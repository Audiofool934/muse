"""
MLLM Bridge perception encoder.

Converts any visual/multimodal input to a text description via a
Multimodal LLM, then encodes the description with T5. This is the
strategy used by WeaveWave — it provides zero-shot capability for
any modality without modality-specific training.

Architecture:
    Image/Video → MLLM (Gemma-3) → text description → T5 → embeddings

Tradeoff vs. direct encoders (CLIP, ViViT):
    + Works zero-shot for any modality
    + Leverages MLLM's rich understanding
    - Two-stage: slower inference
    - Information bottleneck through text
    - Not end-to-end differentiable
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from muse.perception.base import ConditioningOutput, PerceptionEncoder


# Default system prompts (from WeaveWave)
SYSTEM_PROMPTS = {
    "image": (
        "You are a music composer who generates short, concise description "
        "of music inspired by visual input. When provided with an image, "
        "interpret its elements such as colors, mood, and content — and "
        "translate them into musical terms. Describe the tone, rhythm, "
        "genre, and instruments most suited to the image's atmosphere. "
        "Respond with only the music description, 1-2 sentences."
    ),
    "video": (
        "You are a music composer who generates short, concise description "
        "of music inspired by video content. Analyze the motion, pacing, "
        "mood, and visual narrative — and translate them into musical terms. "
        "Describe the tone, rhythm, genre, and instruments most suited to "
        "the video's atmosphere and dynamics. "
        "Respond with only the music description, 1-2 sentences."
    ),
}


class MLLMBridgeEncoder(PerceptionEncoder):
    """
    MLLM-based perception encoder for zero-shot multimodal conditioning.

    Uses a Multimodal LLM to describe visual input, then encodes the
    description with T5 for flow matching conditioning.

    Args:
        mllm_model_name: HuggingFace MLLM model identifier.
        t5_model_name: T5 model for text encoding.
        input_modality: "image" or "video" — selects system prompt.
        device: Device to load models on.
        mllm_max_new_tokens: Max tokens for MLLM generation.
    """

    def __init__(
        self,
        mllm_model_name: str = "google/gemma-3-12b-it",
        t5_model_name: str = "google/flan-t5-base",
        input_modality: str = "image",
        device: Union[str, torch.device] = "cuda",
        mllm_max_new_tokens: int = 200,
    ) -> None:
        super().__init__()

        self.device_ = device if isinstance(device, torch.device) else torch.device(device)
        self.input_modality = input_modality
        self.mllm_max_new_tokens = mllm_max_new_tokens
        self.system_prompt = SYSTEM_PROMPTS.get(input_modality, SYSTEM_PROMPTS["image"])

        # Load MLLM
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.mllm_processor = AutoProcessor.from_pretrained(mllm_model_name)
        self.mllm_model = AutoModelForCausalLM.from_pretrained(
            mllm_model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device_)
        self.mllm_model.eval()

        # Load T5 text encoder
        from muse.perception.text import T5PerceptionEncoder

        self.t5_encoder = T5PerceptionEncoder(
            model_name=t5_model_name,
            device=device,
        )

        self._dim = self.t5_encoder.output_dim

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def modality(self) -> str:
        return self.input_modality

    @torch.no_grad()
    def encode(self, inputs: Any, user_prompt: str = "", **kwargs) -> ConditioningOutput:
        """
        Encode visual input via MLLM bridge.

        Args:
            inputs: Image/video file path(s).
            user_prompt: Optional user guidance for the MLLM.
        """
        if isinstance(inputs, (str, Path)):
            inputs = [str(inputs)]

        descriptions = []
        for path in inputs:
            desc = self._describe(path, user_prompt)
            descriptions.append(desc)

        # Encode descriptions with T5
        condition = self.t5_encoder.encode(descriptions)
        condition.metadata["descriptions"] = descriptions
        condition.metadata["source_modality"] = self.input_modality
        return condition

    def _describe(self, media_path: str, user_prompt: str = "") -> str:
        """Generate music description from visual input using MLLM."""
        from PIL import Image

        if self.input_modality == "image":
            image = Image.open(media_path).convert("RGB")
            content = [{"type": "image", "image": image}]
        elif self.input_modality == "video":
            # Extract middle frame (same as WeaveWave's approach)
            from decord import VideoReader, cpu as decord_cpu
            vr = VideoReader(media_path, ctx=decord_cpu(0))
            mid_frame = vr[len(vr) // 2].asnumpy()
            image = Image.fromarray(mid_frame)
            content = [{"type": "image", "image": image}]
        else:
            raise ValueError(f"Unsupported modality: {self.input_modality}")

        if user_prompt:
            content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        prompt = self.mllm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.mllm_processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device_)

        output_ids = self.mllm_model.generate(
            **model_inputs,
            max_new_tokens=self.mllm_max_new_tokens,
            do_sample=False,
        )
        new_tokens = output_ids[:, model_inputs["input_ids"].shape[1] :]
        description = self.mllm_processor.decode(new_tokens[0], skip_special_tokens=True)

        # Free MLLM VRAM
        del model_inputs, output_ids
        torch.cuda.empty_cache()

        return description.strip()
