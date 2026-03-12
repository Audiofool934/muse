"""
Video perception encoder.

Encodes video frames into a temporal sequence of embeddings for
conditioning music generation on visual motion and scene dynamics.

Strategy: extract N frames uniformly, encode each with a vision
encoder, producing [B, N * patches_per_frame, D] or a pooled
[B, N, D] sequence. The temporal ordering is preserved so the
flow matching transformer can attend to motion.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from muse.perception.base import ConditioningOutput, PerceptionEncoder


class VideoFramePerceptionEncoder(PerceptionEncoder):
    """
    Video encoder that samples frames and encodes them with a frozen
    vision backbone (CLIP ViT).

    Each frame is encoded to a single pooled vector, yielding a
    temporal sequence [B, num_frames, D] that captures scene dynamics.

    Args:
        vision_model_name: HuggingFace vision model for frame encoding.
        num_frames: Number of frames to sample uniformly from the video.
        target_dim: Output embedding dimension.
        pool_patches: If True, pool patch tokens per frame to a single
            vector. If False, concatenate all patches (much longer sequence).
        device: Device to load model on.
    """

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-large-patch14",
        num_frames: int = 16,
        target_dim: Optional[int] = 768,
        pool_patches: bool = True,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()

        from transformers import CLIPModel, CLIPProcessor

        self.device_ = device if isinstance(device, torch.device) else torch.device(device)
        self.num_frames = num_frames
        self.pool_patches = pool_patches

        self.processor = CLIPProcessor.from_pretrained(vision_model_name)
        clip_model = CLIPModel.from_pretrained(vision_model_name).to(self.device_)
        self.vision_model = clip_model.vision_model
        self.vision_model.eval()
        self.vision_model.requires_grad_(False)

        clip_dim = self.vision_model.config.hidden_size

        if target_dim is not None and target_dim != clip_dim:
            self.projection = nn.Linear(clip_dim, target_dim, bias=False).to(self.device_)
            self._dim = target_dim
        else:
            self.projection = None
            self._dim = clip_dim

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def modality(self) -> str:
        return "video"

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, Path, List[str]],
        **kwargs,
    ) -> ConditioningOutput:
        """
        Encode video file(s) to conditioning output.

        Args:
            inputs: Path(s) to video file(s).
        """
        if isinstance(inputs, (str, Path)):
            inputs = [inputs]

        all_embeds = []
        for video_path in inputs:
            frames = self._extract_frames(str(video_path))
            embeds = self._encode_frames(frames)  # [num_frames, D] or [num_patches*num_frames, D]
            all_embeds.append(embeds)

        # Pad to same length and stack
        max_len = max(e.shape[0] for e in all_embeds)
        B = len(all_embeds)
        padded = torch.zeros(B, max_len, self._dim, device=self.device_)
        mask = torch.zeros(B, max_len, device=self.device_, dtype=torch.bool)

        for i, e in enumerate(all_embeds):
            L = e.shape[0]
            padded[i, :L] = e
            mask[i, :L] = True

        return ConditioningOutput(
            embeddings=padded,
            mask=mask,
            modality="video",
        )

    def _extract_frames(self, video_path: str) -> List:
        """Extract uniformly-sampled frames from video using decord."""
        from decord import VideoReader, cpu
        from PIL import Image
        import numpy as np

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)

        if total <= self.num_frames:
            indices = list(range(total))
        else:
            indices = [int(i * total / self.num_frames) for i in range(self.num_frames)]

        frames_np = vr.get_batch(indices).asnumpy()  # [N, H, W, C]
        return [Image.fromarray(f) for f in frames_np]

    def _encode_frames(self, frames) -> Tensor:
        """Encode a list of PIL frames to embeddings."""
        pixel_values = self.processor(
            images=frames, return_tensors="pt"
        ).pixel_values.to(self.device_)  # [N, C, H, W]

        vision_out = self.vision_model(pixel_values=pixel_values)

        if self.pool_patches:
            # Use [CLS] token as per-frame representation
            frame_embeds = vision_out.last_hidden_state[:, 0, :]  # [N, D]
        else:
            # Concatenate all patch tokens across frames
            frame_embeds = vision_out.last_hidden_state[:, 1:, :]  # [N, P, D]
            frame_embeds = frame_embeds.reshape(-1, frame_embeds.shape[-1])  # [N*P, D]

        if self.projection is not None:
            frame_embeds = self.projection(frame_embeds)

        return frame_embeds
