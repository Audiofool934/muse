"""
TwoStageFlowPipeline: Perception -> Flow Matching -> DiT -> Audio.

This is the core MUSE pipeline. It composes:
  1. PerceptionEncoder (text / image / video)
  2. Cond2LatentFlow (Stage 1: condition -> MuQ-MuLan latent)
  3. LatentToAudioDiT (Stage 2: latent -> 44.1kHz audio)

The same pipeline class works for t2m, i2m, and v2m — only the
perception encoder changes.

Usage:
    # Text-to-Music (same as t2m)
    pipe = TwoStageFlowPipeline.from_config("configs/t2m_flow.yaml")
    audio = pipe.generate("A calm piano melody")

    # Image-to-Music
    pipe = TwoStageFlowPipeline.from_config("configs/i2m_flow.yaml")
    audio = pipe.generate("path/to/image.jpg")

    # Video-to-Music
    pipe = TwoStageFlowPipeline.from_config("configs/v2m_flow.yaml")
    audio = pipe.generate("path/to/video.mp4")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from muse.perception.base import ConditioningOutput, PerceptionEncoder
from muse.generation.flow_matching.transformer import Cond2LatentFlow
from muse.generation.flow_matching.dit import LatentToAudioDiT
from muse.pipelines.base import BasePipeline


# =============================================================================
# Encoder Registry
# =============================================================================

_ENCODER_REGISTRY = {
    "t5": ("muse.perception.text", "T5PerceptionEncoder"),
    "clip": ("muse.perception.image", "CLIPPerceptionEncoder"),
    "siglip": ("muse.perception.image", "SigLIPPerceptionEncoder"),
    "video_frames": ("muse.perception.video", "VideoFramePerceptionEncoder"),
    "muq_mulan": ("muse.perception.audio", "MuQMuLanPerceptionEncoder"),
    "mllm_bridge": ("muse.perception.mllm_bridge", "MLLMBridgeEncoder"),
}


def _build_encoder(config: dict, device: str) -> PerceptionEncoder:
    """Instantiate a perception encoder from config dict."""
    enc_type = config["type"]

    if enc_type in _ENCODER_REGISTRY:
        module_path, class_name = _ENCODER_REGISTRY[enc_type]
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    else:
        # Allow fully qualified class path
        module_path, class_name = enc_type.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

    params = {k: v for k, v in config.items() if k != "type"}
    params["device"] = device
    return cls(**params)


# =============================================================================
# Pipeline
# =============================================================================

class TwoStageFlowPipeline(BasePipeline):
    """
    Unified two-stage flow matching pipeline for any-modality-to-music.

    Args:
        perception: Any PerceptionEncoder (text, image, video, ...).
        stage1: Cond2LatentFlow model.
        stage2: LatentToAudioDiT model.
        device: Compute device.
        dtype: Model dtype.
    """

    def __init__(
        self,
        perception: PerceptionEncoder,
        stage1: Cond2LatentFlow,
        stage2: LatentToAudioDiT,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        stage1_num_steps: int = 50,
        stage1_guidance: float = 1.5,
        stage1_method: str = "midpoint",
        stage2_num_steps: int = 50,
        stage2_guidance: float = 3.0,
        stage2_method: str = "dopri5",
        sample_rate: int = 44100,
    ) -> None:
        self.perception = perception
        self.stage1 = stage1
        self.stage2 = stage2
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        self.stage1_num_steps = stage1_num_steps
        self.stage1_guidance = stage1_guidance
        self.stage1_method = stage1_method
        self.stage2_num_steps = stage2_num_steps
        self.stage2_guidance = stage2_guidance
        self.stage2_method = stage2_method
        self.sample_rate = sample_rate

        # Move to device and eval
        for m in [self.stage1, self.stage2]:
            m.to(self.device)
            m.eval()

    def to(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.stage1.to(self.device)
        self.stage2.to(self.device)
        return self

    @torch.no_grad()
    def generate(
        self,
        inputs: Any,
        seed: Optional[int] = None,
        num_steps_stage1: Optional[int] = None,
        num_steps_stage2: Optional[int] = None,
        guidance_scale_stage1: Optional[float] = None,
        guidance_scale_stage2: Optional[float] = None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        End-to-end generation from any modality input.

        Args:
            inputs: Modality-specific input (text string, image path, video path, etc.)
            seed: Random seed for reproducibility.
            return_intermediate: If True, return dict with latents too.

        Returns:
            Audio tensor [B, C, T] at 44.1kHz, or dict with intermediates.
        """
        generator = None
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        s1_steps = num_steps_stage1 or self.stage1_num_steps
        s1_cfg = guidance_scale_stage1 if guidance_scale_stage1 is not None else self.stage1_guidance
        s2_steps = num_steps_stage2 or self.stage2_num_steps
        s2_cfg = guidance_scale_stage2 if guidance_scale_stage2 is not None else self.stage2_guidance

        # Step 1: Perception
        condition = self.perception.encode(inputs)
        condition = condition.to(self.device, dtype=self.dtype)

        # Step 2: Stage 1 — condition -> latent
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            latents = self.stage1.generate(
                condition,
                num_steps=s1_steps,
                guidance_scale=s1_cfg,
                method=self.stage1_method,
                generator=generator,
            )

        # Pool latents for Stage 2
        latent_pooled = latents.mean(dim=1)  # [B, D]
        if latent_pooled.shape[-1] == 512:
            latent_pooled = F.normalize(latent_pooled, p=2, dim=-1)

        # Step 3: Stage 2 — latent -> audio
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            audio = self.stage2.sample(
                conditioning=latent_pooled.to(self.dtype),
                num_steps=s2_steps,
                guidance_scale=s2_cfg,
                generator=generator,
                method=self.stage2_method,
            )

        if return_intermediate:
            return {
                "audio": audio,
                "latents": latents,
                "latent_pooled": latent_pooled,
                "condition": condition,
            }

        return audio

    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: str = "cuda",
    ) -> "TwoStageFlowPipeline":
        """
        Build pipeline from a YAML config file.

        Config structure:
            perception:
              type: t5 | clip | siglip | video_frames
              model_name: ...
              target_dim: 768
            stage1:
              checkpoint: path/to/stage1.pt
              params:
                cond_dim: 768
                latent_dim: 512
                ...
            stage2:
              checkpoint: path/to/stage2.pt
              params:
                cond_dim: 512
                ...
            inference:
              stage1_num_steps: 50
              ...
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Build perception encoder
        perception = _build_encoder(cfg["perception"], device)

        # Build Stage 1
        stage1 = Cond2LatentFlow(**cfg["stage1"]["params"])
        s1_ckpt = cfg["stage1"].get("checkpoint")
        if s1_ckpt:
            _load_checkpoint(stage1, s1_ckpt, device)

        # Build Stage 2
        stage2 = LatentToAudioDiT(**cfg["stage2"]["params"])
        s2_ckpt = cfg["stage2"].get("checkpoint")
        if s2_ckpt:
            _load_checkpoint(stage2, s2_ckpt, device)

        # Inference settings
        inf = cfg.get("inference", {})

        return cls(
            perception=perception,
            stage1=stage1,
            stage2=stage2,
            device=device,
            stage1_num_steps=inf.get("stage1_num_steps", 50),
            stage1_guidance=inf.get("stage1_guidance", 1.5),
            stage1_method=inf.get("stage1_method", "midpoint"),
            stage2_num_steps=inf.get("stage2_num_steps", 50),
            stage2_guidance=inf.get("stage2_guidance", 3.0),
            stage2_method=inf.get("stage2_method", "dopri5"),
            sample_rate=inf.get("sample_rate", 44100),
        )


def _load_checkpoint(model: nn.Module, path: str, device: str) -> None:
    """Load checkpoint with DDP prefix handling."""
    ckpt = torch.load(path, map_location=device, weights_only=True)

    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # Strip DDP "module." prefix
    cleaned = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(cleaned)
