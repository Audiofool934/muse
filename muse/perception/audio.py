"""
Audio perception encoder using MuQ-MuLan.

Encodes audio waveforms to ConditioningOutput, enabling audio-conditioned
music generation (style transfer, audio-to-audio variation).

This encoder bridges the gap between t2m and WeaveWave — when combined
with the flow matching pipeline, it enables audio-to-music generation
where the input audio provides stylistic/timbral conditioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from muse.perception.base import ConditioningOutput, PerceptionEncoder


class MuQMuLanPerceptionEncoder(PerceptionEncoder):
    """
    Frozen MuQ-MuLan audio encoder as perception encoder.

    Produces a single global embedding [B, 1, 512] per audio clip,
    wrapped as ConditioningOutput for downstream flow matching.

    Use case: audio style transfer — condition music generation on
    the sonic characteristics of a reference audio.

    Args:
        model_name: HuggingFace model identifier or local path.
        device: Device to load model on.
        target_dim: If set, project 512 -> target_dim for Stage 1 compatibility.
    """

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-MuLan-large",
        device: Union[str, torch.device] = "cuda",
        target_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        from muq import MuQMuLan

        self.device_ = device if isinstance(device, torch.device) else torch.device(device)

        if Path(model_name).is_dir():
            import json
            import os
            cfg_path = os.path.join(model_name, "config.json")
            weights_path = os.path.join(model_name, "pytorch_model.bin")
            with open(cfg_path) as f:
                cfg = json.load(f)
            original_cache = os.environ.get("HF_HUB_CACHE")
            os.environ["HF_HUB_CACHE"] = model_name
            try:
                self.encoder = MuQMuLan(config=cfg, hf_hub_cache_dir=model_name)
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                self.encoder.load_state_dict(state, strict=True)
                self.encoder = self.encoder.to(self.device_)
            finally:
                if original_cache is not None:
                    os.environ["HF_HUB_CACHE"] = original_cache
                elif "HF_HUB_CACHE" in os.environ:
                    del os.environ["HF_HUB_CACHE"]
        else:
            self.encoder = MuQMuLan.from_pretrained(model_name).to(self.device_)

        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self._native_dim = 512

        if target_dim is not None and target_dim != self._native_dim:
            self.projection = nn.Linear(self._native_dim, target_dim, bias=False).to(self.device_)
            self._dim = target_dim
        else:
            self.projection = None
            self._dim = self._native_dim

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def modality(self) -> str:
        return "audio"

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str], Tensor],
        sample_rate: int = 24000,
        **kwargs,
    ) -> ConditioningOutput:
        """
        Encode audio to conditioning output.

        Args:
            inputs: Audio file path(s) or waveform tensor [B, samples].
            sample_rate: Sample rate of input audio.
        """
        if isinstance(inputs, (str, Path)):
            inputs = [str(inputs)]
        if isinstance(inputs, list) and isinstance(inputs[0], str):
            wavs = self._load_audios(inputs, sample_rate)
        elif isinstance(inputs, Tensor):
            wavs = inputs if inputs.dim() == 2 else inputs.squeeze(1)
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

        wavs = wavs.to(self.device_)
        embeddings = self.encoder(wavs=wavs)  # [B, 512]

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        # Wrap as [B, 1, D] sequence
        embeddings = embeddings.unsqueeze(1)
        B = embeddings.shape[0]
        mask = torch.ones(B, 1, device=self.device_, dtype=torch.bool)

        return ConditioningOutput(
            embeddings=embeddings,
            mask=mask,
            modality="audio",
        )

    def _load_audios(self, paths: List[str], target_sr: int) -> Tensor:
        import torchaudio

        wavs = []
        for p in paths:
            wav, sr = torchaudio.load(p)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            wavs.append(wav.squeeze(0))

        # Pad to same length
        max_len = max(w.shape[0] for w in wavs)
        padded = torch.zeros(len(wavs), max_len)
        for i, w in enumerate(wavs):
            padded[i, : w.shape[0]] = w

        return padded
