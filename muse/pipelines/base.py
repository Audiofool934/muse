"""
Base pipeline interface.

Pipelines compose perception + generation + vocoding into end-to-end
systems. They handle device management, dtype casting, and the
sequential flow between stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


class BasePipeline(ABC):
    """Abstract pipeline interface."""

    @abstractmethod
    def generate(self, inputs: Any, **kwargs) -> Tensor:
        """End-to-end generation: raw input -> audio waveform."""
        ...

    @abstractmethod
    def to(self, device: Union[str, torch.device]) -> "BasePipeline":
        """Move all models to device."""
        ...

    def save_audio(
        self,
        audio: Tensor,
        path: str,
        sample_rate: int = 44100,
        normalize: bool = True,
    ) -> None:
        import torchaudio

        audio = audio.detach().cpu().float()
        if audio.dim() == 3:
            audio = audio[0]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if normalize:
            peak = audio.abs().max()
            if peak > 0:
                audio = audio / peak * 0.95

        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torchaudio.save(path, audio, sample_rate)
