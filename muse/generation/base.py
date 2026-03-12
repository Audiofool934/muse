"""
Base generation model interface.

Generation models consume ConditioningOutput from perception encoders
and produce either latent embeddings (Stage 1) or audio waveforms (Stage 2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from muse.perception.base import ConditioningOutput


@dataclass
class GenerationOutput:
    """Output from a generation model."""

    result: Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


class LatentGenerator(nn.Module, ABC):
    """
    Abstract base for Stage 1: condition -> latent space.

    Maps ConditioningOutput to a target latent space (e.g., MuQ-MuLan 512-dim).
    The latent can then be fed to Stage 2 (audio synthesis).
    """

    @abstractmethod
    def forward(
        self,
        condition: ConditioningOutput,
        target_latents: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Training forward pass.

        Args:
            condition: From perception encoder.
            target_latents: Ground truth latents [B, L, D].

        Returns:
            (loss, metrics_dict)
        """
        ...

    @abstractmethod
    @torch.inference_mode()
    def generate(
        self,
        condition: ConditioningOutput,
        num_steps: int = 50,
        guidance_scale: float = 3.0,
        **kwargs,
    ) -> Tensor:
        """
        Generate latents from conditioning.

        Returns:
            Generated latents [B, L, D].
        """
        ...


class AudioSynthesizer(nn.Module, ABC):
    """
    Abstract base for Stage 2: latent -> audio waveform.

    Maps latent conditioning to audio using a generative model
    (e.g., Stable Audio DiT with flow matching).
    """

    @abstractmethod
    def forward(
        self,
        audio_waveform: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        """
        Training forward: compute loss.

        Args:
            audio_waveform: Target audio [B, C, T].
            conditioning: Latent conditioning [B, D].

        Returns:
            Scalar loss.
        """
        ...

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        num_steps: int = 50,
        guidance_scale: float = 3.0,
        **kwargs,
    ) -> Tensor:
        """
        Generate audio from conditioning.

        Returns:
            Audio waveform [B, C, T].
        """
        ...
