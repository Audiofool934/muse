"""
Base perception encoder interface.

All modality-specific encoders (text, image, video, audio) implement
PerceptionEncoder and return a ConditioningOutput. Downstream generation
models consume ConditioningOutput without knowing the source modality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ConditioningOutput:
    """
    Unified conditioning representation from any perception encoder.

    This is the contract between perception and generation layers.
    All encoders produce this; all generators consume this.

    Attributes:
        embeddings: Condition embeddings [B, L, D].
        mask: Padding mask [B, L], True = valid token, False = padding.
        modality: Source modality identifier ("text", "image", "video", "audio").
        metadata: Optional extra info (e.g., original text, image path).
    """

    embeddings: Tensor
    mask: Tensor
    modality: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.embeddings.shape[0]

    @property
    def seq_len(self) -> int:
        return self.embeddings.shape[1]

    @property
    def dim(self) -> int:
        return self.embeddings.shape[2]

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "ConditioningOutput":
        kw = {"device": device}
        if dtype is not None:
            kw["dtype"] = dtype
        return ConditioningOutput(
            embeddings=self.embeddings.to(**kw),
            mask=self.mask.to(device=device),
            modality=self.modality,
            metadata=self.metadata,
        )


class PerceptionEncoder(nn.Module, ABC):
    """
    Abstract base class for all perception encoders.

    Subclasses must implement:
        - encode(): modality-specific input -> ConditioningOutput
        - output_dim: the embedding dimension D

    Encoders are frozen by default (no gradient computation).
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Embedding dimension of the encoder output."""
        ...

    @property
    @abstractmethod
    def modality(self) -> str:
        """Modality name this encoder handles."""
        ...

    @abstractmethod
    @torch.no_grad()
    def encode(self, inputs: Any, **kwargs) -> ConditioningOutput:
        """
        Encode modality-specific inputs to unified conditioning.

        Args:
            inputs: Modality-specific input (str/list[str] for text,
                    Tensor for image/video, etc.)

        Returns:
            ConditioningOutput with embeddings [B, L, D] and mask [B, L].
        """
        ...

    def freeze(self) -> None:
        """Freeze all parameters and set to eval mode."""
        self.eval()
        self.requires_grad_(False)
