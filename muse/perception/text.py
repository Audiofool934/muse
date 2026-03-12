"""
Text perception encoder using Flan-T5.

Wraps a frozen T5 encoder to produce ConditioningOutput from text strings.
This is the same encoder used in t2m's Stage 1.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from muse.perception.base import ConditioningOutput, PerceptionEncoder


class T5PerceptionEncoder(PerceptionEncoder):
    """
    Frozen Flan-T5-Base text encoder.

    Args:
        model_name: HuggingFace model identifier.
        device: Device to load model on.
        max_length: Maximum tokenization length.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Union[str, torch.device] = "cuda",
        max_length: int = 128,
    ) -> None:
        super().__init__()

        from transformers import AutoTokenizer, T5EncoderModel

        self.device_ = device if isinstance(device, torch.device) else torch.device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name).to(self.device_)
        self._dim = self.encoder.config.d_model

        self.freeze()

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def modality(self) -> str:
        return "text"

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str]],
        **kwargs,
    ) -> ConditioningOutput:
        if isinstance(inputs, str):
            inputs = [inputs]

        tokens = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device_)

        embeddings = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].bool()

        return ConditioningOutput(
            embeddings=embeddings,
            mask=mask,
            modality="text",
            metadata={"texts": inputs},
        )
