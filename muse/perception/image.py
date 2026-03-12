"""
Image perception encoder using CLIP / SigLIP.

Encodes images to ConditioningOutput compatible with the flow matching
generation stage. Uses patch embeddings (sequence output) rather than
the pooled [CLS] token, providing richer spatial conditioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from muse.perception.base import ConditioningOutput, PerceptionEncoder


class CLIPPerceptionEncoder(PerceptionEncoder):
    """
    Frozen CLIP ViT image encoder.

    Outputs patch-level embeddings [B, num_patches, D] so the flow matching
    transformer can cross-attend to spatial features.

    A learned linear projection maps CLIP dim -> target_dim to match the
    text encoder's output dimension (e.g., 768 for T5-base compatibility).

    Args:
        model_name: HuggingFace CLIP model identifier.
        target_dim: Output embedding dimension (should match text encoder).
            If None, uses the native CLIP dimension (no projection).
        device: Device to load model on.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        target_dim: Optional[int] = 768,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()

        from transformers import CLIPModel, CLIPProcessor

        self.device_ = device if isinstance(device, torch.device) else torch.device(device)

        self.processor = CLIPProcessor.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name).to(self.device_)
        self.vision_model = clip_model.vision_model
        self.vision_model.eval()
        self.vision_model.requires_grad_(False)

        clip_dim = self.vision_model.config.hidden_size

        # Optional projection to match text encoder dim
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
        return "image"

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str], Tensor, "PIL.Image.Image", List["PIL.Image.Image"]],
        **kwargs,
    ) -> ConditioningOutput:
        """
        Encode images to conditioning output.

        Args:
            inputs: File path(s), PIL Image(s), or pre-loaded tensor [B, C, H, W].
        """
        images = self._load_images(inputs)

        pixel_values = self.processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(self.device_)

        # Get patch embeddings (skip [CLS] token at position 0)
        vision_out = self.vision_model(pixel_values=pixel_values)
        patch_embeds = vision_out.last_hidden_state[:, 1:, :]  # [B, num_patches, D]

        if self.projection is not None:
            patch_embeds = self.projection(patch_embeds)

        B, L, _ = patch_embeds.shape
        mask = torch.ones(B, L, device=self.device_, dtype=torch.bool)

        return ConditioningOutput(
            embeddings=patch_embeds,
            mask=mask,
            modality="image",
        )

    def _load_images(self, inputs):
        from PIL import Image

        if isinstance(inputs, (str, Path)):
            return [Image.open(inputs).convert("RGB")]
        if isinstance(inputs, Image.Image):
            return [inputs]
        if isinstance(inputs, list):
            out = []
            for x in inputs:
                if isinstance(x, (str, Path)):
                    out.append(Image.open(x).convert("RGB"))
                elif isinstance(x, Image.Image):
                    out.append(x)
                else:
                    raise TypeError(f"Unsupported image input type: {type(x)}")
            return out
        raise TypeError(f"Unsupported input type: {type(inputs)}")


class SigLIPPerceptionEncoder(PerceptionEncoder):
    """
    Frozen SigLIP image encoder (Google's improved CLIP variant).

    SigLIP uses sigmoid loss instead of softmax, often producing better
    aligned embeddings. Same interface as CLIPPerceptionEncoder.

    Args:
        model_name: HuggingFace SigLIP model identifier.
        target_dim: Output embedding dimension.
        device: Device to load model on.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        target_dim: Optional[int] = 768,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()

        from transformers import AutoModel, AutoProcessor

        self.device_ = device if isinstance(device, torch.device) else torch.device(device)

        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device_)
        self.vision_model = model.vision_model
        self.vision_model.eval()
        self.vision_model.requires_grad_(False)

        siglip_dim = self.vision_model.config.hidden_size

        if target_dim is not None and target_dim != siglip_dim:
            self.projection = nn.Linear(siglip_dim, target_dim, bias=False).to(self.device_)
            self._dim = target_dim
        else:
            self.projection = None
            self._dim = siglip_dim

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def modality(self) -> str:
        return "image"

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str], "PIL.Image.Image", List["PIL.Image.Image"]],
        **kwargs,
    ) -> ConditioningOutput:
        from PIL import Image

        if isinstance(inputs, (str, Path)):
            images = [Image.open(inputs).convert("RGB")]
        elif isinstance(inputs, Image.Image):
            images = [inputs]
        elif isinstance(inputs, list):
            images = [
                Image.open(x).convert("RGB") if isinstance(x, (str, Path)) else x
                for x in inputs
            ]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

        pixel_values = self.processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(self.device_)

        # SigLIP has no [CLS] token; all tokens are patch embeddings
        vision_out = self.vision_model(pixel_values=pixel_values)
        patch_embeds = vision_out.last_hidden_state  # [B, num_patches, D]

        if self.projection is not None:
            patch_embeds = self.projection(patch_embeds)

        B, L, _ = patch_embeds.shape
        mask = torch.ones(B, L, device=self.device_, dtype=torch.bool)

        return ConditioningOutput(
            embeddings=patch_embeds,
            mask=mask,
            modality="image",
        )
