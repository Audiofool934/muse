"""
Cond2LatentFlow: Flow Matching Transformer for conditioning -> latent generation.

Generalized from t2m's Text2MuQFlow. The key change: instead of hardcoding
text_dim, this model accepts any ConditioningOutput and cross-attends to it.

This means the same architecture works for text->MuQ, image->MuQ, video->MuQ
simply by swapping the perception encoder upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

from muse.perception.base import ConditioningOutput
from muse.generation.base import LatentGenerator


# =============================================================================
# Latent Space Configuration
# =============================================================================

@dataclass(frozen=True)
class LatentSpaceConfig:
    name: str
    dim: int
    seq_len: int
    normalize_input: bool
    normalize_output: bool


LATENT_SPACES = {
    "muq_mulan": LatentSpaceConfig("muq_mulan", 512, 1, False, True),
    "muq": LatentSpaceConfig("muq", 1024, 35, True, False),
}


def _detect_latent_space(latent_dim: int, max_seq_len: int) -> LatentSpaceConfig:
    if latent_dim == 512 and max_seq_len == 1:
        return LATENT_SPACES["muq_mulan"]
    if latent_dim == 1024 and max_seq_len == 35:
        return LATENT_SPACES["muq"]
    return LatentSpaceConfig("custom", latent_dim, max_seq_len, True, False)


# =============================================================================
# Transformer Components (same as t2m)
# =============================================================================

class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h)[0])
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, context, context, key_padding_mask=mask)[0])
        x = x + self.ffn(self.norm3(x))
        return x


class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-torch.arange(half) * (2.0 * torch.log(torch.tensor(10000.0)) / half))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: Tensor) -> Tensor:
        args = t * self.freqs.to(t.device)
        embed = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.proj(embed)


# =============================================================================
# Main Model
# =============================================================================

class Cond2LatentFlow(LatentGenerator):
    """
    Flow Matching model: ConditioningOutput -> target latent space.

    Generalized from t2m's Text2MuQFlow. The model is modality-agnostic:
    it cross-attends to whatever embeddings the perception encoder provides.

    Args:
        cond_dim: Dimension of conditioning embeddings from perception encoder.
        latent_dim: Target latent dimension (512 for MuQ-MuLan).
        max_seq_len: Latent sequence length (1 for MuQ-MuLan global).
        d_model: Transformer hidden dimension.
        depth: Number of transformer layers.
        heads: Number of attention heads.
        dropout: Dropout rate.
        unconditional_prob: CFG dropout probability during training.
    """

    def __init__(
        self,
        cond_dim: int = 768,
        latent_dim: int = 512,
        max_seq_len: int = 1,
        d_model: int = 1024,
        depth: int = 16,
        heads: int = 16,
        dropout: float = 0.1,
        unconditional_prob: float = 0.1,
    ) -> None:
        super().__init__()

        self._latent_config = _detect_latent_space(latent_dim, max_seq_len)

        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.unconditional_prob = unconditional_prob

        # Projections
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.velocity_head = nn.Linear(d_model, latent_dim)

        # Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.time_embed = _SinusoidalTimeEmbed(d_model)
        self.null_cond_embed = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)

        # Transformer
        self.layers = nn.ModuleList([
            _TransformerLayer(d_model, heads, d_model * 4, dropout)
            for _ in range(depth)
        ])

        # Flow Matching path
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _preprocess_latents(self, x: Tensor) -> Tensor:
        if self._latent_config.normalize_input:
            return F.instance_norm(x)
        return x

    def _postprocess_latents(self, x: Tensor) -> Tensor:
        if self._latent_config.normalize_output:
            return F.normalize(x, p=2, dim=-1)
        return x

    def forward(
        self,
        condition: ConditioningOutput,
        target_latents: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Flow matching training loss.

        Args:
            condition: ConditioningOutput from any perception encoder.
            target_latents: [B, L_latent, latent_dim].
        """
        B = condition.batch_size
        device = condition.embeddings.device

        x1 = self._preprocess_latents(target_latents)
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=device)

        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        x_t = path_sample.x_t
        target_velocity = path_sample.dx_t

        # Project conditioning
        cond = self.cond_proj(condition.embeddings)

        # CFG dropout
        if self.training and self.unconditional_prob > 0:
            drop = torch.rand(B, device=device) < self.unconditional_prob
            null_cond = self.cond_proj(self.null_cond_embed.expand(B, -1, -1))
            cond = torch.where(drop[:, None, None], null_cond, cond)

        # Build mask: invert valid mask to get padding mask for cross-attention
        padding_mask = ~condition.mask  # True = padding

        pred = self._predict_velocity(x_t, t, cond, padding_mask)
        loss = F.mse_loss(pred, target_velocity)

        return loss, {"loss": loss.item(), "t_mean": float(t.mean())}

    def _predict_velocity(self, x, t, cond, mask):
        x = self.latent_proj(x) + self.pos_embed
        t_embed = self.time_embed(t.unsqueeze(-1)).unsqueeze(1)
        x = x + t_embed
        for layer in self.layers:
            x = layer(x, cond, mask)
        return self.velocity_head(x)

    @torch.inference_mode()
    def generate(
        self,
        condition: ConditioningOutput,
        num_steps: int = 50,
        guidance_scale: float = 3.0,
        method: Literal["euler", "midpoint", "rk4", "dopri5"] = "midpoint",
        generator: Optional[torch.Generator] = None,
        x0: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        device = condition.embeddings.device
        B = condition.batch_size

        cond = self.cond_proj(condition.embeddings)
        padding_mask = ~condition.mask

        null_cond = None
        null_mask = None
        if guidance_scale > 1.0:
            null_cond = self.cond_proj(self.null_cond_embed.expand(B, -1, -1))
            null_mask = torch.zeros(B, null_cond.shape[1], device=device, dtype=torch.bool)

        if x0 is not None:
            x = x0.to(device=device, dtype=torch.float32)
        else:
            x = torch.randn(
                B, self.max_seq_len, self.latent_dim,
                generator=generator, device=device, dtype=torch.float32,
            )

        velocity_fn = _CFGVelocity(self, cond, null_cond, guidance_scale, padding_mask, null_mask)
        solver = ODESolver(velocity_model=velocity_fn)

        result = solver.sample(
            x_init=x,
            step_size=1.0 / num_steps,
            method=method,
            time_grid=torch.linspace(0, 1, num_steps + 1, device=device),
            return_intermediates=False,
        )

        return self._postprocess_latents(result)


class _CFGVelocity(nn.Module):
    def __init__(self, model, cond, null_cond, scale, cond_mask, null_mask):
        super().__init__()
        self.model = model
        self.cond = cond
        self.null_cond = null_cond
        self.scale = scale
        self.cond_mask = cond_mask
        self.null_mask = null_mask

    def forward(self, x, t):
        B = x.shape[0]
        if t.dim() == 0:
            t = t.expand(B)

        x_proj = self.model.latent_proj(x) + self.model.pos_embed
        t_embed = self.model.time_embed(t.unsqueeze(-1)).unsqueeze(1)
        x_in = x_proj + t_embed

        h = x_in.clone()
        for layer in self.model.layers:
            h = layer(h, self.cond, self.cond_mask)
        v_cond = self.model.velocity_head(h)

        if self.null_cond is not None and self.scale > 1.0:
            h = x_in.clone()
            for layer in self.model.layers:
                h = layer(h, self.null_cond, self.null_mask)
            v_uncond = self.model.velocity_head(h)
            return v_uncond + self.scale * (v_cond - v_uncond)

        return v_cond
