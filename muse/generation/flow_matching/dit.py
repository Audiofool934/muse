"""
LatentToAudioDiT: Stable Audio DiT conditioned on latent embeddings.

Generalized from t2m's StableAudioMuQ. This Stage 2 model takes any
latent conditioning vector and generates audio via flow matching + DiT.

It is fully modality-agnostic — it doesn't know whether the conditioning
came from text, image, or video. That decoupling happens at Stage 1.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusers import AutoencoderOobleck
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from muse.generation.base import AudioSynthesizer


class _GuidedVelocity(ModelWrapper):
    """CFG wrapper for DiT velocity prediction."""

    def forward(self, x, t, c, w=3.0, rotary_embedding=None, **extras):
        t = t.unsqueeze(0).repeat(x.shape[0]).to(x)

        c_uncond = torch.zeros_like(c).to(x)
        uncond_pred = self.model(
            x, t,
            encoder_hidden_states=c_uncond,
            global_hidden_states=None,
            rotary_embedding=rotary_embedding,
        ).sample

        cond_pred = self.model(
            x, t,
            encoder_hidden_states=c,
            global_hidden_states=None,
            rotary_embedding=rotary_embedding,
        ).sample

        return uncond_pred + w * (cond_pred - uncond_pred)


class LatentToAudioDiT(AudioSynthesizer):
    """
    Stage 2: Latent conditioning -> audio waveform via Stable Audio DiT.

    Generalized from t2m's StableAudioMuQ. The only assumption is that
    conditioning is a 1D vector of dimension `cond_dim`.

    Args:
        audiocodec_ckpt_path: Path to Oobleck VAE checkpoint.
        ckpt_dir_audio_dit: Path to Stable Audio DiT checkpoint.
        cond_dim: Dimension of input conditioning vector.
        latent_length: Audio latent sequence length (256 ~ 12s).
        dit_num_layers: Number of DiT transformer layers.
        scale_factor: Latent scaling factor.
        unconditional_prob: CFG dropout during training.
        sample_steps: Default ODE solver steps.
        guidance_scale: Default CFG scale.
        freeze_dit: If True, freeze DiT (train only projection).
    """

    def __init__(
        self,
        audiocodec_ckpt_path: str,
        ckpt_dir_audio_dit: str,
        cond_dim: int = 512,
        latent_length: int = 256,
        dit_num_layers: int = 24,
        scale_factor: float = 1.0,
        unconditional_prob: float = 0.3,
        sample_steps: int = 50,
        guidance_scale: float = 3.0,
        freeze_dit: bool = False,
    ) -> None:
        super().__init__()

        # Frozen VAE codec
        self.audio_codec = AutoencoderOobleck.from_pretrained(audiocodec_ckpt_path)
        self.audio_codec.requires_grad_(False)
        self.audio_codec.eval()

        # Resolve Stable Audio DiT import
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        _stable_audio_root = os.environ.get("STABLE_AUDIO_ROOT", _project_root)
        if _stable_audio_root not in sys.path:
            sys.path.append(_stable_audio_root)

        from model.stable_audio.stable_audio_transformer import StableAudioDiTModel

        self.music_flow = StableAudioDiTModel.from_pretrained(
            ckpt_dir_audio_dit,
            num_layers=dit_num_layers,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )

        if freeze_dit:
            self.music_flow.requires_grad_(False)
            self.music_flow.eval()

        # Conditioning projection
        dit_cross_dim = self.music_flow.config.cross_attention_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, dit_cross_dim, bias=False),
            nn.SiLU(),
            nn.Linear(dit_cross_dim, dit_cross_dim, bias=False),
        )

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.scale_factor = scale_factor
        self.latent_length = latent_length
        self.latent_in_dim = getattr(self.music_flow.config, "in_channels", 64)
        self.rotary_embed_dim = self.music_flow.config.attention_head_dim // 2
        self.unconditional_prob = unconditional_prob
        self.sample_steps = sample_steps
        self.guidance_scale = guidance_scale

    def _get_rotary_embedding(self, seq_len, device):
        return get_1d_rotary_pos_embed(
            self.rotary_embed_dim, seq_len + 1,
            use_real=True, repeat_interleave_real=False,
        )

    def _encode_audio(self, audio: Tensor) -> Tensor:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        with torch.no_grad():
            with torch.autocast(device_type=audio.device.type, enabled=False):
                latent_dist = self.audio_codec.encode(audio.float())
                return latent_dist.latent_dist.sample().detach() * self.scale_factor

    def forward(self, audio_waveform: Tensor, conditioning: Tensor) -> Tensor:
        B = audio_waveform.shape[0]
        device = audio_waveform.device

        audio_latent = self._encode_audio(audio_waveform)

        cond = self.cond_proj(conditioning.detach()).unsqueeze(1)

        if self.training and self.unconditional_prob > 0:
            drop = torch.rand(B, device=device) < self.unconditional_prob
            cond = cond * (1.0 - drop.view(-1, 1, 1).float())

        noise = torch.randn_like(audio_latent)
        t = torch.rand(B, device=device)

        path_sample = self.path.sample(t=t, x_0=noise, x_1=audio_latent)
        rotary = self._get_rotary_embedding(path_sample.x_t.shape[2], device)

        pred = self.music_flow(
            path_sample.x_t, path_sample.t,
            encoder_hidden_states=cond,
            global_hidden_states=None,
            rotary_embedding=rotary,
        ).sample

        return F.mse_loss(pred, path_sample.dx_t)

    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        method: str = "dopri5",
        **kwargs,
    ) -> Tensor:
        device = conditioning.device
        B = conditioning.shape[0]

        steps = num_steps or self.sample_steps
        w = guidance_scale if guidance_scale is not None else self.guidance_scale

        cond = self.cond_proj(conditioning.detach()).unsqueeze(1)

        wrapped = _GuidedVelocity(self.music_flow)
        solver = ODESolver(velocity_model=wrapped)

        init_latent = randn_tensor(
            (B, self.latent_in_dim, self.latent_length),
            device=device, generator=generator,
        )

        rotary = self._get_rotary_embedding(init_latent.shape[2], device)

        time_grid = torch.linspace(0, 1, steps + 1).to(device)
        audio_latent = solver.sample(
            time_grid=time_grid,
            x_init=init_latent,
            method=method,
            return_intermediates=False,
            step_size=None,
            c=cond, w=w,
            rotary_embedding=rotary,
        )

        audio_latent = audio_latent / self.scale_factor
        return self.audio_codec.decode(audio_latent).sample
