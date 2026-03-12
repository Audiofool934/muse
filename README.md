# MUSE: Music Unified Synthesis Engine

A modular framework for generating music from **any input modality** — text, image, video, or audio — through a unified two-stage flow matching architecture.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │           Perception Layer (Frozen)           │
                    │                                              │
  Text ──────────── │  T5 Encoder ──────────────┐                  │
  Image ─────────── │  CLIP / SigLIP ───────────┤ ConditioningOutput│
  Video ─────────── │  Frame Encoder ───────────┤   [B, L, 768]   │
  Audio ─────────── │  MuQ-MuLan ───────────────┤                  │
  Any (zero-shot) ─ │  MLLM Bridge (Gemma-3) ──┘                  │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │      Stage 1: Conditional Flow Matching       │
                    │   Cross-Attention Transformer (16L, 1024d)   │
                    │         ODE: noise → MuQ-MuLan [512]         │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │         Stage 2: Audio Synthesis              │
                    │   Stable Audio DiT (24L) + Oobleck VAE       │
                    │     ODE: noise → audio latent → 44.1kHz      │
                    └──────────────────────────────────────────────┘
```

**Key insight**: Stage 2 is modality-agnostic — it only sees a 512-dim latent vector. Adding a new input modality only requires a new perception encoder and training Stage 1.

## Results (Text-to-Music baseline)

Evaluated on MusicBench (2,811 samples), compared against AudioLDM and MusicGen:

| Model | FAD ↓ | CLAP ↑ | KL Sigmoid ↑ |
|-------|-------|--------|--------------|
| AudioLDM | 3.82 | 0.368 | 0.744 |
| MusicGen | 5.36 | 0.307 | 0.844 |
| **MUSE (t2m)** | **2.25** | 0.326 | **0.925** |

- **FAD 2.25** — 41% improvement over AudioLDM, indicating higher audio quality and distribution match.
- **KL Sigmoid 0.925** — best prompt-audio semantic alignment.
- Output: stereo 44.1kHz, ~12 seconds.

### Probabilistic Generation

The flow matching formulation models a full distribution $p(z \mid c)$ rather than a deterministic mapping. This enables:

- **Diverse outputs**: multiple distinct samples from the same prompt
- **Uncertainty calibration**: vague prompts produce higher variance (APD 1.037) than specific prompts (APD 0.963)
- **Latent multimodality**: ambiguous prompts yield clustered sub-genres (e.g., "Cyberpunk city" → synthwave / ambient / industrial clusters)
- **Smooth interpolation**: continuous latent space traversal via noise-space SLERP

## Supported Pipelines

| Config | Input | Encoder | Training Required |
|--------|-------|---------|-------------------|
| `t2m_flow.yaml` | Text | Flan-T5 | Stage 1 + 2 (done) |
| `i2m_flow.yaml` | Image | CLIP ViT | Stage 1 only |
| `i2m_bridge.yaml` | Image | Gemma-3 → T5 | None (zero-shot) |
| `v2m_flow.yaml` | Video | CLIP frames | Stage 1 only |
| `a2m_flow.yaml` | Audio | MuQ-MuLan | Stage 1 only |

All pipelines reuse the same Stage 2 weights — only the perception encoder and Stage 1 differ.

## Usage

```python
from muse.pipelines import TwoStageFlowPipeline

# Text → Music
pipe = TwoStageFlowPipeline.from_config("configs/t2m_flow.yaml")
audio = pipe.generate("A melancholic cello solo over soft rain")
pipe.save_audio(audio, "output.wav")

# Image → Music (zero-shot, no additional training)
pipe = TwoStageFlowPipeline.from_config("configs/i2m_bridge.yaml")
audio = pipe.generate("sunset.jpg")

# Video → Music
pipe = TwoStageFlowPipeline.from_config("configs/v2m_flow.yaml")
audio = pipe.generate("timelapse.mp4")
```

```bash
# CLI
python scripts/generate.py --config configs/t2m_flow.yaml \
    --input "A cheerful piano melody" --output output.wav
```

## Design

### Three-layer decoupling

| Layer | Role | Interface |
|-------|------|-----------|
| Perception | Modality → embeddings | `PerceptionEncoder.encode() → ConditioningOutput` |
| Generation | Embeddings → MuQ-MuLan | `Cond2LatentFlow.generate() → [B, 512]` |
| Synthesis | MuQ-MuLan → waveform | `LatentToAudioDiT.sample() → [B, 2, T]` |

`ConditioningOutput` is the universal contract between layers — a `[B, L, D]` embedding tensor with a padding mask. All encoders produce it; all generators consume it.

### Config-driven assembly

Switch modality by changing one field in YAML:

```yaml
perception:
  type: clip                      # ← swap this
  model_name: openai/clip-vit-large-patch14
  target_dim: 768
```

The encoder registry maps type names to classes. Custom encoders can be registered by fully qualified class path.

### Sampling strategies

For latent exploration beyond single-sample generation:

| Strategy | Description |
|----------|-------------|
| `random` | N i.i.d. samples from the flow model |
| `mean` | Closest point to distribution centroid |
| `diverse` | Greedy max-min distance selection |
| `dbscan` | Adaptive density-based clustering |
| `kmeans` | Elbow-method auto-K clustering |

## Project Structure

```
MUSE/
├── muse/
│   ├── perception/          # Modality encoders
│   │   ├── text.py          #   T5 (768d)
│   │   ├── image.py         #   CLIP / SigLIP
│   │   ├── video.py         #   Temporal frame encoding
│   │   ├── audio.py         #   MuQ-MuLan (512d)
│   │   └── mllm_bridge.py   #   Gemma-3 zero-shot bridge
│   ├── generation/
│   │   └── flow_matching/
│   │       ├── transformer.py  # Stage 1: Cond2LatentFlow
│   │       └── dit.py          # Stage 2: LatentToAudioDiT
│   ├── pipelines/
│   │   └── two_stage_flow.py   # TwoStageFlowPipeline
│   ├── sampling/
│   │   └── strategies.py       # Latent selection strategies
│   ├── data/
│   │   └── dataset.py          # MultiModalMusicDataset
│   └── training/
│       └── trainer.py          # MuseTrainer interface
├── configs/                 # Pipeline YAML configs
├── scripts/                 # CLI tools
└── course/                  # Academic report & slides
```

## Technical Details

### Flow Matching

Both stages use Conditional Flow Matching (Lipman et al., 2023) with the optimal-transport affine path:

$$x_t = (1-t)\,x_0 + t\,x_1, \quad x_0 \sim \mathcal{N}(0, I),\; x_1 = z$$

The model learns a velocity field $v_\theta(x_t, t, c)$ with MSE loss against the target velocity $u_t = x_1 - x_0$. At inference, an ODE solver (Euler / midpoint / DOPRI5) integrates from noise to data.

Classifier-free guidance interpolates conditional and unconditional velocities:

$$v_\text{guided} = v_\text{uncond} + w\,(v_\text{cond} - v_\text{uncond})$$

### MuQ-MuLan bottleneck

All modalities are mapped through the MuQ-MuLan latent space (512-dim, L2-normalized). This provides:
- Compact, semantically aligned representation
- Shared interface for Stage 2
- Contrastive text-audio alignment for evaluation

## Dependencies

```
torch, torchaudio, transformers, diffusers, flow-matching
scikit-learn (sampling), decord (video), muq (audio encoder)
```

## References

- Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)
- Stable Audio Open (Evans et al., 2024)
- MuQ-MuLan: contrastive audio-text embeddings
