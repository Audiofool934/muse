# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MUSE (Music Unified Synthesis Engine) is a unified framework for multi-modal music generation. It supports text, image, video, and audio conditioned music synthesis through a modular three-layer architecture.

### Architecture: Three Decoupled Layers

```
Layer 1: Perception    — Any modality → ConditioningOutput [B, L, D]
Layer 2: Generation    — ConditioningOutput → MuQ-MuLan latent [B, 512]
Layer 3: Synthesis     — Latent → 44.1kHz audio waveform
```

Each layer is independently swappable via config. The same Stage 1 + Stage 2 generation backbone works for t2m, i2m, v2m, and a2m by only changing the perception encoder.

### Relationship to Other Projects

- `../t2m/` — The original text-to-music implementation. MUSE generalizes t2m's architecture to multiple modalities. Stage 1/2 weights are directly reusable.
- `../🗄️/WeaveWave/` — MLLM text-bridging approach (Gemma-3 → MusicGen). Integrated as `MLLMBridgeEncoder` in the perception layer.

## Project Structure

```
MUSE/
├── muse/                              # Core library
│   ├── perception/                    # Layer 1: Modality encoders
│   │   ├── base.py                    # PerceptionEncoder ABC + ConditioningOutput
│   │   ├── text.py                    # T5PerceptionEncoder (from t2m)
│   │   ├── image.py                   # CLIPPerceptionEncoder, SigLIPPerceptionEncoder
│   │   ├── video.py                   # VideoFramePerceptionEncoder
│   │   ├── audio.py                   # MuQMuLanPerceptionEncoder (style transfer)
│   │   └── mllm_bridge.py            # MLLMBridgeEncoder (from WeaveWave)
│   ├── generation/                    # Layer 2: Generative models
│   │   ├── base.py                    # LatentGenerator, AudioSynthesizer ABCs
│   │   └── flow_matching/
│   │       ├── transformer.py         # Cond2LatentFlow (Stage 1)
│   │       └── dit.py                 # LatentToAudioDiT (Stage 2)
│   ├── pipelines/                     # End-to-end assembly
│   │   ├── base.py                    # BasePipeline
│   │   └── two_stage_flow.py          # TwoStageFlowPipeline + encoder registry
│   ├── sampling/
│   │   └── strategies.py              # Peak, diverse, DBSCAN, k-means selection
│   ├── data/
│   │   └── dataset.py                 # MultiModalMusicDataset
│   └── training/
│       └── trainer.py                 # MuseTrainer interface (impl deferred)
├── configs/                           # YAML pipeline configs
│   ├── t2m_flow.yaml                  # Text → Music (direct)
│   ├── i2m_flow.yaml                  # Image → Music (CLIP direct)
│   ├── i2m_bridge.yaml                # Image → Music (MLLM bridge, zero-shot)
│   ├── v2m_flow.yaml                  # Video → Music (frame encoding)
│   └── a2m_flow.yaml                  # Audio → Music (style transfer)
├── scripts/
│   └── generate.py                    # Unified CLI inference
└── course/                            # Academic report & slides (XeLaTeX)
```

## Quick Start

```python
from muse.pipelines import TwoStageFlowPipeline

# Text-to-Music (reuses t2m weights)
pipe = TwoStageFlowPipeline.from_config("configs/t2m_flow.yaml")
audio = pipe.generate("A calm piano melody")

# Image-to-Music (zero-shot via MLLM bridge — no training needed)
pipe = TwoStageFlowPipeline.from_config("configs/i2m_bridge.yaml")
audio = pipe.generate("path/to/image.jpg")

# Image-to-Music (direct CLIP encoding — requires i2m training)
pipe = TwoStageFlowPipeline.from_config("configs/i2m_flow.yaml")
audio = pipe.generate("path/to/image.jpg")

# Video-to-Music
pipe = TwoStageFlowPipeline.from_config("configs/v2m_flow.yaml")
audio = pipe.generate("path/to/video.mp4")
```

## Perception Encoder Registry

| Type | Class | Modality | Training? | Source |
|------|-------|----------|-----------|--------|
| `t5` | T5PerceptionEncoder | text | t2m Stage 1 | t2m |
| `clip` | CLIPPerceptionEncoder | image | Needs i2m Stage 1 | New |
| `siglip` | SigLIPPerceptionEncoder | image | Needs i2m Stage 1 | New |
| `video_frames` | VideoFramePerceptionEncoder | video | Needs v2m Stage 1 | New |
| `muq_mulan` | MuQMuLanPerceptionEncoder | audio | Needs a2m Stage 1 | t2m |
| `mllm_bridge` | MLLMBridgeEncoder | image/video | Zero-shot (reuse t2m) | WeaveWave |

## Key Design Decisions

1. **ConditioningOutput** is the universal contract between perception and generation.
2. **Stage 2 is fully reusable** across all modalities — it only sees a 512-dim latent.
3. **Perception encoders are frozen** and output-projected to 768-dim for Stage 1 compatibility.
4. **Config-driven assembly** — switch modality by changing `perception.type` in YAML.
5. **Two approaches to multimodal**: direct encoding (needs training) vs. MLLM bridge (zero-shot).

## Build Commands (Course Materials)

```bash
cd course/report && xelatex report.tex   # Build report (run twice)
cd course/slides && xelatex slides.tex   # Build slides (run twice)
```

## Dependencies

Core: `torch`, `transformers`, `diffusers`, `flow-matching`, `torchaudio`
Image: `Pillow` (via transformers)
Video: `decord`
Sampling: `scikit-learn` (for clustering strategies)
MLLM bridge: `transformers>=4.50` (for Gemma-3)
Audio encoder: `muq`
