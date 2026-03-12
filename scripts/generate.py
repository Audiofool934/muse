"""
Unified MUSE inference script.

Usage:
    # Text-to-Music
    python scripts/generate.py --config configs/t2m_flow.yaml \
        --input "A calm piano melody" --output output_t2m.wav

    # Image-to-Music
    python scripts/generate.py --config configs/i2m_flow.yaml \
        --input path/to/image.jpg --output output_i2m.wav

    # Video-to-Music
    python scripts/generate.py --config configs/v2m_flow.yaml \
        --input path/to/video.mp4 --output output_v2m.wav
"""

import argparse

import torch

from muse.pipelines import TwoStageFlowPipeline


def main():
    parser = argparse.ArgumentParser(description="MUSE: Unified Music Generation")
    parser.add_argument("--config", type=str, required=True, help="Pipeline config YAML")
    parser.add_argument("--input", type=str, required=True, help="Input (text, image path, or video path)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    pipe = TwoStageFlowPipeline.from_config(args.config, device=args.device)

    print(f"Generating music from: {args.input}")
    audio = pipe.generate(args.input, seed=args.seed)

    pipe.save_audio(audio, args.output, sample_rate=pipe.sample_rate)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
