from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "models" / "dinov2-small"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public DINOv2 ViT-S/14 checkpoint.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--repo-id", default="facebook/dinov2-small")
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.model_dir),
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "README.md",
        ],
    )
    required = [args.model_dir / "config.json"]
    if not required[0].is_file() or not (args.model_dir / "model.safetensors").is_file():
        raise RuntimeError(f"Model download incomplete; inspect {path}")
    print(f"DINOv2-S/14 downloaded to: {args.model_dir.resolve()}")


if __name__ == "__main__":
    main()
