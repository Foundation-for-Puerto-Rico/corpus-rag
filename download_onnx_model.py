#!/usr/bin/env python3
"""Download the Nomic Embed Text v1.5 ONNX model from HuggingFace Hub.

Downloads:
  - onnx/model.onnx (FP32, ~547 MB)
  - tokenizer.json, tokenizer_config.json, special_tokens_map.json

Destination: ~/wcrp-rag/models/nomic-embed-text-v1.5/

Usage:
    python download_onnx_model.py
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_DIR = Path.home() / "wcrp-rag" / "models" / "nomic-embed-text-v1.5"

FILES = [
    ("onnx/model.onnx", "model.onnx"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
]


def download():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for remote_path, local_name in FILES:
        dest = MODEL_DIR / local_name
        if dest.exists():
            print(f"  Already exists: {dest}")
            continue
        print(f"  Downloading {remote_path} ...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            local_dir=str(MODEL_DIR),
        )
        # hf_hub_download may place files in subdirectories; move to flat layout
        downloaded_path = Path(downloaded)
        if downloaded_path != dest:
            downloaded_path.rename(dest)
            # Clean up empty parent dirs left by hf_hub_download
            for parent in downloaded_path.parents:
                if parent == MODEL_DIR:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
        print(f"  Saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    print(f"\nModel ready at: {MODEL_DIR}")


if __name__ == "__main__":
    download()
