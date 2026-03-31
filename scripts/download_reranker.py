"""Download the reranker model to a local directory for offline use.

Supports two download sources:
  1. HuggingFace Hub (default, requires network to huggingface.co)
  2. ModelScope (Chinese mirror, requires `pip install modelscope`)

Usage:
    # From HuggingFace (if accessible):
    python scripts/download_reranker.py

    # From ModelScope mirror (recommended in China):
    python scripts/download_reranker.py --source modelscope

    # Custom output directory:
    python scripts/download_reranker.py --output ./models/bge-reranker-v2-m3
"""

import argparse
import os
import ssl
import sys
import warnings
from pathlib import Path

import requests
from urllib3.exceptions import InsecureRequestWarning

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_OUTPUT = str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3")
HF_MIRROR = "https://hf-mirror.com"


# Files required for CrossEncoder to work
MODEL_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "model.safetensors",
]


def download_from_huggingface(model_id: str, output_dir: str) -> None:
    """Download model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., 'BAAI/bge-reranker-v2-m3').
        output_dir: Local directory to save the model.
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading from HuggingFace: {model_id}")
    print(f"Target directory: {output_dir}")

    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )

    print(f"[OK] Download complete: {output_dir}")


def download_from_mirror(model_id: str, output_dir: str) -> None:
    """Download model files from hf-mirror.com (works in China/corporate proxy).

    Downloads files individually with SSL verification disabled
    to bypass corporate proxy certificate issues.

    Args:
        model_id: HuggingFace model ID (e.g., 'BAAI/bge-reranker-v2-m3').
        output_dir: Local directory to save the model.
    """
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading from hf-mirror.com: {model_id}")
    print(f"Target directory: {output_dir}")
    print(f"(SSL verification disabled for corporate proxy compatibility)")
    print()

    session = requests.Session()
    session.verify = False

    for filename in MODEL_FILES:
        url = f"{HF_MIRROR}/{model_id}/resolve/main/{filename}"
        target = Path(output_dir) / filename

        if target.exists() and target.stat().st_size > 0:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  [OK] {filename} already exists ({size_mb:.1f} MB), skipping")
            continue

        print(f"  [DL] {filename} ... ", end="", flush=True)
        try:
            resp = session.get(url, timeout=300, stream=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        size_mb = downloaded / (1024 * 1024)
                        print(f"\r  [DL] {filename} ... {size_mb:.1f} MB ({pct}%)", end="", flush=True)

            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"\r  [OK] {filename} ({size_mb:.1f} MB)                    ")
        except Exception as e:
            print(f"\r  [FAIL] {filename}: {e}                    ")
            if target.exists():
                target.unlink()

    print(f"\n[OK] Download complete: {output_dir}")


def download_from_modelscope(model_id: str, output_dir: str) -> None:
    """Download model from ModelScope (Chinese mirror).

    Args:
        model_id: ModelScope model ID (e.g., 'BAAI/bge-reranker-v2-m3').
        output_dir: Local directory to save the model.
    """
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError:
        print(f"[FAIL] modelscope not installed. Install with: pip install modelscope")
        sys.exit(1)

    print(f"Downloading from ModelScope: {model_id}")
    print(f"Target directory: {output_dir}")

    ms_download(
        model_id=model_id,
        local_dir=output_dir,
    )

    print(f"[OK] Download complete: {output_dir}")


def verify_model(model_dir: str) -> bool:
    """Verify that the downloaded model can be loaded.

    Args:
        model_dir: Path to the downloaded model directory.

    Returns:
        True if the model loads successfully.
    """
    print(f"\nVerifying model at: {model_dir}")

    # Check required files exist
    model_path = Path(model_dir)
    required_files = ["config.json", "tokenizer_config.json"]
    weight_files = ["model.safetensors", "pytorch_model.bin"]

    for f in required_files:
        if not (model_path / f).exists():
            print(f"  [FAIL] Missing: {f}")
            return False
        print(f"  [OK] Found: {f}")

    has_weights = any((model_path / f).exists() for f in weight_files)
    if not has_weights:
        print(f"  [FAIL] Missing model weights (need one of: {weight_files})")
        return False
    print(f"  [OK] Model weights found")

    # Try loading with CrossEncoder
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_dir)
        # Quick test
        score = model.predict([("test query", "test document")])
        print(f"  [OK] Model loads and runs (test score: {score[0]:.4f})")
        return True
    except Exception as e:
        print(f"  [FAIL] Model load failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download reranker model for offline use")
    parser.add_argument(
        "--source",
        choices=["huggingface", "mirror", "modelscope"],
        default="mirror",
        help="Download source (default: mirror = hf-mirror.com)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing download, don't download",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    if args.verify_only:
        ok = verify_model(output_dir)
        sys.exit(0 if ok else 1)

    # Download
    if args.source == "mirror":
        download_from_mirror(args.model, output_dir)
    elif args.source == "modelscope":
        download_from_modelscope(args.model, output_dir)
    else:
        download_from_huggingface(args.model, output_dir)

    # Verify
    verify_model(output_dir)

    # Print config hint
    print(f"\n[INFO] Add this to config/settings.yaml under 'reranker':")
    print(f"  local_model_path: \"{output_dir}\"")


if __name__ == "__main__":
    main()
