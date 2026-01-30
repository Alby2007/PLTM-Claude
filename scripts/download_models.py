"""Download and cache models for offline use"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings


def download_models() -> None:
    """Download models to cache"""
    print("Model download script")
    print(f"Cache directory: {settings.MODEL_CACHE}")
    print(f"Small model: {settings.SMALL_MODEL}")
    print(f"Judge model: {settings.JUDGE_MODEL}")
    print("\nModels will auto-download on first use.")
    print("To pre-download, run:")
    print(f"  python -c 'from transformers import AutoModel; AutoModel.from_pretrained(\"{settings.SMALL_MODEL}\")'")
    print(f"  python -c 'from transformers import AutoModel; AutoModel.from_pretrained(\"{settings.JUDGE_MODEL}\")'")


if __name__ == "__main__":
    download_models()
