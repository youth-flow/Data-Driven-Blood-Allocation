"""Standalone SAA experiment entrypoint."""

from pathlib import Path
import sys

MODEL_ROOT = Path(__file__).resolve().parents[1]
ROOT = MODEL_ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MODEL_ROOT / "src"))
from dat_study.cli import model_main  # noqa: E402

if __name__ == "__main__":
    model_main("SAA", MODEL_ROOT / "configs/main.yaml")
