"""Root A/B/C launcher. Never generates data implicitly or opens real tables."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dat_study.synthetic_suite import main  # noqa: E402

if __name__ == "__main__":
    main(default_config=ROOT / "configs/synthetic.yaml", variants=ROOT / "DAT/configs/experiments.yaml")
