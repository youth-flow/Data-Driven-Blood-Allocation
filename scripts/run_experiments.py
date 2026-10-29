"""Run selected independent models against the shared root data directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dat_study.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "--config", str(ROOT / "configs/experiment.yaml"), *sys.argv[1:]])
