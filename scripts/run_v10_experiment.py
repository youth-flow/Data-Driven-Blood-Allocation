"""Run one frozen AXID-DC v10 stage from the attention environment."""

from pathlib import Path
import os
import sys

if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "synthetic/src"):
    sys.path.insert(0, str(source))

from axid_synthetic.pipeline_v10 import main  # noqa: E402


if __name__ == "__main__":
    main(root=ROOT)
