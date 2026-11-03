"""One root entry for theory and performance experiments on artificial data."""

from pathlib import Path
import os
import sys

# Avoid loading competing MKL/PyTorch OpenMP runtimes on Windows/Conda.
# PyTorch still uses the requested training threads; MKL runs sequentially.
if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "synthetic/src"):
    sys.path.insert(0, str(source))
from axid_synthetic.study import main  # noqa: E402

if __name__ == "__main__":
    main()
