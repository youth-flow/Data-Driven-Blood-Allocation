"""Load only the requested model. Sibling model packages are never imported."""

from importlib import import_module
from pathlib import Path
import sys

PACKAGES = {
    "DAT": "datransformer",
    "DAT-3stage": "datransformer",
    "SAA": "saa_experiment",
    "wSAA": "wsaa_experiment",
    "Flat-ETO": "flat_eto",
    "Set-ETO": "set_eto",
}


def backend(method, operation):
    if method not in PACKAGES:
        raise ValueError("Unknown method")
    folder = "DAT" if method == "DAT-3stage" else method
    source = Path(__file__).resolve().parents[2] / folder / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = "trainer" if method.startswith("DAT") and operation == "training" else operation
    return import_module(f"{PACKAGES[method]}.{module}")
