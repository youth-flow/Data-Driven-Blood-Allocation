"""Load only the requested model. Sibling model packages are never imported."""

from importlib import import_module
from pathlib import Path
import sys

PACKAGES = {
    "DAT": "datransformer",
    "SAA": "saa_experiment",
    "wSAA": "wsaa_experiment",
    "Flat-ETO": "flat_eto",
    "Set-ETO": "set_eto",
    "Rolling-SAA": "rolling_saa",
    "LQR": "lqr_experiment",
    "GBQR": "gbqr_experiment",
    "DeepQR": "deep_qr",
    "Historical-Receipt": "historical_receipt",
}

MODEL_PATHS = {
    "DAT": "DAT",
    "SAA": "SAA/SAA",
    "wSAA": "SAA/wSAA",
    "Rolling-SAA": "SAA/Rolling-SAA",
    "Flat-ETO": "ETO/Flat-ETO",
    "Set-ETO": "ETO/Set-ETO",
    "LQR": "QR/LQR",
    "GBQR": "QR/GBQR",
    "DeepQR": "QR/DeepQR",
    "Historical-Receipt": "Historical-Receipt",
}


def backend(method, operation):
    if method not in PACKAGES:
        raise ValueError("Unknown method")
    folder = MODEL_PATHS[method]
    source = Path(__file__).resolve().parents[2] / folder / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = "trainer" if method.startswith("DAT") and operation == "training" else operation
    return import_module(f"{PACKAGES[method]}.{module}")
