"""Model-neutral integrity checks and lazy predictor dispatch."""

from pathlib import Path
import joblib
from .data import FoldData
from .io import read_json, sha256
from .config import model_config
from .registry import backend


def predict_artifact(payload, data):
    if payload["data_identity"] != data.identity:
        raise ValueError("Predictor requires its original preprocessing state and fold")
    method = payload["method"]
    return backend(method, "inference").predict_artifact(
        payload["artifact"], data, model_config(payload["config"], method)
    )


def load_local_prediction(run_directory, method, data_directory):
    root = Path(run_directory).resolve()
    manifest = read_json(root / "manifest.json")
    if not manifest["complete"] or method not in manifest["methods"]:
        raise ValueError("Incomplete run or unavailable method")
    name = f"{method}/predictor.joblib"
    expected = manifest["artifacts"].get(name, manifest["artifacts"].get(name.replace("/", "\\")))
    if expected is None or sha256(root / name) != expected:
        raise ValueError("Predictor integrity mismatch")
    # Joblib/pickle is for artifacts generated locally by this project, never an untrusted download.
    payload = joblib.load(root / name)
    data = FoldData(data_directory)
    if manifest["smoke"]:
        data.limit_for_smoke()
    return data.daily_split("eval"), predict_artifact(payload, data), payload["config"]["taus"]
