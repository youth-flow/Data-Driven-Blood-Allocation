from copy import deepcopy
from pathlib import Path
import yaml


METHODS = ("SAA", "wSAA", "Set-ETO", "Flat-ETO", "DAT", "DAT-3stage")
FOLDS = ("val2017", "val2018", "val2019", "final2020")


def _merge(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def load_config(path, _visited=None):
    path = Path(path).resolve()
    visited = set() if _visited is None else set(_visited)
    if path in visited:
        raise ValueError("Configuration inheritance cycle")
    visited.add(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent is not None:
        config = _merge(load_config(path.parent / parent, visited), config)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment configuration version")
    for key in ("data_root", "output_root"):
        config[key] = str((path.parent / config[key]).resolve())
    if "model_configs" in config:
        config["model_configs"] = {
            method: str((path.parent / filename).resolve()) for method, filename in config["model_configs"].items()
        }
    for key in ("taus", "targets"):
        values = config[key]
        if not values or values != sorted(set(values)):
            raise ValueError(f"{key} must be nonempty, unique and increasing")
    if any(not 0 < t < 1 for t in config["taus"]):
        raise ValueError("Quantile levels must be in (0, 1)")
    if "loss" in config and config["taus"] != config["loss"]["taus"]:
        raise ValueError("Model and evaluation quantile levels must agree")
    if "three_stage" in config and config["three_stage"]["finetune"] not in ("lora", "full"):
        raise ValueError("Unknown fine-tuning mode")
    return config


def smoke_config(config):
    config = deepcopy(config)
    config["_smoke"] = True
    if "training" in config:
        config["training"].update(epochs=1, batch_size=8, patience=1)
    if "three_stage" in config:
        config["three_stage"].update(order_epochs=1, synthetic_epochs=1, scenarios_per_day=1)
    if "wsaa" in config:
        config["wsaa"]["n_estimators"] = 8
    return config


def model_config(config, method):
    """Resolve one model's own file; the root controls only common experiment settings."""
    base_method = "DAT" if method == "DAT-3stage" else method
    if method not in METHODS:
        raise ValueError("Unknown method")
    if "model_configs" in config:
        selected = load_config(config["model_configs"][base_method])
        if selected.get("method") != base_method:
            raise ValueError("Model configuration belongs to a different method")
        for key in ("data_root", "output_root", "seed", "taus", "targets"):
            selected[key] = deepcopy(config[key])
        # Optional common runtime overrides, e.g. CPU-only smoke verification.
        selected["training"] = _merge(selected["training"], config.get("training", {}))
        if "loss" in selected:
            selected["loss"]["taus"] = list(selected["taus"])
    else:
        selected = deepcopy(config)
        if selected.get("method", base_method) != base_method:
            raise ValueError("Model configuration belongs to a different method")
    return smoke_config(selected) if config.get("_smoke") else selected
