"""Root-level dispatch only; each model owns its training implementation."""

from .config import model_config
from .registry import backend
from .runtime import seed_everything as seed_everything


def fit_predict(method, data, config, recipe=None):
    selected = model_config(config, method)
    kwargs = {"three_stage": method == "DAT-3stage"} if method.startswith("DAT") else {}
    return backend(method, "training").fit_predict(data, selected, recipe, **kwargs)
