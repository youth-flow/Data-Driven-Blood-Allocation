"""Root-level dispatch only; each model owns its training implementation."""

from .config import model_config
from .registry import backend
from .runtime import seed_everything as seed_everything


def fit_predict(method, data, config, recipe=None):
    selected = model_config(config, method)
    return backend(method, "training").fit_predict(data, selected, recipe)
