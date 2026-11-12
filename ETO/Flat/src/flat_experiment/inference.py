"""Flat v7 artifact reload."""

from dat_study.runtime import device_for, seed_everything

from .model import FlatMSE
from .training import _predict


def predict_artifact(state, data, config):
    seed_everything(config["seed"])
    settings = config["flat"]
    model = FlatMSE(settings["patient_hidden"], settings["residual_hidden"], settings["dropout"])
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device_for(config))
    model.eval()
    return _predict(model, data, config)
