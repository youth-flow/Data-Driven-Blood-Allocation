"""Reload the independently trained scalar models without fitting to evaluation data."""

import numpy as np
from dat_study.data import DailySets
from dat_study.runtime import device_for
from .model import build_model
from .trainer import predict


def predict_artifact(state, data, config):
    if "scalar_models" not in state:
        raise ValueError("Legacy DAT weights are incompatible with the simplified architecture")
    result = []
    for fitted in state["scalar_models"]:
        selected = fitted["config"]
        model = build_model(selected, state["metadata"]).to(device_for(selected))
        model.load_state_dict(fitted["model_state"], strict=True)
        result.append(predict(model, "DAT", DailySets(data, "eval"), data, selected))
    return np.concatenate(result, axis=1)
