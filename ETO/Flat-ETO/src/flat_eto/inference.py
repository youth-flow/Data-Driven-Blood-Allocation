from dat_study.runtime import device_for, seed_everything

from .model import FlatPatientPMF
from .training import _predict


def predict_artifact(state, data, config):
    seed_everything(config["seed"])
    model = FlatPatientPMF(
        22,
        config["flat_eto"]["patient_hidden_dims"],
        config["flat_eto"]["dropout"],
    )
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device_for(config))
    return _predict(
        model,
        state["forest"],
        state["train_leaves"],
        state["residual"],
        state["features"],
        data,
        config,
    )
