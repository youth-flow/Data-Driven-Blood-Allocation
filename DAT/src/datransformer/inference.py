from dat_study.data import DailySets
from dat_study.runtime import device_for, seed_everything
from .trainer import new_dat, predict


def predict_artifact(state, data, config):
    seed_everything(config["seed"])
    model = new_dat(data, config)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device_for(config))
    return predict(model, "DAT", DailySets(data, "eval"), data, config)
