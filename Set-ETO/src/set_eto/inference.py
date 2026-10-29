from dat_study.data import DailySets
from dat_study.runtime import seed_everything
from .training import new_model, predict


def predict_artifact(state, data, config):
    seed_everything(config["seed"])
    model = new_model(data, config)
    model.load_state_dict(state["model_state"], strict=True)
    return predict(model, "Set-ETO", DailySets(data, "eval"), data, config)
