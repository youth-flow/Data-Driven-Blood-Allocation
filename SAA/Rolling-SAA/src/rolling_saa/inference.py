from .estimator import rolling_quantiles


def predict_artifact(state, data, config):
    return rolling_quantiles(data.daily_split("train"), data.daily_split("eval"), config["taus"], **state["parameters"])
