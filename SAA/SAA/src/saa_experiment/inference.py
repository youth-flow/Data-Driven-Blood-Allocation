import numpy as np


def predict_artifact(state, data, config):
    return np.tile(state["saa_quantiles"], (len(data.daily_split("eval")), 1))
