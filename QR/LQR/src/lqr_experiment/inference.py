import numpy as np
from dat_study.tabular import ordered_quantiles


def predict_artifact(state, data, config):
    x = state["features"].transform(data, data.daily_split("eval"))
    return ordered_quantiles(np.column_stack([m.predict(x) for m in state["models"]]))
