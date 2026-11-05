from dat_study.runtime import prepare_fit

from .estimator import rolling_quantiles


def fit_predict(data, config, recipe=None):
    prepare_fit(data, config, recipe)
    history, evaluation = data.daily_split("train"), data.daily_split("eval")
    parameters = dict(config["rolling_saa"])
    prediction = rolling_quantiles(history, evaluation, config["taus"], **parameters)
    return prediction, {"epochs": 0, "parameters": parameters}, {"parameters": parameters}
