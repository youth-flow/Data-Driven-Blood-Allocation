from dat_study.runtime import prepare_fit
from dat_study.selection import selection_candidates, validation_score
from .estimator import rolling_quantiles


def fit_predict(data, config, recipe=None):
    prepare_fit(data, config, recipe)
    history, evaluation = data.daily_split("train"), data.daily_split("eval")
    best, trials = None, []
    for params in selection_candidates(data, config, "rolling_saa", recipe):
        if params["same_weekday"] and params["window"] < 91:
            continue
        prediction = rolling_quantiles(history, evaluation, config["taus"], **params)
        score = validation_score(prediction, data, config) if data.eval_role == "validation" else None
        trials.append({"parameters": params, "validation_mean_P": score})
        if best is None or (score is not None and score > best[0]):
            best = (score, params, prediction)
    if best is None:
        raise ValueError("No valid rolling SAA candidates")
    _, params, prediction = best
    return prediction, {"epochs": 0, "selection": params, "trials": trials}, {"parameters": params}
