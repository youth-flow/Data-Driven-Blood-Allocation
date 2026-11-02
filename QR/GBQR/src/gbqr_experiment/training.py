import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from dat_study.runtime import prepare_fit
from dat_study.tabular import TabularFeatures, ordered_quantiles
from dat_study.selection import selection_candidates, validation_score


def fit_predict(data, config, recipe=None):
    prepare_fit(data, config, recipe)
    features = TabularFeatures(config["gbqr"].get("input_set", "basic")).fit(data)
    train, evaluation = data.daily_split("train"), data.daily_split("eval")
    x, xe = features.transform(data, train), features.transform(data, evaluation)
    best, trials = None, []
    for params in selection_candidates(data, config, "gbqr", recipe):
        models = [
            HistGradientBoostingRegressor(
                loss="quantile", quantile=t, early_stopping=False, random_state=config["seed"], **params
            ).fit(x, train.D)
            for t in config["taus"]
        ]
        prediction = ordered_quantiles(np.column_stack([m.predict(xe) for m in models]))
        score = validation_score(prediction, data, config) if data.eval_role == "validation" else None
        trials.append({"parameters": params, "validation_mean_P": score})
        if best is None or (score is not None and score > best[0]):
            best = (score, params, prediction, models)
    _, params, prediction, models = best
    return prediction, {"epochs": 0, "selection": params, "trials": trials}, {"features": features, "models": models}
