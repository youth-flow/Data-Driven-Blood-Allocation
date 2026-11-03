"""wSAA fits context weights using training days only."""

from dat_study.runtime import prepare_fit
from .estimators import RandomForestWeights, predict_weighted_quantiles
from dat_study.selection import candidates, validation_score


def fit_predict(data, config, recipe=None):
    training, evaluation, _, _ = prepare_fit(data, config, recipe)
    settings = dict(config["wsaa"])
    grid = settings.pop("grid", None)
    demand = training.daily.D.to_numpy()
    if data.eval_role == "test" and grid and (recipe is None or "selection" not in recipe):
        raise ValueError("Test wSAA requires frozen hyperparameters")
    choices = [recipe["selection"]] if recipe and "selection" in recipe else candidates(grid) if grid else [{}]
    best, trials = None, []
    for params in choices:
        estimator = RandomForestWeights(seed=config["seed"], n_jobs=1, **{**settings, **params})
        estimator.fit(data.context(training.daily), demand)
        weights = estimator.weights(data.context(evaluation.daily))
        prediction = predict_weighted_quantiles(weights, demand, config["taus"])
        score = validation_score(prediction, data, config) if data.eval_role == "validation" else None
        trials.append({"parameters": params, "validation_score": score})
        if best is None or (score is not None and score > best[0]):
            best = (score, params, prediction, estimator)
    _, params, prediction, estimator = best
    return (
        prediction,
        {"epochs": 0, "selection": params, "trials": trials},
        {"estimator": estimator, "training_demand": demand},
    )
