"""wSAA fits context weights using training days only."""

from dat_study.runtime import prepare_fit
from .estimators import RandomForestWeights, predict_weighted_quantiles


def fit_predict(data, config, recipe=None):
    training, evaluation, _, _ = prepare_fit(data, config, recipe)
    estimator = RandomForestWeights(seed=config["seed"], n_jobs=1, **config["wsaa"])
    demand = training.daily.D.to_numpy()
    estimator.fit(data.context(training.daily), demand)
    weights = estimator.weights(data.context(evaluation.daily))
    return (
        predict_weighted_quantiles(weights, demand, config["taus"]),
        {"epochs": 0},
        {"estimator": estimator, "training_demand": demand},
    )
