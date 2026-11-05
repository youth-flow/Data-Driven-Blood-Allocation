"""wSAA fits fixed random-forest leaf weights using training days only."""

from dat_study.runtime import prepare_fit
from dat_study.tabular import PsiFeatures

from .estimators import RandomForestWeights, predict_weighted_quantiles


def fit_predict(data, config, recipe=None):
    training, evaluation, _, _ = prepare_fit(data, config, recipe)
    features = PsiFeatures().fit(data)
    settings = dict(config["wsaa"])
    demand = training.daily.D.to_numpy()
    estimator = RandomForestWeights(seed=config["seed"], n_jobs=1, **settings)
    estimator.fit(features.transform(data, training.daily), demand)
    weights = estimator.weights(features.transform(data, evaluation.daily))
    prediction = predict_weighted_quantiles(weights, demand, config["taus"])
    return (
        prediction,
        {"epochs": 0, "parameters": settings},
        {"estimator": estimator, "features": features, "training_demand": demand},
    )
