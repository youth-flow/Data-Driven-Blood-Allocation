"""SAA fits only the training demand distribution."""

from dat_study.runtime import prepare_fit
from .estimator import SAAEstimator


def fit_predict(data, config, recipe=None):
    training, evaluation, _, _ = prepare_fit(data, config, recipe)
    estimator = SAAEstimator().fit(training.daily.D.to_numpy())
    prediction = estimator.predict_quantiles(len(evaluation), config["taus"])
    return prediction, {"epochs": 0}, {"saa_quantiles": prediction[0]}
