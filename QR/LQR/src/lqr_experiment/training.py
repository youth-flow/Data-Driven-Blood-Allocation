import numpy as np
from sklearn.linear_model import QuantileRegressor

from dat_study.runtime import prepare_fit
from dat_study.tabular import PsiFeatures, ordered_quantiles


def fit_predict(data, config, recipe=None):
    prepare_fit(data, config, recipe)
    features = PsiFeatures().fit(data)
    train, evaluation = data.daily_split("train"), data.daily_split("eval")
    x, xe = features.transform(data, train), features.transform(data, evaluation)
    alpha = float(config["lqr"]["alpha"])
    models = [QuantileRegressor(quantile=tau, alpha=alpha, solver="highs").fit(x, train.D) for tau in config["taus"]]
    prediction = ordered_quantiles(np.column_stack([model.predict(xe) for model in models]))
    return prediction, {"epochs": 0, "alpha": alpha}, {"features": features, "models": models}
