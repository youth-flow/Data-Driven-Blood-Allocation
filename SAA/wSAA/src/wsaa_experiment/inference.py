from .estimators import predict_weighted_quantiles


def predict_artifact(state, data, config):
    features = state["features"]
    weights = state["estimator"].weights(features.transform(data, data.daily_split("eval")))
    return predict_weighted_quantiles(weights, state["training_demand"], config["taus"])
