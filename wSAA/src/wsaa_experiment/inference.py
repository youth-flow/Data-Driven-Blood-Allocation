from .estimators import predict_weighted_quantiles


def predict_artifact(state, data, config):
    weights = state["estimator"].weights(data.context(data.daily_split("eval")))
    return predict_weighted_quantiles(weights, state["training_demand"], config["taus"])
