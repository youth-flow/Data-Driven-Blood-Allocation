from .policy import receipt_allocations


def fit_predict(data, config, recipe=None):
    return receipt_allocations(data, config["taus"]), {"epochs": 0, "fitted": False}, {"output_kind": "allocation"}
