from .policy import receipt_allocations


def predict_artifact(state, data, config):
    return receipt_allocations(data, config["taus"])
