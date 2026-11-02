"""Shared selection bookkeeping, independent of model implementations."""

from itertools import product
from .evaluation import evaluate, selection_score


def candidates(grid):
    keys = list(grid)
    if not keys or any(not grid[key] for key in keys):
        raise ValueError("Selection grid cannot be empty")
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def validation_score(prediction, data, config):
    daily = data.daily_split("eval")
    return selection_score(
        evaluate(
            prediction,
            daily.D.to_numpy(),
            daily.It.to_numpy(),
            data.daily_split("train").D.to_numpy(),
            config["taus"],
            config["targets"],
        )
    )


def selection_candidates(data, config, section, recipe):
    if data.eval_role == "test":
        if recipe is None or "selection" not in recipe:
            raise ValueError("Test fitting requires frozen hyperparameters")
        return [recipe["selection"]]
    return candidates(config[section]["grid"])
