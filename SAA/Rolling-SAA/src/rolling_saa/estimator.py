import numpy as np


def rolling_quantiles(history, evaluation, taus, window, same_weekday=False):
    """Fixed window of the training demand sequence; no test-label updates."""
    if window < 1:
        raise ValueError("Rolling SAA requires a positive window")
    if same_weekday:
        raise ValueError("The S4 protocol does not use weekday filters")
    recent = history[["decision_day", "D"]].sort_values("decision_day").tail(window)
    if recent.empty:
        raise ValueError("Rolling SAA requires at least one training demand")
    quantiles = np.quantile(recent.D.to_numpy(float), np.asarray(taus, float), method="higher")
    return np.tile(quantiles, (len(evaluation), 1))
