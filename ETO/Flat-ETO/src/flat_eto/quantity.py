"""Patient PMF convolution and weighted empirical new-demand CDF."""

import numpy as np


def convolve_patient_pmfs(probabilities):
    """Convolve one seven-class PMF per visible patient on the 0.5 grid."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 7:
        raise ValueError("Patient probabilities must have shape [N,7]")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid patient probabilities")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True).clip(1e-300)
    pmf = np.array([1.0])
    for row in probabilities:
        pmf = np.convolve(pmf, row)
    return pmf


class NewDemandCDF:
    """Weighted empirical CDF of the continuous residual demand R_t."""

    def __init__(self, values, weights):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.ndim != 1 or weights.shape != values.shape or not len(values):
            raise ValueError("Invalid residual-demand sample")
        order = np.argsort(values, kind="stable")
        self.values = values[order]
        cumulative = np.cumsum(weights[order])
        total = float(cumulative[-1])
        if total <= 0:
            raise ValueError("Residual-demand weights must have positive mass")
        self.cumulative = cumulative / total

    def __call__(self, point):
        index = int(np.searchsorted(self.values, float(point), side="right"))
        return 0.0 if index == 0 else float(self.cumulative[index - 1])


def flat_quantile(patient_probabilities, new_demand: NewDemandCDF, tau, upper, step=0.5):
    pmf = convolve_patient_pmfs(patient_probabilities)
    grid = float(step) * np.arange(len(pmf))

    def cdf(point):
        return float(np.sum(pmf * np.array([new_demand(point - atom) for atom in grid])))

    if cdf(float(upper)) < tau:
        return float(upper)
    low, high = 0.0, float(upper)
    for _ in range(50):
        middle = 0.5 * (low + high)
        if cdf(middle) >= tau:
            high = middle
        else:
            low = middle
    return float(high)
