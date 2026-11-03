from __future__ import annotations

import numpy as np
from scipy import stats


def distribution_quantiles(distribution, parameters, taus):
    tau = np.asarray(taus, dtype=np.float64)[None, :]
    if distribution == "Normal":
        values = parameters["mu"][:, None] + parameters["sigma"][:, None] * stats.norm.ppf(tau)
        return np.maximum(values, 0.0)
    if distribution in {"NegativeBinomial", "SmoothedNegativeBinomial"}:
        mean = parameters["mean_count"][:, None]
        dispersion = parameters["dispersion"][:, None]
        probability = dispersion / (dispersion + mean)
        count = stats.nbinom.ppf(tau, dispersion, probability)
        if distribution == "SmoothedNegativeBinomial":
            previous = stats.nbinom.cdf(count - 1, dispersion, probability)
            mass = stats.nbinom.pmf(count, dispersion, probability)
            return count / 2 + 0.5 * np.clip((tau - previous) / np.maximum(mass, 1e-300), 0, 1)
        return count / 2.0
    zero_probability = 1.0 / (1.0 + np.exp(-parameters["zero_logit"]))
    positive_tau = np.clip(
        (tau - zero_probability[:, None]) / np.maximum(1.0 - zero_probability[:, None], 1e-12),
        1e-10,
        1.0 - 1e-10,
    )
    positive = stats.gamma.ppf(
        positive_tau,
        a=parameters["shape"][:, None],
        scale=1.0 / parameters["rate"][:, None],
    )
    return np.where(tau <= zero_probability[:, None], 0.0, positive)
