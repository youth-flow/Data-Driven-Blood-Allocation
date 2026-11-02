"""Unconditional empirical-demand baseline."""

import numpy as np
from dat_study.evaluation import higher_quantile


class SAAEstimator:
    def fit(self, demand):
        values = np.asarray(demand, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("SAA requires finite nonnegative training demand")
        self.demand_ = values.copy()
        return self

    def predict_quantiles(self, n_days, taus):
        if not hasattr(self, "demand_"):
            raise RuntimeError("SAA must be fitted before prediction")
        values = np.array([higher_quantile(self.demand_, tau) for tau in taus])
        return np.repeat(values[None, :], int(n_days), axis=0)
