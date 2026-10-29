"""Training-only conditional remainder for full-day demand.

The morning set is fixed before t. R = D - known_recent_D includes new arrivals
and earlier requests outside its age window. Neither current R nor current D
is fed to a prediction model.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class ConditionalRemainder:
    def __init__(self, neighbors=30):
        self.neighbors = int(neighbors)

    def fit(self, context, demand, known):
        remainder = np.asarray(demand, float) - np.asarray(known, float)
        if not np.isfinite(remainder).all() or (remainder < -1e-6).any():
            raise ValueError("Full-day labels require a known nonnegative training remainder")
        self.values = np.maximum(remainder, 0)
        self.model = NearestNeighbors(n_neighbors=min(self.neighbors, len(context))).fit(context)
        return self

    def pools(self, context):
        return self.values[self.model.kneighbors(context, return_distance=False)]

    def pmfs(self, context):
        result = []
        for values in self.pools(context):
            if not np.allclose(values * 2, np.round(values * 2), atol=1e-6):
                raise ValueError("Flat convolution requires half-unit quantities")
            counts = np.bincount(np.rint(values * 2).astype(int))
            result.append(counts / counts.sum())
        return result


def fit_remainder(data, neighbors):
    training = data.daily_split("train")
    return ConditionalRemainder(neighbors).fit(data.context(training), training.D, training.known_recent_D)
