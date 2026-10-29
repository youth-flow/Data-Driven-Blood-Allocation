from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from dat_study.evaluation import higher_quantile


def weighted_higher_quantile(values: np.ndarray, weights: np.ndarray, tau: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("Values and weights must be aligned vectors.")
    if (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError("Weights must be finite and nonnegative.")
    total = float(weights.sum())
    if total <= 0.0:
        return higher_quantile(values, tau)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / total
    index = int(np.searchsorted(cumulative, float(tau), side="left"))
    return float(values[order[min(index, len(values) - 1)]])


@dataclass
class KNNWeights:
    k: int

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNWeights":
        self.n_train_ = len(x)
        self.model_ = NearestNeighbors(n_neighbors=min(self.k, len(x))).fit(x)
        return self

    def weights(self, x: np.ndarray) -> np.ndarray:
        indices = self.model_.kneighbors(x, return_distance=False)
        result = np.zeros((len(x), self.n_train_), dtype=np.float64)
        result[np.arange(len(x))[:, None], indices] = 1.0 / indices.shape[1]
        return result


@dataclass
class TricubicKernelWeights:
    k: int

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TricubicKernelWeights":
        self.x_train_ = np.asarray(x, dtype=np.float64)
        self.neighbors_ = NearestNeighbors(n_neighbors=min(self.k, len(x))).fit(x)
        return self

    def weights(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        bandwidth = np.maximum(self.neighbors_.kneighbors(x)[0][:, -1], 1e-12)
        distances = np.sqrt(((x[:, None] - self.x_train_[None]) ** 2).sum(axis=2))
        ratio = distances / bandwidth[:, None]
        weights = np.where(ratio <= 1.0, (1.0 - ratio**3) ** 3, 0.0)
        zero = weights.sum(axis=1) <= 0.0
        if zero.any():
            weights[zero] = 0.0
            weights[np.flatnonzero(zero), np.argmin(distances[zero], axis=1)] = 1.0
        return weights / weights.sum(axis=1, keepdims=True)


@dataclass
class RandomForestWeights:
    n_estimators: int
    min_samples_leaf: int
    max_features: str | float
    seed: int
    n_jobs: int = -1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestWeights":
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            bootstrap=True,
        ).fit(x, y)
        self.train_leaves_ = self.model_.apply(x)
        return self

    def weights(self, x: np.ndarray) -> np.ndarray:
        query = self.model_.apply(x)
        result = np.zeros((len(x), len(self.train_leaves_)), dtype=np.float64)
        for tree in range(self.train_leaves_.shape[1]):
            train_leaf = self.train_leaves_[:, tree]
            for row, leaf in enumerate(query[:, tree]):
                members = train_leaf == leaf
                result[row, members] += 1.0 / members.sum()
        result /= self.train_leaves_.shape[1]
        return result / result.sum(axis=1, keepdims=True)


def predict_weighted_quantiles(weights, train_demand, taus):
    return np.array(
        [[weighted_higher_quantile(train_demand, row, tau) for tau in taus] for row in weights], dtype=np.float64
    )


def weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    return {
        "mean_nonzero_weights": float((weights > 0).sum(axis=1).mean()),
        "mean_effective_sample_size": float((1.0 / np.maximum((weights**2).sum(1), 1e-12)).mean()),
        "mean_max_weight": float(weights.max(axis=1).mean()),
    }
