from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
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
    max_samples: float | None = None
    max_depth: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestWeights":
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            bootstrap=True,
            max_samples=self.max_samples,
            max_depth=self.max_depth,
        ).fit(x, y)
        self.train_leaves_ = self.model_.apply(x)
        self.leaf_members_ = []
        for tree in range(self.train_leaves_.shape[1]):
            leaves = self.train_leaves_[:, tree]
            order = np.argsort(leaves, kind="stable")
            values, starts = np.unique(leaves[order], return_index=True)
            ends = np.r_[starts[1:], len(order)]
            self.leaf_members_.append(
                {int(value): order[start:end].astype(np.int32) for value, start, end in zip(values, starts, ends)}
            )
        return self

    def weights(self, x: np.ndarray) -> np.ndarray:
        if len(x) * len(self.train_leaves_) > 50_000_000:
            raise MemoryError("Dense forest weights are disabled at v10 scale; use predict_quantiles")
        query = self.model_.apply(x)
        result = np.zeros((len(x), len(self.train_leaves_)), dtype=np.float64)
        for tree in range(self.train_leaves_.shape[1]):
            train_leaf = self.train_leaves_[:, tree]
            for row, leaf in enumerate(query[:, tree]):
                members = train_leaf == leaf
                result[row, members] += 1.0 / members.sum()
        result /= self.train_leaves_.shape[1]
        return result / result.sum(axis=1, keepdims=True)

    def predict_quantiles(self, x, train_demand, taus, n_jobs=None):
        """Exact forest-proximity quantiles without a dense test-by-train matrix."""
        query = self.model_.apply(np.asarray(x, dtype=float))
        demand = np.asarray(train_demand, dtype=float)
        taus = tuple(float(tau) for tau in taus)
        trees = query.shape[1]

        def one(row):
            members = [self.leaf_members_[tree][int(row[tree])] for tree in range(trees)]
            indices = np.concatenate(members)
            weights = np.concatenate(
                [np.full(len(index), 1.0 / (trees * len(index)), dtype=np.float64) for index in members]
            )
            values = demand[indices]
            order = np.argsort(values, kind="stable")
            cumulative = np.cumsum(weights[order])
            sorted_values = values[order]
            return [sorted_values[min(np.searchsorted(cumulative, tau, side="left"), len(order) - 1)] for tau in taus]

        workers = self.n_jobs if n_jobs is None else int(n_jobs)
        rows = Parallel(n_jobs=workers, prefer="threads", batch_size=16)(delayed(one)(row) for row in query)
        return np.asarray(rows, dtype=np.float64)


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
