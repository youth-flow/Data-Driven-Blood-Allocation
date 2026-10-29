from __future__ import annotations
import numpy as np


def quantity_keys(frame):
    use = frame["申请用血量"].to_numpy(dtype=float)
    reserve = frame["申请备血量"].to_numpy(dtype=float)
    kind = (use > 0).astype(int) + 2 * (reserve > 0).astype(int)
    age = frame["delta"].to_numpy(dtype=int)
    amount = np.digitize(np.maximum(use, reserve), [2.0, 4.0, 8.0], right=True)
    department = frame["科室高频分组"].fillna("未知").astype(str).to_numpy()
    return [list(zip(kind, age, amount, department)), list(zip(kind, age, amount)), list(zip(kind, age))]


class ConditionalQuantityPMF:
    """Positive quantity PMFs learned exclusively from training outcomes."""

    def fit(self, frame):
        outcomes = frame["real_demand"].to_numpy(dtype=float)
        if not np.isfinite(outcomes).all() or (outcomes < 0).any():
            raise ValueError("Invalid order outcomes.")
        if not np.allclose(outcomes * 2, np.round(outcomes * 2), rtol=0.0, atol=1e-7):
            raise ValueError("Quantities must be on the half-unit grid.")
        positive = np.flatnonzero(outcomes > 0)
        if not len(positive):
            raise ValueError("No positive training quantities.")
        keys = quantity_keys(frame)
        self.pools = [outcomes[positive]]
        self.maps = []
        for level, threshold in enumerate([30, 20, 10]):
            grouped = {}
            for i in positive:
                grouped.setdefault(keys[level][i], []).append(outcomes[i])
            mapping = {}
            for key, amounts in grouped.items():
                if len(amounts) >= threshold:
                    mapping[key] = len(self.pools)
                    self.pools.append(np.asarray(amounts, dtype=float))
            self.maps.append(mapping)
        self.pmfs = []
        for pool in self.pools:
            counts = np.bincount(np.rint(pool * 2).astype(np.int64))
            self.pmfs.append(counts.astype(float) / counts.sum())
        self.means = np.array([pool.mean() for pool in self.pools])
        self.fit_states = len(frame)
        self.fit_positives = len(positive)
        return self

    def pool_ids(self, frame):
        keys = quantity_keys(frame)
        ids = np.zeros(len(frame), dtype=np.int64)
        for i in range(len(frame)):
            for level in range(3):
                candidate = self.maps[level].get(keys[level][i])
                if candidate is not None:
                    ids[i] = candidate
                    break
        return ids

    def state(self):
        def plain(value):
            return value.item() if isinstance(value, np.generic) else value

        return {
            "fit_split": "explicit training split only",
            "fit_states": self.fit_states,
            "fit_positive_states": self.fit_positives,
            "thresholds": [30, 20, 10],
            "quantity_bins": [2.0, 4.0, 8.0],
            "quantity_bin_input": "max(request_use, request_reserve)",
            "maps": [
                [{"key": [plain(v) for v in key], "pool": pool} for key, pool in mapping.items()]
                for mapping in self.maps
            ],
            "positive_pmfs_half_units": [pmf.tolist() for pmf in self.pmfs],
            "pool_sample_counts": [len(pool) for pool in self.pools],
        }
