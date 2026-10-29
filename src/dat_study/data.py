"""Read only morning_t_v1 derived artifacts; no original-table interface."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .io import read_json, sha256


class NumericScaler:
    def __init__(self, columns, state):
        self.columns = list(columns)
        self.mean = np.array([state[c]["mean"] for c in columns])
        self.scale = np.array([state[c]["std"] for c in columns])
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid training-only scaler state")

    def transform(self, frame):
        result = (frame[self.columns].to_numpy(float) - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("Model inputs contain missing or infinite values")
        return result.astype(np.float32)


class FoldData:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.schema = read_json(self.directory / "data_feature_schema.json")
        if self.schema.get("schema_version") != "morning_t_v1":
            raise ValueError("Only morning_t_v1 derived data are accepted")
        manifest = read_json(self.directory / "manifest.json")
        if not manifest.get("complete"):
            raise ValueError("Incomplete preprocessing output")
        for name, expected in manifest["artifacts"].items():
            if Path(name).name != name or sha256(self.directory / name) != expected:
                raise ValueError("Derived artifact integrity mismatch")
        self.identity = sha256(self.directory / "manifest.json")
        self.split = read_json(self.directory / "split_manifest.json")
        self.state = read_json(self.directory / "preprocessing_state.json")
        if self.state["feature_fit_end"] != self.split["train_end"]:
            raise ValueError("Feature fitting and training boundary mismatch")
        self.eval_role = self.split["evaluation_role"]
        self.order_num_cols = self.schema["order_num_cols"]
        self.order_cat_cols = self.schema["order_cat_cols"]
        self.day_cols = self.schema["day_feature_cols"]
        inputs = set(self.order_num_cols + self.order_cat_cols + self.day_cols)
        forbidden = set(self.schema["excluded_from_model_inputs"]) | {
            "D",
            "q_hist_receipt",
            "real_demand",
            "tag",
            "order_key",
        }
        if inputs & forbidden or "It" in self.day_cols:
            raise ValueError("Diagnostic or outcome fields cannot be model inputs")
        self.daily = pd.read_csv(
            self.directory / "daily_features.csv",
            parse_dates=["decision_day", "target_day", "snapshot_day", "information_cutoff"],
        )
        columns = list(
            dict.fromkeys(
                [
                    "decision_day",
                    "order_key",
                    "real_demand",
                    "order_label_observed",
                    *self.order_num_cols,
                    *self.order_cat_cols,
                ]
            )
        )
        self.orders = pd.read_csv(self.directory / "data.csv", usecols=columns, parse_dates=["decision_day"])
        if self.daily.decision_day.duplicated().any() or self.orders.duplicated(["decision_day", "order_key"]).any():
            raise ValueError("Duplicate daily or order snapshot keys")
        if not (
            self.daily.decision_day.eq(self.daily.target_day).all()
            and self.daily.snapshot_day.eq(self.daily.decision_day - pd.Timedelta(days=1)).all()
            and self.daily.information_cutoff.eq(self.daily.decision_day).all()
        ):
            raise ValueError("Morning timing contract mismatch")
        eligible = self.daily.supervised_eligible.astype(bool)
        if not (
            self.daily.loc[eligible, "demand_observed"].all() and self.daily.loc[eligible, "inventory_observed"].all()
        ):
            raise ValueError("Unobserved supervised target")
        self.daily = self.daily.sort_values("decision_day").reset_index(drop=True)
        fitted = self.state["numeric_scalers"]
        self.order_scaler = NumericScaler(self.order_num_cols, fitted["order"])
        self.day_scaler = NumericScaler(self.day_cols, fitted["day"])
        self.inventory_scaler = NumericScaler(["It"], fitted["inventory"])
        training = self.orders[self.orders.decision_day.between(self.split["train_start"], self.split["train_end"])]
        self.category_maps = {
            c: {value: i + 1 for i, value in enumerate(sorted(training[c].fillna("missing").astype(str).unique()))}
            for c in self.order_cat_cols
        }
        self.cardinalities = [len(self.category_maps[c]) + 1 for c in self.order_cat_cols]

    def daily_split(self, split):
        role = "train" if split == "train" else self.eval_role
        if split not in ("train", "eval"):
            raise ValueError("Split must be train or eval")
        return self.daily.loc[self.daily["split"].eq(role) & self.daily.supervised_eligible].copy()

    def order_split(self, split):
        return self.orders[self.orders.decision_day.isin(self.daily_split(split).decision_day)].copy()

    def categories(self, frame):
        if not self.order_cat_cols:
            return np.empty((len(frame), 0), dtype=np.int64)
        return np.column_stack(
            [
                frame[c].fillna("missing").astype(str).map(self.category_maps[c]).fillna(0).to_numpy(np.int64)
                for c in self.order_cat_cols
            ]
        )

    def context(self, daily):
        return np.column_stack([self.day_scaler.transform(daily), self.inventory_scaler.transform(daily)])

    def metadata(self):
        return {
            "order_num_dim": len(self.order_num_cols),
            "order_cat_cardinalities": self.cardinalities,
            "day_num_dim": len(self.day_cols),
            "inventory_mean": float(self.inventory_scaler.mean[0]),
            "inventory_std": float(self.inventory_scaler.scale[0]),
        }

    def audit_summary(self):
        return {
            "schema": "morning_t_v1",
            "data_identity": self.identity,
            "evaluation_role": self.eval_role,
            "train_days": len(self.daily_split("train")),
            "evaluation_days": len(self.daily_split("eval")),
            "order_numeric_dim": len(self.order_num_cols),
            "order_categorical_dim": len(self.order_cat_cols),
            "day_feature_dim": len(self.day_cols),
            "empty_set_days": int(self.daily.n_orders.eq(0).sum()),
            "retrospective": bool(self.split["evaluation_is_retrospective"]),
        }

    def limit_for_smoke(self, train_days=32, eval_days=8):
        selected = pd.concat([self.daily_split("train").head(train_days), self.daily_split("eval").head(eval_days)])
        self.daily = selected.copy().reset_index(drop=True)
        self.orders = self.orders[self.orders.decision_day.isin(selected.decision_day)].copy()


class DailySets(Dataset):
    def __init__(self, data, split):
        self.daily = data.daily_split(split).reset_index(drop=True)
        self.frames = []
        self.samples = []
        groups = dict(tuple(data.order_split(split).groupby("decision_day", sort=False)))
        for row in self.daily.itertuples():
            frame = groups.get(row.decision_day, data.orders.iloc[:0])
            context = self.daily.loc[self.daily.decision_day.eq(row.decision_day)]
            self.frames.append(frame)
            self.samples.append(
                {
                    "decision_day": row.decision_day.strftime("%Y-%m-%d"),
                    "order_num": torch.from_numpy(data.order_scaler.transform(frame)),
                    "order_cat": torch.from_numpy(data.categories(frame)),
                    "day_num": torch.from_numpy(data.day_scaler.transform(context)[0]),
                    "inventory": torch.tensor(row.It, dtype=torch.float32),
                    "demand": torch.tensor(row.D, dtype=torch.float32),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(samples):
    count, maximum = len(samples), max(1, max(len(s["order_num"]) for s in samples))
    result = {
        "decision_day": [s["decision_day"] for s in samples],
        "order_num": torch.zeros(count, maximum, samples[0]["order_num"].shape[1]),
        "order_cat": torch.zeros(count, maximum, samples[0]["order_cat"].shape[1], dtype=torch.long),
        "order_mask": torch.zeros(count, maximum, dtype=torch.bool),
    }
    for i, sample in enumerate(samples):
        n = len(sample["order_num"])
        result["order_num"][i, :n] = sample["order_num"]
        result["order_cat"][i, :n] = sample["order_cat"]
        result["order_mask"][i, :n] = True
    for name in ("day_num", "inventory", "demand"):
        result[name] = torch.stack([s[name] for s in samples])
    return result
