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
        if self.schema.get("schema_version") not in {
            "morning_t_v1",
            "synthetic_axid_v2",
            "synthetic_axid_v3",
            "synthetic_axid_axid_v3",
            "synthetic_axid_axid_v4",
            "synthetic_axid_dc_v6",
            "synthetic_axid_dc_v7",
            "synthetic_axid_dc_v7_final",
            "synthetic_axid_dc_v8",
            "synthetic_axid_dc_v10",
        }:
            raise ValueError("Unsupported data contract")
        manifest = read_json(self.directory / "manifest.json")
        if not manifest.get("complete"):
            raise ValueError("Incomplete preprocessing output")
        for name, expected in manifest["artifacts"].items():
            if Path(name).name != name or sha256(self.directory / name) != expected:
                raise ValueError("Derived artifact integrity mismatch")
        self.identity = sha256(self.directory / "manifest.json")
        self.split = read_json(self.directory / "split_manifest.json")
        self.state = read_json(self.directory / "preprocessing_state.json")
        # The synthetic protocol fixes X=(c1,c2,I,N/M,sum u/M,sum v/M,mean r,mean r^2).
        # Real-derived folds retain the historical layout with inventory appended last.
        self.inventory_position = (
            2
            if self.schema.get("schema_version") in {"synthetic_axid_v2", "synthetic_axid_v3", "synthetic_axid_axid_v3", "synthetic_axid_axid_v4", "synthetic_axid_dc_v6", "synthetic_axid_dc_v7", "synthetic_axid_dc_v7_final", "synthetic_axid_dc_v8", "synthetic_axid_dc_v10"}
            else "end"
        )
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
            "D_new",
            "known_recent_D",
            "latent_shock",
            "oracle_quantile",
        }
        if inputs & forbidden or "It" in self.day_cols:
            raise ValueError("Diagnostic or outcome fields cannot be model inputs")
        integer_days = bool(self.schema.get("integer_decision_day", False))
        self.compact_format = self.schema.get("compact_format")
        if self.compact_format == "axid_v10_csr_v1":
            files = self.schema.get("compact_files", {})
            self._compact_paths = {
                "daily": self.directory / files.get("daily", "daily.npz"),
                "patients": self.directory / files.get("patients", "patients.npy"),
                "labels": self.directory / files.get("patient_labels", "patient_labels.npy"),
                "offsets": self.directory / files.get("patient_offsets", "patient_offsets.npy"),
            }
            archive = np.load(self._compact_paths["daily"], allow_pickle=False)
            self.daily = pd.DataFrame({name: archive[name] for name in archive.files})
            archive.close()
            if "split" not in self.daily:
                raise ValueError("Compact v10 daily arrays omit the split field")
            if np.issubdtype(self.daily["split"].dtype, np.number):
                self.daily["split"] = self.daily["split"].map({0: "train", 1: self.eval_role})
            self.daily["_compact_index"] = np.arange(len(self.daily), dtype=np.int64)
            self.daily["target_day"] = self.daily["decision_day"]
            self.daily["snapshot_day"] = self.daily["decision_day"] - 1
            self.daily["information_cutoff"] = self.daily["decision_day"]
            self.daily["source_split"] = self.daily["split"]
            for name in ("supervised_eligible", "demand_observed", "inventory_observed"):
                self.daily[name] = self.daily[name].astype(bool) if name in self.daily else True
            if "n_orders" not in self.daily and "n_registered" in self.daily:
                self.daily["n_orders"] = self.daily["n_registered"]
            self._compact_patients = np.load(self._compact_paths["patients"], mmap_mode="r")
            self._compact_labels = np.load(self._compact_paths["labels"], mmap_mode="r")
            self._compact_offsets = np.load(self._compact_paths["offsets"], mmap_mode="r")
            if (
                self._compact_patients.ndim != 2
                or self._compact_patients.shape[1] != len(self.order_num_cols)
                or self._compact_labels.shape != (len(self._compact_patients),)
                or self._compact_offsets.shape != (len(self.daily) + 1,)
                or int(self._compact_offsets[-1]) != len(self._compact_patients)
            ):
                raise ValueError("Invalid compact v10 patient arrays")
            self.orders = None
        elif integer_days:
            self.daily = pd.read_csv(self.directory / "daily_features.csv")
        else:
            self.daily = pd.read_csv(
                self.directory / "daily_features.csv",
                parse_dates=["decision_day", "target_day", "snapshot_day", "information_cutoff"],
            )
        if self.compact_format != "axid_v10_csr_v1":
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
            self.orders = pd.read_csv(self.directory / "data.csv", usecols=columns)
            if not integer_days:
                self.orders["decision_day"] = pd.to_datetime(self.orders["decision_day"])
        if self.daily.decision_day.duplicated().any() or (
            self.orders is not None and self.orders.duplicated(["decision_day", "order_key"]).any()
        ):
            raise ValueError("Duplicate daily or order snapshot keys")
        if not integer_days and not (
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
        training = (
            None
            if self.orders is None
            else self.orders[self.orders.decision_day.between(self.split["train_start"], self.split["train_end"])]
        )
        self.category_maps = (
            {}
            if training is None
            else {
                c: {
                    value: i + 1
                    for i, value in enumerate(sorted(training[c].fillna("missing").astype(str).unique()))
                }
                for c in self.order_cat_cols
            }
        )
        self.cardinalities = [len(self.category_maps[c]) + 1 for c in self.order_cat_cols]

    def daily_split(self, split):
        role = "train" if split == "train" else self.eval_role
        if split not in ("train", "eval"):
            raise ValueError("Split must be train or eval")
        return self.daily.loc[self.daily["split"].eq(role) & self.daily.supervised_eligible].copy()

    def order_split(self, split):
        if self.orders is None:
            patients, labels, offsets, days = self.patient_arrays(split)
            if len(patients) > 1_000_000:
                raise RuntimeError("Compact v10 patient rows must be consumed through patient_arrays, not a DataFrame")
            counts = np.diff(offsets)
            return pd.DataFrame(
                {
                    "decision_day": np.repeat(days, counts),
                    "order_key": np.arange(len(patients), dtype=np.int64),
                    **{name: patients[:, j] for j, name in enumerate(self.order_num_cols)},
                    "real_demand": labels,
                    "order_label_observed": True,
                }
            )
        return self.orders[self.orders.decision_day.isin(self.daily_split(split).decision_day)].copy()

    def patient_arrays(self, split):
        """Return visible-patient arrays, labels, split-local CSR offsets, and day ids."""
        if self.compact_format != "axid_v10_csr_v1":
            frame = self.order_split(split)
            daily = self.daily_split(split)
            groups = frame.groupby("decision_day", sort=False).size().reindex(daily.decision_day, fill_value=0)
            offsets = np.concatenate([[0], np.cumsum(groups.to_numpy(np.int64))])
            return (
                frame[self.order_num_cols].to_numpy(np.float32),
                frame.real_demand.to_numpy(np.float32),
                offsets,
                daily.decision_day.to_numpy(),
            )
        daily = self.daily_split(split).sort_values("_compact_index")
        indices = daily["_compact_index"].to_numpy(np.int64)
        contiguous = len(indices) == 0 or np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
        if contiguous and len(indices):
            left, right = int(self._compact_offsets[indices[0]]), int(self._compact_offsets[indices[-1] + 1])
            patients = self._compact_patients[left:right]
            labels = self._compact_labels[left:right]
            offsets = np.asarray(self._compact_offsets[indices[0] : indices[-1] + 2] - left, dtype=np.int64)
        elif len(indices):
            pieces = [self._compact_patients[self._compact_offsets[i] : self._compact_offsets[i + 1]] for i in indices]
            label_pieces = [self._compact_labels[self._compact_offsets[i] : self._compact_offsets[i + 1]] for i in indices]
            patients = np.concatenate(pieces)
            labels = np.concatenate(label_pieces)
            offsets = np.concatenate([[0], np.cumsum([len(piece) for piece in pieces])]).astype(np.int64)
        else:
            patients = np.empty((0, len(self.order_num_cols)), dtype=np.float32)
            labels = np.empty(0, dtype=np.float32)
            offsets = np.zeros(1, dtype=np.int64)
        return patients, labels, offsets, daily.decision_day.to_numpy()

    def categories(self, frame):
        if not self.order_cat_cols:
            return np.empty((len(frame), 0), dtype=np.int64)
        return np.column_stack(
            [
                frame[c].fillna("missing").astype(str).map(self.category_maps[c]).fillna(0).to_numpy(np.int64)
                for c in self.order_cat_cols
            ]
        )

    def day_matrix(self, daily):
        """Model input X; inventory is inserted at position 3 for synthetic data."""
        base = self.day_scaler.transform(daily)
        inventory = self.inventory_scaler.transform(daily)
        if self.inventory_position == 2:
            if base.shape[1] != len(self.day_cols):
                raise ValueError("Synthetic X contract requires one non-inventory coordinate per day column")
            return np.column_stack([base[:, :2], inventory, base[:, 2:]])
        return np.column_stack([base, inventory])

    def context(self, daily):
        return self.day_matrix(daily)

    def metadata(self):
        return {
            "order_num_dim": len(self.order_num_cols),
            "order_cat_cardinalities": self.cardinalities,
            "day_num_dim": len(self.day_cols),
            "day_feature_dim": len(self.day_cols) + 1,
            "inventory_position": self.inventory_position,
            "inventory_mean": float(self.inventory_scaler.mean[0]),
            "inventory_std": float(self.inventory_scaler.scale[0]),
        }

    def audit_summary(self):
        return {
            "schema": self.schema["schema_version"],
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

    def close(self):
        """Release compact memory maps so Windows can move or delete a fold."""
        for name in ("_compact_patients", "_compact_labels", "_compact_offsets"):
            value = getattr(self, name, None)
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
            if hasattr(self, name):
                setattr(self, name, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def limit_for_smoke(self, train_days=32, eval_days=8):
        selected = pd.concat([self.daily_split("train").head(train_days), self.daily_split("eval").head(eval_days)])
        self.daily = selected.copy().reset_index(drop=True)
        if self.orders is not None:
            self.orders = self.orders[self.orders.decision_day.isin(selected.decision_day)].copy()


class DailySets(Dataset):
    def __init__(self, data, split):
        selected_daily = data.daily_split(split).reset_index(drop=True)
        if data.compact_format == "axid_v10_csr_v1":
            # Tabular baselines consume the same DailySets.daily object and need
            # the authorized day/Psi columns; workers only read these arrays.
            self.daily = selected_daily.copy()
            self.samples = None
            self.frames = None
            self._patient_path = str(data._compact_paths["patients"])
            self._offset_path = str(data._compact_paths["offsets"])
            self._patients = None
            self._offsets = None
            self._day_indices = selected_daily["_compact_index"].to_numpy(np.int64)
            self.lengths = np.asarray(
                data._compact_offsets[self._day_indices + 1] - data._compact_offsets[self._day_indices],
                dtype=np.int16,
            )
            self._order_mean = data.order_scaler.mean.astype(np.float32)
            self._order_scale = data.order_scaler.scale.astype(np.float32)
            self._day_num = torch.from_numpy(data.day_scaler.transform(selected_daily))
            self._day_features = torch.from_numpy(data.day_matrix(selected_daily).astype(np.float32))
            self._inventory = torch.from_numpy(selected_daily.It.to_numpy(np.float32))
            self._demand = torch.from_numpy(selected_daily.D.to_numpy(np.float32))
            return
        self.daily = selected_daily
        self.frames = []
        self.samples = []
        orders = data.order_split(split).reset_index(drop=True)
        groups = orders.groupby("decision_day", sort=False).indices
        all_numeric = data.order_scaler.transform(orders)
        all_categories = data.categories(orders)
        all_days = data.day_scaler.transform(self.daily)
        all_features = data.day_matrix(self.daily)
        for index, row in enumerate(self.daily.itertuples()):
            positions = groups.get(row.decision_day, np.array([], dtype=int))
            frame = orders.iloc[positions]
            self.frames.append(frame)
            self.samples.append(
                {
                    "decision_day": row.decision_day.strftime("%Y-%m-%d"),
                    "order_num": torch.from_numpy(all_numeric[positions]),
                    "order_cat": torch.from_numpy(all_categories[positions]),
                    "day_num": torch.from_numpy(all_days[index]),
                    "day_features": torch.from_numpy(all_features[index]),
                    "inventory": torch.tensor(row.It, dtype=torch.float32),
                    "demand": torch.tensor(row.D, dtype=torch.float32),
                }
            )

    def __len__(self):
        return len(self.daily) if self.samples is None else len(self.samples)

    def __getitem__(self, i):
        if self.samples is None:
            if self._patients is None:
                self._patients = np.load(self._patient_path, mmap_mode="r")
                self._offsets = np.load(self._offset_path, mmap_mode="r")
            day_index = int(self._day_indices[i])
            left, right = int(self._offsets[day_index]), int(self._offsets[day_index + 1])
            numeric = (np.asarray(self._patients[left:right]) - self._order_mean) / self._order_scale
            return {
                "decision_day": int(self.daily.decision_day.iloc[i]),
                "order_num": torch.from_numpy(np.asarray(numeric, dtype=np.float32)),
                "order_cat": torch.empty((right - left, 0), dtype=torch.long),
                "day_num": self._day_num[i],
                "day_features": self._day_features[i],
                "inventory": self._inventory[i],
                "demand": self._demand[i],
            }
        return self.samples[i]

    def __getitems__(self, indices):
        """Bulk-fetch a compact batch without constructing per-day Python dicts."""
        if self.samples is not None:
            return [self.samples[i] for i in indices]
        if self._patients is None:
            self._patients = np.load(self._patient_path, mmap_mode="r")
            self._offsets = np.load(self._offset_path, mmap_mode="r")
        indices = np.asarray(indices, dtype=np.int64)
        day_indices = self._day_indices[indices]
        left = np.asarray(self._offsets[day_indices], dtype=np.int64)
        right = np.asarray(self._offsets[day_indices + 1], dtype=np.int64)
        counts = right - left
        maximum = max(1, int(counts.max(initial=0)))
        numeric = torch.zeros(len(indices), maximum, len(self._order_mean), dtype=torch.float32)
        mask = torch.zeros(len(indices), maximum, dtype=torch.bool)
        for row, (start, stop) in enumerate(zip(left, right)):
            count = int(stop - start)
            if count:
                values = (np.asarray(self._patients[start:stop]) - self._order_mean) / self._order_scale
                numeric[row, :count] = torch.from_numpy(np.asarray(values, dtype=np.float32))
                mask[row, :count] = True
        tensor_indices = torch.from_numpy(indices)
        return {
            "decision_day": self.daily.decision_day.iloc[indices].tolist(),
            "order_num": numeric,
            "order_cat": torch.empty((len(indices), maximum, 0), dtype=torch.long),
            "order_mask": mask,
            "day_num": self._day_num[tensor_indices],
            "day_features": self._day_features[tensor_indices],
            "inventory": self._inventory[tensor_indices],
            "demand": self._demand[tensor_indices],
        }

    def close(self):
        for name in ("_patients", "_offsets"):
            value = getattr(self, name, None)
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
            if hasattr(self, name):
                setattr(self, name, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def collate(samples):
    if isinstance(samples, dict):
        return samples
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
    for name in ("day_num", "day_features", "inventory", "demand"):
        result[name] = torch.stack([s[name] for s in samples])
    return result
