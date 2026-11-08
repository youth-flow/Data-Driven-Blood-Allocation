"""AXID-v4 synthetic generator: one i.i.d. daily mechanism, no real-data reader."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

SCHEMA = "synthetic_axid_axid_v4"
GENERATOR = "AXID-v4"
ORDER_NUM = ["u", "v", "risk", "surgery_type"]
DAY = [
    "scheduled_load",
    "emergency_pressure",
    "n_orders",
    "request_use_sum",
    "request_reserve_sum",
    "risk_mean",
    "risk_second_moment",
    "type_mean",
]
UPPER = 60.5


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30.0, 30.0)))



def mechanism_features(u, v, risk, surgery_type, c1, c2):
    """AXID-v4 local case-mix mechanism shared by generator and Oracle."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    risk = np.asarray(risk, float)
    surgery_type = np.asarray(surgery_type, float)
    high = (risk - 0.15) / 0.70
    z = 2.0 * high - 1.0
    match = -((u - (0.5 + 0.3 * c1)) ** 2 + (v - (0.5 + 0.3 * c2)) ** 2) / (2.0 * 0.23**2)
    match += 0.5 * (2.0 * surgery_type - 1.0) * c2
    omega = np.exp(match - match.max())
    omega /= omega.sum()
    d2 = (u[:, None] - u[None, :]) ** 2 + (v[:, None] - v[None, :]) ** 2
    kernel = np.exp(-d2 / (2.0 * 0.22**2) - 1.5 * (surgery_type[:, None] - surgery_type[None, :]) ** 2)
    np.fill_diagonal(kernel, 0.0)
    p_local = (kernel @ high) / kernel.sum(1)
    p_other = (high.sum() - high) / (len(high) - 1)
    local = high * np.tanh(2.0 * (p_local - p_other))
    h_local = float(omega @ local)
    h_ctx = 0.55 * float(omega @ z) + 0.20 * float(omega @ ((2.0 * u - 1.0) * (2.0 * surgery_type - 1.0)))
    return z, omega, local, h_local, h_ctx


def patient_parameters(u, v, risk, surgery_type, c1, c2, shock):
    z, _, local, _, h_ctx = mechanism_features(u, v, risk, surgery_type, c1, c2)
    length = (
        -1.0
        + 0.6 * u
        + 0.4 * v
        + 0.6 * z
        + 0.4 * surgery_type
        + 3.0 * h_ctx
        + 0.3 * c1
        + 0.2 * c2
        + 2.2 * local
        + 0.45 * shock
    )
    quantity = sigmoid(-0.4 + 0.5 * u + 0.3 * v + h_ctx + 0.9 * local)
    return length, quantity


def new_parameters(h_ctx, h_local, zbar, c1, c2, shock):
    logit_new = -0.7 + 3.5 * h_ctx + 0.5 * zbar + 0.5 * c1 + 0.2 * c2 + 3.0 * h_local
    p_new = sigmoid(logit_new + 0.45 * shock)
    pi_new = sigmoid(-0.1 + 1.2 * h_ctx + 0.3 * c2 + 0.15 * shock)
    return p_new, pi_new


def day_sample(rng):
    c1, c2 = rng.uniform(-1.0, 1.0, 2)
    n = 3 + int(rng.binomial(5, sigmoid(0.25 + 0.45 * c1)))
    u = rng.uniform(0.05, 0.95, n)
    v = rng.uniform(0.05, 0.95, n)
    high = rng.binomial(1, 0.5, n).astype(bool)
    risk = 0.15 + 0.70 * high
    surgery = rng.binomial(1, 0.5, n).astype(float)
    z, _, local, h_local, h_ctx = mechanism_features(u, v, risk, surgery, c1, c2)
    u_shock = float(rng.normal())
    length, quantity_probability = patient_parameters(u, v, risk, surgery, c1, c2, u_shock)
    issued = rng.binomial(1, sigmoid(length), n).astype(bool)
    quantity = 0.5 * (1.0 + rng.binomial(5, quantity_probability))
    y = issued * quantity
    p_new, pi_new = new_parameters(h_ctx, h_local, float(z.mean()), c1, c2, u_shock)
    arrivals = int(rng.binomial(12, p_new))
    new_zero = 0.5 * (arrivals + rng.binomial(5 * arrivals, pi_new))
    remainder = float(rng.uniform(0.0, 0.5))
    demand_new = float(new_zero) + remainder
    known = float(y.sum())
    demand = known + demand_new
    if not 0.0 <= demand <= UPPER:
        raise RuntimeError("AXID-v4 demand violated the hard support")
    return {
        "context": {"scheduled_load": c1, "emergency_pressure": c2},
        "n": n,
        "u": u,
        "v": v,
        "risk": risk,
        "surgery_type": surgery,
        "known": known,
        "demand_new": demand_new,
        "demand": demand,
        "y": y,
    }


def simulate_split(config, count, start, seed, label):
    split_key = [int(v) for v in seed.generate_state(4)]
    days, rows = [], []
    dates = pd.date_range(start, periods=count)
    for index, day in enumerate(dates):
        rng = np.random.default_rng(np.random.SeedSequence([*split_key, index, 0]))
        sample = day_sample(rng)
        context = sample["context"]
        inventory = (
            float(config["inventory_center"])
            + 20.0 * context["scheduled_load"]
            + float(rng.uniform(-config["inventory_half_width"], config["inventory_half_width"]))
        )
        days.append(
            {
                "decision_day": day,
                "target_day": day,
                "snapshot_day": day - pd.Timedelta(days=1),
                "information_cutoff": day,
                "source_split": label,
                "D": sample["demand"],
                "D_new": sample["demand_new"],
                "known_recent_D": sample["known"],
                "It": inventory,
                "n_orders": sample["n"],
                "supervised_eligible": True,
                "demand_observed": True,
                "inventory_observed": True,
                **context,
                "request_use_sum": float(sample["u"].sum()),
                "request_reserve_sum": float(sample["v"].sum()),
                "risk_mean": float(sample["risk"].mean()),
                "risk_second_moment": float((sample["risk"] ** 2).mean()),
                "type_mean": float(sample["surgery_type"].mean()),
            }
        )
        frame = pd.DataFrame(
            {
                "decision_day": day,
                "order_key": [f"axidv3-{label}-{index}-{j}" for j in range(sample["n"])],
                "u": sample["u"],
                "v": sample["v"],
                "risk": sample["risk"],
                "surgery_type": sample["surgery_type"],
                "real_demand": sample["y"],
                "order_label_observed": True,
            }
        )
        rows.append(frame)
    return pd.DataFrame(days), pd.concat(rows, ignore_index=True)


def fixed_scalers():
    order = {
        "u": {"mean": 0.0, "std": 1.0},
        "v": {"mean": 0.0, "std": 1.0},
        "risk": {"mean": 0.0, "std": 1.0},
        "surgery_type": {"mean": 0.0, "std": 1.0},
    }
    day = {
        "scheduled_load": {"mean": -1.0, "std": 2.0},
        "emergency_pressure": {"mean": -1.0, "std": 2.0},
        "n_orders": {"mean": 0.0, "std": 8.0},
        "request_use_sum": {"mean": 0.0, "std": 8.0},
        "request_reserve_sum": {"mean": 0.0, "std": 8.0},
        "risk_mean": {"mean": 0.0, "std": 1.0},
        "risk_second_moment": {"mean": 0.0, "std": 1.0},
        "type_mean": {"mean": 0.0, "std": 1.0},
    }
    return {"order": order, "day": day, "inventory": {"It": {"mean": 225.0, "std": 180.0}}}


def schema(config):
    return {
        "schema_version": SCHEMA,
        "generator_version": 4,
        "generator_name": GENERATOR,
        "order_num_cols": ORDER_NUM,
        "order_cat_cols": [],
        "day_feature_cols": DAY,
        "inventory_feature": "It",
        "inventory_position": 2,
        "excluded_from_model_inputs": [
            "D",
            "D_new",
            "known_recent_D",
            "real_demand",
            "latent_U",
            "H",
            "omega",
            "peer",
            "L",
            "J",
            "oracle_quantile",
        ],
        "information_groups": {
            "day_context": ["scheduled_load", "emergency_pressure"],
            "day_basic": ["n_orders", "request_use_sum", "request_reserve_sum"],
            "day_clinical": ["risk_mean", "risk_second_moment", "type_mean"],
            "order_basic_num": ["u", "v", "surgery_type"],
            "order_basic_cat": [],
        },
        "max_orders": int(config["max_visible_orders"]),
        "max_new_orders": int(config["max_new_orders"]),
        "demand_support_upper": UPPER,
        "demand_grid_step": 0.5,
    }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def export_fold_with_orders(folder, train_days, train_orders, test_days, test_orders, config):
    folder.mkdir(parents=True, exist_ok=False)
    daily = pd.concat([train_days, test_days], ignore_index=True)
    daily["split"] = ["train"] * len(train_days) + ["test"] * len(test_days)
    daily = daily.sort_values("decision_day").reset_index(drop=True)
    orders = pd.concat([train_orders, test_orders], ignore_index=True)
    orders["split"] = orders.decision_day.map(dict(zip(daily.decision_day, daily.split)))
    orders = orders.sort_values(["decision_day", "order_key"]).reset_index(drop=True)
    daily.to_csv(folder / "daily_features.csv", index=False)
    orders.to_csv(folder / "data.csv", index=False)
    save_json(folder / "data_feature_schema.json", schema(config))
    save_json(folder / "generation.yaml", config)
    save_json(
        folder / "split_manifest.json",
        {
            "train_start": str(train_days.decision_day.min().date()),
            "train_end": str(train_days.decision_day.max().date()),
            "test_start": str(test_days.decision_day.min().date()),
            "test_end": str(test_days.decision_day.max().date()),
            "evaluation_role": "test",
            "evaluation_is_retrospective": False,
            "train_days": len(train_days),
            "test_days": len(test_days),
        },
    )
    save_json(
        folder / "preprocessing_state.json",
        {
            "feature_fit_end": str(train_days.decision_day.max().date()),
            "numeric_scalers": fixed_scalers(),
        },
    )
    artifacts = {}
    for name in (
        "daily_features.csv",
        "data.csv",
        "data_feature_schema.json",
        "split_manifest.json",
        "preprocessing_state.json",
    ):
        artifacts[name] = file_hash(folder / name)
    save_json(folder / "manifest.json", {"complete": True, "artifacts": artifacts})
    return folder


def generate_performance(config, output):
    config = dict(config)
    if config.get("schema_version") not in {4, "4"}:
        raise ValueError("AXID-v4 reference must declare schema_version 4")
    if config.get("generator_version") not in {4, "4"}:
        raise ValueError("AXID-v4 generator version mismatch")
    if int(config["max_visible_orders"]) != 8 or int(config["max_new_orders"]) != 12:
        raise ValueError("AXID-v4 fixes M=8 and K_new=12")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Synthetic output already exists; choose a fresh directory")
    seed = np.random.SeedSequence([int(config["data_seed"]), 731])
    train_seed, test_seed = seed.spawn(2)
    train_days, train_orders = simulate_split(config, int(config["n_train"]), "2020-01-01", train_seed, "train")
    test_start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(config["n_train"]) + 30)
    test_days, test_orders = simulate_split(config, int(config["n_test"]), str(test_start.date()), test_seed, "test")
    export_fold_with_orders(output, train_days, train_orders, test_days, test_orders, config)
    return {
        "schema_version": SCHEMA,
        "train_days": len(train_days),
        "test_days": len(test_days),
        "train_orders": len(train_orders),
        "test_orders": len(test_orders),
        "demand_upper": UPPER,
    }
