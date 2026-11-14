"""AXID-DC v8: HB-context two-level doctor-choice synthetic demand."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

SCHEMA = "synthetic_axid_dc_v8"
GENERATOR = "AXID-DC v8"
ORDER_NUM = ["u", "r", "hb"]
DAY = [
    "planned_load",
    "emergency_load",
    "n_registered",
    "request_use_sum",
    "risk_sum",
    "hb_shortage_sum",
]
UPPER = 61.0


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30.0, 30.0)))


def draw_profiles(rng, cp, ce, count):
    if count == 0:
        return np.empty((0, 3), dtype=float)
    u = rng.beta(2.0, 2.0, count)
    r = rng.beta(2.0, 3.0, count)
    hb = rng.beta(2.0, 2.0, count)
    return np.column_stack([u, r, hb]).astype(float)


def mechanism(A, b, cp, ce, inventory):
    u, r, hb = A.T
    shortage = 1.0 - hb
    zeta = 1.5 * r + 0.75 * u + 0.5 * shortage + 0.75 * shortage**2
    zeta += 0.75 * r * shortage + 0.5 * (1.0 - b)
    diff = zeta[None, :] - zeta[:, None]
    crowding = (np.maximum(diff, 0.0) * u[None, :]).sum(1) / 10.0
    T = -crowding
    rt = r + 0.4 * shortage + 0.6 * T
    g1 = 0.5 + 0.3 * cp
    g2 = 0.5 * (ce + 1.0)
    scarcity = sigmoid((270.0 - inventory) / 12.0)
    e = -((u - g1) ** 2) / (2.0 * 0.22**2) + 1.5 * g2 * rt + 0.4 * scarcity * zeta
    omega = np.exp(e - e.max())
    omega /= omega.sum()
    R = float(omega @ rt)
    return zeta, T, rt, omega, R, scarcity


def action_probabilities(A, b, cp, ce, inventory, shock):
    zeta, T, _, _, R, scarcity = mechanism(A, b, cp, ce, inventory)
    h = -1.9 + 0.35 * zeta + 3.0 * scarcity * T + 2.4 * R + 0.175 * shock
    utilities = np.zeros((len(A), 7))
    for y in range(1, 7):
        utilities[:, y] = -0.7 + y * h - 0.105 * y * y
    utilities -= utilities.max(1, keepdims=True)
    probabilities = np.exp(utilities)
    probabilities /= probabilities.sum(1, keepdims=True)
    return probabilities


def sample_actions(A, b, cp, ce, inventory, shock, rng):
    probabilities = action_probabilities(A, b, cp, ce, inventory, shock)
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(len(A))
    return (draws[:, None] > cumulative).sum(1).astype(int)


def day_sample(rng):
    cp, ce = rng.uniform(-1.0, 1.0, 2)
    inventory = 270.0 + 15.0 * cp + rng.uniform(-50.0, 50.0)
    n_old = 3 + int(rng.binomial(5, sigmoid(-0.1 + 0.4 * cp)))
    n_new = int(rng.binomial(2, sigmoid(-0.4 + 0.5 * ce)))
    old = draw_profiles(rng, cp, ce, n_old)
    new = draw_profiles(rng, cp, ce, n_new)
    all_patients = np.vstack([old, new])
    b = np.concatenate([np.zeros(n_old), np.ones(n_new)])
    shock = float(rng.normal())
    actions = sample_actions(all_patients, b, cp, ce, inventory, shock, rng)
    demand = float(actions.sum() + rng.uniform(0.0, 1.0))
    if not 0.0 <= demand <= UPPER:
        raise RuntimeError("AXID-DC v8 demand violated the hard support")
    known = float(actions[:n_old].sum())
    return {
        "cp": cp,
        "ce": ce,
        "inventory": float(inventory),
        "n_old": n_old,
        "n_new": n_new,
        "old": old,
        "old_y": actions[:n_old],
        "demand": demand,
        "known": known,
        "demand_new": demand - known,
    }


def simulate_split(config, count, start, seed, label):
    split_key = [int(v) for v in seed.generate_state(4)]
    days, rows = [], []
    for index, day in enumerate(pd.date_range(start, periods=count)):
        rng = np.random.default_rng(np.random.SeedSequence([*split_key, index, 0]))
        sample = day_sample(rng)
        old = sample["old"]
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
                "It": sample["inventory"],
                "n_registered": sample["n_old"],
                "supervised_eligible": True,
                "demand_observed": True,
                "inventory_observed": True,
                "planned_load": sample["cp"],
                "emergency_load": sample["ce"],
                "request_use_sum": float(old[:, 0].sum()),
                "risk_sum": float(old[:, 1].sum()),
                "hb_shortage_sum": float((1.0 - old[:, 2]).sum()),
            }
        )
        rows.append(
            pd.DataFrame(
                {
                    "decision_day": day,
                    "order_key": [f"axid-dc-v8-{label}-{index}-{j}" for j in range(sample["n_old"])],
                    "u": old[:, 0],
                    "r": old[:, 1],
                    "hb": old[:, 2],
                    "real_demand": sample["old_y"],
                    "order_label_observed": True,
                }
            )
        )
    return pd.DataFrame(days), pd.concat(rows, ignore_index=True)


def fixed_scalers():
    order = {name: {"mean": 0.0, "std": 1.0} for name in ORDER_NUM}
    day = {
        "planned_load": {"mean": -1.0, "std": 2.0},
        "emergency_load": {"mean": -1.0, "std": 2.0},
        "n_registered": {"mean": 0.0, "std": 8.0},
        "request_use_sum": {"mean": 0.0, "std": 8.0},
        "risk_sum": {"mean": 0.0, "std": 8.0},
        "hb_shortage_sum": {"mean": 0.0, "std": 8.0},
    }
    return {"order": order, "day": day, "inventory": {"It": {"mean": 205.0, "std": 130.0}}}


def schema(config):
    return {
        "schema_version": SCHEMA,
        "generator_version": 9,
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
            "A_plus",
            "U",
            "T",
            "omega",
            "Y",
            "q",
        ],
        "information_groups": {
            "day_context": ["planned_load", "emergency_load"],
            "day_basic": ["n_registered", "request_use_sum"],
            "day_clinical": ["risk_sum", "hb_shortage_sum"],
            "order_basic_num": ["u", "r", "hb"],
            "order_basic_cat": [],
        },
        "max_orders": int(config["max_visible_orders"]),
        "max_new_orders": int(config["max_new_orders"]),
        "demand_support_upper": UPPER,
        "demand_grid_step": 1.0,
    }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def export_fold(folder, train_days, train_orders, test_days, test_orders, config):
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
        {"feature_fit_end": str(train_days.decision_day.max().date()), "numeric_scalers": fixed_scalers()},
    )
    artifacts = {}
    for name in ("daily_features.csv", "data.csv", "data_feature_schema.json", "split_manifest.json", "preprocessing_state.json"):
        artifacts[name] = file_hash(folder / name)
    save_json(folder / "manifest.json", {"complete": True, "artifacts": artifacts})
    return folder


def generate_performance(config, output):
    config = dict(config)
    if config.get("schema_version") not in {9, "9"} or config.get("generator_version") not in {9, "9"}:
        raise ValueError("AXID-DC v8 reference must declare schema/generator version 9")
    if int(config["max_visible_orders"]) != 8 or int(config["max_new_orders"]) != 2:
        raise ValueError("AXID-DC v7-Final fixes M=8 and N+<=2")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Synthetic output already exists; choose a fresh directory")
    train_seed, test_seed = np.random.SeedSequence([int(config["data_seed"]), 731]).spawn(2)
    train_days, train_orders = simulate_split(config, int(config["n_train"]), "2020-01-01", train_seed, "train")
    test_start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(config["n_train"]) + 30)
    test_days, test_orders = simulate_split(config, int(config["n_test"]), str(test_start.date()), test_seed, "test")
    export_fold(output, train_days, train_orders, test_days, test_orders, config)
    return {
        "schema_version": SCHEMA,
        "train_days": len(train_days),
        "test_days": len(test_days),
        "train_orders": len(train_orders),
        "test_orders": len(test_orders),
        "demand_upper": UPPER,
    }
