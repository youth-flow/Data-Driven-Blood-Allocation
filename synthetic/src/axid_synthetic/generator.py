"""Versioned, permutation-invariant single-period simulator, independent of every predictor."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml

SCENARIOS = {"S1": (0.0, 0.0, 0.0), "S2": (1.0, 0.0, 0.0), "S3": (1.0, 1.5, 0.0), "S4": (1.0, 1.5, 0.8)}
KINDS = ["急救用血", "手术备血", "病房用血", "病房备血", "其他"]
NUM_BASIC = ["delta", "申请用血量", "申请备血量"]
NUM_CLINICAL = ["血红蛋白", "红细胞压积", "planned_use_offset_days", "planned_use_missing"]
CAT_BASIC = ["输血要求分组", "科室高频分组"]
DAY_CONTEXT = ["dow_sin", "dow_cos", "is_weekend", "scheduled_load", "emergency_pressure"]
DAY_BASIC = ["n_orders", "request_use_sum", "request_reserve_sum", "order_age_mean"]
DAY_CLINICAL = ["hb_mean", "hct_mean", "urgent_request_rate", "surgery_prepare_rate"]


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def coefficients(config):
    if config["scenario"] not in SCENARIOS:
        raise ValueError("Scenario must be S1, S2, S3 or S4")
    return SCENARIOS[config["scenario"]]


def make_orders(rng, context, config):
    n = rng.binomial(config["max_visible_orders"], sigmoid(-0.2 + 0.4 * context["scheduled_load"]))
    if rng.random() < config.get("empty_set_probability", 0.05):
        n = 0
    # Allow empty sets naturally; every observable is generated before demand randomness.
    kind = rng.choice(KINDS, size=n, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    risk = rng.beta(2, 3, size=n)
    use = 0.5 * rng.binomial(12, 0.25 + 0.25 * risk)
    reserve = 0.5 * rng.binomial(16, 0.15 + 0.20 * risk)
    hb = np.clip(160 - 100 * risk + rng.normal(0, 8, n), 40, 180)
    missing = rng.random(n) < 0.15
    planned = rng.choice([-1, 0, 1, 2], size=n, p=[0.1, 0.5, 0.3, 0.1])
    return pd.DataFrame(
        {
            "delta": rng.integers(0, 3, size=n),
            "申请用血量": use,
            "申请备血量": reserve,
            "血红蛋白": hb,
            "红细胞压积": np.clip(0.003 * hb + rng.normal(0, 0.02, n), 0.1, 0.55),
            "planned_use_offset_days": np.where(missing, 0, planned),
            "planned_use_missing": missing.astype(int),
            "输血要求分组": kind,
            "科室高频分组": [f"dept_{d}" for d in rng.integers(0, 6, size=n)],
        }
    )


def order_parameters(orders, context, config):
    alpha, gamma, _ = coefficients(config)
    use, reserve = orders["申请用血量"].to_numpy(), orders["申请备血量"].to_numpy()
    risk = np.clip((160 - orders["血红蛋白"].to_numpy()) / 100, 0, 1)
    urgent = orders["输血要求分组"].eq("急救用血").to_numpy()
    surgery = orders["输血要求分组"].eq("手术备血").to_numpy()
    planned_today = orders.planned_use_offset_days.eq(0).to_numpy() & orders.planned_use_missing.eq(0).to_numpy()
    # Same-department peer composition, excluding the order itself. No learned network.
    dept = orders["科室高频分组"].to_numpy()
    peers = (dept[:, None] == dept[None, :]) & ~np.eye(len(orders), dtype=bool)
    peer_risk = (peers @ risk) / np.maximum(peers.sum(axis=1), 1)
    logit = (
        -2
        + 0.35 * np.log1p(use)
        + 0.15 * np.log1p(reserve)
        + 1.2 * risk
        + 0.5 * urgent
        - 0.25 * orders.delta.to_numpy()
        + 0.4 * planned_today
        + alpha * (0.8 * context["scheduled_load"] * surgery + 0.6 * context["emergency_pressure"] * urgent)
        + gamma * peer_risk * risk
    )
    quantity_p = sigmoid(-0.5 + 0.3 * np.log1p(use) + 0.15 * np.log1p(reserve) + 0.7 * risk)
    return logit, quantity_p


def sample_demand(orders, context, config, rng, repeats=1):
    """Resample only future uncertainty at fixed observable inputs (also used by MC oracle)."""
    _, _, rho = coefficients(config)
    logit, quantity_p = order_parameters(orders, context, config)
    shock = rng.normal(size=repeats)
    issued = rng.random((repeats, len(orders))) < sigmoid(logit[None, :] + rho * shock[:, None])
    quantity = 0.5 * (1 + rng.binomial(5, quantity_p, size=(repeats, len(orders))))
    contributions = issued * quantity
    new_p = sigmoid(-0.7 + 0.4 * context["emergency_pressure"] + 0.3 * context["scheduled_load"] + rho * shock)
    n_new = rng.binomial(config["max_new_orders"], new_p)
    new_quantity_p = sigmoid(-0.4 + 0.5 * context["emergency_pressure"] + 0.2 * rho * shock)
    # Sum of n_new independent (0.5 + 0.5 Binomial(5,p)) quantities.
    new_demand = 0.5 * (n_new + rng.binomial(5 * n_new, new_quantity_p))
    return contributions, new_demand


def oracle_quantiles(orders, context, config, taus, repeats=20000, seed=10000):
    """Full-information conditional MC oracle; never conditions on the realized shock/outcomes."""
    contributions, new = sample_demand(orders, context, config, np.random.default_rng(seed), repeats)
    return np.quantile(contributions.sum(axis=1) + new, taus, method="higher")


def simulate_split(config, count, start, seed, label):
    split_key = [int(v) for v in seed.generate_state(4)]
    rows, days = [], []
    for day in pd.date_range(start, periods=count):
        # A day owns its random streams. Extending the training window cannot
        # change any overlapping day or either holdout set.
        day_key = day.toordinal()
        feature_rng = np.random.default_rng(np.random.SeedSequence([*split_key, day_key, 0]))
        demand_rng = np.random.default_rng(np.random.SeedSequence([*split_key, day_key, 1]))
        context = {"scheduled_load": feature_rng.uniform(-1, 1), "emergency_pressure": feature_rng.uniform(-1, 1)}
        orders = make_orders(feature_rng, context, config)
        inventory = (
            config["inventory_center"]
            + 20 * context["scheduled_load"]
            + feature_rng.uniform(-config["inventory_half_width"], config["inventory_half_width"])
        )
        contributions, new = sample_demand(orders, context, config, demand_rng)
        known, new = float(contributions.sum()), float(new[0])
        n = len(orders)
        days.append(
            {
                "decision_day": day,
                "target_day": day,
                "snapshot_day": day - pd.Timedelta(days=1),
                "information_cutoff": day,
                "source_split": label,
                "D": known + new,
                "D_new": new,
                "known_recent_D": known,
                "It": inventory,
                "n_orders": n,
                "supervised_eligible": True,
                "demand_observed": True,
                "inventory_observed": True,
                "dow_sin": np.sin(2 * np.pi * day.dayofweek / 7),
                "dow_cos": np.cos(2 * np.pi * day.dayofweek / 7),
                "is_weekend": float(day.dayofweek >= 5),
                **context,
                "request_use_sum": float(orders["申请用血量"].sum()),
                "request_reserve_sum": float(orders["申请备血量"].sum()),
                "order_age_mean": float(orders.delta.mean() + 1) if n else 0.0,
                "hb_mean": float(orders["血红蛋白"].mean()) if n else 0.0,
                "hct_mean": float(orders["红细胞压积"].mean()) if n else 0.0,
                "urgent_request_rate": float(orders["输血要求分组"].eq("急救用血").mean()) if n else 0.0,
                "surgery_prepare_rate": float(orders["输血要求分组"].eq("手术备血").mean()) if n else 0.0,
            }
        )
        orders["decision_day"] = day
        orders["order_key"] = [f"synthetic-{label}-{day_key}-{j}" for j in range(n)]
        orders["real_demand"] = contributions[0]
        orders["order_label_observed"] = True
        rows.append(orders)
    return pd.DataFrame(days), pd.concat(rows, ignore_index=True)


def scales(frame, columns):
    result = {}
    for col in columns:
        values = frame[col].to_numpy(float)
        mean, std = (float(values.mean()), float(values.std())) if len(values) else (0.0, 1.0)
        result[col] = {"mean": mean, "std": std if std > 1e-8 else 1.0}
    return result


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_fold(folder, train, evaluation, role, config):
    folder.mkdir(parents=True, exist_ok=False)
    train_daily, train_orders = train
    eval_daily, eval_orders = evaluation
    daily = pd.concat([train_daily.assign(split="train"), eval_daily.assign(split=role)], ignore_index=True)
    orders = pd.concat([train_orders, eval_orders], ignore_index=True)
    num, cat, day_cols = NUM_BASIC + NUM_CLINICAL, CAT_BASIC, DAY_CONTEXT + DAY_BASIC + DAY_CLINICAL
    train_end = str(train_daily.decision_day.max().date())
    schema = {
        "schema_version": "synthetic_axid_v1",
        "order_num_cols": num,
        "order_cat_cols": cat,
        "day_feature_cols": day_cols,
        "excluded_from_model_inputs": [
            "D",
            "D_new",
            "known_recent_D",
            "real_demand",
            "order_key",
            "latent_shock",
            "source_split",
        ],
        "information_groups": {
            "day_context": DAY_CONTEXT,
            "day_basic": DAY_BASIC,
            "day_clinical": DAY_CLINICAL,
            "order_basic_num": NUM_BASIC,
            "order_basic_cat": CAT_BASIC,
            "order_clinical_num": NUM_CLINICAL,
        },
        "feature_dependencies": {},
        "demand_support_upper": 3 * (config["max_visible_orders"] + config["max_new_orders"]),
    }
    state = {
        "feature_fit_end": train_end,
        "numeric_scalers": {
            "order": scales(train_orders, num),
            "day": scales(train_daily, day_cols),
            "inventory": scales(train_daily, ["It"]),
        },
    }
    split = {
        "train_start": str(train_daily.decision_day.min().date()),
        "train_end": train_end,
        "evaluation_start": str(eval_daily.decision_day.min().date()),
        "evaluation_end": str(eval_daily.decision_day.max().date()),
        "evaluation_role": role,
        "evaluation_is_retrospective": False,
        "source": "fully synthetic; no real records",
        "generator": config,
    }
    save_json(folder / "manifest.json", {"complete": False})
    daily.to_csv(folder / "daily_features.csv", index=False)
    orders.to_csv(folder / "data.csv", index=False)
    for name, value in [
        ("data_feature_schema.json", schema),
        ("preprocessing_state.json", state),
        ("split_manifest.json", split),
    ]:
        save_json(folder / name, value)
    save_json(
        folder / "manifest.json",
        {"complete": True, "artifacts": {p.name: file_hash(p) for p in folder.iterdir() if p.name != "manifest.json"}},
    )


def generate(config, output):
    config = dict(config)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported synthetic generator configuration")
    if config.get("generator_version", 2) != 2:
        raise ValueError("Unsupported synthetic random-stream version")
    config["generator_version"] = 2
    config.setdefault("validation_start", "2020-01-01")
    coefficients(config)
    if not 0 <= config.get("empty_set_probability", 0.05) < 1:
        raise ValueError("Empty-set probability must be in [0,1)")
    for key in ("n_train", "n_validation", "n_test", "max_visible_orders", "max_new_orders"):
        if not isinstance(config[key], int) or config[key] < (0 if key == "max_new_orders" else 1):
            raise ValueError(f"Invalid positive count: {key}")
    if config["inventory_center"] <= config["inventory_half_width"] + 20 or config["inventory_half_width"] < 0:
        raise ValueError("Inventory support must be nonnegative")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Synthetic output already exists; choose a fresh directory")
    seeds = np.random.SeedSequence(config["data_seed"]).spawn(3)
    anchor = pd.Timestamp(config["validation_start"])
    if pd.isna(anchor) or anchor != anchor.normalize() or anchor.tz is not None:
        raise ValueError("Validation start must be a valid timezone-naive calendar day")
    starts = [anchor - pd.Timedelta(days=config["n_train"]), anchor, anchor + pd.Timedelta(days=config["n_validation"])]
    splits = []
    for label, key, seed, start in zip(
        ("train", "validation", "test"), ("n_train", "n_validation", "n_test"), seeds, starts
    ):
        splits.append(simulate_split(config, config[key], start, seed, label))
    output.mkdir(parents=True, exist_ok=False)
    (output / "generation.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    export_fold(output / "validation", splits[0], splits[1], "validation", config)
    refit = tuple(pd.concat([splits[0][i], splits[1][i]], ignore_index=True) for i in (0, 1))
    export_fold(output / "test", refit, splits[2], "test", config)
    summary = []
    for label, (daily, orders) in zip(("train", "validation", "test"), splits):
        summary.append(
            {
                "split": label,
                "days": len(daily),
                "orders": len(orders),
                "mean_D": float(daily.D.mean()),
                "std_D": float(daily.D.std(ddof=0)),
                "q50_D": float(daily.D.quantile(0.5, interpolation="higher")),
                "q90_D": float(daily.D.quantile(0.9, interpolation="higher")),
                "q95_D": float(daily.D.quantile(0.95, interpolation="higher")),
                "max_D": float(daily.D.max()),
                "mean_I": float(daily.It.mean()),
                "mean_orders_per_day": float(daily.n_orders.mean()),
                "empty_set_days": int(daily.n_orders.eq(0).sum()),
                "new_demand_share": float(daily.D_new.sum() / daily.D.sum()) if daily.D.sum() else None,
            }
        )
    save_json(output / "summary.json", summary)
    return summary
