"""Version 3: bounded IID compact AXID; no real-data or predictor dependency."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml

SCENARIOS = {
    "S4": (1.0, 1.5, 0.8),
    "C0": (0.0, 0.0, 0.0),
}
NUM_BASIC = ["申请用血量", "申请备血量"]
NUM_CLINICAL = ["clinical_risk"]
CAT_BASIC = []
DAY_CONTEXT = ["scheduled_load", "emergency_pressure"]
DAY_BASIC = ["n_orders", "request_use_sum", "request_reserve_sum"]
DAY_CLINICAL = ["risk_mean", "risk_second_moment"]


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def coefficients(config):
    return SCENARIOS[config["scenario"]]


def make_orders(rng, context, config):
    maximum = config["max_visible_orders"]
    n = (
        maximum
        if config["scenario"] == "C0"
        else 1 + rng.binomial(maximum - 1, sigmoid(-0.2 + 0.4 * context["scheduled_load"]))
    )
    risk = rng.beta(2, 3, size=n)
    return pd.DataFrame(
        {
            "申请用血量": 0.5 * rng.binomial(12, 0.25 + 0.25 * risk),
            "申请备血量": 0.5 * rng.binomial(16, 0.15 + 0.2 * risk),
            "clinical_risk": risk,
        }
    )


def order_parameters(orders, context, config):
    if config["scenario"] == "C0":
        return np.full(len(orders), np.log(0.35 / 0.65)), np.full(len(orders), 0.4)
    alpha, gamma, _ = coefficients(config)
    use, reserve = orders["申请用血量"].to_numpy(float), orders["申请备血量"].to_numpy(float)
    risk = orders.clinical_risk.to_numpy(float)
    u = use / 6
    kernel = np.exp(-((u[:, None] - u[None, :]) ** 2) / (2 * 0.25**2))
    np.fill_diagonal(kernel, 0)
    peer = (kernel * risk[None, :]).sum(1) / np.maximum(kernel.sum(1), 1e-12)
    logit = (
        -2
        + 0.35 * np.log1p(use)
        + 0.15 * np.log1p(reserve)
        + 1.2 * risk
        + alpha * (0.8 * context["scheduled_load"] * u + 0.6 * context["emergency_pressure"] * risk)
        + gamma * risk * peer
    )
    return logit, sigmoid(-0.5 + 0.3 * np.log1p(use) + 0.15 * np.log1p(reserve) + 0.7 * risk)


def new_parameters(context, config, shock):
    if config["scenario"] == "C0":
        return np.zeros_like(shock) + 0.3, np.zeros_like(shock) + 0.4
    rho = coefficients(config)[2]
    return (
        sigmoid(-0.7 + 0.4 * context["emergency_pressure"] + 0.3 * context["scheduled_load"] + rho * shock),
        sigmoid(-0.4 + 0.5 * context["emergency_pressure"] + 0.2 * rho * shock),
    )


def sample_demand(orders, context, config, rng, repeats=1):
    rho = coefficients(config)[2]
    logit, qp = order_parameters(orders, context, config)
    shock = rng.normal(size=repeats)
    issued = rng.random((repeats, len(orders))) < sigmoid(logit[None, :] + rho * shock[:, None])
    quantity = 0.5 * (1 + rng.binomial(5, qp, size=(repeats, len(orders))))
    probability, amount = new_parameters(context, config, shock)
    count = rng.binomial(config["max_new_orders"], probability)
    new = 0.5 * (count + rng.binomial(5 * count, amount))
    new += rng.uniform(0, config.get("remainder_width", 0.5), repeats)
    return issued * quantity, new


def oracle_quantiles(orders, context, config, taus, repeats=None, seed=None):
    from .oracle import conditional_pmf, mixture_quantiles

    pmf, _ = conditional_pmf(orders, context, config)
    return mixture_quantiles(pmf, taus, config.get("remainder_width", 0.5))


def simulate_split(config, count, start, seed, label):
    split_key = [int(v) for v in seed.generate_state(4)]
    rows, days = [], []
    for day in pd.date_range(start, periods=count):
        key = day.toordinal()
        rng = np.random.default_rng(np.random.SeedSequence([*split_key, key, 0]))
        demand_rng = np.random.default_rng(np.random.SeedSequence([*split_key, key, 1]))
        context = dict(zip(DAY_CONTEXT, rng.uniform(-1, 1, 2)))
        orders = make_orders(rng, context, config)
        inventory = (
            config["inventory_center"]
            + 20 * context["scheduled_load"]
            + rng.uniform(-config["inventory_half_width"], config["inventory_half_width"])
        )
        contributions, new = sample_demand(orders, context, config, demand_rng)
        known = float(contributions.sum())
        days.append(
            {
                "decision_day": day,
                "target_day": day,
                "snapshot_day": day - pd.Timedelta(days=1),
                "information_cutoff": day,
                "source_split": label,
                "D": known + float(new[0]),
                "D_new": float(new[0]),
                "known_recent_D": known,
                "It": inventory,
                "n_orders": len(orders),
                "supervised_eligible": True,
                "demand_observed": True,
                "inventory_observed": True,
                **context,
                "request_use_sum": float(orders["申请用血量"].sum()),
                "request_reserve_sum": float(orders["申请备血量"].sum()),
                "risk_mean": float(orders.clinical_risk.mean()),
                "risk_second_moment": float(orders.clinical_risk.pow(2).mean()),
            }
        )
        orders["decision_day"] = day
        orders["order_key"] = [f"synthetic-{label}-{key}-{j}" for j in range(len(orders))]
        orders["real_demand"] = contributions[0]
        orders["order_label_observed"] = True
        rows.append(orders)
    return pd.DataFrame(days), pd.concat(rows, ignore_index=True)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support_upper(config):
    return 3 * (config["max_visible_orders"] + config["max_new_orders"]) + config.get("remainder_width", 0.5)


def fixed_scalers(config):
    m = config["max_visible_orders"]
    bounds = {
        "申请用血量": (0, 6),
        "申请备血量": (0, 8),
        "clinical_risk": (0, 1),
        "scheduled_load": (-1, 2),
        "emergency_pressure": (-1, 2),
        "n_orders": (0, m),
        "request_use_sum": (0, 6 * m),
        "request_reserve_sum": (0, 8 * m),
        "risk_mean": (0, 1),
        "risk_second_moment": (0, 1),
        "It": (
            config["inventory_center"] - 20 - config["inventory_half_width"],
            40 + 2 * config["inventory_half_width"],
        ),
    }

    def section(cols):
        return {c: {"mean": bounds[c][0], "std": bounds[c][1]} for c in cols}

    return {
        "order": section(NUM_BASIC + NUM_CLINICAL),
        "day": section(DAY_CONTEXT + DAY_BASIC + DAY_CLINICAL),
        "inventory": section(["It"]),
    }


def export_fold(folder, train, evaluation, role, config, create=True):
    if create:
        folder.mkdir(parents=True, exist_ok=False)
    train_daily, train_orders = train
    eval_daily, eval_orders = evaluation
    daily = pd.concat([train_daily.assign(split="train"), eval_daily.assign(split=role)], ignore_index=True)
    orders = pd.concat([train_orders, eval_orders], ignore_index=True)
    train_end = str(train_daily.decision_day.max().date())
    schema = {
        "schema_version": "synthetic_axid_v3",
        "generator_version": 4,
        "order_num_cols": NUM_BASIC + NUM_CLINICAL,
        "order_cat_cols": [],
        "day_feature_cols": DAY_CONTEXT + DAY_BASIC + DAY_CLINICAL,
        "excluded_from_model_inputs": [
            "D",
            "D_new",
            "known_recent_D",
            "real_demand",
            "order_key",
            "latent_shock",
            "source_split",
            "oracle_quantile",
        ],
        "information_groups": {
            "day_context": DAY_CONTEXT,
            "day_basic": DAY_BASIC,
            "day_clinical": DAY_CLINICAL,
            "order_basic_num": NUM_BASIC,
            "order_basic_cat": [],
            "order_clinical_num": NUM_CLINICAL,
        },
        "feature_dependencies": {},
        "demand_support_upper": support_upper(config),
        "max_orders": config["max_visible_orders"],
        "scaling": "fixed_known_support",
        "iid_days": True,
    }
    state = {
        "feature_fit_end": train_end,
        "numeric_scalers": fixed_scalers(config),
        "scaling_source": "predeclared_support_not_fitted_to_samples",
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


def validate_config(config):
    if config.get("schema_version") != 2 or config.get("generator_version") != 4:
        raise ValueError("Use the current version-4 synthetic configuration")
    if config["scenario"] != "S4":
        raise ValueError("Performance data generation is S4-only")
    for key in ("n_train", "n_test", "max_visible_orders", "max_new_orders"):
        if not isinstance(config[key], int) or config[key] < (0 if key == "max_new_orders" else 1):
            raise ValueError(f"Invalid count: {key}")
    if config.get("remainder_width", 0.5) not in (0.0, 0.5):
        raise ValueError("Use full-cell continuous demand or the explicitly discrete sensitivity")
    if config["inventory_center"] <= config["inventory_half_width"] + 20 or config["inventory_half_width"] < 0:
        raise ValueError("Inventory support must be positive")


def generate_performance(config, output):
    """S4 performance fold: 2,500 training days and 3,000 independent test days."""
    config = dict(config)
    validate_config(config)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Synthetic output already exists; choose a fresh directory")
    seeds = np.random.SeedSequence(config["data_seed"]).spawn(2)
    anchor = pd.Timestamp(config.get("validation_start", "2020-01-01"))
    train = simulate_split(config, config["n_train"], anchor - pd.Timedelta(days=config["n_train"]), seeds[0], "train")
    test = simulate_split(config, config["n_test"], anchor, seeds[1], "test")
    output.mkdir(parents=True, exist_ok=False)
    (output / "generation.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    export_fold(output, train, test, "test", config, create=False)
    summary = [
        {
            "split": label,
            "days": len(d),
            "orders": len(o),
            "mean_D": float(d.D.mean()),
            "std_D": float(d.D.std(ddof=0)),
            "mean_I": float(d.It.mean()),
            "max_D": float(d.D.max()),
            "empty_set_days": int(d.n_orders.eq(0).sum()),
            "new_demand_share": float(d.D_new.sum() / d.D.sum()),
        }
        for label, (d, o) in zip(("train", "test"), (train, test))
    ]
    save_json(output / "summary.json", summary)
    return summary
