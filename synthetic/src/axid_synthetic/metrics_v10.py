"""AXID-DC v10 operational metrics for the single common target ``G=280``.

The functions in this module deliberately operate on frozen per-day arrays.  In
particular, the paired bootstrap never refits a model, regenerates a day, or
reruns the conditional Oracle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


TAU = 0.9
COMMON_G = 280.0
BOOTSTRAP_REPLICATIONS = 2_000


def _vector(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def left_quantile(values, tau=TAU):
    """Return the empirical left quantile ``inf{x: F_n(x) >= tau}``."""
    sample = _vector(values, "values")
    if not 0 < tau < 1:
        raise ValueError("tau must lie strictly between zero and one")
    return float(np.quantile(sample, tau, method="inverted_cdf"))


def replenishment(prediction, inventory, target=COMMON_G):
    prediction = np.asarray(prediction, dtype=float)
    inventory = np.asarray(inventory, dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(inventory).all():
        raise ValueError("prediction and inventory must be finite")
    if (prediction < 0).any() or (inventory < 0).any() or target <= 0:
        raise ValueError("prediction, inventory, and target must be nonnegative")
    return np.maximum(prediction + float(target) - inventory, 0.0)


def daily_outcomes(prediction, demand, inventory, tau=TAU, target=COMMON_G):
    """Compute policy outcomes, quantile coverage, and target service."""
    prediction = _vector(prediction, "prediction")
    demand = _vector(demand, "demand")
    inventory = _vector(inventory, "inventory")
    if prediction.shape != demand.shape or inventory.shape != demand.shape:
        raise ValueError("prediction, demand, and inventory must have equal shapes")
    if (prediction < 0).any() or (demand < 0).any() or (inventory < 0).any():
        raise ValueError("prediction, demand, and inventory must be nonnegative")
    if not 0 < tau < 1 or target <= 0:
        raise ValueError("invalid tau or target")
    order = replenishment(prediction, inventory, target)
    ending = inventory + order - demand
    deviation = ending - float(target)
    cost = tau * np.maximum(-deviation, 0.0) + (1.0 - tau) * np.maximum(deviation, 0.0)
    return {
        "replenishment": order,
        "ending_inventory": ending,
        "cost": cost,
        "coverage": demand <= prediction,
        "service": ending >= float(target),
    }


def saa_prediction(training_demand, size, tau=TAU):
    if int(size) <= 0:
        raise ValueError("size must be positive")
    return np.full(int(size), left_quantile(training_demand, tau), dtype=float)


def perfect_foresight_outcomes(demand, inventory, tau=TAU, target=COMMON_G):
    """Evaluate nonnegative-order perfect foresight; its cost need not be zero."""
    return daily_outcomes(demand, demand, inventory, tau=tau, target=target)


def summarize_outcomes(outcomes):
    cost = _vector(outcomes["cost"], "cost")
    coverage = np.asarray(outcomes["coverage"])
    service = np.asarray(outcomes["service"])
    if (
        coverage.ndim != 1
        or coverage.shape != cost.shape
        or service.ndim != 1
        or service.shape != cost.shape
    ):
        raise ValueError("coverage and service must be one-dimensional arrays aligned with cost")
    return {
        "MeanCost": float(cost.mean()),
        "Coverage_0.9": float(coverage.astype(float).mean()),
        "SL_G": float(service.astype(float).mean()),
    }


def prescriptiveness(method_cost, saa_cost, reference_cost, *, atol=1e-12, rtol=1e-10):
    """Cost-difference Prescriptiveness without clipping to ``[0, 1]``."""
    method_cost, saa_cost, reference_cost = map(float, (method_cost, saa_cost, reference_cost))
    if not np.isfinite([method_cost, saa_cost, reference_cost]).all():
        raise ValueError("costs must be finite")
    numerator = saa_cost - method_cost
    denominator = saa_cost - reference_cost
    tolerance = max(float(atol), float(rtol) * max(1.0, abs(saa_cost)))
    defined = bool(denominator > tolerance)
    return {
        "value": numerator / denominator if defined else np.nan,
        "defined": defined,
        "numerator": numerator,
        "denominator": denominator,
    }


def evaluate_single_method(
    prediction,
    demand,
    inventory,
    training_demand,
    oracle_prediction,
    *,
    tau=TAU,
    target=COMMON_G,
):
    """Return frozen daily arrays and the four v10 performance metrics."""
    demand = _vector(demand, "demand")
    inventory = _vector(inventory, "inventory")
    if demand.shape != inventory.shape:
        raise ValueError("demand and inventory must have equal shapes")
    method = daily_outcomes(prediction, demand, inventory, tau=tau, target=target)
    saa = daily_outcomes(
        saa_prediction(training_demand, len(demand), tau), demand, inventory, tau=tau, target=target
    )
    oracle = daily_outcomes(oracle_prediction, demand, inventory, tau=tau, target=target)
    pf = perfect_foresight_outcomes(demand, inventory, tau=tau, target=target)
    method_summary = summarize_outcomes(method)
    saa_cost = float(np.mean(saa["cost"]))
    pf_p = prescriptiveness(method_summary["MeanCost"], saa_cost, float(np.mean(pf["cost"])))
    oracle_p = prescriptiveness(method_summary["MeanCost"], saa_cost, float(np.mean(oracle["cost"])))
    return {
        **method_summary,
        "P_PF": pf_p["value"],
        "P_PF_defined": pf_p["defined"],
        "P_Oracle": oracle_p["value"],
        "P_Oracle_defined": oracle_p["defined"],
        "daily": {"method": method, "SAA": saa, "PF": pf, "Oracle": oracle},
    }


def _aligned_arrays(values: Mapping[str, Sequence[float]], name):
    if not values:
        raise ValueError(f"{name} cannot be empty")
    converted = {key: _vector(value, f"{name}[{key!r}]") for key, value in values.items()}
    lengths = {len(value) for value in converted.values()}
    if len(lengths) != 1:
        raise ValueError(f"all {name} arrays must be aligned")
    return converted


def paired_bootstrap(
    method_costs: Mapping[str, Sequence[float]],
    method_services: Mapping[str, Sequence[float]],
    *,
    method_coverages: Mapping[str, Sequence[float]],
    saa_cost: Sequence[float],
    pf_cost: Sequence[float],
    oracle_cost: Sequence[float],
    replications=BOOTSTRAP_REPLICATIONS,
    seed=2026111406,
    chunk_size=100,
):
    """Paired day bootstrap for MeanCost, two service rates, PF-P, and Oracle-P.

    One common resampled day-index vector is used for every method and all
    three references in each replication.  This routine consumes saved arrays;
    it does not refit or call an Oracle.
    """
    costs = _aligned_arrays(method_costs, "method_costs")
    services = _aligned_arrays(method_services, "method_services")
    coverages = _aligned_arrays(method_coverages, "method_coverages")
    if set(costs) != set(services) or set(costs) != set(coverages):
        raise ValueError("method costs, coverages, and services must have identical keys")
    n = len(next(iter(costs.values())))
    references = {
        "SAA": _vector(saa_cost, "saa_cost"),
        "PF": _vector(pf_cost, "pf_cost"),
        "Oracle": _vector(oracle_cost, "oracle_cost"),
    }
    if any(len(value) != n for value in references.values()):
        raise ValueError("reference costs must align with method arrays")
    replications, chunk_size = int(replications), int(chunk_size)
    if replications <= 0 or chunk_size <= 0:
        raise ValueError("replications and chunk_size must be positive")

    names = list(costs)
    cost_matrix = np.stack([costs[name] for name in names])
    service_matrix = np.stack([services[name] for name in names])
    coverage_matrix = np.stack([coverages[name] for name in names])
    rng = np.random.default_rng(seed)
    draws = {
        name: {
            metric: []
            for metric in ("MeanCost", "Coverage_0.9", "SL_G", "P_PF", "P_Oracle")
        }
        for name in names
    }
    for start in range(0, replications, chunk_size):
        width = min(chunk_size, replications - start)
        indices = rng.integers(0, n, size=(width, n))
        saa_mean = references["SAA"][indices].mean(axis=1)
        pf_mean = references["PF"][indices].mean(axis=1)
        oracle_mean = references["Oracle"][indices].mean(axis=1)
        for row, name in enumerate(names):
            cost_mean = cost_matrix[row][indices].mean(axis=1)
            service_mean = service_matrix[row][indices].mean(axis=1)
            coverage_mean = coverage_matrix[row][indices].mean(axis=1)
            pf_denominator = saa_mean - pf_mean
            oracle_denominator = saa_mean - oracle_mean
            scale = np.maximum(1.0, np.abs(saa_mean))
            pf_valid = pf_denominator > np.maximum(1e-12, 1e-10 * scale)
            oracle_valid = oracle_denominator > np.maximum(1e-12, 1e-10 * scale)
            p_pf = np.full(width, np.nan)
            p_oracle = np.full(width, np.nan)
            np.divide(saa_mean - cost_mean, pf_denominator, out=p_pf, where=pf_valid)
            np.divide(saa_mean - cost_mean, oracle_denominator, out=p_oracle, where=oracle_valid)
            for metric, values in (
                ("MeanCost", cost_mean),
                ("Coverage_0.9", coverage_mean),
                ("SL_G", service_mean),
                ("P_PF", p_pf),
                ("P_Oracle", p_oracle),
            ):
                draws[name][metric].append(values)

    rows = []
    saa_estimate = float(references["SAA"].mean())
    pf_estimate = float(references["PF"].mean())
    oracle_estimate = float(references["Oracle"].mean())
    for name in names:
        estimates = {
            "MeanCost": float(costs[name].mean()),
            "Coverage_0.9": float(coverages[name].mean()),
            "SL_G": float(services[name].mean()),
            "P_PF": prescriptiveness(float(costs[name].mean()), saa_estimate, pf_estimate)["value"],
            "P_Oracle": prescriptiveness(float(costs[name].mean()), saa_estimate, oracle_estimate)["value"],
        }
        for metric, chunks in draws[name].items():
            values = np.concatenate(chunks)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "method": name,
                    "metric": metric,
                    "estimate": estimates[metric],
                    "ci_lower": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_upper": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "bootstrap_defined": int(len(finite)),
                    "bootstrap_replications": replications,
                    "n_test": n,
                    "seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def paired_difference_bootstrap(
    left,
    right,
    *,
    replications=BOOTSTRAP_REPLICATIONS,
    seed=2026111406,
    chunk_size=100,
):
    """Bootstrap the mean of a same-day ``left - right`` difference."""
    left = _vector(left, "left")
    right = _vector(right, "right")
    if left.shape != right.shape:
        raise ValueError("left and right must be aligned daily arrays")
    replications, chunk_size = int(replications), int(chunk_size)
    if replications <= 0 or chunk_size <= 0:
        raise ValueError("replications and chunk_size must be positive")
    delta = left - right
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replications, chunk_size):
        width = min(chunk_size, replications - start)
        indices = rng.integers(0, len(delta), size=(width, len(delta)))
        means.append(delta[indices].mean(axis=1))
    values = np.concatenate(means)
    return {
        "estimate": float(delta.mean()),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "bootstrap_replications": replications,
        "n_test": len(delta),
        "seed": int(seed),
    }
