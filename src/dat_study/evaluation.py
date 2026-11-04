"""One decision and evaluation contract for every method."""

import numpy as np


def higher_quantile(values, tau):
    return float(np.quantile(np.asarray(values, float), tau, method="higher"))


def decision(quantile, inventory, target):
    return np.maximum(np.asarray(quantile) + target - np.asarray(inventory), 0.0)


def cost(quantile, demand, inventory, tau, target):
    ending_deviation = inventory + decision(quantile, inventory, target) - demand - target
    return tau * np.maximum(-ending_deviation, 0) + (1 - tau) * np.maximum(ending_deviation, 0)


def evaluate(predictions, demand, inventory, training_demand, taus, targets):
    p, d, inventory = (np.asarray(v, dtype=float) for v in (predictions, demand, inventory))
    if p.shape != (len(d), len(taus)) or inventory.shape != d.shape or not len(d):
        raise ValueError("Invalid daily evaluation dimensions")
    if not all(np.isfinite(v).all() for v in (p, d, inventory)) or (p < 0).any() or (d < 0).any():
        raise ValueError("Evaluation requires finite nonnegative demand and quantiles")
    crossing_share = float((np.diff(p, axis=1) < -1e-5).any(axis=1).mean()) if len(taus) > 1 else 0.0
    saa = np.array([higher_quantile(training_demand, t) for t in taus])
    rows = []
    for k, tau in enumerate(taus):
        residual = d - p[:, k]
        pinball = np.maximum(tau * residual, (tau - 1) * residual).mean()
        for target in targets:
            baseline = cost(saa[k], d, inventory, tau, target).mean()
            value = cost(p[:, k], d, inventory, tau, target).mean()
            allocation = decision(p[:, k], inventory, target)
            ending = inventory + allocation - d
            saa_allocation = decision(saa[k], inventory, target)
            saa_ending = inventory + saa_allocation - d
            rows.append(
                {
                    "tau": float(tau),
                    "target": float(target),
                    "days": len(d),
                    "n_test": len(d),
                    "mean_cost": float(value),
                    "TC": float(value * len(d)),
                    "SL_G": float((ending >= target).mean()),
                    "saa_cost": float(baseline),
                    "saa_TC": float(baseline * len(d)),
                    "saa_SL_G": float((saa_ending >= target).mean()),
                    "P": float(1 - value / baseline) if baseline > 0 else None,
                    "P_defined": bool(baseline > 0),
                    "pinball": float(pinball),
                    "coverage": float((d <= p[:, k]).mean()),
                    "crossing_share": crossing_share,
                    "mean_q": float(allocation.mean()),
                    "binding_share": float((allocation == 0).mean()),
                }
            )
    return rows


def selection_score(rows):
    valid = [r["P"] for r in rows if r["P_defined"]]
    if not valid:
        raise ValueError("Every SAA reference cost is zero; cannot select by P")
    return float(np.mean(valid))


def evaluate_allocations(allocations, demand, inventory, training_demand, taus, targets):
    """Evaluate observed actions directly; they are not demand quantile forecasts."""
    q, d, inventory = (np.asarray(v, float) for v in (allocations, demand, inventory))
    if q.shape != (len(d), len(taus)) or inventory.shape != d.shape or not len(d):
        raise ValueError("Invalid allocation evaluation dimensions")
    if not all(np.isfinite(v).all() for v in (q, d, inventory)) or (q < 0).any() or (d < 0).any():
        raise ValueError("Invalid observed allocation or demand")
    rows = []
    for j, tau in enumerate(taus):
        reference = higher_quantile(training_demand, tau)
        for target in targets:
            deviation = inventory + q[:, j] - d - target
            value = float((tau * np.maximum(-deviation, 0) + (1 - tau) * np.maximum(deviation, 0)).mean())
            baseline = float(cost(reference, d, inventory, tau, target).mean())
            ending = inventory + q[:, j] - d
            rows.append(
                {
                    "tau": float(tau),
                    "target": float(target),
                    "days": len(d),
                    "n_test": len(d),
                    "mean_cost": value,
                    "TC": float(value * len(d)),
                    "SL_G": float((ending >= target).mean()),
                    "saa_cost": baseline,
                    "saa_TC": float(baseline * len(d)),
                    "P": 1 - value / baseline if baseline > 0 else None,
                    "P_defined": baseline > 0,
                    "pinball": None,
                    "coverage": None,
                    "mean_q": float(q[:, j].mean()),
                    "binding_share": float((q[:, j] == 0).mean()),
                    "output_kind": "allocation",
                }
            )
    return rows
