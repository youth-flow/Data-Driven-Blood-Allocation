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
    if (np.diff(p, axis=1) < -1e-5).any():
        raise ValueError("Quantile crossing")
    saa = np.array([higher_quantile(training_demand, t) for t in taus])
    rows = []
    for k, tau in enumerate(taus):
        residual = d - p[:, k]
        pinball = np.maximum(tau * residual, (tau - 1) * residual).mean()
        for target in targets:
            baseline = cost(saa[k], d, inventory, tau, target).mean()
            value = cost(p[:, k], d, inventory, tau, target).mean()
            rows.append(
                {
                    "tau": float(tau),
                    "target": float(target),
                    "days": len(d),
                    "mean_cost": float(value),
                    "saa_cost": float(baseline),
                    "P": float(1 - value / baseline) if baseline > 0 else None,
                    "P_defined": bool(baseline > 0),
                    "pinball": float(pinball),
                    "coverage": float((d <= p[:, k]).mean()),
                    "mean_q": float(decision(p[:, k], inventory, target).mean()),
                }
            )
    return rows


def selection_score(rows):
    valid = [r["P"] for r in rows if r["P_defined"]]
    if not valid:
        raise ValueError("Every SAA reference cost is zero; cannot select by P")
    return float(np.mean(valid))
