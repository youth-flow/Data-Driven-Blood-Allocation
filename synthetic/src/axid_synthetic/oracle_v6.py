"""AXID-DC v6 Full Oracle over hidden A+ and shared doctor-choice shock."""


import numpy as np
import pandas as pd

from .generator_v6 import action_probabilities, profile_probabilities, sigmoid
from .oracle import _GH_25
from dat_study.data import FoldData

TAU = 0.9
TARGETS = [260, 280, 300, 320, 340]
UPPER = 61.0


def conditional_pmf(orders, cp, ce, inventory):
    old = orders[["u", "r", "s"]].to_numpy(float)
    profiles, profile_probability = profile_probabilities(cp, ce)
    p_plus = sigmoid(-0.4 + 0.5 * ce)
    state_probability = np.concatenate([[1.0 - p_plus], p_plus * profile_probability])
    state_probability /= state_probability.sum()
    combinations = [(i, j) for i in range(9) for j in range(i, 9)]
    combination_weights = np.array(
        [(2.0 if i != j else 1.0) * state_probability[i] * state_probability[j] for i, j in combinations]
    )
    combination_weights /= combination_weights.sum()
    points, weights = _GH_25
    width = 6 * (len(old) + 2) + 1
    law = np.zeros(width)
    for (left, right), combination_weight in zip(combinations, combination_weights):
        additions = []
        if left:
            additions.append(profiles[left - 1])
        if right:
            additions.append(profiles[right - 1])
        patients = np.vstack([old, *additions]) if additions else old
        for shock, quadrature_weight in zip(points, weights):
            probabilities = action_probabilities(patients, cp, ce, inventory, shock)
            pmf = np.ones(1)
            for row in probabilities:
                pmf = np.convolve(pmf, row)
            law[: len(pmf)] += combination_weight * quadrature_weight * pmf
    if law.min() < -1e-13 or abs(law.sum() - 1.0) > 1e-9:
        raise ArithmeticError("AXID-DC v6 conditional PMF is not normalized")
    return law


def quantile(pmf, tau=TAU):
    cumulative = np.cumsum(pmf)
    k = int(np.searchsorted(cumulative, tau, side="left"))
    previous = cumulative[k - 1] if k else 0.0
    return float(k) + (tau - previous) / pmf[k]


def cost_and_sl(qhat, inventory, demand, target):
    decision = np.maximum(qhat + target - inventory, 0.0)
    ending = inventory + decision - demand
    cost = float(np.mean(0.9 * np.maximum(target - ending, 0.0) + 0.1 * np.maximum(ending - target, 0.0)))
    service = float(np.mean(ending >= target))
    return cost, service


def oracle_costs(output):
    rows = []
    for repeat in range(10):
        data = FoldData(output / "datasets" / f"AXIDv6_r{repeat}")
        daily = data.daily_split("eval").reset_index(drop=True)
        orders = data.order_split("eval")
        groups = dict(tuple(orders.groupby("decision_day", sort=False)))
        quantiles = np.empty(len(daily))
        for index, row in enumerate(daily.to_dict("records")):
            frame = groups[row["decision_day"]]
            pmf = conditional_pmf(frame, row["planned_load"], row["emergency_load"], row["It"])
            quantiles[index] = quantile(pmf)
        inventory = daily.It.to_numpy(float)
        demand = daily.D.to_numpy(float)
        for target in TARGETS:
            cost, service = cost_and_sl(quantiles, inventory, demand, target)
            rows.append(
                {
                    "data_repeat": repeat,
                    "target": float(target),
                    "OracleMeanCost": cost,
                    "OracleSL_G": service,
                    "OracleQuantileMean": float(quantiles.mean()),
                }
            )
        print("oracle repeat", repeat, "done", flush=True)
    return pd.DataFrame(rows)
