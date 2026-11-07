"""Full-information AXID-v3 Oracle and PrescriptiveP (phase two)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "synthetic/src"))

from axid_synthetic.generator_v3 import mechanism_features, new_parameters, sigmoid  # noqa: E402
from axid_synthetic.oracle import _GH_25  # noqa: E402
from dat_study.data import FoldData  # noqa: E402

TAU = 0.9
TARGETS = [260, 280, 300, 320, 340]


def _patient_pmf(length, quantity):
    positive = binom.pmf(np.arange(6)[None, :], 5, quantity[:, None])
    probability = sigmoid(length)
    pmf = np.empty((len(length), 7))
    pmf[:, 0] = 1.0 - probability
    pmf[:, 1:] = probability[:, None] * positive
    return pmf


def _new_pmf(arrival, amount):
    positive = binom.pmf(np.arange(6), 5, amount)
    pmf = np.empty(7)
    pmf[0] = 1.0 - arrival
    pmf[1:] = arrival * positive
    return pmf


def conditional_pmf(orders, c1, c2):
    """Joint patient + 12 potential-new-order PMF after integrating the common shock."""
    u = orders["u"].to_numpy(float)
    v = orders["v"].to_numpy(float)
    risk = orders["risk"].to_numpy(float)
    surgery = orders["surgery_type"].to_numpy(float)
    z, _, _, h = mechanism_features(u, v, risk, surgery, c1, c2)
    zbar = float(z.mean())
    points, weights = _GH_25
    laws = []
    for shock in points:
        length = -1.0 + 0.6 * u + 0.4 * v + 0.6 * z + 0.4 * surgery + 3.0 * h + 0.3 * c1 + 0.2 * c2 + 0.45 * shock
        quantity = sigmoid(-0.4 + 0.5 * u + 0.3 * v + h)
        p_new, pi_new = new_parameters(h, zbar, c1, c2, shock)
        pmf = np.ones(1)
        for one in _patient_pmf(length, quantity):
            pmf = np.convolve(pmf, one)
        one_new = _new_pmf(p_new, pi_new)
        for _ in range(12):
            pmf = np.convolve(pmf, one_new)
        laws.append(pmf)
    width = max(len(law) for law in laws)
    stacked = np.zeros((len(laws), width))
    for index, law in enumerate(laws):
        stacked[index, : len(law)] = law
    result = weights @ stacked
    result = np.maximum(result, 0.0)
    result /= result.sum()
    return result


def quantile(pmf, tau=TAU):
    cumulative = np.cumsum(pmf)
    cumulative[-1] = 1.0
    k = int(np.searchsorted(cumulative, tau, side="left"))
    k = min(k, len(pmf) - 1)
    previous = cumulative[k - 1] if k else 0.0
    return 0.5 * k + 0.5 * (tau - previous) / pmf[k]


def cost_for_target(quantile, inventory, demand, target):
    decision = np.maximum(quantile + target - inventory, 0.0)
    ending = inventory + decision - demand
    return float(np.mean(0.9 * np.maximum(target - ending, 0.0) + 0.1 * np.maximum(ending - target, 0.0)))


def oracle_costs(output):
    rows = []
    for repeat in range(10):
        data = FoldData(output / "datasets" / f"AXIDv3_r{repeat}")
        daily = data.daily_split("eval").reset_index(drop=True)
        orders = data.order_split("eval")
        groups = dict(tuple(orders.groupby("decision_day", sort=False)))
        quantiles = np.empty(len(daily))
        for index, row in enumerate(daily.to_dict("records")):
            frame = groups[row["decision_day"]]
            pmf = conditional_pmf(frame, row["scheduled_load"], row["emergency_pressure"])
            quantiles[index] = quantile(pmf)
        inventory = daily.It.to_numpy(float)
        demand = daily.D.to_numpy(float)
        for target in TARGETS:
            rows.append(
                {
                    "data_repeat": repeat,
                    "target": float(target),
                    "OracleMeanCost": cost_for_target(quantiles, inventory, demand, target),
                    "OracleQuantileMean": float(quantiles.mean()),
                }
            )
        print("oracle repeat", repeat, "done", flush=True)
    return pd.DataFrame(rows)


def prescriptive(output, oracle):
    metrics = pd.read_csv(output / "performance_metrics.csv")
    keys = ["experiment", "method", "structure_variant", "information_level", "tau", "target"]
    per_repeat = metrics.groupby([*keys, "data_repeat"], dropna=False)[["MeanCost", "SL_G"]].mean().reset_index()
    saa = metrics.groupby(["data_repeat", "target"]).saa_MeanCost.first().reset_index()
    oracle = oracle.rename(columns={"OracleMeanCost": "oracle_cost"})
    rows = []
    for experiment, index in (
        ("comparison", "method"),
        ("ablation", "structure_variant"),
        ("information", "information_level"),
    ):
        block = per_repeat[per_repeat.experiment.eq(experiment)].copy()
        block["information_level"] = block.information_level.fillna("")
        for key, table in block.groupby(index, dropna=False):
            if experiment == "comparison" and key == "SAA":
                continue
            merged = table.merge(saa, on=["data_repeat", "target"], how="left").merge(
                oracle, on=["data_repeat", "target"], how="left"
            )
            for target, group in merged.groupby("target"):
                numerator = float((group.saa_MeanCost - group.MeanCost).mean())
                denominator = float((group.saa_MeanCost - group.oracle_cost).mean())
                rows.append(
                    {
                        "experiment": experiment,
                        "condition": key,
                        "target": float(target),
                        "n_repeats": len(group),
                        "SAA_cost": group.saa_MeanCost.mean(),
                        "method_cost": group.MeanCost.mean(),
                        "oracle_cost": group.oracle_cost.mean(),
                        "numerator": numerator,
                        "denominator": denominator,
                        "PrescriptiveP": numerator / denominator if abs(denominator) > 1e-12 else np.nan,
                    }
                )
            # Five-G aggregate: sum cost differences before the single ratio.
            numerator_5g = float((merged.saa_MeanCost - merged.MeanCost).sum() / merged.data_repeat.nunique())
            denominator_5g = float((merged.saa_MeanCost - merged.oracle_cost).sum() / merged.data_repeat.nunique())
            rows.append(
                {
                    "experiment": experiment,
                    "condition": key,
                    "target": 0.0,
                    "n_repeats": merged.data_repeat.nunique(),
                    "SAA_cost": merged.groupby("data_repeat").saa_MeanCost.first().mean(),
                    "method_cost": merged.groupby("data_repeat").MeanCost.first().mean(),
                    "oracle_cost": merged.groupby("data_repeat").oracle_cost.first().mean(),
                    "numerator": numerator_5g,
                    "denominator": denominator_5g,
                    "PrescriptiveP": numerator_5g / denominator_5g if abs(denominator_5g) > 1e-12 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main():
    output = ROOT / "outputs/performance_axid_v3_main"
    report = ROOT / "outputs/performance_axid_v3_report"
    report.mkdir(parents=True, exist_ok=True)
    oracle_path = report / "oracle_costs.csv"
    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
    else:
        oracle = oracle_costs(output)
        oracle.to_csv(oracle_path, index=False)
    p_table = prescriptive(output, oracle)
    p_table.to_csv(report / "prescriptive_p.csv", index=False)
    print(p_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
