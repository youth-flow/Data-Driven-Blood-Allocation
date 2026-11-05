"""Aggregate S4-only independent replicates and build the frozen report figures."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def _ci95(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        return float("nan")
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def _save(frame, path):
    frame.to_csv(path, index=False, float_format="%.8g")


def _theory_tables(output, data):
    c0 = data[data.group.eq("estimation") & data.scenario.eq("C0") & data.estimator.eq("exact_bias_only")]
    if not c0.empty:
        rows = []
        for n_train, table in c0.groupby("n_train"):
            for delta in ("0.1", "0.05"):
                rows.append(
                    {
                        "n_train": n_train,
                        "repetitions": table.data_repeat.nunique(),
                        "mean_excess": table.excess_pinball.mean(),
                        "median_excess": table.excess_pinball.median(),
                        "q90_excess": table.excess_pinball.quantile(0.9),
                        "q95_excess": table.excess_pinball.quantile(0.95),
                        "delta": float(delta),
                        "theoretical_bound": table[f"bound_delta_{delta}"].iloc[0],
                        "exceedance_rate": table[f"exceeds_bound_delta_{delta}"].mean(),
                        "bound_is_trivial": bool(table.bound_is_trivial.iloc[0]),
                    }
                )
        _save(pd.DataFrame(rows), output / "T1_C0_summary.csv")

    dat = data[data.group.eq("estimation") & data.scenario.eq("S4")]
    if not dat.empty:
        rows = []
        for n_train, table in dat.groupby("n_train"):
            rows.append(
                {
                    "scenario": "S4",
                    "n_train": n_train,
                    "repetitions": table.data_repeat.nunique(),
                    "mean_excess": table.excess_pinball.mean(),
                    "median_excess": table.excess_pinball.median(),
                    "q90_excess": table.excess_pinball.quantile(0.9),
                    "q95_excess": table.excess_pinball.quantile(0.95),
                    "mean_regret": table.operational_regret.mean(),
                }
            )
        _save(pd.DataFrame(rows), output / "T1_DAT_summary.csv")

    approximation = data[data.group.eq("approximation")]
    if not approximation.empty:
        rows = []
        for parameters, table in approximation.groupby("parameters"):
            rows.append(
                {
                    "parameters": parameters,
                    "hidden": table.hidden.iloc[0],
                    "repetitions": table.data_repeat.nunique(),
                    "oracle_L1_mean": table.oracle_L1.mean(),
                    "oracle_L2_mean": table.oracle_L2.mean(),
                    "attainable_excess_mean": table.excess_pinball.mean(),
                    "eval_point_max_mean": table.eval_point_max.mean(),
                }
            )
        _save(pd.DataFrame(rows).sort_values("parameters"), output / "T2_summary.csv")

    consistency = data[data.group.eq("consistency")]
    if not consistency.empty:
        rows = []
        for (schedule, n_train), table in consistency.groupby(["schedule", "n_train"]):
            rows.append(
                {
                    "scenario": "S4",
                    "schedule": schedule,
                    "n_train": n_train,
                    "repetitions": table.data_repeat.nunique(),
                    "mean_excess": table.excess_pinball.mean(),
                    "median_excess": table.excess_pinball.median(),
                    "q90_excess": table.excess_pinball.quantile(0.9),
                    "mean_regret_G300": table.loc[table.target.eq(300), "operational_regret"].mean(),
                    "parameters": table.parameters.iloc[0],
                    "parameter_radius": table.parameter_radius.iloc[0],
                    "parameter_norm": table.parameter_norm.mean(),
                    "complexity_ratio": table.complexity_ratio.iloc[0],
                }
            )
        _save(pd.DataFrame(rows), output / "T3_summary.csv")


def _performance_tables(output, data):
    keys = ["experiment", "method", "structure_variant", "information_level", "tau", "target"]
    per_repeat = data.groupby([*keys, "data_repeat"], dropna=False)[["MeanCost", "SL_G", "n_test"]].mean().reset_index()
    rows, paired = [], []
    for key, table in per_repeat.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row.update(
            n_repeats=len(table),
            MeanCost_mean=table.MeanCost.mean(),
            MeanCost_median=table.MeanCost.median(),
            MeanCost_std=table.MeanCost.std(ddof=1),
            MeanCost_CI95=_ci95(table.MeanCost),
            SL_G_mean=table.SL_G.mean(),
            SL_G_median=table.SL_G.median(),
            SL_G_std=table.SL_G.std(ddof=1),
            SL_G_CI95=_ci95(table.SL_G),
            n_test=table.n_test.mean(),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    _save(summary, output / "performance_summary.csv")

    for experiment, reference in (("ablation", ("DAT", "full", "")), ("information", ("DAT", "full", "L4"))):
        block = per_repeat[per_repeat.experiment.eq(experiment)]
        if block.empty:
            continue
        ref = block[
            block.method.eq(reference[0])
            & block.structure_variant.eq(reference[1])
            & block.information_level.fillna("").eq(reference[2])
        ]
        ref = ref[["tau", "target", "data_repeat", "MeanCost", "SL_G"]].rename(
            columns={"MeanCost": "reference_MeanCost", "SL_G": "reference_SL_G"}
        )
        joined = block.merge(ref, on=["tau", "target", "data_repeat"], how="inner")
        joined["MeanCost_minus_reference"] = joined.MeanCost - joined.reference_MeanCost
        joined["SL_G_minus_reference"] = joined.SL_G - joined.reference_SL_G
        paired.append(joined)
    if paired:
        differences = pd.concat(paired, ignore_index=True)
        _save(differences, output / "paired_MeanCost_SL_differences.csv")
        rows = []
        for key, table in differences.groupby(keys, dropna=False):
            row = dict(zip(keys, key))
            row.update(
                n_repeats=len(table),
                mean_MeanCost_difference=table.MeanCost_minus_reference.mean(),
                MeanCost_difference_CI95=_ci95(table.MeanCost_minus_reference),
                mean_SL_G_difference=table.SL_G_minus_reference.mean(),
                SL_G_difference_CI95=_ci95(table.SL_G_minus_reference),
            )
            rows.append(row)
        _save(pd.DataFrame(rows), output / "paired_MeanCost_SL_summary.csv")
    return summary


def _setup_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _theory_figures(output, data, figures, plt):
    c0 = data[data.group.eq("estimation") & data.scenario.eq("C0") & data.estimator.eq("exact_bias_only")]
    if not c0.empty:
        table = c0[c0.target.eq(c0.target.iloc[0])]
        curve = table.groupby("n_train").excess_pinball.agg(
            mean="mean", median="median", q90=lambda x: x.quantile(0.9), q95=lambda x: x.quantile(0.95)
        )
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for column, color in (("mean", "tab:blue"), ("median", "tab:orange"), ("q90", "tab:green"), ("q95", "tab:red")):
            ax.plot(curve.index, curve[column], marker="o", color=color, label=column)
        for delta, style in (("0.1", "--"), ("0.05", ":")):
            bound = table.groupby("n_train")[f"bound_delta_{delta}"].first()
            ax.plot(bound.index, bound, style, color="black", label=f"bound δ={delta}")
        ax.set(
            xscale="log",
            yscale="symlog",
            ylim=(1e-6, 20),
            xlabel="Training days T",
            ylabel="Exact excess pinball risk / bound",
        )
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "T1_C0_estimation_bound.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for delta, color in (("0.1", "tab:blue"), ("0.05", "tab:red")):
            rates = table.groupby("n_train")[f"exceeds_bound_delta_{delta}"].mean()
            ax.plot(rates.index, rates, marker="o", color=color, label=f"δ={delta}")
            ax.axhline(float(delta), color=color, linestyle="--", linewidth=0.8)
        ax.set(xscale="log", xlabel="Training days T", ylabel="Empirical exceedance rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "T1_C0_bound_exceedance.png", dpi=170)
        plt.close(fig)

    dat = data[data.group.eq("estimation") & data.scenario.eq("S4")]
    if not dat.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        curve = dat.groupby("n_train").excess_pinball.agg(["mean", "std", "count"])
        ax.plot(curve.index, curve["mean"], marker="o")
        half = 1.96 * curve["std"] / np.sqrt(curve["count"])
        ax.fill_between(curve.index, curve["mean"] - half, curve["mean"] + half, alpha=0.15)
        ax.set(xscale="log", xlabel="Training days T", ylabel="Excess pinball risk")
        fig.tight_layout()
        fig.savefig(figures / "T1_DAT_total_excess_S4.png", dpi=170)
        plt.close(fig)

    approximation = data[data.group.eq("approximation")]
    if not approximation.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, (metric, label) in zip(
            axes,
            [
                ("oracle_L1", "Oracle quantile L1"),
                ("oracle_L2", "Oracle quantile L2"),
                ("excess_pinball", "Attainable excess risk"),
            ],
        ):
            curve = approximation.groupby("parameters")[metric].mean()
            ax.plot(curve.index, curve.values, marker="o")
            ax.set(xscale="log", xlabel="Parameters P", ylabel=label)
        fig.tight_layout()
        fig.savefig(figures / "T2_S4_oracle_diagnostics.png", dpi=170)
        plt.close(fig)

    consistency = data[data.group.eq("consistency")]
    if not consistency.empty:
        for metric, filename, label in (
            ("excess_pinball", "T3_S4_risk_convergence.png", "Excess pinball risk"),
            ("operational_regret", "T3_S4_operational_regret.png", "Operational regret G=300"),
        ):
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            table = consistency if metric == "excess_pinball" else consistency[consistency.target.eq(300)]
            for schedule, rows in table.groupby("schedule"):
                curve = rows.groupby("n_train")[metric].mean()
                ax.plot(curve.index, curve.values, marker="o", label=schedule)
            ax.set(xscale="log", xlabel="Training days T", ylabel=label)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures / filename, dpi=170)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for schedule, rows in consistency.groupby("schedule"):
            curve = rows.groupby("n_train").complexity_ratio.first()
            ax.plot(curve.index, curve.values, marker="o", label=schedule)
        ax.set(xscale="log", xlabel="Training days T", ylabel="P_T log(e+P_T)/T")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "T3_S4_complexity_proxy.png", dpi=170)
        plt.close(fig)


def _performance_figures(output, data, figures, plt):
    labels = {"comparison": "P1", "ablation": "P2", "information": "P3"}
    for experiment, prefix in labels.items():
        block = data[data.experiment.eq(experiment)]
        if block.empty:
            continue
        x_name = "method" if experiment == "comparison" else "structure_variant"
        if experiment == "information":
            x_name = "information_level"
        for metric, ylabel, suffix in (
            ("MeanCost", "Mean cost", "MeanCost"),
            ("SL_G", "Target service level SL_G", "SL"),
        ):
            targets = sorted(block.target.unique())
            fig, axes = plt.subplots(1, len(targets), figsize=(3.2 * len(targets), 3.8), squeeze=False)
            for ax, target in zip(axes[0], targets):
                panel = block[block.target.eq(target)]
                curve = panel.groupby(x_name)[metric].mean().sort_index()
                ax.bar(range(len(curve)), curve.values)
                ax.set_xticks(range(len(curve)), curve.index, rotation=90, fontsize=6)
                ax.set(title=f"G={target:g}", xlabel=x_name, ylabel=ylabel)
            fig.tight_layout()
            if experiment == "ablation":
                stem = f"{prefix}_S4_{suffix}_ablation"
            elif experiment == "information":
                stem = f"{prefix}_S4_{suffix}_information"
            else:
                stem = f"{prefix}_S4_{suffix}_by_method"
            fig.savefig(figures / f"{stem}.png", dpi=160)
            plt.close(fig)


def summarize(output, theory, performance):
    output = Path(output)
    if theory:
        data = pd.DataFrame(theory)
        data.to_csv(output / "theory_metrics.csv", index=False)
        _theory_tables(output, data)
    if performance:
        data = pd.DataFrame(performance)
        data.to_csv(output / "performance_metrics.csv", index=False)
        _performance_tables(output, data)
    try:
        plt = _setup_plotting()
    except ImportError:
        return {"figures": "matplotlib unavailable; CSV summaries complete"}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    if theory:
        _theory_figures(output, pd.DataFrame(theory), figures, plt)
    if performance:
        _performance_figures(output, pd.DataFrame(performance), figures, plt)
    return {"figures": "complete"}
