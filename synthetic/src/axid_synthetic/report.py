"""Aggregate independent data repetitions, not initialization seeds; make local figures."""

import numpy as np
import pandas as pd
from scipy.stats import t


def summarize(output, theory, performance):
    if performance:
        data = pd.DataFrame(performance)
        keys = ["group", "scenario", "method", "variant", "tau", "target"]
        independent = data.groupby([*keys, "data_repeat"], dropna=False)[["P", "mean_cost"]].mean().reset_index()
        summary = independent.groupby(keys, dropna=False).P.agg(["mean", "std", "count"]).reset_index()
        summary["CI95_half_width"] = t.ppf(0.975, summary["count"] - 1) * summary["std"] / np.sqrt(summary["count"])
        summary.to_csv(output / "performance_summary.csv", index=False)
        paired = []
        for group, rows in independent.groupby("group"):
            reference = rows[rows.method.eq("DAT") & rows.variant.eq("full" if group != "information" else "L4")]
            ref = reference[["scenario", "tau", "target", "data_repeat", "P"]].rename(columns={"P": "reference_P"})
            joined = rows.merge(ref, on=["scenario", "tau", "target", "data_repeat"], how="inner")
            joined["P_minus_full_DAT"] = joined.P - joined.reference_P
            paired.append(joined)
        if paired:
            differences = pd.concat(paired, ignore_index=True)
            differences.to_csv(output / "paired_P_differences.csv", index=False)
            paired_summary = differences.groupby(keys).P_minus_full_DAT.agg(["mean", "std", "count"]).reset_index()
            paired_summary["CI95_half_width"] = (
                t.ppf(0.975, paired_summary["count"] - 1) * paired_summary["std"] / np.sqrt(paired_summary["count"])
            )
            paired_summary.to_csv(output / "paired_P_summary.csv", index=False)
    if theory:
        data = pd.DataFrame(theory)
        keys = ["group", "scenario", "estimator", "schedule", "n_train", "hidden", "parameters", "tau", "target"]
        summary = (
            data.groupby(keys)
            .agg(
                mean_excess=("excess_pinball", "mean"),
                median_excess=("excess_pinball", "median"),
                q90_excess=("excess_pinball", lambda x: x.quantile(0.9)),
                q95_excess=("excess_pinball", lambda x: x.quantile(0.95)),
                repetitions=("data_repeat", "nunique"),
                mean_regret=("operational_regret", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(output / "theory_summary.csv", index=False)
        calibration = data[data.group.eq("estimation") & data.scenario.eq("C0") & data.exact_ERM_certified]
        tail_columns = [c for c in data.columns if c.startswith("exceeds_bound_delta_")]
        if not calibration.empty:
            calibration.groupby(["n_train", "estimator", "target"])[tail_columns].mean().reset_index().to_csv(
                output / "exact_erm_tail_frequencies.csv", index=False
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"figures": "matplotlib unavailable; CSV summaries complete"}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    if performance:
        data = pd.read_csv(output / "performance_summary.csv")
        for (group, scenario), table in data.groupby(["group", "scenario"]):
            targets = sorted(table.target.unique())
            fig, axes = plt.subplots(1, len(targets), figsize=(3.5 * len(targets), 3.2), squeeze=False)
            for ax, target in zip(axes[0], targets):
                for (method, variant), curve in table[table.target.eq(target)].groupby(["method", "variant"]):
                    curve = curve.sort_values("tau")
                    ax.plot(curve.tau, curve["mean"], marker="o", label=f"{method}/{variant}")
                    ax.fill_between(
                        curve.tau,
                        curve["mean"] - curve.CI95_half_width,
                        curve["mean"] + curve.CI95_half_width,
                        alpha=0.08,
                    )
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.set(title=f"Target={target:g}", xlabel="tau", ylabel="P vs SAA")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7)
            fig.tight_layout(rect=(0, 0, 1, 0.80))
            fig.savefig(figures / f"{group}_{scenario}.png", dpi=160)
            plt.close(fig)
    if theory:
        data = pd.DataFrame(theory)
        data = data[data.target.eq(data.target.iloc[0])]
        for group, table in data.groupby("group"):
            fig, ax = plt.subplots(figsize=(6, 4))
            x = "parameters" if group == "approximation" else "n_train"
            for (scenario, schedule, estimator), rows in table.groupby(["scenario", "schedule", "estimator"]):
                curve = rows.groupby(x).excess_pinball.mean().sort_index()
                ax.plot(curve.index, curve, marker="o", label=f"{scenario}/{schedule}/{estimator}")
            ax.set(xscale="log", xlabel=x, ylabel="Excess pinball risk", title=group)
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(figures / f"theory_{group}.png", dpi=160)
            plt.close(fig)
        estimation = data[data.group.eq("estimation")]
        if not estimation.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for (scenario, estimator), rows in estimation.groupby(["scenario", "estimator"]):
                tail = rows.groupby("n_train").excess_pinball.quantile(0.95).sort_index()
                ax.plot(tail.index, tail, marker="o", label=f"{scenario}/{estimator}: q95")
                if estimator == "exact_bias_only":
                    bound = rows.groupby("n_train").bound_capped.first().sort_index()
                    ax.plot(bound.index, bound, linestyle="--", label="C0 exact subclass: capped bound")
            ax.set(
                xscale="log",
                xlabel="Training days",
                ylabel="Excess pinball risk",
                title="Repeated-sample tail calibration",
            )
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(figures / "theory_tail_calibration.png", dpi=160)
            plt.close(fig)
        approximation = data[data.group.eq("approximation")]
        if not approximation.empty:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
            for ax, metric in zip(axes, ["oracle_L1", "oracle_L2", "sampled_max_error"]):
                for scenario, rows in approximation.groupby("scenario"):
                    curve = rows.groupby("parameters")[metric].mean().sort_index()
                    ax.plot(curve.index, curve, marker="o", label=scenario)
                ax.set(xscale="log", xlabel="Parameters", ylabel=metric)
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(figures / "theory_oracle_approximation.png", dpi=160)
            plt.close(fig)
        consistency = data[data.group.eq("consistency")]
        if not consistency.empty:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            for ax, metric in zip(axes, ["operational_regret", "complexity_ratio"]):
                for (scenario, schedule), rows in consistency.groupby(["scenario", "schedule"]):
                    curve = rows.groupby("n_train")[metric].mean().sort_index()
                    ax.plot(curve.index, curve, marker="o", label=f"{scenario}/{schedule}")
                ax.set(xscale="log", xlabel="Training days", ylabel=metric)
                ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(figures / "theory_consistency_diagnostics.png", dpi=160)
            plt.close(fig)
    return {"figures": "complete"}
