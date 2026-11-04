"""Aggregate independent data repetitions and build the protocol figures."""

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
    keys = ["group", "scenario", "estimator", "schedule", "n_train", "hidden", "parameters", "tau", "target"]
    summary = (
        data.groupby(keys, dropna=False)
        .agg(
            mean_excess=("excess_pinball", "mean"),
            median_excess=("excess_pinball", "median"),
            q90_excess=("excess_pinball", lambda x: x.quantile(0.9)),
            q95_excess=("excess_pinball", lambda x: x.quantile(0.95)),
            repetitions=("data_repeat", "nunique"),
            mean_regret=("operational_regret", "mean"),
            mean_oracle_L1=("oracle_L1", "mean"),
            mean_oracle_L2=("oracle_L2", "mean"),
            mean_point_max=("eval_point_max", "mean"),
        )
        .reset_index()
    )
    _save(summary, output / "theory_summary.csv")

    c0 = data[data.group.eq("estimation") & data.scenario.eq("C0") & data.estimator.eq("exact_bias_only")].copy()
    if not c0.empty:
        rows = []
        for n_train, table in c0.groupby("n_train"):
            for delta in ("0.1", "0.05"):
                column = f"exceeds_bound_delta_{delta}"
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
                        "exceedance_rate": table[column].mean(),
                        "bound_is_trivial": bool(table.bound_is_trivial.iloc[0]),
                    }
                )
        _save(pd.DataFrame(rows), output / "T1_C0_summary.csv")
        # Compatibility file used by earlier smoke runs.
        tail_columns = [c for c in c0.columns if c.startswith("exceeds_bound_delta_")]
        c0.groupby(["n_train", "estimator", "target"])[tail_columns].mean().reset_index().to_csv(
            output / "exact_erm_tail_frequencies.csv", index=False
        )

    dat = data[data.group.eq("estimation") & data.scenario.ne("C0")]
    if not dat.empty:
        rows = []
        for (scenario, n_train), table in dat.groupby(["scenario", "n_train"]):
            rows.append(
                {
                    "scenario": scenario,
                    "n_train": n_train,
                    "repetitions": table.data_repeat.nunique(),
                    "mean_excess": table.excess_pinball.mean(),
                    "median_excess": table.excess_pinball.median(),
                    "q90_excess": table.excess_pinball.quantile(0.9),
                    "q95_excess": table.excess_pinball.quantile(0.95),
                    "mean_oracle_L1": table.oracle_L1.mean(),
                    "mean_regret": table.operational_regret.mean(),
                }
            )
        _save(pd.DataFrame(rows), output / "T1_DAT_summary.csv")

    approximation = data[data.group.eq("approximation")]
    if not approximation.empty:
        rows = []
        for (scenario, parameters, hidden), table in approximation.groupby(["scenario", "parameters", "hidden"]):
            rows.append(
                {
                    "scenario": scenario,
                    "parameters": parameters,
                    "hidden": hidden,
                    "repetitions": table.data_repeat.nunique(),
                    "oracle_L1_mean": table.oracle_L1.mean(),
                    "oracle_L1_median": table.oracle_L1.median(),
                    "oracle_L2_mean": table.oracle_L2.mean(),
                    "attainable_excess_mean": table.excess_pinball.mean(),
                    "eval_point_max_mean": table.eval_point_max.mean(),
                }
            )
        _save(pd.DataFrame(rows), output / "T2_summary.csv")

    consistency = data[data.group.eq("consistency")]
    if not consistency.empty:
        rows = []
        for (scenario, schedule, n_train), table in consistency.groupby(["scenario", "schedule", "n_train"]):
            rows.append(
                {
                    "scenario": scenario,
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
    return summary


def _performance_tables(output, data):
    keys = ["group", "scenario", "method", "variant", "tau", "target"]
    per_repeat = data.groupby([*keys, "data_repeat"], dropna=False)[["TC", "SL_G", "n_test"]].mean().reset_index()
    rows, paired = [], []
    for key, table in per_repeat.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row.update(
            n_repeats=len(table),
            TC_mean=table.TC.mean(),
            TC_median=table.TC.median(),
            TC_std=table.TC.std(ddof=1),
            TC_CI95=_ci95(table.TC),
            SL_G_mean=table.SL_G.mean(),
            SL_G_median=table.SL_G.median(),
            SL_G_std=table.SL_G.std(ddof=1),
            SL_G_CI95=_ci95(table.SL_G),
            n_test=table.n_test.mean(),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    _save(summary, output / "performance_summary.csv")

    for group, reference in (("ablation", ("DAT", "full")), ("information", ("DAT", "L4"))):
        block = per_repeat[per_repeat.group.eq(group)]
        if block.empty:
            continue
        ref = block[(block.method.eq(reference[0])) & (block.variant.eq(reference[1]))]
        ref = ref[["scenario", "tau", "target", "data_repeat", "TC", "SL_G"]].rename(
            columns={"TC": "reference_TC", "SL_G": "reference_SL_G"}
        )
        joined = block.merge(ref, on=["scenario", "tau", "target", "data_repeat"], how="inner")
        joined["TC_minus_reference"] = joined.TC - joined.reference_TC
        joined["SL_G_minus_reference"] = joined.SL_G - joined.reference_SL_G
        paired.append(joined)
    if paired:
        differences = pd.concat(paired, ignore_index=True)
        _save(differences, output / "paired_TC_SL_differences.csv")
        rows = []
        for key, table in differences.groupby(keys, dropna=False):
            row = dict(zip(keys, key))
            row.update(
                n_repeats=len(table),
                mean_TC_difference=table.TC_minus_reference.mean(),
                TC_difference_CI95=_ci95(table.TC_minus_reference),
                mean_SL_G_difference=table.SL_G_minus_reference.mean(),
                SL_G_difference_CI95=_ci95(table.SL_G_minus_reference),
            )
            rows.append(row)
        _save(pd.DataFrame(rows), output / "paired_TC_SL_summary.csv")
    return summary


def _setup_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _theory_figures(output, data, figures, plt):
    c0 = data[data.group.eq("estimation") & data.scenario.eq("C0") & data.estimator.eq("exact_bias_only")]
    if not c0.empty:
        target = sorted(c0.target.unique())[0]
        table = c0[c0.target.eq(target)]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        curve = (
            table.groupby("n_train")
            .excess_pinball.agg(
                mean="mean", median="median", q90=lambda x: x.quantile(0.9), q95=lambda x: x.quantile(0.95)
            )
            .sort_index()
        )
        for column, color in (("mean", "tab:blue"), ("median", "tab:orange"), ("q90", "tab:green"), ("q95", "tab:red")):
            ax.plot(curve.index, curve[column].values, marker="o", color=color, label=f"exact ERM {column}")
        for delta, style in (("0.1", "--"), ("0.05", ":")):
            bound = table.groupby("n_train")[f"bound_delta_{delta}"].first().sort_index()
            ax.plot(bound.index, bound.values, style, color="black", label=f"theoretical bound δ={delta}")
        ax.set(
            xscale="log",
            yscale="symlog",
            ylim=(1e-6, 20),
            xlabel="Training days T",
            ylabel="Exact excess pinball risk / theoretical bound",
        )
        ax.set_title("T1-C0: exact ERM estimation error and high-probability bounds")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "T1_C0_error_bound.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for delta, color in (("0.1", "tab:blue"), ("0.05", "tab:red")):
            rates = table.groupby("n_train")[f"exceeds_bound_delta_{delta}"].mean().sort_index()
            ax.plot(rates.index, rates.values, marker="o", color=color, label=f"δ={delta}")
            ax.axhline(float(delta), color=color, linewidth=0.8, linestyle="--")
        ax.set(xscale="log", xlabel="Training days T", ylabel="Empirical exceedance rate")
        ax.set_title("T1-C0: bound exceedance versus nominal δ")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "T1_C0_exceedance.png", dpi=170)
        plt.close(fig)

    dat = data[data.group.eq("estimation") & data.scenario.ne("C0")]
    if not dat.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for scenario, table in dat.groupby("scenario"):
            curve = table.groupby("n_train").excess_pinball.agg(["mean", "median", "std", "count"]).sort_index()
            ax.plot(curve.index, curve["mean"], marker="o", label=f"{scenario} mean")
            half = 1.96 * curve["std"] / np.sqrt(curve["count"])
            ax.fill_between(curve.index, curve["mean"] - half, curve["mean"] + half, alpha=0.12)
        ax.set(xscale="log", xlabel="Training days T", ylabel="Total excess pinball risk vs oracle")
        ax.set_title("T1-DAT: fixed-capacity learning curves")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "T1_DAT_total_excess.png", dpi=170)
        plt.close(fig)

    approximation = data[data.group.eq("approximation")]
    if not approximation.empty:
        metrics = [
            ("oracle_L1", "Oracle quantile L1"),
            ("oracle_L2", "Oracle quantile L2"),
            ("excess_pinball", "Attainable excess pinball risk"),
            ("eval_point_max", "Evaluation-point maximum error"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
        for ax, (metric, label) in zip(axes.ravel(), metrics):
            for scenario, table in approximation.groupby("scenario"):
                curve = table.groupby("parameters")[metric].mean().sort_index()
                ax.plot(curve.index, curve.values, marker="o", label=scenario)
            ax.set(xscale="log", xlabel="Actual parameter count P", ylabel=label)
            ax.legend(fontsize=7)
        axes[0, 0].set_title("T2: constructive approximation diagnostics")
        fig.tight_layout()
        fig.savefig(figures / "T2_oracle_diagnostics.png", dpi=170)
        plt.close(fig)

    consistency = data[data.group.eq("consistency")]
    if not consistency.empty:
        for metric, filename, title in (
            ("excess_pinball", "T3_risk_convergence.png", "T3: total excess pinball risk"),
            ("operational_regret", "T3_operational_regret.png", "T3: operational regret at G=300"),
        ):
            label = "Excess pinball risk" if metric == "excess_pinball" else "Operational regret"
            fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), squeeze=False)
            for ax, scenario in zip(axes[0], sorted(consistency.scenario.unique())):
                table = consistency[consistency.scenario.eq(scenario)]
                if metric == "operational_regret":
                    table = table[table.target.eq(300)]
                for schedule, rows in table.groupby("schedule"):
                    curve = rows.groupby("n_train")[metric].mean().sort_index()
                    ax.plot(curve.index, curve.values, marker="o", label=schedule)
                ax.set(xscale="log", xlabel="Training days T", ylabel=label)
                ax.set_title(str(scenario))
                ax.legend(fontsize=7)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(figures / filename, dpi=170)
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
        for (scenario, schedule), table in consistency.groupby(["scenario", "schedule"]):
            curve = table.groupby("n_train").complexity_ratio.first().sort_index()
            axes[0].plot(curve.index, curve.values, marker="o", label=f"{scenario}/{schedule}")
            pcurve = table.groupby("n_train").parameters.first().sort_index()
            axes[1].plot(pcurve.index, pcurve.values, marker="o", label=f"{scenario}/{schedule}")
        for ax, label in zip(axes, ("P_T log(e+P_T)/T", "Actual parameter count P_T")):
            ax.set(xscale="log", xlabel="Training days T", ylabel=label)
            ax.legend(fontsize=6)
        axes[0].set_title("T3: complexity proxy")
        fig.tight_layout()
        fig.savefig(figures / "T3_complexity_proxy.png", dpi=170)
        plt.close(fig)


def _performance_figures(output, data, figures, plt):
    comparison = data[data.group.eq("comparison")]
    if not comparison.empty:
        for scenario, table in comparison.groupby("scenario"):
            methods = sorted(table.method.unique())
            fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=False)
            for ax, tau in zip(axes, sorted(table.tau.unique())):
                panel = table[table.tau.eq(tau)]
                for method in methods:
                    curve = panel[panel.method.eq(method)].groupby("target").TC.mean().sort_index()
                    ax.plot(curve.index, curve.values, marker="o", label=method)
                ax.set(title=f"τ={tau:g}", xlabel="Target G", ylabel="Total cost TC")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5, fontsize=7)
            fig.suptitle(f"P1 total cost, {scenario}")
            fig.tight_layout(rect=(0, 0, 1, 0.86))
            fig.savefig(figures / f"P1_TC_by_method_{scenario}.png", dpi=160)
            plt.close(fig)

            fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=False)
            for ax, tau in zip(axes, sorted(table.tau.unique())):
                panel = table[table.tau.eq(tau)]
                for method in methods:
                    curve = panel[panel.method.eq(method)].groupby("target").SL_G.mean().sort_index()
                    ax.plot(curve.index, curve.values, marker="o", label=method)
                ax.set(title=f"τ={tau:g}", xlabel="Target G", ylabel="Target service level SL_G")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5, fontsize=7)
            fig.suptitle(f"P1 target service level, {scenario}")
            fig.tight_layout(rect=(0, 0, 1, 0.86))
            fig.savefig(figures / f"P1_SL_by_method_{scenario}.png", dpi=160)
            plt.close(fig)

            # The absolute TC/SL_G lines overlap when between-method differences are
            # small; fixed (tau, G) panels make those differences visible.
            taus, targets = sorted(table.tau.unique()), sorted(table.target.unique())
            for metric, label, filename in (
                ("TC", "Total cost TC", "P1_TC_method_panels"),
                ("SL_G", "Target service level SL_G", "P1_SL_method_panels"),
            ):
                fig, axes = plt.subplots(len(taus), len(targets), figsize=(18, 12), sharex=True, squeeze=False)
                for row, tau in enumerate(taus):
                    for column, target in enumerate(targets):
                        ax = axes[row, column]
                        panel = table[table.tau.eq(tau) & table.target.eq(target)]
                        values = panel.groupby("method")[metric].mean().reindex(methods)
                        ax.plot(range(len(methods)), values.values, marker="o", linestyle="none", color="tab:blue")
                        span = max(float(values.max() - values.min()), abs(float(values.mean())) * 1e-4, 1e-8)
                        ax.set_ylim(float(values.min()) - 0.25 * span, float(values.max()) + 0.25 * span)
                        ax.grid(axis="y", alpha=0.3)
                        ax.set_xticks(range(len(methods)), methods, rotation=90, fontsize=6)
                        ax.set_title(f"τ={tau:g}, G={target:g}", fontsize=8)
                        if row == len(taus) - 1:
                            ax.set_xlabel("Method")
                        if column == 0:
                            ax.set_ylabel(label)
                fig.suptitle(f"P1 {label} by (τ, G), {scenario}")
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                fig.savefig(figures / f"{filename}_{scenario}.png", dpi=150)
                plt.close(fig)

        # τ×G heatmaps for the two reference methods.
        for method in [m for m in ("DAT", "SAA") if m in comparison.method.unique()]:
            for metric, filename in (("TC", "P1_TC_tau_G"), ("SL_G", "P1_SL_tau_G")):
                for scenario, table in comparison[comparison.method.eq(method)].groupby("scenario"):
                    pivot = table.pivot_table(index="tau", columns="target", values=metric, aggfunc="mean")
                    fig, ax = plt.subplots(figsize=(5.4, 3.4))
                    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
                    ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
                    ax.set_yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
                    ax.set(xlabel="Target G", ylabel="τ", title=f"{method} {metric} ({scenario})")
                    fig.colorbar(image, ax=ax, label=metric)
                    fig.tight_layout()
                    fig.savefig(figures / f"{filename}_{scenario}_{method}.png", dpi=160)
                    plt.close(fig)

    for group, filename, title in (
        ("ablation", "P2", "P2 structural ablation"),
        ("information", "P3", "P3 information ladder"),
    ):
        block = data[data.group.eq(group)]
        if block.empty:
            continue
        x_name = "variant" if group == "ablation" else "variant"
        for metric, label in (("TC", "Total cost TC"), ("SL_G", "Target service level SL_G")):
            for scenario, table in block.groupby("scenario"):
                taus = sorted(table.tau.unique())
                fig, axes = plt.subplots(1, len(taus), figsize=(3.5 * len(taus), 3.6), squeeze=False)
                for ax, tau in zip(axes[0], taus):
                    panel = table[table.tau.eq(tau)]
                    for target, rows in panel.groupby("target"):
                        curve = rows.groupby(x_name)[metric].mean()
                        order = sorted(curve.index)
                        ax.plot(order, curve.loc[order].values, marker="o", label=f"G={target:g}")
                    ax.set(xlabel=x_name, ylabel=label, title=f"τ={tau:g}")
                    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5, fontsize=7)
                fig.suptitle(f"{title}: {metric}, {scenario}")
                fig.tight_layout(rect=(0, 0, 1, 0.84))
                name = f"{filename}_{metric}_{'ablation' if group == 'ablation' else 'information'}_{scenario}.png"
                fig.savefig(figures / name, dpi=160)
                plt.close(fig)

        # Paired differences relative to the full DAT / L4 reference. These are
        # still TC and SL_G values (not a new metric) but remove the large
        # between-G level so component effects are visible.
        reference = "full" if group == "ablation" else "L4"
        keys = ["scenario", "tau", "target", "data_repeat", "variant"]
        for metric, label in (("TC", "Total cost TC"), ("SL_G", "Target service level SL_G")):
            per_repeat = block.groupby(keys, dropna=False)[metric].mean().reset_index()
            anchor = per_repeat[per_repeat.variant.eq(reference)][
                ["scenario", "tau", "target", "data_repeat", metric]
            ].rename(columns={metric: "reference"})
            joined = per_repeat.merge(anchor, on=["scenario", "tau", "target", "data_repeat"], how="inner")
            joined["difference"] = joined[metric] - joined.reference
            for scenario, table in joined.groupby("scenario"):
                taus = sorted(table.tau.unique())
                fig, axes = plt.subplots(1, len(taus), figsize=(3.5 * len(taus), 3.6), squeeze=False)
                for ax, tau in zip(axes[0], taus):
                    panel = table[table.tau.eq(tau)]
                    for target, rows in panel.groupby("target"):
                        curve = rows.groupby("variant").difference.mean()
                        order = sorted(curve.index)
                        ax.plot(order, curve.loc[order].values, marker="o", label=f"G={target:g}")
                    ax.axhline(0.0, color="black", linewidth=0.7)
                    ax.set(
                        xlabel="variant",
                        ylabel=f"{label} − reference ({reference})",
                        title=f"τ={tau:g}",
                    )
                    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5, fontsize=7)
                fig.suptitle(f"{title}: {label} difference vs {reference}, {scenario}")
                fig.tight_layout(rect=(0, 0, 1, 0.84))
                name = f"{filename}_{metric}_difference_{scenario}.png"
                fig.savefig(figures / name, dpi=160)
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
