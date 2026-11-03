"""Paired comparison, structural intervention and information-value experiments."""

from copy import deepcopy
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from dat_study.config import load_config
from dat_study.runner import run
from dat_study.data import FoldData
from dat_study.io import read_json
from .generator import generate
from .jobs import run_job
from .oracle import cached_laws, assess_laws


def search_space(method, smoke, settings):
    if smoke or method not in {"DAT", "DeepQR", "Flat-ETO", "Set-ETO"}:
        return [{}]
    search = settings["set_eto_search" if method == "Set-ETO" else "neural_search"]
    return [
        {"learning_rate": lr, "batch_size": batch}
        for lr, batch in product(search["learning_rates"], search["batch_sizes"])
    ]


def experiment_tasks(group, settings):
    if group == "comparison":
        for method in settings["comparison"]["methods"]:
            for variant in ["basic", "enhanced"] if method in {"LQR", "GBQR"} else ["full"]:
                yield method, variant
    elif group == "ablation":
        for variant in settings["ablation"]["variants"]:
            yield "DAT", variant
        for variant in settings["ablation"].get("equal_capacity_diagnostic", []):
            yield "DAT", variant + "_matched"
    else:
        for level in settings["information"]["levels"]:
            yield "DAT", level


def run_performance(args, protocol, generator, root, output, groups, code_hash):
    settings = protocol["performance_experiments"]
    repeats = args.repetitions or (1 if args.smoke else settings["independent_data_repetitions"])
    models = [0] if args.smoke else settings["model_seeds"]
    all_rows = []
    for group in groups:
        scenarios = args.scenarios or (["S2"] if args.smoke else settings[group]["scenarios"])
        for scenario in scenarios:
            for repeat in range(repeats):
                profile = protocol["profiles"]["performance"]
                g = {
                    **generator,
                    "scenario": scenario,
                    "data_seed": args.data_seed + repeat,
                    "max_visible_orders": profile["max_orders"],
                    "n_train": settings["n_train"],
                    "n_validation": settings["n_validation"],
                    "n_test": settings["n_test"],
                }
                if args.smoke:
                    g.update(n_train=48, n_validation=12, n_test=12)
                if args.eval_days:
                    g["n_test"] = args.eval_days
                data_root = output / "datasets" / f"{scenario}_r{repeat}"
                if data_root.exists():
                    existing = yaml.safe_load((data_root / "generation.yaml").read_text(encoding="utf-8"))
                    if existing != g:
                        raise ValueError("Existing synthetic samples have a different recipe")
                else:
                    generate(g, data_root)
                for method, variant in experiment_tasks(group, settings):
                    if args.methods and method not in args.methods:
                        continue
                    for seed in models[:1] if method in {"SAA", "Rolling-SAA", "LQR", "GBQR"} else models:
                        name = f"P_{group}_{scenario}_r{repeat}_{method}_{variant}_s{seed}"
                        signature = {
                            "protocol": protocol,
                            "generator": g,
                            "method": method,
                            "variant": variant,
                            "seed": seed,
                            "smoke": args.smoke,
                            "code": code_hash,
                            "device": args.device,
                            "threads": args.threads,
                        }

                        def action(folder):
                            base = load_config(root / "configs/synthetic.yaml")
                            base.update(
                                seed=seed,
                                data_root=str(data_root),
                                output_root=str(folder),
                                taus=settings["taus"],
                                targets=settings["targets"],
                                selection_metric="pinball",
                            )
                            base["training"] = {"device": args.device, "threads": args.threads}
                            if method in {"DAT", "DeepQR", "Flat-ETO", "Set-ETO"}:
                                base["training"].update(
                                    epochs=settings["neural_search"]["max_epochs"],
                                    patience=settings["neural_search"]["patience"],
                                )
                            override = base.setdefault("model_overrides", {}).setdefault(method, {})
                            if method == "DAT":
                                override["model"] = {
                                    "d_model": profile["d_model"],
                                    "patient_encoder": {"hidden_dims": [profile["patient_hidden"]]},
                                    "self_attention": {"num_heads": protocol["dat"]["self_attention_heads"]},
                                    "cross_attention": {
                                        "num_heads": protocol["dat"]["cross_attention_heads"],
                                        "ff_dim": profile["day_ff"],
                                    },
                                    "head": {"output_upper_bound": profile["demand_upper"]},
                                }
                                override["constraints"] = {
                                    key: profile["radius"]
                                    for key in (
                                        "global_parameter_radius",
                                        "operator_norm_bound",
                                        "bias_bound",
                                        "layer_norm_gain_bound",
                                        "layer_norm_shift_bound",
                                    )
                                }
                            elif method == "DeepQR":
                                override.setdefault("deep_qr", {})["output_upper_bound"] = profile["demand_upper"]
                            elif method == "Set-ETO":
                                override.setdefault("set_eto", {})["distributions"] = settings["comparison"][
                                    "set_eto_primary_families"
                                ]
                            if group == "ablation":
                                override["ablation"] = variant.removesuffix("_matched")
                                if variant.endswith("_matched"):
                                    d = profile["d_model"]
                                    inactive = 2 * (d * d + d) + 2 * d
                                    override.setdefault("model", {}).setdefault("cross_attention", {})["ff_dim"] = (
                                        profile["day_ff"] + round(inactive / (2 * d + 1))
                                    )
                            elif group == "information":
                                override["information"] = variant
                            elif method in {"LQR", "GBQR"}:
                                override.setdefault(method.lower(), {})["input_set"] = variant
                            candidates = search_space(method, args.smoke, settings)
                            trials, best = [], None
                            for index, parameters in enumerate(candidates):
                                cfg = deepcopy(base)
                                cfg["training"].update(parameters)
                                dev = run(cfg, "validation", [method], f"candidate_{index:02d}", args.smoke)
                                frame = pd.read_csv(dev / "comparison.csv")
                                score = float(frame.groupby("tau").pinball.first().mean())
                                trials.append({"candidate": index, "training": parameters, "validation_pinball": score})
                                if best is None or score < best[0]:
                                    best = (score, cfg, dev)
                            _, selected, recipe_root = best
                            test = run(selected, "test", [method], "test", args.smoke, recipe_root)
                            evaluation = FoldData(data_root / "test")
                            if args.smoke:
                                evaluation.limit_for_smoke()
                            daily = evaluation.daily_split("eval")
                            predictions = (
                                pd.read_csv(test / method / "daily_quantiles.csv").filter(regex="^Q_").to_numpy()
                            )
                            law = cached_laws(
                                output / "oracle_cache",
                                {"data": evaluation.identity, "code": code_hash},
                                daily,
                                evaluation.order_split("eval"),
                                g,
                            )
                            assessed = assess_laws(law, predictions, settings["taus"], settings["targets"])
                            rows = pd.read_csv(test / "comparison.csv").replace({np.nan: None}).to_dict("records")
                            for row in rows:
                                ti = settings["taus"].index(row["tau"])
                                gi = settings["targets"].index(row["target"])
                                row.update(
                                    group=group,
                                    scenario=scenario,
                                    data_repeat=repeat,
                                    model_seed=seed,
                                    variant=variant,
                                    fit_days=read_json(test / "data_contract.json")["train_days"],
                                    selection_metric="validation_pinball",
                                    operational_regret=float(assessed["operational_regret"][:, ti, gi].mean()),
                                    excess_pinball=float(assessed["excess_pinball"][:, ti].mean()),
                                )
                            return {
                                "rows": rows,
                                "validation_trials": trials,
                                "selected_recipe": str(Path(recipe_root).relative_to(folder)),
                            }

                        result = run_job(output / "jobs", name, signature, action, args.resume)
                        all_rows.extend(result["rows"])
                        pd.DataFrame(all_rows).to_csv(output / "performance_metrics.csv", index=False)
    return all_rows
