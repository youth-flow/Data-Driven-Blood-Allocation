"""AXID-DC v7 fixed-configuration performance experiments P1-P3."""

import numpy as np
import pandas as pd
import yaml

from dat_study.config import load_config
from dat_study.data import FoldData
from dat_study.evaluation import evaluate
from dat_study.training import fit_predict
from .generator_v7 import generate_performance
from .jobs import run_job


def _specs(group, settings):
    if group == "comparison":
        for method in settings["comparison"]["methods"]:
            yield method, "full", None
    elif group == "ablation":
        for variant in settings["ablation"]["variants"]:
            yield "DAT", variant, None
    elif group == "information":
        for level in settings["information"]["levels"]:
            yield "DAT", "full", level
    else:
        raise ValueError(f"Unknown performance group: {group}")


def _data_recipe(protocol, generator, args, repeat):
    settings = protocol["performance_experiments"]
    return {
        **generator,
        "scenario": "AXID-DC v7",
        "data_seed": args.data_seed + repeat,
        "max_visible_orders": protocol["profiles"]["performance"]["max_orders"],
        "n_train": 32 if args.smoke else settings["n_train"],
        "n_test": 16 if args.smoke else settings["n_test"],
    }


def _config(root, output, data_root, protocol, args, seed):
    settings = protocol["performance_experiments"]
    config = load_config(root / "configs/synthetic.yaml")
    config.update(
        seed=int(seed),
        taus=[settings["tau"]],
        targets=list(settings["targets"]),
        data_root=str(data_root),
        output_root=str(output),
    )
    config["training"] = dict(config.get("training", {}))
    config["training"].update(device=args.device, threads=args.threads)
    if args.smoke:
        config["training"].update(epochs=1, batch_size=8)
    return config


def _matched_override(config, protocol, variant):
    profile = protocol["profiles"]["performance"]
    width = int(profile["d_model"])
    day_ff = int(profile["day_ff"])
    if variant == "no_self_attention_param_matched":
        inactive = 4 * width * width + 6 * width
        ablation = "no_self_attention"
    else:
        inactive = 2 * (width * width + width) + 2 * width
        ablation = variant.removesuffix("_param_matched")
    extra = round(inactive / (2 * width + 1))
    config.setdefault("model_overrides", {}).setdefault("DAT", {}).setdefault("model", {}).setdefault(
        "cross_attention", {}
    )["ff_dim"] = day_ff + extra
    config["ablation"] = ablation


def run_performance(args, protocol, generator, root, output, groups, code_hash):
    settings = protocol["performance_experiments"]
    repeats = args.repetitions or (1 if args.smoke else settings["independent_data_repetitions"])
    seeds = [0] if args.smoke else (args.model_seeds or settings["model_seeds"])
    all_rows = []
    for repeat in range(args.repeat_start, args.repeat_start + repeats):
        recipe = _data_recipe(protocol, generator, args, repeat)
        data_root = output / "datasets" / f"AXIDv7_r{repeat}"
        if data_root.exists():
            existing = yaml.safe_load((data_root / "generation.yaml").read_text(encoding="utf-8"))
            if existing != recipe:
                raise ValueError("Existing AXID-DC v7 samples have a different recipe")
        else:
            generate_performance(recipe, data_root)
        data = FoldData(data_root)
        for group in groups:
            for method, variant, level in _specs(group, settings):
                if args.methods and method not in args.methods:
                    continue
                run_seeds = seeds if method in settings["stochastic_methods"] else [0]
                for seed in run_seeds:
                    name = f"{group}_AXIDv7_r{repeat}_{method}_{variant}_{level or 'full'}_s{seed}"
                    signature = {
                        "protocol": protocol,
                        "generator": recipe,
                        "method": method,
                        "variant": variant,
                        "information": level,
                        "seed": seed,
                        "smoke": args.smoke,
                        "code": code_hash,
                    }

                    def action(folder, method=method, variant=variant, level=level, seed=seed, group=group):
                        config = _config(root, folder, data_root, protocol, args, seed)
                        if group == "ablation":
                            if variant.endswith("_param_matched"):
                                _matched_override(config, protocol, variant)
                            else:
                                config["ablation"] = variant
                        if level is not None:
                            config["information"] = level
                        prediction, result, _ = fit_predict(method, data, config)
                        prediction = np.clip(prediction, 0.0, float(data.schema["demand_support_upper"]))
                        daily = data.daily_split("eval")
                        rows = evaluate(
                            prediction,
                            daily.D.to_numpy(),
                            daily.It.to_numpy(),
                            data.daily_split("train").D.to_numpy(),
                            config["taus"],
                            config["targets"],
                        )
                        for row in rows:
                            row.update(
                                experiment=group,
                                scenario="AXID-DC v7",
                                method=method,
                                structure_variant=variant,
                                information_level=level or "",
                                data_repeat=repeat,
                                train_seed=seed,
                                selection_metric="fixed_configuration",
                                epochs=result.get("epochs", 0),
                            )
                        return {"rows": rows}

                    result = run_job(output / "jobs", name, signature, action, args.resume)
                    all_rows.extend(result["rows"])
                    pd.DataFrame(all_rows).to_csv(output / "performance_metrics.csv", index=False)
    return all_rows
