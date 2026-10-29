"""Experiment orchestration. Existing completed runs are never overwritten."""

from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import pandas as pd
import joblib
from .config import METHODS, smoke_config, model_config
from .data import FoldData
from .evaluation import evaluate
from .io import read_json, save_json, sha256
from .training import fit_predict


def recipe_signature(config):
    # Data paths and output names may differ across collaborators and folds.
    relevant = {k: v for k, v in config.items() if k not in {"data_root", "output_root", "seed"}}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def run(config, fold, methods, run_name, smoke=False, recipe_dir=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_name):
        raise ValueError("Run name must be a simple directory name")
    if any(method not in METHODS for method in methods):
        raise ValueError("Unknown method")
    config = smoke_config(config) if smoke else deepcopy(config)
    configurations = {method: model_config(config, method) for method in methods}
    data = FoldData(Path(config["data_root"]) / fold)
    if smoke:
        data.limit_for_smoke()
    if data.eval_role == "test" and recipe_dir is None:
        raise ValueError("Use a frozen development run's recipes for final2020")
    output = Path(config["output_root"]) / run_name
    if output.exists():
        raise ValueError("Run directory exists; choose a fresh run name")
    recipes = {}
    for method in methods:
        if recipe_dir is not None:
            recipe = read_json(Path(recipe_dir) / method / "recipe.json")
            if recipe.get("method") != method or recipe.get("signature") != recipe_signature(configurations[method]):
                raise ValueError("Recipe method/configuration mismatch")
            if recipe.get("source_role") != "validation" or recipe.get("smoke"):
                raise ValueError("Final refit requires a non-smoke validation recipe")
            if pd.Timestamp(recipe["source_evaluation_end"]) >= pd.Timestamp(data.split["evaluation_start"]):
                raise ValueError("Recipe must precede evaluation year")
            recipes[method] = recipe
    output.mkdir(parents=True)
    save_json(output / "config.json", config)
    save_json(output / "manifest.json", {"complete": False, "smoke": smoke, "methods": list(methods), "fold": fold})
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "torch", "scikit-learn", "scipy", "PyYAML")
    }
    save_json(output / "environment.json", {"python": platform.python_version(), "packages": versions})
    save_json(output / "data_contract.json", data.audit_summary())
    all_metrics = []
    for method in methods:
        folder = output / method
        folder.mkdir()
        selected = configurations[method]
        save_json(folder / "config.json", selected)
        prediction, result, artifact = fit_predict(method, data, selected, recipes.get(method))
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
            row.update(method=method, seed=config["seed"], fold=fold, smoke=smoke)
        all_metrics.extend(rows)
        # Local outputs are ignored by Git. No raw IDs, source strings or order keys.
        predictions = pd.DataFrame(prediction, columns=[f"Q_{t:.2f}" for t in config["taus"]])
        predictions.insert(0, "decision_day", daily.decision_day.to_numpy())
        predictions.to_csv(folder / "daily_quantiles.csv", index=False)
        recipe = {
            "method": method,
            "signature": recipe_signature(selected),
            "epochs": result["epochs"],
            "source_role": data.eval_role,
            "source_evaluation_end": data.split["evaluation_end"],
            "source_data_identity": data.identity,
            "smoke": smoke,
        }
        if "three_stage" in result:
            recipe["three_stage"] = result["three_stage"]
        save_json(folder / "recipe.json", recipe)
        save_json(folder / "training.json", result)
        save_json(folder / "metrics.json", rows)
        joblib.dump(
            {
                "method": method,
                "artifact": artifact,
                "data_identity": data.identity,
                "category_maps": data.category_maps,
                "preprocessing_state": data.state,
                "config": selected,
            },
            folder / "predictor.joblib",
        )
        save_json(folder / "complete.json", {"complete": True, "method": method, "smoke": smoke})
        print(
            json.dumps({"completed": method, "evaluation_days": len(daily), "scenarios": len(rows), "smoke": smoke}),
            flush=True,
        )
    pd.DataFrame(all_metrics).to_csv(output / "comparison.csv", index=False)
    save_json(
        output / "manifest.json",
        {
            "complete": True,
            "smoke": smoke,
            "methods": list(methods),
            "fold": fold,
            "data_identity": data.identity,
            "seed": config["seed"],
            "evaluation_role": data.eval_role,
            "P_definition": "1 - mean_cost / mean_SAA_cost",
            "artifacts": {
                str(p.relative_to(output)): sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return output
