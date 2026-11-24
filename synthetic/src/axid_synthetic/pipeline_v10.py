"""Resumable staged runner for the frozen AXID-DC v10 performance study.

The stages deliberately mirror the preregistered execution order.  In
particular P2 reuses the P3-L3 checkpoint, P1 reuses the same DAT checkpoint,
and P1 cannot start without an explicit command-line approval flag.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

from dat_study.config import load_config
from dat_study.data import FoldData
from dat_study.io import read_json, save_json, sha256
from dat_study.training import fit_predict

from .contract_v10 import validate_contract
from .generator_v10 import generate_performance
from .jobs import run_job, source_fingerprint
from .metrics_v10 import (
    COMMON_G,
    TAU,
    evaluate_single_method,
    paired_bootstrap,
)
from .oracle_v10 import ObservedState, conditional_oracle_quantiles


P3_SPECS = (
    ("p3_L0", "DAT", "full", "L0"),
    ("p3_L1", "DAT", "full", "L1"),
    ("p3_L2", "DAT", "full", "L2"),
    ("p3_L3", "DAT", "full", "L3"),
)
P2_SPECS = (
    ("p2_no_self_attention", "DAT", "no_self_attention", None),
    ("p2_no_day_query", "DAT", "no_day_query", None),
    ("p2_mean_pool", "DAT", "mean_pool", None),
    ("p2_sum_pool", "DAT", "sum_pool", None),
)
P1_SPECS = tuple(
    (f"p1_{method.lower().replace('-', '_')}", method, "full", None)
    for method in ("SAA", "wSAA", "Flat", "Set-ETO", "LQR", "GBQR", "DeepQR")
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def load_frozen_specs(root):
    root = Path(root).resolve()
    protocol = yaml.safe_load((root / "synthetic/configs/confirmed_protocol_v10.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load((root / "synthetic/configs/reference_v10.yaml").read_text(encoding="utf-8"))
    validate_contract(protocol, reference)
    return protocol, reference


def _stage_file(output, stage):
    return Path(output) / "stages" / f"{stage}.json"


def _require_stage(output, stage):
    path = _stage_file(output, stage)
    if not path.exists():
        raise RuntimeError(f"Stage {stage} must be completed first")
    return read_json(path)


def _mark_stage(output, stage, value):
    save_json(_stage_file(output, stage), {"stage": stage, "complete": True, **_jsonable(value)})


def _validate_run_mode(data, protocol, args):
    experiment = protocol["performance_experiments"]
    expected = (128, 32) if args.smoke else (int(experiment["n_train"]), int(experiment["n_test"]))
    observed = (len(data.daily_split("train")), len(data.daily_split("eval")))
    if observed != expected:
        raise RuntimeError(f"Requested smoke/formal mode expects {expected}, but the frozen data contain {observed}")


def _data_recipe(protocol, reference, smoke):
    recipe = dict(reference)
    if smoke:
        recipe.update(n_train=128, n_test=32, export_csv=False)
    else:
        experiment = protocol["performance_experiments"]
        recipe.update(n_train=int(experiment["n_train"]), n_test=int(experiment["n_test"]), export_csv=False)
    return recipe


def run_data_stage(root, output, protocol, reference, *, smoke=False, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = _data_recipe(protocol, reference, smoke)
    folder = output / "dataset"
    if folder.exists():
        if not resume:
            raise ValueError("v10 dataset already exists; use --resume")
        existing = read_json(folder / "generation.yaml")
        if existing != recipe:
            raise ValueError("Existing v10 dataset was generated from a different recipe")
    else:
        generate_performance(recipe, folder)
    data = FoldData(folder)
    expected = (int(recipe["n_train"]), int(recipe["n_test"]))
    observed = (len(data.daily_split("train")), len(data.daily_split("eval")))
    if observed != expected:
        raise RuntimeError(f"v10 dataset counts {observed} do not match {expected}")
    _mark_stage(output, "data", {"recipe": recipe, "data_identity": data.identity})
    return data


def _runtime_config(root, folder, data_root, protocol, method, variant, information, args):
    experiment = protocol["performance_experiments"]
    runtime = protocol["runtime"]
    smoke = bool(args.smoke)
    batch_size = 8 if smoke else int(experiment["batch_size"])
    steps = 16 if smoke else int(experiment["optimizer_steps"])
    config = load_config(Path(root) / "configs/synthetic.yaml")
    config.update(
        seed=int(experiment["seeds"]["forest"] if method == "wSAA" else experiment["seeds"]["model_init"]),
        taus=[float(experiment["taus"][0])],
        targets=[float(experiment["targets"][0])],
        data_root=str(data_root),
        output_root=str(folder),
        selection_metric="fixed_configuration",
    )
    config["training"] = {
        "device": args.device,
        "threads": int(args.threads),
        "epochs": 1,
        "batch_size": batch_size,
        "max_steps": steps,
        "gradient_clip": float(experiment["gradient_clip"]),
        "shuffle_seed": int(experiment["seeds"]["shuffle"]),
        "num_workers": 0 if smoke else int(runtime["dataloader_workers"]),
        "pin_memory": bool(runtime["pin_memory"] and args.device != "cpu"),
        "persistent_workers": bool(runtime["persistent_workers"] and not smoke),
        "prefetch_factor": 4,
        "bucket_by_length": bool(runtime["bucket_by_visible_orders"]),
    }
    neural_training = {
        "learning_rate": float(experiment["peak_learning_rate"]),
        "warmup_steps": (
            min(int(experiment["warmup_steps"]), max(1, steps // 4))
            if smoke
            else int(experiment["warmup_steps"])
        ),
        "warmup_start_lr": float(experiment["warmup_start_learning_rate"]),
        "minimum_lr": float(experiment["final_learning_rate"]),
        "weight_decay": float(experiment["weight_decay"]),
        "betas": list(experiment["betas"]),
        "eps": float(experiment["optimizer_eps"]),
    }
    config["model_overrides"] = {
        "DAT": {"training": neural_training},
        "DeepQR": {
            "training": neural_training,
            "deep_qr": {"output_upper_bound": 520.0},
        },
        "Set-ETO": {"training": neural_training},
        "wSAA": {
            "wsaa": {
                "n_estimators": 8 if smoke else 128,
                "max_depth": 8 if smoke else 16,
                "min_samples_leaf": 8 if smoke else 128,
                "max_features": 0.75,
                "max_samples": 0.5,
            }
        },
    }
    if variant != "full":
        config["ablation"] = variant
    if information is not None:
        config["information"] = information
    return config


def _job_attempt(output, name):
    job = Path(output) / "jobs" / name
    record = read_json(job / "complete.json")
    result = (job / record["result"]).resolve()
    attempt = result.parent
    relative = str((attempt / "prediction.npy").relative_to(job))
    if relative not in record["artifacts"] or sha256(attempt / "prediction.npy") != record["artifacts"][relative]:
        raise ValueError(f"Prediction integrity check failed for {name}")
    return attempt


def load_prediction(output, name):
    return np.load(_job_attempt(output, name) / "prediction.npy", allow_pickle=False)


def _run_model_job(root, output, data, protocol, spec, args, code_hash):
    name, method, variant, information = spec
    signature = {
        "scenario": "AXID-DC-v10",
        "protocol": protocol,
        "data_identity": data.identity,
        "name": name,
        "method": method,
        "variant": variant,
        "information": information,
        "smoke": bool(args.smoke),
        "code": code_hash,
    }

    def action(folder):
        job_started = perf_counter()
        config = _runtime_config(root, folder, data.directory, protocol, method, variant, information, args)
        if args.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        started = perf_counter()
        prediction, result, artifact = fit_predict(method, data, config)
        if args.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        fit_duration_seconds = perf_counter() - started
        prediction = np.asarray(prediction, dtype=float)
        if prediction.ndim != 2 or prediction.shape != (len(data.daily_split("eval")), 1):
            raise RuntimeError(f"{name} returned an invalid prediction shape {prediction.shape}")
        prediction = np.clip(prediction[:, 0], 0.0, float(data.schema["demand_support_upper"]))
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{name} returned nonfinite predictions")
        np.save(folder / "prediction.npy", prediction.astype(np.float32), allow_pickle=False)
        # A fitted v10 forest carries hundreds of MB of leaf-membership caches
        # used only to produce the already frozen test prediction.  Persist its
        # recipe/result rather than duplicating that transient cache on disk.
        if method == "wSAA" and isinstance(artifact, dict) and "estimator" in artifact:
            estimator = artifact["estimator"]
            for transient in ("train_leaves_", "leaf_members_"):
                if hasattr(estimator, transient):
                    delattr(estimator, transient)
            artifact["leaf_cache_rebuild_from_frozen_training_pool"] = True
        stored_artifact = artifact
        torch.save(stored_artifact, folder / "artifact.pt")
        save_json(folder / "fit_result.json", _jsonable(result))
        return {
            "name": name,
            "method": method,
            "variant": variant,
            "information": information or "",
            "n_test": len(prediction),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "fit_duration_seconds": fit_duration_seconds,
            "duration_seconds": perf_counter() - job_started,
        }

    return run_job(Path(output) / "jobs", name, signature, action, resume=args.resume)


def run_model_stage(root, output, protocol, stage, args):
    _require_stage(output, "data")
    if stage == "p3":
        specs = P3_SPECS
    elif stage == "p2":
        _require_stage(output, "p3")
        specs = P2_SPECS
    elif stage == "p1":
        _require_stage(output, "p2")
        if not args.approve_p1:
            raise PermissionError("P1 requires the explicit --approve-p1 flag after reviewing P3/P2")
        specs = P1_SPECS
    else:
        raise ValueError(f"Unknown model stage {stage}")
    data = FoldData(Path(output) / "dataset")
    _validate_run_mode(data, protocol, args)
    code_hash = source_fingerprint(root)
    if stage in {"p2", "p1"}:
        previous = _require_stage(output, "p3")
        if previous.get("code_fingerprint") != code_hash:
            raise RuntimeError("Code changed after P3; do not reuse its Full DAT checkpoint")
    completed = [_run_model_job(root, output, data, protocol, spec, args, code_hash) for spec in specs]
    aliases = {}
    if stage == "p2":
        aliases["p2_full"] = "p3_L3"
    elif stage == "p1":
        aliases["p1_dat"] = "p3_L3"
    _mark_stage(output, stage, {"jobs": completed, "aliases": aliases, "code_fingerprint": code_hash})
    return completed


def _all_prediction_jobs():
    return {
        "P3/L0": "p3_L0",
        "P3/L1": "p3_L1",
        "P3/L2": "p3_L2",
        "P3/L3": "p3_L3",
        "P2/Full": "p3_L3",
        "P2/No-SA": "p2_no_self_attention",
        "P2/No-Day-Query": "p2_no_day_query",
        "P2/Mean-Pool": "p2_mean_pool",
        "P2/Sum-Pool": "p2_sum_pool",
        "P1/DAT": "p3_L3",
        "P1/SAA": "p1_saa",
        "P1/wSAA": "p1_wsaa",
        "P1/Flat": "p1_flat",
        "P1/Set-ETO": "p1_set_eto",
        "P1/LQR": "p1_lqr",
        "P1/GBQR": "p1_gbqr",
        "P1/DeepQR": "p1_deepqr",
    }


def _observed_states(data):
    patients, _, offsets, _ = data.patient_arrays("eval")
    daily = data.daily_split("eval").reset_index(drop=True)
    if len(offsets) != len(daily) + 1:
        raise RuntimeError("Oracle patient offsets do not align with test days")
    return tuple(
        ObservedState(
            patients[int(offsets[index]) : int(offsets[index + 1])],
            daily.planned_load.iloc[index],
            daily.emergency_load.iloc[index],
            daily.It.iloc[index],
        )
        for index in range(len(daily))
    )


def _oracle_predictions(root, output, data, protocol, args):
    folder = Path(output) / "oracle"
    prediction_file = folder / "prediction.npy"
    settings = protocol["evaluation"]["conditional_oracle"]
    experiment = protocol["performance_experiments"]
    draws = 256 if args.smoke else int(settings["paths_per_test_day"])
    batch = min(256 if args.smoke else 1024, draws)
    recipe = {
        "data_identity": data.identity,
        "n_test": len(data.daily_split("eval")),
        "draws": draws,
        "batch": batch,
        "tau": float(experiment["taus"][0]),
        "seed": int(experiment["seeds"]["oracle"]),
        "code_fingerprint": source_fingerprint(root),
    }
    manifest_file = folder / "manifest.json"
    if prediction_file.exists():
        if not args.resume:
            raise ValueError("Oracle output exists; use --resume")
        manifest = read_json(manifest_file)
        if manifest.get("recipe") != recipe or sha256(prediction_file) != manifest.get("prediction_sha256"):
            raise ValueError("Saved Oracle recipe or prediction integrity mismatch")
        prediction = np.load(prediction_file, allow_pickle=False)
        if prediction.shape != (len(data.daily_split("eval")),):
            raise ValueError("Saved Oracle prediction shape mismatch")
        return prediction
    folder.mkdir(parents=True, exist_ok=True)
    states = _observed_states(data)
    prediction, estimates = conditional_oracle_quantiles(
        states,
        n_draws=draws,
        batch_size=batch,
        tau=float(experiment["taus"][0]),
        seed=int(experiment["seeds"]["oracle"]),
        workers=int(args.oracle_workers),
    )
    np.save(prediction_file, prediction.astype(np.float32), allow_pickle=False)
    block = np.asarray([estimate.block_quantiles for estimate in estimates], dtype=np.float32)
    np.save(folder / "block_quantiles.npy", block, allow_pickle=False)

    audit_days = min(5 if args.smoke else int(settings["stability_check_days"]), len(states))
    audit_draws = 512 if args.smoke else int(settings["stability_check_paths"])
    audit, _ = conditional_oracle_quantiles(
        states[:audit_days],
        n_draws=audit_draws,
        batch_size=min(1024, audit_draws),
        tau=float(experiment["taus"][0]),
        seed=int(experiment["seeds"]["oracle"]) + 1,
        workers=int(args.oracle_workers),
    )
    delta = audit - prediction[:audit_days]
    save_json(
        folder / "stability.json",
        {
            "days": audit_days,
            "main_draws": draws,
            "audit_draws": audit_draws,
            "mean_absolute_quantile_difference": float(np.abs(delta).mean()),
            "max_absolute_quantile_difference": float(np.abs(delta).max()),
        },
    )
    save_json(manifest_file, {"recipe": recipe, "prediction_sha256": sha256(prediction_file)})
    return prediction


def run_metrics_stage(root, output, protocol, args):
    _require_stage(output, "p1")
    data = FoldData(Path(output) / "dataset")
    _validate_run_mode(data, protocol, args)
    current_code = source_fingerprint(root)
    if _require_stage(output, "p3").get("code_fingerprint") != current_code:
        raise RuntimeError("Code changed after P3; final metrics cannot mix incompatible checkpoints")
    train = data.daily_split("train")
    test = data.daily_split("eval")
    demand = test.D.to_numpy(float)
    inventory = test.It.to_numpy(float)
    oracle_prediction = _oracle_predictions(root, output, data, protocol, args)
    predictions = {name: load_prediction(output, job) for name, job in _all_prediction_jobs().items()}
    summaries, costs, services = [], {}, {}
    references = None
    for name, prediction in predictions.items():
        result = evaluate_single_method(
            prediction,
            demand,
            inventory,
            train.D.to_numpy(float),
            oracle_prediction,
            tau=TAU,
            target=COMMON_G,
        )
        summaries.append(
            {
                "configuration": name,
                "P_PF": result["P_PF"],
                "P_PF_defined": result["P_PF_defined"],
                "P_Oracle": result["P_Oracle"],
                "P_Oracle_defined": result["P_Oracle_defined"],
                "MeanCost": result["MeanCost"],
                "SL_256": result["SL_G"],
                "n_test": len(demand),
            }
        )
        costs[name] = result["daily"]["method"]["cost"]
        services[name] = result["daily"]["method"]["service"].astype(float)
        if references is None:
            references = result["daily"]
    metrics = pd.DataFrame(summaries)
    metrics.to_csv(Path(output) / "final_metrics.csv", index=False)
    daily_archive = {}
    for index, name in enumerate(predictions):
        daily_archive[f"cost_{index:02d}"] = costs[name].astype(np.float32)
        daily_archive[f"service_{index:02d}"] = services[name].astype(np.uint8)
    np.savez(Path(output) / "daily_metric_arrays.npz", **daily_archive)
    save_json(Path(output) / "daily_metric_names.json", {str(i): name for i, name in enumerate(predictions)})

    bootstrap_repetitions = 100 if args.smoke else int(protocol["evaluation"]["bootstrap"]["repetitions"])
    bootstrap = paired_bootstrap(
        costs,
        services,
        saa_cost=references["SAA"]["cost"],
        pf_cost=references["PF"]["cost"],
        oracle_cost=references["Oracle"]["cost"],
        replications=bootstrap_repetitions,
        seed=int(protocol["performance_experiments"]["seeds"]["bootstrap"]),
    )
    bootstrap["metric"] = bootstrap.metric.replace({"SL_G": "SL_256"})
    bootstrap.rename(columns={"method": "configuration"}, inplace=True)
    bootstrap.to_csv(Path(output) / "bootstrap_intervals.csv", index=False)
    _mark_stage(
        output,
        "metrics",
        {
            "configurations": len(metrics),
            "n_test": len(test),
            "bootstrap_replications": bootstrap_repetitions,
            "oracle_paths": 256
            if args.smoke
            else int(protocol["evaluation"]["conditional_oracle"]["paths_per_test_day"]),
        },
    )
    return metrics, bootstrap


def build_parser(root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("data", "p3", "p2", "p1", "metrics"))
    parser.add_argument("--output", type=Path, default=Path(root) / "outputs/axid_v10_fast")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--oracle-workers", type=int, default=6)
    parser.add_argument("--approve-p1", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None, root=None):
    root = Path(root or Path(__file__).resolve().parents[3]).resolve()
    args = build_parser(root).parse_args(argv)
    if args.threads < 1 or args.oracle_workers < 1:
        raise ValueError("threads and oracle-workers must be positive")
    protocol, reference = load_frozen_specs(root)
    args.output = args.output.resolve()
    if args.stage == "data":
        run_data_stage(root, args.output, protocol, reference, smoke=args.smoke, resume=args.resume)
    elif args.stage in {"p3", "p2", "p1"}:
        run_model_stage(root, args.output, protocol, args.stage, args)
    else:
        run_metrics_stage(root, args.output, protocol, args)
    print(json.dumps({"stage": args.stage, "complete": True, "output": str(args.output)}), flush=True)

