"""Three experiment families on one explicitly synthetic data contract."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import re
import pandas as pd
import yaml
from .config import load_config, METHODS
from .data import FoldData
from .runner import run
from .io import read_json, save_json


def tasks(group, methods, variants):
    if group == "A":
        for method in methods:
            for variant in ["basic", "enhanced"] if method in {"LQR", "GBQR"} else ["full"]:
                yield method, variant
    else:
        for variant in variants["structural" if group == "B" else "information"]:
            yield "DAT", variant


def main(argv=None, default_config="configs/synthetic.yaml", variants="DAT/configs/experiments.yaml"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--group", choices=["A", "B", "C", "all"], default="A")
    parser.add_argument("--fold", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=[m for m in METHODS if m != "Historical-Receipt"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--recipe-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip verified completed runs; no within-epoch resume")
    args = parser.parse_args(argv)
    if "Historical-Receipt" in args.methods:
        raise ValueError(
            "Fully synthetic data have no genuine Historical Receipt; use the standalone real-data adapter"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.name):
        raise ValueError("Suite name must be a simple directory name")
    if args.fold == "test" and args.recipe_root is None:
        raise ValueError("Test suite requires --recipe-root pointing at a completed development suite")
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = str(args.data_root.resolve())
    data = FoldData(Path(config["data_root"]) / args.fold)
    if data.schema["schema_version"] != "synthetic_axid_v1":
        raise ValueError("This launcher accepts fully synthetic data only")
    protocol = yaml.safe_load(Path(variants).read_text(encoding="utf-8"))
    output = Path(config["output_root"]) / args.name
    signature = {
        "config": config,
        "group": args.group,
        "fold": args.fold,
        "methods": args.methods,
        "seeds": args.seeds,
        "smoke": args.smoke,
        "data_identity": data.identity,
        "recipe_root": str(args.recipe_root.resolve()) if args.recipe_root else None,
        "variants": protocol,
    }
    if output.exists():
        if not args.resume or read_json(output / "suite_config.json") != signature:
            raise ValueError("Suite exists; use --resume with the identical protocol")
    else:
        output.mkdir(parents=True)
        save_json(output / "suite_config.json", signature)
    config["output_root"] = str(output)
    metrics = []
    groups = ["A", "B", "C"] if args.group == "all" else [args.group]
    for group in groups:
        for method, variant in tasks(group, args.methods, protocol):
            seeds = args.seeds if method not in {"SAA", "Rolling-SAA", "LQR", "GBQR"} else args.seeds[:1]
            for seed in seeds:
                selected = deepcopy(config)
                selected["seed"] = seed
                override = selected.setdefault("model_overrides", {}).setdefault(method, {})
                if group == "B":
                    override["ablation"] = variant
                elif group == "C":
                    override["information"] = variant
                elif method in {"LQR", "GBQR"}:
                    override.setdefault(method.lower(), {})["input_set"] = variant
                run_name = f"{group}_{method}_{variant}_s{seed}"
                run_dir = output / run_name
                if run_dir.exists():
                    if not args.resume:
                        raise ValueError("Run already exists")
                    manifest = read_json(run_dir / "manifest.json")
                    from .io import sha256

                    if not manifest["complete"] or any(
                        sha256(run_dir / p) != digest for p, digest in manifest["artifacts"].items()
                    ):
                        raise ValueError("Incomplete/changed run cannot be skipped; use a fresh suite name")
                else:
                    recipe = args.recipe_root / run_name if args.recipe_root is not None else None
                    run(selected, args.fold, [method], run_name, args.smoke, recipe)
                rows = pd.read_csv(run_dir / "comparison.csv")
                rows["experiment"], rows["variant"] = group, variant
                metrics.append(rows)
    combined = pd.concat(metrics, ignore_index=True)
    combined.to_csv(output / "metrics_long.csv", index=False)
    save_json(output / "suite_complete.json", {"complete": True, "rows": len(combined), "smoke": args.smoke})
    print(f"Completed synthetic suite: {output}")
