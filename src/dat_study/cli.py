"""Portable command-line entry point."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from .config import FOLDS, METHODS, load_config

    parser = argparse.ArgumentParser(prog="dat-study")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("preprocess", help="Read local originals and write fresh derived folds")
    prep.add_argument("--data-root", type=Path, default=Path("data"))
    prep.add_argument("--output", type=Path)
    prep.add_argument("--fold", choices=["all", *FOLDS], default="all")
    audit = sub.add_parser("audit", help="Verify derived artifacts; report aggregates only")
    audit.add_argument("--data", type=Path, required=True)
    execute = sub.add_parser("run", help="Train and evaluate with a common contract")
    execute.add_argument("--config", default="configs/experiment.yaml")
    execute.add_argument("--fold", choices=FOLDS, default="val2019")
    execute.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:5]))
    execute.add_argument("--name", required=True)
    execute.add_argument("--smoke", action="store_true")
    execute.add_argument("--recipe-dir", type=Path)
    inference = sub.add_parser("predict", help="Reload a trusted local run without retraining")
    inference.add_argument("--run", type=Path, required=True)
    inference.add_argument("--method", choices=METHODS, required=True)
    inference.add_argument("--data", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        from runpy import run_path

        script = Path(__file__).resolve().parents[2] / "data" / "数据预处理.py"
        if not script.is_file():
            raise FileNotFoundError("Run preprocessing from the source checkout: python data/数据预处理.py")
        preprocess = run_path(str(script))["main"]

        forwarded = ["--data-root", str(args.data_root), "--fold", args.fold]
        if args.output is not None:
            forwarded += ["--output", str(args.output)]
        preprocess(forwarded)
    elif args.command == "audit":
        from .data import FoldData

        print(json.dumps(FoldData(args.data).audit_summary(), ensure_ascii=False))
    elif args.command == "predict":
        import pandas as pd
        from .inference import load_local_prediction

        if args.output.exists():
            raise ValueError("Prediction output exists; refusing to overwrite")
        daily, values, taus = load_local_prediction(args.run, args.method, args.data)
        result = pd.DataFrame(values, columns=[f"Q_{t:.2f}" for t in taus])
        result.insert(0, "decision_day", daily.decision_day.to_numpy())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(json.dumps({"predicted_days": len(result), "quantiles": len(taus)}))
    else:
        from .runner import run

        result = run(load_config(args.config), args.fold, args.methods, args.name, args.smoke, args.recipe_dir)
        print(f"Completed local run: {result}")


def model_main(method, default_config, argv=None):
    """Keep the five original model directories' standalone entrypoints."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from .config import FOLDS, load_config
    from .runner import run

    parser = argparse.ArgumentParser(prog=f"{method}/scripts/run_experiment.py")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--fold", choices=FOLDS, default="val2019")
    parser.add_argument("--name", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recipe-dir", type=Path)
    if method == "DAT":
        parser.add_argument("--three-stage", action="store_true")
    args = parser.parse_args(argv)
    selected = "DAT-3stage" if method == "DAT" and args.three_stage else method
    result = run(load_config(args.config), args.fold, [selected], args.name, args.smoke, args.recipe_dir)
    print(f"Completed local run: {result}")


if __name__ == "__main__":
    main()
