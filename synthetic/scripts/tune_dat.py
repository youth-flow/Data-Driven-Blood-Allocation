"""Tune DAT training hyperparameters on fixed validation folds.

Only the DAT model is trained. Validation folds are reused from an existing
synthetic study; no test data and no other method are touched.
"""

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dat_study.config import load_config  # noqa: E402
from dat_study.io import read_json  # noqa: E402
from dat_study.runner import run  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs="+", type=Path, required=True, help="Each directory contains validation/")
    parser.add_argument("--name", required=True, help="Fresh tuning study name")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/DAT_tuning")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.0001, 0.0003, 0.001])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for data_root in args.data_roots:
        if not (data_root / "validation").is_dir():
            raise ValueError(f"{data_root} must contain a validation/ fold")
    study = args.output_root / args.name
    if study.exists():
        if not args.force:
            raise ValueError("Tuning study already exists; choose a new name or pass --force")
        shutil.rmtree(study)
    study.mkdir(parents=True)

    base = load_config(ROOT / "configs/synthetic.yaml")
    base["output_root"] = str(study)
    base["seed"] = args.seed
    base["selection_metric"] = "pinball"
    base["training"] = {
        "device": args.device,
        "threads": args.threads,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": 128,
        "learning_rate": 0.0003,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
    }

    rows = []
    combinations = list(itertools.product(args.learning_rates, args.batch_sizes))
    for learning_rate, batch_size in combinations:
        label = f"lr{learning_rate:g}_bs{batch_size}"
        print(f"[tuning] {label}", flush=True)
        per_scenario = []
        tau_epochs = {}
        for data_root in args.data_roots:
            tag = data_root.name
            run_name = f"{label}__{tag}"
            config = deepcopy(base)
            config["data_root"] = str(data_root.resolve())
            config["training"]["learning_rate"] = learning_rate
            config["training"]["batch_size"] = batch_size
            run(config, "validation", ["DAT"], run_name, smoke=False)
            folder = study / run_name
            frame = pd.read_csv(folder / "comparison.csv")
            training_path = folder / "DAT/training.json"
            epochs = {}
            if training_path.exists():
                training = read_json(training_path)
                epochs = {str(tau): int(result["epochs"]) for tau, result in training.get("scalar_fits", {}).items()}
                tau_epochs[tag] = epochs
            per_scenario.append(
                {
                    "scenario": tag,
                    "pinball": float(frame.groupby("tau").pinball.first().mean()),
                    "average_cost": float(frame.average_cost.mean()),
                    "SL_G": float(frame.SL_G.mean()),
                    "mean_q": float(frame.mean_q.mean()),
                    "binding_share": float(frame.binding_share.mean()),
                    "tau_epochs": epochs,
                }
            )
        rows.append(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs_requested": args.epochs,
                "patience": args.patience,
                "seed": args.seed,
                "scenarios": len(per_scenario),
                "validation_pinball": float(pd.Series([row["pinball"] for row in per_scenario]).mean()),
                "validation_average_cost": float(pd.Series([row["average_cost"] for row in per_scenario]).mean()),
                "validation_SL_G": float(pd.Series([row["SL_G"] for row in per_scenario]).mean()),
                "validation_mean_q": float(pd.Series([row["mean_q"] for row in per_scenario]).mean()),
                "validation_binding_share": float(pd.Series([row["binding_share"] for row in per_scenario]).mean()),
                "per_scenario": json.dumps(per_scenario),
                "tau_epochs": json.dumps(tau_epochs),
            }
        )
        pd.DataFrame(rows).to_csv(study / "dat_tuning.csv", index=False)
        print(
            f"[tuning] {label} pinball={rows[-1]['validation_pinball']:.6g} "
            f"AC={rows[-1]['validation_average_cost']:.6g} "
            f"SL={rows[-1]['validation_SL_G']:.4f}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table.to_csv(study / "dat_tuning.csv", index=False)
    print(table.sort_values(["validation_pinball", "validation_average_cost"]).to_string(index=False))
    print("\nBest by validation pinball:")
    print(table.nsmallest(1, "validation_pinball").to_string(index=False))
    print("\nBest by validation average cost:")
    print(table.nsmallest(1, "validation_average_cost").to_string(index=False))


if __name__ == "__main__":
    main()
