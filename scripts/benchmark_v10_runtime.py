"""Local, disposable AXID-DC v10 generation and one-DAT throughput benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import torch  # Load the conda environment's single OpenMP owner before SciPy.
import yaml


ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "DAT/src", ROOT / "synthetic/src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from axid_synthetic.generator_v10 import generate_performance  # noqa: E402
from dat_study.config import load_config  # noqa: E402
from dat_study.data import FoldData  # noqa: E402
from dat_study.training import fit_predict  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-days", type=int, default=32768)
    parser.add_argument("--test-days", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.train_days % args.batch_size:
        raise ValueError("train-days must be divisible by batch-size")

    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v10.yaml").read_text(encoding="utf-8"))
    reference.update(n_train=args.train_days, n_test=args.test_days, train_seed=101, test_seed=102)
    with tempfile.TemporaryDirectory(prefix="axid_v10_benchmark_") as temporary:
        folder = Path(temporary) / "fold"
        started = perf_counter()
        generate_performance(reference, folder)
        generated = perf_counter()
        data = FoldData(folder)
        loaded = perf_counter()
        config = load_config(ROOT / "configs/synthetic.yaml")
        config.update(
            seed=2026111404,
            taus=[0.9],
            targets=[256.0],
            data_root=str(folder),
            output_root=temporary,
        )
        config["training"] = {
            "device": args.device,
            "threads": args.threads,
            "epochs": 1,
            "batch_size": args.batch_size,
            "max_steps": args.train_days // args.batch_size,
            "num_workers": args.workers,
            "pin_memory": args.device == "cuda",
            "persistent_workers": args.workers > 0,
            "prefetch_factor": 4,
            "bucket_by_length": True,
            "shuffle_seed": 2026111403,
            "learning_rate": 3e-4,
            "warmup_steps": min(300, args.train_days // args.batch_size // 4),
            "warmup_start_lr": 3e-5,
            "minimum_lr": 3e-5,
            "weight_decay": 1e-4,
        }
        prediction, result, _ = fit_predict("DAT", data, config)
        finished = perf_counter()
        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "train_days": args.train_days,
                    "test_days": args.test_days,
                    "batch_size": args.batch_size,
                    "workers": args.workers,
                    "generation_seconds": generated - started,
                    "load_seconds": loaded - generated,
                    "dat_fit_predict_seconds": finished - loaded,
                    "optimizer_steps": result["scalar_fits"]["0.9"]["optimizer_steps"],
                    "prediction_days": len(prediction),
                },
                indent=2,
            )
        )
        data.close()


if __name__ == "__main__":
    main()
