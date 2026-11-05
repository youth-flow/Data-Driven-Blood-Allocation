"""Root entry for the S4-only theory (T1-T3) and performance (P1-P3) experiments."""

import argparse
from pathlib import Path
import os
import re
import json
import yaml
import sys
from importlib.metadata import version, PackageNotFoundError

from dat_study.io import save_json, read_json

from .jobs import source_fingerprint

THEORY = ["estimation", "approximation", "consistency"]
PERFORMANCE = ["comparison", "ablation", "information"]


def main(argv=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["theory", "performance", "all"], default="all")
    parser.add_argument("--group", choices=[*THEORY, *PERFORMANCE, "all"], default="all")
    parser.add_argument("--name", required=True)
    parser.add_argument("--protocol", type=Path, default=root / "synthetic/configs/confirmed_protocol.yaml")
    parser.add_argument("--output-root", type=Path, default=root / "outputs")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--train-sizes", nargs="+", type=int)
    parser.add_argument("--widths", nargs="+", type=int)
    parser.add_argument("--model-seeds", nargs="+", type=int)
    parser.add_argument("--eval-days", type=int)
    parser.add_argument("--data-seed", type=int, default=20261102)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.name):
        raise ValueError("Use a simple experiment name")
    if (args.repetitions is not None and args.repetitions < 1) or (args.eval_days is not None and args.eval_days < 2):
        raise ValueError("Positive repetitions and at least two evaluation cases required")
    if args.repeat_start < 0:
        raise ValueError("Repeat start must be nonnegative")
    if args.train_sizes and (any(x < 2 for x in args.train_sizes) or args.train_sizes != sorted(set(args.train_sizes))):
        raise ValueError("Training sizes must be unique increasing counts >= 2")
    if args.widths and (any(x < 1 for x in args.widths) or args.widths != sorted(set(args.widths))):
        raise ValueError("Widths must be unique increasing positive counts")
    if args.model_seeds and (
        any(x < 0 for x in args.model_seeds) or len(set(args.model_seeds)) != len(args.model_seeds)
    ):
        raise ValueError("Model seeds must be unique nonnegative integers")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    from dat_study.config import METHODS

    if args.methods and any(x not in METHODS or x == "Historical-Receipt" for x in args.methods):
        raise ValueError("Unknown method or unavailable historical receipts")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    generator = yaml.safe_load((root / "synthetic/configs/reference.yaml").read_text(encoding="utf-8"))
    from .contract import validate_contract

    validate_contract(protocol, generator)
    allowed = (
        THEORY if args.family == "theory" else PERFORMANCE if args.family == "performance" else THEORY + PERFORMANCE
    )
    groups = allowed if args.group == "all" else [args.group]
    if not set(groups) <= set(allowed):
        raise ValueError("Group does not belong to the selected family")
    plan = {
        "family": args.family,
        "groups": groups,
        "smoke": args.smoke,
        "protocol": protocol,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in {"resume", "dry_run"}
        },
    }
    environment = {"python": sys.version.split()[0], "MKL_THREADING_LAYER": os.getenv("MKL_THREADING_LAYER")}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "PyYAML", "matplotlib"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    plan["environment"] = environment
    print(json.dumps({"groups": groups, "smoke": args.smoke, "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        return
    output = args.output_root.resolve() / args.name
    plan["code_fingerprint"] = source_fingerprint(root)
    if output.exists():
        if not args.resume or read_json(output / "study_plan.json") != plan:
            raise ValueError("Study exists or recipe changed; use --resume with the same recipe or a new name")
    else:
        output.mkdir(parents=True)
        save_json(output / "study_plan.json", plan)
    theory, performance = [], []
    if any(x in THEORY for x in groups):
        from .theory import run_theory

        theory = run_theory(
            args, protocol, generator, root, output, [x for x in groups if x in THEORY], plan["code_fingerprint"]
        )
    if any(x in PERFORMANCE for x in groups):
        from .performance import run_performance

        performance = run_performance(
            args, protocol, generator, root, output, [x for x in groups if x in PERFORMANCE], plan["code_fingerprint"]
        )
    from .report import summarize

    plots = summarize(output, theory, performance)
    save_json(
        output / "study_complete.json",
        {
            "complete": True,
            "smoke": args.smoke,
            "groups": groups,
            "theory_rows": len(theory),
            "performance_rows": len(performance),
            **plots,
        },
    )
    print(
        json.dumps({"study_complete": str(output), "theory_rows": len(theory), "performance_rows": len(performance)}),
        flush=True,
    )
