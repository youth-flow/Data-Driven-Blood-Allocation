"""Report synthetic demand moments for preregistered v10 log-phi intercept candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synthetic/src"))

import axid_synthetic.generator_v10 as dgp  # noqa: E402


def moments(intercept, gamma_scale, days, seed):
    dgp.LOG_PHI_INTERCEPT = float(intercept)
    dgp.GAMMA_SCALE = float(gamma_scale)
    rng = np.random.default_rng(seed)
    context = dgp.draw_daily_context(rng, size=days)
    demand = np.empty(days)
    for index in range(days):
        demand[index] = dgp.sample_given_context(rng, *(coordinate[index] for coordinate in context))["demand"]
    return {
        "intercept": float(intercept),
        "gamma_scale": float(gamma_scale),
        "days": int(days),
        "mean": float(demand.mean()),
        "sd": float(demand.std()),
        "q50": float(np.quantile(demand, 0.5)),
        "q90": float(np.quantile(demand, 0.9)),
        "q99": float(np.quantile(demand, 0.99)),
        "zero_share": float((demand <= 1e-9).mean()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026111401)
    parser.add_argument("--intercepts", nargs="+", type=float, default=[-2.4, -2.2, -2.0, -1.8])
    parser.add_argument("--gamma-scales", nargs="+", type=float, default=[5.5])
    args = parser.parse_args(argv)
    if args.days < 100:
        raise ValueError("Use at least 100 days for a calibration diagnostic")
    print(
        json.dumps(
            [
                moments(intercept, gamma_scale, args.days, args.seed)
                for gamma_scale in args.gamma_scales
                for intercept in args.intercepts
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
