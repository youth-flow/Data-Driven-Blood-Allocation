"""Generate an entirely artificial S4 local fold; never opens real data."""

import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from axid_synthetic.generator import generate_performance  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reference.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.data_seed is not None:
        config["data_seed"] = args.data_seed
    if args.n_train is not None:
        config["n_train"] = args.n_train
    if args.n_test is not None:
        config["n_test"] = args.n_test
    if args.smoke:
        config.update(n_train=48, n_test=16)
    print(json.dumps(generate_performance(config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
