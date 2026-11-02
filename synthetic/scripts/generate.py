"""Generate entirely artificial local data, without opening any real data directory."""

import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from axid_synthetic.generator import generate, SCENARIOS  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reference.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-validation", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.scenario is not None:
        config["scenario"] = args.scenario
    if args.data_seed is not None:
        config["data_seed"] = args.data_seed
    for name in ("n_train", "n_validation", "n_test"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.smoke:
        config.update(n_train=48, n_validation=12, n_test=12)
    print(json.dumps(generate(config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
