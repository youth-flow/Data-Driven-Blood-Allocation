"""Merge disjoint study shards and rebuild the frozen summaries and figures."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "synthetic/src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)
    destination = (args.output_root / args.name).resolve()
    if destination.exists():
        raise ValueError("Merged study already exists; choose a fresh name")
    destination.mkdir(parents=True)
    theory, performance = [], []
    for source in args.sources:
        source = source.resolve()
        for filename, store in (("theory_metrics.csv", theory), ("performance_metrics.csv", performance)):
            path = source / filename
            if path.exists():
                frame = pd.read_csv(path)
                if not frame.empty:
                    store.append(frame)
    theory_rows = pd.concat(theory, ignore_index=True).drop_duplicates() if theory else pd.DataFrame()
    performance_rows = pd.concat(performance, ignore_index=True).drop_duplicates() if performance else pd.DataFrame()
    if theory_rows.empty and performance_rows.empty:
        raise ValueError("No theory or performance rows found in the shards")
    if not theory_rows.empty:
        theory_rows.to_csv(destination / "theory_metrics.csv", index=False)
    if not performance_rows.empty:
        performance_rows.to_csv(destination / "performance_metrics.csv", index=False)
    from axid_synthetic.report import summarize

    plots = summarize(
        destination,
        theory_rows.to_dict("records") if not theory_rows.empty else [],
        performance_rows.to_dict("records") if not performance_rows.empty else [],
    )
    (destination / "study_complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "merged_from": [str(path) for path in args.sources],
                "theory_rows": len(theory_rows),
                "performance_rows": len(performance_rows),
                **plots,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"merged": str(destination), "theory": len(theory_rows), "performance": len(performance_rows)}))


if __name__ == "__main__":
    main()
