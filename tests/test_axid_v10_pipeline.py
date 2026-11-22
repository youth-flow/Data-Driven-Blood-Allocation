"""Stage ordering and smoke-data tests for the v10 runner."""

from argparse import Namespace
from pathlib import Path

import pytest

from axid_synthetic.pipeline_v10 import (
    P1_SPECS,
    P2_SPECS,
    P3_SPECS,
    _all_prediction_jobs,
    _runtime_config,
    load_frozen_specs,
    run_data_stage,
    run_model_stage,
)


ROOT = Path(__file__).resolve().parents[1]


def _args(**updates):
    values = dict(
        smoke=True,
        resume=False,
        device="cpu",
        threads=1,
        oracle_workers=1,
        approve_p1=False,
    )
    values.update(updates)
    return Namespace(**values)


def test_stage_specs_and_reuse_map_are_frozen():
    assert [spec[3] for spec in P3_SPECS] == ["L0", "L1", "L2", "L3"]
    assert [spec[2] for spec in P2_SPECS] == [
        "no_self_attention",
        "no_day_query",
        "mean_pool",
        "sum_pool",
    ]
    assert [spec[1] for spec in P1_SPECS] == ["SAA", "wSAA", "Flat", "Set-ETO", "LQR", "GBQR", "DeepQR"]
    jobs = _all_prediction_jobs()
    assert jobs["P3/L3"] == jobs["P2/Full"] == jobs["P1/DAT"] == "p3_L3"
    assert len(jobs) == 17


def test_smoke_data_stage_and_runtime_config(tmp_path):
    protocol, reference = load_frozen_specs(ROOT)
    output = tmp_path / "v10"
    data = run_data_stage(ROOT, output, protocol, reference, smoke=True)
    assert len(data.daily_split("train")) == 128
    assert len(data.daily_split("eval")) == 32
    config = _runtime_config(ROOT, output, data.directory, protocol, "DAT", "full", "L3", _args())
    assert config["taus"] == [0.9] and config["targets"] == [256.0]
    assert config["training"]["batch_size"] == 8
    assert config["training"]["max_steps"] == 16
    assert config["training"]["num_workers"] == 0
    assert config["model_overrides"]["DAT"]["training"]["learning_rate"] == 3e-4


def test_p1_requires_completed_stages_and_explicit_approval(tmp_path):
    protocol, _ = load_frozen_specs(ROOT)
    output = tmp_path / "v10"
    (output / "stages").mkdir(parents=True)
    for stage in ("data", "p3", "p2"):
        (output / "stages" / f"{stage}.json").write_text(
            f'{{"stage":"{stage}","complete":true}}', encoding="utf-8"
        )
    with pytest.raises(PermissionError, match="--approve-p1"):
        run_model_stage(ROOT, output, protocol, "p1", _args())
