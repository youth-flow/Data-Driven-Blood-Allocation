"""Frozen AXID-DC v10 protocol/reference contract tests."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from axid_synthetic.contract_v10 import FINAL_METRICS, validate_contract


ROOT = Path(__file__).resolve().parents[1]


def _load():
    protocol = yaml.safe_load(
        (ROOT / "synthetic/configs/confirmed_protocol_v10.yaml").read_text(encoding="utf-8")
    )
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v10.yaml").read_text(encoding="utf-8"))
    return protocol, reference


def test_v10_contract_is_frozen_and_self_consistent():
    protocol, reference = _load()
    assert validate_contract(protocol, reference) == {
        "status": "frozen",
        "scenario": "AXID-DC-v10",
        "G": 256.0,
        "methods": 8,
        "variants": 5,
        "levels": 4,
        "metrics": 4,
        "n_train": 524288,
        "n_test": 10000,
        "optimizer_steps": 4096,
    }
    assert protocol["performance_experiments"]["n_train"] == 128 * 4096
    assert protocol["evaluation"]["final_metrics"] == FINAL_METRICS
    assert protocol["performance_experiments"]["targets"] == [256.0]


def test_v10_rejects_two_different_G_values():
    protocol, reference = _load()
    changed = deepcopy(protocol)
    changed["decision"]["blood_center_target_G"] = 260.0
    with pytest.raises(ValueError, match="same 256"):
        validate_contract(changed, reference)


def test_v10_rejects_old_data_support_or_size():
    protocol, reference = _load()
    changed = deepcopy(protocol)
    changed["sampling"]["max_visible_orders"] = 8
    with pytest.raises(ValueError, match="138 visible"):
        validate_contract(changed, reference)

    changed = deepcopy(protocol)
    changed["performance_experiments"]["n_test"] = 50000
    with pytest.raises(ValueError, match="10,000 test"):
        validate_contract(changed, reference)


def test_v10_rejects_metric_or_experiment_drift():
    protocol, reference = _load()
    changed = deepcopy(protocol)
    changed["evaluation"]["final_metrics"].append("Pinball")
    with pytest.raises(ValueError, match="exactly two P"):
        validate_contract(changed, reference)

    changed = deepcopy(protocol)
    changed["performance_experiments"]["p2_ablation"]["variants"].append("param_matched")
    with pytest.raises(ValueError, match="P2 variants"):
        validate_contract(changed, reference)


def test_v10_rejects_silent_precision_or_oracle_budget_change():
    protocol, reference = _load()
    changed = deepcopy(protocol)
    changed["runtime"]["use_bf16"] = True
    with pytest.raises(ValueError, match="silent precision"):
        validate_contract(changed, reference)

    changed = deepcopy(protocol)
    changed["evaluation"]["conditional_oracle"]["paths_per_test_day"] = 1024
    with pytest.raises(ValueError, match="4,096-path"):
        validate_contract(changed, reference)
