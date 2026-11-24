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
    assert protocol["sampling"]["model_day_features"][-1] == "sum_hb_scaled"
    assert protocol["sampling"]["day_scaling"]["sum_hb"] == "sum((hb - 42) / 128) / 138"
    assert protocol["sampling"]["tabular_summary"]["dimension"] == 16
    assert protocol["dat"]["output_clip"] == [0.0, 520.0]
    assert reference["max_new_orders"] == 73
    assert reference["gamma_scale"] == 0.43
    assert reference["log_phi_intercept"] == -0.956


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
    changed["sampling"]["demand_upper"] = 519.0
    with pytest.raises(ValueError, match=r"\[0,520\]"):
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

    changed = deepcopy(protocol)
    changed["performance_experiments"]["p3_information"]["day_gates"]["L1"][-1] = 1
    with pytest.raises(ValueError, match="information hierarchy"):
        validate_contract(changed, reference)


def test_v10_rejects_hb_shortage_or_old_mdcev_scale():
    protocol, reference = _load()
    changed = deepcopy(protocol)
    changed["sampling"]["model_day_features"][-1] = "sum_hb_shortage_scaled"
    with pytest.raises(ValueError, match="direct-Hb"):
        validate_contract(changed, reference)

    changed_reference = deepcopy(reference)
    changed_reference["gamma_scale"] = 0.7
    with pytest.raises(ValueError, match="gamma scale/intercept"):
        validate_contract(protocol, changed_reference)


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
