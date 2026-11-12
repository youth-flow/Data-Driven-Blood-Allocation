"""AXID-DC v6 data, priority mechanism, Oracle and parameter contracts."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from axid_synthetic.contract_v6 import validate_contract
from axid_synthetic.generator_v6 import generate_performance
from axid_synthetic.oracle_v6 import conditional_pmf, quantile
from dat_study.config import load_config, model_config
from dat_study.data import FoldData
from dat_study.tabular import PsiFeatures
from datransformer.model import structural_parameter_counts
from datransformer.trainer import new_dat

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v6(tmp_path_factory):
    folder = tmp_path_factory.mktemp("axidv6") / "fold"
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v6.yaml").read_text(encoding="utf-8"))
    reference.update(data_seed=12345, n_train=64, n_test=24)
    generate_performance(reference, folder)
    return FoldData(folder)


def test_v6_contract_and_data(v6):
    protocol = yaml.safe_load((ROOT / "synthetic/configs/confirmed_protocol_v6.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v6.yaml").read_text(encoding="utf-8"))
    result = validate_contract(protocol, reference)
    assert result["methods"] == 8 and result["variants"] == 8 and result["levels"] == 4
    assert v6.schema["schema_version"] == "synthetic_axid_dc_v6"
    assert v6.context(v6.daily_split("eval")).shape[1] == 7
    assert len(v6.order_num_cols) == 3
    assert v6.daily.n_registered.between(3, 8).all()
    assert v6.daily.D.between(0.0, 61.0).all()


def test_v6_features(v6):
    features = PsiFeatures().fit(v6)
    assert features.transform(v6, v6.daily_split("train")).shape == (64, 16)
    assert features.patient_input(v6, v6.order_split("train").head(5)).shape == (5, 19)


def test_v6_lists():
    protocol = yaml.safe_load((ROOT / "synthetic/configs/confirmed_protocol_v6.yaml").read_text(encoding="utf-8"))
    settings = protocol["performance_experiments"]
    assert settings["information"]["levels"] == ["L0", "L1", "L2", "L3"]
    assert len(settings["comparison"]["methods"]) == 8
    assert "no_self_attention_param_matched" in settings["ablation"]["variants"]


def test_v6_parameter_matching(v6):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["ablation"] = "full"
    full = structural_parameter_counts(new_dat(v6, config))
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["ablation"] = "no_self_attention"
    config["model"]["cross_attention"]["ff_dim"] = 290
    matched = structural_parameter_counts(new_dat(v6, config))
    assert full["structurally_active_parameters"] == 81873
    assert abs(matched["structurally_active_parameters"] - 81873) / 81873 < 0.05


def test_v6_oracle(v6):
    daily = v6.daily_split("eval")
    orders = v6.order_split("eval")
    groups = dict(tuple(orders.groupby("decision_day", sort=False)))
    row = daily.iloc[0]
    frame = groups[row.decision_day]
    pmf = conditional_pmf(frame, row.planned_load, row.emergency_load, row.It)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-10)
    assert 0.0 <= quantile(pmf) <= 61.0
    permuted = conditional_pmf(
        frame.sample(frac=1, random_state=2), row.planned_load, row.emergency_load, row.It
    )
    assert np.max(np.abs(pmf - permuted)) < 1e-12
