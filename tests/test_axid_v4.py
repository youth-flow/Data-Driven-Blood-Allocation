"""AXID-v4 data, mechanism and parameter-matching contracts."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from axid_synthetic.contract_v4 import validate_contract
from axid_synthetic.generator_v4 import generate_performance
from axid_synthetic.oracle_v4 import conditional_pmf, quantile
from dat_study.config import load_config, model_config
from dat_study.data import FoldData
from dat_study.tabular import PsiFeatures
from datransformer.model import structural_parameter_counts
from datransformer.trainer import new_dat

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v4(tmp_path_factory):
    folder = tmp_path_factory.mktemp("axidv4") / "fold"
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v4.yaml").read_text(encoding="utf-8"))
    reference.update(data_seed=12345, n_train=64, n_test=24)
    generate_performance(reference, folder)
    return FoldData(folder)


def test_v4_contract_and_data(v4):
    protocol = yaml.safe_load((ROOT / "synthetic/configs/confirmed_protocol_v4.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v4.yaml").read_text(encoding="utf-8"))
    assert validate_contract(protocol, reference)["methods"] == 8
    assert v4.schema["schema_version"] == "synthetic_axid_axid_v4"
    assert len(v4.daily_split("train")) == 64 and len(v4.daily_split("eval")) == 24
    assert v4.daily.n_orders.between(3, 8).all() and v4.daily.D.between(0.0, 60.5).all()
    assert v4.context(v4.daily_split("eval")).shape[1] == 9
    assert PsiFeatures().fit(v4).transform(v4, v4.daily_split("train")).shape == (64, 23)


def test_v4_design_lists():
    protocol = yaml.safe_load((ROOT / "synthetic/configs/confirmed_protocol_v4.yaml").read_text(encoding="utf-8"))
    settings = protocol["performance_experiments"]
    assert len(settings["comparison"]["methods"]) == 8
    assert "Rolling-SAA" not in settings["comparison"]["methods"]
    assert len(settings["ablation"]["variants"]) == 8
    assert "no_self_attention_param_matched" in settings["ablation"]["variants"]
    assert settings["information"]["levels"] == ["L0", "L1", "L2", "L4"]


def test_v4_parameter_matching(v4):
    base = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    base["ablation"] = "full"
    full = structural_parameter_counts(new_dat(v4, base))
    matched = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    matched["ablation"] = "no_self_attention"
    matched["model"]["cross_attention"]["ff_dim"] = 290
    counts = structural_parameter_counts(new_dat(v4, matched))
    assert full["structurally_active_parameters"] == 82129
    assert abs(counts["structurally_active_parameters"] - 82129) / 82129 < 0.05


def test_v4_oracle(v4):
    daily = v4.daily_split("eval")
    orders = v4.order_split("eval")
    groups = dict(tuple(orders.groupby("decision_day", sort=False)))
    row = daily.iloc[0]
    frame = groups[row.decision_day]
    pmf = conditional_pmf(frame, row.scheduled_load, row.emergency_pressure)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-10)
    assert 0.0 <= quantile(pmf) <= 60.5
    permuted = conditional_pmf(frame.sample(frac=1, random_state=3), row.scheduled_load, row.emergency_pressure)
    assert np.max(np.abs(pmf - permuted)) < 1e-12
