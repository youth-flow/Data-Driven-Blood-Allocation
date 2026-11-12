"""AXID-DC v7 data, Flat baseline and Oracle contracts."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from axid_synthetic.contract_v7 import validate_contract
from axid_synthetic.generator_v7 import generate_performance
from axid_synthetic.oracle_v7 import conditional_pmf, quantile
from dat_study.config import load_config, model_config
from dat_study.data import FoldData
from dat_study.training import fit_predict
from dat_study.tabular import PsiFeatures
from datransformer.model import structural_parameter_counts
from datransformer.trainer import new_dat

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v7(tmp_path_factory):
    folder = tmp_path_factory.mktemp("axidv7") / "fold"
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v7.yaml").read_text(encoding="utf-8"))
    reference.update(data_seed=12345, n_train=64, n_test=24)
    generate_performance(reference, folder)
    return FoldData(folder)


def test_v7_contract_and_data(v7):
    protocol = yaml.safe_load((ROOT / "synthetic/configs/confirmed_protocol_v7.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v7.yaml").read_text(encoding="utf-8"))
    result = validate_contract(protocol, reference)
    assert result["methods"] == 8 and result["variants"] == 8 and result["levels"] == 4
    assert v7.schema["schema_version"] == "synthetic_axid_dc_v7"
    assert v7.context(v7.daily_split("eval")).shape[1] == 7
    assert v7.daily.n_registered.between(3, 8).all()
    assert v7.daily.D.between(0.0, 61.0).all()
    features = PsiFeatures().fit(v7)
    assert features.transform(v7, v7.daily_split("train")).shape == (64, 16)
    assert features.patient_input(v7, v7.order_split("train").head(4)).shape == (4, 19)


def test_v7_flat_method(v7):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "Flat")
    config.update(seed=0)
    config["training"].update(device="cpu", threads=1, epochs=2, batch_size=8)
    prediction, result, artifact = fit_predict("Flat", v7, config)
    assert prediction.shape == (24, 1)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 61.0)).all()
    assert artifact


def test_v7_parameter_matching(v7):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["ablation"] = "full"
    full = structural_parameter_counts(new_dat(v7, config))
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["ablation"] = "no_self_attention"
    config["model"]["cross_attention"]["ff_dim"] = 290
    matched = structural_parameter_counts(new_dat(v7, config))
    assert full["structurally_active_parameters"] == 81873
    assert abs(matched["structurally_active_parameters"] - 81873) / 81873 < 0.05


def test_v7_oracle(v7):
    daily = v7.daily_split("eval")
    orders = v7.order_split("eval")
    groups = dict(tuple(orders.groupby("decision_day", sort=False)))
    row = daily.iloc[0]
    frame = groups[row.decision_day]
    pmf = conditional_pmf(frame, row.planned_load, row.emergency_load, row.It)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-10)
    assert 0.0 <= quantile(pmf) <= 61.0
    permuted = conditional_pmf(
        frame.sample(frac=1, random_state=3), row.planned_load, row.emergency_load, row.It
    )
    assert np.max(np.abs(pmf - permuted)) < 1e-12
