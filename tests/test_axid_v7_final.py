"""AXID-DC v7-Final data, Flat and contract checks."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from axid_synthetic.contract_v7_final import validate_contract
from axid_synthetic.generator_v7_final import generate_performance
from dat_study.config import load_config, model_config
from dat_study.data import FoldData
from dat_study.training import fit_predict
from dat_study.tabular import PsiFeatures
from datransformer.model import structural_parameter_counts
from datransformer.trainer import new_dat

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def final(tmp_path_factory):
    folder = tmp_path_factory.mktemp("axidv7final") / "fold"
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v7_final.yaml").read_text(encoding="utf-8"))
    reference.update(data_seed=12345, n_train=64, n_test=24)
    generate_performance(reference, folder)
    return FoldData(folder)


def test_final_contract_and_data(final):
    protocol = yaml.safe_load((ROOT / "synthetic/configs/confirmed_protocol_v7_final.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v7_final.yaml").read_text(encoding="utf-8"))
    result = validate_contract(protocol, reference)
    assert result["methods"] == 8 and result["variants"] == 8 and result["levels"] == 4
    assert final.schema["schema_version"] == "synthetic_axid_dc_v7_final"
    assert final.context(final.daily_split("eval")).shape[1] == 7
    assert final.daily.D.between(0.0, 61.0).all()
    assert PsiFeatures().fit(final).transform(final, final.daily_split("train")).shape == (64, 16)


def test_final_flat(final):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "Flat")
    config["training"].update(device="cpu", threads=1, epochs=2, batch_size=8)
    prediction, _, artifact = fit_predict("Flat", final, config)
    assert prediction.shape == (24, 1)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 61.0)).all()
    assert artifact


def test_final_parameter_matching(final):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["ablation"] = "full"
    full = structural_parameter_counts(new_dat(final, config))
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["ablation"] = "no_self_attention"
    config["model"]["cross_attention"]["ff_dim"] = 290
    matched = structural_parameter_counts(new_dat(final, config))
    assert full["structurally_active_parameters"] == 81873
    assert abs(matched["structurally_active_parameters"] - 81873) / 81873 < 0.05
