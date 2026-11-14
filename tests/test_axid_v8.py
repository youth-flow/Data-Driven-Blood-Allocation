"""AXID-DC v8 data, Hb context, Flat and parameter contracts."""

from pathlib import Path
import numpy as np
import pytest
import yaml
from axid_synthetic.contract_v8 import validate_contract
from axid_synthetic.generator_v8 import generate_performance
from dat_study.config import load_config, model_config
from dat_study.data import FoldData
from dat_study.tabular import PsiFeatures
from dat_study.training import fit_predict
from datransformer.trainer import new_dat
from datransformer.model import structural_parameter_counts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v8(tmp_path_factory):
    folder = tmp_path_factory.mktemp("axidv8") / "fold"
    ref = yaml.safe_load((ROOT / "synthetic/configs/reference_v8.yaml").read_text(encoding="utf-8"))
    ref.update(data_seed=12345, n_train=64, n_test=24)
    generate_performance(ref, folder)
    return FoldData(folder)


def test_contract_data(v8):
    p = yaml.safe_load((ROOT / "synthetic/configs/confirmed_protocol_v8.yaml").read_text(encoding="utf-8"))
    r = yaml.safe_load((ROOT / "synthetic/configs/reference_v8.yaml").read_text(encoding="utf-8"))
    out = validate_contract(p, r)
    assert out == {"status": "frozen", "methods": 8, "variants": 8, "levels": 4, "d_model": 64}
    assert v8.schema["schema_version"] == "synthetic_axid_dc_v8"
    assert v8.context(v8.daily_split("eval")).shape[1] == 7
    assert v8.daily.D.between(0, 61).all()
    assert v8.order_num_cols == ["u", "r", "hb"]
    f = PsiFeatures().fit(v8)
    assert f.transform(v8, v8.daily_split("train")).shape == (64, 16)
    assert f.patient_input(v8, v8.order_split("train").head(4)).shape == (4, 19)


def test_flat(v8):
    c = model_config(load_config(ROOT / "configs/synthetic.yaml"), "Flat")
    c["training"].update(device="cpu", threads=1, epochs=2, batch_size=8)
    pred, _, a = fit_predict("Flat", v8, c)
    assert pred.shape == (24, 1)
    assert np.isfinite(pred).all()
    assert a


def test_params(v8):
    c = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    c["ablation"] = "full"
    full = structural_parameter_counts(new_dat(v8, c))
    assert full["structurally_active_parameters"] == 51105


def test_lite_parameter_counts(v8):
    expected = {
        "full": 43441,
        "no_self_attention": 26673,
        "no_day_query": 43505,
        "lite_mean_pool": 30833,
        "lite_sum_pool": 30833,
    }
    for variant, count in expected.items():
        c = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
        c["ablation"] = variant
        c["model"]["d_model"] = 64
        c["model"]["patient_encoder"]["hidden_dims"] = [32]
        c["model"]["day_encoder"]["hidden_dims"] = [16]
        c["model"]["cross_attention"]["ff_dim"] = 48
        assert structural_parameter_counts(new_dat(v8, c))["structurally_active_parameters"] == count
