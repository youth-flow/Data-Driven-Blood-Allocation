"""AXID-v3 data, DAT interface and all-method smoke contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from axid_synthetic.generator_v3 import generate_performance
from dat_study.config import load_config, model_config
from dat_study.data import DailySets, FoldData, collate
from dat_study.tabular import PsiFeatures
from dat_study.training import fit_predict
from datransformer.model import structural_parameter_counts
from datransformer.trainer import new_dat

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["DAT", "SAA", "wSAA", "Rolling-SAA", "Flat-ETO", "Set-ETO", "LQR", "GBQR", "DeepQR"]


@pytest.fixture(scope="module")
def axid(tmp_path_factory):
    folder = tmp_path_factory.mktemp("axidv3") / "fold"
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v3.yaml").read_text(encoding="utf-8"))
    reference.update(data_seed=12345, n_train=64, n_test=24)
    generate_performance(reference, folder)
    return FoldData(folder)


def test_data_contract(axid):
    assert axid.schema["schema_version"] == "synthetic_axid_axid_v3"
    assert axid.schema["generator_name"] == "AXID-v3"
    assert len(axid.daily_split("train")) == 64
    assert len(axid.daily_split("eval")) == 24
    assert len(axid.order_num_cols) == 4
    assert len(axid.day_cols) == 8
    assert axid.daily.n_orders.between(3, 8).all()
    assert axid.daily.D.between(0.0, 60.5).all()
    assert set(axid.orders.real_demand.unique()) <= {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
    assert "H" not in axid.daily and "latent_U" not in axid.daily


def test_x_order_and_count(axid):
    daily = axid.daily_split("eval")
    x = axid.context(daily)
    assert x.shape[1] == 9
    assert np.allclose(x[:, 2], (daily.It.to_numpy() - 225.0) / 180.0)
    batch = collate([DailySets(axid, "eval")[0], DailySets(axid, "eval")[1]])
    assert np.allclose(batch["day_features"][:, 2].numpy(), (batch["inventory"].numpy() - 225.0) / 180.0)
    assert np.allclose(batch["day_features"][:, 3].numpy(), batch["order_mask"].sum(1).numpy() / 8.0)


def test_psi_and_flat_inputs(axid):
    features = PsiFeatures().fit(axid)
    psi = features.transform(axid, axid.daily_split("train"))
    patient = features.patient_input(axid, axid.order_split("train").head(10))
    assert psi.shape == (64, 23)
    assert patient.shape == (10, 27)


def _dat_config(axid, variant="full", information=None):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config.update(seed=0)
    config["training"].update(device="cpu", threads=1, epochs=1, batch_size=8)
    config["ablation"] = variant.removesuffix("_param_matched")
    if variant.endswith("_param_matched"):
        width = 80
        inactive = 2 * (width * width + width) + 2 * width
        config["model"]["cross_attention"]["ff_dim"] += round(inactive / (2 * width + 1))
    if information is not None:
        config["information"] = information
    return config


def test_dat_output_and_variants(axid):
    model = new_dat(axid, _dat_config(axid)).eval()
    batch = collate([DailySets(axid, "eval")[0], DailySets(axid, "eval")[1]])
    with torch.no_grad():
        prediction = model(batch)["quantile"]
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 60.5)).all()

    full = structural_parameter_counts(new_dat(axid, _dat_config(axid)))
    assert full["structurally_active_parameters"] == 82129
    for variant in ("mean_pool_param_matched", "sum_pool_param_matched"):
        counts = structural_parameter_counts(new_dat(axid, _dat_config(axid, variant)))
        assert (
            abs(counts["structurally_active_parameters"] - full["structurally_active_parameters"])
            / full["structurally_active_parameters"]
            < 0.05
        )


def test_information_gates(axid):
    l0 = new_dat(axid, _dat_config(axid, information="L0"))
    l3 = new_dat(axid, _dat_config(axid, information="L3"))
    l4 = new_dat(axid, _dat_config(axid, information="L4"))
    assert l0.drop_set and not l0.day_gate[2:].any()
    assert not l3.drop_set and not bool(l3.order_num_gate[2])
    assert not l3.order_num_gate[2]
    assert bool(l4.order_num_gate[2])


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_fit(axid, method):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), method)
    config.update(seed=0)
    config["training"].update(device="cpu", threads=1, epochs=1, batch_size=8)
    prediction, result, artifact = fit_predict(method, axid, config)
    assert prediction.shape == (24, 1)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 60.5)).all()
    assert artifact
