"""DAT architecture and interface tests for the S4 contract."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from axid_synthetic.generator import generate_performance
from dat_study.config import load_config, model_config
from dat_study.data import DailySets, FoldData, collate
from dat_study.evaluation import decision
from datransformer.constraints import project_parameters
from datransformer.trainer import new_dat

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def s4_data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dat_s4") / "fold"
    config = yaml.safe_load((ROOT / "synthetic/configs/reference.yaml").read_text(encoding="utf-8"))
    config.update(n_train=48, n_test=16, scenario="S4")
    generate_performance(config, folder)
    return FoldData(folder)


@pytest.fixture(scope="module")
def dat_model(s4_data):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["training"].update(device="cpu", threads=1, epochs=1)
    return new_dat(s4_data, config), config


def _batch(data, count=3):
    dataset = DailySets(data, "eval")
    return collate([dataset[index] for index in range(count)])


def test_output_shape_and_bounds(s4_data, dat_model):
    model, _ = dat_model
    batch = _batch(s4_data)
    with torch.no_grad():
        prediction = model(batch)["quantile"]
    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 36.5)).all()


def test_patient_permutation_invariance(s4_data, dat_model):
    model, _ = dat_model
    batch = _batch(s4_data)
    changed = dict(batch)
    for name in ("order_num", "order_cat", "order_mask"):
        changed[name] = batch[name].flip(1)
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["quantile"], model(changed)["quantile"], atol=1e-5, rtol=1e-5)


def test_padding_invariance(s4_data, dat_model):
    model, _ = dat_model
    batch = _batch(s4_data, count=1)
    padded = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    padded["order_num"] = torch.cat([padded["order_num"], torch.zeros(1, 2, 3)], dim=1)
    padded["order_cat"] = torch.cat(
        [padded["order_cat"], torch.zeros(1, 2, padded["order_cat"].shape[-1], dtype=torch.long)], dim=1
    )
    padded["order_mask"] = torch.cat([padded["order_mask"], torch.zeros(1, 2, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["quantile"], model(padded)["quantile"], atol=1e-5, rtol=1e-5)


def test_count_coordinate(s4_data):
    batch = _batch(s4_data, count=2)
    assert torch.allclose(batch["day_features"][:, 3], batch["order_mask"].sum(1).float() / 8)


def test_parameter_counts():
    protocol = yaml.safe_load((ROOT / "synthetic/configs/confirmed_protocol.yaml").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("check_protocol", ROOT / "synthetic/scripts/check_protocol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.validate(protocol)
    assert result["profiles"]["theory"]["parameters"] == 5801
    assert result["profiles"]["performance"]["parameters"] == 76161
    assert [item["parameters"] for item in result["capacity"]] == [4761, 5801, 7881, 12041, 20361, 37001]


def test_architecture_contract(s4_data, dat_model):
    model, _ = dat_model
    assert sum(isinstance(module, torch.nn.LayerNorm) for module in model.modules()) == 3
    assert not hasattr(model, "output_norm")
    assert model.self_attention.num_heads == 4 and model.cross_attention.num_heads == 4
    assert tuple(model.self_attention.in_proj_weight.shape) == (3 * 64, 64)


def test_global_projection(s4_data, dat_model):
    model, config = dat_model
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(10.0)
    project_parameters(model, config)
    norm = torch.sqrt(sum(parameter.square().sum() for parameter in model.parameters()))
    assert float(norm) <= float(config["constraints"]["global_parameter_radius"]) + 1e-4


def test_independent_heads(s4_data):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["training"].update(device="cpu", threads=1, epochs=1)
    config["model"]["self_attention"]["num_heads"] = 2
    config["model"]["cross_attention"]["num_heads"] = 4
    model = new_dat(s4_data, config)
    assert model.self_attention.num_heads == 2 and model.cross_attention.num_heads == 4


def test_g_is_external(s4_data, dat_model):
    model, _ = dat_model
    batch = _batch(s4_data)
    with torch.no_grad():
        prediction = model(batch)["quantile"].numpy()
    inventory = batch["inventory"].numpy()
    assert np.allclose(decision(prediction, inventory, 300), np.maximum(prediction + 300 - inventory, 0))
    assert np.allclose(decision(prediction, inventory, 340), np.maximum(prediction + 340 - inventory, 0))
    with torch.no_grad():
        assert np.allclose(model(batch)["quantile"].numpy(), prediction)


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3", "L4"])
def test_information_levels_fixed_parameters(s4_data, level):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["training"].update(device="cpu", threads=1, epochs=1)
    full = new_dat(s4_data, config)
    config["information"] = level
    model = new_dat(s4_data, config).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel() for parameter in full.parameters()
    )
    batch = _batch(s4_data, count=2)
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    changed["day_num"][:, ~model.day_gate] += 100
    changed["order_num"][:, :, ~model.order_num_gate] += 100
    if model.drop_set:
        changed["order_num"] += 100
        changed["order_mask"] = ~changed["order_mask"]
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["quantiles"], model(changed)["quantiles"], atol=1e-6, rtol=0)


@pytest.mark.parametrize("variant", ["full", "no_self_attention", "mean_pool", "sum_pool", "no_day_query"])
def test_structural_variants(s4_data, variant):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    config["training"].update(device="cpu", threads=1, epochs=1)
    config["ablation"] = variant
    model = new_dat(s4_data, config).eval()
    batch = _batch(s4_data, count=2)
    with torch.no_grad():
        prediction = model(batch)["quantile"]
    assert prediction.shape == (2,) and torch.isfinite(prediction).all()


def test_scalar_tau_contract(s4_data):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT")
    assert config["taus"] == [0.9]
    config["training"].update(device="cpu", threads=1, epochs=1)
    model = new_dat(s4_data, config)
    assert model.tau == 0.9
