"""All nine fixed S4 methods fit and predict on a small contract fixture."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from axid_synthetic.generator import generate_performance
from dat_study.config import load_config, model_config
from dat_study.data import FoldData
from dat_study.training import fit_predict

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["DAT", "SAA", "wSAA", "Rolling-SAA", "Flat-ETO", "Set-ETO", "LQR", "GBQR", "DeepQR"]


@pytest.fixture(scope="module")
def s4_data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("methods_s4") / "fold"
    config = yaml.safe_load((ROOT / "synthetic/configs/reference.yaml").read_text(encoding="utf-8"))
    config.update(n_train=32, n_test=12, scenario="S4")
    generate_performance(config, folder)
    return FoldData(folder)


def _config(method, seed=0):
    config = model_config(load_config(ROOT / "configs/synthetic.yaml"), method)
    config.update(seed=seed)
    config["training"].update(device="cpu", threads=1, epochs=1, batch_size=8)
    return config


@pytest.mark.parametrize("method", METHODS)
def test_method_fit_predict(s4_data, method):
    prediction, result, artifact = fit_predict(method, s4_data, _config(method))
    assert prediction.shape == (12, 1)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 36.5)).all()
    assert result.get("epochs", 0) >= 0
    assert artifact


def test_psi_features_are_19d(s4_data):
    from dat_study.tabular import PsiFeatures

    features = PsiFeatures().fit(s4_data)
    values = features.transform(s4_data, s4_data.daily_split("train"))
    assert values.shape == (len(s4_data.daily_split("train")), 19)
    patient = features.patient_input(s4_data, s4_data.order_split("train").head(5))
    assert patient.shape[1] == 22


def test_ablation_and_information_are_propagated():
    root_config = load_config(ROOT / "configs/synthetic.yaml")
    root_config["ablation"] = "mean_pool"
    root_config["information"] = "L0"
    selected = model_config(root_config, "DAT")
    assert selected["ablation"] == "mean_pool"
    assert selected["information"] == "L0"
