"""Compact v10 fold integration with the shared data and DAT interfaces."""

from pathlib import Path

import numpy as np
import yaml

from axid_synthetic.generator_v10 import generate_performance
from dat_study.config import load_config
from dat_study.data import FoldData
from dat_study.tabular import PsiFeatures
from dat_study.training import fit_predict
from datransformer.experiments import experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_compact_fold_and_dat_smoke(tmp_path):
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v10.yaml").read_text(encoding="utf-8"))
    reference.update(n_train=64, n_test=16, train_seed=1234, test_seed=5678, export_csv=False)
    folder = tmp_path / "v10_compact"
    generate_performance(reference, folder)

    data = FoldData(folder)
    assert data.schema["schema_version"] == "synthetic_axid_dc_v10"
    assert data.context(data.daily_split("train")).shape == (64, 7)
    patients, labels, offsets, days = data.patient_arrays("train")
    assert patients.shape[1] == 3 and labels.shape == (len(patients),)
    assert offsets.shape == (65,) and days.shape == (64,)
    assert PsiFeatures().fit(data).transform(data, data.daily_split("train")).shape == (64, 16)

    config = load_config(ROOT / "configs/synthetic.yaml")
    config.update(seed=91, taus=[0.9], targets=[280.0], data_root=str(folder), output_root=str(tmp_path))
    config["training"] = {
        "device": "cpu",
        "threads": 1,
        "epochs": 1,
        "batch_size": 8,
        "max_steps": 8,
        "num_workers": 0,
        "shuffle_seed": 92,
    }
    prediction, result, _ = fit_predict("DAT", data, config)
    assert prediction.shape == (16, 1)
    assert np.isfinite(prediction).all()
    assert result["scalar_fits"]["0.9"]["optimizer_steps"] == 8


def test_v10_information_gates_match_the_frozen_hierarchy(tmp_path):
    reference = yaml.safe_load((ROOT / "synthetic/configs/reference_v10.yaml").read_text(encoding="utf-8"))
    reference.update(n_train=8, n_test=4, train_seed=901, test_seed=902, export_csv=False)
    folder = tmp_path / "v10_information"
    generate_performance(reference, folder)
    data = FoldData(folder)
    base = load_config(ROOT / "DAT/configs/main.yaml")
    expected = {
        "L0": ([True, True, False, False, False, False], True),
        "L1": ([True, True, True, True, False, False], True),
        "L2": ([True, True, True, True, True, True], True),
        "L3": ([True, True, True, True, True, True], False),
    }
    for level, (day_gate, drop_set) in expected.items():
        config = dict(base)
        config["information"] = level
        controls = experiment_config(config, data)["input_controls"]
        assert controls["day"] == day_gate
        assert controls["drop_set"] is drop_set
