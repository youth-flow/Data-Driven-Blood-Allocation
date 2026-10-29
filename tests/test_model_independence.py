"""Verify sibling models can be unavailable during training and predictor reloading."""

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from dat_study.config import load_config, model_config


@pytest.mark.parametrize("method", ["SAA", "wSAA", "Flat-ETO", "Set-ETO", "DAT", "DAT-3stage"])
def test_selected_model_never_imports_siblings(synthetic_fold, config, tmp_path, method):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    source = r"""
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np

method, fold, config_path = sys.argv[1:]
packages = {'SAA': 'saa_experiment', 'wSAA': 'wsaa_experiment', 'Flat-ETO': 'flat_eto',
            'Set-ETO': 'set_eto', 'DAT': 'datransformer', 'DAT-3stage': 'datransformer'}
forbidden = set(packages.values()) - {packages[method]}

class BlockSiblingModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise ImportError('Sibling model imported: ' + fullname)

sys.meta_path.insert(0, BlockSiblingModels())
from dat_study.data import FoldData
from dat_study.training import fit_predict
from dat_study.inference import predict_artifact
config = json.loads(Path(config_path).read_text(encoding='utf-8'))
data = FoldData(fold)
prediction, result, artifact = fit_predict(method, data, config)
reloaded = predict_artifact({'method': method, 'artifact': artifact, 'config': config,
                            'data_identity': data.identity}, data)
assert np.allclose(prediction, reloaded, atol=1e-5)
assert not any(name.split('.')[0] in forbidden for name in sys.modules)
"""
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", source, method, str(synthetic_fold), str(config_file)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_configs_share_data_but_own_parameters():
    root = Path(__file__).resolve().parents[1]
    experiment = load_config(root / "configs/experiment.yaml")
    for method in ["DAT", "SAA", "wSAA", "Flat-ETO", "Set-ETO"]:
        standalone = load_config(root / method / "configs/main.yaml")
        selected = model_config(experiment, method)
        assert selected == standalone
        assert Path(selected["data_root"]) == root / "data/processed"
        assert "model_configs" not in selected
    assert "model" not in experiment and "training" not in experiment


def test_wrong_model_config_rejected():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="different method"):
        model_config(load_config(root / "DAT/configs/main.yaml"), "Set-ETO")
