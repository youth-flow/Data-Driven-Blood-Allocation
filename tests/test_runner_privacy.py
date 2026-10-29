from pathlib import Path
import importlib.util
import subprocess
import shutil
import numpy as np
import pandas as pd
import pytest
from dat_study.runner import run
from dat_study.inference import load_local_prediction
from dat_study.io import read_json
from dat_study.training import fit_predict
from dat_study.data import FoldData


def test_completed_run_reload_and_refuse_overwrite(synthetic_fold, config, tmp_path):
    config["data_root"] = str(synthetic_fold.parent)
    config["output_root"] = str(tmp_path / "outputs")
    methods = ["SAA", "wSAA", "Set-ETO", "Flat-ETO", "DAT", "DAT-3stage"]
    root = run(config, "val2019", methods, "roundtrip", smoke=True)
    assert read_json(root / "manifest.json")["complete"]
    for method in methods:
        _, actual, _ = load_local_prediction(root, method, synthetic_fold)
        expected = pd.read_csv(root / method / "daily_quantiles.csv").iloc[:, 1:].to_numpy()
        assert np.allclose(actual, expected, atol=1e-5)
    with pytest.raises(ValueError, match="exists"):
        run(config, "val2019", ["SAA"], "roundtrip", smoke=True)


def test_final_refit_does_not_select_on_test_labels(synthetic_fold, config):
    data = FoldData(synthetic_fold)
    data.eval_role = "test"
    data.daily.loc[data.daily["split"].eq("validation"), "split"] = "test"
    first, _, _ = fit_predict("DAT", data, config, recipe={"epochs": 1})
    data.daily.loc[data.daily["split"].eq("test"), "D"] += 10000
    second, _, _ = fit_predict("DAT", data, config, recipe={"epochs": 1})
    assert np.array_equal(first, second)


def test_git_privacy_paths():
    path = Path(__file__).resolve().parents[1] / "scripts/check_git_privacy.py"
    spec = importlib.util.spec_from_file_location("privacy_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.allowed("src/dat_study/data.py")
    assert module.allowed("tests/test_contracts.py")
    for model in ("DAT", "SAA", "wSAA", "Flat-ETO", "Set-ETO"):
        assert module.allowed(f"{model}/configs/main.yaml")
        assert module.allowed(f"{model}/scripts/run_experiment.py")
        assert not module.allowed(f"{model}/results/model.pt")
        assert not module.allowed(f"{model}/src/private.csv")
    for name in (
        "data/morning/.private_order_key_salt",
        "data/申请表明细.csv",
        "outputs/model.pt",
        "docs/private.csv",
        "参考文献/report.pdf",
        "src/.env",
    ):
        assert not module.allowed(name)


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is not installed")
def test_force_staged_data_blocked(tmp_path):
    import sys

    guard = Path(__file__).resolve().parents[1] / "scripts/check_git_privacy.py"
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/fabricated.csv").write_text("artificial,value\nexample,1\n")
    subprocess.run(["git", "add", "-f", "data/fabricated.csv"], cwd=tmp_path, check=True)
    result = subprocess.run([sys.executable, str(guard)], cwd=tmp_path, capture_output=True)
    assert result.returncode == 1
