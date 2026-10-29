import numpy as np
import pytest
import torch
from dat_study.data import FoldData, DailySets, collate
from dat_study.evaluation import evaluate, decision
from dat_study.training import fit_predict, seed_everything
from dat_study.config import model_config
from datransformer.trainer import new_dat
from datransformer.three_stage import install_lora
from datransformer.adaptation import merge_lora
from flat_eto.quantity import independent_mixture_pmf
from dat_study.synthetic import ConditionalRemainder
from dat_study.io import read_json, save_json, sha256


def test_empty_sets_and_independent_total(synthetic_fold):
    data = FoldData(synthetic_fold)
    ds = DailySets(data, "train")
    assert len(ds) == 12 and len(ds[2]["order_num"]) == 0
    assert data.daily_split("train").D.sum() > data.order_split("train").real_demand.sum()
    batch = collate([ds[2]])
    assert batch["order_mask"].shape == (1, 1) and not batch["order_mask"].any()


def test_integrity_refuses_changed_artifact(synthetic_fold):
    with (synthetic_fold / "data.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="integrity"):
        FoldData(synthetic_fold)


def test_saa_P_zero_and_zero_denominator():
    d = np.array([1.0, 2.0, 3.0])
    inv = np.array([0.0, 0.0, 0.0])
    taus = [0.5, 0.9]
    p = np.tile([2.0, 3.0], (3, 1))
    rows = evaluate(p, d, inv, d, taus, [2.0])
    assert all(r["P"] == 0 for r in rows)
    rows = evaluate(np.ones((2, 1)), np.ones(2), np.zeros(2), np.ones(2), [0.5], [2.0])
    assert rows[0]["P"] is None and not rows[0]["P_defined"]
    assert np.array_equal(decision(np.array([3.0, 3.0]), np.array([10.0, 1.0]), 2.0), [0.0, 4.0])


def test_remainder_and_convolution():
    x = np.array([[0.0], [1.0]])
    model = ConditionalRemainder(1).fit(x, [5.0, 8.0], [2.0, 3.0])
    assert model.pools(x).ravel().tolist() == [3.0, 5.0]
    pmf = independent_mixture_pmf([0.5], [np.array([0.0, 0.0, 1.0])])
    assert np.allclose(pmf, [0.5, 0.0, 0.5])
    with pytest.raises(ValueError):
        ConditionalRemainder().fit(x, [1.0, 1.0], [2.0, 3.0])


def test_dat_empty_and_lora_merge(synthetic_fold, config):
    seed_everything(42)
    data = FoldData(synthetic_fold)
    ds = DailySets(data, "train")
    model = new_dat(data, model_config(config, "DAT")).eval()
    batch = collate([ds[0], ds[2]])
    with torch.no_grad():
        before = model(batch)["quantiles"]
    assert torch.isfinite(before).all() and (before.diff(dim=1) >= 0).all()
    install_lora(model, 4)
    model.eval()
    with torch.no_grad():
        adapted = model(batch)["quantiles"]
    assert torch.allclose(before, adapted, atol=1e-6)
    assert not any(p.requires_grad for p in model.patient_encoder.parameters())
    merge_lora(model)
    with torch.no_grad():
        after = model(batch)["quantiles"]
    assert torch.allclose(before, after, atol=1e-6)


@pytest.mark.parametrize("method", ["SAA", "wSAA", "Set-ETO", "Flat-ETO", "DAT", "DAT-3stage"])
def test_every_method_trains_and_evaluates_synthetic(synthetic_fold, config, method):
    data = FoldData(synthetic_fold)
    prediction, result, artifact = fit_predict(method, data, config)
    assert prediction.shape == (6, 5) and np.isfinite(prediction).all()
    assert (np.diff(prediction, axis=1) >= -1e-5).all()
    assert result["epochs"] in (0, 1)
    from dat_study.inference import predict_artifact

    restored = predict_artifact(
        {"method": method, "artifact": artifact, "config": config, "data_identity": data.identity}, data
    )
    assert np.allclose(prediction, restored, atol=1e-5)


def test_final_year_requires_frozen_recipe(synthetic_fold, config):
    data = FoldData(synthetic_fold)
    data.eval_role = "test"
    with pytest.raises(ValueError, match="frozen"):
        fit_predict("DAT", data, config)


def test_label_feature_rejected(synthetic_fold):
    schema = read_json(synthetic_fold / "data_feature_schema.json")
    schema["day_feature_cols"].append("D")
    save_json(synthetic_fold / "data_feature_schema.json", schema)
    manifest = read_json(synthetic_fold / "manifest.json")
    manifest["artifacts"]["data_feature_schema.json"] = sha256(synthetic_fold / "data_feature_schema.json")
    save_json(synthetic_fold / "manifest.json", manifest)
    with pytest.raises(ValueError, match="outcome"):
        FoldData(synthetic_fold)
