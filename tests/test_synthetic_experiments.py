"""Synthetic experiments never need real tables; these tests cover leakage and policy contracts."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from axid_synthetic.generator import generate, order_parameters, oracle_quantiles, NUM_BASIC, NUM_CLINICAL, CAT_BASIC
from dat_study.config import load_config, model_config, smoke_config
from dat_study.data import FoldData, DailySets, collate
from dat_study.training import fit_predict
from dat_study.inference import predict_artifact
from dat_study.tabular import TabularFeatures
from datransformer.trainer import new_dat
from datransformer.experiments import ABLATIONS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generated(tmp_path):
    settings = yaml.safe_load((ROOT / "synthetic/configs/reference.yaml").read_text(encoding="utf-8"))
    settings.update(n_train=32, n_validation=8, n_test=8)
    folder = tmp_path / "generated"
    generate(settings, folder)
    return folder, settings


@pytest.fixture
def synthetic_config():
    config = smoke_config(load_config(ROOT / "configs/synthetic.yaml"))
    config["training"] = {"device": "cpu", "threads": 1}
    return config


def test_generator_contract_and_reproducibility(generated, tmp_path):
    folder, settings = generated
    other = tmp_path / "repeat"
    generate(settings, other)
    a, b = FoldData(folder / "validation"), FoldData(other / "validation")
    pd.testing.assert_frame_equal(a.daily, b.daily)
    pd.testing.assert_frame_equal(a.orders, b.orders)
    assert a.identity == b.identity
    assert a.schema["schema_version"] == "synthetic_axid_v2"
    daily = a.daily.set_index("decision_day")
    known = a.orders.groupby("decision_day").real_demand.sum().reindex(daily.index, fill_value=0)
    assert np.allclose(daily.D, known + daily.D_new)
    assert (daily.D <= a.schema["demand_support_upper"]).all()
    assert set(a.day_cols + a.order_num_cols + a.order_cat_cols).isdisjoint(
        {"D", "D_new", "known_recent_D", "latent_shock"}
    )
    assert not any("latent_shock" in frame for frame in (a.daily, a.orders))
    test = FoldData(folder / "test")
    assert len(test.daily_split("train")) == settings["n_train"] + settings["n_validation"]
    assert test.eval_role == "test"
    with pytest.raises(ValueError, match="already exists"):
        generate(settings, folder)


def test_scenarios_share_observables_and_test_size_cannot_change_training(generated, tmp_path):
    folder, settings = generated
    changed = {**settings, "scenario": "S4", "n_test": 13}
    generate(changed, tmp_path / "s4")
    a, b = FoldData(folder / "validation"), FoldData(tmp_path / "s4/validation")
    pd.testing.assert_frame_equal(
        a.orders[NUM_BASIC + NUM_CLINICAL + CAT_BASIC], b.orders[NUM_BASIC + NUM_CLINICAL + CAT_BASIC]
    )
    pd.testing.assert_frame_equal(a.daily[[*a.day_cols, "It"]], b.daily[[*b.day_cols, "It"]])
    assert a.state == b.state


def test_learning_curve_keeps_holdouts_identical_and_training_nested(generated, tmp_path):
    folder, settings = generated
    larger = {**settings, "n_train": settings["n_train"] + 17}
    generate(larger, tmp_path / "larger")
    small_dev, big_dev = FoldData(folder / "validation"), FoldData(tmp_path / "larger/validation")
    pd.testing.assert_frame_equal(
        small_dev.daily_split("eval").reset_index(drop=True), big_dev.daily_split("eval").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        small_dev.order_split("eval").reset_index(drop=True), big_dev.order_split("eval").reset_index(drop=True)
    )
    small_train = small_dev.daily_split("train")
    overlap = big_dev.daily_split("train").query("decision_day >= @small_train.decision_day.min()")
    pd.testing.assert_frame_equal(small_train.reset_index(drop=True), overlap.reset_index(drop=True))
    small_orders = small_dev.order_split("train")
    overlap_orders = big_dev.order_split("train")
    overlap_orders = overlap_orders[overlap_orders.decision_day.isin(small_train.decision_day)]
    pd.testing.assert_frame_equal(small_orders.reset_index(drop=True), overlap_orders.reset_index(drop=True))
    small_test, big_test = FoldData(folder / "test"), FoldData(tmp_path / "larger/test")
    pd.testing.assert_frame_equal(
        small_test.daily_split("eval").reset_index(drop=True), big_test.daily_split("eval").reset_index(drop=True)
    )


def test_plain_dat_has_no_retired_training_route():
    from dat_study.config import METHODS
    from dat_study.registry import PACKAGES
    from importlib.util import find_spec

    assert "DAT-3stage" not in METHODS and "DAT-3stage" not in PACKAGES
    assert "three_stage" not in load_config(ROOT / "DAT/configs/main.yaml")
    assert find_spec("datransformer.three_stage") is None
    assert find_spec("datransformer.adaptation") is None


def test_order_mechanism_permutation_invariant_and_oracle_separate(generated):
    folder, settings = generated
    data = FoldData(folder / "validation")
    day = data.orders.decision_day.iloc[0]
    orders = data.orders[data.orders.decision_day.eq(day)].reset_index(drop=True)
    context = data.daily[data.daily.decision_day.eq(day)].iloc[0].to_dict()
    settings = {**settings, "scenario": "S3"}
    forward = order_parameters(orders, context, settings)
    reverse = order_parameters(orders.iloc[::-1], context, settings)
    for a, b in zip(forward, reverse):
        assert np.allclose(a, b[::-1])
    q = oracle_quantiles(orders, context, settings, [0.5, 0.9], repeats=500, seed=1)
    orders["real_demand"] = 10000
    context["D"] = 10000
    assert np.array_equal(q, oracle_quantiles(orders, context, settings, [0.5, 0.9], repeats=500, seed=1))


@pytest.mark.parametrize("variant", ABLATIONS)
def test_dat_ablation_train_reload(generated, synthetic_config, variant):
    data = FoldData(generated[0] / "validation")
    cfg = model_config(synthetic_config, "DAT")
    cfg["ablation"] = variant
    prediction, _, artifact = fit_predict("DAT", data, cfg)
    restored = predict_artifact(
        {"method": "DAT", "artifact": artifact, "config": cfg, "data_identity": data.identity}, data
    )
    assert prediction.shape == (8, 5)
    assert np.isfinite(prediction).all()
    assert np.allclose(prediction, restored, atol=1e-5)


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3", "L4"])
def test_information_masks_remove_hidden_inputs_with_fixed_parameters(generated, synthetic_config, level):
    data = FoldData(generated[0] / "validation")
    cfg = model_config(synthetic_config, "DAT")
    full = new_dat(data, cfg)
    cfg["information"] = level
    model = new_dat(data, cfg).eval()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in full.parameters())
    dataset = DailySets(data, "eval")
    batch = collate([dataset[0], dataset[1]])
    changed = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    changed["day_num"][:, ~model.day_gate] += 100
    changed["order_num"][:, :, ~model.order_num_gate] += 100
    if model.drop_set:
        changed["order_num"] += 100
        changed["order_mask"] = ~changed["order_mask"]
    with torch.no_grad():
        assert torch.allclose(model(batch)["quantiles"], model(changed)["quantiles"], atol=1e-6)
    prediction, _, artifact = fit_predict("DAT", data, cfg)
    restored = predict_artifact(
        {"method": "DAT", "artifact": artifact, "config": cfg, "data_identity": data.identity}, data
    )
    assert np.allclose(prediction, restored, atol=1e-5)


def test_enhanced_features_ignore_outcomes(generated):
    data = FoldData(generated[0] / "validation")
    features = TabularFeatures("enhanced").fit(data)
    before = features.transform(data, data.daily_split("eval"))
    data.orders["real_demand"] = 12345
    data.daily["D"] = 12345
    after = features.transform(data, data.daily_split("eval"))
    assert np.array_equal(before, after)


@pytest.mark.parametrize("method", ["LQR", "GBQR", "Rolling-SAA", "DeepQR", "wSAA", "Set-ETO", "DAT"])
def test_frozen_selection_and_test_outcome_separation(generated, synthetic_config, method):
    cfg = model_config(synthetic_config, method)
    validation = FoldData(generated[0] / "validation")
    _, selected, _ = fit_predict(method, validation, cfg)
    test = FoldData(generated[0] / "test")
    recipe = {"epochs": selected["epochs"]}
    if "selection" in selected:
        recipe["selection"] = selected["selection"]
    prediction, result, _ = fit_predict(method, test, cfg, recipe)
    test.daily.loc[test.daily["split"].eq("test"), "D"] += 1000
    altered, _, _ = fit_predict(method, test, cfg, recipe)
    if method == "Rolling-SAA":
        assert np.array_equal(prediction[0], altered[0])
    else:
        assert np.allclose(prediction, altered, atol=1e-6)
    assert all("validation_mean_P" not in row for row in result.get("history", []))


def test_rolling_does_not_see_current_or_future_labels():
    from rolling_saa.estimator import rolling_quantiles

    history = pd.DataFrame({"decision_day": pd.date_range("2000-01-01", periods=4), "D": [1.0, 2.0, 3.0, 4.0]})
    evaluation = pd.DataFrame({"decision_day": pd.date_range("2000-01-05", periods=3), "D": [2.0, 2.0, 2.0]})
    original = rolling_quantiles(history, evaluation, [0.5, 0.9], 28)
    changed = evaluation.copy()
    changed.loc[1:, "D"] = 10000
    altered = rolling_quantiles(history, changed, [0.5, 0.9], 28)
    assert np.array_equal(original[:2], altered[:2])


def test_suite_rejects_wrong_family_group(tmp_path):
    from axid_synthetic.study import main

    with pytest.raises(ValueError, match="Group"):
        main(["--family", "theory", "--group", "comparison", "--name", "no_recipe"])


def test_synthetic_generated_files_cannot_be_committed():
    from runpy import run_path

    guard = run_path(str(ROOT / "scripts/check_git_privacy.py"))
    assert guard["allowed"]("synthetic/src/axid_synthetic/generator.py")
    assert not guard["allowed"]("synthetic/generated/reference/generation.yaml")
    assert not guard["allowed"]("synthetic/generated/reference/validation/data.csv")
    assert guard["allowed"]("QR/DeepQR/src/deep_qr/model.py")


def test_historical_receipt_uses_actions_and_rejects_synthetic(generated, synthetic_fold):
    from historical_receipt.policy import receipt_allocations
    from dat_study.evaluation import evaluate_allocations

    with pytest.raises(ValueError, match="historical"):
        receipt_allocations(FoldData(generated[0] / "validation"), [0.5, 0.9])
    data = FoldData(synthetic_fold)  # Fabricated fixture with the retained real-derived schema.
    data.daily["q_hist_receipt"] = 7.0
    q = receipt_allocations(data, [0.5, 0.9])
    daily = data.daily_split("eval")
    rows = evaluate_allocations(q, daily.D, daily.It, data.daily_split("train").D, [0.5, 0.9], [260, 300])
    assert all(row["mean_q"] == 7.0 and row["pinball"] is None and row["coverage"] is None for row in rows)
    deviation = daily.It.to_numpy() + 7 - daily.D.to_numpy() - 260
    assert rows[0]["average_cost"] == pytest.approx(np.abs(deviation).mean() / 2)
    data.daily.loc[data.daily["split"].eq("validation"), "q_hist_receipt"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        receipt_allocations(data, [0.5, 0.9])


@pytest.mark.parametrize("family", ["Normal", "SmoothedNegativeBinomial", "HurdleGamma"])
def test_set_eto_distribution_presets_train_reload(generated, synthetic_config, family):
    data = FoldData(generated[0] / "validation")
    cfg = model_config(synthetic_config, "Set-ETO")
    cfg["set_eto"].pop("distributions", None)
    cfg["set_eto"]["distribution"] = family
    values, _, artifact = fit_predict("Set-ETO", data, cfg)
    reloaded = predict_artifact(
        {"method": "Set-ETO", "artifact": artifact, "config": cfg, "data_identity": data.identity}, data
    )
    assert np.isfinite(values).all() and np.allclose(values, reloaded)
    assert (np.diff(values, axis=1) >= -1e-5).all()


def test_historical_runner_writes_actions_not_quantiles(synthetic_fold, config, tmp_path):
    from dat_study.runner import run
    from dat_study.io import read_json, save_json, sha256
    from dat_study.inference import load_local_prediction

    daily_path = synthetic_fold / "daily_features.csv"
    daily = pd.read_csv(daily_path)
    daily["q_hist_receipt"] = 7.0
    daily.to_csv(daily_path, index=False)
    manifest = read_json(synthetic_fold / "manifest.json")
    manifest["artifacts"]["daily_features.csv"] = sha256(daily_path)
    save_json(synthetic_fold / "manifest.json", manifest)
    config["data_root"], config["output_root"] = str(synthetic_fold.parent), str(tmp_path / "outputs")
    result = run(config, synthetic_fold.name, ["Historical-Receipt"], "receipt_check", smoke=True)
    assert (result / "Historical-Receipt/daily_allocations.csv").is_file()
    assert not (result / "Historical-Receipt/daily_quantiles.csv").exists()
    _, values, _ = load_local_prediction(result, "Historical-Receipt", synthetic_fold)
    assert (values == 7).all()
