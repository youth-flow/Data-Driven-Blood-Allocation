"""Scientific contracts exercised only on artificial inputs and temporary outputs."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from axid_synthetic.generator import make_orders, sample_demand, generate
from axid_synthetic.oracle import conditional_pmf, mixture_quantiles, conditional_pinball, prepare_laws, assess_laws
from axid_synthetic.jobs import run_job
from dat_study.config import load_config, model_config
from dat_study.data import FoldData, DailySets, collate
from dat_study.evaluation import evaluate
from datransformer.trainer import new_dat, scalar_config
from datransformer.model import structural_parameter_counts
from set_eto.model import distribution_nll
from set_eto.distributions import distribution_quantiles

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generated_v3(tmp_path):
    settings = yaml.safe_load((ROOT / "synthetic/configs/reference.yaml").read_text(encoding="utf-8"))
    settings.update(n_train=48, n_validation=8, n_test=8)
    generate(settings, tmp_path / "new")
    return FoldData(tmp_path / "new/validation"), settings


def test_pdf_equations_and_only_three_layernorms(generated_v3):
    data, _ = generated_v3
    config = scalar_config(model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT"), 0.9)
    config["training"]["device"] = "cpu"
    config["model"]["self_attention"]["num_heads"] = 2
    model = new_dat(data, config).eval()
    assert model.self_attention.num_heads == 2 and model.cross_attention.num_heads == 4
    assert sum(isinstance(m, torch.nn.LayerNorm) for m in model.modules()) == 3
    assert not hasattr(model, "output_norm") and not hasattr(model, "patient_stack")
    batch = collate([DailySets(data, "eval")[0], DailySets(data, "eval")[1]])
    with torch.no_grad():
        e0 = model.patient_encoder(batch["order_num"], batch["order_cat"])
        normalized = model.sa_norm(e0)
        e = (
            e0
            + model.self_attention(
                normalized, normalized, normalized, key_padding_mask=~batch["order_mask"], need_weights=False
            )[0]
        )
        x = torch.cat(
            [batch["day_num"], ((batch["inventory"] - model.inventory_mean) / model.inventory_std)[:, None]], -1
        )
        u = model.day_encoder(x).unsqueeze(1)
        v = (
            u
            + model.cross_attention(
                model.query_norm(u),
                model.memory_norm(e),
                model.memory_norm(e),
                key_padding_mask=~batch["order_mask"],
                need_weights=False,
            )[0]
        )
        expected = model.head(v + model.day_ffn(v)).flatten().clamp(0, model.upper)
        torch.testing.assert_close(model(batch)["quantile"], expected)
        changed = dict(batch)
        for name in ("order_num", "order_cat", "order_mask"):
            changed[name] = batch[name].flip(1)
        torch.testing.assert_close(model(batch)["quantile"], model(changed)["quantile"])


def test_fixed_scaling_nonempty_sets_and_no_labels_in_inputs(generated_v3):
    data, settings = generated_v3
    assert data.daily.n_orders.between(1, settings["max_visible_orders"]).all()
    for split in ("train", "eval"):
        for sample in DailySets(data, split).samples:
            assert ((sample["order_num"] >= 0) & (sample["order_num"] <= 1)).all()
            assert ((sample["day_num"] >= 0) & (sample["day_num"] <= 1)).all()
    assert data.inventory_scaler.mean[0] == 225 and data.inventory_scaler.scale[0] == 180
    assert not np.allclose(data.daily.D * 2, np.round(data.daily.D * 2))
    assert data.state["scaling_source"] == "predeclared_support_not_fitted_to_samples"


@pytest.mark.parametrize("scenario", ["S1", "S3", "S4", "C0"])
def test_oracle_agrees_with_simulation_and_ignores_realized_labels(generated_v3, scenario):
    _, config = generated_v3
    config = {**config, "scenario": scenario}
    context = {"scheduled_load": 0.2, "emergency_pressure": -0.3}
    orders = make_orders(np.random.default_rng(41), context, config)
    pmf, _ = conditional_pmf(orders, context, config)
    assert np.isclose(pmf.sum(), 1) and (pmf >= 0).all()
    q = mixture_quantiles(pmf, [0.5, 0.9])
    known, new = sample_demand(orders, context, config, np.random.default_rng(76), 40000)
    demand = known.sum(1) + new
    expected = np.sum(pmf * (np.arange(len(pmf)) * 0.5 + 0.25))
    assert abs(demand.mean() - expected) < 6 * demand.std() / np.sqrt(len(demand))
    assert abs((demand <= q[1]).mean() - 0.9) < 0.015
    orders["real_demand"] = 1e8
    poisoned = {**context, "D": -999, "D_new": 999, "latent_shock": 100}
    np.testing.assert_allclose(conditional_pmf(orders, poisoned, config)[0], pmf)


def test_operational_bridge_uses_expected_cost_not_realized_perfect_information(generated_v3):
    data, config = generated_v3
    daily, orders = data.daily_split("eval"), data.order_split("eval")
    law = prepare_laws(daily, orders, config)
    prediction = np.random.default_rng(9).uniform(0, 36.5, (len(daily), 2))
    result = assess_laws(law, prediction, [0.5, 0.9], [200, 280, 340, 450])
    assert (result["excess_pinball"] >= -1e-9).all()
    assert (result["operational_regret"] >= -1e-9).all()
    assert (result["operational_regret"] <= result["excess_pinball"][..., None] + 1e-9).all()
    for i, pmf in enumerate(law["pmfs"]):
        risk = conditional_pinball(pmf, result["oracle_quantiles"][i, 1], 0.9)
        assert risk == pytest.approx(result["oracle_risk"][i, 1])


def test_scalar_quantile_crossing_is_reported_without_rearranging():
    predictions = np.array([[3.0, 1.0], [1.0, 2.0]])
    original = predictions.copy()
    rows = evaluate(predictions, [1.0, 3.0], [1.0, 2.0], [1.0, 2.0, 3.0], [0.5, 0.9], [1.0])
    assert all(row["crossing_share"] == 0.5 for row in rows)
    np.testing.assert_array_equal(predictions, original)


def test_smoothed_negative_binomial_is_a_density_not_rounded_counts():
    parameters = {"mean_count": torch.tensor([4.0, 4.0]), "dispersion": torch.tensor([2.0, 2.0])}
    d = torch.tensor([1.1, 1.4])
    with pytest.raises(ValueError, match="round"):
        distribution_nll("NegativeBinomial", parameters, d)
    expected = -torch.distributions.NegativeBinomial(
        total_count=parameters["dispersion"], logits=torch.log(parameters["mean_count"] / parameters["dispersion"])
    ).log_prob(torch.floor(2 * d)).mean() - np.log(2)
    assert distribution_nll("SmoothedNegativeBinomial", parameters, d) == pytest.approx(float(expected))
    q = distribution_quantiles("SmoothedNegativeBinomial", {k: v.numpy() for k, v in parameters.items()}, [0.5, 0.9])
    assert (np.diff(q, axis=1) > 0).all()


def test_ablation_reports_inactive_parameters(generated_v3):
    data, _ = generated_v3
    cfg = scalar_config(model_config(load_config(ROOT / "configs/synthetic.yaml"), "DAT"), 0.9)
    cfg["training"]["device"] = "cpu"
    full = structural_parameter_counts(new_dat(data, cfg))
    assert full["allocated_parameters"] == full["structurally_active_parameters"]
    for intervention in ("no_self_attention", "mean_pool", "sum_pool"):
        selected = deepcopy(cfg)
        selected["ablation"] = intervention
        counts = structural_parameter_counts(new_dat(data, selected))
        assert counts["allocated_parameters"] == full["allocated_parameters"]
        assert counts["structurally_active_parameters"] < full["structurally_active_parameters"]


def test_resume_verifies_artifacts_and_preserves_interrupted_attempt(tmp_path):
    calls = []

    def action(folder):
        calls.append(1)
        (folder / "weights.txt").write_text("artificial", encoding="utf-8")
        return {"rows": [1]}

    result = run_job(tmp_path, "test", {"seed": 1}, action)
    assert run_job(tmp_path, "test", {"seed": 1}, action, True) == result and len(calls) == 1
    with pytest.raises(ValueError, match="recipe"):
        run_job(tmp_path, "test", {"seed": 2}, action, True)
    (tmp_path / "test/attempt_0000/weights.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        run_job(tmp_path, "test", {"seed": 1}, action, True)
    interrupted = tmp_path / "interrupted/attempt_0000"
    interrupted.mkdir(parents=True)
    run_job(tmp_path, "interrupted", {"seed": 1}, action, True)
    assert interrupted.exists() and (tmp_path / "interrupted/attempt_0001/result.json").exists()
