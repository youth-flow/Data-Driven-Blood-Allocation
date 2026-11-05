"""S4 data, oracle and evaluation contracts."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from axid_synthetic.generator import generate_performance, make_orders, sample_demand
from axid_synthetic.oracle import assess_laws, conditional_pmf, mixture_quantiles, prepare_laws
from dat_study.data import DailySets, FoldData, collate
from dat_study.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def s4_fold(tmp_path):
    config = yaml.safe_load((ROOT / "synthetic/configs/reference.yaml").read_text(encoding="utf-8"))
    config.update(n_train=48, n_test=16, scenario="S4")
    folder = tmp_path / "s4"
    generate_performance(config, folder)
    return folder, config


def test_performance_split_and_schema(s4_fold):
    folder, config = s4_fold
    data = FoldData(folder)
    assert data.schema["schema_version"] == "synthetic_axid_v3"
    assert data.schema["generator_version"] == 4
    assert len(data.daily_split("train")) == 48
    assert len(data.daily_split("eval")) == 16
    assert data.eval_role == "test"
    assert data.schema["demand_support_upper"] == 36.5
    assert set(data.daily.split.unique()) == {"train", "test"}
    assert data.daily.D.between(0, 36.5).all()
    assert data.daily.n_orders.between(1, 8).all()


def test_x_order_inventory_and_count(s4_fold):
    folder, _ = s4_fold
    data = FoldData(folder)
    dataset = DailySets(data, "eval")
    batch = collate([dataset[0], dataset[1]])
    inventory = (batch["inventory"].numpy() - 225.0) / 180.0
    assert np.allclose(batch["day_features"][:, 2].numpy(), inventory)
    assert np.allclose(
        batch["day_features"][:, 3].numpy(),
        batch["order_mask"].sum(1).numpy() / 8,
    )
    assert np.allclose(data.daily.It.to_numpy(), 225 + 180 * data.context(data.daily)[:, 2])


def test_no_future_fields(s4_fold):
    folder, _ = s4_fold
    data = FoldData(folder)
    inputs = set(data.day_cols + data.order_num_cols + data.order_cat_cols)
    assert inputs.isdisjoint({"D", "D_new", "known_recent_D", "real_demand", "latent_U", "L", "J"})
    assert not any(column in data.daily for column in ("latent_U", "J"))


def test_s4_oracle_pmf_and_quantile(s4_fold):
    folder, config = s4_fold
    data = FoldData(folder)
    day = data.daily.decision_day.iloc[0]
    orders = data.orders[data.orders.decision_day.eq(day)]
    context = data.daily[data.daily.decision_day.eq(day)].iloc[0].to_dict()
    pmf, diagnostic = conditional_pmf(orders, context, config)
    assert np.isfinite(pmf).all() and (pmf >= 0).all()
    assert pmf.sum() == pytest.approx(1.0, abs=1e-8)
    q = mixture_quantiles(pmf, [0.9])
    assert 0 <= q[0] <= 36.5
    assert diagnostic["method"] == "gauss_hermite"


def test_s4_simulation_and_oracle_mean(s4_fold):
    folder, config = s4_fold
    data = FoldData(folder)
    day = data.daily.decision_day.iloc[0]
    orders = data.orders[data.orders.decision_day.eq(day)]
    context = data.daily[data.daily.decision_day.eq(day)].iloc[0].to_dict()
    pmf, _ = conditional_pmf(orders, context, config)
    known, new = sample_demand(orders, context, config, np.random.default_rng(7), 30000)
    demand = known.sum(1) + new
    expected = np.sum(pmf * (np.arange(len(pmf)) * 0.5 + 0.25))
    assert abs(demand.mean() - expected) < 6 * demand.std() / np.sqrt(len(demand))


def test_c0_is_constant_calibration(s4_fold):
    _, config = s4_fold
    config = {**config, "scenario": "C0"}
    context = {"scheduled_load": 0.1, "emergency_pressure": -0.2}
    orders = make_orders(np.random.default_rng(11), context, config)
    pmf, _ = conditional_pmf(orders, context, config)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-10)
    assert 0 <= mixture_quantiles(pmf, [0.9])[0] <= 18.5


def test_mean_cost_and_sl_evaluation():
    demand = np.array([1.0, 2.0, 3.0, 4.0])
    inventory = np.array([300.0, 300.0, 300.0, 300.0])
    predictions = np.array([[2.0], [2.0], [2.0], [2.0]])
    rows = evaluate(predictions, demand, inventory, demand, [0.9], [300.0])
    ending = inventory + np.maximum(predictions[:, 0] + 300.0 - inventory, 0.0) - demand
    expected = np.mean(0.9 * np.maximum(300.0 - ending, 0) + 0.1 * np.maximum(ending - 300.0, 0))
    assert rows[0]["MeanCost"] == pytest.approx(expected, abs=1e-12)
    assert rows[0]["SL_G"] == pytest.approx(0.5)
    assert "average_cost" not in rows[0]


def test_assess_laws_shapes(s4_fold):
    folder, config = s4_fold
    data = FoldData(folder)
    daily, orders = data.daily_split("eval"), data.order_split("eval")
    laws = prepare_laws(daily, orders, config)
    predictions = np.full((len(daily), 1), 10.0)
    result = assess_laws(laws, predictions, [0.9], [300.0])
    assert result["excess_pinball"].shape == (len(daily), 1)
    assert result["operational_regret"].shape == (len(daily), 1, 1)
