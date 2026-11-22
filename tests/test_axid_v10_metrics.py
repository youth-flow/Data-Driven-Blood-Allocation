"""Single-G operational metrics and conditional-Oracle infrastructure for v10."""

import numpy as np

from axid_synthetic.metrics_v10 import (
    COMMON_G,
    daily_outcomes,
    evaluate_single_method,
    paired_bootstrap,
    paired_difference_bootstrap,
    perfect_foresight_outcomes,
    prescriptiveness,
)
from axid_synthetic.oracle_v10 import ObservedState, conditional_oracle_quantile, conditional_oracle_quantiles


def test_common_g_cost_service_and_nonnegative_pf():
    demand = np.array([10.0, 20.0, 30.0])
    inventory = np.array([250.0, 300.0, 240.0])
    prediction = np.array([20.0, 20.0, 20.0])
    outcome = daily_outcomes(prediction, demand, inventory)
    assert COMMON_G == 256.0
    assert np.allclose(outcome["replenishment"], [26.0, 0.0, 36.0])
    assert np.allclose(outcome["ending_inventory"], [266.0, 280.0, 246.0])
    assert np.allclose(outcome["cost"], [1.0, 2.4, 9.0])
    assert outcome["service"].tolist() == [True, True, False]

    pf = perfect_foresight_outcomes(demand, inventory)
    # Day two starts above G + D and cannot place a negative order.
    assert np.allclose(pf["cost"], [0.0, 2.4, 0.0])


def test_two_p_and_complete_evaluation():
    defined = prescriptiveness(method_cost=2.0, saa_cost=4.0, reference_cost=1.0)
    assert defined["defined"]
    assert np.isclose(defined["value"], 2.0 / 3.0)
    undefined = prescriptiveness(method_cost=2.0, saa_cost=1.0, reference_cost=1.0)
    assert not undefined["defined"]
    assert np.isnan(undefined["value"])

    result = evaluate_single_method(
        prediction=np.array([10.0, 20.0, 30.0, 40.0]),
        demand=np.array([8.0, 25.0, 25.0, 45.0]),
        inventory=np.array([230.0, 260.0, 250.0, 280.0]),
        training_demand=np.arange(1.0, 21.0),
        oracle_prediction=np.array([8.0, 25.0, 25.0, 45.0]),
    )
    assert set(("MeanCost", "SL_G", "P_PF", "P_Oracle")) <= set(result)
    assert len(result["daily"]["method"]["cost"]) == 4


def test_paired_bootstrap_is_deterministic_and_paired():
    n = 80
    x = np.linspace(0.0, 1.0, n)
    costs = {"DAT": 2.0 + x, "DeepQR": 2.2 + x}
    services = {"DAT": x >= 0.1, "DeepQR": x >= 0.2}
    kwargs = {
        "saa_cost": 3.0 + x,
        "pf_cost": 1.0 + x,
        "oracle_cost": 1.5 + x,
        "replications": 40,
        "seed": 123,
        "chunk_size": 7,
    }
    first = paired_bootstrap(costs, services, **kwargs)
    second = paired_bootstrap(costs, services, **kwargs)
    assert first.equals(second)
    assert len(first) == 2 * 4
    dat = first[first.method.eq("DAT")].set_index("metric")
    assert np.isclose(dat.loc["MeanCost", "estimate"], costs["DAT"].mean())
    assert np.isclose(dat.loc["P_PF", "estimate"], 0.5)
    assert np.isclose(dat.loc["P_Oracle", "estimate"], 2.0 / 3.0)
    assert (first.ci_lower <= first.ci_upper).all()
    difference = paired_difference_bootstrap(costs["DeepQR"], costs["DAT"], replications=40, seed=4)
    assert np.isclose(difference["estimate"], 0.2)
    assert difference["ci_lower"] <= difference["estimate"] <= difference["ci_upper"]


def _visible_only_simulator(state, n_draws, rng):
    # The callback has no realized A+, shock, Y, or D field available to inspect.
    return np.clip(state.cp * 10.0 + rng.normal(0.0, 1.0, n_draws), 0.0, state.inventory)


def test_conditional_oracle_uses_only_observed_state_and_is_reproducible():
    state = ObservedState(np.array([[2.0, 0.8, 60.0], [4.0, 0.2, 110.0]]), 0.7, 0.4, 256.0)
    first = conditional_oracle_quantile(
        state, simulator=_visible_only_simulator, n_draws=256, batch_size=64, seed=91
    )
    second = conditional_oracle_quantile(
        state, simulator=_visible_only_simulator, n_draws=256, batch_size=64, seed=91
    )
    assert first == second
    assert 6.0 < first.quantile < 9.0
    assert len(first.block_quantiles) == 4

    quantiles, estimates = conditional_oracle_quantiles(
        [state, ObservedState(np.array([[1.0, 0.1, 130.0]]), 0.2, 0.5, 300.0)],
        simulator=_visible_only_simulator,
        n_draws=128,
        batch_size=64,
        seed=22,
    )
    assert quantiles.shape == (2,)
    assert len(estimates) == 2
    assert quantiles[0] > quantiles[1]


def test_conditional_oracle_generator_adapter():
    old = np.tile(np.array([[2.0, 0.5, 90.0]]), (58, 1))
    state = ObservedState(old, 0.6, 0.4, 256.0)
    estimate = conditional_oracle_quantile(state, n_draws=16, batch_size=8, seed=8)
    assert 0.0 <= estimate.quantile <= state.inventory
    assert estimate.n_draws == 16
