"""AXID-DC v10 marginal, MDCEV, Oracle-draw and compact-file tests."""

import json
from pathlib import Path

import numpy as np
import yaml

from axid_synthetic.generator_v10 import (
    COPULA_CORRELATION,
    DEMAND_UPPER,
    HB_LOWER,
    HB_UPPER,
    MAX_NEW_ORDERS,
    MAX_VISIBLE_ORDERS,
    TARGET_INVENTORY,
    conditional_new_counts,
    day_sample,
    draw_daily_context,
    draw_profiles,
    generate_performance,
    inventory_cdf,
    inventory_ppf,
    mdcev_allocate,
    mdcev_demand_batch,
    mechanism_parameters,
    simulate_conditional_demand,
)


ROOT = Path(__file__).resolve().parents[1]


def test_copula_support_and_inventory_inverse():
    rng = np.random.default_rng(12)
    n_old, n_new, inventory, cp, ce = draw_daily_context(rng, size=4000)
    assert np.linalg.eigvalsh(COPULA_CORRELATION).min() > 0
    assert np.all((1 <= n_old) & (n_old <= MAX_VISIBLE_ORDERS))
    assert np.all((0 <= n_new) & (n_new <= MAX_NEW_ORDERS))
    assert np.all((0 < inventory) & (inventory <= DEMAND_UPPER))
    assert np.all((0 <= cp) & (cp <= 1))
    assert np.all((0 <= ce) & (ce <= 1))
    probability = np.linspace(0.01, 0.99, 51)
    np.testing.assert_allclose(inventory_cdf(inventory_ppf(probability)), probability, atol=2e-10)
    assert 50 < n_old.mean() < 65
    assert 24 < n_new.mean() < 34


def test_profiles_mapping_and_mdcev_kkt():
    rng = np.random.default_rng(23)
    old = draw_profiles(rng, 2000, "old")
    new = draw_profiles(rng, 2000, "new")
    assert set(np.unique(old[:, 0])).issubset({1, 2, 3, 4, 5, 6, 8, 10})
    assert np.all((0 <= old[:, 1]) & (old[:, 1] <= 1))
    assert np.all((HB_LOWER <= old[:, 2]) & (old[:, 2] <= HB_UPPER))
    # Main scenario does not manufacture a high-risk hidden-arrival advantage.
    assert abs(new[:, 1].mean() - old[:, 1].mean()) < 0.03
    gamma, phi, ell, deficit = mechanism_parameters(old[:70], 0.7, 0.6, 0.0, np.zeros(70))
    assert np.all((0.39 <= gamma) & (gamma <= 0.87))
    assert np.all(phi > 0) and np.all((0 <= ell) & (ell <= 1)) and np.all((0 <= deficit) & (deficit <= 4))
    patient_cap = old[:70, 0]
    doses, reserve, multiplier = mdcev_allocate(gamma, phi, 270.0, cap=patient_cap)
    assert np.all((0 <= doses) & (doses <= patient_cap))
    assert multiplier > 0 and reserve > 0
    assert abs(reserve + doses.sum() - 270.0) < 1e-6
    active = (doses > 1e-8) & (doses < patient_cap - 1e-8)
    np.testing.assert_allclose(doses[active], gamma[active] * (phi[active] / multiplier - 1), atol=1e-8)

    gamma_batch = np.zeros((2, 70))
    phi_batch = np.ones((2, 70))
    mask = np.zeros((2, 70), dtype=bool)
    gamma_batch[0], phi_batch[0], mask[0] = gamma, phi, True
    gamma_batch[1, :35], phi_batch[1, :35], mask[1, :35] = gamma[:35], phi[:35], True
    caps = np.zeros_like(gamma_batch)
    caps[0], caps[1, :35] = patient_cap, patient_cap[:35]
    batched = mdcev_demand_batch(
        gamma_batch, phi_batch, mask, np.array([270.0, 240.0]), cap=caps
    )
    scalar_second = mdcev_allocate(gamma[:35], phi[:35], 240.0, cap=patient_cap[:35])[0].sum()
    np.testing.assert_allclose(batched, [doses.sum(), scalar_second], atol=2e-6)


def test_daily_and_conditional_demand_support():
    rng = np.random.default_rng(34)
    for _ in range(10):
        sample = day_sample(rng)
        assert 0 <= sample["demand"] <= sample["inventory"] <= DEMAND_UPPER
        assert len(sample["old"]) == sample["n_old"]
        assert len(sample["new"]) == sample["n_new"]
        patients = np.vstack([sample["old"], sample["new"]])
        assert np.all(sample["Y"] <= patients[:, 0] + 1e-10)
        assert abs(sample["R"] + sample["demand"] - sample["inventory"]) < 1e-6
    old = draw_profiles(rng, 25, "old")
    counts = conditional_new_counts(rng, 25, 0.7, 0.4, 280.0, 100)
    assert counts.shape == (100,) and np.all((0 <= counts) & (counts <= MAX_NEW_ORDERS))
    draws = simulate_conditional_demand(old, 0.7, 0.4, 280.0, 24, rng)
    assert draws.shape == (24,) and np.isfinite(draws).all()
    assert np.all((0 <= draws) & (draws <= 280.0))


def test_compact_fold_export(tmp_path):
    config = yaml.safe_load((ROOT / "synthetic/configs/reference_v10.yaml").read_text(encoding="utf-8"))
    config.update(data_seed=123, n_train=8, n_test=4)
    output = tmp_path / "fold"
    result = generate_performance(config, output)
    assert result["target_inventory"] == TARGET_INVENTORY
    schema = json.loads((output / "data_feature_schema.json").read_text(encoding="utf-8"))
    split = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    daily = np.load(output / "daily.npz")
    patients = np.load(output / "patients.npy", mmap_mode="r")
    labels = np.load(output / "patient_labels.npy", mmap_mode="r")
    offsets = np.load(output / "patient_offsets.npy", mmap_mode="r")
    assert schema["schema_version"] == "synthetic_axid_dc_v10"
    assert schema["compact_format"] == "axid_v10_csr_v1"
    assert schema["max_orders"] == MAX_VISIBLE_ORDERS
    assert schema["day_feature_cols"][-1] == "hb_sum"
    assert "hb_shortage_sum" not in schema["day_feature_cols"]
    assert split["train_start"] == 0 and split["train_end"] == 7
    assert split["test_start"] == 38 and split["test_end"] == 41
    assert len(daily["D"]) == 12 and offsets.shape == (13,)
    assert patients.shape == (int(offsets[-1]), 3) and labels.shape == (int(offsets[-1]),)
    assert all(name in daily for name in schema["psi_summary_cols"])
    expected_hb_sum = np.array(
        [((patients[offsets[j] : offsets[j + 1], 2] - 42.0) / 128.0).sum() for j in range(12)]
    )
    np.testing.assert_allclose(daily["hb_sum"], expected_hb_sum, rtol=2e-6, atol=2e-6)
