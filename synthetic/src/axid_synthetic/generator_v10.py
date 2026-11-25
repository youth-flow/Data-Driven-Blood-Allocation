"""AXID-DC v10: copula-calibrated patient arrivals and joint MDCEV allocation.

The blood-centre order is intentionally absent: demand is generated from the
morning state and the hospital's initial inventory, before the evaluated order.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import nbinom, norm, truncnorm


SCHEMA = "synthetic_axid_dc_v10"
GENERATOR = "AXID-DC v10 MDCEV"
GENERATOR_VERSION = 10
ORDER_NUM = ["u", "r", "hb"]
DAY = [
    "planned_load",
    "emergency_load",
    "n_registered",
    "request_use_sum",
    "risk_sum",
    "hb_sum",
]
PSI_COLUMNS = [
    "psi_ur",
    "psi_u_hb",
    "psi_r_hb",
    "psi_u_min",
    "psi_u_max",
    "psi_r_min",
    "psi_r_max",
    "psi_hb_min",
    "psi_hb_max",
]

TARGET_INVENTORY = 280.0
KAPPA = 1.0
DOSE_CAP = 10.0
INVENTORY_LOWER = 0.0
INVENTORY_UPPER = 520.0
DEMAND_UPPER = INVENTORY_UPPER
MAX_VISIBLE_ORDERS = 138
MAX_NEW_ORDERS = 73

N_OLD_R, N_OLD_P = 6.704872412573668, 0.10530530406692609
N_NEW_R, N_NEW_P = 4.676776468869735, 0.14289110003578717
COPULA_ORDER = ("n_old", "n_new", "inventory", "planned_load", "emergency_load")
COPULA_CORRELATION = np.array(
    [
        [1.000, 0.220, -0.194, 0.891, 0.520],
        [0.220, 1.000, -0.112, 0.147, 0.268],
        [-0.194, -0.112, 1.000, -0.048, -0.439],
        [0.891, 0.147, -0.048, 1.000, 0.240],
        [0.520, 0.268, -0.439, 0.240, 1.000],
    ],
    dtype=float,
)

U_VALUES = np.array([1, 2, 3, 4, 5, 6, 8, 10], dtype=float)
# Three zero/missing records are assigned to the minimum positive request;
# the 17 sparse 7/9-unit records are pooled with their adjacent upper bins.
U_PROBABILITIES = np.array([65, 11766, 4404, 33664, 200, 1423, 1227, 556], dtype=float)
U_PROBABILITIES /= U_PROBABILITIES.sum()
RISK_VALUES = np.arange(14, dtype=float) / 13.0
RISK_PROBABILITIES = np.array(
    [
        0.295695, 0.191127, 0.149705, 0.127568, 0.012982, 0.008630, 0.065153,
        0.052734, 0.057030, 0.018291, 0.010168, 0.008217, 0.002007, 0.000694,
    ],
    dtype=float,
)
RISK_PROBABILITIES /= RISK_PROBABILITIES.sum()
HB_COMPONENT_WEIGHTS = ((0.201, 0.799), (0.764, 0.236), (0.831, 0.169))
HB_COMPONENT_MEANS = ((79.8, 135.6), (59.8, 110.2), (57.8, 101.1))
HB_COMPONENT_SDS = ((21.0, 16.0), (9.4, 26.0), (8.4, 27.2))
HB_LOWER, HB_UPPER = 42.0, 170.0
HB_IMPUTATION, HB_MISSING_PROBABILITY = 122.0, 803.0 / 53305.0

INVENTORY_WEIGHTS = np.array([0.472753, 0.527247])
INVENTORY_MEANS = np.array([246.70438, 314.7556])
INVENTORY_SDS = np.array([37.565011, 38.24376])
# This full-precision shift keeps every MDCEV allocation unchanged when
# G moves from 256 to 280: phi and the shadow price scale by 280 / 256.
LOG_PHI_INTERCEPT = -0.8663878413103128
GAMMA_SCALE = 0.43
PLANNED_BETA = (6.60, 1.41)
EMERGENCY_BETA = (2.95, 2.05)


def _unit(value):
    return np.clip(np.asarray(value, dtype=float), 1e-12, 1.0 - 1e-12)


def _tn_cdf(value, mean, sd, lower, upper):
    return truncnorm.cdf(value, (lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)


def _tn_ppf(probability, mean, sd, lower, upper):
    return truncnorm.ppf(
        _unit(probability), (lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd
    )


def inventory_cdf(value):
    value = np.asarray(value, dtype=float)
    result = np.zeros_like(value)
    for weight, mean, sd in zip(INVENTORY_WEIGHTS, INVENTORY_MEANS, INVENTORY_SDS):
        result += weight * _tn_cdf(value, mean, sd, INVENTORY_LOWER, INVENTORY_UPPER)
    return np.clip(result, 0.0, 1.0)


def inventory_ppf(probability, iterations=48):
    """Vectorised inverse of the truncated-normal mixture CDF."""
    probability = _unit(probability)
    low = np.full_like(probability, INVENTORY_LOWER)
    high = np.full_like(probability, INVENTORY_UPPER)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        below = inventory_cdf(middle) < probability
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    return 0.5 * (low + high)


def draw_daily_context(rng, size=None):
    """Draw ``(N-,N+,I,cp,ce)`` in the frozen latent-Gaussian order."""
    if np.linalg.eigvalsh(COPULA_CORRELATION).min() <= 0:
        raise RuntimeError("v10 copula correlation is not positive definite")
    z = rng.multivariate_normal(np.zeros(5), COPULA_CORRELATION, size=size)
    p = _unit(norm.cdf(z))
    old_raw = nbinom.ppf(p[..., 0], N_OLD_R, N_OLD_P).astype(np.int64)
    new_raw = nbinom.ppf(p[..., 1], N_NEW_R, N_NEW_P).astype(np.int64)
    n_old = np.clip(old_raw, 1, MAX_VISIBLE_ORDERS)
    n_new = np.clip(new_raw, 0, MAX_NEW_ORDERS)
    inventory = inventory_ppf(p[..., 2])
    cp = beta_dist.ppf(p[..., 3], *PLANNED_BETA)
    ce = beta_dist.ppf(p[..., 4], *EMERGENCY_BETA)
    if size is None:
        return int(n_old), int(n_new), float(inventory), float(cp), float(ce)
    return n_old, n_new, inventory, cp, ce


def draw_profiles(rng, count, cohort):
    """Draw raw ``(application units, risk, Hb g/L)`` patient features."""
    count = int(count)
    if count < 0 or cohort not in {"old", "new"}:
        raise ValueError("cohort must be old/new and count must be nonnegative")
    if count == 0:
        return np.empty((0, 3), dtype=float)
    application = rng.choice(U_VALUES, size=count, p=U_PROBABILITIES)
    risk = rng.choice(RISK_VALUES, size=count, p=RISK_PROBABILITIES)
    hb = np.empty(count)
    tier = np.where(risk < 0.25, 0, np.where(risk < 0.55, 1, 2))
    for level in range(3):
        selected = np.flatnonzero(tier == level)
        if not len(selected):
            continue
        component = rng.choice(2, size=len(selected), p=HB_COMPONENT_WEIGHTS[level])
        for mixture in range(2):
            positions = selected[component == mixture]
            if not len(positions):
                continue
            hb[positions] = _tn_ppf(
                rng.random(len(positions)),
                HB_COMPONENT_MEANS[level][mixture],
                HB_COMPONENT_SDS[level][mixture],
                HB_LOWER,
                HB_UPPER,
            )
    missing = rng.random(count) < HB_MISSING_PROBABILITY
    hb[missing] = HB_IMPUTATION
    return np.column_stack([application, risk, hb])


def mechanism_parameters(patients, cp, ce, eta, epsilon):
    patients = np.asarray(patients, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    if patients.ndim != 2 or patients.shape[1] != 3 or epsilon.shape != (len(patients),):
        raise ValueError("patients=(u,r,hb) and one epsilon per patient are required")
    application, risk, hb = patients.T
    ell = np.log1p(application) / np.log(11.0)
    deficit = np.maximum((70.0 + 10.0 * risk - hb) / 10.0, 0.0)
    gamma = GAMMA_SCALE * (0.5 + 1.5 * ell)
    log_phi = (
        LOG_PHI_INTERCEPT
        + 0.15 * ell
        + 1.20 * risk
        + 0.45 * deficit
        + 0.40 * float(cp) * ell
        + 0.60 * float(ce) * risk
        + float(eta)
        + epsilon
    )
    return gamma, np.exp(log_phi), ell, deficit


def mdcev_allocate(gamma, phi, inventory, target=TARGET_INVENTORY, cap=DOSE_CAP, iterations=56):
    """Solve the unique joint allocation through its scalar KKT equation."""
    gamma, phi = np.asarray(gamma, dtype=float), np.asarray(phi, dtype=float)
    inventory = float(inventory)
    cap = np.broadcast_to(np.asarray(cap, dtype=float), gamma.shape)
    if gamma.shape != phi.shape or gamma.ndim != 1 or not len(gamma):
        raise ValueError("gamma and phi must be matching nonempty vectors")
    if (
        (gamma <= 0).any()
        or (phi <= 0).any()
        or (cap < 0).any()
        or not INVENTORY_LOWER < inventory <= INVENTORY_UPPER
    ):
        raise ValueError("invalid positive MDCEV inputs")

    def dose(multiplier):
        return np.minimum(cap, gamma * np.maximum(phi / multiplier - 1.0, 0.0))

    low = float(target) / inventory
    high = max(low, float(phi.max()))
    if dose(low).sum() <= 1e-14:
        multiplier = low
    else:
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            if float(target) / middle + dose(middle).sum() > inventory:
                low = middle
            else:
                high = middle
        multiplier = 0.5 * (low + high)
    doses = dose(multiplier)
    reserve = float(target) / multiplier
    if abs(reserve + doses.sum() - inventory) > 2e-8 * inventory:
        raise RuntimeError("MDCEV resource equation did not converge")
    return doses, reserve, multiplier


def doctor_allocate(patients, cp, ce, inventory, rng):
    eta = float(rng.normal(0.0, 0.08))
    epsilon = rng.gumbel(-0.12 * np.euler_gamma, 0.12, size=len(patients))
    gamma, phi, ell, deficit = mechanism_parameters(patients, cp, ce, eta, epsilon)
    doses, reserve, multiplier = mdcev_allocate(gamma, phi, inventory, cap=patients[:, 0])
    return {
        "Y": doses,
        "R": reserve,
        "lambda": multiplier,
        "gamma": gamma,
        "phi": phi,
        "ell": ell,
        "deficit": deficit,
        "eta": eta,
        "epsilon": epsilon,
    }


def sample_given_context(rng, n_old, n_new, inventory, cp, ce):
    old = draw_profiles(rng, n_old, "old")
    new = draw_profiles(rng, n_new, "new")
    allocation = doctor_allocate(np.vstack([old, new]), cp, ce, inventory, rng)
    doses = allocation["Y"]
    demand = float(doses.sum())
    known = float(doses[:n_old].sum())
    if demand < -1e-10 or demand > inventory + 1e-8 or demand > DEMAND_UPPER + 1e-8:
        raise RuntimeError("v10 demand violated the inventory-derived support")
    return {
        "cp": float(cp),
        "ce": float(ce),
        "inventory": float(inventory),
        "n_old": int(n_old),
        "n_new": int(n_new),
        "old": old,
        "new": new,
        "old_y": doses[:n_old],
        "new_y": doses[n_old:],
        "demand": demand,
        "known": known,
        "demand_new": demand - known,
        **allocation,
    }


def day_sample(rng):
    return sample_given_context(rng, *draw_daily_context(rng))


def _old_latent_bounds(n_old):
    n_old = int(n_old)
    if not 1 <= n_old <= MAX_VISIBLE_ORDERS:
        raise ValueError("observed N- is outside its top-coded support")
    if n_old == 1:
        p_low, p_high = 0.0, nbinom.cdf(1, N_OLD_R, N_OLD_P)
    elif n_old == MAX_VISIBLE_ORDERS:
        p_low, p_high = nbinom.cdf(MAX_VISIBLE_ORDERS - 1, N_OLD_R, N_OLD_P), 1.0
    else:
        p_low = nbinom.cdf(n_old - 1, N_OLD_R, N_OLD_P)
        p_high = nbinom.cdf(n_old, N_OLD_R, N_OLD_P)
    return (-np.inf if p_low == 0 else norm.ppf(p_low), np.inf if p_high == 1 else norm.ppf(p_high))


def conditional_new_counts(rng, observed_n_old, cp, ce, inventory, size):
    """Draw N+ conditional on continuous observations and the N- latent interval."""
    size = int(size)
    if size <= 0:
        raise ValueError("size must be positive")
    p_observed = _unit(
        [
            inventory_cdf(inventory),
            beta_dist.cdf(cp, *PLANNED_BETA),
            beta_dist.cdf(ce, *EMERGENCY_BETA),
        ]
    )
    z_observed = norm.ppf(p_observed)
    targets, observed = np.array([0, 1]), np.array([2, 3, 4])
    sigma_to = COPULA_CORRELATION[np.ix_(targets, observed)]
    sigma_oo = COPULA_CORRELATION[np.ix_(observed, observed)]
    solve = np.linalg.solve(sigma_oo, z_observed)
    mean = sigma_to @ solve
    covariance = COPULA_CORRELATION[np.ix_(targets, targets)]
    covariance -= sigma_to @ np.linalg.solve(sigma_oo, sigma_to.T)
    mean_old, mean_new = mean
    var_old, var_new, cov = covariance[0, 0], covariance[1, 1], covariance[1, 0]
    lower, upper = _old_latent_bounds(observed_n_old)
    z_old = truncnorm.ppf(
        _unit(rng.random(size)),
        (lower - mean_old) / np.sqrt(var_old),
        (upper - mean_old) / np.sqrt(var_old),
        loc=mean_old,
        scale=np.sqrt(var_old),
    )
    z_new = mean_new + cov / var_old * (z_old - mean_old)
    z_new += np.sqrt(max(var_new - cov * cov / var_old, 0.0)) * rng.normal(size=size)
    raw = nbinom.ppf(_unit(norm.cdf(z_new)), N_NEW_R, N_NEW_P).astype(np.int64)
    return np.clip(raw, 0, MAX_NEW_ORDERS)


def mdcev_allocate_batch(gamma, phi, valid, inventory, target=TARGET_INVENTORY, cap=DOSE_CAP, iterations=48):
    """Vectorised KKT allocation for independently padded patient sets."""
    gamma, phi, valid = np.asarray(gamma), np.asarray(phi), np.asarray(valid, dtype=bool)
    if gamma.shape != phi.shape or gamma.shape != valid.shape or gamma.ndim != 2:
        raise ValueError("batched gamma, phi and mask must have the same 2-D shape")
    inventory = np.broadcast_to(np.asarray(inventory, dtype=float), (len(gamma),))
    cap = np.broadcast_to(np.asarray(cap, dtype=float), gamma.shape)
    if (inventory <= INVENTORY_LOWER).any() or (inventory > INVENTORY_UPPER).any() or (cap < 0).any():
        raise ValueError("batched inventory is outside the v10 support")

    def dose(multiplier):
        raw = gamma * np.maximum(phi / multiplier[:, None] - 1.0, 0.0)
        return np.where(valid, np.minimum(cap, raw), 0.0)

    low = target / inventory
    masked_phi = np.where(valid, phi, 0.0)
    high = np.maximum(low, masked_phi.max(axis=1))
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        resource = target / middle + dose(middle).sum(axis=1)
        above = resource > inventory
        low = np.where(above, middle, low)
        high = np.where(above, high, middle)
    multiplier = 0.5 * (low + high)
    doses = dose(multiplier)
    demand = doses.sum(axis=1)
    residual = target / multiplier + demand - inventory
    if np.max(np.abs(residual)) > 3e-6:
        raise RuntimeError("batched MDCEV solver failed its resource equation")
    return doses, target / multiplier, multiplier


def mdcev_demand_batch(gamma, phi, valid, inventory, target=TARGET_INVENTORY, cap=DOSE_CAP, iterations=48):
    """Return total issue demand from the vectorised KKT allocation."""
    doses, _, _ = mdcev_allocate_batch(
        gamma, phi, valid, inventory, target=target, cap=cap, iterations=iterations
    )
    return doses.sum(axis=1)


def _conditional_demand_batch(old_patients, cp, ce, inventory, counts, rng):
    draws = len(counts)
    n_old = len(old_patients)
    maximum_new = int(np.max(counts, initial=0))
    width = n_old + maximum_new
    patients = np.zeros((draws, width, 3), dtype=float)
    patients[:, :n_old, :] = old_patients[None, :, :]
    valid = np.zeros((draws, width), dtype=bool)
    valid[:, :n_old] = True
    if maximum_new:
        new_valid = np.arange(maximum_new)[None, :] < counts[:, None]
        new_flat = draw_profiles(rng, int(counts.sum()), "new")
        patients[:, n_old:, :][new_valid] = new_flat
        valid[:, n_old:] = new_valid
    application, risk, hb = np.moveaxis(patients, -1, 0)
    ell = np.log1p(application) / np.log(11.0)
    deficit = np.maximum((70.0 + 10.0 * risk - hb) / 10.0, 0.0)
    gamma = GAMMA_SCALE * (0.5 + 1.5 * ell)
    eta = rng.normal(0.0, 0.08, size=(draws, 1))
    epsilon = rng.gumbel(-0.12 * np.euler_gamma, 0.12, size=(draws, width))
    log_phi = (
        LOG_PHI_INTERCEPT + 0.15 * ell + 1.20 * risk + 0.45 * deficit
        + 0.40 * float(cp) * ell + 0.60 * float(ce) * risk + eta + epsilon
    )
    return mdcev_demand_batch(gamma, np.exp(log_phi), valid, inventory, cap=application)


def simulate_conditional_demand(
    old_patients, cp, ce, inventory, n_draws, rng, observed_n_old=None, batch_size=4096
):
    """Draw from D | A-,cp,ce,I using vectorised conditional-MC batches."""
    old_patients = np.asarray(old_patients, dtype=float)
    if old_patients.ndim != 2 or old_patients.shape[1] != 3:
        raise ValueError("old_patients must have columns (u,r,hb)")
    observed_n_old = len(old_patients) if observed_n_old is None else int(observed_n_old)
    if observed_n_old != len(old_patients):
        raise ValueError("N- must equal the visible set size")
    n_draws, batch_size = int(n_draws), int(batch_size)
    if n_draws <= 0 or batch_size <= 0:
        raise ValueError("n_draws and batch_size must be positive")
    counts = conditional_new_counts(rng, observed_n_old, cp, ce, inventory, n_draws)
    demand = np.empty(n_draws)
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        demand[start:stop] = _conditional_demand_batch(
            old_patients, cp, ce, inventory, counts[start:stop], rng
        )
    return demand


def patient_summaries(old):
    """Return authorized X summaries and the nine extra 16-D Psi statistics."""
    normalized = np.column_stack([old[:, 0] / 10.0, old[:, 1], (old[:, 2] - HB_LOWER) / 128.0])
    cross = [
        np.mean(normalized[:, 0] * normalized[:, 1]),
        np.mean(normalized[:, 0] * normalized[:, 2]),
        np.mean(normalized[:, 1] * normalized[:, 2]),
    ]
    extrema = [
        normalized[:, 0].min(),
        normalized[:, 0].max(),
        normalized[:, 1].min(),
        normalized[:, 1].max(),
        normalized[:, 2].min(),
        normalized[:, 2].max(),
    ]
    return {
        "request_use_sum": float(old[:, 0].sum()),
        "risk_sum": float(old[:, 1].sum()),
        "hb_sum": float(normalized[:, 2].sum()),
        **dict(zip(PSI_COLUMNS, [*cross, *extrema])),
    }


def fixed_scalers():
    return {
        "order": {
            "u": {"mean": 0.0, "std": 10.0},
            "r": {"mean": 0.0, "std": 1.0},
            "hb": {"mean": HB_LOWER, "std": 128.0},
        },
        "day": {
            "planned_load": {"mean": 0.0, "std": 1.0},
            "emergency_load": {"mean": 0.0, "std": 1.0},
            "n_registered": {"mean": 0.0, "std": float(MAX_VISIBLE_ORDERS)},
            "request_use_sum": {"mean": 0.0, "std": 10.0 * MAX_VISIBLE_ORDERS},
            "risk_sum": {"mean": 0.0, "std": float(MAX_VISIBLE_ORDERS)},
            "hb_sum": {"mean": 0.0, "std": float(MAX_VISIBLE_ORDERS)},
        },
        "inventory": {"It": {"mean": 0.0, "std": INVENTORY_UPPER}},
    }


def schema(config):
    return {
        "schema_version": SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "generator_name": GENERATOR,
        "integer_decision_day": True,
        "compact_format": "axid_v10_csr_v1",
        "compact_files": {
            "daily": "daily.npz",
            "patients": "patients.npy",
            "patient_labels": "patient_labels.npy",
            "patient_offsets": "patient_offsets.npy",
        },
        "order_num_cols": ORDER_NUM,
        "order_cat_cols": [],
        "day_feature_cols": DAY,
        "psi_summary_cols": PSI_COLUMNS,
        "inventory_feature": "It",
        "inventory_position": 2,
        "excluded_from_model_inputs": [
            "D", "D_new", "known_recent_D", "real_demand", "A_plus", "eta", "epsilon",
            "gamma", "phi", "lambda", "Y", "R", "q", *PSI_COLUMNS,
        ],
        "information_groups": {
            "day_context": ["planned_load", "emergency_load"],
            "day_basic": ["n_registered", "request_use_sum"],
            "day_clinical": ["risk_sum", "hb_sum"],
            "order_basic_num": ORDER_NUM,
            "order_basic_cat": [],
        },
        "max_orders": int(config["max_visible_orders"]),
        "max_new_orders": int(config["max_new_orders"]),
        "enforce_constructive_width": False,
        "demand_support_upper": DEMAND_UPPER,
        "demand_grid_step": None,
        "target_inventory": TARGET_INVENTORY,
        "dose_cap": DOSE_CAP,
        "copula_order": list(COPULA_ORDER),
    }


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _empty_daily(total):
    float_fields = [
        "D", "D_new", "known_recent_D", "It", "planned_load", "emergency_load",
        "request_use_sum", "risk_sum", "hb_sum", *PSI_COLUMNS,
    ]
    daily = {name: np.empty(total, dtype=np.float32) for name in float_fields}
    daily.update(
        decision_day=np.empty(total, dtype=np.int64),
        split=np.empty(total, dtype=np.uint8),
        n_orders=np.empty(total, dtype=np.int16),
        n_registered=np.empty(total, dtype=np.int16),
        supervised_eligible=np.ones(total, dtype=np.uint8),
        demand_observed=np.ones(total, dtype=np.uint8),
        inventory_observed=np.ones(total, dtype=np.uint8),
    )
    return daily


def _write_compat_csv(folder, daily, patients, labels, offsets):
    """Small-fold compatibility only; formal runs use the compact arrays."""
    total = len(daily["decision_day"])
    if total > 100_000:
        raise ValueError("CSV compatibility export is forbidden above 100,000 days")
    frame = pd.DataFrame({name: values for name, values in daily.items()})
    frame["target_day"] = frame.decision_day
    frame["snapshot_day"] = frame.decision_day - 1
    frame["information_cutoff"] = frame.decision_day
    frame["source_split"] = np.where(frame.split.eq(0), "train", "test")
    frame["split"] = frame["source_split"]
    frame.to_csv(folder / "daily_features.csv", index=False)
    rows = []
    for day in range(total):
        start, stop = int(offsets[day]), int(offsets[day + 1])
        if start == stop:
            continue
        part = pd.DataFrame(patients[start:stop], columns=ORDER_NUM)
        part.insert(0, "order_key", [f"axid-dc-v10-{day}-{j}" for j in range(stop - start)])
        part.insert(0, "decision_day", int(daily["decision_day"][day]))
        part["real_demand"] = labels[start:stop]
        part["order_label_observed"] = True
        part["split"] = "train" if daily["split"][day] == 0 else "test"
        rows.append(part)
    pd.concat(rows, ignore_index=True).to_csv(folder / "data.csv", index=False)


def _generate_chunk(rng, start, stop, contexts, offsets, patients_out, labels_out, daily):
    """Generate one vectorised block and write it directly to the compact arrays."""
    n_old, n_new, inventory, cp, ce = (np.asarray(value[start:stop]) for value in contexts)
    counts = n_old + n_new
    rows, width = len(counts), int(counts.max())
    positions = np.arange(width)[None, :]
    valid = positions < counts[:, None]
    old_valid = positions < n_old[:, None]
    profiles = np.zeros((rows, width, 3), dtype=float)
    profiles[valid] = draw_profiles(rng, int(counts.sum()), "old")
    application, risk, hb = np.moveaxis(profiles, -1, 0)
    ell = np.log1p(application) / np.log(11.0)
    deficit = np.maximum((70.0 + 10.0 * risk - hb) / 10.0, 0.0)
    gamma = GAMMA_SCALE * (0.5 + 1.5 * ell)
    eta = rng.normal(0.0, 0.08, size=(rows, 1))
    epsilon = rng.gumbel(-0.12 * np.euler_gamma, 0.12, size=(rows, width))
    log_phi = (
        LOG_PHI_INTERCEPT + 0.15 * ell + 1.20 * risk + 0.45 * deficit
        + 0.40 * cp[:, None] * ell + 0.60 * ce[:, None] * risk + eta + epsilon
    )
    doses, _, _ = mdcev_allocate_batch(
        gamma, np.exp(log_phi), valid, inventory, cap=application
    )

    left, right = int(offsets[start]), int(offsets[stop])
    patients_out[left:right] = profiles[old_valid].astype(np.float32)
    labels_out[left:right] = doses[old_valid].astype(np.float32)
    demand = doses.sum(axis=1)
    known = np.where(old_valid, doses, 0.0).sum(axis=1)
    normalized = np.stack([application / 10.0, risk, (hb - HB_LOWER) / 128.0], axis=-1)
    old_float = old_valid.astype(float)
    old_count = n_old.astype(float)
    cross = (
        normalized[:, :, (0, 0, 1)] * normalized[:, :, (1, 2, 2)]
    )
    cross_mean = (cross * old_float[:, :, None]).sum(axis=1) / old_count[:, None]
    minima = np.where(old_valid[:, :, None], normalized, np.inf).min(axis=1)
    maxima = np.where(old_valid[:, :, None], normalized, -np.inf).max(axis=1)
    psi = np.column_stack(
        [cross_mean, minima[:, 0], maxima[:, 0], minima[:, 1], maxima[:, 1], minima[:, 2], maxima[:, 2]]
    )
    values = {
        "D": demand,
        "D_new": demand - known,
        "known_recent_D": known,
        "It": inventory,
        "planned_load": cp,
        "emergency_load": ce,
        "request_use_sum": np.where(old_valid, application, 0.0).sum(axis=1),
        "risk_sum": np.where(old_valid, risk, 0.0).sum(axis=1),
        "hb_sum": np.where(old_valid, normalized[:, :, 2], 0.0).sum(axis=1),
        **{name: psi[:, j] for j, name in enumerate(PSI_COLUMNS)},
    }
    for name, value in values.items():
        daily[name][start:stop] = value
    daily["n_orders"][start:stop] = n_old
    daily["n_registered"][start:stop] = n_old


def generate_performance(config, output):
    """Generate compact CSR-like artifacts without per-day Python objects."""
    config = dict(config)
    if config.get("schema_version") not in {10, "10"} or config.get("generator_version") not in {10, "10"}:
        raise ValueError("v10 reference must declare schema/generator version 10")
    if int(config["max_visible_orders"]) != MAX_VISIBLE_ORDERS or int(config["max_new_orders"]) != MAX_NEW_ORDERS:
        raise ValueError("v10 count supports are fixed at 138 and 73")
    if float(config.get("target_inventory", TARGET_INVENTORY)) != TARGET_INVENTORY:
        raise ValueError("v10 uses the single common G=280")
    if float(config.get("gamma_scale", GAMMA_SCALE)) != GAMMA_SCALE or float(
        config.get("log_phi_intercept", LOG_PHI_INTERCEPT)
    ) != LOG_PHI_INTERCEPT:
        raise ValueError("v10 gamma scale and log-phi intercept are frozen by aggregate moment calibration")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output exists; choose a fresh directory")
    output.mkdir(parents=True)
    n_train, n_test = int(config["n_train"]), int(config["n_test"])
    total = n_train + n_test
    if "train_seed" in config and "test_seed" in config:
        train_rng = np.random.default_rng(int(config["train_seed"]))
        test_rng = np.random.default_rng(int(config["test_seed"]))
    else:
        train_rng, test_rng = [
            np.random.default_rng(seed)
            for seed in np.random.SeedSequence([int(config["data_seed"]), 1010]).spawn(2)
        ]
    train_context = draw_daily_context(train_rng, n_train)
    test_context = draw_daily_context(test_rng, n_test)
    contexts = tuple(np.concatenate([train_context[j], test_context[j]]) for j in range(5))
    n_old, n_new, inventory, cp, ce = contexts
    offsets = np.empty(total + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(n_old, out=offsets[1:])
    patients = np.lib.format.open_memmap(
        output / "patients.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), 3)
    )
    labels = np.lib.format.open_memmap(
        output / "patient_labels.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]),)
    )
    daily = _empty_daily(total)
    daily["decision_day"][:n_train] = np.arange(n_train)
    daily["decision_day"][n_train:] = n_train + 30 + np.arange(n_test)
    daily["split"][:n_train] = 0
    daily["split"][n_train:] = 1
    chunk_days = max(1, int(config.get("generation_chunk_days", 2048)))
    for split_start, split_stop, rng in ((0, n_train, train_rng), (n_train, total, test_rng)):
        for start in range(split_start, split_stop, chunk_days):
            stop = min(split_stop, start + chunk_days)
            _generate_chunk(rng, start, stop, contexts, offsets, patients, labels, daily)
    patients.flush()
    labels.flush()
    np.save(output / "patient_offsets.npy", offsets, allow_pickle=False)
    np.savez(output / "daily.npz", **daily)
    _save_json(output / "data_feature_schema.json", schema(config))
    _save_json(output / "generation.yaml", config)
    split_manifest = {
        "train_start": 0, "train_end": n_train - 1,
        "test_start": n_train + 30, "test_end": n_train + 30 + n_test - 1,
        "evaluation_role": "test", "evaluation_is_retrospective": False,
        "train_days": n_train, "test_days": n_test,
    }
    _save_json(output / "split_manifest.json", split_manifest)
    _save_json(
        output / "preprocessing_state.json",
        {"feature_fit_end": n_train - 1, "numeric_scalers": fixed_scalers()},
    )
    if bool(config.get("export_csv", False)):
        _write_compat_csv(output, daily, patients, labels, offsets)
    artifacts = {}
    for name in (
        "daily.npz", "patients.npy", "patient_labels.npy", "patient_offsets.npy",
        "data_feature_schema.json", "generation.yaml", "split_manifest.json", "preprocessing_state.json",
    ):
        artifacts[name] = _file_hash(output / name)
    if bool(config.get("export_csv", False)):
        artifacts.update({name: _file_hash(output / name) for name in ("daily_features.csv", "data.csv")})
    _save_json(output / "manifest.json", {"complete": True, "artifacts": artifacts})
    return {
        "schema_version": SCHEMA, "train_days": n_train, "test_days": n_test,
        "train_orders": int(offsets[n_train]), "test_orders": int(offsets[-1] - offsets[n_train]),
        "demand_upper": DEMAND_UPPER, "target_inventory": TARGET_INVENTORY,
        "compact_format": "axid_v10_csr_v1",
    }
