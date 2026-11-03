"""Validate the executable protocol before generating any samples."""

from math import comb
from .generator import SCENARIOS


def validate_contract(protocol, generator):
    if protocol["protocol_version"] != 2 or generator["generator_version"] != 3:
        raise ValueError("Use protocol 2 with generator 3")
    sampling, architecture = protocol["sampling"], protocol["dat"]
    fixed = {
        "order_dimension": 3,
        "day_dimension_excluding_count": 7,
        "inventory_context_coefficient": 20.0,
        "peer_kernel_bandwidth": 0.25,
        "demand_grid_step": 0.5,
        "continuous_remainder_width": 0.5,
    }
    if any(sampling[k] != value for k, value in fixed.items()):
        raise ValueError("Canonical DGP formulas and declared support dimensions differ")
    for key, source in (
        ("n_new_max", "max_new_orders"),
        ("inventory_center", "inventory_center"),
        ("inventory_noise_half_width", "inventory_half_width"),
        ("continuous_remainder_width", "remainder_width"),
    ):
        if sampling[key] != generator[source]:
            raise ValueError(f"Generator/protocol mismatch: {key}")
    for scenario, spec in protocol["scenarios"].items():
        if tuple(spec[k] for k in ("context", "peer", "shock")) != SCENARIOS[scenario]:
            raise ValueError("Scenario coefficients disagree with the executable DGP")
    if (
        architecture["patient_ffn"]
        or architecture["day_ffn_pre_norm"]
        or architecture["output_norm"]
        or architecture["pretraining"]
        or architecture["lora"]
        or architecture["dropout"] != 0
        or architecture["head"] != "clipped_scalar_affine"
        or architecture["loss"] != "pinball"
    ):
        raise ValueError("Protocol is not the simplified theory DAT")
    for name, profile in protocol["profiles"].items():
        m, da, dx = profile["max_orders"], 3, 7
        dm = min(comb(da + m, m) - 1, 2 * m * da + 1)
        if profile["d_model"] < max(dm + 2, dm + dx + 1) or any(
            profile["d_model"] % architecture[k] for k in ("self_attention_heads", "cross_attention_heads")
        ):
            raise ValueError(f"Invalid approximation width or attention heads: {name}")
        if profile["demand_upper"] != 3 * (m + sampling["n_new_max"]) + 0.5:
            raise ValueError(f"Output support differs from DGP support: {name}")
    settings = protocol["performance_experiments"]
    for kind, multiplier in (
        ("neural_search", 1),
        ("set_eto_search", len(settings["comparison"]["set_eto_primary_families"])),
    ):
        budget = len(settings[kind]["learning_rates"]) * len(settings[kind]["batch_sizes"]) * multiplier
        if budget > settings["max_validation_candidates_per_method"]:
            raise ValueError(f"Validation search exceeds the registered budget: {kind}")
