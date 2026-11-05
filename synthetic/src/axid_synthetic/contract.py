"""Validate the S4-only executable protocol before generating any samples."""

from math import comb

from .generator import SCENARIOS


def validate_contract(protocol, generator):
    if protocol["protocol_version"] != 4 or generator["generator_version"] != 4:
        raise ValueError("Use protocol 4 with generator 4")
    if protocol["scenario"] != "S4" or protocol["primary_tau"] != 0.9:
        raise ValueError("The frozen protocol is S4-only with tau=0.9")
    sampling, architecture = protocol["sampling"], protocol["dat"]
    fixed = {
        "order_dimension": 3,
        "day_dimension_excluding_count": 7,
        "inventory_context_coefficient": 20.0,
        "peer_kernel_bandwidth": 0.25,
        "demand_grid_step": 0.5,
        "continuous_remainder_width": 0.5,
    }
    if any(sampling[key] != value for key, value in fixed.items()):
        raise ValueError("Canonical DGP formulas and declared support dimensions differ")
    for key, source in (
        ("n_new_max", "max_new_orders"),
        ("inventory_center", "inventory_center"),
        ("inventory_noise_half_width", "inventory_half_width"),
        ("continuous_remainder_width", "remainder_width"),
    ):
        if sampling[key] != generator[source]:
            raise ValueError(f"Generator/protocol mismatch: {key}")
    if generator["schema_version"] != 2 or generator["scenario"] != "S4":
        raise ValueError("Reference generator configuration is not the S4 version-4 contract")
    for scenario in ("S4", "C0"):
        spec = protocol["scenarios"][scenario]
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
    theory = protocol["theory_experiments"]
    if theory["scenario"] != "S4" or theory["calibration_scenario"] != "C0" or theory["tau"] != 0.9:
        raise ValueError("Theory experiments must be S4 with a separate C0 calibration")
    performance = protocol["performance_experiments"]
    if performance["scenario"] != "S4" or performance["tau"] != 0.9:
        raise ValueError("Performance experiments must be S4-only with tau=0.9")
    if performance["n_train"] != 2500 or performance["n_test"] != 3000:
        raise ValueError("Performance split must be 2500 training and 3000 test days")
    if performance["hyperparameter_search"] or performance["validation_selection"]:
        raise ValueError("The S4 protocol forbids hyperparameter search and validation selection")
    if len(performance["comparison"]["methods"]) != 9:
        raise ValueError("P1 must contain exactly nine methods")
    if len(performance["ablation"]["variants"]) != 7:
        raise ValueError("P2 must contain five main and two matched variants")
    if performance["information"]["levels"] != ["L0", "L1", "L2", "L3", "L4"]:
        raise ValueError("P3 must contain L0-L4")
