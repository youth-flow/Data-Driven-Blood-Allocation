"""Frozen AXID-DC v6 performance contract checks; no training or test access."""

from math import comb

METHODS = ["DAT", "SAA", "wSAA", "Flat-ETO", "Set-ETO", "LQR", "GBQR", "DeepQR"]
VARIANTS = [
    "full",
    "no_self_attention",
    "no_self_attention_param_matched",
    "mean_pool",
    "sum_pool",
    "no_day_query",
    "mean_pool_param_matched",
    "sum_pool_param_matched",
]
LEVELS = ["L0", "L1", "L2", "L3"]


def validate_contract(protocol, reference):
    if int(protocol["protocol_version"]) < 7 or protocol["scenario"] != "AXID-DC-v6":
        raise ValueError("This contract is only for the frozen AXID-DC v6 performance protocol")
    if int(reference.get("schema_version", 0)) != 6 or int(reference.get("generator_version", 0)) != 6:
        raise ValueError("AXID-DC v6 reference version mismatch")
    if protocol["primary_tau"] != 0.9 or protocol["performance_experiments"]["tau"] != 0.9:
        raise ValueError("Primary tau must be 0.9")
    sampling = protocol["sampling"]
    if sampling["order_dimension"] != 3 or sampling["day_dimension_excluding_count"] != 6:
        raise ValueError("AXID-DC v6 uses three patient coordinates and six non-inventory day coordinates")
    if sampling["n_new_max"] != 2 or sampling["max_visible_orders"] != 8:
        raise ValueError("AXID-DC v6 fixes M=8 and N+<=2")
    if int(reference["max_new_orders"]) != 2 or int(reference["max_visible_orders"]) != 8:
        raise ValueError("Generator reference does not match AXID-DC v6")
    if float(protocol["profiles"]["performance"]["demand_upper"]) != 61.0:
        raise ValueError("AXID-DC v6 hard demand upper bound B must be 61")
    profile = protocol["profiles"]["performance"]
    if profile["d_model"] < 57:
        raise ValueError("AXID-DC v6 d_model must satisfy the declared sufficient dimension")
    performance = protocol["performance_experiments"]
    if performance["n_train"] != 2500 or performance["n_test"] != 3000:
        raise ValueError("AXID-DC v6 fixes 2500 train and 3000 test days")
    if performance["independent_data_repetitions"] != 10 or performance["model_seeds"] != [0, 1, 2]:
        raise ValueError("AXID-DC v6 fixes ten independent repeats and three training seeds")
    if performance["hyperparameter_search"] or performance["validation_selection"]:
        raise ValueError("AXID-DC v6 forbids search and validation selection")
    if performance["comparison"]["methods"] != METHODS:
        raise ValueError("AXID-DC v6 fixes eight comparison methods")
    if performance["ablation"]["variants"] != VARIANTS:
        raise ValueError("AXID-DC v6 fixes eight structural variants")
    if performance["information"]["levels"] != LEVELS:
        raise ValueError("AXID-DC v6 fixes four information levels")
    if profile["patient_hidden"] != 96 or profile["day_ff"] != 128:
        raise ValueError("Frozen AXID-DC v6 DAT candidate widths changed")
    if comb(3 + 8, 8) - 1 != 164:
        raise ValueError("Unexpected patient-set latent dimension accounting")
    return {"status": "frozen", "methods": 8, "variants": 8, "levels": 4, "d_model": profile["d_model"]}
