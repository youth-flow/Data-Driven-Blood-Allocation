"""Frozen AXID-v3 performance contract checks; no training or test access."""

from math import comb

METHODS = ["DAT", "SAA", "wSAA", "Rolling-SAA", "Flat-ETO", "Set-ETO", "LQR", "GBQR", "DeepQR"]
VARIANTS = [
    "full",
    "no_self_attention",
    "mean_pool",
    "sum_pool",
    "no_day_query",
    "mean_pool_param_matched",
    "sum_pool_param_matched",
]
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def validate_contract(protocol, reference):
    if int(protocol["protocol_version"]) < 5 or protocol["scenario"] != "AXID-v3":
        raise ValueError("This contract is only for the frozen AXID-v3 performance protocol")
    if int(reference.get("schema_version", 0)) != 3 or int(reference.get("generator_version", 0)) != 3:
        raise ValueError("AXID-v3 reference version mismatch")
    if protocol["primary_tau"] != 0.9 or protocol["performance_experiments"]["tau"] != 0.9:
        raise ValueError("Primary tau must be 0.9")
    sampling = protocol["sampling"]
    if sampling["order_dimension"] != 4 or sampling["day_dimension_excluding_count"] != 8:
        raise ValueError("AXID-v3 uses four patient coordinates and eight non-inventory day coordinates")
    if sampling["n_new_max"] != 12 or sampling["max_visible_orders"] != 8:
        raise ValueError("AXID-v3 fixes M=8 and K_new=12")
    if int(reference["max_new_orders"]) != 12 or int(reference["max_visible_orders"]) != 8:
        raise ValueError("Generator reference does not match AXID-v3")
    if float(protocol["profiles"]["performance"]["demand_upper"]) != 60.5:
        raise ValueError("AXID-v3 hard demand upper bound B must be 60.5")
    profile = protocol["profiles"]["performance"]
    if profile["d_model"] < 74:
        raise ValueError("AXID-v3 d_model must satisfy the declared sufficient dimension")
    performance = protocol["performance_experiments"]
    if performance["n_train"] != 2500 or performance["n_test"] != 3000:
        raise ValueError("AXID-v3 fixes 2500 train and 3000 test days")
    if performance["independent_data_repetitions"] != 10 or performance["model_seeds"] != [0, 1, 2]:
        raise ValueError("AXID-v3 fixes ten independent repeats and three training seeds")
    if performance["hyperparameter_search"] or performance["validation_selection"]:
        raise ValueError("AXID-v3 forbids search and validation selection")
    if performance["comparison"]["methods"] != METHODS:
        raise ValueError("AXID-v3 fixes nine comparison methods")
    if performance["ablation"]["variants"] != VARIANTS:
        raise ValueError("AXID-v3 fixes seven structural variants")
    if performance["information"]["levels"] != LEVELS:
        raise ValueError("AXID-v3 fixes five information levels")
    if profile["patient_hidden"] != 96 or profile["day_ff"] != 128:
        raise ValueError("Frozen AXID-v3 DAT candidate widths changed")
    if comb(4 + 8, 8) - 1 != 494:
        raise ValueError("Unexpected patient-set latent dimension accounting")
    return {"status": "frozen", "methods": 9, "variants": 7, "levels": 5, "d_model": profile["d_model"]}
