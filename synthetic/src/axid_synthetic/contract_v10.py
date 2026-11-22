"""Frozen AXID-DC v10 DGP/performance contract checks.

The DGP/schema version and the performance-protocol version are deliberately
separate: v10 uses generator/schema version 10, while protocol version 11 is
the next protocol identifier after the historical v8 protocol used 10.
"""

from __future__ import annotations

from math import isclose


METHODS = ["DAT", "SAA", "wSAA", "Flat", "Set-ETO", "LQR", "GBQR", "DeepQR"]
VARIANTS = ["full", "no_self_attention", "no_day_query", "mean_pool", "sum_pool"]
LEVELS = ["L0", "L1", "L2", "L3"]
FINAL_METRICS = ["P_PF", "P_Oracle", "MeanCost", "SL_256"]
EXECUTION_ORDER = ["data", "P3", "P2", "H1", "P1", "PF_and_Oracle"]
SEEDS = {
    "dgp_train": 2026111401,
    "dgp_test": 2026111402,
    "shuffle": 2026111403,
    "model_init": 2026111404,
    "forest": 2026111405,
    "bootstrap": 2026111406,
    "oracle": 2026111407,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_number(left, right) -> bool:
    return isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def validate_contract(protocol, reference):
    """Validate the complete frozen v10 contract without reading test outcomes."""

    _require(int(protocol.get("protocol_version", 0)) == 11, "v10 performance protocol_version must be 11")
    _require(protocol.get("status") == "frozen", "v10 protocol must be frozen")
    _require(protocol.get("scenario") == "AXID-DC-v10", "v10 scenario mismatch")
    _require(
        protocol.get("generator_family") == "axid_dc_v10_mdcev_real_scale",
        "v10 generator family mismatch",
    )
    _require(int(reference.get("schema_version", 0)) == 10, "v10 reference schema_version must be 10")
    _require(int(reference.get("generator_version", 0)) == 10, "v10 reference generator_version must be 10")
    _require(reference.get("scenario") == "AXID-DC-v10", "v10 reference scenario mismatch")
    calibration = reference.get("parameter_calibration", {})
    _require(
        calibration.get("status") == "hybrid_structural_calibration"
        and calibration.get("kappa") == "fixed_scale_normalization",
        "v10 must disclose hybrid calibration and the kappa scale normalization",
    )
    _require(
        calibration.get("gamma_phi_coefficients")
        == "transparent_moment_calibration_not_full_structural_mle",
        "v10 gamma/phi coefficients must not be mislabeled as fully identified MLEs",
    )

    decision = protocol["decision"]
    _require(_same_number(decision["tau"], 0.9), "v10 tau must be 0.9")
    for field in ("common_G", "hospital_mdcev_G", "blood_center_target_G"):
        _require(_same_number(decision[field], 256.0), "hospital and blood-center G must be the same 256")
    _require(
        _same_number(decision["shortage_weight"], 0.9)
        and _same_number(decision["excess_weight"], 0.1),
        "v10 cost weights must be 0.9/0.1",
    )
    _require(decision["nonnegative_replenishment"] is True, "v10 replenishment must be nonnegative")

    sampling = protocol["sampling"]
    _require(sampling["patient_features"] == ["u", "r", "hb"], "v10 patient features changed")
    _require(sampling["day_state_features"] == ["c_p", "c_e", "inventory"], "v10 day states changed")
    _require(int(sampling["order_dimension"]) == 3, "v10 patient input dimension must be three")
    _require(int(sampling["day_dimension"]) == 7, "v10 model day input dimension must be seven")
    _require(
        int(sampling["max_visible_orders"]) == 138 and int(sampling["max_new_orders"]) == 76,
        "v10 order caps must be 138 visible and 76 new",
    )
    _require(sampling["visible_set_nonempty"] is True, "v10 visible patient set must be nonempty")
    _require(
        _same_number(sampling["demand_lower"], 0.0) and _same_number(sampling["demand_upper"], 519.0),
        "v10 demand support must be [0,519]",
    )
    _require(int(reference["max_visible_orders"]) == 138, "v10 reference visible cap mismatch")
    _require(int(reference["max_new_orders"]) == 76, "v10 reference new-patient cap mismatch")
    _require(_same_number(reference["demand_support_upper"], 519.0), "v10 reference demand cap mismatch")
    _require(_same_number(reference["target_inventory"], 256.0), "v10 reference common G mismatch")
    _require(
        _same_number(reference["gamma_scale"], 0.7)
        and _same_number(reference["log_phi_intercept"], -1.78),
        "v10 moment-calibrated gamma scale/intercept changed",
    )

    dat = protocol["dat"]
    _require(int(dat["expected_parameter_count"]) == 51105, "v10 Full DAT must have 51,105 parameters")
    _require(int(dat["d_model"]) == 64, "v10 d_model must remain 64")
    _require(dat["patient_encoder"] == [3, 64, 64], "v10 patient encoder changed")
    _require(
        int(dat["self_attention_layers"]) == 1
        and int(dat["cross_attention_layers"]) == 1
        and int(dat["self_attention_heads"]) == 4
        and int(dat["cross_attention_heads"]) == 4,
        "v10 must retain one four-head SA and one four-head CA",
    )
    _require(int(dat["day_ff"]) == 96, "v10 day FF width must remain 96")
    _require(dat["layer_norms"] == ["LN_sa", "LN_q", "LN_m"], "v10 must use exactly the three frozen LNs")
    _require(dat["dropout"] == 0.0 and dat["positional_encoding"] is False, "v10 DAT regularization changed")
    _require(dat["output_clip"] == [0.0, 519.0], "v10 DAT output support must be [0,519]")
    _require(
        dat["constructive_width_claim_for_M138"] is False,
        "v10 cannot claim the old M=8 constructive width result for M=138",
    )

    performance = protocol["performance_experiments"]
    _require(
        int(performance["n_train"]) == 524288 and int(performance["n_test"]) == 10000,
        "v10 fast protocol fixes 524,288 train and 10,000 test days",
    )
    _require(
        int(performance["batch_size"]) == 128
        and int(performance["optimizer_steps"]) == 4096
        and int(performance["epochs"]) == 1,
        "v10 training budget must be batch128 x 4096 steps x one epoch",
    )
    _require(
        int(performance["n_train"])
        == int(performance["batch_size"]) * int(performance["optimizer_steps"]),
        "v10 training days must exactly equal batch_size times optimizer_steps",
    )
    _require(
        int(performance["train_pools"]) == 1
        and int(performance["test_pools"]) == 1
        and int(performance["validation_days"]) == 0,
        "v10 uses one train pool, one test pool, and no validation pool",
    )
    _require(
        performance["validation_selection"] is False
        and performance["hyperparameter_search"] is False
        and performance["early_stopping"] is False,
        "v10 forbids validation selection, search, and early stopping",
    )
    _require(performance["optimizer"] == "AdamW", "v10 optimizer must be AdamW")
    _require(performance["betas"] == [0.9, 0.999], "v10 AdamW betas changed")
    _require(int(performance["warmup_steps"]) == 300, "v10 warmup must be 300 steps")
    _require(performance["schedule"] == "cosine", "v10 learning-rate schedule must be cosine")
    _require(_same_number(performance["weight_decay"], 1e-4), "v10 weight decay must be 1e-4")
    _require(performance["precision"] == "fp32", "v10 formal training precision must be FP32")
    _require(performance["taus"] == [0.9] and performance["targets"] == [256.0], "v10 has one tau and one G")
    _require(performance["model_seeds"] == [2026111404], "v10 fixes one model initialization")
    _require(performance["seeds"] == SEEDS, "v10 independent RNG seeds changed")
    _require(
        int(reference.get("train_seed", -1)) == SEEDS["dgp_train"]
        and int(reference.get("test_seed", -1)) == SEEDS["dgp_test"],
        "v10 generator reference must use the frozen independent train/test seeds",
    )
    _require(performance["p3_information"]["levels"] == LEVELS, "v10 P3 levels changed")
    _require(performance["p2_ablation"]["variants"] == VARIANTS, "v10 P2 variants changed")
    _require(
        performance["p2_ablation"]["parameter_matched_variants"] is False,
        "v10 excludes parameter-matched P2 variants",
    )
    _require(performance["p1_comparison"]["methods"] == METHODS, "v10 P1 methods changed")
    _require(performance["execution_order"] == EXECUTION_ORDER, "v10 staged execution order changed")
    _require(performance["manual_gate_after_p2"] is True, "v10 H1 manual gate must remain active")

    evaluation = protocol["evaluation"]
    _require(evaluation["final_metrics"] == FINAL_METRICS, "v10 must report exactly two P, MeanCost, and SL_256")
    _require(_same_number(evaluation["nominal_service_rate"], 0.9), "v10 nominal service rate must be 0.9")
    _require(evaluation["clip_prescriptiveness"] is False, "v10 P metrics must not be clipped")
    bootstrap = evaluation["bootstrap"]
    _require(
        bootstrap["enabled"] is True
        and bootstrap["paired_by_test_day"] is True
        and int(bootstrap["repetitions"]) == 2000,
        "v10 requires 2,000 same-day paired bootstrap resamples",
    )
    _require(
        bootstrap["retrain_models"] is False
        and bootstrap["rerun_dgp"] is False
        and bootstrap["rerun_oracle"] is False,
        "v10 bootstrap is array resampling only",
    )
    oracle = evaluation["conditional_oracle"]
    _require(
        oracle["required_for_final_metrics"] is True
        and oracle["method"] == "conditional_monte_carlo"
        and int(oracle["paths_per_test_day"]) == 4096,
        "v10 Oracle-P requires 4,096-path conditional Monte Carlo",
    )
    _require(
        int(oracle["stability_check_days"]) == 500 and int(oracle["stability_check_paths"]) == 16384,
        "v10 Oracle stability audit must use 500 days and 16,384 paths",
    )

    runtime = protocol["runtime"]
    _require(runtime["conda_environment"] == "attention", "v10 must run in the attention environment")
    _require(runtime["device"] == "cuda", "v10 neural models must use CUDA")
    _require(runtime["formal_precision"] == "fp32", "v10 runtime precision must be FP32")
    _require(runtime["allow_tf32"] is False and runtime["use_bf16"] is False, "v10 forbids silent precision changes")
    _require(runtime["data_format"] == "csr_memmap", "v10 large data must use CSR/memmap storage")
    _require(int(runtime["dataloader_workers"]) == 4, "v10 fixes four DataLoader workers")
    _require(
        runtime["pin_memory"] is True
        and runtime["persistent_workers"] is True
        and runtime["non_blocking_device_transfer"] is True,
        "v10 GPU input pipeline acceleration changed",
    )

    return {
        "status": "frozen",
        "scenario": "AXID-DC-v10",
        "G": 256.0,
        "methods": len(METHODS),
        "variants": len(VARIANTS),
        "levels": len(LEVELS),
        "metrics": len(FINAL_METRICS),
        "n_train": int(performance["n_train"]),
        "n_test": int(performance["n_test"]),
        "optimizer_steps": int(performance["optimizer_steps"]),
    }
