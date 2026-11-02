"""DAT-owned structural ablations and fixed-width information interventions."""

from copy import deepcopy

ABLATIONS = (
    "full",
    "no_self_attention",
    "mean_pool",
    "sum_pool",
    "no_orders",
    "no_inventory",
    "pinball",
    "softplus_beta6",
    "no_planned_use",
)
INFORMATION = ("full", "L0", "L1", "L2", "L3", "L4")


def experiment_config(config, data):
    config = deepcopy(config)
    variant = config.get("ablation", "full")
    level = config.get("information", "full")
    if variant not in ABLATIONS or level not in INFORMATION:
        raise ValueError("Unknown DAT ablation or information profile")
    if variant != "full" and level != "full":
        raise ValueError("Change structure or information in one experiment, not both")
    if variant == "no_self_attention":
        config["model"]["self_attention"]["enabled"] = False
    if variant in {"mean_pool", "sum_pool"}:
        config["model"]["cross_attention"]["readout"] = variant.removesuffix("_pool")
    if variant == "no_inventory":
        config["model"]["context"]["include_inventory"] = False
    if variant == "pinball":
        config["loss"]["type"] = "pinball"
    if variant == "softplus_beta6":
        config["loss"]["beta"] = 6.0
    active_day, active_num, active_cat = set(data.day_cols), set(data.order_num_cols), set(data.order_cat_cols)
    drop_set = variant == "no_orders"
    if level not in {"full", "L4"}:
        groups = data.schema.get("information_groups")
        if not groups:
            raise ValueError("Information profiles require an explicit feature-group schema")
        required = {"day_context", "day_basic", "day_clinical", "order_basic_num", "order_basic_cat"}
        if not required <= groups.keys():
            raise ValueError("Incomplete information-group schema")
        active_day = set(groups["day_context"])
        if level in {"L1", "L2", "L3"}:
            active_day.update(groups["day_basic"])
        if level in {"L2", "L3"}:
            active_day.update(groups["day_clinical"])
        active_num, active_cat = set(groups["order_basic_num"]), set(groups["order_basic_cat"])
        drop_set = level in {"L0", "L1", "L2"}
    if variant == "no_planned_use":
        active_num = {c for c in active_num if not c.startswith("planned_use_")}
        active_day = {c for c in active_day if not c.startswith("planned_use_")}
        dependencies = data.schema.get("feature_dependencies", {})
        active_day = {c for c in active_day if not any(v.startswith("planned_use_") for v in dependencies.get(c, []))}
    config["input_controls"] = {
        "day": [c in active_day for c in data.day_cols],
        "order_num": [c in active_num for c in data.order_num_cols],
        "order_cat": [c in active_cat for c in data.order_cat_cols],
        "drop_set": drop_set,
    }
    return config
