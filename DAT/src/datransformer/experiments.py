"""DAT-owned structural ablations and fixed-width information interventions."""

from copy import deepcopy

ABLATIONS = (
    "full",
    "no_self_attention",
    "mean_pool",
    "sum_pool",
    "no_day_query",
    "lite_mean_pool",
    "lite_sum_pool",
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
    active_day, active_num, active_cat = set(data.day_cols), set(data.order_num_cols), set(data.order_cat_cols)
    drop_set = False
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
    config["input_controls"] = {
        "day": [c in active_day for c in data.day_cols],
        "order_num": [c in active_num for c in data.order_num_cols],
        "order_cat": [c in active_cat for c in data.order_cat_cols],
        "drop_set": drop_set,
    }
    return config
