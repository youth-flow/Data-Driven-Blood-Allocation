"""Check the frozen design against the real DAT module; never access datasets.

This validates architecture arithmetic and deterministic forward invariance only.
It does not implement the v3 generator, certify ERM, or run any formal experiment.
"""

import argparse
from copy import deepcopy
import json
from math import comb, floor, sqrt
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DAT" / "src"))


def model_config(protocol, profile, hidden=None, radius=None):
    """A constructor config, not a legacy experiment-runner configuration."""
    spec = deepcopy(profile)
    if hidden is not None:
        spec["patient_hidden"] = spec["day_ff"] = hidden
    radius = spec["radius"] if radius is None else radius
    heads = protocol["dat"]["heads"]
    return {
        "model": {
            "d_model": spec["d_model"],
            "layer_norm_eps": protocol["dat"]["layer_norm_eps"],
            "context": {
                "include_inventory": True,
                "inventory_transform": "standardize",
                "inventory_clip": None,
            },
            "patient_encoder": {"hidden_dims": [spec["patient_hidden"]], "activation": "relu", "dropout": 0.0},
            "self_attention": {
                "enabled": True,
                "num_layers": 1,
                "num_heads": heads,
                "ff_dim": spec["patient_ff"],
                "activation": "relu",
                "dropout": 0.0,
            },
            "day_encoder": {"hidden_dims": [], "activation": "relu", "dropout": 0.0},
            "cross_attention": {
                "num_layers": 1,
                "num_heads": heads,
                "ff_dim": spec["day_ff"],
                "activation": "relu",
                "dropout": 0.0,
            },
            "head": {
                "hidden_dims": [],
                "activation": "relu",
                "dropout": 0.0,
                "output_lower_bound": 0.0,
                "output_upper_bound": spec["demand_upper"],
            },
            "quantile_queries": {"enabled": False},
        },
        "loss": {"type": "pinball", "tau": protocol["theory_experiments"]["tau"]},
        "constraints": {
            "enabled": True,
            "global_parameter_radius": radius,
            "operator_norm_bound": radius,
            "bias_bound": radius,
            "layer_norm_gain_bound": radius,
            "layer_norm_shift_bound": radius,
        },
    }


def construct(protocol, profile, hidden=None, radius=None):
    from datransformer.model import DecisionAwareTransformer

    config = model_config(protocol, profile, hidden, radius)
    # 7 non-inventory network coordinates + fixed-affine inventory = 8 X coordinates.
    net = DecisionAwareTransformer(3, [], 7, config, inventory_mean=225.0, inventory_std=180.0)
    return net, config


def validate(protocol):
    from datransformer.constraints import project_parameters

    sampling = protocol["sampling"]
    if sampling["order_dimension"] != 3 or sampling["day_dimension_excluding_count"] != 7:
        raise ValueError("Update the design and checker together if the canonical inputs change")
    if sampling["continuous_remainder_width"] != sampling["demand_grid_step"]:
        raise ValueError("The registered continuity argument requires a full grid-cell uniform remainder")
    if protocol["dat"]["head"] != "clipped_scalar_affine" or protocol["dat"]["loss"] != "pinball":
        raise ValueError("The theory contract is scalar clipped DAT with pinball loss")
    result = {"status": protocol["status"], "profiles": {}, "capacity": [], "joint_schedule": []}
    for name, profile in protocol["profiles"].items():
        m, da, dx = profile["max_orders"], sampling["order_dimension"], sampling["day_dimension_excluding_count"]
        dm = min(comb(da + m, m) - 1, 2 * m * da + 1)
        required_width = 2 * (dm + dx) + 2
        if profile["d_model"] < required_width or profile["d_model"] % protocol["dat"]["heads"]:
            raise ValueError(f"Latent width is invalid for profile {name}")
        support = 3 * (m + sampling["n_new_max"]) + sampling["continuous_remainder_width"]
        if profile["demand_upper"] != support:
            raise ValueError(f"Demand clipping bound differs from the DGP support for profile {name}")
        net, config = construct(protocol, profile)
        project_parameters(net, config)
        net.eval()
        batch = {
            "order_num": torch.rand(4, m, 3),
            "order_cat": torch.empty(4, m, 0, dtype=torch.long),
            "order_mask": torch.ones(4, m, dtype=torch.bool),
            "day_num": torch.rand(4, 7),
            "inventory": 225 + 180 * torch.rand(4),
        }
        batch["day_num"][:, 2] = 1.0  # N/M in this full-size artificial batch.
        reversed_batch = {
            key: value.flip(1) if key in {"order_num", "order_cat", "order_mask"} else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            pred = net(batch)["quantile"]
            reverse = net(reversed_batch)["quantile"]
        torch.testing.assert_close(pred, reverse, atol=1e-5, rtol=1e-5)
        if pred.shape != (4,) or not torch.isfinite(pred).all() or not ((pred >= 0) & (pred <= support)).all():
            raise ValueError("Scalar output contract failed")
        norm = sqrt(sum(float(p.detach().square().sum()) for p in net.parameters()))
        if norm > profile["radius"] + 1e-4:
            raise ValueError("Projection did not satisfy the declared global ball")
        groups = {key: sum(p.numel() for p in component.parameters()) for key, component in net.named_children()}
        result["profiles"][name] = {
            "d_m": dm,
            "d_eff": dm + dx,
            "required_d_model": required_width,
            "d_model": profile["d_model"],
            "parameters": sum(p.numel() for p in net.parameters()),
            "parameter_groups": groups,
            "permutation_check": "passed",
            "bounded_scalar_output": "passed",
            "projection_check": "passed",
        }
    theory = protocol["profiles"]["theory"]
    p0 = result["profiles"]["theory"]["parameters"]
    for hidden in protocol["theory_experiments"]["approximation"]["hidden_widths"]:
        net, _ = construct(protocol, theory, hidden=hidden)
        p = sum(x.numel() for x in net.parameters())
        result["capacity"].append({"hidden": hidden, "parameters": p, "radius": 100 * (p / p0) ** 2})
    for count in protocol["theory_experiments"]["train_sizes"]:
        hidden = max(1, floor(32 * sqrt(count / 2000)))
        net, _ = construct(protocol, theory, hidden=hidden)
        p = sum(x.numel() for x in net.parameters())
        result["joint_schedule"].append({"train_days": count, "hidden": hidden, "parameters": p})
    result["formal_experiments_run"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "synthetic/configs/confirmed_protocol.yaml")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    torch.manual_seed(0)
    torch.set_num_threads(1)
    print(json.dumps(validate(protocol), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
