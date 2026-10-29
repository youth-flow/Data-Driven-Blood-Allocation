from __future__ import annotations

from typing import Any

import torch
from torch import nn


@torch.no_grad()
def project_parameters(model: nn.Module, config: dict[str, Any]) -> None:
    constraint = config.get("constraints", {})
    if not bool(constraint.get("enabled", True)):
        return

    operator_bound = float(constraint["operator_norm_bound"])
    bias_bound = float(constraint["bias_bound"])
    layer_norm_gain_bound = float(constraint["layer_norm_gain_bound"])
    layer_norm_shift_bound = float(constraint["layer_norm_shift_bound"])

    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.clamp_(-layer_norm_gain_bound, layer_norm_gain_bound)
            if module.bias is not None:
                module.bias.clamp_(-layer_norm_shift_bound, layer_norm_shift_bound)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            _project_matrix(module.weight, operator_bound)
            if getattr(module, "bias", None) is not None:
                module.bias.clamp_(-bias_bound, bias_bound)
        elif isinstance(module, nn.MultiheadAttention):
            _project_matrix(module.in_proj_weight, operator_bound)
            if module.in_proj_bias is not None:
                module.in_proj_bias.clamp_(-bias_bound, bias_bound)

    radius = float(constraint["global_parameter_radius"])
    squared_norm = sum(parameter.detach().pow(2).sum() for parameter in model.parameters())
    norm = squared_norm.sqrt()
    if norm > radius:
        scale = radius / norm
        for parameter in model.parameters():
            parameter.mul_(scale)


@torch.no_grad()
def _project_matrix(weight: torch.Tensor, bound: float) -> None:
    if weight.ndim != 2:
        return
    norm = torch.linalg.matrix_norm(weight, ord=2)
    if norm > bound:
        weight.mul_(bound / norm)


@torch.no_grad()
def parameter_diagnostics(model: nn.Module) -> dict[str, float | int]:
    parameters = list(model.parameters())
    parameter_count = sum(item.numel() for item in parameters)
    global_norm = torch.sqrt(sum(item.detach().pow(2).sum() for item in parameters)).item()
    operator_norms = [torch.linalg.matrix_norm(item.detach(), ord=2).item() for item in parameters if item.ndim == 2]
    return {
        "parameter_count": int(parameter_count),
        "global_parameter_norm": float(global_norm),
        "max_matrix_operator_norm": float(max(operator_norms, default=0.0)),
    }
