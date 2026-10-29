from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from .losses import quantile_loss


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class OrderAuxiliary(nn.Module):
    def __init__(self, encoder, dimension=48):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.issue_head = nn.Linear(dimension, 1)
        self.quantity_head = nn.Linear(dimension, 1)

    def forward(self, numeric, categorical):
        h = self.encoder(numeric, categorical)
        return self.issue_head(h).squeeze(-1), F.softplus(self.quantity_head(h).squeeze(-1))


def order_objective(logits, quantity, y, scale):
    positive = y.gt(0)
    classification = F.binary_cross_entropy_with_logits(logits, positive.float())
    regression = (
        quantile_loss(quantity[positive], y[positive] / scale, 0.9, "pinball") if positive.any() else quantity.sum() * 0
    )
    return classification + regression


class LowRankUpdate(nn.Module):
    def __init__(self, rows, columns, rank=4, blocks=1, alpha=4.0):
        super().__init__()
        self.blocks, self.scaling = blocks, alpha / rank
        self.A = nn.Parameter(torch.empty(blocks, rank, columns))
        self.B = nn.Parameter(torch.zeros(blocks, rows // blocks, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

    def delta(self):
        return (self.B @ self.A).flatten(0, 1) * self.scaling

    def forward(self, base):
        return base + self.delta()


def merge_lora(model):
    for module in list(model.modules()):
        for name in ("in_proj_weight", "weight"):
            if parametrize.is_parametrized(module, name):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)
    return model


@torch.no_grad()
def project_trainable(model, config):
    """Constrain effective weights, without modifying frozen base parameters."""
    c = config["constraints"]
    if not bool(c.get("enabled", True)):
        return
    bound = float(c["operator_norm_bound"])
    for module in model.modules():
        for name in ("in_proj_weight", "weight"):
            if parametrize.is_parametrized(module, name):
                plist = getattr(module.parametrizations, name)
                base, adapter = plist.original, plist[0]
                if torch.linalg.matrix_norm(getattr(module, name), ord=2) > bound + 1e-6:
                    delta = adapter.delta()
                    left, right = 0.0, 1.0
                    for _ in range(18):
                        mid = (left + right) / 2
                        if torch.linalg.matrix_norm(base + mid * delta, ord=2) <= bound + 1e-6:
                            left = mid
                        else:
                            right = mid
                    adapter.B.mul_(left)
        if isinstance(module, nn.LayerNorm):
            for name, limit in [("weight", c["layer_norm_gain_bound"]), ("bias", c["layer_norm_shift_bound"])]:
                p = getattr(module, name)
                if p is not None and p.requires_grad:
                    p.clamp_(-limit, limit)
        elif isinstance(module, (nn.Linear, nn.Embedding, nn.MultiheadAttention)):
            name = "in_proj_weight" if isinstance(module, nn.MultiheadAttention) else "weight"
            if not parametrize.is_parametrized(module, name):
                weight = getattr(module, name)
                if weight.requires_grad:
                    magnitude = torch.linalg.matrix_norm(weight, ord=2)
                    if magnitude > bound:
                        weight.mul_(bound / magnitude)
            bias = getattr(module, "in_proj_bias" if isinstance(module, nn.MultiheadAttention) else "bias", None)
            if bias is not None and bias.requires_grad:
                bias.clamp_(-c["bias_bound"], c["bias_bound"])
    # The original hypothesis-class radius concerns the merged DAT, not the
    # arbitrary factorization of its low-rank update. Exclude each base/A/B
    # parameter set and replace it by its effective matrix exactly once.
    replaced_parameters, effective_weights = set(), []
    for module in model.modules():
        for name in ("in_proj_weight", "weight"):
            if parametrize.is_parametrized(module, name):
                plist = getattr(module.parametrizations, name)
                replaced_parameters.update(id(p) for p in plist.parameters())
                effective_weights.append(getattr(module, name))
    norm_squared = sum(p.square().sum() for p in model.parameters() if id(p) not in replaced_parameters)
    norm_squared += sum(weight.square().sum() for weight in effective_weights)
    # Fail rather than silently change frozen matrices if the radius activates.
    if norm_squared.sqrt() > float(c["global_parameter_radius"]):
        raise RuntimeError("Global radius became active; frozen-safe projection needs review")


def frozen_parameters(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad}


def assert_frozen(model, state):
    current = dict(model.named_parameters())
    if not all(torch.equal(current[name].detach().cpu(), value) for name, value in state.items()):
        raise AssertionError("A frozen parameter changed")
