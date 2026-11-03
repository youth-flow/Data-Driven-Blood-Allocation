"""Original scalar pinball loss and the fixed operational contract."""

import torch


def pinball_loss(prediction, demand, tau):
    residual = demand - prediction
    return torch.maximum(tau * residual, (tau - 1) * residual).mean()


def plugin_allocation(quantile, inventory, target):
    return (quantile - inventory + target).clamp_min(0)


def inventory_cost(allocation, demand, inventory, target, tau):
    deviation = inventory + allocation - demand - target
    return tau * (-deviation).clamp_min(0) + (1 - tau) * deviation.clamp_min(0)
