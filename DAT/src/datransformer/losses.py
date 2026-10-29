from __future__ import annotations

import torch
from torch.nn import functional as F


def pinball_loss(prediction: torch.Tensor, demand: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    tau_tensor = torch.as_tensor(tau, dtype=prediction.dtype, device=prediction.device)
    residual = demand - prediction
    return torch.maximum(tau_tensor * residual, (tau_tensor - 1.0) * residual).mean()


def softplus_pinball_loss(
    prediction: torch.Tensor,
    demand: torch.Tensor,
    tau: float | torch.Tensor,
    beta: float,
) -> torch.Tensor:
    tau_tensor = torch.as_tensor(tau, dtype=prediction.dtype, device=prediction.device)
    residual = demand - prediction
    return (
        tau_tensor * F.softplus(residual, beta=float(beta))
        + (1.0 - tau_tensor) * F.softplus(-residual, beta=float(beta))
    ).mean()


def huber_pinball_loss(
    prediction: torch.Tensor,
    demand: torch.Tensor,
    tau: float | torch.Tensor,
    delta: float,
) -> torch.Tensor:
    tau_tensor = torch.as_tensor(tau, dtype=prediction.dtype, device=prediction.device)
    residual = demand - prediction
    magnitude = residual.abs()
    delta_value = float(delta)
    huber = torch.where(
        magnitude < delta_value,
        magnitude.square() / (2.0 * delta_value),
        magnitude - delta_value / 2.0,
    )
    weight = torch.where(residual >= 0.0, tau_tensor, 1.0 - tau_tensor)
    return (weight * huber).mean()


def quantile_loss(
    prediction: torch.Tensor,
    demand: torch.Tensor,
    tau: float | torch.Tensor,
    loss_type: str,
    beta: float = 6.0,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    normalized = str(loss_type).lower()
    if normalized == "pinball":
        return pinball_loss(prediction, demand, tau)
    if normalized in {"softplus", "softplus_pinball"}:
        return softplus_pinball_loss(prediction, demand, tau, beta)
    if normalized in {"huber", "huber_pinball"}:
        return huber_pinball_loss(prediction, demand, tau, huber_delta)
    raise ValueError(f"Unsupported quantile loss: {loss_type}")


def multi_quantile_loss(
    predictions: torch.Tensor,
    demand: torch.Tensor,
    taus: torch.Tensor | list[float] | tuple[float, ...],
    loss_type: str,
    beta: float = 6.0,
    huber_delta: float = 1.0,
    tau_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    if predictions.ndim != 2:
        raise ValueError("Multi-quantile predictions must have shape [batch, n_tau].")
    if demand.ndim != 1 or demand.shape[0] != predictions.shape[0]:
        raise ValueError("Demand must have shape [batch] and align with predictions.")
    tau = torch.as_tensor(taus, dtype=predictions.dtype, device=predictions.device)
    if tau.ndim != 1 or tau.numel() != predictions.shape[1]:
        raise ValueError("taus must have shape [n_tau] and align with predictions.")
    residual = demand.unsqueeze(-1) - predictions
    tau = tau.unsqueeze(0)
    normalized = str(loss_type).lower()
    if normalized == "pinball":
        losses = torch.maximum(tau * residual, (tau - 1.0) * residual)
    elif normalized in {"softplus", "softplus_pinball"}:
        losses = tau * torch.nn.functional.softplus(residual, beta=float(beta)) + (
            1.0 - tau
        ) * torch.nn.functional.softplus(-residual, beta=float(beta))
    elif normalized in {"huber", "huber_pinball"}:
        absolute = residual.abs()
        delta = float(huber_delta)
        huber = torch.where(
            absolute <= delta,
            0.5 * absolute.square() / delta,
            absolute - 0.5 * delta,
        )
        losses = torch.where(residual >= 0.0, tau, 1.0 - tau) * huber
    else:
        raise ValueError(f"Unsupported quantile loss: {loss_type}")
    if tau_weights is not None:
        weights = torch.as_tensor(tau_weights, dtype=predictions.dtype, device=predictions.device)
        if weights.ndim != 1 or weights.numel() != predictions.shape[1]:
            raise ValueError("tau_weights must have shape [n_tau].")
        if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
            raise ValueError("tau_weights must be finite and positive.")
        losses = losses * (weights / weights.mean()).unsqueeze(0)
    return losses.mean()


def plugin_allocation(quantile: torch.Tensor, inventory: torch.Tensor, target: float) -> torch.Tensor:
    return torch.clamp(quantile - inventory + float(target), min=0.0)


def inventory_cost(
    allocation: torch.Tensor,
    demand: torch.Tensor,
    inventory: torch.Tensor,
    target: float,
    tau: float | torch.Tensor,
) -> torch.Tensor:
    tau_tensor = torch.as_tensor(tau, dtype=demand.dtype, device=demand.device)
    deviation = inventory + allocation - demand - float(target)
    return tau_tensor * F.relu(-deviation) + (1.0 - tau_tensor) * F.relu(deviation)
