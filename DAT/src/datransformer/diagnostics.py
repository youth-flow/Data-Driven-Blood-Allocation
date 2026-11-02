"""Aggregate inventory sensitivity, with every non-inventory input held fixed."""

import numpy as np
import torch
from dat_study.runtime import loader, to_device, device_for


@torch.no_grad()
def inventory_sensitivity(model, dataset, config, step=10.0):
    if step <= 0:
        raise ValueError("Finite-difference step must be positive")
    model.eval()
    values = []
    for raw in loader(dataset, config):
        batch = to_device(raw, device_for(config))
        high, low = dict(batch), dict(batch)
        high["inventory"] = batch["inventory"] + step
        low["inventory"] = (batch["inventory"] - step).clamp_min(0)
        derivative = (model(high)["quantiles"] - model(low)["quantiles"]) / (high["inventory"] - low["inventory"])[
            :, None
        ]
        values.append(derivative.cpu().numpy())
    derivative = np.concatenate(values)
    return [
        {
            "tau": tau,
            "mean_dQ_dI": float(derivative[:, j].mean()),
            "p05": float(np.quantile(derivative[:, j], 0.05)),
            "p50": float(np.quantile(derivative[:, j], 0.5)),
            "p95": float(np.quantile(derivative[:, j], 0.95)),
            "share_above_one": float((derivative[:, j] > 1).mean()),
        }
        for j, tau in enumerate(config["taus"])
    ]
