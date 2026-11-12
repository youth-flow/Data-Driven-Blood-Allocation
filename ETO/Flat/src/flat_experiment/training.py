"""Flat v7 training: patient MSE plus residual-demand MSE, no PMF/convolution."""

import numpy as np
import torch
from torch import nn

from dat_study.runtime import cpu_state, device_for, prepare_fit, seed_everything

from .model import FlatMSE


def _fit_module(module, x, y, config, seed):
    seed_everything(seed)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_function = nn.MSELoss()
    features = torch.from_numpy(np.asarray(x, dtype=np.float32))
    target = torch.from_numpy(np.asarray(y, dtype=np.float32))
    batch_size = int(config["training"]["batch_size"])
    for epoch in range(int(config["training"]["epochs"])):
        module.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(len(target), generator=generator)
        for start in range(0, len(target), batch_size):
            index = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(
                module(features[index].to(device_for(config))).squeeze(-1),
                target[index].to(device_for(config)),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite Flat loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
    module.eval()
    return module


def _patient_frame(orders):
    return orders[["u", "r", "s"]].to_numpy(float), orders["real_demand"].to_numpy(float)


def _residual_target(data, daily):
    orders = data.order_split("train")
    known = orders.groupby("decision_day").real_demand.sum().reindex(daily.decision_day).fillna(0.0)
    return daily.D.to_numpy(float) - known.to_numpy(float)


@torch.no_grad()
def _predict(model, data, config):
    daily = data.daily_split("eval").reset_index(drop=True)
    orders = data.order_split("eval")
    groups = dict(tuple(orders.groupby("decision_day", sort=False)))
    x = data.context(daily).astype(np.float32)
    residual = model.residual_prediction(torch.from_numpy(x).to(device_for(config))).cpu().numpy()
    residual = np.clip(residual, 0.0, 13.0)
    values = []
    for index, day in enumerate(daily.decision_day):
        frame = groups.get(day)
        if frame is None or len(frame) == 0:
            known = 0.0
        else:
            patient = frame[["u", "r", "s"]].to_numpy(np.float32)
            prediction = model.patient_prediction(torch.from_numpy(patient).to(device_for(config))).cpu().numpy()
            known = float(np.clip(prediction, 0.0, 6.0).sum())
        values.append(np.clip(known + residual[index], 0.0, float(data.schema["demand_support_upper"])))
    return np.asarray(values)[:, None]


def fit_predict(data, config, recipe=None):
    prepare_fit(data, config, recipe)
    training = data.daily_split("train")
    orders = data.order_split("train")
    patient_x, patient_y = _patient_frame(orders)
    residual_x = data.context(training).astype(np.float32)
    residual_y = _residual_target(data, training)
    settings = config["flat"]
    model = FlatMSE(settings["patient_hidden"], settings["residual_hidden"], settings["dropout"]).to(
        device_for(config)
    )
    _fit_module(model.patient, patient_x, patient_y, config, config["seed"] + 1)
    _fit_module(model.residual, residual_x, residual_y, config, config["seed"] + 2)
    prediction = _predict(model, data, config)
    return prediction, {"epochs": int(config["training"]["epochs"])}, {"model_state": cpu_state(model)}
