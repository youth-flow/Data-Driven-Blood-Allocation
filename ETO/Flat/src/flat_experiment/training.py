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
    max_steps = config["training"].get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    step = 0
    for epoch in range(int(config["training"]["epochs"])):
        module.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        if max_steps is None:
            permutation = torch.randperm(len(target), generator=generator)
            indices = (
                permutation[start : start + batch_size]
                for start in range(0, len(target), batch_size)
            )
        elif len(target) <= 2_000_000:
            permutation = torch.randperm(len(target), generator=generator)
            indices = (
                permutation[start : start + batch_size]
                for start in range(0, min(len(target), (max_steps - step) * batch_size), batch_size)
            )
        else:
            needed = (max_steps - step) * batch_size
            if needed > len(target):
                raise RuntimeError("Flat fixed-step sample exceeds the available unique records")
            selection = np.random.default_rng(seed + epoch).choice(len(target), size=needed, replace=False)
            permutation = torch.from_numpy(selection.astype(np.int64, copy=False))
            indices = (
                permutation[start : start + batch_size] for start in range(0, needed, batch_size)
            )
        for index in indices:
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
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    if max_steps is not None and step != max_steps:
        raise RuntimeError(f"Flat module produced {step} optimizer steps; expected {max_steps}")
    module.eval()
    return module


def _patient_columns(orders):
    return ["u", "r", "hb"] if "hb" in orders.columns else ["u", "r", "s"]


def _patient_frame(orders):
    return orders[_patient_columns(orders)].to_numpy(float), orders["real_demand"].to_numpy(float)


def _residual_target(data, daily):
    if data.compact_format == "axid_v10_csr_v1":
        _, labels, offsets, _ = data.patient_arrays("train")
        known = np.add.reduceat(np.asarray(labels), offsets[:-1])
        return daily.D.to_numpy(float) - known
    orders = data.order_split("train")
    known = orders.groupby("decision_day").real_demand.sum().reindex(daily.decision_day).fillna(0.0)
    return daily.D.to_numpy(float) - known.to_numpy(float)


@torch.no_grad()
def _predict(model, data, config):
    daily = data.daily_split("eval").reset_index(drop=True)
    x = data.context(daily).astype(np.float32)
    residual = model.residual_prediction(torch.from_numpy(x).to(device_for(config))).cpu().numpy()
    residual = np.clip(residual, 0.0, float(data.schema["demand_support_upper"]))
    if data.compact_format == "axid_v10_csr_v1":
        patients, _, offsets, _ = data.patient_arrays("eval")
        patients = ((np.asarray(patients) - data.order_scaler.mean) / data.order_scaler.scale).astype(np.float32)
        chunks = []
        for start in range(0, len(patients), 65536):
            tensor = torch.from_numpy(patients[start : start + 65536]).to(device_for(config))
            chunks.append(model.patient_prediction(tensor).cpu().numpy())
        patient_prediction = np.clip(np.concatenate(chunks), 0.0, 10.0)
        known = np.add.reduceat(patient_prediction, offsets[:-1])
        return np.clip(known + residual, 0.0, float(data.schema["demand_support_upper"]))[:, None]
    orders = data.order_split("eval")
    groups = dict(tuple(orders.groupby("decision_day", sort=False)))
    values = []
    for index, day in enumerate(daily.decision_day):
        frame = groups.get(day)
        if frame is None or len(frame) == 0:
            known = 0.0
        else:
            patient = frame[_patient_columns(frame)].to_numpy(np.float32)
            prediction = model.patient_prediction(torch.from_numpy(patient).to(device_for(config))).cpu().numpy()
            known = float(np.clip(prediction, 0.0, 6.0).sum())
        values.append(np.clip(known + residual[index], 0.0, float(data.schema["demand_support_upper"])))
    return np.asarray(values)[:, None]


def fit_predict(data, config, recipe=None):
    prepare_fit(data, config, recipe)
    training = data.daily_split("train")
    if data.compact_format == "axid_v10_csr_v1":
        patients, patient_y, _, _ = data.patient_arrays("train")
        patient_x = ((np.asarray(patients) - data.order_scaler.mean) / data.order_scaler.scale).astype(np.float32)
    else:
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
