"""Set-ETO training, validation selection and prediction."""

from math import cos, pi

import numpy as np
import torch
from dat_study.runtime import device_for, loader, to_device, score_prediction, prepare_fit, cpu_state
from .model import DeepSetsDistribution, distribution_nll
from .distributions import distribution_quantiles


def set_context(batch, data):
    result = dict(batch)
    if "day_features" not in batch:
        inv = (batch["inventory"] - float(data.inventory_scaler.mean[0])) / float(data.inventory_scaler.scale[0])
        result["day_features"] = torch.cat([batch["day_num"], inv[:, None]], -1)
    return result


@torch.no_grad()
def predict(model, kind, dataset, data, config):
    model.eval()
    values = []
    for raw in loader(dataset, config):
        batch = to_device(raw, device_for(config))
        parameters = {k: v.cpu().numpy() for k, v in model(set_context(batch, data)).items()}
        value = distribution_quantiles(config["set_eto"]["distribution"], parameters, config["taus"])
        values.append(value)
    return np.concatenate(values)


def fit_daily(model, kind, training, data, config, epochs, validation=None):
    settings = config["training"]
    decay = float(settings["weight_decay"])
    decay_parameters, no_decay_parameters = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay_parameters if parameter.ndim > 1 else no_decay_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=float(settings["learning_rate"]),
        betas=tuple(settings.get("betas", (0.9, 0.999))),
        eps=float(settings.get("eps", 1e-8)),
    )
    max_steps = settings.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    peak_lr = float(settings["learning_rate"])
    if max_steps is None:
        warmup_steps, start_lr, minimum_lr = 0, peak_lr, peak_lr
    else:
        warmup_steps = min(int(settings.get("warmup_steps", 300)), max(1, max_steps // 4))
        start_lr = float(settings.get("warmup_start_lr", 3e-5))
        minimum_lr = float(settings.get("minimum_lr", 3e-5))

    def learning_rate(step):
        if max_steps is None:
            return peak_lr
        if warmup_steps and step < warmup_steps:
            return start_lr + (peak_lr - start_lr) * step / warmup_steps
        denominator = max(1, max_steps - warmup_steps - 1)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
        return minimum_lr + 0.5 * (peak_lr - minimum_lr) * (1.0 + cos(pi * progress))

    best, best_epoch, stale = -np.inf, 0, 0
    state, history = None, []
    global_step = 0
    for epoch in range(int(epochs)):
        model.train()
        loss_sum, count = 0.0, 0
        for raw in loader(training, config, True, epoch):
            if max_steps is not None and global_step >= max_steps:
                break
            batch = to_device(raw, device_for(config))
            lr = learning_rate(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = distribution_nll(config["set_eto"]["distribution"], model(set_context(batch, data)), batch["demand"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; no source records logged")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            global_step += 1
            loss_sum += float(loss.detach()) * len(batch["demand"])
            count += len(batch["demand"])
        row = {
            "epoch": epoch + 1,
            "training_loss": loss_sum / count,
            "optimizer_steps": global_step,
            "learning_rate": learning_rate(max(0, global_step - 1)),
        }
        if validation is not None:
            score = score_prediction(predict(model, kind, validation, data, config), validation, data, config)
            row["validation_score"] = score
            if score > best:
                best, best_epoch, stale, state = score, epoch + 1, 0, cpu_state(model)
            else:
                stale += 1
        else:
            best_epoch = epoch + 1
        history.append(row)
        if validation is not None and stale >= config["training"]["patience"]:
            break
        if max_steps is not None and global_step >= max_steps:
            break
    if max_steps is not None and global_step != max_steps:
        raise RuntimeError(f"Set-ETO produced {global_step} optimizer steps; expected {max_steps}")
    if state is not None:
        model.load_state_dict(state)
    return {"epochs": best_epoch, "optimizer_steps": global_step, "history": history}


def new_model(data, config):
    return DeepSetsDistribution(
        len(data.order_num_cols),
        len(data.day_cols) + 1,
        data.cardinalities,
        config["set_eto"],
        config["set_eto"]["distribution"],
    ).to(device_for(config))


def fit_predict(data, config, recipe=None):
    if config["set_eto"]["distribution"] != "HurdleGamma":
        raise ValueError("The S4 protocol fixes Set-ETO to HurdleGamma")
    training, evaluation, validation, epochs = prepare_fit(data, config, recipe)
    model = new_model(data, config)
    result = fit_daily(model, "Set-ETO", training, data, config, epochs, validation)
    prediction = predict(model, "Set-ETO", evaluation, data, config)
    return prediction, result, {"model_state": cpu_state(model), "metadata": data.metadata()}
