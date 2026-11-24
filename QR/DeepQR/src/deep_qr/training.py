from math import cos, pi

import numpy as np
import torch
from dat_study.runtime import prepare_fit, device_for, loader, to_device, cpu_state, score_prediction
from .model import DeepSetsQuantiles, quantile_loss


def new_model(data, config):
    initial = np.quantile(data.daily_split("train").D, config["taus"], method="higher")
    for j in range(len(initial)):
        initial[j] = max(initial[j], initial[j - 1] + 0.001 if j else 0.001)
    if initial[-1] > config["deep_qr"]["output_upper_bound"]:
        raise ValueError("Training quantiles exceed configured output bound")
    return DeepSetsQuantiles(data.metadata(), config, initial).to(device_for(config))


@torch.no_grad()
def predict(model, dataset, config):
    model.eval()
    return np.concatenate([model(to_device(b, device_for(config))).cpu().numpy() for b in loader(dataset, config)])


def fit_predict(data, config, recipe=None):
    training, evaluation, validation, epochs = prepare_fit(data, config, recipe)
    model = new_model(data, config)
    settings = config["training"]
    decay = float(settings["weight_decay"])
    decay_parameters, no_decay_parameters = [], []
    for parameter in model.parameters():
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

    best, stale, best_epoch, state, history = -np.inf, 0, 0, None, []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for raw in loader(training, config, True, epoch):
            if max_steps is not None and global_step >= max_steps:
                break
            batch = to_device(raw, device_for(config))
            lr = learning_rate(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = quantile_loss(model(batch), batch["demand"], config)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite quantile training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            global_step += 1
            total += float(loss.detach()) * len(batch["demand"])
            count += len(batch["demand"])
        row = {
            "epoch": epoch + 1,
            "training_loss": total / count,
            "optimizer_steps": global_step,
            "learning_rate": learning_rate(max(0, global_step - 1)),
        }
        if validation is not None:
            score = score_prediction(predict(model, validation, config), validation, data, config)
            row["validation_score"] = score
            if score > best:
                best, stale, best_epoch, state = score, 0, epoch + 1, cpu_state(model)
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
        raise RuntimeError(f"DeepQR produced {global_step} optimizer steps; expected {max_steps}")
    if state is not None:
        model.load_state_dict(state)
    return (
        predict(model, evaluation, config),
        {"epochs": best_epoch, "optimizer_steps": global_step, "history": history},
        {"model_state": cpu_state(model)},
    )
