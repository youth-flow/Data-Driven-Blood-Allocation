"""DAT training, validation selection and prediction."""

import numpy as np
import torch
from dat_study.runtime import device_for, loader, to_device, score_prediction, prepare_fit, cpu_state
from .model import build_model
from .constraints import project_parameters
from .losses import multi_quantile_loss
from .experiments import experiment_config


def new_dat(data, config):
    config = experiment_config(config, data)
    initial = np.quantile(data.daily_split("train").D, config["taus"], method="higher")
    for j in range(len(initial)):
        initial[j] = max(initial[j], initial[j - 1] + 0.001 if j else 0.001)
    if initial[-1] > config["model"]["head"]["output_upper_bound"]:
        raise ValueError("Training quantiles exceed the configured output bound")
    config["model"]["head"]["initial_quantiles"] = initial.tolist()
    model = build_model(config, data.metadata()).to(device_for(config))
    project_parameters(model, config)
    return model


@torch.no_grad()
def predict(model, kind, dataset, data, config):
    model.eval()
    values = []
    for raw in loader(dataset, config):
        batch = to_device(raw, device_for(config))
        value = model(batch)["quantiles"].cpu().numpy()
        values.append(value)
    return np.concatenate(values)


def fit_daily(model, kind, training, data, config, epochs, validation=None):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    best, best_epoch, stale = -np.inf, 0, 0
    state, history = None, []
    for epoch in range(int(epochs)):
        model.train()
        loss_sum, count = 0.0, 0
        for raw in loader(training, config, True, epoch):
            batch = to_device(raw, device_for(config))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)["quantiles"]
            loss = multi_quantile_loss(
                prediction, batch["demand"], config["taus"], config["loss"]["type"], beta=config["loss"]["beta"]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; no source records logged")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            project_parameters(model, config)
            loss_sum += float(loss.detach()) * len(batch["demand"])
            count += len(batch["demand"])
        row = {"epoch": epoch + 1, "training_loss": loss_sum / count}
        if validation is not None:
            score = score_prediction(predict(model, kind, validation, data, config), validation, data, config)
            row["validation_mean_P"] = score
            if score > best:
                best, best_epoch, stale, state = score, epoch + 1, 0, cpu_state(model)
            else:
                stale += 1
        else:
            best_epoch = epoch + 1
        history.append(row)
        if validation is not None and stale >= config["training"]["patience"]:
            break
    if state is not None:
        model.load_state_dict(state)
    return {"epochs": best_epoch, "history": history}


def fit_predict(data, config, recipe=None):
    config = experiment_config(config, data)
    training, evaluation, validation, epochs = prepare_fit(data, config, recipe)
    model = new_dat(data, config)
    result = fit_daily(model, "DAT", training, data, config, epochs, validation)
    prediction = predict(model, "DAT", evaluation, data, config)
    return prediction, result, {"model_state": cpu_state(model), "metadata": data.metadata()}
