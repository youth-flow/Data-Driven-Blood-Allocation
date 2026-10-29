"""Set-ETO training, validation selection and prediction."""

import numpy as np
import torch
from dat_study.runtime import device_for, loader, to_device, score_prediction, prepare_fit, cpu_state
from .model import DeepSetsDistribution, distribution_nll
from .distributions import distribution_quantiles


def set_context(batch, data):
    result = dict(batch)
    inv = (batch["inventory"] - float(data.inventory_scaler.mean[0])) / float(data.inventory_scaler.scale[0])
    result["day_num"] = torch.cat([batch["day_num"], inv[:, None]], -1)
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
            loss = distribution_nll(config["set_eto"]["distribution"], model(set_context(batch, data)), batch["demand"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; no source records logged")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
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


def new_model(data, config):
    return DeepSetsDistribution(
        len(data.order_num_cols),
        len(data.day_cols) + 1,
        data.cardinalities,
        config["set_eto"],
        config["set_eto"]["distribution"],
    ).to(device_for(config))


def fit_predict(data, config, recipe=None):
    training, evaluation, validation, epochs = prepare_fit(data, config, recipe)
    model = new_model(data, config)
    result = fit_daily(model, "Set-ETO", training, data, config, epochs, validation)
    prediction = predict(model, "Set-ETO", evaluation, data, config)
    return prediction, result, {"model_state": cpu_state(model), "metadata": data.metadata()}
