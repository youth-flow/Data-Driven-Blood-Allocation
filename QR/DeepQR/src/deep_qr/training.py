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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    best, stale, best_epoch, state, history = -np.inf, 0, 0, None, []
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for raw in loader(training, config, True, epoch):
            batch = to_device(raw, device_for(config))
            optimizer.zero_grad(set_to_none=True)
            loss = quantile_loss(model(batch), batch["demand"], config)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite quantile training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(batch["demand"])
            count += len(batch["demand"])
        row = {"epoch": epoch + 1, "training_loss": total / count}
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
    if state is not None:
        model.load_state_dict(state)
    return (
        predict(model, evaluation, config),
        {"epochs": best_epoch, "history": history},
        {"model_state": cpu_state(model)},
    )
