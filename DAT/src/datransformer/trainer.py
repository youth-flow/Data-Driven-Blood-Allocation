"""Independent scalar DAT fits at each tau; no shared multi-quantile model."""

from copy import deepcopy
from math import comb
import numpy as np
import torch
from dat_study.runtime import device_for, loader, to_device, prepare_fit, cpu_state
from .model import build_model
from .constraints import project_parameters
from .experiments import experiment_config
from .losses import pinball_loss


def scalar_config(config, tau):
    selected = deepcopy(config)
    selected["taus"] = [float(tau)]
    selected["loss"] = {"type": "pinball", "tau": float(tau), "taus": [float(tau)]}
    return selected


def new_dat(data, config):
    config = experiment_config(config, data)
    if len(config["taus"]) != 1:
        config = scalar_config(config, config["taus"][-1])
    if config["loss"]["type"] != "pinball":
        raise ValueError("DAT trains with original pinball loss")
    if "demand_support_upper" in data.schema:
        config["model"]["head"]["output_upper_bound"] = float(data.schema["demand_support_upper"])
    if "max_orders" in data.schema:
        m, da = data.schema["max_orders"], len(data.order_num_cols)
        dm = min(comb(da + m, m) - 1, 2 * m * da + 1)
        if config["model"]["d_model"] < max(dm + 2, dm + len(data.day_cols) + 1):
            raise ValueError("DAT latent width violates the declared synthetic approximation profile")
    model = build_model(config, data.metadata()).to(device_for(config))
    project_parameters(model, config)
    return model


@torch.no_grad()
def predict(model, kind, dataset, data, config):
    model.eval()
    return np.concatenate(
        [model(to_device(b, device_for(config)))["quantiles"].cpu().numpy() for b in loader(dataset, config)]
    )


def fit_daily(model, kind, training, data, config, epochs, validation=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.0)
    best, stale, best_epoch, state, history = np.inf, 0, 0, None, []
    tau = config["taus"][0]
    for epoch in range(int(epochs)):
        model.train()
        total, count = 0.0, 0
        for raw in loader(training, config, True, epoch):
            batch = to_device(raw, device_for(config))
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(batch)["quantile"], batch["demand"], tau)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite DAT loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            project_parameters(model, config)
            total += loss.item() * len(batch["demand"])
            count += len(batch["demand"])
        row = {"epoch": epoch + 1, "training_pinball": total / count}
        if validation is not None:
            q = predict(model, kind, validation, data, config)[:, 0]
            residual = validation.daily.D.to_numpy() - q
            score = float(np.maximum(tau * residual, (tau - 1) * residual).mean())
            row["validation_pinball"] = score
            if score < best:
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
    from .model import structural_parameter_counts

    return {"epochs": best_epoch, "history": history, **structural_parameter_counts(model)}


def fit_predict(data, config, recipe=None):
    predictions, models, results = [], [], {}
    for tau in config["taus"]:
        selected = scalar_config(config, tau)
        selected["seed"] = config["seed"] + int(round(tau * 10000))
        frozen = None
        if recipe is not None:
            if str(tau) not in recipe.get("selection", {}).get("tau_epochs", {}):
                raise ValueError("Scalar DAT refit requires frozen epochs for every tau")
            frozen = {"epochs": recipe["selection"]["tau_epochs"][str(tau)]}
        training, evaluation, validation, epochs = prepare_fit(data, selected, frozen)
        model = new_dat(data, selected)
        result = fit_daily(model, "DAT", training, data, selected, epochs, validation)
        predictions.append(predict(model, "DAT", evaluation, data, selected))
        results[str(tau)] = result
        if "demand_support_upper" in data.schema:
            selected["model"]["head"]["output_upper_bound"] = float(data.schema["demand_support_upper"])
        models.append({"tau": tau, "config": experiment_config(selected, data), "model_state": cpu_state(model)})
    return (
        np.concatenate(predictions, axis=1),
        {
            "epochs": max(r["epochs"] for r in results.values()),
            "selection": {"tau_epochs": {t: r["epochs"] for t, r in results.items()}},
            "scalar_fits": results,
            "training_mode": "independent_scalar_pinball",
        },
        {"scalar_models": models, "metadata": data.metadata()},
    )
