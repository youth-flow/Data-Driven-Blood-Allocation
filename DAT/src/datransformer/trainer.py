"""Independent scalar DAT fits at each tau; no shared multi-quantile model."""

from copy import deepcopy
from math import comb, cos, pi
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
    enforce_width = data.schema.get("enforce_constructive_width", True)
    if "max_orders" in data.schema and enforce_width:
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
    packed = _packed_dataset(dataset, device_for(config), config)
    if packed is not None:
        values = []
        for start in range(0, len(dataset), 1024):
            batch = {key: value[start : start + 1024] for key, value in packed.items()}
            values.append(model(batch)["quantiles"].cpu().numpy())
        return np.concatenate(values)
    return np.concatenate(
        [model(to_device(b, device_for(config)))["quantiles"].cpu().numpy() for b in loader(dataset, config)]
    )


def _packed_dataset(dataset, device, config):
    """Packed tensors avoid per-batch collation for the small tabular sets used here."""
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None
    maximum = max(len(sample["order_num"]) for sample in samples)
    if maximum > 32 or len(samples) > 100_000:
        return None
    numeric_dim = samples[0]["order_num"].shape[1]
    categorical_dim = samples[0]["order_cat"].shape[1]
    order_num = torch.zeros(len(samples), maximum, numeric_dim)
    order_cat = torch.zeros(len(samples), maximum, categorical_dim, dtype=torch.long)
    order_mask = torch.zeros(len(samples), maximum, dtype=torch.bool)
    for i, sample in enumerate(samples):
        n = len(sample["order_num"])
        order_num[i, :n] = sample["order_num"]
        order_cat[i, :n] = sample["order_cat"]
        order_mask[i, :n] = True
    result = {
        "order_num": order_num,
        "order_cat": order_cat,
        "order_mask": order_mask,
        "day_features": torch.stack([sample["day_features"] for sample in samples]),
        "inventory": torch.stack([sample["inventory"] for sample in samples]),
        "demand": torch.stack([sample["demand"] for sample in samples]),
    }
    return {key: value.to(device) for key, value in result.items()}


@torch.no_grad()
def _predict_packed(model, packed):
    values = []
    for start in range(0, len(packed["demand"]), 1024):
        batch = {key: value[start : start + 1024] for key, value in packed.items()}
        values.append(model(batch)["quantiles"].cpu().numpy())
    return np.concatenate(values)


def fit_daily(model, kind, training, data, config, epochs, validation=None):
    training_config = config["training"]
    decay = float(training_config.get("weight_decay", 0.0))
    decay_parameters, no_decay_parameters = [], []
    for parameter in model.parameters():
        (decay_parameters if parameter.ndim > 1 else no_decay_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=float(training_config["learning_rate"]),
        betas=tuple(training_config.get("betas", (0.9, 0.999))),
        eps=float(training_config.get("eps", 1e-8)),
    )
    best, stale, best_epoch, state, history = np.inf, 0, 0, None, []
    tau = config["taus"][0]
    device = device_for(config)
    packed = _packed_dataset(training, device, config)
    validation_packed = _packed_dataset(validation, device, config) if validation is not None else None
    max_steps = training_config.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    warmup_steps = int(training_config.get("warmup_steps", 0))
    peak_lr = float(training_config["learning_rate"])
    start_lr = float(training_config.get("warmup_start_lr", peak_lr))
    minimum_lr = float(training_config.get("minimum_lr", peak_lr))
    global_step = 0

    def learning_rate(step):
        if max_steps is None:
            return peak_lr
        if warmup_steps and step < warmup_steps:
            return start_lr + (peak_lr - start_lr) * step / warmup_steps
        denominator = max(1, max_steps - warmup_steps - 1)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
        return minimum_lr + 0.5 * (peak_lr - minimum_lr) * (1.0 + cos(pi * progress))

    for epoch in range(int(epochs)):
        model.train()
        total, count = 0.0, 0
        if packed is not None:
            generator = torch.Generator().manual_seed(config["seed"] + epoch)
            permutation = torch.randperm(len(packed["demand"]), generator=generator).to(device)
            batches = (
                {
                    key: value[permutation[start : start + config["training"]["batch_size"]]]
                    for key, value in packed.items()
                }
                for start in range(0, len(permutation), config["training"]["batch_size"])
            )
        else:
            batches = (to_device(b, device) for b in loader(training, config, True, epoch))
        for batch in batches:
            if max_steps is not None and global_step >= max_steps:
                break
            lr = learning_rate(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(batch)["quantile"], batch["demand"], tau)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite DAT loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            project_parameters(model, config)
            global_step += 1
            total += loss.item() * len(batch["demand"])
            count += len(batch["demand"])
        row = {
            "epoch": epoch + 1,
            "training_pinball": total / count,
            "optimizer_steps": global_step,
            "learning_rate": learning_rate(max(0, global_step - 1)),
        }
        if validation is not None:
            q = (
                _predict_packed(model, validation_packed)[:, 0]
                if validation_packed is not None
                else predict(model, kind, validation, data, config)[:, 0]
            )
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
        if max_steps is not None and global_step >= max_steps:
            break
    if max_steps is not None and global_step != max_steps:
        raise RuntimeError(f"Training produced {global_step} optimizer steps; expected {max_steps}")
    if state is not None:
        model.load_state_dict(state)
    from .model import structural_parameter_counts

    return {
        "epochs": best_epoch,
        "optimizer_steps": global_step,
        "history": history,
        **structural_parameter_counts(model),
    }


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
        selected["model"]["head"]["initial_quantile"] = float(
            np.quantile(training.daily.D.to_numpy(float), tau, method="higher")
        )
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
