"""Flat-ETO order-level probability training."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from dat_study.runtime import device_for, score_prediction, prepare_fit, cpu_state
from dat_study.synthetic import fit_remainder
from .model import FlatProbabilityMLP
from .quantity import ConditionalQuantityPMF
from .inference import predict_flat


def fit_flat(data, config, epochs, validation=None):
    frame = data.order_split("train")
    frame = frame.loc[frame.order_label_observed & frame.real_demand.notna()].copy()
    quantity = ConditionalQuantityPMF().fit(frame)
    remainder = fit_remainder(data, config["flat_eto"]["residual_neighbors"])
    model = FlatProbabilityMLP(
        len(data.order_num_cols),
        len(data.day_cols) + 1,
        data.cardinalities,
        config["flat_eto"]["hidden_dims"],
        config["flat_eto"]["dropout"],
        config["flat_eto"]["embedding_dim_cap"],
    ).to(device_for(config))
    daily = data.daily_split("train").set_index("decision_day")
    context = daily.loc[frame.decision_day].reset_index()
    tensors = [
        torch.from_numpy(data.order_scaler.transform(frame)),
        torch.from_numpy(data.categories(frame)),
        torch.from_numpy(data.context(context)),
        torch.tensor(frame.real_demand.gt(0).to_numpy(), dtype=torch.float32),
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    best, state, best_epoch, stale, history = -np.inf, None, 0, 0, []
    for epoch in range(int(epochs)):
        model.train()
        batches = DataLoader(
            TensorDataset(*tensors),
            batch_size=1024,
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"] + epoch),
        )
        total, count = 0.0, 0
        for num, cat, day, y in batches:
            num, cat, day, y = [t.to(device_for(config)) for t in (num, cat, day, y)]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(num, cat, day), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        row = {"epoch": epoch + 1, "training_loss": total / count}
        if validation is not None:
            score = score_prediction(
                predict_flat(model, quantity, remainder, validation, data, config), validation, data, config
            )
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
    return model, quantity, remainder, {"epochs": best_epoch, "history": history}


def fit_predict(data, config, recipe=None):
    _, evaluation, validation, epochs = prepare_fit(data, config, recipe)
    model, quantities, remainder, result = fit_flat(data, config, epochs, validation)
    prediction = predict_flat(model, quantities, remainder, evaluation, data, config)
    return (
        prediction,
        result,
        {
            "model_state": cpu_state(model),
            "metadata": data.metadata(),
            "quantities": quantities,
            "remainder": remainder,
        },
    )
