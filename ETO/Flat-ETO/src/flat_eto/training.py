"""Flat-ETO: patient seven-class PMF plus continuous residual-demand forest."""

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn

from dat_study.runtime import cpu_state, device_for, prepare_fit, seed_everything
from dat_study.tabular import PsiFeatures

from .model import FlatPatientPMF
from .quantity import NewDemandCDF, flat_quantile


def _patient_data(data, features, daily):
    orders = data.orders[
        data.orders.decision_day.isin(daily.decision_day)
        & data.orders.order_label_observed.astype(bool)
        & data.orders.real_demand.notna()
    ].copy()
    x = features.patient_input(data, orders)
    y = np.rint(2.0 * orders.real_demand.to_numpy(float)).astype(np.int64)
    if ((y < 0) | (y > 6)).any():
        raise ValueError("Flat-ETO patient labels must lie on the seven-class grid")
    day_lookup = {day: index for index, day in enumerate(daily.decision_day.to_numpy())}
    day_index = orders.decision_day.map(day_lookup).to_numpy(np.int64)
    return x, y, day_index


def _fit_patient_model(x, y, day_index, config):
    settings = config["flat_eto"]
    seed_everything(config["seed"])
    model = FlatPatientPMF(x.shape[1], settings["patient_hidden_dims"], settings["dropout"]).to(device_for(config))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_function = nn.CrossEntropyLoss(reduction="none")
    features = torch.from_numpy(x).float()
    labels = torch.from_numpy(y).long()
    days = torch.from_numpy(day_index).long()
    batch_size = int(config["training"]["batch_size"])
    count = len(labels)
    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        permutation = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            index = permutation[start : start + batch_size]
            batch_x = features[index].to(device_for(config))
            batch_y = labels[index].to(device_for(config))
            batch_days = days[index].to(device_for(config))
            optimizer.zero_grad(set_to_none=True)
            losses = loss_function(model(batch_x), batch_y)
            day_means = torch.stack([losses[batch_days == day].mean() for day in batch_days.unique()])
            loss = day_means.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite Flat-ETO patient loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
    model.eval()
    return model


def _fit_new_demand(data, features, config):
    daily = data.daily_split("train")
    psi = features.transform(data, daily)
    orders = data.order_split("train")
    known = orders.groupby("decision_day").real_demand.sum().reindex(daily.decision_day).fillna(0.0)
    residual = daily.D.to_numpy(float) - known.to_numpy(float)
    if (residual < 0).any():
        raise ValueError("Residual demand cannot be negative")
    forest = RandomForestRegressor(
        random_state=config["seed"],
        n_jobs=1,
        bootstrap=True,
        **config["flat_eto"]["new_demand_forest"],
    ).fit(psi, residual)
    return forest, forest.apply(psi), residual


def _forest_weights(forest, train_leaves, psi):
    query = forest.apply(psi)
    result = np.zeros((len(query), len(train_leaves)), dtype=np.float64)
    for tree_index in range(train_leaves.shape[1]):
        train_leaf = train_leaves[:, tree_index]
        for row, leaf in enumerate(query[:, tree_index]):
            members = train_leaf == leaf
            result[row, members] += 1.0 / members.sum()
    result /= train_leaves.shape[1]
    return result / result.sum(axis=1, keepdims=True)


@torch.no_grad()
def _predict(model, forest, train_leaves, residual, features, data, config):
    evaluation = data.daily_split("eval")
    orders = data.order_split("eval")
    groups = orders.groupby("decision_day", sort=False)
    psi = features.transform(data, evaluation)
    weights = _forest_weights(forest, train_leaves, psi)
    upper = float(data.schema["demand_support_upper"])
    values = []
    for row_index, day in enumerate(evaluation.decision_day):
        frame = groups.get_group(day)
        patient_x = features.patient_input(data, frame)
        probabilities = (
            torch.softmax(model(torch.from_numpy(patient_x).float().to(device_for(config))), dim=-1).cpu().numpy()
        )
        cdf = NewDemandCDF(residual, weights[row_index])
        values.append([flat_quantile(probabilities, cdf, tau, upper) for tau in config["taus"]])
    return np.asarray(values)


def fit_predict(data, config, recipe=None):
    prepare_fit(data, config, recipe)
    features = PsiFeatures().fit(data)
    daily = data.daily_split("train")
    x, y, day_index = _patient_data(data, features, daily)
    model = _fit_patient_model(x, y, day_index, config)
    forest, train_leaves, residual = _fit_new_demand(data, features, config)
    prediction = _predict(model, forest, train_leaves, residual, features, data, config)
    artifact = {
        "model_state": cpu_state(model),
        "forest": forest,
        "train_leaves": train_leaves,
        "residual": residual,
        "features": features,
    }
    return prediction, {"epochs": int(config["training"]["epochs"])}, artifact
