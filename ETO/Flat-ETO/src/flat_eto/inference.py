"""Flat-ETO distribution convolution and predictor reloading."""

import numpy as np
import torch
from dat_study.data import DailySets
from dat_study.runtime import device_for, seed_everything
from .model import FlatProbabilityMLP
from .quantity import independent_mixture_pmf, pmf_quantile


@torch.no_grad()
def predict_flat(model, quantity, remainder, dataset, data, config):
    model.eval()
    background = remainder.pmfs(data.context(dataset.daily))
    values = []
    for j, (sample, frame) in enumerate(zip(dataset.samples, dataset.frames)):
        if len(frame):
            day = data.context(dataset.daily.iloc[[j]])
            probability = (
                torch.sigmoid(
                    model(
                        sample["order_num"].to(device_for(config)),
                        sample["order_cat"].to(device_for(config)),
                        torch.from_numpy(np.repeat(day, len(frame), axis=0)).to(device_for(config)),
                    )
                )
                .cpu()
                .numpy()
            )
            ids = quantity.pool_ids(frame)
            known = independent_mixture_pmf(probability, [quantity.pmfs[k] for k in ids])
        else:
            known = np.array([1.0])
        full = np.convolve(known, background[j])
        full = np.maximum(full, 0)
        full /= full.sum()
        values.append([pmf_quantile(full, tau) for tau in config["taus"]])
    return np.asarray(values)


def predict_artifact(state, data, config):
    seed_everything(config["seed"])
    model = FlatProbabilityMLP(
        len(data.order_num_cols),
        len(data.day_cols) + 1,
        data.cardinalities,
        config["flat_eto"]["hidden_dims"],
        config["flat_eto"]["dropout"],
        config["flat_eto"]["embedding_dim_cap"],
    )
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device_for(config))
    return predict_flat(model, state["quantities"], state["remainder"], DailySets(data, "eval"), data, config)
