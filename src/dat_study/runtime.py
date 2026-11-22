"""Model-neutral runtime helpers; never imports a model package."""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import DailySets, collate
from .evaluation import evaluate, selection_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Keep the same attention implementation across training and evaluation.
    torch.backends.mha.set_fastpath_enabled(False)


def device_for(config):
    name = config["training"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def loader(dataset, config, shuffle=False, epoch=0):
    shuffle_seed = int(config["training"].get("shuffle_seed", config["seed"]))
    generator = torch.Generator().manual_seed(shuffle_seed + epoch)
    training = config["training"]
    workers = int(training.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    pin_memory = bool(training.get("pin_memory", device_for(config).type == "cuda"))
    kwargs = {
        "dataset": dataset,
        "collate_fn": collate,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    batch_size = int(training["batch_size"])
    lengths = getattr(dataset, "lengths", None)
    if shuffle and bool(training.get("bucket_by_length", False)) and lengths is not None:
        order = np.argsort(np.asarray(lengths), kind="stable")
        batches = [order[start : start + batch_size].tolist() for start in range(0, len(order), batch_size)]
        permutation = torch.randperm(len(batches), generator=generator).tolist()
        kwargs["batch_sampler"] = [batches[i] for i in permutation]
    else:
        kwargs.update(batch_size=batch_size, shuffle=shuffle, generator=generator)
    if workers:
        kwargs.update(
            persistent_workers=bool(training.get("persistent_workers", True)),
            prefetch_factor=int(training.get("prefetch_factor", 4)),
        )
    return DataLoader(**kwargs)


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def score_prediction(prediction, dataset, data, config):
    daily = dataset.daily
    if config.get("selection_metric") == "pinball":
        residual = daily.D.to_numpy()[:, None] - prediction
        taus = np.asarray(config["taus"])[None, :]
        return -float(np.maximum(taus * residual, (taus - 1) * residual).mean())
    return selection_score(
        evaluate(
            prediction,
            daily.D.to_numpy(),
            daily.It.to_numpy(),
            data.daily_split("train").D.to_numpy(),
            config["taus"],
            config["targets"],
        )
    )


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def prepare_fit(data, config, recipe=None):
    seed_everything(config["seed"])
    torch.set_num_threads(config["training"]["threads"])
    training, evaluation = DailySets(data, "train"), DailySets(data, "eval")
    validation = evaluation if data.eval_role == "validation" else None
    epochs = int(config["training"]["epochs"])
    return training, evaluation, validation, epochs
