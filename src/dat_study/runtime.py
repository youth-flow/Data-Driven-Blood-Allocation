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
    # PyTorch's fused MHA inference can bypass parametrized LoRA out projections.
    torch.backends.mha.set_fastpath_enabled(False)


def device_for(config):
    name = config["training"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def loader(dataset, config, shuffle=False, epoch=0):
    generator = torch.Generator().manual_seed(config["seed"] + epoch)
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
        generator=generator,
    )


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def score_prediction(prediction, dataset, data, config):
    daily = dataset.daily
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
    if data.eval_role == "test" and recipe is None:
        raise ValueError("Final-year fitting requires a frozen development recipe")
    epochs = int(recipe["epochs"] if recipe else config["training"]["epochs"])
    return training, evaluation, validation, epochs
