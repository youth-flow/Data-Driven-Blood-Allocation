import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, widths, output_dim, dropout):
    layers = []
    for width in widths:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)])
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class DeepSetsQuantiles(nn.Module):
    def __init__(self, metadata, config, initial):
        super().__init__()
        settings = config["deep_qr"]
        sizes = [
            min(settings["embedding_dim_cap"], max(2, math.ceil(c**0.25 * 2)))
            for c in metadata["order_cat_cardinalities"]
        ]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, s, padding_idx=0) for c, s in zip(metadata["order_cat_cardinalities"], sizes)]
        )
        patient, day = settings["patient_hidden_dims"], settings["day_hidden_dims"]
        dropout = settings["dropout"]
        self.phi = mlp(metadata["order_num_dim"] + sum(sizes), patient[:-1], patient[-1], dropout)
        self.day_encoder = mlp(metadata["day_num_dim"] + 1, day[:-1], day[-1], dropout)
        self.head = mlp(patient[-1] + day[-1], settings["head_hidden_dims"], len(initial), dropout)
        self.upper = float(settings["output_upper_bound"])
        increments = np.diff(np.r_[0.0, initial]).clip(0.001)
        nn.init.normal_(self.head[-1].weight, std=0.01)
        with torch.no_grad():
            bias = increments + np.log(-np.expm1(-increments))
            self.head[-1].bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
        self.register_buffer("inventory_mean", torch.tensor(metadata["inventory_mean"]))
        self.register_buffer("inventory_std", torch.tensor(metadata["inventory_std"]))

    def forward(self, batch):
        embedded = [layer(batch["order_cat"][:, :, j]) for j, layer in enumerate(self.embeddings)]
        tokens = self.phi(torch.cat([batch["order_num"], *embedded], dim=-1))
        pooled = (tokens * batch["order_mask"].unsqueeze(-1)).sum(dim=1)
        inventory = (batch["inventory"] - self.inventory_mean) / self.inventory_std
        day = self.day_encoder(torch.cat([batch["day_num"], inventory[:, None]], dim=-1))
        raw = self.head(torch.cat([pooled, day], dim=-1))
        return F.softplus(raw).cumsum(dim=-1).clamp(max=self.upper)


def quantile_loss(prediction, demand, config):
    residual = demand[:, None] - prediction
    tau = prediction.new_tensor(config["taus"])[None, :]
    if config["loss"]["type"] == "pinball":
        return torch.maximum(tau * residual, (tau - 1) * residual).mean()
    if config["loss"]["type"] != "softplus_pinball":
        raise ValueError("Unsupported quantile loss")
    beta = config["loss"]["beta"]
    return (tau * F.softplus(residual, beta=beta) + (1 - tau) * F.softplus(-residual, beta=beta)).mean()
