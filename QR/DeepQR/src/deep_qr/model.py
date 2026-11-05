import math
import torch
from torch import nn


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
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.constant_(self.head[-1].bias, min(float(initial[0]), self.upper / 2))
        self.register_buffer("inventory_mean", torch.tensor(metadata["inventory_mean"]))
        self.register_buffer("inventory_std", torch.tensor(metadata["inventory_std"]))
        self.inventory_position = metadata.get("inventory_position", "end")

    def forward(self, batch):
        embedded = [layer(batch["order_cat"][:, :, j]) for j, layer in enumerate(self.embeddings)]
        tokens = self.phi(torch.cat([batch["order_num"], *embedded], dim=-1))
        mask = batch["order_mask"].to(tokens.dtype)
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        if "day_features" in batch:
            day_input = batch["day_features"]
        else:
            inventory = (batch["inventory"] - self.inventory_mean) / self.inventory_std
            day_input = torch.cat([batch["day_num"], inventory[:, None]], dim=-1)
        day = self.day_encoder(day_input)
        return self.head(torch.cat([pooled, day], dim=-1)).clamp(0.0, self.upper)


def quantile_loss(prediction, demand, config):
    residual = demand[:, None] - prediction
    tau = prediction.new_tensor(config["taus"])[None, :]
    return torch.maximum(tau * residual, (tau - 1) * residual).mean()
