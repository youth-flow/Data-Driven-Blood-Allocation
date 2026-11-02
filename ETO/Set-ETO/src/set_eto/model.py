from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SUPPORTED_DISTRIBUTIONS = ("Normal", "NegativeBinomial", "HurdleGamma")


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float):
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)])
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class DeepSetsDistribution(nn.Module):
    """Pure DeepSets model: patient phi, sum pooling, day fusion, distribution head."""

    def __init__(self, order_num_dim, day_dim, category_cardinalities, config, distribution):
        super().__init__()
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution: {distribution}")
        self.distribution = distribution
        self.embeddings = nn.ModuleList()
        embedding_dims = []
        for cardinality in category_cardinalities:
            dimension = min(
                int(config["embedding_dim_cap"]),
                max(2, int(math.ceil(cardinality**0.25 * 2))),
            )
            self.embeddings.append(nn.Embedding(cardinality, dimension, padding_idx=0))
            embedding_dims.append(dimension)
        patient_dims = list(config["patient_hidden_dims"])
        day_dims = list(config["day_hidden_dims"])
        if not patient_dims or not day_dims:
            raise ValueError("Patient and day hidden dimensions must be nonempty.")
        dropout = float(config["dropout"])
        self.phi = _mlp(order_num_dim + sum(embedding_dims), patient_dims[:-1], patient_dims[-1], dropout)
        self.day_encoder = _mlp(day_dim, day_dims[:-1], day_dims[-1], dropout)
        output_dim = {"Normal": 2, "NegativeBinomial": 2, "HurdleGamma": 3}[distribution]
        self.head = _mlp(
            patient_dims[-1] + day_dims[-1],
            list(config["head_hidden_dims"]),
            output_dim,
            dropout,
        )
        self.parameter_floor = float(config.get("parameter_floor", 1e-3))

    def forward(self, batch):
        embedded = [layer(batch["order_cat"][:, :, index]) for index, layer in enumerate(self.embeddings)]
        patient = self.phi(torch.cat([batch["order_num"], *embedded], dim=-1))
        pooled = (patient * batch["order_mask"].unsqueeze(-1)).sum(dim=1)
        day = self.day_encoder(batch["day_num"])
        raw = self.head(torch.cat([pooled, day], dim=-1))
        floor = self.parameter_floor
        if self.distribution == "Normal":
            return {"mu": F.softplus(raw[:, 0]), "sigma": F.softplus(raw[:, 1]) + floor}
        if self.distribution == "NegativeBinomial":
            return {
                "mean_count": F.softplus(raw[:, 0]),
                "dispersion": F.softplus(raw[:, 1]) + floor,
            }
        return {
            "zero_logit": raw[:, 0],
            "shape": F.softplus(raw[:, 1]) + floor,
            "rate": F.softplus(raw[:, 2]) + floor,
        }


def distribution_nll(distribution, parameters, demand):
    if distribution == "Normal":
        return -torch.distributions.Normal(parameters["mu"], parameters["sigma"]).log_prob(demand).mean()
    if distribution == "NegativeBinomial":
        count = torch.round(demand * 2.0)
        mean = parameters["mean_count"].clamp_min(1e-8)
        dispersion = parameters["dispersion"]
        distribution_object = torch.distributions.NegativeBinomial(
            total_count=dispersion, logits=torch.log(mean / dispersion)
        )
        return -distribution_object.log_prob(count).mean()
    zero = demand <= 1e-8
    gamma = torch.distributions.Gamma(parameters["shape"], parameters["rate"])
    log_probability = torch.where(
        zero,
        F.logsigmoid(parameters["zero_logit"]),
        F.logsigmoid(-parameters["zero_logit"]) + gamma.log_prob(demand.clamp_min(1e-8)),
    )
    return -log_probability.mean()
