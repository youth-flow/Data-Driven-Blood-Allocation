from __future__ import annotations

import math

import torch
from torch import nn


class FlatProbabilityMLP(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        day_dim: int,
        category_cardinalities: list[int],
        hidden_dims: list[int],
        dropout: float,
        embedding_dim_cap: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        embedding_dims = []
        for cardinality in category_cardinalities:
            dimension = min(
                int(embedding_dim_cap),
                max(2, int(math.ceil(cardinality**0.25 * 2))),
            )
            self.embeddings.append(nn.Embedding(cardinality, dimension, padding_idx=0))
            embedding_dims.append(dimension)
        current = numeric_dim + day_dim + sum(embedding_dims)
        layers: list[nn.Module] = []
        for width in hidden_dims:
            layers.extend([nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)])
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, order_num, order_cat, day_num):
        embedded = [layer(order_cat[:, index]) for index, layer in enumerate(self.embeddings)]
        return self.network(torch.cat([order_num, day_num, *embedded], dim=-1)).squeeze(-1)
