from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def build_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class ConfigurableMlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(value) for value in hidden_dims], int(output_dim)]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(build_activation(activation))
                if dropout > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def embedding_dim(cardinality: int) -> int:
    return int(min(32, max(4, round(1.6 * cardinality**0.56))))


class PatientEncoder(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: Sequence[int],
        hidden_dims: Sequence[int],
        d_model: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.numeric_dim = int(numeric_dim)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(int(cardinality), embedding_dim(int(cardinality)))
                for cardinality in categorical_cardinalities
            ]
        )
        input_dim = self.numeric_dim + sum(item.embedding_dim for item in self.embeddings)
        self.mlp = ConfigurableMlp(input_dim, hidden_dims, d_model, activation, dropout)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.numeric_dim:
            parts.append(numeric)
        for index, embedding in enumerate(self.embeddings):
            parts.append(embedding(categorical[..., index]))
        return self.mlp(torch.cat(parts, dim=-1))


class PreNormSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, patient_tokens: torch.Tensor, patient_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(patient_tokens)
        empty = ~patient_mask.any(dim=1)
        safe_mask = patient_mask.clone()
        safe_mask[empty, 0] = True
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        attended = attended.masked_fill(empty[:, None, None], 0.0)
        state = patient_tokens + self.attention_dropout(attended)
        return state + self.ffn_dropout(self.ffn(self.ffn_norm(state)))


class PatientSelfAttentionStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        num_heads: int,
        ff_dim: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PreNormSelfAttentionBlock(
                    d_model,
                    num_heads,
                    ff_dim,
                    activation,
                    dropout,
                    layer_norm_eps,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, patient_tokens: torch.Tensor, patient_mask: torch.Tensor) -> torch.Tensor:
        state = patient_tokens
        for block in self.blocks:
            state = block(state, patient_mask)
        return state


class PreNormCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.memory_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        day_token: torch.Tensor,
        patient_tokens: torch.Tensor,
        patient_mask: torch.Tensor,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self.query_norm(day_token)
        memory = self.memory_norm(patient_tokens)
        empty = ~patient_mask.any(dim=1)
        safe_mask = patient_mask.clone()
        safe_mask[empty, 0] = True
        attended, weights = self.attention(
            query,
            memory,
            memory,
            key_padding_mask=~safe_mask,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        attended = attended.masked_fill(empty[:, None, None], 0.0)
        if weights is not None:
            weights = weights.masked_fill(empty[:, None, None], 0.0)
        state = day_token + self.attention_dropout(attended)
        state = state + self.ffn_dropout(self.ffn(self.ffn_norm(state)))
        attention = weights.squeeze(1) if weights is not None else None
        return state, attention


class DecisionAwareTransformer(nn.Module):
    """Pre-norm DAT with either a scalar or monotone multi-quantile head."""

    def __init__(
        self,
        order_num_dim: int,
        order_cat_cardinalities: Sequence[int],
        day_num_dim: int,
        config: dict[str, Any],
        inventory_mean: float = 0.0,
        inventory_std: float = 1.0,
    ) -> None:
        super().__init__()
        model_config = config["model"]
        d_model = int(model_config["d_model"])
        layer_norm_eps = float(model_config.get("layer_norm_eps", 1e-5))
        patient_config = model_config["patient_encoder"]
        self_attention_config = model_config["self_attention"]
        day_config = model_config["day_encoder"]
        cross_config = model_config["cross_attention"]
        head_config = model_config["head"]
        self_heads = int(self_attention_config["num_heads"])
        cross_heads = int(cross_config["num_heads"])
        for label, heads in (("self_attention", self_heads), ("cross_attention", cross_heads)):
            if d_model % heads:
                raise ValueError(f"d_model={d_model} must be divisible by {label}.num_heads={heads}.")

        self.patient_encoder = PatientEncoder(
            order_num_dim,
            order_cat_cardinalities,
            patient_config["hidden_dims"],
            d_model,
            str(patient_config.get("activation", "relu")),
            float(patient_config.get("dropout", 0.0)),
        )
        self.use_self_attention = bool(self_attention_config.get("enabled", True))
        self.patient_stack = (
            PatientSelfAttentionStack(
                d_model=d_model,
                depth=int(self_attention_config["num_layers"]),
                num_heads=self_heads,
                ff_dim=int(self_attention_config["ff_dim"]),
                activation=str(self_attention_config.get("activation", "relu")),
                dropout=float(self_attention_config.get("dropout", 0.0)),
                layer_norm_eps=layer_norm_eps,
            )
            if self.use_self_attention
            else None
        )
        context_config = model_config["context"]
        self.include_inventory = bool(context_config.get("include_inventory", True))
        self.inventory_transform = str(context_config.get("inventory_transform", "standardize")).lower()
        inventory_clip = context_config.get("inventory_clip", 6.0)
        self.inventory_context_clip = None if inventory_clip is None else float(inventory_clip)
        day_input_dim = int(day_num_dim) + (1 if self.include_inventory else 0)
        self.day_encoder = ConfigurableMlp(
            day_input_dim,
            day_config["hidden_dims"],
            d_model,
            str(day_config.get("activation", "relu")),
            float(day_config.get("dropout", 0.0)),
        )
        self.register_buffer("inventory_mean", torch.tensor(float(inventory_mean), dtype=torch.float32))
        self.register_buffer("inventory_std", torch.tensor(max(float(inventory_std), 1.0), dtype=torch.float32))
        self.cross_blocks = nn.ModuleList(
            [
                PreNormCrossAttentionBlock(
                    d_model=d_model,
                    num_heads=cross_heads,
                    ff_dim=int(cross_config["ff_dim"]),
                    activation=str(cross_config.get("activation", "relu")),
                    dropout=float(cross_config.get("dropout", 0.0)),
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(int(cross_config.get("num_layers", 1)))
            ]
        )
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        loss_config = config.get("loss", {})
        configured_taus = loss_config.get("taus")
        self.quantile_levels = (
            tuple(float(value) for value in configured_taus)
            if configured_taus is not None
            else (float(loss_config.get("tau", 0.9)),)
        )
        if (
            not self.quantile_levels
            or tuple(sorted(set(self.quantile_levels))) != self.quantile_levels
            or any(not 0.0 < value < 1.0 for value in self.quantile_levels)
        ):
            raise ValueError("loss.taus must be unique, sorted, and lie in (0, 1).")
        self.multi_quantile = len(self.quantile_levels) > 1
        self.primary_tau = float(loss_config.get("primary_tau", loss_config.get("tau", 0.9)))
        if self.primary_tau not in self.quantile_levels:
            raise ValueError("loss.primary_tau must be one of the configured quantile levels.")
        self.primary_tau_index = self.quantile_levels.index(self.primary_tau)
        quantile_query_config = model_config.get("quantile_queries", {})
        self.use_quantile_queries = bool(quantile_query_config.get("enabled", False))
        if self.use_quantile_queries and not self.multi_quantile:
            raise ValueError("Quantile-conditioned queries require multiple quantile levels.")
        self.quantile_query_encoder: ConfigurableMlp | None = None
        self.quantile_increment_bias: nn.Parameter | None = None
        if self.use_quantile_queries:
            self.quantile_query_encoder = ConfigurableMlp(
                2,
                quantile_query_config.get("hidden_dims", [32]),
                d_model,
                str(quantile_query_config.get("activation", "relu")),
                float(quantile_query_config.get("dropout", 0.0)),
            )
            tau_tensor = torch.tensor(self.quantile_levels, dtype=torch.float32)
            tau_inputs = torch.stack(
                [tau_tensor, torch.logit(tau_tensor)],
                dim=-1,
            )
            self.register_buffer("quantile_query_inputs", tau_inputs)
            self.quantile_increment_bias = nn.Parameter(torch.empty(len(self.quantile_levels), dtype=torch.float32))
        self.quantile_head = ConfigurableMlp(
            d_model,
            head_config.get("hidden_dims", []),
            1 if self.use_quantile_queries else len(self.quantile_levels),
            str(head_config.get("activation", "relu")),
            float(head_config.get("dropout", 0.0)),
        )
        self.output_lower_bound = float(head_config.get("output_lower_bound", 0.0))
        self.output_upper_bound = float(head_config["output_upper_bound"])
        if self.multi_quantile:
            initial_quantiles = head_config.get("initial_quantiles")
            if initial_quantiles is None:
                raise ValueError("A multi-quantile DAT head requires model.head.initial_quantiles.")
            self._initialize_multi_head([float(value) for value in initial_quantiles])
        else:
            self._initialize_head(float(head_config.get("initial_quantile", self.output_upper_bound / 4.0)))

    def _initialize_head(self, initial_quantile: float) -> None:
        final_linear = next(module for module in reversed(self.quantile_head.net) if isinstance(module, nn.Linear))
        nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(
            final_linear.bias,
            min(max(initial_quantile, self.output_lower_bound), self.output_upper_bound),
        )

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0.0:
            raise ValueError("Monotone quantile initialization increments must be positive.")
        return float(value + math.log(-math.expm1(-value)))

    def _initialize_multi_head(self, initial_quantiles: list[float]) -> None:
        if len(initial_quantiles) != len(self.quantile_levels):
            raise ValueError("model.head.initial_quantiles must match loss.taus.")
        if initial_quantiles != sorted(initial_quantiles):
            raise ValueError("model.head.initial_quantiles must be sorted.")
        if initial_quantiles[0] <= self.output_lower_bound:
            raise ValueError("The first initial quantile must exceed output_lower_bound.")
        if initial_quantiles[-1] > self.output_upper_bound:
            raise ValueError("Initial quantiles must not exceed output_upper_bound.")
        increments = [
            initial_quantiles[0] - self.output_lower_bound,
            *[current - previous for previous, current in zip(initial_quantiles, initial_quantiles[1:])],
        ]
        if any(value <= 0.0 for value in increments):
            raise ValueError("Initial quantiles must be strictly increasing.")
        final_linear = next(module for module in reversed(self.quantile_head.net) if isinstance(module, nn.Linear))
        nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
        inverse_increments = torch.tensor(
            [self._inverse_softplus(value) for value in increments],
            dtype=final_linear.bias.dtype,
            device=final_linear.bias.device,
        )
        with torch.no_grad():
            if self.use_quantile_queries:
                if self.quantile_increment_bias is None:
                    raise RuntimeError("Quantile increment bias was not initialized.")
                final_linear.bias.zero_()
                self.quantile_increment_bias.copy_(inverse_increments)
            else:
                final_linear.bias.copy_(inverse_increments)

    def _day_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.include_inventory:
            return batch["day_num"]
        inventory = batch["inventory"].to(dtype=batch["day_num"].dtype)
        inventory_context = (
            (inventory - self.inventory_mean) / self.inventory_std
            if self.inventory_transform == "standardize"
            else inventory
        )
        if self.inventory_context_clip is not None:
            inventory_context = inventory_context.clamp(
                min=-self.inventory_context_clip,
                max=self.inventory_context_clip,
            )
        return torch.cat([batch["day_num"], inventory_context.unsqueeze(-1)], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor], return_attention: bool = False) -> dict[str, torch.Tensor | None]:
        patient_tokens = self.patient_encoder(batch["order_num"], batch["order_cat"])
        if self.patient_stack is not None:
            patient_tokens = self.patient_stack(patient_tokens, batch["order_mask"])
        day_token = self.day_encoder(self._day_context(batch)).unsqueeze(1)
        if self.use_quantile_queries:
            if self.quantile_query_encoder is None:
                raise RuntimeError("Quantile query encoder was not initialized.")
            query_offsets = self.quantile_query_encoder(self.quantile_query_inputs)
            day_state = day_token + query_offsets.unsqueeze(0)
        else:
            day_state = day_token
        attention = None
        for cross_block in self.cross_blocks:
            day_state, attention = cross_block(
                day_state,
                patient_tokens,
                batch["order_mask"],
                return_attention,
            )
        head_output = self.quantile_head(self.output_norm(day_state))
        if self.use_quantile_queries:
            if self.quantile_increment_bias is None:
                raise RuntimeError("Quantile increment bias was not initialized.")
            raw = head_output.squeeze(-1) + self.quantile_increment_bias.unsqueeze(0)
        else:
            raw = head_output.squeeze(1)
        if self.multi_quantile:
            positive_parts = F.softplus(raw)
            quantiles = torch.cumsum(positive_parts, dim=-1) + self.output_lower_bound
            quantiles = torch.clamp(quantiles, max=self.output_upper_bound)
            primary = quantiles[:, self.primary_tau_index]
            return {
                "quantile": primary,
                "quantiles": quantiles,
                "raw_quantiles": raw,
                "quantile_levels": torch.as_tensor(
                    self.quantile_levels,
                    dtype=quantiles.dtype,
                    device=quantiles.device,
                ),
                "attention_weights": attention,
            }
        raw_quantile = raw.squeeze(-1)
        quantile = torch.clamp(
            raw_quantile,
            min=self.output_lower_bound,
            max=self.output_upper_bound,
        )
        return {
            "quantile": quantile,
            "raw_quantile": raw_quantile,
            "attention_weights": attention,
        }


def build_model(config: dict[str, Any], metadata: dict[str, Any]) -> DecisionAwareTransformer:
    return DecisionAwareTransformer(
        order_num_dim=int(metadata["order_num_dim"]),
        order_cat_cardinalities=[int(value) for value in metadata["order_cat_cardinalities"]],
        day_num_dim=int(metadata["day_num_dim"]),
        config=config,
        inventory_mean=float(metadata["inventory_mean"]),
        inventory_std=float(metadata["inventory_std"]),
    )
