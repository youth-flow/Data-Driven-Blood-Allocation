"""Single-layer scalar DAT: equations (6)-(7) in the 2026-11-03 theory PDF."""

import torch
from torch import nn


class ConfigurableMlp(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", dropout=0.0):
        super().__init__()
        if activation != "relu" or dropout != 0 or len(hidden_dims) > 1:
            raise ValueError("DAT uses affine or two-affine-layer ReLU encoders and zero dropout")
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for j, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(a, b))
            if j < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatientEncoder(nn.Module):
    def __init__(self, numeric_dim, cardinalities, hidden, width):
        super().__init__()
        self.cardinalities = tuple(cardinalities)
        self.mlp = ConfigurableMlp(numeric_dim + sum(cardinalities), hidden, width)

    def forward(self, numeric, categorical):
        parts = [numeric]
        for j, count in enumerate(self.cardinalities):
            parts.append(torch.nn.functional.one_hot(categorical[..., j], count).to(numeric.dtype))
        return self.mlp(torch.cat(parts, -1))


class DecisionAwareTransformer(nn.Module):
    def __init__(
        self,
        order_num_dim,
        order_cat_cardinalities,
        day_num_dim,
        config,
        inventory_mean=0.0,
        inventory_std=1.0,
        inventory_position="end",
    ):
        super().__init__()
        model = config["model"]
        width, eps = int(model["d_model"]), float(model.get("layer_norm_eps", 1e-5))
        sa, ca = model["self_attention"], model["cross_attention"]
        if inventory_position not in {2, "end"}:
            raise ValueError("Inventory coordinate is either the synthetic X_3 or the final coordinate")
        self.inventory_position = inventory_position
        if eps <= 0 or any(width % int(section["num_heads"]) for section in (sa, ca)):
            raise ValueError("Positive LN epsilon and independently compatible head counts required")
        if any(int(section.get("num_layers", 1)) != 1 for section in (sa, ca)):
            raise ValueError("Exactly one self-attention and one cross-attention layer required")
        if "quantile_queries" in model or model["head"].get("hidden_dims", []):
            raise ValueError("Only the theory's scalar affine head is supported")
        if any(
            float(model[name].get("dropout", 0)) != 0
            for name in ("patient_encoder", "day_encoder", "self_attention", "cross_attention", "head")
        ):
            raise ValueError("Theory DAT disables dropout")
        levels = config.get("loss", {}).get("taus", [config.get("loss", {}).get("tau", 0.9)])
        if len(levels) != 1:
            raise ValueError("Construct one scalar DAT per tau")
        self.tau = float(levels[0])
        self.patient_encoder = PatientEncoder(
            order_num_dim, order_cat_cardinalities, model["patient_encoder"]["hidden_dims"], width
        )
        self.sa_norm = nn.LayerNorm(width, eps=eps)
        self.self_attention = nn.MultiheadAttention(width, int(sa["num_heads"]), batch_first=True)
        self.day_encoder = ConfigurableMlp(day_num_dim + 1, model["day_encoder"]["hidden_dims"], width)
        self.query_norm = nn.LayerNorm(width, eps=eps)
        self.memory_norm = nn.LayerNorm(width, eps=eps)
        self.cross_attention = nn.MultiheadAttention(width, int(ca["num_heads"]), batch_first=True)
        self.day_ffn = ConfigurableMlp(width, [int(ca["ff_dim"])], width)
        self.head = nn.Linear(width, 1)
        self.lower = float(model["head"].get("output_lower_bound", 0))
        self.upper = float(model["head"]["output_upper_bound"])
        if self.lower != 0 or self.upper <= 0 or inventory_std <= 0:
            raise ValueError("Invalid demand support or fixed inventory scaling")
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, float(model["head"].get("initial_quantile", self.upper / 4)))
        self.register_buffer("inventory_mean", torch.tensor(float(inventory_mean)))
        self.register_buffer("inventory_std", torch.tensor(float(inventory_std)))
        self.ablation = config.get("ablation", "full")
        if self.ablation not in {"full", "no_self_attention", "mean_pool", "sum_pool", "no_day_query"}:
            raise ValueError("Unknown structural intervention")
        if self.ablation == "no_day_query":
            self.constant_query = nn.Parameter(torch.zeros(1, 1, width))
        controls = config.get("input_controls", {})
        self.drop_set = bool(controls.get("drop_set", False))
        for name, count in (
            ("day", day_num_dim),
            ("order_num", order_num_dim),
            ("order_cat", len(order_cat_cardinalities)),
        ):
            values = controls.get(name, [True] * count)
            if len(values) != count:
                raise ValueError("Input mask width mismatch")
            self.register_buffer(f"{name}_gate", torch.tensor(values, dtype=torch.bool), persistent=False)

    def _day_input(self, batch):
        """Assemble the fixed eight-dimensional X; the inventory slot is always active."""
        if "day_features" in batch:
            day = batch["day_features"].clone()
            if self.inventory_position == 2:
                day[:, :2] = day[:, :2] * self.day_gate[:2]
                day[:, 3:] = day[:, 3:] * self.day_gate[2:]
            else:
                day[:, :-1] = day[:, :-1] * self.day_gate
            return day
        day_num = batch["day_num"] * self.day_gate
        inv = (batch["inventory"] - self.inventory_mean) / self.inventory_std
        if self.inventory_position == 2:
            return torch.cat([day_num[:, :2], inv[:, None], day_num[:, 2:]], -1)
        return torch.cat([day_num, inv[:, None]], -1)

    def forward(self, batch, return_attention=False):
        numeric = batch["order_num"] * self.order_num_gate
        categorical = batch["order_cat"] * self.order_cat_gate.to(torch.long)
        mask = batch["order_mask"]
        if self.drop_set:
            numeric, categorical = numeric[:, :1] * 0, categorical[:, :1] * 0
            mask = torch.zeros_like(mask[:, :1])
        empty = ~mask.any(1)
        safe_mask = mask.clone()
        safe_mask[empty, 0] = True
        tokens = self.patient_encoder(numeric, categorical)
        if self.ablation != "no_self_attention":
            norm = self.sa_norm(tokens)
            attention, _ = self.self_attention(norm, norm, norm, key_padding_mask=~safe_mask, need_weights=False)
            tokens = tokens + attention
        day = self._day_input(batch)
        u = self.day_encoder(day).unsqueeze(1)
        weights = None
        if self.ablation in {"mean_pool", "sum_pool"}:
            memory = self.memory_norm(tokens)
            w = self.cross_attention.in_proj_weight.chunk(3)[2]
            b = self.cross_attention.in_proj_bias.chunk(3)[2]
            values = torch.nn.functional.linear(memory, w, b) * mask[..., None]
            pooled = values.sum(1, keepdim=True)
            if self.ablation == "mean_pool":
                pooled = pooled / mask.sum(1).clamp_min(1)[:, None, None]
            attended = self.cross_attention.out_proj(pooled)
        else:
            query = self.constant_query.expand_as(u) if self.ablation == "no_day_query" else u
            memory = self.memory_norm(tokens)
            attended, weights = self.cross_attention(
                self.query_norm(query), memory, memory, key_padding_mask=~safe_mask, need_weights=return_attention
            )
        attended = attended.masked_fill(empty[:, None, None], 0)
        v = u + attended
        fused = v + self.day_ffn(v)
        raw = self.head(fused).squeeze(-1).squeeze(-1)
        quantile = raw.clamp(self.lower, self.upper)
        return {"quantile": quantile, "quantiles": quantile[:, None], "raw_quantile": raw, "attention_weights": weights}


def build_model(config, metadata):
    return DecisionAwareTransformer(
        metadata["order_num_dim"],
        metadata["order_cat_cardinalities"],
        metadata["day_num_dim"],
        config,
        metadata["inventory_mean"],
        metadata["inventory_std"],
        metadata.get("inventory_position", "end"),
    )


def structural_parameter_counts(model):
    allocated = sum(p.numel() for p in model.parameters())
    inactive = 0
    if model.ablation == "no_self_attention":
        inactive += sum(p.numel() for module in (model.sa_norm, model.self_attention) for p in module.parameters())
    if model.ablation in {"mean_pool", "sum_pool"}:
        width = model.cross_attention.embed_dim
        inactive += 2 * (width * width + width) + sum(p.numel() for p in model.query_norm.parameters())
    return {"allocated_parameters": allocated, "structurally_active_parameters": allocated - inactive}
