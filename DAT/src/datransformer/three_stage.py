"""Real order encoder pretraining, semi-synthetic days, real daily fine-tuning."""

from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .adaptation import (
    OrderAuxiliary,
    order_objective,
    LowRankUpdate,
    merge_lora,
    frozen_parameters,
    assert_frozen,
)
from .constraints import project_parameters
from dat_study.quantity import ConditionalQuantityPMF
from dat_study.synthetic import fit_remainder


def install_lora(model, rank):
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be positive")
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        if isinstance(module, nn.MultiheadAttention):
            weight = module.in_proj_weight
            parametrize.register_parametrization(
                module,
                "in_proj_weight",
                LowRankUpdate(*weight.shape, rank=rank, blocks=3, alpha=float(rank)).to(weight),
            )
            weight = module.out_proj.weight
            parametrize.register_parametrization(
                module.out_proj, "weight", LowRankUpdate(*weight.shape, rank=rank, alpha=float(rank)).to(weight)
            )
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                p.requires_grad_(True)
    for module in (model.day_encoder, model.quantile_head):
        for p in module.parameters():
            p.requires_grad_(True)
    return model


class ScenarioDays(Dataset):
    def __init__(self, real, labels):
        self.samples = []
        for sample, values in zip(real.samples, labels):
            for value in values:
                self.samples.append({**sample, "demand": torch.tensor(value, dtype=torch.float32)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def pretrain_encoder(model, data, config, epochs):
    from dat_study.runtime import device_for

    frame = data.order_split("train")
    frame = frame.loc[frame.order_label_observed & frame.real_demand.notna()].copy()
    if frame.empty:
        raise ValueError("No observed training order labels")
    auxiliary = OrderAuxiliary(model.patient_encoder, config["model"]["d_model"]).to(device_for(config))
    y = frame.real_demand.to_numpy(dtype=np.float32)
    scale = float(np.median(y[y > 0])) if (y > 0).any() else 1.0
    tensors = TensorDataset(
        torch.from_numpy(data.order_scaler.transform(frame)),
        torch.from_numpy(data.categories(frame)),
        torch.from_numpy(y),
    )
    optimizer = torch.optim.AdamW(
        auxiliary.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    history = []
    for epoch in range(int(epochs)):
        auxiliary.train()
        total = 0.0
        for num, cat, labels in DataLoader(
            tensors, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(config["seed"] + epoch)
        ):
            num, cat, labels = [x.to(device_for(config)) for x in (num, cat, labels)]
            optimizer.zero_grad(set_to_none=True)
            logits, quantity = auxiliary(num, cat)
            loss = order_objective(logits, quantity, labels, scale)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite order pretraining objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(auxiliary.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            project_parameters(auxiliary, config)
            total += float(loss.detach()) * len(labels)
        history.append({"epoch": epoch + 1, "loss": total / len(frame)})
    model.patient_encoder.load_state_dict(auxiliary.encoder.state_dict())
    project_parameters(model, config)
    auxiliary.eval()
    return auxiliary, frame, history


@torch.no_grad()
def make_scenarios(auxiliary, frame, training, data, config):
    from dat_study.runtime import device_for

    quantity = ConditionalQuantityPMF().fit(frame)
    remainder = fit_remainder(data, config["three_stage"]["residual_neighbors"])
    pools = remainder.pools(data.context(training.daily))
    rng = np.random.default_rng(config["seed"] + 17)
    repeats = int(config["three_stage"]["scenarios_per_day"])
    labels = []
    for j, (sample, orders) in enumerate(zip(training.samples, training.frames)):
        background = rng.choice(pools[j], repeats, replace=True)
        known = np.zeros(repeats)
        if len(orders):
            logits, _ = auxiliary(
                sample["order_num"].to(device_for(config)), sample["order_cat"].to(device_for(config))
            )
            probability = logits.sigmoid().cpu().numpy()
            ids = quantity.pool_ids(orders)
            for p, pool_id in zip(probability, ids):
                known += (rng.random(repeats) < p) * rng.choice(quantity.pools[pool_id], repeats, replace=True)
        labels.append(known + background)
    return ScenarioDays(training, labels)


def fit_three_stage(model, training, validation, data, config, final_epochs, recipe=None):
    from .trainer import fit_daily

    stage_config = deepcopy(config["three_stage"])
    if recipe is not None:
        stage_config.update(recipe["three_stage"])
    local_config = deepcopy(config)
    local_config["three_stage"] = stage_config
    auxiliary, frame, history1 = pretrain_encoder(model, data, local_config, stage_config["order_epochs"])
    scenarios = make_scenarios(auxiliary, frame, training, data, local_config)
    for p in model.patient_encoder.parameters():
        p.requires_grad_(False)
    frozen = frozen_parameters(model)
    stage2 = fit_daily(
        model, "DAT", scenarios, data, local_config, stage_config["synthetic_epochs"], frozen_projection=True
    )
    assert_frozen(model, frozen)
    if stage_config["finetune"] == "lora":
        install_lora(model, stage_config["lora_rank"])
        frozen = frozen_parameters(model)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    adapted_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = fit_daily(
        model,
        "DAT",
        training,
        data,
        local_config,
        final_epochs,
        validation,
        frozen_projection=stage_config["finetune"] == "lora",
    )
    if stage_config["finetune"] == "lora":
        assert_frozen(model, frozen)
        merge_lora(model)
    result.update(
        three_stage=stage_config,
        stage1_history=history1,
        stage2_history=stage2["history"],
        stage2_scenarios=len(scenarios),
        stage3_trainable_parameters=adapted_count,
        synthetic_demand="sampled visible-order issuance + training-only conditional remainder",
        exact_erm_claimed=False,
    )
    return result
