"""Estimation, approximation and joint-consistency numerical experiments."""

from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from scipy import integrate
from dat_study.config import load_config
from dat_study.data import FoldData, NumericScaler, DailySets
from dat_study.runtime import device_for, seed_everything, cpu_state
from dat_study.registry import backend
from dat_study.io import save_json
from .generator import (
    simulate_split,
    fixed_scalers,
    support_upper,
    NUM_BASIC,
    NUM_CLINICAL,
    DAY_CONTEXT,
    DAY_BASIC,
    DAY_CLINICAL,
)
from .oracle import conditional_pmf, mixture_quantiles, cached_laws, assess_laws
from .jobs import run_job


class SimulationData(FoldData):
    """The normal model data interface backed only by freshly simulated memory."""

    def __init__(self, train, evaluation, config):
        self.eval_role = "validation"
        self.inventory_position = 2
        self.order_num_cols, self.order_cat_cols = NUM_BASIC + NUM_CLINICAL, []
        self.day_cols = DAY_CONTEXT + DAY_BASIC + DAY_CLINICAL
        self.daily = pd.concat(
            [train[0].assign(split="train"), evaluation[0].assign(split="validation")], ignore_index=True
        )
        self.orders = pd.concat([train[1], evaluation[1]], ignore_index=True)
        scales = fixed_scalers(config)
        self.order_scaler, self.day_scaler = (
            NumericScaler(self.order_num_cols, scales["order"]),
            NumericScaler(self.day_cols, scales["day"]),
        )
        self.inventory_scaler = NumericScaler(["It"], scales["inventory"])
        self.category_maps, self.cardinalities = {}, []
        self.schema = {
            "demand_support_upper": support_upper(config),
            "generator_version": 3,
            "max_orders": config["max_visible_orders"],
        }
        self.state = {"numeric_scalers": scales}


def dat_configuration(root, profile, tau, hidden, seed, architecture=None):
    cfg = load_config(root / "DAT/configs/main.yaml")
    cfg["seed"], cfg["taus"] = seed, [tau]
    cfg["loss"] = {"type": "pinball", "tau": tau, "taus": [tau]}
    cfg["model"]["d_model"] = profile["d_model"]
    cfg["model"]["patient_encoder"]["hidden_dims"] = [hidden]
    cfg["model"]["cross_attention"]["ff_dim"] = hidden
    cfg["model"]["head"]["output_upper_bound"] = profile["demand_upper"]
    if architecture:
        cfg["model"]["self_attention"]["num_heads"] = architecture["self_attention_heads"]
        cfg["model"]["cross_attention"]["num_heads"] = architecture["cross_attention_heads"]
        cfg["model"]["layer_norm_eps"] = architecture["layer_norm_eps"]
    for key in cfg["constraints"]:
        if key.endswith("bound") or key == "global_parameter_radius":
            cfg["constraints"][key] = profile["radius"]
    return cfg


_BOUND_CACHE = {}


def uniform_bound(config, count, tau, sample_size, delta, da=3, dx=8, bias_only=False):
    """Conservative legal parameter-Lipschitz envelope from PDF section 5.2.
    All operator/bias/LN envelopes are dominated by the declared Euclidean radius.
    The returned cap may be trivial; it is not fitted to observed errors.
    """
    radius = config["constraints"]["global_parameter_radius"]
    d = config["model"]["d_model"]

    def mlp(bound, dimensions):
        sensitivity = 0.0
        for _ in dimensions:
            sensitivity = np.sqrt(radius**2 * sensitivity**2 + bound**2 + 1)
            bound = radius * bound + radius
        return bound, sensitivity

    bpat, lpat = mlp(np.sqrt(da), config["model"]["patient_encoder"]["hidden_dims"] + [d])
    bday, lday = mlp(np.sqrt(dx), config["model"]["day_encoder"]["hidden_dims"] + [d])
    bn, ln, cn = radius * (np.sqrt(d) + 1), radius / np.sqrt(config["model"]["layer_norm_eps"]), np.sqrt(d + 1)

    def attention(heads):
        dh = d / heads
        bq = bu = np.sqrt(bn**2 + 1)
        ba = radius**2 * np.sqrt(heads) * bu + radius
        lq = radius**4 * np.sqrt(heads) * bu**2 / np.sqrt(dh)
        lu = radius**2 * np.sqrt(heads) + radius**4 * np.sqrt(heads) * bq * bu / np.sqrt(dh)
        ca2 = radius**2 * bu**2 * (heads + 1 + 2 * radius**4 * bq**2 / dh) + 1
        return ba, lq, lu, ca2

    _, lq, lu, ca2 = attention(config["model"]["self_attention"]["num_heads"])
    ls = lq + lu
    lset2 = (1 + ls * ln) ** 2 * lpat**2 + ca2 + (ls * cn) ** 2
    ba, lq, lu, ca2 = attention(config["model"]["cross_attention"]["num_heads"])
    bv = bday + ba
    au, ae = 1 + lq * ln, lu * ln
    cv2 = ca2 + (lq * cn) ** 2 + (lu * cn) ** 2
    cf2 = (radius * bv + radius) ** 2 + radius**2 * bv**2 + radius**2 + 1
    af = 1 + radius**2
    bplus = bv + radius * (radius * bv + radius) + radius
    lam = np.sqrt(radius**2 * ((af * au * lday) ** 2 + (af * ae) ** 2 * lset2 + af**2 * cv2 + cf2) + bplus**2 + 1)
    if bias_only:
        lam = 1.0
    b = config["model"]["head"]["output_upper_bound"]
    lt = max(tau, 1 - tau)
    key = (
        int(count),
        int(sample_size),
        round(float(radius), 10),
        round(float(lam), 10),
        round(float(b), 10),
        round(float(tau), 10),
        round(float(delta), 10),
        bool(bias_only),
    )
    if key in _BOUND_CACHE:
        return dict(_BOUND_CACHE[key])
    closed = 6 * lt * b * np.sqrt(count / sample_size) * (np.sqrt(np.log1p(8 * radius * lam / b)) + np.sqrt(np.pi) / 2)
    closed += lt * b * np.sqrt(2 * np.log(1 / delta) / sample_size)

    def entropy_log(s):
        return count * np.log1p(2 * radius * lam / s)

    def jensen_integral(a):
        if a <= 0:
            value, _ = integrate.quad(lambda s: np.sqrt(entropy_log(s)), 0.0, b / 4, epsabs=1e-10, limit=200)
        else:
            value, _ = integrate.quad(lambda s: np.sqrt(entropy_log(s)), a, b / 4, epsabs=1e-10, limit=200)
        return 4 * a + 12 / np.sqrt(sample_size) * value

    candidates = [0.0, *np.geomspace(b / 4 * 1e-7, b / 4 * 0.999999, 160)]
    j_value = min(b / 2, min(jensen_integral(a) for a in candidates))
    integral = min(lt * b, 2 * lt * j_value + lt * b * np.sqrt(2 * np.log(1 / delta) / sample_size))
    result = {
        "parameter_lipschitz_envelope": float(lam),
        "complexity_integral": float(j_value),
        "bound_closed": float(closed),
        "bound_integral": float(integral),
        "bound_capped": float(min(lt * b, closed, integral)),
        "bound_is_trivial": bool(min(closed, integral) >= lt * b),
    }
    _BOUND_CACHE[key] = result
    return dict(result)


def fit_network(data, config, epochs, objective, oracle_targets=None):
    trainer = backend("DAT", "training")
    from datransformer.constraints import project_parameters
    from datransformer.losses import pinball_loss

    seed_everything(config["seed"])
    torch.set_num_threads(config["training"]["threads"])
    model = trainer.new_dat(data, config)
    dataset = DailySets(data, "train")
    if oracle_targets is not None:
        for sample, target in zip(dataset.samples, oracle_targets):
            sample["demand"] = torch.tensor(float(target), dtype=torch.float32)
    device = device_for(config)
    batch_size = int(config["training"]["batch_size"])
    packed = _packed_dataset(dataset, device)
    count = len(dataset)

    def batches(indices):
        for start in range(0, count, batch_size):
            selection = indices[start : start + batch_size]
            yield {key: value[selection] for key, value in packed.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.0)
    best, state, history = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        permutation = torch.randperm(count, generator=generator).to(device)
        for batch in batches(permutation):
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)["quantile"]
            loss = (
                ((pred - batch["demand"]) ** 2).mean()
                if objective == "oracle_mse"
                else pinball_loss(pred, batch["demand"], config["taus"][0])
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite theory-training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            project_parameters(model, config)
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in batches(torch.arange(count, device=device)):
                pred = model(batch)["quantile"]
                value = (
                    ((pred - batch["demand"]) ** 2).mean()
                    if objective == "oracle_mse"
                    else pinball_loss(pred, batch["demand"], config["taus"][0])
                )
                total += value.item() * len(pred)
        score = total / count
        history.append({"epoch": epoch + 1, "full_training_objective": score})
        if score < best:
            best, state = score, cpu_state(model)
    model.load_state_dict(state)
    return model, best, history


def _packed_dataset(dataset, device):
    samples = dataset.samples
    count = len(samples)
    max_orders = max(1, max(len(sample["order_num"]) for sample in samples))
    numeric_dim = samples[0]["order_num"].shape[1]
    categorical_dim = samples[0]["order_cat"].shape[1]
    order_num = torch.zeros(count, max_orders, numeric_dim)
    order_cat = torch.zeros(count, max_orders, categorical_dim, dtype=torch.long)
    order_mask = torch.zeros(count, max_orders, dtype=torch.bool)
    for i, sample in enumerate(samples):
        n = len(sample["order_num"])
        order_num[i, :n] = sample["order_num"]
        order_cat[i, :n] = sample["order_cat"]
        order_mask[i, :n] = True
    packed = {
        "order_num": order_num,
        "order_cat": order_cat,
        "order_mask": order_mask,
        "day_features": torch.stack([sample["day_features"] for sample in samples]),
        "inventory": torch.stack([sample["inventory"] for sample in samples]),
        "demand": torch.stack([sample["demand"] for sample in samples]),
    }
    return {key: value.to(device) for key, value in packed.items()}


def _targets(sample, config, tau):
    groups = dict(tuple(sample[1].groupby("decision_day")))
    return np.array(
        [
            mixture_quantiles(
                conditional_pmf(groups[row["decision_day"]], row, config)[0], [tau], config.get("remainder_width", 0.5)
            )[0]
            for row in sample[0].to_dict("records")
        ]
    )


def run_theory(args, protocol, generator, root, output, groups, code_hash):
    settings, profile = protocol["theory_experiments"], protocol["profiles"]["theory"]
    tau = settings["tau"]
    sizes = args.train_sizes or ([24, 48] if args.smoke else settings["train_sizes"])
    widths = args.widths or ([16, 32] if args.smoke else settings["approximation"]["hidden_widths"])
    repeats = args.repetitions or (1 if args.smoke else settings["independent_data_repetitions"])
    model_seeds = [0] if args.smoke else (args.model_seeds or settings["model_seeds"])
    eval_count = args.eval_days or (12 if args.smoke else settings["evaluation_contexts_initial"])
    approximation_count = 24 if args.smoke else settings["approximation"].get("training_contexts", 10000)
    all_rows = []
    joint = settings["joint_consistency"]

    def growing_width(n):
        return max(1, int(joint["width_anchor"] * (n / joint["sample_anchor"]) ** joint["exponent"]))

    for group in groups:
        configured = (
            settings["estimation"].get("main_scenarios")
            if group == "estimation"
            else settings["approximation" if group == "approximation" else "joint_consistency"]["scenarios"]
        )
        scenarios = args.scenarios or (["S2"] if args.smoke else configured)
        if group == "estimation" and (args.scenarios is None or "C0" in args.scenarios):
            scenarios = list(dict.fromkeys([*scenarios, "C0"]))
        for scenario in scenarios:
            for repeat in range(args.repeat_start, args.repeat_start + repeats):
                g = {**generator, "max_visible_orders": profile["max_orders"], "scenario": scenario}
                data_seed = args.data_seed + repeat
                streams = np.random.SeedSequence([data_seed, 731]).spawn(2)
                nmax = approximation_count if group == "approximation" else max(sizes)
                training = simulate_split(g, nmax, "1850-01-01", streams[0], "train")
                evaluation = simulate_split(g, eval_count, "1700-01-01", streams[1], "evaluation")
                laws = cached_laws(
                    output / "oracle_cache",
                    {"seed": data_seed, "code": code_hash, "track": "theory"},
                    evaluation[0],
                    evaluation[1],
                    g,
                )
                choices = (
                    [(approximation_count, h, "capacity") for h in widths]
                    if group == "approximation"
                    else [(n, settings["estimation"]["hidden_width"], "fixed") for n in sizes]
                    if group == "estimation"
                    else [
                        (n, h, label)
                        for n in sizes
                        for h, label in [
                            (growing_width(n), "growing"),
                            *[(h, f"fixed{h}") for h in joint["fixed_width_controls"]],
                        ]
                    ]
                )
                oracle_training = _targets(training, g, tau) if group == "approximation" else None
                for n, hidden, schedule in choices:
                    modes = ["exact_bias_only"] if scenario == "C0" else ["numerical_DAT"]
                    for mode in modes:
                        name = f"T_{group}_{scenario}_r{repeat}_n{n}_h{hidden}_{schedule}_{mode}"
                        signature = {
                            "group": group,
                            "generator": g,
                            "data_seed": data_seed,
                            "n": n,
                            "hidden": hidden,
                            "schedule": schedule,
                            "mode": mode,
                            "evaluation": eval_count,
                            "protocol": protocol,
                            "smoke": args.smoke,
                            "code": code_hash,
                        }

                        def action(folder, n=n, hidden=hidden, mode=mode, schedule=schedule):
                            daily = training[0].iloc[:n].copy()
                            orders = training[1][training[1].decision_day.isin(daily.decision_day)].copy()
                            data = SimulationData((daily, orders), evaluation, g)
                            cfg = dat_configuration(root, profile, tau, hidden, 0, protocol["dat"])
                            trainer = backend("DAT", "training")
                            base = trainer.new_dat(
                                data,
                                dat_configuration(root, profile, tau, profile["patient_hidden"], 0, protocol["dat"]),
                            )
                            base_count = sum(p.numel() for p in base.parameters())
                            probe = trainer.new_dat(data, cfg)
                            parameter_count = sum(p.numel() for p in probe.parameters())
                            radius = (
                                profile["radius"]
                                if group == "estimation"
                                else profile["radius"] * (parameter_count / base_count) ** 2
                            )
                            for key in list(cfg["constraints"]):
                                if key.endswith("bound") or key == "global_parameter_radius":
                                    cfg["constraints"][key] = radius
                            cfg["training"].update(settings["training"])
                            cfg["training"].update(device=args.device, threads=args.threads)
                            if args.epochs is not None:
                                cfg["training"]["epochs"] = args.epochs
                            if args.patience is not None:
                                cfg["training"]["patience"] = args.patience
                            if args.smoke:
                                cfg["training"].update(epochs=1, batch_size=8)
                            if mode == "exact_bias_only":
                                values = np.sort(daily.D.to_numpy(float))
                                rank = int(np.ceil(tau * len(values)))
                                q = float(values[max(rank - 1, 0)])
                                prediction = np.full((eval_count, 1), q)
                                save_json(
                                    folder / "exact_erm.json",
                                    {
                                        "empirical_quantile": q,
                                        "class": "constant_output_bias",
                                        "definition": "inf{y: empirical_CDF(y) >= tau}",
                                    },
                                )
                                best, best_model, best_cfg, history = None, None, cfg, []
                            else:
                                best, best_model, best_cfg, history = np.inf, None, None, []
                                for seed in model_seeds:
                                    selected = deepcopy(cfg)
                                    selected["seed"] = seed
                                    model, score, trace = fit_network(
                                        data,
                                        selected,
                                        cfg["training"]["epochs"],
                                        "oracle_mse" if group == "approximation" else "pinball",
                                        oracle_training,
                                    )
                                    history.append({"seed": seed, "objective": score, "epochs": trace})
                                    if score < best:
                                        best, best_model, best_cfg = score, model, selected
                                prediction = trainer.predict(best_model, "DAT", DailySets(data, "eval"), data, best_cfg)
                                torch.save(
                                    {"state": cpu_state(best_model), "config": best_cfg, "metadata": data.metadata()},
                                    folder / "model.pt",
                                )
                            targets = protocol["performance_experiments"]["targets"]
                            assessed = assess_laws(laws, prediction, [tau], targets)
                            excess = assessed["excess_pinball"][:, 0]
                            error = prediction[:, 0] - assessed["oracle_quantiles"][:, 0]
                            class_parameters = 1 if mode == "exact_bias_only" else parameter_count
                            bounds = {
                                str(delta): uniform_bound(
                                    best_cfg, class_parameters, tau, n, delta, bias_only=mode == "exact_bias_only"
                                )
                                for delta in settings["estimation"]["confidence_delta"]
                            }
                            bound = uniform_bound(
                                best_cfg, class_parameters, tau, n, 0.05, bias_only=mode == "exact_bias_only"
                            )
                            parameter_norm = (
                                float(torch.sqrt(sum(p.detach().square().sum() for p in best_model.parameters())))
                                if best_model
                                else abs(q)
                            )
                            rows = []
                            for k, target in enumerate(targets):
                                regret = assessed["operational_regret"][:, 0, k]
                                rows.append(
                                    {
                                        "group": group,
                                        "scenario": scenario,
                                        "data_repeat": repeat,
                                        "n_train": n,
                                        "hidden": hidden,
                                        "schedule": schedule,
                                        "estimator": mode,
                                        "tau": tau,
                                        "target": target,
                                        "parameters": class_parameters,
                                        "network_parameters": parameter_count,
                                        "class_parameters": class_parameters,
                                        "parameter_radius": radius,
                                        "parameter_norm": parameter_norm,
                                        "complexity_ratio": float(
                                            class_parameters * np.log(np.e + class_parameters) / n
                                        ),
                                        "excess_pinball": float(excess.mean()),
                                        "evaluation_MCSE": float(excess.std(ddof=1) / np.sqrt(eval_count)),
                                        "operational_regret": float(regret.mean()),
                                        "oracle_L1": float(np.abs(error).mean()),
                                        "oracle_L2": float(np.sqrt(np.mean(error**2))),
                                        "sampled_max_error": float(np.max(np.abs(error))),
                                        "eval_point_max": float(np.max(np.abs(error))),
                                        "quantity": "achievable_approximation_error"
                                        if group == "approximation"
                                        else "estimation_error_realizable"
                                        if scenario == "C0"
                                        else "total_excess_risk",
                                        "exact_ERM_certified": mode == "exact_bias_only",
                                        "smoke": args.smoke,
                                        "oracle_quadrature_error": assessed["integration_error_max"],
                                        **bound,
                                        **{
                                            f"bound_delta_{delta}": item["bound_capped"]
                                            for delta, item in bounds.items()
                                        },
                                        **{
                                            f"exceeds_bound_delta_{delta}": bool(excess.mean() > item["bound_capped"])
                                            for delta, item in bounds.items()
                                        },
                                    }
                                )
                            np.savez_compressed(folder / "evaluation.npz", predictions=prediction, **assessed)
                            save_json(
                                folder / "training.json",
                                {
                                    "selection": "minimum_training_objective",
                                    "objective": "oracle_mse" if group == "approximation" else "pinball",
                                    "seeds": history,
                                    "selected_objective": best,
                                },
                            )
                            return {"rows": rows}

                        result = run_job(output / "jobs", name, signature, action, args.resume)
                        all_rows.extend(result["rows"])
                        pd.DataFrame(all_rows).to_csv(output / "theory_metrics.csv", index=False)
    return all_rows
