"""Conditional law oracle; neither a class-optimal DAT nor a realized-future oracle."""

import numpy as np
from scipy.integrate import quad_vec
from scipy.stats import binom, norm
from .generator import coefficients, order_parameters, new_parameters, sigmoid

_GH_CACHE = {}


def _gauss_hermite(nodes):
    if nodes not in _GH_CACHE:
        points, weights = np.polynomial.hermite_e.hermegauss(nodes)
        _GH_CACHE[nodes] = (points, weights / np.sqrt(2 * np.pi))
    return _GH_CACHE[nodes]


def _pmf_at_shocks(orders, context, config, shocks):
    """Conditional PMF of D_0 on the half-unit grid for each fixed latent shock."""
    logit, quantity = order_parameters(orders, context, config)
    rho = coefficients(config)[2]
    shocks = np.atleast_1d(np.asarray(shocks, dtype=float))
    count = len(shocks)
    positive = binom.pmf(np.arange(6)[None, :], 5, quantity[:, None])
    probabilities = sigmoid(logit[None, :] + rho * shocks[:, None])
    pmf = np.ones((count, 1))
    for i, probability in enumerate(probabilities.T):
        one = np.empty((count, 7), dtype=float)
        one[:, 0] = 1.0 - probability
        one[:, 1:] = probability[:, None] * positive[i]
        pmf = np.stack([np.convolve(row, column) for row, column in zip(pmf, one)])
    arrival, amount = new_parameters(context, config, shocks)
    arrival, amount = np.broadcast_to(arrival, (count,)), np.broadcast_to(amount, (count,))
    for _ in range(config["max_new_orders"]):
        one = np.empty((count, 7), dtype=float)
        one[:, 0] = 1.0 - arrival
        one[:, 1:] = arrival[:, None] * binom.pmf(np.arange(6)[None, :], 5, amount[:, None])
        pmf = np.stack([np.convolve(row, column) for row, column in zip(pmf, one)])
    return pmf


def conditional_pmf(orders, context, config):
    rho = coefficients(config)[2]
    if rho == 0:
        pmf = _pmf_at_shocks(orders, context, config, [0.0])[0]
        return pmf, {"method": "finite_convolution", "quadrature_error_estimate": 0.0, "tail_mass_bound": 0.0}

    method = config.get("oracle_quadrature", "gauss_hermite")
    if method == "gauss_hermite":
        nodes = int(config.get("oracle_nodes", 25))
        points, weights = _gauss_hermite(nodes)
        values = _pmf_at_shocks(orders, context, config, points)
        pmf = np.tensordot(weights, values, axes=(0, 0))
        tail = float(2 * norm.sf(points[-1]))
        return pmf, {
            "method": "gauss_hermite",
            "nodes": nodes,
            "quadrature_error_estimate": tail,
            "tail_mass_bound": tail,
            "error_is_numerical_estimate_not_certificate": True,
        }

    cutoff = float(config.get("oracle_normal_cutoff", 9))
    tolerance = float(config.get("oracle_atol", 1e-10))
    pmf, error = quad_vec(
        lambda x: _pmf_at_shocks(orders, context, config, [x])[0] * norm.pdf(x),
        -cutoff,
        cutoff,
        epsabs=tolerance,
        epsrel=1e-9,
    )
    tail, mass = float(2 * norm.sf(cutoff)), float(pmf.sum())
    if not np.isfinite(pmf).all() or mass <= 0 or abs(mass - 1) > max(1e-7, 10 * (tail + error)):
        raise RuntimeError("Conditional law integration failed")
    pmf = np.maximum(pmf, 0) / mass
    return pmf, {
        "method": "adaptive_normal_integration",
        "quadrature_error_estimate": float(error),
        "tail_mass_bound": tail,
        "error_is_numerical_estimate_not_certificate": True,
    }


def mixture_quantiles(pmf, taus, width=0.5):
    cumulative = np.cumsum(pmf)
    cumulative[-1] = 1.0
    taus = np.asarray(taus, float)
    k = np.searchsorted(cumulative, taus, side="left")
    k = np.minimum(k, len(pmf) - 1)
    previous = np.where(k > 0, cumulative[np.maximum(k - 1, 0)], 0)
    return 0.5 * k + (width * (taus - previous) / pmf[k] if width else 0)


def conditional_pinball(pmf, prediction, tau, width=0.5):
    lower = 0.5 * np.arange(len(pmf))
    q = np.asarray(prediction, float)[..., None]
    if width:
        shortfall = (np.maximum(q - lower, 0) ** 2 - np.maximum(q - lower - width, 0) ** 2) / (2 * width)
        mean = lower + width / 2
    else:
        shortfall, mean = np.maximum(q - lower, 0), lower
    return np.sum(pmf * (tau * (mean - q) + shortfall), axis=-1)


def prepare_laws(daily, orders, config):
    groups = dict(tuple(orders.groupby("decision_day", sort=False)))
    size = 6 * (config["max_visible_orders"] + config["max_new_orders"]) + 1
    laws = np.zeros((len(daily), size))
    if config.get("scenario") == "C0" and len(daily):
        first = daily.iloc[0]
        pmf, diagnostic = conditional_pmf(groups[first["decision_day"]], first, config)
        laws[:, : len(pmf)] = pmf
        return {
            "pmfs": laws,
            "inventory": daily.It.to_numpy(float),
            "width": np.asarray(config.get("remainder_width", 0.5)),
            "integration_error_max": np.asarray(diagnostic["quadrature_error_estimate"]),
            "tail_mass_bound_max": np.asarray(diagnostic["tail_mass_bound"]),
        }
    errors, tails = [], []
    for j, row in enumerate(daily.to_dict("records")):
        pmf, diagnostic = conditional_pmf(groups[row["decision_day"]], row, config)
        laws[j, : len(pmf)] = pmf
        errors.append(diagnostic["quadrature_error_estimate"])
        tails.append(diagnostic["tail_mass_bound"])
    return {
        "pmfs": laws,
        "inventory": daily.It.to_numpy(float),
        "width": np.asarray(config.get("remainder_width", 0.5)),
        "integration_error_max": np.asarray(max(errors)),
        "tail_mass_bound_max": np.asarray(max(tails)),
    }


def assess_laws(cache, predictions, taus, targets):
    pmfs, width = cache["pmfs"], float(cache["width"])
    predictions = np.asarray(predictions, float)
    taus = np.asarray(taus, float)
    if predictions.shape != (len(pmfs), len(taus)):
        raise ValueError("Oracle evaluation shape mismatch")
    quantiles = np.stack([mixture_quantiles(p, taus, width) for p in pmfs])
    lower = (0.5 * np.arange(pmfs.shape[1]))[None, None, :]

    def risk(q):
        q = q[..., None]
        if width:
            shortfall = (np.maximum(q - lower, 0) ** 2 - np.maximum(q - lower - width, 0) ** 2) / (2 * width)
        else:
            shortfall = np.maximum(q - lower, 0)
        return (pmfs[:, None, :] * (taus[None, :, None] * (lower + width / 2 - q) + shortfall)).sum(-1)

    oracle_risk = risk(quantiles)
    excess = risk(predictions) - oracle_risk
    regrets = []
    for target in targets:
        boundary = cache["inventory"][:, None] - target
        regrets.append(risk(np.maximum(predictions, boundary)) - risk(np.maximum(quantiles, boundary)))
    return {
        "oracle_quantiles": quantiles,
        "excess_pinball": excess,
        "operational_regret": np.stack(regrets, -1),
        "oracle_risk": oracle_risk,
        "integration_error_max": float(cache["integration_error_max"]),
        "tail_mass_bound_max": float(cache["tail_mass_bound_max"]),
    }


def evaluate_oracle(daily, orders, config, predictions, taus, targets):
    return assess_laws(prepare_laws(daily, orders, config), predictions, taus, targets)


def cached_laws(directory, identity, daily, orders, config):
    from pathlib import Path
    from .jobs import digest
    from dat_study.io import read_json, save_json, sha256

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = digest({"identity": identity, "generator": config, "days": len(daily)})
    path, metadata = directory / f"{key}.npz", directory / f"{key}.json"
    if metadata.exists():
        if sha256(path) != read_json(metadata)["sha256"]:
            raise ValueError("Conditional law cache was changed")
        with np.load(path, allow_pickle=False) as stored:
            return {k: stored[k] for k in stored.files}
    result = prepare_laws(daily, orders, config)
    np.savez_compressed(path, **result)
    save_json(metadata, {"sha256": sha256(path), "evaluation_only": True})
    return result
