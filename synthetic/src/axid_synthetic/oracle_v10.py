"""Conditional Monte Carlo Oracle for the AXID-DC v10 information boundary.

The Oracle receives only the blood centre's pre-decision information
``Z=(A_minus, c_p, c_e, I)``.  Hidden arrivals, patient profiles, the shared day
shock, and patient shocks must be integrated by the supplied simulator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .metrics_v10 import TAU, left_quantile


DEFAULT_MC_DRAWS = 4_096
DEFAULT_MC_BATCH = 1_024


@dataclass(frozen=True)
class ObservedState:
    """Exactly the information visible when the replenishment decision is made."""

    old_patients: np.ndarray
    cp: float
    ce: float
    inventory: float

    def __post_init__(self):
        patients = np.array(self.old_patients, dtype=float, copy=True)
        if patients.ndim != 2 or patients.shape[1] != 3 or len(patients) < 1:
            raise ValueError("old_patients must have shape [N_minus, 3] with N_minus >= 1")
        if not np.isfinite(patients).all():
            raise ValueError("old patient features must be finite")
        if (patients[:, 0] < 1).any() or (patients[:, 0] > 10).any():
            raise ValueError("patient application amount must lie in [1, 10]")
        if (patients[:, 1] < 0).any() or (patients[:, 1] > 1).any():
            raise ValueError("patient risk must lie in [0, 1]")
        if (patients[:, 2] < 42).any() or (patients[:, 2] > 170).any():
            raise ValueError("patient Hb must lie in [42, 170] g/L")
        scalar = np.asarray([self.cp, self.ce, self.inventory], dtype=float)
        if not np.isfinite(scalar).all() or not 0 <= self.cp <= 1 or not 0 <= self.ce <= 1:
            raise ValueError("cp and ce must be finite values in [0, 1]")
        if not 0 < self.inventory <= 520:
            raise ValueError("inventory must lie in the frozen (0, 520] support")
        patients.setflags(write=False)
        object.__setattr__(self, "old_patients", patients)
        object.__setattr__(self, "cp", float(self.cp))
        object.__setattr__(self, "ce", float(self.ce))
        object.__setattr__(self, "inventory", float(self.inventory))

    @property
    def n_old(self):
        return len(self.old_patients)


# A simulator may only receive an ObservedState, a draw count, and a private RNG.
ConditionalDemandSimulator = Callable[[ObservedState, int, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class OracleEstimate:
    quantile: float
    n_draws: int
    seed_entropy: int
    block_quantiles: tuple[float, ...]


def generator_simulator(state: ObservedState, n_draws: int, rng: np.random.Generator):
    """Adapter to the frozen v10 DGP's conditional simulator."""
    from .generator_v10 import simulate_conditional_demand

    return simulate_conditional_demand(
        state.old_patients,
        state.cp,
        state.ce,
        state.inventory,
        int(n_draws),
        rng,
        observed_n_old=state.n_old,
    )


def _validated_demands(values, expected, inventory):
    result = np.asarray(values, dtype=float)
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ValueError("conditional simulator must return one finite demand per requested draw")
    if (result < -1e-9).any() or (result > inventory + 1e-7).any():
        raise ValueError("v10 realized issue demand must lie in [0, observed inventory]")
    return np.clip(result, 0.0, inventory)


def conditional_oracle_quantile(
    state: ObservedState,
    *,
    simulator: ConditionalDemandSimulator = generator_simulator,
    n_draws=DEFAULT_MC_DRAWS,
    batch_size=DEFAULT_MC_BATCH,
    tau=TAU,
    seed=0,
):
    """Estimate ``Q_tau(D | Z=state)`` using reproducible conditional MC."""
    n_draws, batch_size = int(n_draws), int(batch_size)
    if n_draws <= 0 or batch_size <= 0 or not 0 < tau < 1:
        raise ValueError("n_draws, batch_size, and tau are invalid")
    seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_sequence)
    samples = []
    block_quantiles = []
    for start in range(0, n_draws, batch_size):
        width = min(batch_size, n_draws - start)
        block = _validated_demands(simulator(state, width, rng), width, state.inventory)
        samples.append(block)
        block_quantiles.append(left_quantile(block, tau))
    demand = np.concatenate(samples)
    entropy = seed_sequence.entropy
    if not isinstance(entropy, (int, np.integer)):
        # SeedSequence also accepts an integer sequence; retain a stable compact fingerprint.
        entropy = int(np.asarray(entropy, dtype=np.uint32).sum(dtype=np.uint64))
    return OracleEstimate(
        quantile=left_quantile(demand, tau),
        n_draws=n_draws,
        seed_entropy=int(entropy),
        block_quantiles=tuple(block_quantiles),
    )


def conditional_oracle_quantiles(
    states: Iterable[ObservedState],
    *,
    simulator: ConditionalDemandSimulator = generator_simulator,
    n_draws=DEFAULT_MC_DRAWS,
    batch_size=DEFAULT_MC_BATCH,
    tau=TAU,
    seed=2026111410,
    workers=1,
):
    """Evaluate states with independent, order-stable child random streams."""
    states = tuple(states)
    if not states:
        raise ValueError("at least one observed state is required")
    children = np.random.SeedSequence(seed).spawn(len(states))
    workers = int(workers)
    if workers <= 0:
        raise ValueError("workers must be positive")
    arguments = tuple(zip(states, children))
    if workers == 1:
        estimates = [
            conditional_oracle_quantile(
                state,
                simulator=simulator,
                n_draws=n_draws,
                batch_size=batch_size,
                tau=tau,
                seed=child,
            )
            for state, child in arguments
        ]
    else:
        # Process parallelism is effective for the current conditional simulator,
        # whose per-state MDCEV draws otherwise run serially in Python.  The
        # caller's executable entry point must use the normal Windows main guard.
        from joblib import Parallel, delayed

        estimates = Parallel(n_jobs=workers, backend="loky", batch_size=1)(
            delayed(conditional_oracle_quantile)(
                state,
                simulator=simulator,
                n_draws=n_draws,
                batch_size=batch_size,
                tau=tau,
                seed=child,
            )
            for state, child in arguments
        )
    return np.asarray([estimate.quantile for estimate in estimates]), estimates
