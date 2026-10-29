"""Flat-ETO convolution plus shared empirical quantity estimation."""

import numpy as np
from dat_study.quantity import ConditionalQuantityPMF as ConditionalQuantityPMF, quantity_keys as quantity_keys


def independent_mixture_pmf(probabilities, positive_pmfs):
    """Convolve (1-p) delta_0 + p F_positive without circular aliasing."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(positive_pmfs):
        raise ValueError("Probabilities and positive PMFs must align.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must lie in [0,1].")
    checked = []
    for pmf in positive_pmfs:
        pmf = np.asarray(pmf, dtype=float)
        if (
            pmf.ndim != 1
            or not len(pmf)
            or not np.isfinite(pmf).all()
            or (pmf < 0).any()
            or not np.isclose(pmf.sum(), 1.0)
        ):
            raise ValueError("Invalid positive PMF.")
        if pmf[0] != 0:
            raise ValueError("Conditional positive PMF must have zero mass at zero.")
        checked.append(pmf)
    max_sum = sum(len(pmf) - 1 for p, pmf in zip(probabilities, checked) if p > 0)
    if max_sum == 0:
        return np.array([1.0])
    fft_size = 1 << max_sum.bit_length()
    characteristic = np.ones(fft_size // 2 + 1, dtype=np.complex128)
    cache = {}
    for probability, pmf in zip(probabilities, checked):
        if probability == 0:
            continue
        key = (pmf.shape, pmf.tobytes())
        transformed = cache.get(key)
        if transformed is None:
            transformed = np.fft.rfft(pmf, n=fft_size)
            cache[key] = transformed
        characteristic *= (1.0 - probability) + probability * transformed
    result = np.fft.irfft(characteristic, n=fft_size)[: max_sum + 1]
    if result.min() < -1e-9:
        raise RuntimeError("Numerically invalid convolution.")
    result = np.maximum(result, 0.0)
    result /= result.sum()
    # np.dot loads a second Intel OpenMP runtime after CUDA training in this
    # Windows environment; this equivalent elementwise sum avoids BLAS entirely.
    expected = sum(p * np.sum(np.arange(len(pmf)) / 2.0 * pmf) for p, pmf in zip(probabilities, checked))
    obtained = float(np.sum(np.arange(len(result)) / 2.0 * result))
    if not np.isclose(expected, obtained, rtol=1e-8, atol=1e-7):
        raise RuntimeError("FFT mean does not match sum of individual means.")
    return result


def pmf_quantile(pmf, tau=0.9):
    # Numerical rounding can leave total mass infinitesimally below one.
    cumulative = np.cumsum(pmf)
    cumulative[-1] = 1.0
    return min(int(np.searchsorted(cumulative, tau, side="left")), len(pmf) - 1) / 2.0
