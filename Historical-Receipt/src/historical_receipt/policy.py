import numpy as np


def receipt_allocations(data, taus):
    if data.schema["schema_version"].startswith("synthetic_"):
        raise ValueError("Synthetic data have no genuine historical hospital receipts")
    daily = data.daily_split("eval")
    if "q_hist_receipt" not in daily:
        raise ValueError("Historical Receipt requires observed q_hist_receipt on every evaluation day")
    values = daily.q_hist_receipt.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Missing/invalid receipts cannot be treated as zero or silently drop days")
    return np.repeat(values[:, None], len(taus), axis=1)
