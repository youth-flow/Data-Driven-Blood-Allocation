import numpy as np
import pandas as pd


def rolling_quantiles(history, evaluation, taus, window, same_weekday=False):
    if window < 1 or (same_weekday and window < 91):
        raise ValueError("Same-weekday rolling SAA requires a window of at least 91 calendar days")
    past = history[["decision_day", "D"]].copy()
    values = []
    for row in evaluation.sort_values("decision_day").itertuples():
        # Strictly prior calendar days: today's label enters only after its decision.
        eligible = past[past.decision_day.lt(row.decision_day)]
        recent = eligible[eligible.decision_day.ge(row.decision_day - pd.Timedelta(days=window))]
        if same_weekday:
            recent = recent[recent.decision_day.dt.dayofweek.eq(row.decision_day.dayofweek)]
        if recent.empty:
            recent = eligible  # Predeclared cold-start fallback; no future labels.
        if recent.empty:
            raise ValueError("Rolling SAA requires at least one previously observed demand")
        values.append(np.quantile(recent.D, taus, method="higher"))
        past = pd.concat([past, pd.DataFrame({"decision_day": [row.decision_day], "D": [row.D]})], ignore_index=True)
    return np.asarray(values)
