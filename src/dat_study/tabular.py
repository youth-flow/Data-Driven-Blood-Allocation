"""Training-fitted tabular inputs built exclusively from observable pre-decision fields."""

import numpy as np


def ordered_quantiles(values):
    """Fixed rearrangement for independently fitted quantiles; never uses labels."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid quantile predictions")
    return np.sort(np.maximum(values, 0), axis=1)


class PsiFeatures:
    """Fixed pre-decision summary: 23 dims for v3/v4, 16 dims for AXID-DC v6."""

    V6_COLUMNS = ["ur", "us", "rs", "u_min", "u_max", "r_min", "r_max", "s_min", "s_max"]
    V4_COLUMNS = [
        "u2",
        "v2",
        "uv",
        "ur",
        "vr",
        "us",
        "vs",
        "rs",
        "u_min",
        "u_max",
        "v_min",
        "v_max",
        "r_min",
        "r_max",
    ]

    def fit(self, data):
        self.version = int(data.schema.get("generator_version", 4))
        self.columns = self.V6_COLUMNS if self.version in {6, 7, 8} else self.V4_COLUMNS
        return self

    def _daily_summary(self, data, daily):
        orders = data.orders[data.orders.decision_day.isin(daily.decision_day)].copy()
        if self.version in {6, 7, 8}:
            orders["ur"] = orders["u"] * orders["r"]
            orders["us"] = orders["u"] * orders["s"]
            orders["rs"] = orders["r"] * orders["s"]
            summary = orders.groupby("decision_day").agg(
                ur=("ur", "mean"),
                us=("us", "mean"),
                rs=("rs", "mean"),
                u_min=("u", "min"),
                u_max=("u", "max"),
                r_min=("r", "min"),
                r_max=("r", "max"),
                s_min=("s", "min"),
                s_max=("s", "max"),
            )
        else:
            orders["u2"] = orders["u"] ** 2
            orders["v2"] = orders["v"] ** 2
            orders["uv"] = orders["u"] * orders["v"]
            orders["ur"] = orders["u"] * orders["risk"]
            orders["vr"] = orders["v"] * orders["risk"]
            orders["us"] = orders["u"] * orders["surgery_type"]
            orders["vs"] = orders["v"] * orders["surgery_type"]
            orders["rs"] = orders["risk"] * orders["surgery_type"]
            summary = orders.groupby("decision_day").agg(
                u2=("u2", "mean"),
                v2=("v2", "mean"),
                uv=("uv", "mean"),
                ur=("ur", "mean"),
                vr=("vr", "mean"),
                us=("us", "mean"),
                vs=("vs", "mean"),
                rs=("rs", "mean"),
                u_min=("u", "min"),
                u_max=("u", "max"),
                v_min=("v", "min"),
                v_max=("v", "max"),
                r_min=("risk", "min"),
                r_max=("risk", "max"),
            )
        aligned = daily[["decision_day"]].merge(summary, left_on="decision_day", right_index=True, how="left")
        return aligned[self.columns].fillna(0.0).to_numpy(float)

    def transform(self, data, daily):
        return np.column_stack([data.context(daily), self._daily_summary(data, daily)])

    def patient_input(self, data, orders):
        unique_days = (
            data.daily.loc[data.daily.decision_day.isin(orders.decision_day), ["decision_day", *data.day_cols, "It"]]
            .drop_duplicates("decision_day")
            .reset_index(drop=True)
        )
        psi = self.transform(data, unique_days)
        lookup = {day: value for day, value in zip(unique_days.decision_day.to_numpy(), psi)}
        psi_rows = np.row_stack([lookup[day] for day in orders.decision_day.to_numpy()])
        if self.version in {6, 7, 8}:
            return np.column_stack(
                [
                    orders["u"].to_numpy(float),
                    orders["r"].to_numpy(float),
                    orders["s"].to_numpy(float),
                    psi_rows,
                ]
            )
        return np.column_stack(
            [
                orders["u"].to_numpy(float),
                orders["v"].to_numpy(float),
                orders["risk"].to_numpy(float),
                orders["surgery_type"].to_numpy(float),
                psi_rows,
            ]
        )
