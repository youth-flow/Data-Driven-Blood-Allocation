"""Training-fitted tabular inputs built exclusively from observable order fields."""

import numpy as np
import pandas as pd


def ordered_quantiles(values):
    """Fixed rearrangement for independently fitted quantiles; never uses labels."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid quantile predictions")
    return np.sort(np.maximum(values, 0), axis=1)


class PsiFeatures:
    """The fixed 19-dimensional pre-decision summary psi(A,X) from the S4 protocol."""

    columns = [
        "u2",
        "v2",
        "uv",
        "ur",
        "vr",
        "u_min",
        "u_max",
        "v_min",
        "v_max",
        "r_min",
        "r_max",
    ]

    def fit(self, data):
        return self

    def _daily_summary(self, data, daily):
        orders = data.orders[data.orders.decision_day.isin(daily.decision_day)].copy()
        orders["u"] = orders["申请用血量"].to_numpy(float) / 6.0
        orders["v"] = orders["申请备血量"].to_numpy(float) / 8.0
        orders["r"] = orders["clinical_risk"].to_numpy(float)
        orders["u2"] = orders["u"] ** 2
        orders["v2"] = orders["v"] ** 2
        orders["uv"] = orders["u"] * orders["v"]
        orders["ur"] = orders["u"] * orders["r"]
        orders["vr"] = orders["v"] * orders["r"]
        summary = orders.groupby("decision_day").agg(
            u2=("u2", "mean"),
            v2=("v2", "mean"),
            uv=("uv", "mean"),
            ur=("ur", "mean"),
            vr=("vr", "mean"),
            u_min=("u", "min"),
            u_max=("u", "max"),
            v_min=("v", "min"),
            v_max=("v", "max"),
            r_min=("r", "min"),
            r_max=("r", "max"),
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
        u = orders["申请用血量"].to_numpy(float) / 6.0
        v = orders["申请备血量"].to_numpy(float) / 8.0
        r = orders["clinical_risk"].to_numpy(float)
        psi_rows = np.row_stack([lookup[day] for day in orders.decision_day.to_numpy()])
        return np.column_stack([u, v, r, psi_rows])


class TabularFeatures:
    def __init__(self, input_set="basic"):
        if input_set not in {"basic", "enhanced"}:
            raise ValueError("Input set must be basic or enhanced")
        self.input_set = input_set

    def fit(self, data):
        orders = data.order_split("train")
        dept = "科室高频分组"
        counts = orders[dept].astype(str).value_counts() if dept in orders else pd.Series(dtype=int)
        self.departments = sorted(counts.index, key=lambda v: (-counts[v], v))[:10]
        # Fixed semantic categories, not categories selected from evaluation records.
        self.kinds = ["急救用血", "手术备血", "病房用血", "病房备血", "其他"]
        extra = self.extra(data, data.daily_split("train"))
        self.extra_mean = extra.mean(axis=0)
        self.extra_scale = extra.std(axis=0)
        self.extra_scale[self.extra_scale < 1e-8] = 1.0
        return self

    def extra(self, data, daily):
        if self.input_set == "basic":
            return np.empty((len(daily), 0))
        frame = data.orders.copy()
        if data.schema.get("generator_version") == 3:
            columns = []
            for col in data.order_num_cols:
                grouped = frame.groupby("decision_day")[col]
                for values in (grouped.mean(), grouped.std().fillna(0), grouped.min(), grouped.max()):
                    columns.append(daily.decision_day.map(values).fillna(0).to_numpy(float))
            for left, right in (("申请用血量", "clinical_risk"), ("申请备血量", "clinical_risk")):
                values = (frame[left] * frame[right]).groupby(frame.decision_day).sum()
                columns.append(daily.decision_day.map(values).fillna(0).to_numpy(float))
            return np.column_stack(columns)
        use = frame["申请用血量"].to_numpy(float)
        reserve = frame["申请备血量"].to_numpy(float)
        age = frame["delta"].to_numpy(float) + 1
        kind = frame.get("输血要求分组", pd.Series("其他", index=frame.index)).astype(str)
        department = frame.get("科室高频分组", pd.Series("missing", index=frame.index)).astype(str)
        columns = []

        def add(mask, quantities):
            for quantity in quantities:
                grouped = (
                    pd.Series(np.where(mask, quantity, 0.0)).groupby(frame.decision_day.reset_index(drop=True)).sum()
                )
                columns.append(daily.decision_day.map(grouped).fillna(0).to_numpy(float))

        for a in (1, 2, 3):
            for k in self.kinds:
                add((age == a) & kind.eq(k).to_numpy(), [np.ones(len(frame)), use, reserve])
        for dept in self.departments:
            add(department.eq(dept).to_numpy(), [np.ones(len(frame)), use, reserve])
        if "planned_use_offset_days" in frame:
            missing = frame["planned_use_missing"].to_numpy(bool)
            for offset in (0, 1):
                add(
                    (frame.planned_use_offset_days.to_numpy() == offset) & ~missing, [np.ones(len(frame)), use, reserve]
                )
            add(missing, [np.ones(len(frame))])
        return np.column_stack(columns)

    def transform(self, data, daily):
        extra = (self.extra(data, daily) - self.extra_mean) / self.extra_scale
        return np.column_stack([data.context(daily), extra])
