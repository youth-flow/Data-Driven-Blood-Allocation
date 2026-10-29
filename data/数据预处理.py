"""完整的本机数据预处理脚本。

本文件包含字段清洗、订单快照、日需求、期初库存、历史及订单特征、
训练期拟合、时间划分、隐私处理、完整性校验和数据导出的全部实现。
原始表只读；不输出原始明细；默认输出到本脚本旁的 processed/，拒绝覆盖。
直接运行：python data/数据预处理.py --help
仅依赖 Python 标准库、NumPy 和 pandas，不依赖工程 src 或模型代码。
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets

import numpy as np
import pandas as pd

# 一、原始字段清洗、日需求、订单快照、库存与历史特征

RED_CELL_CATEGORY_CODES = {"HX", "XS", "XS1", "FD", "XH", "JX", "BJ", "FU", "FJ"}
START_DAY = pd.Timestamp("2013-01-01")
TRAIN_END_DAY = pd.Timestamp("2017-12-31")
CUTOFF_DAY = pd.Timestamp("2020-01-01")
MAX_ORDER_AGE_DAYS = 3


def _source(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(np.nan, index=frame.index, dtype=object)


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _amount(series: pd.Series) -> pd.Series:
    number = pd.to_numeric(series, errors="coerce").astype(float)
    return number.where(np.isfinite(number) & number.ge(0.0))


def _flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").eq(1) | _text(series).str.lower().isin(["是", "true", "yes", "y"])


def _calendar(calendar) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce")).normalize()
    if days.hasnans or days.has_duplicates:
        raise ValueError("Decision calendar must contain unique valid dates.")
    return days.sort_values()


def _calendar_frame(calendar) -> pd.DataFrame:
    days = _calendar(calendar)
    return pd.DataFrame(
        {
            "decision_day": days,
            "target_day": days,
            "snapshot_day": days - pd.Timedelta(days=1),
            "information_cutoff": days,
        }
    )


def normalize_requests(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply product scope only; retain incomplete features and all issue delays."""
    if not raw.columns.is_unique:
        raise ValueError("Request source columns must be unique.")
    scope = _text(_source(raw, "ABO血型")).str.upper().eq("O") & _text(_source(raw, "输血品种")).str.contains(
        "红细胞", regex=False
    )
    req = raw.loc[scope].copy().reset_index(drop=True)
    # A request identifier is used only for a duplicate integrity check.  It
    # remains private in this internal frame and is never part of snapshots.
    key_names = [
        "申请编码",
        "申请单号",
        "申请号",
        "申请编号",
        "申请单编号",
        "申请ID",
        "申请单ID",
        "application_id",
        "request_id",
    ]
    for key in key_names:
        if key in req.columns:
            values = _text(req[key])
            if values.loc[values.ne("")].duplicated().any():
                raise ValueError("Duplicate application keys prevent unique request normalization.")
    req["order_idx"] = np.arange(len(req), dtype=np.int64)
    for output, aliases in [
        ("application_time", ("申请日期", "application_time")),
        ("issue_time", ("输血科发血日期", "issue_time")),
        ("cancel_time", ("撤销日期", "撤消日期", "cancel_time")),
        ("planned_time", ("输用日期", "预定输血日期", "计划输血日期", "planned_time")),
    ]:
        req[output] = _datetime(_source(req, *aliases))
    for prefix in ["application", "issue", "cancel", "planned"]:
        req[f"{prefix}_day"] = req[f"{prefix}_time"].dt.normalize()
    raw_issue_amount = pd.to_numeric(_source(req, "发血量", "issue_amount"), errors="coerce")
    req["issue_amount"] = _amount(raw_issue_amount)
    req["has_issue_event"] = (
        req["issue_time"].notna()
        | _flag(_source(req, "是否发血"))
        | (raw_issue_amount.notna() & raw_issue_amount.ne(0.0))
    )
    req["undated_issue_event"] = req["has_issue_event"] & req["issue_time"].isna()
    req["undated_cancellation"] = _flag(_source(req, "撤消标志", "撤销标志")) & req["cancel_time"].isna()
    req["request_use"] = _amount(_source(req, "申请用血量"))
    req["request_reserve"] = _amount(_source(req, "申请备血量"))
    stats = {
        "input_request_records": int(len(raw)),
        "product_scope_request_records": int(len(req)),
        "excluded_other_product_records": int((~scope).sum()),
        "missing_application_time_records": int(req["application_time"].isna().sum()),
        "dated_issue_records": int(req["issue_time"].notna().sum()),
        "undated_issue_event_records": int(req["undated_issue_event"].sum()),
        "undated_cancellation_records": int(req["undated_cancellation"].sum()),
        "invalid_issue_amount_event_records": int((req["has_issue_event"] & req["issue_amount"].isna()).sum()),
        "issue_precedes_application_records": int(req["issue_time"].lt(req["application_time"]).sum()),
    }
    return req, stats


def _daily_sum(days: pd.Series, values: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    sums = values.groupby(days).sum(min_count=1)
    return sums.reindex(calendar, fill_value=0.0).to_numpy(dtype=float, copy=True)


def _daily_count(days: pd.Series, mask: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    return days.loc[mask].value_counts().reindex(calendar, fill_value=0).to_numpy(dtype=np.int64)


def build_demand(req: pd.DataFrame, calendar) -> tuple[pd.DataFrame, dict]:
    """Daily issuance labels independent of eligibility for order snapshots."""
    daily = _calendar_frame(calendar)
    days = pd.DatetimeIndex(daily["decision_day"])
    dated = req["issue_day"].notna()
    valid = req["issue_amount"].notna()
    unknown = dated & ~valid
    existing = dated & req["application_day"].lt(req["issue_day"])
    new = dated & req["application_day"].eq(req["issue_day"])
    source_unknown = dated & ~(existing | new)
    coverage = pd.Series(False, index=daily.index)
    if dated.any():
        first, last = req.loc[dated, "issue_day"].min(), req.loc[dated, "issue_day"].max()
        coverage = daily["decision_day"].between(first, last)
    daily["event_coverage_supported"] = coverage
    daily["unknown_issue_records"] = _daily_count(req["issue_day"], unknown, days)
    daily["dated_issue_records"] = _daily_count(req["issue_day"], dated, days)
    daily["source_unknown_issue_records"] = _daily_count(req["issue_day"], source_unknown, days)
    for column, mask in [("D", dated), ("D_existing", existing), ("D_new", new), ("D_source_unknown", source_unknown)]:
        daily[column] = _daily_sum(req.loc[mask & valid, "issue_day"], req.loc[mask & valid, "issue_amount"], days)
        unknown_component = _daily_count(req["issue_day"], mask & ~valid, days) > 0
        daily.loc[~coverage | unknown_component, column] = np.nan
    daily["demand_observed"] = coverage & daily["unknown_issue_records"].eq(0)
    in_calendar = req["issue_day"].isin(days)
    stats = {
        "dated_issue_records_in_calendar": int((dated & in_calendar).sum()),
        "valid_issue_quantity_in_calendar": float(req.loc[dated & valid & in_calendar, "issue_amount"].sum()),
        "unknown_issue_records_in_calendar": int((unknown & in_calendar).sum()),
        "source_unknown_issue_records_in_calendar": int((source_unknown & in_calendar).sum()),
        "undated_issue_event_records": int(req["undated_issue_event"].sum()),
        "event_coverage_supported_days": int(coverage.sum()),
        "demand_unobserved_days": int((~daily["demand_observed"]).sum()),
    }
    return daily, stats


def build_snapshots(req: pd.DataFrame, calendar) -> pd.DataFrame:
    """Requests observable strictly before each target day's midnight cutoff."""
    days = _calendar(calendar)
    pieces = []
    for age in range(1, MAX_ORDER_AGE_DAYS + 1):
        target = req["application_day"] + pd.Timedelta(days=age)
        eligible = (
            target.isin(days)
            & req["application_time"].lt(target)
            & (req["issue_time"].isna() | req["issue_time"].ge(target))
            & (req["cancel_time"].isna() | req["cancel_time"].ge(target))
        )
        selected = req.loc[eligible]
        target_day = target.loc[eligible]
        issue_today = selected["issue_day"].eq(target_day)
        unknown_label = (issue_today & selected["issue_amount"].isna()) | selected["undated_issue_event"]
        demand = selected["issue_amount"].where(issue_today, 0.0).mask(unknown_label)
        tags = issue_today.astype(float).mask(selected["undated_issue_event"])
        frame = pd.DataFrame(
            {
                "order_idx": selected["order_idx"].to_numpy(dtype=np.int64),
                "decision_day": target_day.to_numpy(),
                "target_day": target_day.to_numpy(),
                "snapshot_day": (target_day - pd.Timedelta(days=1)).to_numpy(),
                "information_cutoff": target_day.to_numpy(),
                "delta": age - 1,
                "target_age_days": age,
                "application_time": selected["application_time"].to_numpy(),
                "real_demand": demand.to_numpy(dtype=float),
                "tag": tags.to_numpy(dtype=float),
                "order_label_observed": (~unknown_label).to_numpy(dtype=bool),
                "planned_use_offset_days": (selected["planned_day"] - target_day).dt.days.to_numpy(dtype=float),
                "planned_use_missing": selected["planned_time"].isna().to_numpy(dtype=np.int64),
            }
        )
        pieces.append(frame)
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["decision_day", "order_idx"], kind="stable")
        .reset_index(drop=True)
    )


def _interval_sum(start: pd.Series, end: pd.Series, values: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    """Accumulate inclusive daily intervals using an O(n + calendar) delta array."""
    difference = np.zeros(len(calendar) + 1, dtype=float)
    valid = start.notna() & end.notna() & start.le(end) & values.notna()
    if valid.any() and len(calendar):
        left = calendar.searchsorted(pd.DatetimeIndex(start.loc[valid]), side="left")
        right = calendar.searchsorted(pd.DatetimeIndex(end.loc[valid]), side="right")
        amount = values.loc[valid].to_numpy(dtype=float)
        np.add.at(difference, left, amount)
        np.add.at(difference, right, -amount)
    return np.cumsum(difference[:-1])


def _event_quantity(event_days: pd.Series, amount: pd.Series, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    valid = event_days.notna() & amount.notna()
    values = _daily_sum(event_days.loc[valid], amount.loc[valid], days)
    unknown = _daily_count(event_days, event_days.notna() & amount.isna(), days)
    values[unknown > 0] = np.nan
    return values, unknown


def build_inventory(raw: pd.DataFrame, calendar) -> tuple[pd.DataFrame, dict]:
    """Opening stock: receipt-day + 1 through actual/fallback departure day."""
    days = _calendar(calendar)
    result = pd.DataFrame({"decision_day": days})
    if not raw.columns.is_unique:
        raise ValueError("Inventory source columns must be unique.")
    if "血液类别" in raw.columns:
        red = _text(raw["血液类别"]).str.upper().isin(RED_CELL_CATEGORY_CODES)
    else:
        red = _text(_source(raw, "血液名称")).str.contains("红细胞", regex=False)
    unit = _text(_source(raw, "单位")).str.replace("Ｕ", "U", regex=False).str.upper()
    scope = red & _text(_source(raw, "ABO血型")).str.upper().eq("O") & unit.eq("U")
    inv = raw.loc[scope].copy().reset_index(drop=True)
    amount = _amount(_source(inv, "血量"))
    receipt = _datetime(_source(inv, "入库日期")).dt.normalize()
    actual_end = _datetime(_source(inv, "出库日期")).dt.normalize()
    expiry = _datetime(_source(inv, "失效日期")).dt.normalize()
    fallback = actual_end.isna() & expiry.notna()
    end = actual_end.fillna(expiry)
    start = receipt + pd.Timedelta(days=1)
    missing_end = end.isna()
    invalid_chronology = end.notna() & receipt.notna() & end.lt(receipt)
    unresolved = missing_end | receipt.isna() | amount.isna() | invalid_chronology
    if len(days):
        uncertain_start = start.fillna(days.min())
        uncertain_end = end.mask(invalid_chronology).fillna(days.max())
    else:
        uncertain_start, uncertain_end = start, end
    known = ~unresolved
    known_stock = _interval_sum(start.loc[known], end.loc[known], amount.loc[known], days)
    counts = pd.Series(1.0, index=inv.index)
    unresolved_records = _interval_sum(
        uncertain_start.loc[unresolved], uncertain_end.loc[unresolved], counts.loc[unresolved], days
    ).astype(np.int64)
    unresolved_units = _interval_sum(
        uncertain_start.loc[unresolved], uncertain_end.loc[unresolved], amount.loc[unresolved], days
    )
    unknown_quantity_records = _interval_sum(
        uncertain_start.loc[amount.isna()], uncertain_end.loc[amount.isna()], counts.loc[amount.isna()], days
    ).astype(np.int64)
    unresolved_units[unknown_quantity_records > 0] = np.nan
    result["inventory_unresolved_records"] = unresolved_records
    result["inventory_unresolved_units"] = unresolved_units
    result["inventory_unknown_quantity_records"] = unknown_quantity_records
    result["inventory_observed"] = unresolved_records == 0
    result["It"] = np.where(result["inventory_observed"], known_stock, np.nan)
    result["expiry_fallback_units"] = _interval_sum(
        start.loc[fallback & known], end.loc[fallback & known], amount.loc[fallback & known], days
    )
    result["expiry_fallback_records"] = _interval_sum(
        start.loc[fallback & known], end.loc[fallback & known], counts.loc[fallback & known], days
    ).astype(np.int64)
    result["q_hist_receipt"], receipt_unknown = _event_quantity(receipt, amount, days)
    result["inventory_departure"], departure_unknown = _event_quantity(actual_end, amount, days)
    result["expiry_fallback_departure"], fallback_unknown = _event_quantity(expiry.where(fallback), amount, days)
    result["receipt_unknown_quantity_records"] = receipt_unknown
    result["departure_unknown_quantity_records"] = departure_unknown
    result["expiry_fallback_unknown_quantity_records"] = fallback_unknown
    stats = {
        "input_inventory_records": int(len(raw)),
        "product_scope_inventory_records": int(len(inv)),
        "missing_receipt_date_records": int(receipt.isna().sum()),
        "missing_departure_and_expiry_records": int(missing_end.sum()),
        "invalid_inventory_quantity_records": int(amount.isna().sum()),
        "invalid_inventory_chronology_records": int(invalid_chronology.sum()),
        "expiry_fallback_records": int(fallback.sum()),
        "inventory_unobserved_days": int((~result["inventory_observed"]).sum()),
    }
    return result, stats


def build_request_activity(req: pd.DataFrame, calendar) -> pd.DataFrame:
    """Application-day activity for later lagging; no issue outcome is used."""
    days = _calendar(calendar)
    activity = pd.DataFrame({"decision_day": days})
    dated = req["application_day"].notna()
    supported = pd.Series(False, index=activity.index)
    if dated.any():
        supported = activity["decision_day"].between(
            req.loc[dated, "application_day"].min(), req.loc[dated, "application_day"].max()
        )
    activity["request_activity_coverage_supported"] = supported
    activity["new_applications_count"] = _daily_count(req["application_day"], dated, days).astype(float)
    for source, prefix in [("request_use", "new_request_use"), ("request_reserve", "new_request_reserve")]:
        valid = dated & req[source].notna()
        activity[f"{prefix}_sum"] = _daily_sum(req.loc[valid, "application_day"], req.loc[valid, source], days)
        activity[f"{prefix}_missing_count"] = _daily_count(
            req["application_day"], dated & req[source].isna(), days
        ).astype(float)
    values = [column for column in activity.columns if column.startswith("new_")]
    activity.loc[~supported, values] = np.nan
    return activity


def build_history(daily: pd.DataFrame, request_activity: pd.DataFrame) -> pd.DataFrame:
    """Calendar-based history available strictly before the decision cutoff."""
    if daily.empty:
        return pd.DataFrame({"decision_day": pd.Series(dtype="datetime64[ns]")})
    date_values = _datetime(daily["decision_day"]).dt.normalize()
    if date_values.isna().any() or date_values.duplicated().any():
        raise ValueError("Daily history requires one valid row per day.")
    days = pd.date_range(date_values.min(), date_values.max(), freq="D")
    source = daily.copy().assign(decision_day=date_values).set_index("decision_day").reindex(days)
    result = pd.DataFrame(index=days)
    observed = source.get("demand_observed", pd.Series(True, index=days)).fillna(False).astype(bool)
    if "event_coverage_supported" in source:
        observed &= source["event_coverage_supported"].fillna(False).astype(bool)
    demand = pd.to_numeric(source["D"], errors="coerce").where(observed)
    prior = demand.shift(1)
    result["demand_lag1"] = prior
    for window in [7, 14, 28]:
        result[f"demand_roll{window}_mean"] = prior.rolling(window, min_periods=window).mean()
    result["demand_roll7_std"] = prior.rolling(7, min_periods=7).std(ddof=1)
    result["demand_same_weekday_lag7"] = demand.shift(7)
    new_demand = (
        pd.to_numeric(source.get("D_new", pd.Series(np.nan, index=days)), errors="coerce").where(observed).shift(1)
    )
    result["past_new_demand_lag1"] = new_demand
    for window in [7, 28]:
        result[f"past_new_demand_roll{window}_mean"] = new_demand.rolling(window, min_periods=window).mean()
    activity = request_activity.copy()
    if "decision_day" not in activity:
        raise ValueError("Request activity requires decision dates.")
    activity["decision_day"] = _datetime(activity["decision_day"]).dt.normalize()
    if activity["decision_day"].isna().any() or activity["decision_day"].duplicated().any():
        raise ValueError("Request activity requires one valid row per day.")
    activity = activity.set_index("decision_day").reindex(days)
    for column in activity.columns:
        if not column.startswith("new_"):
            continue
        values = pd.to_numeric(activity[column], errors="coerce")
        if "request_activity_coverage_supported" in activity:
            values = values.where(activity["request_activity_coverage_supported"].fillna(False))
        lagged = values.shift(1)
        result[f"past_{column}_lag1"] = lagged
        for window in [7, 28]:
            result[f"past_{column}_roll{window}_mean"] = lagged.rolling(window, min_periods=window).mean()
    return result.rename_axis("decision_day").reset_index()


# 二、训练期拟合的订单特征

DEPT_TOP_N = 30
SURGERY_TOP_N = 30
ICD_TOP_N = 50
CATEGORY_MIN_SUPPORT = 5
REQUEST_LARGE_THRESHOLD = 8.0

# The caller adds delta after expanding requests into decision snapshots.
ORDER_NUM_COLS = [
    "delta",
    "申请用血量",
    "申请备血量",
    "log1p_request_use",
    "log1p_request_reserve",
    "request_use_ge_8",
    "request_reserve_ge_8",
    "血红蛋白",
    "红细胞压积",
    "年龄(岁)",
    "ALT数值",
    "ALT有数值",
    "诊断_肿瘤",
    "诊断_白血病",
    "诊断_出血",
    "诊断_创伤",
    "诊断_肝病",
    "病史缺失",
    "病史_既往输血史",
    "病史_妊娠史",
    "病史_流产史",
    "病史_输血反应史",
    "手术缺失",
    "手术_切除",
    "手术_置换",
    "手术_移植",
    "手术_开颅",
    "手术_开胸",
    "手术_剖宫产",
    "手术_搭桥",
    "request_use_missing",
    "request_reserve_missing",
    "hb_missing",
    "hct_missing",
    "age_missing",
]

ORDER_CAT_COLS = [
    "输血要求分组",
    "年龄分组",
    "科室高频分组",
    "手术高频分组",
    "入院诊断ICD分组",
]

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ICD = re.compile(r"[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?", re.IGNORECASE)


def _column(requests: pd.DataFrame, name: str) -> pd.Series:
    if name in requests.columns:
        return requests[name]
    return pd.Series(np.nan, index=requests.index, dtype=object)


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", "", regex=True)


def _first_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(_NUMBER, str(value).strip())
    if match is None:
        return np.nan
    try:
        number = float(match.group(0))
    except (ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def parse_age_to_years(value: object) -> float:
    """Read numeric years and explicit Chinese year/month/day ages."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    number = _first_number(text)
    if not np.isfinite(number) or number < 0.0:
        return np.nan
    # Explicit units also correctly handle combined ages such as 1岁6个月.
    components = re.findall(rf"({_NUMBER})\s*(岁|年|个月|月|天|日)", text)
    if components:
        years = 0.0
        for amount_text, unit in components:
            amount = float(amount_text)
            if not np.isfinite(amount) or amount < 0.0:
                return np.nan
            divisor = 12.0 if unit in {"个月", "月"} else 365.0 if unit in {"天", "日"} else 1.0
            years += amount / divisor
        return years if np.isfinite(years) else np.nan
    return number


def _nonnegative_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(np.isfinite(values) & values.ge(0.0))


def _fit_numeric(
    values: pd.Series,
    fit_mask: pd.Series,
) -> tuple[pd.Series, dict[str, int | float | None]]:
    training = values.loc[fit_mask].dropna()
    state: dict[str, int | float | None] = {
        "fit_valid_count": int(len(training)),
        "fit_missing_count": int(fit_mask.sum() - len(training)),
        "lower_quantile": 0.01,
        "upper_quantile": 0.99,
        "clip_lower": None,
        "clip_upper": None,
        "median": 0.0,
    }
    if training.empty:
        # No held-out observations are used to invent clipping thresholds.
        return values.fillna(0.0).astype(float), state
    lower = float(training.quantile(0.01))
    upper = float(training.quantile(0.99))
    median = float(training.median())
    state.update(clip_lower=lower, clip_upper=upper, median=median)
    return values.clip(lower=lower, upper=upper).fillna(median).astype(float), state


def _purpose(text: str) -> str:
    if not text or text == "未知":
        return "未知"
    if "急救" in text or "急诊" in text:
        return "急救用血"
    if "手术" in text and "备血" in text:
        return "手术备血"
    if "病房" in text and "备血" in text:
        return "病房备血"
    if "病房" in text and "用血" in text:
        return "病房用血"
    if "门诊" in text and "备血" in text:
        return "门诊备血"
    if "门诊" in text and "用血" in text:
        return "门诊用血"
    if "备血" in text:
        return "备血"
    if "用血" in text:
        return "用血"
    return "其他"


def _age_group(value: float) -> str:
    if pd.isna(value):
        return "未知"
    if value < 18.0:
        return "<18"
    if value < 45.0:
        return "18-44"
    if value < 65.0:
        return "45-64"
    return "65+"


def _icd_code(text: str) -> str:
    # Only a code can become a category; arbitrary angle-bracket text cannot.
    normalized = text.replace("＜", "<").replace("＞", ">")
    bracket = re.search(r"<([^<>]+)>", normalized)
    if bracket is not None:
        match = _ICD.search(bracket.group(1))
        if match is not None:
            return match.group(0).upper()
    match = _ICD.search(normalized)
    return match.group(0).upper() if match is not None else ""


def _opaque_categories(
    values: pd.Series,
    fit_mask: pd.Series,
    prefix: str,
    top_n: int,
) -> tuple[pd.Series, dict]:
    missing = values.isin(["", "未知"])
    training = values.loc[fit_mask & ~missing]
    counts = training.value_counts()
    # Tie order depends only on the training vocabulary, never held-out rows.
    retained = sorted(
        ((str(value), int(count)) for value, count in counts.items() if count >= CATEGORY_MIN_SUPPORT),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]
    vocabulary = {value: f"{prefix}_{position:03d}" for position, (value, _) in enumerate(retained, 1)}
    encoded = values.map(vocabulary).fillna(f"{prefix}_unknown")
    encoded = encoded.where(~missing, f"{prefix}_missing").astype(str)
    # Counts suffice for auditing support and never disclose the vocabulary.
    state = {
        "top_n": int(top_n),
        "min_support": CATEGORY_MIN_SUPPORT,
        "fit_nonmissing_count": int(len(training)),
        "fit_missing_count": int((fit_mask & missing).sum()),
        "fit_unique_count": int(len(counts)),
        "retained_count": int(len(retained)),
        "fit_unknown_count": int(encoded.loc[fit_mask].eq(f"{prefix}_unknown").sum()),
        "retained_support": {vocabulary[value]: count for value, count in retained},
    }
    return encoded, state


def build_order_features(
    requests: pd.DataFrame,
    fit_mask: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Build an allowlisted frame and a JSON-serializable, text-free fit summary.

    ``requests`` must contain distinct requests with a unique internal index.
    ``fit_mask`` is aligned by that index and must cover exactly these requests.
    It is the caller's responsibility to mark only requests visible during
    training, counting each request once rather than each expanded snapshot.
    No issue outcome, cancellation, date, identifier, or raw text is exported.
    The summary intentionally cannot reconstruct the private category mapping;
    apply this function once to the combined requests with the training mask.
    """
    if not isinstance(requests, pd.DataFrame) or not isinstance(fit_mask, pd.Series):
        raise TypeError("Feature input must be a DataFrame and an indexed Series mask.")
    if not requests.index.is_unique or not requests.columns.is_unique or not fit_mask.index.is_unique:
        raise ValueError("Feature inputs require unique indices and columns.")
    if len(fit_mask) != len(requests) or not requests.index.isin(fit_mask.index).all():
        raise ValueError("Training mask must cover exactly the feature input index.")
    fit_mask = fit_mask.reindex(requests.index)
    if fit_mask.isna().any() or not fit_mask.isin([True, False]).all():
        raise ValueError("Training mask must contain only nonmissing boolean values.")
    fit_mask = fit_mask.astype(bool)
    if not fit_mask.any():
        raise ValueError("No training requests are available to fit order features.")

    features = pd.DataFrame(index=requests.index)
    state: dict = {"fit_request_count": int(fit_mask.sum()), "numeric": {}, "amounts": {}, "categories": {}}

    for column, prefix in [("申请用血量", "request_use"), ("申请备血量", "request_reserve")]:
        values = _nonnegative_numeric(_column(requests, column))
        features[f"{prefix}_missing"] = values.isna().astype(int)
        features[column] = values.fillna(0.0)
        features[f"log1p_{prefix}"] = np.log1p(features[column])
        features[f"{prefix}_ge_8"] = values.ge(REQUEST_LARGE_THRESHOLD).astype(int)
        state["amounts"][column] = {
            "fit_valid_count": int(values.loc[fit_mask].notna().sum()),
            "fit_missing_count": int(values.loc[fit_mask].isna().sum()),
        }

    # Retain missingness before imputation and retain valid zero-valued ages.
    age_source = _column(requests, "年龄") if "年龄" in requests.columns else _column(requests, "年龄(岁)")
    ages = _nonnegative_numeric(age_source.map(parse_age_to_years))
    alt = _nonnegative_numeric(_column(requests, "ALT").map(_first_number))
    numeric_inputs = [
        ("血红蛋白", _nonnegative_numeric(_column(requests, "血红蛋白")), "hb_missing"),
        ("红细胞压积", _nonnegative_numeric(_column(requests, "红细胞压积")), "hct_missing"),
        ("年龄(岁)", ages, "age_missing"),
        ("ALT数值", alt, None),
    ]
    for column, values, missing_column in numeric_inputs:
        if missing_column is not None:
            features[missing_column] = values.isna().astype(int)
        features[column], state["numeric"][column] = _fit_numeric(values, fit_mask)
    features["ALT有数值"] = alt.notna().astype(int)
    features["年龄分组"] = ages.map(_age_group)
    features["输血要求分组"] = _clean_text(_column(requests, "输血要求")).map(_purpose)

    diagnosis = _clean_text(_column(requests, "入院诊断"))
    diagnosis_text = (
        diagnosis.str.replace("＜", "<", regex=False)
        .str.replace("＞", ">", regex=False)
        .str.replace(r"<[^<>]*>", "", regex=True)
        .str.strip("<>")
    )
    for suffix, pattern in [
        ("肿瘤", "肿瘤"),
        ("白血病", "白血病"),
        ("出血", "出血"),
        ("创伤", "创伤|外伤"),
        ("肝病", "肝"),
    ]:
        features[f"诊断_{suffix}"] = diagnosis_text.str.contains(pattern, regex=True).astype(int)

    history = _clean_text(_column(requests, "病史情况"))
    features["病史缺失"] = history.eq("").astype(int)
    for suffix, pattern in [
        ("既往输血史", "既往输血史|输血史"),
        ("妊娠史", "妊娠史"),
        ("流产史", "流产史"),
        ("输血反应史", "输血反应史"),
    ]:
        features[f"病史_{suffix}"] = history.str.contains(pattern, regex=True).astype(int)

    surgery = _clean_text(_column(requests, "手术名称"))
    features["手术缺失"] = surgery.eq("").astype(int)
    for suffix in ["切除", "置换", "移植", "开颅", "开胸", "剖宫产", "搭桥"]:
        features[f"手术_{suffix}"] = surgery.str.contains(suffix, regex=False).astype(int)

    department = _clean_text(_column(requests, "科室名称")).str.replace(r"[（(][^）)]*[）)]", "", regex=True)
    for column, values, prefix, top_n in [
        ("科室高频分组", department, "dept", DEPT_TOP_N),
        ("手术高频分组", surgery, "surgery", SURGERY_TOP_N),
        ("入院诊断ICD分组", diagnosis.map(_icd_code), "icd", ICD_TOP_N),
    ]:
        features[column], state["categories"][column] = _opaque_categories(values, fit_mask, prefix, top_n)

    return features.loc[:, ORDER_NUM_COLS[1:] + ORDER_CAT_COLS], state


# 三、时间划分、数据导出与命令行入口

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent
REQUEST_COLUMNS = {
    "申请编码",
    "ABO血型",
    "输血品种",
    "申请日期",
    "输血科发血日期",
    "撤销日期",
    "撤消标志",
    "是否发血",
    "发血量",
    "输用日期",
    "申请用血量",
    "申请备血量",
    "血红蛋白",
    "红细胞压积",
    "年龄",
    "ALT",
    "输血要求",
    "入院诊断",
    "病史情况",
    "科室名称",
    "手术名称",
}
INVENTORY_COLUMNS = {"血液类别", "血液名称", "ABO血型", "单位", "血量", "入库日期", "出库日期", "失效日期"}
FOLDS = {
    "val2017": ("2016-12-28", "2017-01-01", "2017-12-31", "validation"),
    "val2018": ("2017-12-28", "2018-01-01", "2018-12-31", "validation"),
    "val2019": ("2018-12-28", "2019-01-01", "2019-12-31", "validation"),
    "final2020": ("2019-12-28", "2020-01-01", "2020-11-30", "test"),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def scaler_state(frame, columns, mask):
    train = frame.loc[mask, columns].astype(float)
    means = train.mean()
    stds = train.std(ddof=0).replace(0, 1).fillna(1)
    if not np.isfinite(means.to_numpy()).all():
        raise ValueError("Cannot fit numeric scaler from missing training values.")
    return {name: {"mean": float(means[name]), "std": float(stds[name])} for name in columns}


def describe(series):
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return {"count": 0, "mean": None, "median": None, "q90": None, "total": None}
    return {
        "count": len(valid),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "q90": float(valid.quantile(0.9)),
        "total": float(valid.sum()),
    }


def candidate_aggregates(rows):
    rows = rows.copy()
    rows["hb_below80"] = rows["血红蛋白"].lt(80).astype(float)
    rows["hct_below25"] = rows["红细胞压积"].lt(25).astype(float)
    rows["urgent_request"] = rows["输血要求分组"].eq("急救用血").astype(float)
    rows["surgery_prepare"] = rows["输血要求分组"].eq("手术备血").astype(float)
    rows["ward_use"] = rows["输血要求分组"].eq("病房用血").astype(float)
    grouped = rows.groupby("decision_day")
    agg = grouped.agg(
        n_orders=("order_idx", "size"),
        hb_mean=("血红蛋白", "mean"),
        hb_p10=("血红蛋白", lambda s: s.quantile(0.1)),
        hct_mean=("红细胞压积", "mean"),
        hct_p10=("红细胞压积", lambda s: s.quantile(0.1)),
        hb_below80_rate=("hb_below80", "mean"),
        hct_below25_rate=("hct_below25", "mean"),
        urgent_request_rate=("urgent_request", "mean"),
        surgery_prepare_rate=("surgery_prepare", "mean"),
        ward_use_rate=("ward_use", "mean"),
        surgery_missing_rate=("手术缺失", "mean"),
        diagnosis_bleeding_rate=("诊断_出血", "mean"),
        diagnosis_tumor_rate=("诊断_肿瘤", "mean"),
        order_age_mean=("delta", "mean"),
    )
    for prefix, source in (("request_use", "申请用血量"), ("request_reserve", "申请备血量")):
        agg[f"{prefix}_sum"] = grouped[source].sum()
        agg[f"{prefix}_mean"] = grouped[source].mean()
        agg[f"{prefix}_p90"] = grouped[source].quantile(0.9)
        agg[f"{prefix}_ge_8_count"] = grouped[f"{prefix}_ge_8"].sum()
        agg[f"{prefix}_ge_8_rate"] = grouped[f"{prefix}_ge_8"].mean()
    for age in (0, 1, 2):
        agg[f"order_age_{age}_count"] = rows.delta.eq(age).groupby(rows.decision_day).sum()
    return agg.reset_index()


def add_calendar(daily):
    day = daily.decision_day.dt
    result = pd.DataFrame(index=daily.index)
    for prefix, values, period in (
        ("dow", day.dayofweek, 7),
        ("month", day.month - 1, 12),
        ("doy", day.dayofyear - 1, 365.25),
    ):
        result[f"{prefix}_sin"] = np.sin(2 * np.pi * values / period)
        result[f"{prefix}_cos"] = np.cos(2 * np.pi * values / period)
    result["is_weekend"] = day.dayofweek.ge(5).astype(float)
    return result


def export_fold(out_dir, name, req, snapshots, daily_base, history, order_keys, source_hashes, start, end):

    train_end_text, eval_start_text, eval_end_text, role = FOLDS[name]
    train_end, eval_start, eval_end = map(pd.Timestamp, (train_end_text, eval_start_text, eval_end_text))
    fit_indices = snapshots.loc[snapshots.decision_day.between(start, train_end), "order_idx"].unique()
    fit_mask = pd.Series(req.index.isin(fit_indices), index=req.index)
    features, feature_state = build_order_features(req, fit_mask)
    rows = snapshots.merge(features, left_on="order_idx", right_index=True, how="left", validate="many_to_one")
    rows["order_key"] = rows.order_idx.map(order_keys)
    fit_row = rows.decision_day.between(start, train_end)
    planned = rows.loc[fit_row, "planned_use_offset_days"].dropna()
    bounds = [float(planned.quantile(0.01)), float(planned.quantile(0.99))] if len(planned) else [0.0, 0.0]
    rows["planned_use_offset_days"] = rows.planned_use_offset_days.clip(*bounds).fillna(0.0)
    order_num = list(ORDER_NUM_COLS) + ["planned_use_offset_days", "planned_use_missing"]
    daily = daily_base.loc[daily_base.decision_day.between(start, end)].copy().reset_index(drop=True)
    aggregates = candidate_aggregates(rows)
    aggregate_cols = [c for c in aggregates if c != "decision_day"]
    daily = daily.merge(aggregates, on="decision_day", how="left", validate="one_to_one")
    daily[aggregate_cols] = daily[aggregate_cols].fillna(0.0)
    calendar_features = add_calendar(daily)
    daily = pd.concat([daily, calendar_features], axis=1)
    daily = daily.merge(history, on="decision_day", how="left", validate="one_to_one")
    history_cols = [c for c in history if c != "decision_day"]
    train_day = daily.decision_day.between(start, train_end)
    history_state, missing_cols = {}, []
    for column in history_cols:
        missing_name = column + "_missing"
        daily[missing_name] = daily[column].isna().astype(float)
        missing_cols.append(missing_name)
        values = daily.loc[train_day, column].dropna()
        fill = float(values.median()) if len(values) else 0.0
        history_state[column] = {"median": fill, "training_nonmissing": len(values)}
        daily[column] = daily[column].fillna(fill)
    day_cols = list(calendar_features) + aggregate_cols + history_cols + missing_cols
    represented = rows.groupby("decision_day").real_demand.agg(lambda s: s.sum(skipna=False))
    daily["known_recent_D"] = daily.decision_day.map(represented)
    daily.loc[~daily.decision_day.isin(represented.index), "known_recent_D"] = 0.0
    daily["order_label_missing_count"] = (
        daily.decision_day.map(rows.real_demand.isna().groupby(rows.decision_day).sum()).fillna(0).astype(int)
    )
    daily["split"] = "future"
    daily.loc[train_day, "split"] = "train"
    daily.loc[daily.decision_day.gt(train_end) & daily.decision_day.lt(eval_start), "split"] = "gap"
    daily.loc[daily.decision_day.between(eval_start, eval_end), "split"] = role
    daily["supervised_eligible"] = (
        daily.demand_observed & daily.inventory_observed & daily["split"].isin(["train", role])
    )
    daily["test_is_retrospective"] = daily["split"].eq("test")
    rows = rows.merge(
        daily[["decision_day", "D", "It", "supervised_eligible", "split", *day_cols]],
        on="decision_day",
        how="left",
        validate="many_to_one",
    )
    if not np.isfinite(rows[order_num + day_cols].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite model inputs; source row contents suppressed.")
    if not rows.application_time.lt(rows.information_cutoff).all():
        raise ValueError("Snapshot contains an application beyond the cutoff.")
    if rows.duplicated(["decision_day", "order_key"]).any():
        raise ValueError("Duplicate derived order snapshots.")
    observed = daily.demand_observed
    if not np.allclose(
        daily.loc[observed, "D"], daily.loc[observed, ["D_existing", "D_new", "D_source_unknown"]].sum(axis=1)
    ):
        raise ValueError("Daily demand source decomposition failed.")
    return finish_export(
        out_dir,
        name,
        rows,
        daily,
        req,
        fit_mask,
        feature_state,
        bounds,
        history_state,
        order_num,
        list(ORDER_CAT_COLS),
        day_cols,
        source_hashes,
        start,
        end,
        train_end,
        eval_start,
        eval_end,
        role,
    )


def finish_export(
    out_dir,
    name,
    rows,
    daily,
    req,
    fit_mask,
    feature_state,
    bounds,
    history_state,
    order_num,
    order_cat,
    day_cols,
    source_hashes,
    start,
    end,
    train_end,
    eval_start,
    eval_end,
    role,
):
    train_day = daily.decision_day.between(start, train_end)
    train_rows = rows.decision_day.between(start, train_end)
    state = {
        "feature_fit_start": str(start.date()),
        "feature_fit_end": str(train_end.date()),
        "fit_order_count": int(fit_mask.sum()),
        "order_features": feature_state,
        "planned_use_offset_bounds": bounds,
        "history_imputation": history_state,
        "numeric_scalers": {
            "order": scaler_state(rows, order_num, train_rows),
            "day": scaler_state(daily, day_cols, train_day),
            "inventory": scaler_state(daily, ["It"], train_day & daily.inventory_observed),
        },
        "scalers_applied_to_exports": False,
    }
    split = {
        "train_start": str(start.date()),
        "train_end": str(train_end.date()),
        "evaluation_start": str(eval_start.date()),
        "evaluation_end": str(eval_end.date()),
        "evaluation_role": role,
        "purge_days": 3,
        "evaluation_is_retrospective": role == "test",
        "split_key": "target_day",
    }
    identity = {
        "format": "morning_t_v1",
        "start_day": str(start.date()),
        "end_day": str(end.date()),
        "feature_fit_start": str(start.date()),
        "feature_fit_end": str(train_end.date()),
        "fold": name,
        "sources_sha256": source_hashes,
        "pipeline_sha256": file_hash(__file__),
        "preprocessor_sha256": file_hash(Path(__file__).with_name("core.py")),
        "features_sha256": file_hash(Path(__file__).with_name("features.py")),
    }
    excluded = [
        "D",
        "D_existing",
        "D_new",
        "D_source_unknown",
        "known_recent_D",
        "real_demand",
        "tag",
        "q_hist_receipt",
        "inventory_departure",
        "expiry_fallback_departure",
        "order_key",
        "order_idx",
        "split",
        "supervised_eligible",
        "demand_observed",
        "inventory_observed",
        "information_cutoff",
        "target_day",
        "snapshot_day",
        "target_age_days",
        "decision_day",
        "order_label_observed",
        "order_label_missing_count",
        "unknown_issue_records",
        "event_coverage_supported",
        "test_is_retrospective",
        "expiry_fallback_units",
        "inventory_unresolved_records",
        "inventory_unresolved_units",
    ]
    schema = {
        "schema_version": "morning_t_v1",
        "cohort_version": "past_recent_pending_proxy_full_registered_demand",
        "date_col": "decision_day",
        "target_day_col": "target_day",
        "target_col": "D",
        "target_level": "day",
        "target_aggregation": "first",
        "daily_table": "daily_features.csv",
        "order_supervision_table": "order_supervision.csv",
        "order_day_target_col": "real_demand",
        "tag_col": "tag",
        "inventory_col": "It",
        "inventory_context_col": "It",
        "order_num_cols": order_num,
        "order_cat_cols": order_cat,
        "day_feature_cols": day_cols,
        "excluded_from_model_inputs": excluded,
        "temporal": identity,
        "notes": {
            "timing": "decision_day=target_day=t; cutoff=t00 before same-day events; snapshot_day=t-1.",
            "A": "Observable proxy: application dates t-3,t-2,t-1, with no issue/cancellation before cutoff. Not the complete live pending queue or a four-day expiry rule.",
            "D": "All dated valid registered O-red issuance, independent of candidate feature completeness, age window, and future delay. Not proven uncensored clinical demand.",
            "inventory": "Opening stock independently reconstructed. Missing departures use expiry as an approximation, not as an observed departure.",
            "static_application_features": "Recorded application features and planned-use date are treated as request-time covariates; field-version history is unavailable.",
            "privacy": "Only opaque HMAC order keys and coded categories exported. Data remain sensitive and local. No raw identifiers or clinical free text.",
            "history": "Past demand and application-activity summaries use only dates before t. No separate time-series model or residual head.",
            "training": "Use daily supervised_eligible and retain valid empty sets. Numerical exports are imputed/clipped, not standardized; training-only scaler state is saved.",
            "legacy_readers": "Closed-cohort equality assertions and Flat-ETO convolution over visible orders alone do not describe full-day D. Training must explicitly select this new schema.",
        },
    }
    summary = {
        "temporal": identity,
        "final_days": len(daily),
        "final_rows": len(rows),
        "empty_candidate_days": int(daily.n_orders.eq(0).sum()),
        "days_with_unknown_demand": int((~daily.demand_observed).sum()),
        "eligible_days": int(daily.supervised_eligible.sum()),
        "invalid_inventory_days": int((~daily.inventory_observed).sum()),
        "feature_fit_training_applications": int(fit_mask.sum()),
        "order_num_dim": len(order_num),
        "order_cat_dim": len(order_cat),
        "day_feature_dim": len(day_cols),
        "demand": describe(daily.D),
        "inventory": describe(daily.It),
        "candidate_count": describe(daily.n_orders),
        "new_application_demand_fraction": float(daily.D_new.sum() / daily.D.sum()),
        "recent_known_demand_fraction": float(daily.known_recent_D.sum() / daily.D.sum()),
        "split_counts": {str(k): int(v) for k, v in daily["split"].value_counts().items()},
        "private_identifiers_exported": False,
    }
    row_cols = [
        "decision_day",
        "target_day",
        "snapshot_day",
        "information_cutoff",
        "order_key",
        "target_age_days",
        "tag",
        "real_demand",
        "order_label_observed",
        "D",
        *order_num,
        *order_cat,
        *day_cols,
        "It",
        "split",
        "supervised_eligible",
    ]
    supervision_cols = [
        "decision_day",
        "target_day",
        "snapshot_day",
        "information_cutoff",
        "order_key",
        "real_demand",
        "tag",
        "order_label_observed",
        "split",
    ]
    folder = Path(out_dir) / name
    folder.mkdir(exist_ok=False)
    save_json(folder / "manifest.json", {"identity": identity, "complete": False})
    for filename, frame in (
        ("data.csv", rows[row_cols]),
        ("daily_features.csv", daily),
        ("order_supervision.csv", rows[supervision_cols]),
    ):
        frame.to_csv(folder / filename, index=False, encoding="utf-8-sig")
    for filename, content in (
        ("data_feature_schema.json", schema),
        ("data_summary.json", summary),
        ("preprocessing_state.json", state),
        ("split_manifest.json", split),
    ):
        save_json(folder / filename, content)
    artifacts = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"}
    save_json(folder / "manifest.json", {"identity": identity, "complete": True, "artifacts": artifacts})
    return summary


def main(argv=None):

    parser = argparse.ArgumentParser(description="Create separate local morning-t preprocessing folds.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fold", choices=["all", *FOLDS], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)
    data_root = args.data_root.resolve()
    output = (args.output or data_root / "processed").resolve()
    if output.exists():
        raise ValueError("Output already exists. Use a fresh directory; existing data are never overwritten.")
    sources = {"applications": data_root / "申请表明细.csv", "inventory": data_root / "库存明细.csv"}
    hashes_before = {name: file_hash(path) for name, path in sources.items()}
    raw = pd.read_csv(
        sources["applications"], usecols=lambda c: c in REQUEST_COLUMNS, dtype={"申请编码": "string"}, low_memory=False
    )
    req, request_stats = normalize_requests(raw)
    del raw
    inventory_raw = pd.read_csv(sources["inventory"], usecols=lambda c: c in INVENTORY_COLUMNS, low_memory=False)
    output.mkdir(parents=True, exist_ok=False)
    salt = secrets.token_bytes(32)
    (output / ".private_order_key_salt").write_bytes(salt)
    raw_ids = req.get("申请编码", pd.Series(pd.NA, index=req.index, dtype="string")).astype("string")
    fallback_ids = pd.Series("missing-row:" + req.index.astype(str), index=req.index)
    raw_ids = ("source:" + raw_ids).fillna(fallback_ids)
    order_keys = {
        idx: hmac.new(salt, value.encode("utf-8"), hashlib.sha256).hexdigest()[:32] for idx, value in raw_ids.items()
    }
    req = req.drop(columns=["申请编码"], errors="ignore")
    del raw_ids
    start, end = pd.Timestamp("2013-01-01"), pd.Timestamp("2020-11-30")
    calendar = pd.date_range(start - pd.Timedelta(days=28), end)
    daily, demand_stats = build_demand(req, calendar)
    inventory, inventory_stats = build_inventory(inventory_raw, calendar)
    del inventory_raw
    daily = daily.merge(inventory, on="decision_day", how="left", validate="one_to_one")
    activity = build_request_activity(req, calendar)
    history = build_history(daily, activity)
    snapshots = build_snapshots(req, pd.date_range(start, end))
    save_json(
        output / "source_quality.json",
        {
            "requests": request_stats,
            "demand": demand_stats,
            "inventory": inventory_stats,
            "product_scope": "Original O-red request-name and inventory-code scope retained; no new substitution assumptions.",
        },
    )
    summaries = {}
    for name in list(FOLDS) if args.fold == "all" else [args.fold]:
        summaries[name] = export_fold(
            output, name, req, snapshots, daily, history, order_keys, hashes_before, start, end
        )
        print(
            json.dumps(
                {
                    "fold_completed": name,
                    "rows": summaries[name]["final_rows"],
                    "days": summaries[name]["final_days"],
                    "eligible_days": summaries[name]["eligible_days"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    hashes_after = {name: file_hash(path) for name, path in sources.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("A source changed during read-only preprocessing; outputs must not be used.")
    save_json(
        output / "run_manifest.json",
        {
            "complete": True,
            "sources_before_sha256": hashes_before,
            "sources_after_sha256": hashes_after,
            "sources_unchanged": True,
            "folds": list(summaries),
            "source_quality": "source_quality.json",
            "schema": "morning_t_v1",
        },
    )
    print(
        json.dumps(
            {"output_directory": str(output), "sources_unchanged": True, "fold_count": len(summaries)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
