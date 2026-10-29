"""Synthetic-only checks for the morning decision data contract.

Run directly with ``python -m pytest tests/test_preprocessing.py``.  Every
fixture is created in memory; these tests never open the protected source
tables, write derived datasets, or launch model training.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


from 数据预处理 import (
    build_demand,
    build_history,
    build_inventory,
    build_snapshots,
    normalize_requests,
    build_order_features,
)


def test_script_runs_alone_and_refuses_existing_output(tmp_path):
    """A copied script needs no project modules and never overwrites existing data."""
    data_dir = tmp_path / "standalone_data"
    data_dir.mkdir()
    source = Path(__file__).resolve().parents[1] / "data" / "数据预处理.py"
    script = data_dir / source.name
    shutil.copy2(source, script)
    output = data_dir / "processed"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("fabricated fixture", encoding="utf-8")

    def launch(*args):
        return subprocess.run(
            [sys.executable, "-I", "-B", str(script), *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    help_result = launch("--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "--data-root" in help_result.stdout and "--output" in help_result.stdout
    result = launch("--fold", "val2019")
    assert result.returncode != 0 and "Output already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "fabricated fixture"
    assert not (data_dir / "申请表明细.csv").exists()


def test_root_cli_delegates_to_the_standalone_script(tmp_path):
    from dat_study.cli import main

    output = tmp_path / "already_exists"
    output.mkdir()
    with unittest.TestCase().assertRaisesRegex(ValueError, "Output already exists"):
        main(["preprocess", "--data-root", str(tmp_path), "--output", str(output)])


def request(case, application, issue=None, amount=None, cancel=None, **changes):
    """One synthetic O red-cell request, with realistic source column names."""
    row = {
        "synthetic_case": case,
        "申请日期": application,
        "输血科发血日期": issue,
        "撤销日期": cancel,
        "是否发血": int(issue is not None),
        "撤消标志": int(cancel is not None),
        "发血量": amount,
        "输血品种": "悬浮红细胞",
        "ABO血型": "O",
        "申请用血量": 2.0,
        "申请备血量": 0.0,
        "血红蛋白": 100.0,
        "红细胞压积": 30.0,
        "年龄": "40岁",
        "ALT": "20",
        "输血要求": "病房用血",
        "科室名称": "合成科室",
        "手术名称": "合成手术",
        "入院诊断": "<D50>贫血",
        "病史情况": "无",
        "预定输血日期": None,
    }
    row.update(changes)
    return row


def inventory(receipt, departure=None, amount=2.0, expiry=None, **changes):
    row = {
        "血液类别": "HX",
        "血液名称": "悬浮红细胞",
        "ABO血型": "O",
        "单位": "U",
        "血量": amount,
        "入库日期": receipt,
        "出库日期": departure,
        "失效日期": expiry,
    }
    row.update(changes)
    return row


def normalized(*rows):
    return normalize_requests(pd.DataFrame(rows))[0]


def row_on(frame, day):
    selected = frame.loc[pd.to_datetime(frame["decision_day"]).eq(pd.Timestamp(day))]
    if len(selected) != 1:
        raise AssertionError(f"Expected one daily row for {day}, got {len(selected)}")
    return selected.iloc[0]


def case_for_snapshot(req, snapshots):
    """Map the preserved internal row key back to its synthetic fixture label."""
    if "order_idx" in req.columns:
        lookup = req.set_index("order_idx")["synthetic_case"]
    else:
        lookup = req["synthetic_case"]
    return snapshots["order_idx"].map(lookup)


class MorningRequestContractTests(unittest.TestCase):
    def test_normalization_filters_product_only_and_preserves_source(self):
        raw = pd.DataFrame(
            [
                request(
                    "missing_clinical",
                    "2018-01-09",
                    "2018-01-10",
                    2,
                    **{"血红蛋白": np.nan, "红细胞压积": np.nan, "年龄": None},
                ),
                request("large_request", "2018-01-09", "2018-01-10", 3, **{"申请用血量": 1000, "申请备血量": 1000}),
                request(
                    "missing_request", "2018-01-09", "2018-01-10", 4, **{"申请用血量": np.nan, "申请备血量": np.nan}
                ),
                request("unknown_flag", "2018-01-09", **{"是否发血": None}),
                request("late_issue", "2018-01-01", "2018-01-10", 5),
                request("other_abo", "2018-01-09", "2018-01-10", 100, **{"ABO血型": "A"}),
                request("other_product", "2018-01-09", "2018-01-10", 100, **{"输血品种": "血小板"}),
            ]
        )
        original = raw.copy(deep=True)
        req, _ = normalize_requests(raw)
        self.assertEqual(
            set(req["synthetic_case"]),
            {"missing_clinical", "large_request", "missing_request", "unknown_flag", "late_issue"},
        )
        self.assertTrue(req.index.is_unique)
        self.assertTrue(pd.api.types.is_integer_dtype(req.index))
        pd.testing.assert_frame_equal(raw, original)

    def test_demand_uses_all_dated_issues_independently_of_snapshot_cohort(self):
        req = normalized(
            request("pending", "2018-01-09 12:00", "2018-01-10 00:00", 2),
            request("older_than_three_days", "2018-01-01", "2018-01-10 10:00", 5),
            request("same_day_new", "2018-01-10 01:00", "2018-01-10 07:00", 6),
            request(
                "missing_predictors", "2018-01-09", "2018-01-10 18:00", 4, **{"血红蛋白": np.nan, "申请用血量": np.nan}
            ),
        )
        calendar = pd.date_range("2018-01-10", periods=1)
        daily, _ = build_demand(req, calendar)
        day = row_on(daily, "2018-01-10")
        self.assertEqual(day["D"], 17)
        self.assertEqual(day["D_existing"], 11)
        self.assertEqual(day["D_new"], 6)
        self.assertTrue(day["demand_observed"])
        cases = set(case_for_snapshot(req, build_snapshots(req, calendar)))
        self.assertEqual(cases, {"pending", "missing_predictors"})

    def test_missing_application_date_does_not_delete_dated_demand(self):
        req = normalized(request("missing_application", None, "2018-01-10 12:00", 9))
        daily, _ = build_demand(req, pd.date_range("2018-01-10", periods=1))
        self.assertEqual(row_on(daily, "2018-01-10")["D"], 9)

    def test_duplicate_application_keys_fail_before_double_counting(self):
        raw = pd.DataFrame(
            [
                request("duplicate_one", "2018-01-09", "2018-01-10", 2, **{"申请单号": "SYNTHETIC_DUPLICATE_KEY"}),
                request("duplicate_two", "2018-01-09", "2018-01-10", 2, **{"申请单号": "SYNTHETIC_DUPLICATE_KEY"}),
            ]
        )
        original = raw.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            normalize_requests(raw)
        pd.testing.assert_frame_equal(raw, original)

    def test_unknown_event_quantity_is_missing_demand_but_no_event_is_zero(self):
        req = normalized(
            request("known", "2018-01-09", "2018-01-10 12:00", 2),
            request("event_unknown_quantity", "2018-01-09", "2018-01-10 13:00", None),
            request("no_event", "2018-01-09", None, None),
            request("coverage_anchor", "2018-01-11", "2018-01-12", 1),
        )
        daily, _ = build_demand(req, pd.date_range("2018-01-10", periods=2))
        unknown = row_on(daily, "2018-01-10")
        self.assertTrue(pd.isna(unknown["D"]))
        self.assertFalse(unknown["demand_observed"])
        self.assertEqual(unknown["unknown_issue_records"], 1)
        empty = row_on(daily, "2018-01-11")
        self.assertEqual(empty["D"], 0)
        self.assertTrue(empty["demand_observed"])
        self.assertEqual(empty["unknown_issue_records"], 0)

    def test_calendar_contract_keeps_days_without_requests(self):
        req = normalized(
            request("coverage_start", "2018-01-07", "2018-01-08", 1),
            request("one", "2018-01-08", "2018-01-10", 2),
            request("coverage_end", "2018-01-12", "2018-01-13", 1),
        )
        calendar = pd.date_range("2018-01-09", periods=4)
        daily, _ = build_demand(req, calendar)
        pd.testing.assert_index_equal(pd.DatetimeIndex(daily["decision_day"]), calendar, check_names=False)
        pd.testing.assert_series_equal(
            pd.to_datetime(daily["target_day"]), pd.to_datetime(daily["decision_day"]), check_names=False
        )
        pd.testing.assert_series_equal(
            pd.to_datetime(daily["snapshot_day"]),
            pd.to_datetime(daily["decision_day"]) - pd.Timedelta(days=1),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            pd.to_datetime(daily["information_cutoff"]), pd.to_datetime(daily["decision_day"]), check_names=False
        )
        self.assertEqual(list(daily["D"]), [0, 2, 0, 0])

    def test_days_outside_event_coverage_are_not_fabricated_zero_demand(self):
        req = normalized(request("one", "2018-01-09", "2018-01-10", 2))
        daily, _ = build_demand(req, pd.date_range("2018-01-09", periods=3))
        observed = row_on(daily, "2018-01-10")
        self.assertEqual(observed["D"], 2)
        self.assertTrue(observed["demand_observed"])
        for unsupported_day in ["2018-01-09", "2018-01-11"]:
            day = row_on(daily, unsupported_day)
            self.assertFalse(day["event_coverage_supported"])
            self.assertFalse(day["demand_observed"])
            self.assertTrue(pd.isna(day["D"]))


class MorningSnapshotContractTests(unittest.TestCase):
    def test_midnight_boundary_age_and_cancellation_eligibility(self):
        req = normalized(
            request("age_one", "2018-01-09 23:59:59", "2018-01-10 00:00:00", 2),
            request("age_two", "2018-01-08 12:00", "2018-01-10 12:00", 3),
            request("age_three_future_issue", "2018-01-07", "2018-01-11", 4),
            request("age_four", "2018-01-06", "2018-01-10", 5),
            request("new_at_midnight", "2018-01-10 00:00:00", "2018-01-10 08:00", 6),
            request("already_issued", "2018-01-09", "2018-01-09 23:59:59", 7),
            request("cancel_at_midnight", "2018-01-09", cancel="2018-01-10 00:00:00"),
            request("already_cancelled", "2018-01-09", cancel="2018-01-09 23:59:59"),
            request("future_cancellation", "2018-01-09", cancel="2018-01-11 10:00"),
            request("still_pending", "2018-01-09"),
        )
        snapshots = build_snapshots(req, pd.date_range("2018-01-10", periods=1))
        snapshots = snapshots.assign(synthetic_case=case_for_snapshot(req, snapshots)).set_index("synthetic_case")
        self.assertEqual(
            set(snapshots.index),
            {
                "age_one",
                "age_two",
                "age_three_future_issue",
                "cancel_at_midnight",
                "future_cancellation",
                "still_pending",
            },
        )
        for case, delta in [("age_one", 0), ("age_two", 1), ("age_three_future_issue", 2)]:
            self.assertEqual(snapshots.loc[case, "delta"], delta)
            self.assertEqual(snapshots.loc[case, "target_age_days"], delta + 1)
        self.assertEqual(snapshots.loc["age_one", "real_demand"], 2)
        self.assertEqual(snapshots.loc["age_two", "real_demand"], 3)
        self.assertEqual(snapshots.loc["age_three_future_issue", "real_demand"], 0)
        self.assertEqual(snapshots.loc["age_three_future_issue", "tag"], 0)
        self.assertEqual(snapshots.loc["age_one", "tag"], 1)
        self.assertEqual(snapshots.loc["still_pending", "tag"], 0)

    def test_late_future_outcome_does_not_remove_earlier_snapshot(self):
        calendar = pd.date_range("2018-01-08", "2018-01-10")
        first = normalized(request("late", "2018-01-07 10:00", "2018-01-11", 4))
        later = normalized(request("late", "2018-01-07 10:00", "2018-01-20", 100))
        first_rows = build_snapshots(first, calendar)
        later_rows = build_snapshots(later, calendar)
        self.assertEqual(len(first_rows), 3)
        causal_columns = [
            "order_idx",
            "decision_day",
            "target_day",
            "snapshot_day",
            "information_cutoff",
            "delta",
            "target_age_days",
            "application_time",
        ]
        pd.testing.assert_frame_equal(first_rows[causal_columns], later_rows[causal_columns])
        self.assertTrue(first_rows["real_demand"].eq(0).all())
        self.assertTrue(later_rows["real_demand"].eq(0).all())

    def test_future_cancellation_flag_does_not_change_past_eligibility(self):
        calendar = pd.date_range("2018-01-10", periods=1)
        pending = normalized(request("future_cancel", "2018-01-09"))
        cancelled_later = normalized(request("future_cancel", "2018-01-09", cancel="2018-01-12"))
        columns = ["order_idx", "decision_day", "delta", "target_age_days", "real_demand", "tag"]
        pd.testing.assert_frame_equal(
            build_snapshots(pending, calendar)[columns], build_snapshots(cancelled_later, calendar)[columns]
        )

    def test_undated_final_cancellation_flag_does_not_delete_past_snapshots(self):
        calendar = pd.date_range("2018-01-10", periods=3)
        pending = normalized(request("cancel_status", "2018-01-09 12:00"))
        undated_final_cancel = normalized(request("cancel_status", "2018-01-09 12:00", **{"撤消标志": 1}))
        known_cancel = normalized(request("cancel_status", "2018-01-09 12:00", cancel="2018-01-11 00:00:00"))
        identity_columns = [
            "order_idx",
            "decision_day",
            "target_day",
            "snapshot_day",
            "information_cutoff",
            "delta",
            "target_age_days",
            "application_time",
        ]
        pending_rows = build_snapshots(pending, calendar)
        undated_rows = build_snapshots(undated_final_cancel, calendar)
        self.assertEqual(len(pending_rows), 3)
        pd.testing.assert_frame_equal(pending_rows[identity_columns], undated_rows[identity_columns])
        known_rows = build_snapshots(known_cancel, calendar)
        self.assertEqual(
            list(pd.to_datetime(known_rows["decision_day"])),
            [
                pd.Timestamp("2018-01-10"),
                pd.Timestamp("2018-01-11"),
            ],
        )

    def test_missing_event_quantity_is_not_a_negative_snapshot_label(self):
        req = normalized(
            request("unknown", "2018-01-09", "2018-01-10 08:00", None),
            request("none", "2018-01-09"),
        )
        snapshots = build_snapshots(req, pd.date_range("2018-01-10", periods=1))
        snapshots = snapshots.assign(synthetic_case=case_for_snapshot(req, snapshots)).set_index("synthetic_case")
        self.assertTrue(pd.isna(snapshots.loc["unknown", "real_demand"]))
        self.assertEqual(snapshots.loc["none", "real_demand"], 0)
        self.assertEqual(snapshots.loc["none", "tag"], 0)

    def test_undated_issue_evidence_is_not_a_known_negative_label(self):
        req = normalized(
            request("flag_only", "2018-01-09", **{"是否发血": 1}),
            request("amount_only", "2018-01-09", None, 4),
        )
        snapshots = build_snapshots(req, pd.date_range("2018-01-10", periods=1))
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(snapshots["real_demand"].isna().all())
        self.assertTrue(snapshots["tag"].isna().all())
        self.assertTrue((~snapshots["order_label_observed"].astype(bool)).all())

    def test_planned_use_source_is_retained_with_an_explicit_missing_flag(self):
        req = normalized(
            request("planned", "2018-01-09", **{"输用日期": "2018-01-11 15:00"}),
            request("unknown_plan", "2018-01-09"),
        )
        snapshots = build_snapshots(req, pd.date_range("2018-01-10", periods=1))
        snapshots = snapshots.assign(synthetic_case=case_for_snapshot(req, snapshots)).set_index("synthetic_case")
        self.assertEqual(snapshots.loc["planned", "planned_use_offset_days"], 1)
        self.assertEqual(snapshots.loc["planned", "planned_use_missing"], 0)
        self.assertEqual(snapshots.loc["unknown_plan", "planned_use_missing"], 1)

    def test_demand_and_snapshots_do_not_mutate_normalized_source(self):
        req = normalized(request("one", "2018-01-09 12:00", "2018-01-10 12:00", 2))
        original = req.copy(deep=True)
        calendar = pd.date_range("2018-01-09", periods=3)
        build_demand(req, calendar)
        build_snapshots(req, calendar)
        pd.testing.assert_frame_equal(req, original)


class MorningInventoryContractTests(unittest.TestCase):
    def test_timestamp_boundaries_and_same_day_receipts(self):
        raw = pd.DataFrame(
            [
                inventory("2018-01-09 12:00", "2018-01-10 00:00:00", 2),
                inventory("2018-01-10 00:00:00", "2018-01-12", 3),
                inventory("2018-01-09", "2018-01-09 23:59:59", 5),
                inventory("2018-01-09", "2018-01-10 12:00", 7),
                inventory("2018-01-10 12:00", "2018-01-12", 11),
            ]
        )
        original = raw.copy(deep=True)
        result, _ = build_inventory(raw, pd.date_range("2018-01-10", periods=2))
        self.assertEqual(row_on(result, "2018-01-10")["It"], 9)
        self.assertEqual(row_on(result, "2018-01-11")["It"], 14)
        self.assertEqual(row_on(result, "2018-01-10")["q_hist_receipt"], 14)
        self.assertEqual(row_on(result, "2018-01-10")["inventory_departure"], 9)
        pd.testing.assert_frame_equal(raw, original)

    def test_product_and_unit_filter_matches_request_scope(self):
        raw = pd.DataFrame(
            [
                inventory("2018-01-09", "2018-01-12", 2),
                inventory("2018-01-09", "2018-01-12", 3, **{"单位": "Ｕ", "血液类别": "XS1"}),
                inventory("2018-01-09", "2018-01-12", 100, **{"ABO血型": "A"}),
                inventory("2018-01-09", "2018-01-12", 100, **{"单位": "mL"}),
                inventory("2018-01-09", "2018-01-12", 100, **{"血液类别": "PLT"}),
            ]
        )
        result, _ = build_inventory(raw, pd.date_range("2018-01-10", periods=1))
        self.assertEqual(row_on(result, "2018-01-10")["It"], 5)

    def test_expiry_fallback_bounds_missing_departure(self):
        raw = pd.DataFrame(
            [
                inventory("2018-01-08", None, 4, "2018-01-10 00:00:00"),
                inventory("2018-01-08", "2018-01-12", 2),
            ]
        )
        result, _ = build_inventory(raw, pd.date_range("2018-01-10", periods=2))
        first, second = (row_on(result, day) for day in ["2018-01-10", "2018-01-11"])
        self.assertEqual(first["It"], 6)
        self.assertEqual(first["expiry_fallback_units"], 4)
        self.assertEqual(second["It"], 2)
        self.assertEqual(second["expiry_fallback_units"], 0)

    def test_missing_departure_and_expiry_are_explicitly_unresolved(self):
        raw = pd.DataFrame([inventory("2018-01-08", None, 4, None)])
        result, _ = build_inventory(raw, pd.date_range("2018-01-10", periods=2))
        self.assertTrue(result["inventory_unresolved_units"].eq(4).all())
        self.assertTrue((~result["inventory_observed"].astype(bool)).all())
        self.assertTrue(result["It"].isna().all())

    def test_unknown_stock_amount_does_not_become_zero_inventory(self):
        raw = pd.DataFrame(
            [
                inventory("2018-01-08", "2018-01-12", None),
                inventory("2018-01-08", "2018-01-12", 2),
            ]
        )
        result, _ = build_inventory(raw, pd.date_range("2018-01-10", periods=1))
        day = row_on(result, "2018-01-10")
        self.assertFalse(day["inventory_observed"])
        self.assertTrue(pd.isna(day["It"]))


class MorningHistoryContractTests(unittest.TestCase):
    @staticmethod
    def fixtures():
        days = pd.date_range("2017-12-01", periods=41)
        values = np.arange(1.0, len(days) + 1.0)
        daily = pd.DataFrame(
            {
                "decision_day": days,
                "target_day": days,
                "snapshot_day": days - pd.Timedelta(days=1),
                "information_cutoff": days,
                "D": 3 * values,
                "D_existing": 2 * values,
                "D_new": values,
                "demand_observed": True,
                "unknown_issue_records": 0,
            }
        )
        activity = pd.DataFrame(
            {
                "decision_day": days,
                "new_applications_count": values + 10,
                "new_request_use_sum": values + 100,
                "new_request_reserve_sum": values + 200,
            }
        )
        return daily, activity

    def test_history_uses_lagged_complete_calendar(self):
        daily, activity = self.fixtures()
        result = build_history(daily, activity)
        day = row_on(result, "2018-01-10")
        self.assertEqual(day["demand_lag1"], 120)
        self.assertEqual(day["demand_roll7_mean"], np.mean(daily["D"].iloc[33:40]))
        self.assertEqual(day["demand_roll14_mean"], np.mean(daily["D"].iloc[26:40]))
        self.assertEqual(day["demand_roll28_mean"], np.mean(daily["D"].iloc[12:40]))
        self.assertEqual(day["demand_same_weekday_lag7"], daily["D"].iloc[33])
        self.assertEqual(day["past_new_applications_count_lag1"], activity["new_applications_count"].iloc[39])

    def test_short_history_does_not_imply_complete_rolling_window(self):
        daily, activity = self.fixtures()
        result = build_history(daily, activity)
        before_seven = row_on(result, "2017-12-07")
        full_seven = row_on(result, "2017-12-08")
        self.assertTrue(pd.isna(before_seven["demand_roll7_mean"]))
        self.assertEqual(full_seven["demand_roll7_mean"], np.mean(daily["D"].iloc[:7]))

    def test_current_and_future_outcomes_and_activity_do_not_change_features(self):
        daily, activity = self.fixtures()
        target = pd.Timestamp("2018-01-05")
        expected = build_history(daily, activity)
        changed_daily, changed_activity = daily.copy(deep=True), activity.copy(deep=True)
        future = changed_daily["decision_day"].ge(target)
        changed_daily.loc[future, ["D", "D_existing", "D_new"]] = 987654.0
        changed_activity.loc[future, ["new_applications_count", "new_request_use_sum", "new_request_reserve_sum"]] = (
            765432.0
        )
        actual = build_history(changed_daily, changed_activity)
        before_cutoff = pd.to_datetime(expected["decision_day"]).le(target)
        pd.testing.assert_frame_equal(expected.loc[before_cutoff], actual.loc[before_cutoff])

    def test_unknown_previous_demand_does_not_become_zero_lag(self):
        daily, activity = self.fixtures()
        daily.loc[daily["decision_day"].eq(pd.Timestamp("2018-01-09")), "D"] = np.nan
        daily.loc[daily["decision_day"].eq(pd.Timestamp("2018-01-09")), "demand_observed"] = False
        result = build_history(daily, activity)
        self.assertTrue(pd.isna(row_on(result, "2018-01-10")["demand_lag1"]))

    def test_history_does_not_mutate_daily_or_activity_inputs(self):
        daily, activity = self.fixtures()
        original_daily, original_activity = daily.copy(deep=True), activity.copy(deep=True)
        build_history(daily, activity)
        pd.testing.assert_frame_equal(daily, original_daily)
        pd.testing.assert_frame_equal(activity, original_activity)


class MorningFeatureContractTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rows = [
            request(
                f"train_{i}",
                "2017-01-09",
                **{
                    "血红蛋白": 90 + 2 * i,
                    "红细胞压积": 25 + i,
                    "科室名称": "合成训练科室",
                    "手术名称": "合成训练手术",
                    "入院诊断": "<D50>贫血",
                },
            )
            for i in range(6)
        ]
        rows.append(
            request(
                "train_missing",
                "2017-01-09",
                **{
                    "血红蛋白": np.nan,
                    "红细胞压积": np.nan,
                    "年龄": None,
                    "申请用血量": np.nan,
                    "申请备血量": np.nan,
                },
            )
        )
        rows.extend(
            request(
                f"held_out_{i}",
                "2018-01-09",
                **{
                    "血红蛋白": 9999,
                    "科室名称": "合成未来科室",
                    "手术名称": "合成未来手术",
                    "入院诊断": "<C80>合成未来诊断",
                },
            )
            for i in range(8)
        )
        req = normalized(*rows)
        fit_mask = req["synthetic_case"].str.startswith("train_")
        return req, fit_mask

    def test_held_out_rows_do_not_fit_imputation_clipping_or_categories(self):
        req, fit_mask = self.fixture()
        original = req.copy(deep=True)
        features, state = build_order_features(req, fit_mask)
        changed = req.copy(deep=True)
        changed.loc[~fit_mask, "血红蛋白"] = 0.001
        changed.loc[~fit_mask, "红细胞压积"] = 99999
        changed.loc[~fit_mask, "科室名称"] = "另一个合成未来科室"
        changed.loc[~fit_mask, "手术名称"] = "另一个合成未来手术"
        changed.loc[~fit_mask, "入院诊断"] = "<Z99>另一个合成未来诊断"
        changed_features, changed_state = build_order_features(changed, fit_mask)
        self.assertEqual(state, changed_state)
        pd.testing.assert_frame_equal(features.loc[fit_mask], changed_features.loc[fit_mask])
        pd.testing.assert_frame_equal(req, original)
        self.assertEqual(state["fit_request_count"], 7)

    def test_missing_predictors_are_flagged_without_dropping_requests(self):
        req, fit_mask = self.fixture()
        features, _ = build_order_features(req, fit_mask)
        pd.testing.assert_index_equal(features.index, req.index)
        self.assertEqual(len(features), len(req))
        missing = req["synthetic_case"].eq("train_missing")
        for column in ["hb_missing", "hct_missing", "age_missing", "request_use_missing", "request_reserve_missing"]:
            self.assertEqual(features.loc[missing, column].iloc[0], 1)
        numeric = features.select_dtypes(include=[np.number])
        self.assertTrue(np.isfinite(numeric.to_numpy(dtype=float)).all())
        self.assertNotIn("delta", features.columns)
        for raw_column in ["synthetic_case", "科室名称", "手术名称", "病史情况", "入院诊断"]:
            self.assertNotIn(raw_column, features.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
