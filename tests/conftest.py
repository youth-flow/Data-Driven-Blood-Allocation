import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import pandas as pd
import pytest
from dat_study.io import save_json, sha256


@pytest.fixture
def synthetic_fold(tmp_path):
    """Entirely fabricated data; CI never needs patient or source files."""
    folder = tmp_path / "derived" / "val2019"
    folder.mkdir(parents=True)
    days = pd.date_range("2019-01-01", periods=18)
    rows, daily = [], []
    for j, day in enumerate(days):
        n = 0 if j in (2, 15) else 2
        known = 0 if n == 0 else float(1 + j % 2)
        split = "train" if j < 12 else "validation"
        daily.append(
            dict(
                decision_day=day,
                target_day=day,
                snapshot_day=day - pd.Timedelta(days=1),
                information_cutoff=day,
                split=split,
                supervised_eligible=True,
                demand_observed=True,
                inventory_observed=True,
                D=known + 3 + j % 6,
                known_recent_D=known,
                It=260.0 + j * 2,
                n_orders=n,
                calendar_feature=j % 7,
            )
        )
        for k in range(n):
            rows.append(
                dict(
                    decision_day=day,
                    order_key=f"fabricated-{j}-{k}",
                    real_demand=known if k == 0 else 0.0,
                    order_label_observed=True,
                    delta=k,
                    申请用血量=2.0,
                    申请备血量=4.0,
                    科室高频分组="group_a",
                )
            )
    d, r = pd.DataFrame(daily), pd.DataFrame(rows)
    num = ["delta", "申请用血量", "申请备血量"]
    cat = ["科室高频分组"]
    day_cols = ["calendar_feature"]

    def scales(frame, cols):
        return {c: {"mean": float(frame[c].mean()), "std": float(frame[c].std(ddof=0)) or 1.0} for c in cols}

    state = {
        "feature_fit_end": "2019-01-12",
        "numeric_scalers": {
            "order": scales(r[r.decision_day <= days[11]], num),
            "day": scales(d.iloc[:12], day_cols),
            "inventory": scales(d.iloc[:12], ["It"]),
        },
    }
    schema = {
        "schema_version": "morning_t_v1",
        "order_num_cols": num,
        "order_cat_cols": cat,
        "day_feature_cols": day_cols,
        "excluded_from_model_inputs": ["D", "real_demand", "order_key"],
    }
    split = {
        "train_start": "2019-01-01",
        "train_end": "2019-01-12",
        "evaluation_start": "2019-01-13",
        "evaluation_end": "2019-01-18",
        "evaluation_role": "validation",
        "evaluation_is_retrospective": False,
    }
    d.to_csv(folder / "daily_features.csv", index=False)
    r.to_csv(folder / "data.csv", index=False)
    for name, value in [
        ("data_feature_schema.json", schema),
        ("preprocessing_state.json", state),
        ("split_manifest.json", split),
    ]:
        save_json(folder / name, value)
    save_json(folder / "manifest.json", {"complete": True, "artifacts": {p.name: sha256(p) for p in folder.iterdir()}})
    return folder


@pytest.fixture
def config():
    from dat_study.config import load_config, smoke_config
    from pathlib import Path

    c = smoke_config(load_config(Path(__file__).resolve().parents[1] / "configs/experiment.yaml"))
    c.setdefault("training", {})["device"] = "cpu"
    c["training"]["threads"] = 1
    return c
