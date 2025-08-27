import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def detect_drift(reference_data_path: str, current_data_path: str):
    ref = pd.read_csv(reference_data_path)
    cur = pd.read_csv(current_data_path)

    for df in (ref, cur):
        if "target" in df.columns:
            df.drop(columns=["target"], inplace=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    data = report.as_dict()

    drift_detected = data["metrics"][0]["result"]["dataset_drift"]
    by_cols = data["metrics"][1]["result"]["drift_by_columns"]

    chosen = list(by_cols.keys())[:3]
    feature_drifts = {k: float(by_cols[k]["drift_score"]) for k in chosen}
    overall = float(sum(feature_drifts.values()) / len(feature_drifts)) if feature_drifts else 0.0

    out = {
        "drift_detected": bool(drift_detected),
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall,
    }

    with open("reports/drift_report.json", "w") as f:
        json.dump(out, f, indent=2)

    return out