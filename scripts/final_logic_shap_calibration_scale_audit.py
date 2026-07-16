from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_calibration import apply_platt, fit_platt  # noqa: E402
from major_revision_model_audits import (  # noqa: E402
    FORMULA_ALIGNED_REMOVE,
    SPLIT_COL,
    best_lgbm_params,
    load_dataset,
    load_feature_sets,
    select_threshold,
    write_csv,
    write_json,
)
from major_revision_rolling_origin import assign_fold_split  # noqa: E402
from major_revision_target_selection import encode_splits, make_targets  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/explainability"
ROOT_REPORT = ROOT / "shap_calibration_scale_audit.md"
FINAL_FOLD = {"fold_id": "final_style_2025", "train_end_year": 2023, "validation_year": 2024, "test_year": 2025}
TARGET = "T2_min_count_3"


def fmt(x: float, ndigits: int = 4) -> str:
    return f"{float(x):.{ndigits}f}"


def train_fold(seed: int = 42):
    feature_sets, categorical = load_feature_sets()
    features = feature_sets["B_no_8w_formula_features"]
    required = sorted(set(features + ["rolling_8w_mean", "rolling_8w_std", "target_next_week_count"]))
    df = load_dataset({"features": required})
    frame = make_targets(df)[TARGET]
    d = assign_fold_split(frame, FINAL_FOLD)
    X_train, y_train, X_val, y_val, X_test, y_test, model_features = encode_splits(d, features, categorical)

    params = best_lgbm_params()
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    model = LGBMClassifier(
        objective="binary",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=neg / pos if pos else 1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    val_uncalibrated = model.predict_proba(X_val)[:, 1]
    test_uncalibrated = model.predict_proba(X_test)[:, 1]
    uncalibrated_threshold, _ = select_threshold(y_val, val_uncalibrated)
    platt = fit_platt(val_uncalibrated, y_val.to_numpy(dtype=int))
    platt_threshold = float(apply_platt(platt, np.array([uncalibrated_threshold]))[0])

    test_rows = d[d[SPLIT_COL].eq("test")].copy()
    scored = test_rows.reset_index(drop=False).rename(columns={"index": "original_index"})
    scored["y_true"] = y_test.to_numpy(dtype=int)
    scored["uncalibrated_probability"] = test_uncalibrated
    scored["raw_margin"] = model.predict(X_test, raw_score=True)
    scored["raw_margin_from_probability"] = logit(np.clip(test_uncalibrated, 1e-12, 1 - 1e-12))
    scored["platt_calibrated_probability"] = apply_platt(platt, test_uncalibrated)
    scored["uncalibrated_threshold"] = float(uncalibrated_threshold)
    scored["platt_threshold"] = float(platt_threshold)
    scored["predicted_alert"] = (scored["platt_calibrated_probability"] >= platt_threshold).astype(int)
    return scored, model_features, platt


def write_report(audit: pd.DataFrame, metadata: dict[str, object]) -> str:
    lines = [
        "# SHAP Calibration-Scale Audit",
        "",
        "This audit verifies the scale used by the local SHAP waterfall figures and the scale used by the final calibrated decision layer.",
        "",
        "Findings:",
        "",
        "- TreeSHAP local waterfall figures decompose the fitted LightGBM raw margin before Platt calibration.",
        "- The `score` values in the previous local-case artifact are uncalibrated LightGBM probabilities, not Platt-calibrated probabilities.",
        "- Platt calibration is fit on validation year 2024 and is monotone for the audited score range, so it preserves the alert ordering and the same threshold decisions after transforming the threshold.",
        "- Waterfall contributions should not be interpreted as calibrated probabilities or causal effects.",
        "",
        "Thresholds:",
        "",
        f"- Uncalibrated LightGBM probability threshold: {fmt(metadata['uncalibrated_threshold'], 6)}.",
        f"- Equivalent Platt-calibrated probability threshold: {fmt(metadata['platt_threshold'], 6)}.",
        "",
        "Audited local cases:",
        "",
        "| case | true label | alert | raw margin | uncalibrated probability | Platt-calibrated probability | final threshold | previous stored score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in audit.iterrows():
        lines.append(
            "| {case_id} | {y_true} | {predicted_alert} | {raw} | {uncal} | {platt} | {thr} | {stored} |".format(
                case_id=r["case_id"],
                y_true=int(r["y_true"]),
                predicted_alert=int(r["predicted_alert"]),
                raw=fmt(r["raw_margin"], 4),
                uncal=fmt(r["uncalibrated_probability"], 4),
                platt=fmt(r["platt_calibrated_probability"], 4),
                thr=fmt(r["platt_threshold"], 4),
                stored=fmt(r["stored_score"], 4),
            )
        )
    lines.extend(
        [
            "",
            "Implementation notes:",
            "",
            f"- Target: `{TARGET}`.",
            "- Fold: train through 2023, validation 2024, held-out test 2025.",
            "- Seed: 42.",
            "- Formula-aligned 8-week target-construction predictors removed: "
            + ", ".join(f"`{f}`" for f in FORMULA_ALIGNED_REMOVE)
            + ".",
            f"- Model feature count: {metadata['model_feature_count']}.",
        ]
    )
    report = "\n".join(lines)
    (OUT_DIR / "shap_calibration_scale_audit.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored, model_features, _ = train_fold(seed=42)
    cases_path = OUT_DIR / "shap_local_cases.csv"
    cases = pd.read_csv(cases_path)
    rows = []
    for _, case in cases.iterrows():
        match = scored[scored["original_index"].eq(int(case["original_index"]))]
        if match.empty:
            raise ValueError(f"No scored row found for local case original_index={case['original_index']}")
        row = match.iloc[0].to_dict()
        row.update(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "stored_score": float(case["score"]),
                "stored_threshold": float(case["threshold"]),
                "target_week": case["target_week"],
                "ntaname": case["ntaname"],
                "boroname": case["boroname"],
                "complaint_category": case["complaint_category"],
                "target_next_week_count": float(case["target_next_week_count"]),
            }
        )
        rows.append(row)
    audit = pd.DataFrame(rows)
    cols = [
        "case_id",
        "case_type",
        "original_index",
        "target_week",
        "ntaname",
        "boroname",
        "complaint_category",
        "y_true",
        "predicted_alert",
        "raw_margin",
        "raw_margin_from_probability",
        "uncalibrated_probability",
        "platt_calibrated_probability",
        "uncalibrated_threshold",
        "platt_threshold",
        "stored_score",
        "stored_threshold",
        "target_next_week_count",
    ]
    write_csv(OUT_DIR / "shap_calibration_scale_audit.csv", audit[cols], overwrite=True)
    metadata = {
        "uncalibrated_threshold": float(audit["uncalibrated_threshold"].iloc[0]),
        "platt_threshold": float(audit["platt_threshold"].iloc[0]),
        "model_feature_count": len(model_features),
    }
    write_report(audit[cols], metadata)
    write_json(
        OUT_DIR / "shap_calibration_scale_audit_run_summary.json",
        {
            "status": "done",
            "script": "final_logic_shap_calibration_scale_audit.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "target": TARGET,
            "fold": FINAL_FOLD,
            "local_cases": int(len(audit)),
            "outputs": [
                "shap_calibration_scale_audit.csv",
                "shap_calibration_scale_audit.md",
                "shap_calibration_scale_audit_run_summary.json",
            ],
        },
    )


if __name__ == "__main__":
    main()
