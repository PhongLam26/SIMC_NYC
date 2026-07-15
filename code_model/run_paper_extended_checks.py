"""Run compact paper checks and supplemental experiments.

This script is intentionally scoped to reviewer-facing paper evidence. It uses
the existing final dataset and model-ready feature-set definitions, keeps the
chronological train/validation/test split, selects hard-alert thresholds only
on validation, and writes compact CSV/Markdown outputs used by the manuscript.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz"
FEATURE_SETS = ROOT / "data/processed/_model_summaries/inspect_final_dataset_feature_sets.json"
PAPER_TABLES = ROOT / "data/processed/model_results/paper_tables_target_week_fulltrain"
SHAP_DIR = ROOT / "data/processed/model_results/shap_explainability_lgbm_fulltrain"
OUT = ROOT / "data/processed/model_results/paper_extended_checks"
SEED = 42
MAX_TRAIN_ROWS = 300_000
SENSITIVITY_MAX_TRAIN_ROWS = 100_000


def _read_features() -> dict[str, list[str]]:
    raw = json.loads(FEATURE_SETS.read_text(encoding="utf-8"))
    return {k: v["features"] for k, v in raw.items()}


def _drop_unwanted(features: list[str]) -> list[str]:
    # Keep the paper's no-COVID-period convention for compact experiments.
    return [f for f in features if f not in {"is_covid_period", "period_type"}]


def _threshold_grid(scores: np.ndarray) -> np.ndarray:
    base = np.arange(0.01, 0.991, 0.005)
    quantiles = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    return np.unique(np.clip(np.concatenate([base, quantiles]), 0.0, 1.0))


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in _threshold_grid(scores):
        f1 = f1_score(y_true, scores >= t, zero_division=0)
        if f1 > best_f1 or (math.isclose(f1, best_f1) and t > best_t):
            best_t = float(t)
            best_f1 = float(f1)
    return best_t, best_f1


def _metric_row(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    split: str,
) -> dict[str, float | int | str]:
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "split": split,
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "positive_share": float(np.mean(y_true)),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "alert_rows": int(pred.sum()),
        "alert_rate": float(np.mean(pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _fit_score(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    max_train_rows: int = MAX_TRAIN_ROWS,
    n_estimators: int = 500,
) -> tuple[dict[str, dict[str, float | int | str]], dict[str, str | int | float]]:
    train = df[df["time_split"].eq("train")].copy()
    val = df[df["time_split"].eq("validation")].copy()
    test = df[df["time_split"].eq("test")].copy()

    if len(train) > max_train_rows:
        train_fit = train.groupby(target, group_keys=False).sample(
            n=min(max_train_rows // 2, int(train[target].sum())),
            random_state=SEED,
        )
        neg_needed = max_train_rows - len(train_fit)
        train_fit = pd.concat(
            [
                train_fit,
                train[train[target].eq(0)].sample(n=neg_needed, random_state=SEED),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=SEED)
    else:
        train_fit = train

    x_train = train_fit[features]
    y_train = train_fit[target].astype(int).to_numpy()
    x_val = val[features]
    y_val = val[target].astype(int).to_numpy()
    x_test = test[features]
    y_test = test[target].astype(int).to_numpy()

    categorical = [c for c in features if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    numeric = [c for c in features if c not in categorical]

    encoder_kwargs = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=True)
        encoder_kwargs["sparse_output"] = True
    except TypeError:
        encoder_kwargs["sparse"] = True

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(**encoder_kwargs)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    scale_pos_weight = math.sqrt((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    start = time.time()
    x_train_tx = preprocessor.fit_transform(x_train)
    model.fit(x_train_tx, y_train)
    fit_seconds = time.time() - start

    val_scores = model.predict_proba(preprocessor.transform(x_val))[:, 1]
    test_scores = model.predict_proba(preprocessor.transform(x_test))[:, 1]
    threshold, val_f1 = _best_threshold(y_val, val_scores)

    rows = {
        "validation": _metric_row(y_val, val_scores, threshold, "validation"),
        "test": _metric_row(y_test, test_scores, threshold, "test"),
    }
    info = {
        "train_rows_before_sample": int(len(train)),
        "train_rows_fit": int(len(train_fit)),
        "max_train_rows": int(max_train_rows),
        "n_estimators": int(n_estimators),
        "feature_count": int(len(features)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "validation_selected_threshold": threshold,
        "validation_selected_f1": val_f1,
        "fit_seconds": round(fit_seconds, 3),
    }
    return rows, info


def run_ablation(df: pd.DataFrame, features_by_set: dict[str, list[str]]) -> pd.DataFrame:
    configs = [
        ("Core history + identifiers", _drop_unwanted(features_by_set["historical_only"])),
        ("Core + calendar", _drop_unwanted(features_by_set["historical_calendar_no_covid_period"])),
        ("Core + calendar + weather", _drop_unwanted(features_by_set["historical_calendar_weather"])),
        ("Core + calendar + weather + OSM", _drop_unwanted(features_by_set["historical_calendar_weather_osm"])),
        ("Core + calendar + weather + PLUTO", _drop_unwanted(features_by_set["historical_calendar_weather_pluto"])),
        ("Full context", _drop_unwanted(features_by_set["full_without_covid_period_features"])),
    ]
    out_rows: list[dict[str, float | int | str]] = []
    for label, features in configs:
        rows, info = _fit_score(df, features, "abnormal_increase_next_week")
        for split, metrics in rows.items():
            out_rows.append(
                {
                    "analysis": "feature_group_ablation",
                    "configuration": label,
                    **info,
                    **metrics,
                }
            )
    return pd.DataFrame(out_rows)


def run_sensitivity(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    configs = [
        ("4w + 1.5 sigma", "rolling_4w_mean", "rolling_4w_std", 1.5),
        ("8w + 1.0 sigma", "rolling_8w_mean", "rolling_8w_std", 1.0),
        ("8w + 1.5 sigma (reference)", "rolling_8w_mean", "rolling_8w_std", 1.5),
        ("8w + 2.0 sigma", "rolling_8w_mean", "rolling_8w_std", 2.0),
        ("12w + 1.5 sigma", "rolling_12w_mean", "rolling_12w_std", 1.5),
    ]
    out_rows: list[dict[str, float | int | str]] = []
    for label, mean_col, std_col, multiplier in configs:
        target = f"target_sensitivity_{label.replace(' ', '_').replace('+', '').replace('.', 'p').replace('(', '').replace(')', '')}"
        df[target] = (df["target_next_week_count"] > (df[mean_col] + multiplier * df[std_col])).astype(int)
        rows, info = _fit_score(
            df,
            features,
            target,
            max_train_rows=SENSITIVITY_MAX_TRAIN_ROWS,
            n_estimators=250,
        )
        for split, metrics in rows.items():
            out_rows.append(
                {
                    "analysis": "target_sensitivity",
                    "target_definition": label,
                    "window_column": mean_col.replace("rolling_", "").replace("_mean", ""),
                    "sigma_multiplier": multiplier,
                    **info,
                    **metrics,
                }
            )
    return pd.DataFrame(out_rows)


def reconcile_thresholds() -> pd.DataFrame:
    cat = pd.read_csv(PAPER_TABLES / "paper_table_04_category_performance.csv")
    thresholds = pd.read_csv(PAPER_TABLES / "paper_table_05_category_thresholds.csv")
    merged = cat.merge(thresholds[["complaint_category", "threshold"]], on="complaint_category", how="left")
    merged["recomputed_tp_from_alerts_precision"] = merged["predicted_positive_rows"] * merged["precision"]
    merged["recomputed_tp_from_positives_recall"] = merged["positive_rows"] * merged["recall"]
    merged["tp_rounding_error_alerts_precision"] = merged["recomputed_tp_from_alerts_precision"] - merged["tp"]
    merged["tp_rounding_error_positives_recall"] = merged["recomputed_tp_from_positives_recall"] - merged["tp"]
    return merged


def write_report(ablation: pd.DataFrame, sensitivity: pd.DataFrame, threshold_recon: pd.DataFrame) -> None:
    cm = pd.read_csv(PAPER_TABLES / "paper_table_03_final_model_confusion.csv")
    cm_test = cm[cm["split"].str.lower().eq("test")].iloc[0]
    group = pd.read_csv(SHAP_DIR / "shap_group_importance_paper_7groups.csv")
    global_imp = pd.read_csv(SHAP_DIR / "shap_global_importance.csv")

    test_ab = ablation[ablation["split"].eq("test")].copy()
    test_sens = sensitivity[sensitivity["split"].eq("test")].copy()
    lines = [
        "# Numerical Consistency Report",
        "",
        f"Generated by `code_model/run_paper_extended_checks.py` with seed {SEED}.",
        "",
        "## Overall Confusion Matrix",
        "",
        f"- Test positives: TP + FN = {int(cm_test['tp'])} + {int(cm_test['fn'])} = {int(cm_test['tp'] + cm_test['fn'])}.",
        f"- Test alerts: TP + FP = {int(cm_test['tp'])} + {int(cm_test['fp'])} = {int(cm_test['tp'] + cm_test['fp'])}.",
        f"- Precision from confusion matrix: {cm_test['tp'] / (cm_test['tp'] + cm_test['fp']):.6f}.",
        f"- Recall from confusion matrix: {cm_test['tp'] / (cm_test['tp'] + cm_test['fn']):.6f}.",
        f"- F1 from precision/recall: {2 * cm_test['tp'] / (2 * cm_test['tp'] + cm_test['fp'] + cm_test['fn']):.6f}.",
        f"- Alert rate: {(cm_test['tp'] + cm_test['fp']) / (cm_test['tp'] + cm_test['fp'] + cm_test['fn'] + cm_test['tn']):.6f}.",
        "",
        "## Category Reconciliation",
        "",
        f"- Category positive rows sum: {int(threshold_recon['positive_rows'].sum())}.",
        f"- Category alert rows sum: {int(threshold_recon['predicted_positive_rows'].sum())}.",
        f"- Category TP/FP/FN/TN sums: TP {int(threshold_recon['tp'].sum())}, FP {int(threshold_recon['fp'].sum())}, FN {int(threshold_recon['fn'].sum())}, TN {int(threshold_recon['tn'].sum())}.",
        f"- Maximum TP reconstruction error from precision: {threshold_recon['tp_rounding_error_alerts_precision'].abs().max():.4f}.",
        f"- Maximum TP reconstruction error from recall: {threshold_recon['tp_rounding_error_positives_recall'].abs().max():.4f}.",
        "",
        "## Ablation Protocol Checks",
        "",
        f"- Configurations: {', '.join(test_ab['configuration'].tolist())}.",
        f"- Every ablation uses train rows before sample = {int(test_ab['train_rows_before_sample'].iloc[0])}, max sampled training rows = {int(test_ab['max_train_rows'].iloc[0])}, chronological validation/test rows = {int(ablation[ablation['split'].eq('validation')]['rows'].iloc[0])}/{int(test_ab['rows'].iloc[0])}.",
        "- Thresholds are selected on validation; PR-AUC and ROC-AUC are computed from raw scores.",
        "",
        "## Sensitivity Target Prevalence",
        "",
        f"- Sensitivity runs use the same chronological split and validation-threshold policy, with max sampled training rows = {SENSITIVITY_MAX_TRAIN_ROWS} and 250 LightGBM trees for runtime control.",
    ]
    for _, row in test_sens.iterrows():
        lines.append(
            f"- {row['target_definition']}: test positive share {row['positive_share']:.4f}, "
            f"F1 {row['f1']:.4f}, PR-AUC {row['pr_auc']:.4f}, ROC-AUC {row['roc_auc']:.4f}."
        )
    lines.extend(
        [
            "",
            "## SHAP Checks",
            "",
            f"- SHAP group importance percentages sum to {group['percent'].sum():.2f}%.",
            f"- SHAP group feature counts sum to {int(group['feature_count'].sum())}.",
            f"- Top individual features include: {', '.join(global_imp.head(8)['feature'].tolist())}.",
            "- SHAP outputs explain the LightGBM component and are interpreted as predictive associations, not causal effects.",
            "",
        ]
    )
    (ROOT / "numerical_consistency_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features_by_set = _read_features()
    needed = sorted(
        set().union(
            *[
                set(_drop_unwanted(features_by_set[k]))
                for k in [
                    "historical_only",
                    "historical_calendar_no_covid_period",
                    "historical_calendar_weather",
                    "historical_calendar_weather_osm",
                    "historical_calendar_weather_pluto",
                    "full_without_covid_period_features",
                ]
            ],
            {
                "final_train_ready_flag",
                "time_split",
                "target_next_week_count",
                "abnormal_increase_next_week",
                "rolling_4w_mean",
                "rolling_4w_std",
                "rolling_8w_mean",
                "rolling_8w_std",
                "rolling_12w_mean",
                "rolling_12w_std",
            },
        )
    )
    df = pd.read_csv(DATASET, usecols=needed)
    df = df[df["final_train_ready_flag"].eq(1)].copy()
    df["abnormal_increase_next_week"] = df["abnormal_increase_next_week"].astype(int)

    ablation_path = OUT / "feature_group_ablation_compact.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        if "max_train_rows" not in ablation.columns:
            ablation["max_train_rows"] = MAX_TRAIN_ROWS
        if "n_estimators" not in ablation.columns:
            ablation["n_estimators"] = 500
    else:
        ablation = run_ablation(df.copy(), features_by_set)
        ablation.to_csv(ablation_path, index=False)

    sensitivity_features = _drop_unwanted(features_by_set["full_without_covid_period_features"])
    sensitivity = run_sensitivity(df.copy(), sensitivity_features)
    sensitivity.to_csv(OUT / "target_sensitivity_compact.csv", index=False)

    threshold_recon = reconcile_thresholds()
    threshold_recon.to_csv(OUT / "category_threshold_reconciliation.csv", index=False)

    write_report(ablation, sensitivity, threshold_recon)
    summary = {
        "status": "done",
        "seed": SEED,
        "max_train_rows": MAX_TRAIN_ROWS,
        "sensitivity_max_train_rows": SENSITIVITY_MAX_TRAIN_ROWS,
        "ablation_rows": int(len(ablation)),
        "sensitivity_rows": int(len(sensitivity)),
        "output_dir": str(OUT),
    }
    (OUT / "paper_extended_checks_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
