
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10.2 — Train baseline models for NYC 311 abnormal service-demand forecasting.

Project:
    SIMC NYC — Semantics-Aware Explainable Machine Learning for Urban Service
    Demand Forecasting in Smart Cities.

Purpose:
    Train and evaluate baseline models before moving to tree/boosting models.

Baselines included:
    1. majority_class
       Always predicts the majority class from the training split.

    2. train_prevalence_probability
       Uses the training positive-class prevalence as a constant probability score.
       The hard class prediction is still the majority class.

    3. current_week_abnormal_persistence
       Predicts next-week abnormal increase if the current week is already above
       its leakage-safe historical threshold:
           complaint_count > rolling_8w_mean + threshold_multiplier * rolling_8w_std
       This is a non-ML operational baseline.

    4. recent_momentum_above_baseline
       Predicts abnormal increase if current demand is above a short-term baseline
       and increasing:
           complaint_count > rolling_4w_mean and diff_1w_count > 0

    5. logistic_regression
       A leakage-safe statistical ML baseline using selected feature sets from
       model_config.json.

Important leakage rule:
    The script does NOT use:
      - target_next_week_count
      - abnormal_threshold_8w
      - abnormal_increase_next_week as a feature
      - week_start as a feature
      - NTA ID as a main feature
    Row t predicts abnormal increase at t+1.

Expected input:
    data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz
    data/processed/model_ready/model_config.json

Main outputs:
    data/processed/model_results/baselines/baseline_metrics.csv
    data/processed/model_results/baselines/baseline_confusion_matrices.csv
    data/processed/model_results/baselines/baseline_run_summary.json
    data/processed/model_results/baselines/logistic_model_<experiment>_<feature_set>.joblib
    data/processed/model_results/baselines/logistic_feature_names_<experiment>_<feature_set>.csv

Run from project root:
    .\\.venv\\Scripts\\python.exe .\\code_model\\train_baselines.py --overwrite --progress

Recommended first run on laptop:
    .\\.venv\\Scripts\\python.exe .\\code_model\\train_baselines.py --overwrite --progress ^
        --experiments E1_main_2015_2025 ^
        --feature-sets historical_only historical_calendar_weather

Optional COVID checks:
    .\\.venv\\Scripts\\python.exe .\\code_model\\train_baselines.py --overwrite --progress ^
        --experiments E1_main_2015_2025 E2_without_covid_years E3_covid_period_feature_ablation ^
        --feature-sets historical_only historical_calendar_weather full_without_covid_period_features

Notes:
    - This script trains Logistic Regression as the baseline ML model.
    - Heavy tree/boosting models should be handled in Step 10.3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Optional sklearn imports with clear error message
# ---------------------------------------------------------------------

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    print(
        "\n[train_baselines] Missing required package.\n"
        "Install into project .venv using:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn joblib\n",
        file=sys.stderr,
    )
    raise exc


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SCRIPT_NAME = "train_baselines"

TARGET_COL = "abnormal_increase_next_week"
READY_COL = "final_train_ready_flag"
SPLIT_COL = "time_split"
YEAR_COL = "year"
WEEK_COL = "week_start"
TARGET_WEEK_COL = "target_week"
TARGET_YEAR_COL = "target_year"

DEFAULT_FINAL_REL = Path("data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz")
DEFAULT_CONFIG_REL = Path("data/processed/model_ready/model_config.json")
DEFAULT_OUTPUT_DIR_REL = Path("data/processed/model_results/baselines")

DEFAULT_EXPERIMENTS = ["E1_main_2015_2025"]
DEFAULT_FEATURE_SETS = ["historical_only", "historical_calendar_weather"]

BASELINE_MODEL_NAMES = [
    "majority_class",
    "train_prevalence_probability",
    "current_week_abnormal_persistence",
    "recent_momentum_above_baseline",
]

SPLITS = ["train", "validation", "test"]

ID_OUTPUT_COLS = [
    "nta2020",
    "week_start",
    "target_week",
    "target_year",
    "year",
    "period_type",
    "time_split",
    "complaint_category",
    "complaint_count",
    TARGET_COL,
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline models for NYC 311 abnormal service-demand forecasting."
    )

    parser.add_argument(
        "--final-dataset",
        type=str,
        default=None,
        help=f"Path to final dataset. Default: {DEFAULT_FINAL_REL}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to model_config.json from Step 10.1. Default: {DEFAULT_CONFIG_REL}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR_REL}",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help=(
            "Experiments to run. Examples: "
            "E1_main_2015_2025 E2_without_covid_years E3_covid_period_feature_ablation"
        ),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=DEFAULT_FEATURE_SETS,
        help="Feature sets to use for Logistic Regression. Must exist in model_config.json.",
    )
    parser.add_argument(
        "--skip-logistic",
        action="store_true",
        help="Only run non-ML operational baselines.",
    )
    parser.add_argument(
        "--only-logistic",
        action="store_true",
        help="Skip non-ML baselines and run Logistic Regression only.",
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=1.5,
        help="Threshold multiplier for current_week_abnormal_persistence baseline.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=250,
        help="LogisticRegression max_iter.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="LogisticRegression inverse regularization strength C.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Class weight for Logistic Regression.",
    )
    parser.add_argument(
        "--sample-train",
        type=int,
        default=0,
        help=(
            "Optional training sample size for quick debugging. "
            "0 means use all training rows."
        ),
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=42,
        help="Random state for optional train sampling.",
    )
    parser.add_argument(
        "--save-logistic-models",
        action="store_true",
        help="Save fitted Logistic Regression pipelines as joblib files.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=(
            "Save validation/test predictions for Logistic Regression. "
            "Can create large files."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Path and JSON utilities
# ---------------------------------------------------------------------

def get_project_root() -> Path:
    # Expected location: SIMC_PROJECT/code_model/train_baselines.py
    return Path(__file__).resolve().parents[1]


def resolve_path(project_root: Path, maybe_path: Optional[str], default_rel: Path) -> Path:
    if maybe_path:
        p = Path(maybe_path)
        return p if p.is_absolute() else project_root / p
    return project_root / default_rel


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")


def write_json(path: Path, data: dict, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=to_jsonable)


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if pd.isna(x):
        return None
    return x


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found: {config_path}\n"
            "Run Step 10.1 first:\n"
            "  .\\.venv\\Scripts\\python.exe .\\code_model\\inspect_final_dataset.py --overwrite"
        )
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------

def unique_preserve_order(cols: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def get_required_columns(config: dict, feature_sets: List[str]) -> List[str]:
    feature_set_map = config["feature_sets"]

    feature_cols: List[str] = []
    for fs in feature_sets:
        if fs not in feature_set_map:
            available = sorted(feature_set_map.keys())
            raise ValueError(
                f"Feature set not found in model_config.json: {fs}\n"
                f"Available feature sets: {available}"
            )
        feature_cols.extend(feature_set_map[fs]["features"])

    required = (
        feature_cols
        + ID_OUTPUT_COLS
        + [
            READY_COL,
            SPLIT_COL,
            YEAR_COL,
            "rolling_8w_mean",
            "rolling_8w_std",
            "rolling_4w_mean",
            "diff_1w_count",
        ]
    )
    return unique_preserve_order(required)


def load_dataset(final_path: Path, usecols: List[str], progress: bool = False) -> pd.DataFrame:
    if not final_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {final_path}")

    # Read header first to avoid failing if a requested optional column is absent.
    header = pd.read_csv(final_path, nrows=0)
    available = set(header.columns)
    actual_usecols = [c for c in usecols if c in available]
    missing = [c for c in usecols if c not in available]

    if progress:
        print(f"[load] final dataset: {final_path}")
        print(f"[load] requested columns={len(usecols):,}, available used={len(actual_usecols):,}")
        if missing:
            print(f"[load] missing optional/requested columns ignored: {missing}")

    df = pd.read_csv(final_path, usecols=actual_usecols, low_memory=False)

    if progress:
        print(f"[load] loaded rows={len(df):,}, columns={len(df.columns):,}")

    return df


def normalize_flag(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype("int8")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype("int8")
    lower = s.astype(str).str.strip().str.lower()
    return lower.isin({"1", "true", "t", "yes", "y"}).astype("int8")


def add_missing_core_fields_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if YEAR_COL not in df.columns and WEEK_COL in df.columns:
        df[YEAR_COL] = pd.to_datetime(df[WEEK_COL], errors="coerce").dt.year
    if TARGET_WEEK_COL not in df.columns and WEEK_COL in df.columns:
        df[TARGET_WEEK_COL] = pd.to_datetime(df[WEEK_COL], errors="coerce") + pd.Timedelta(days=7)
    if TARGET_YEAR_COL not in df.columns and TARGET_WEEK_COL in df.columns:
        df[TARGET_YEAR_COL] = pd.to_datetime(df[TARGET_WEEK_COL], errors="coerce").dt.year

    if SPLIT_COL not in df.columns:
        y = pd.to_numeric(df[TARGET_YEAR_COL] if TARGET_YEAR_COL in df.columns else df[YEAR_COL], errors="coerce")
        split = pd.Series("unused", index=df.index, dtype="object")
        split.loc[y <= 2022] = "train"
        split.loc[y == 2023] = "validation"
        split.loc[y >= 2024] = "test"
        df[SPLIT_COL] = split

    if "period_type" not in df.columns:
        y = pd.to_numeric(df[YEAR_COL], errors="coerce")
        period = pd.Series("unknown", index=df.index, dtype="object")
        period.loc[y.between(2015, 2019, inclusive="both")] = "pre_covid"
        period.loc[y.between(2020, 2021, inclusive="both")] = "covid_disruption"
        period.loc[y.between(2022, 2025, inclusive="both")] = "post_covid"
        df["period_type"] = period

    return df


def clean_for_modeling(df: pd.DataFrame, all_feature_cols: List[str]) -> pd.DataFrame:
    df = add_missing_core_fields_if_needed(df)

    if READY_COL in df.columns:
        df[READY_COL] = normalize_flag(df[READY_COL])
    else:
        df[READY_COL] = 1

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Replace infinities from ratio features.
    for col in all_feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Make categorical columns explicit strings with missing placeholder later
    # handled by SimpleImputer.
    for col in ["complaint_category", "boroname", "period_type", "poi_dominant_semantic_category"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

    return df


def experiment_mask(df: pd.DataFrame, experiment_id: str, split: str) -> pd.Series:
    ready = df[READY_COL].astype(int) == 1
    target_ok = df[TARGET_COL].notna()
    y = pd.to_numeric(df[TARGET_YEAR_COL] if TARGET_YEAR_COL in df.columns else df[YEAR_COL], errors="coerce")
    time_split = df[SPLIT_COL].astype(str)

    if experiment_id == "E1_main_2015_2025":
        return ready & target_ok & (time_split == split)

    if experiment_id == "E2_without_covid_years":
        no_covid = ~y.isin([2020, 2021])
        return ready & target_ok & no_covid & (time_split == split)

    if experiment_id == "E3_covid_period_feature_ablation":
        return ready & target_ok & (time_split == split)

    if experiment_id == "E4_pre_to_post_generalization":
        if split == "train":
            return ready & target_ok & (y <= 2019)
        if split == "validation":
            return ready & target_ok & y.isin([2022, 2023])
        if split == "test":
            return ready & target_ok & (y >= 2024)
        return pd.Series(False, index=df.index)

    raise ValueError(f"Unknown experiment_id: {experiment_id}")


def get_split_data(
    df: pd.DataFrame,
    experiment_id: str,
    split: str,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mask = experiment_mask(df, experiment_id, split)
    y = df.loc[mask, TARGET_COL].astype(int)

    if feature_cols is None:
        return df.loc[mask], y, mask

    X = df.loc[mask, feature_cols].copy()
    return X, y, mask


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def safe_pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, score))
    except Exception:
        return np.nan


def evaluate_predictions(
    *,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Optional[Iterable[float]],
    experiment_id: str,
    split: str,
    model_name: str,
    feature_set: str,
    train_rows: int,
    threshold: float,
) -> Tuple[dict, dict]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if y_score is None:
        y_score_arr = y_pred_arr.astype(float)
    else:
        y_score_arr = np.asarray(y_score, dtype=float)

    # Ensure finite scores for metric functions.
    y_score_arr = np.nan_to_num(y_score_arr, nan=0.0, posinf=1.0, neginf=0.0)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    metrics = {
        "experiment_id": experiment_id,
        "split": split,
        "model_name": model_name,
        "feature_set": feature_set,
        "rows": int(len(y_true_arr)),
        "train_rows": int(train_rows),
        "positive_rows": int(y_true_arr.sum()),
        "positive_share": float(y_true_arr.mean()) if len(y_true_arr) else np.nan,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "specificity": float(specificity) if not pd.isna(specificity) else np.nan,
        "npv": float(npv) if not pd.isna(npv) else np.nan,
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true_arr, y_score_arr),
        "pr_auc": safe_pr_auc(y_true_arr, y_score_arr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    conf = {
        "experiment_id": experiment_id,
        "split": split,
        "model_name": model_name,
        "feature_set": feature_set,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, conf


# ---------------------------------------------------------------------
# Non-ML baselines
# ---------------------------------------------------------------------

def predict_majority_class(y_train: pd.Series, n: int) -> Tuple[np.ndarray, np.ndarray, int, float]:
    prevalence = float(y_train.mean())
    majority_class = 1 if prevalence >= 0.5 else 0
    y_pred = np.full(n, majority_class, dtype=int)
    y_score = np.full(n, prevalence, dtype=float)
    return y_pred, y_score, majority_class, prevalence


def predict_current_week_abnormal(df_split: pd.DataFrame, multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
    required = ["complaint_count", "rolling_8w_mean", "rolling_8w_std"]
    for c in required:
        if c not in df_split.columns:
            raise ValueError(f"Required column missing for current_week_abnormal_persistence: {c}")

    count = pd.to_numeric(df_split["complaint_count"], errors="coerce")
    mean = pd.to_numeric(df_split["rolling_8w_mean"], errors="coerce")
    std = pd.to_numeric(df_split["rolling_8w_std"], errors="coerce").fillna(0)
    threshold = mean + multiplier * std

    score = count - threshold
    pred = (score > 0).fillna(False).astype(int).to_numpy()
    # Convert margin to monotonic bounded score for AUC/AP.
    score_arr = score.replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_numpy(dtype=float)
    return pred, score_arr


def predict_recent_momentum(df_split: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    required = ["complaint_count", "rolling_4w_mean", "diff_1w_count"]
    for c in required:
        if c not in df_split.columns:
            raise ValueError(f"Required column missing for recent_momentum_above_baseline: {c}")

    count = pd.to_numeric(df_split["complaint_count"], errors="coerce")
    mean4 = pd.to_numeric(df_split["rolling_4w_mean"], errors="coerce")
    diff1 = pd.to_numeric(df_split["diff_1w_count"], errors="coerce")

    margin = (count - mean4).fillna(-1e9)
    pred = ((count > mean4) & (diff1 > 0)).fillna(False).astype(int).to_numpy()
    score = (margin + diff1.fillna(0)).replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_numpy(dtype=float)
    return pred, score


def run_non_ml_baselines(
    df: pd.DataFrame,
    experiments: List[str],
    multiplier: float,
    progress: bool,
) -> Tuple[List[dict], List[dict]]:
    metrics_rows: List[dict] = []
    conf_rows: List[dict] = []

    for exp_id in experiments:
        _, y_train, train_mask = get_split_data(df, exp_id, "train", feature_cols=None)
        train_rows = int(len(y_train))

        if train_rows == 0:
            print(f"[warning] no train rows for experiment {exp_id}; skipping non-ML baselines")
            continue

        if progress:
            print(f"[baseline] experiment={exp_id}, train_rows={train_rows:,}, pos_share={y_train.mean():.4f}")

        for split in SPLITS:
            df_split, y_true, _ = get_split_data(df, exp_id, split, feature_cols=None)
            if len(y_true) == 0:
                continue

            # Majority class
            y_pred, y_score, majority_class, prevalence = predict_majority_class(y_train, len(y_true))
            m, c = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                experiment_id=exp_id,
                split=split,
                model_name="majority_class",
                feature_set="none",
                train_rows=train_rows,
                threshold=0.5,
            )
            m["majority_class"] = int(majority_class)
            m["train_positive_prevalence"] = float(prevalence)
            metrics_rows.append(m)
            conf_rows.append(c)

            # Constant probability baseline; same hard prediction but explicitly named.
            y_pred, y_score, majority_class, prevalence = predict_majority_class(y_train, len(y_true))
            m, c = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                experiment_id=exp_id,
                split=split,
                model_name="train_prevalence_probability",
                feature_set="none",
                train_rows=train_rows,
                threshold=prevalence,
            )
            m["majority_class"] = int(majority_class)
            m["train_positive_prevalence"] = float(prevalence)
            metrics_rows.append(m)
            conf_rows.append(c)

            # Operational persistence baseline
            y_pred, y_score = predict_current_week_abnormal(df_split, multiplier)
            m, c = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                experiment_id=exp_id,
                split=split,
                model_name="current_week_abnormal_persistence",
                feature_set="historical_rule",
                train_rows=train_rows,
                threshold=0.0,
            )
            m["threshold_multiplier"] = float(multiplier)
            metrics_rows.append(m)
            conf_rows.append(c)

            # Operational momentum baseline
            y_pred, y_score = predict_recent_momentum(df_split)
            m, c = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                experiment_id=exp_id,
                split=split,
                model_name="recent_momentum_above_baseline",
                feature_set="historical_rule",
                train_rows=train_rows,
                threshold=0.0,
            )
            metrics_rows.append(m)
            conf_rows.append(c)

    return metrics_rows, conf_rows


# ---------------------------------------------------------------------
# Logistic Regression baseline
# ---------------------------------------------------------------------

def make_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    # sklearn version compatibility for sparse_output.
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
    )


def make_logistic_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    max_iter: int,
    c_value: float,
    class_weight: str,
    random_state: int,
) -> Pipeline:
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    cw = None if class_weight == "none" else "balanced"

    clf = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="saga",
        max_iter=max_iter,
        class_weight=cw,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )


def get_feature_names_from_pipeline(pipe: Pipeline) -> List[str]:
    try:
        pre = pipe.named_steps["preprocess"]
        names = pre.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return []


def split_feature_types(
    features: List[str],
    config: dict,
    df_columns: Iterable[str],
) -> Tuple[List[str], List[str]]:
    df_cols = set(df_columns)
    categorical_all = set(config.get("categorical_features", []))
    present_features = [f for f in features if f in df_cols]
    categorical = [f for f in present_features if f in categorical_all]
    numeric = [f for f in present_features if f not in categorical]
    return numeric, categorical


def sample_training_if_requested(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_train: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if not sample_train or sample_train <= 0 or sample_train >= len(X_train):
        return X_train, y_train

    # Stratified-ish sampling: sample within each class where possible.
    tmp = pd.DataFrame({"y": y_train.values}, index=X_train.index)
    frac = sample_train / len(tmp)

    sampled_idx = (
        tmp.groupby("y", group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=random_state))
        .index
    )
    if len(sampled_idx) > sample_train:
        sampled_idx = pd.Index(sampled_idx).to_series().sample(n=sample_train, random_state=random_state).values

    return X_train.loc[sampled_idx], y_train.loc[sampled_idx]


def run_logistic_models(
    df: pd.DataFrame,
    config: dict,
    experiments: List[str],
    feature_sets: List[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[List[dict], List[dict], List[dict]]:
    metrics_rows: List[dict] = []
    conf_rows: List[dict] = []
    model_rows: List[dict] = []

    feature_set_map = config["feature_sets"]

    for exp_id in experiments:
        for fs in feature_sets:
            if fs not in feature_set_map:
                raise ValueError(f"Feature set not found: {fs}")

            features = feature_set_map[fs]["features"]
            numeric_features, categorical_features = split_feature_types(features, config, df.columns)
            used_features = numeric_features + categorical_features

            if args.progress:
                print(
                    f"[logistic] experiment={exp_id}, feature_set={fs}, "
                    f"features={len(used_features):,}, numeric={len(numeric_features):,}, "
                    f"categorical={len(categorical_features):,}"
                )

            X_train, y_train, _ = get_split_data(df, exp_id, "train", used_features)
            if len(y_train) == 0:
                print(f"[warning] no train rows for {exp_id}/{fs}; skipping logistic")
                continue

            X_train, y_train = sample_training_if_requested(
                X_train,
                y_train,
                args.sample_train,
                args.sample_random_state,
            )

            pipe = make_logistic_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                max_iter=args.max_iter,
                c_value=args.c,
                class_weight=args.class_weight,
                random_state=args.sample_random_state,
            )

            t_fit = time.time()
            pipe.fit(X_train, y_train)
            fit_seconds = time.time() - t_fit

            if args.progress:
                print(
                    f"[logistic] fitted {exp_id}/{fs} in {fit_seconds:.1f}s, "
                    f"train_rows={len(y_train):,}"
                )

            feature_names = get_feature_names_from_pipeline(pipe)
            if feature_names:
                feature_name_df = pd.DataFrame({"transformed_feature": feature_names})
                feature_name_path = output_dir / f"logistic_feature_names_{exp_id}_{fs}.csv"
                write_csv(feature_name_path, feature_name_df, args.overwrite)
            else:
                feature_name_path = None

            model_path = None
            if args.save_logistic_models:
                model_path = output_dir / f"logistic_model_{exp_id}_{fs}.joblib"
                ensure_output_path(model_path, args.overwrite)
                joblib.dump(pipe, model_path)

            # Evaluate on train/validation/test.
            for split in SPLITS:
                X_split, y_true, mask = get_split_data(df, exp_id, split, used_features)
                if len(y_true) == 0:
                    continue

                y_pred = pipe.predict(X_split)
                try:
                    y_score = pipe.predict_proba(X_split)[:, 1]
                except Exception:
                    if hasattr(pipe.named_steps["model"], "decision_function"):
                        y_score = pipe.decision_function(X_split)
                    else:
                        y_score = y_pred.astype(float)

                m, c = evaluate_predictions(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_score=y_score,
                    experiment_id=exp_id,
                    split=split,
                    model_name="logistic_regression",
                    feature_set=fs,
                    train_rows=len(y_train),
                    threshold=0.5,
                )
                m["fit_seconds"] = float(fit_seconds)
                m["selected_feature_count"] = int(len(used_features))
                m["numeric_feature_count"] = int(len(numeric_features))
                m["categorical_feature_count"] = int(len(categorical_features))
                m["class_weight"] = args.class_weight
                m["max_iter"] = int(args.max_iter)
                m["C"] = float(args.c)
                m["sample_train"] = int(args.sample_train)

                metrics_rows.append(m)
                conf_rows.append(c)

                if args.save_predictions and split in {"validation", "test"}:
                    pred_cols = [c for c in ID_OUTPUT_COLS if c in df.columns]
                    pred_df = df.loc[mask, pred_cols].copy()
                    pred_df["y_true"] = y_true.values
                    pred_df["y_pred"] = y_pred
                    pred_df["y_score"] = y_score
                    pred_path = output_dir / f"logistic_predictions_{exp_id}_{fs}_{split}.csv.gz"
                    ensure_output_path(pred_path, args.overwrite)
                    pred_df.to_csv(pred_path, index=False, compression="gzip")

            model_rows.append(
                {
                    "experiment_id": exp_id,
                    "feature_set": fs,
                    "model_name": "logistic_regression",
                    "train_rows": int(len(y_train)),
                    "selected_feature_count": int(len(used_features)),
                    "numeric_feature_count": int(len(numeric_features)),
                    "categorical_feature_count": int(len(categorical_features)),
                    "fit_seconds": float(fit_seconds),
                    "model_path": str(model_path) if model_path else "",
                    "feature_names_path": str(feature_name_path) if feature_name_path else "",
                }
            )

    return metrics_rows, conf_rows, model_rows


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def build_best_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    val = metrics_df[metrics_df["split"] == "validation"].copy()
    if val.empty:
        return pd.DataFrame()

    # Sort with PR-AUC and F1 as main criteria for imbalanced classification.
    sort_cols = ["experiment_id", "pr_auc", "f1", "balanced_accuracy"]
    val = val.sort_values(sort_cols, ascending=[True, False, False, False])
    best = val.groupby("experiment_id", as_index=False).head(10)
    return best


def make_run_summary(
    *,
    args: argparse.Namespace,
    final_path: Path,
    config_path: Path,
    output_dir: Path,
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    model_info_df: pd.DataFrame,
    elapsed_seconds: float,
) -> dict:
    summary = {
        "script": SCRIPT_NAME,
        "status": "done",
        "final_dataset": str(final_path),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "rows_loaded": int(len(df)),
        "experiments": args.experiments,
        "feature_sets": args.feature_sets,
        "skip_logistic": bool(args.skip_logistic),
        "only_logistic": bool(args.only_logistic),
        "threshold_multiplier": float(args.threshold_multiplier),
        "logistic": {
            "max_iter": int(args.max_iter),
            "C": float(args.c),
            "class_weight": args.class_weight,
            "sample_train": int(args.sample_train),
            "save_logistic_models": bool(args.save_logistic_models),
        },
        "metrics_rows": int(len(metrics_df)),
        "confusion_matrix_rows": int(len(conf_df)),
        "model_info_rows": int(len(model_info_df)),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_minutes": round(elapsed_seconds / 60, 3),
    }

    if not metrics_df.empty:
        # Main validation ranking by PR-AUC/F1.
        val = metrics_df[metrics_df["split"] == "validation"].copy()
        if not val.empty:
            top = val.sort_values(["pr_auc", "f1", "balanced_accuracy"], ascending=False).head(10)
            summary["top_validation_models"] = top[
                [
                    "experiment_id",
                    "model_name",
                    "feature_set",
                    "rows",
                    "positive_share",
                    "precision",
                    "recall",
                    "f1",
                    "balanced_accuracy",
                    "roc_auc",
                    "pr_auc",
                ]
            ].to_dict(orient="records")

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    if args.skip_logistic and args.only_logistic:
        raise ValueError("Use either --skip-logistic or --only-logistic, not both.")

    project_root = get_project_root()
    final_path = resolve_path(project_root, args.final_dataset, DEFAULT_FINAL_REL)
    config_path = resolve_path(project_root, args.config, DEFAULT_CONFIG_REL)
    output_dir = resolve_path(project_root, args.output_dir, DEFAULT_OUTPUT_DIR_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    # Validate feature sets and experiments early.
    feature_sets_available = set(config["feature_sets"].keys())
    missing_feature_sets = [fs for fs in args.feature_sets if fs not in feature_sets_available]
    if missing_feature_sets:
        raise ValueError(
            f"Unknown feature sets: {missing_feature_sets}\n"
            f"Available: {sorted(feature_sets_available)}"
        )

    experiments_available = set(config.get("experiment_ids", []))
    missing_experiments = [e for e in args.experiments if e not in experiments_available]
    if missing_experiments:
        raise ValueError(
            f"Unknown experiments: {missing_experiments}\n"
            f"Available: {sorted(experiments_available)}"
        )

    required_cols = get_required_columns(config, args.feature_sets)
    df = load_dataset(final_path, required_cols, progress=args.progress)

    all_feature_cols = unique_preserve_order(
        c
        for fs in args.feature_sets
        for c in config["feature_sets"][fs]["features"]
        if c in df.columns
    )
    df = clean_for_modeling(df, all_feature_cols)

    if args.progress:
        print(f"[data] rows={len(df):,}, columns={len(df.columns):,}")
        print(f"[data] target positive share={df[TARGET_COL].mean(skipna=True):.4f}")

    metrics_rows: List[dict] = []
    conf_rows: List[dict] = []
    model_rows: List[dict] = []

    if not args.only_logistic:
        m_rows, c_rows = run_non_ml_baselines(
            df=df,
            experiments=args.experiments,
            multiplier=args.threshold_multiplier,
            progress=args.progress,
        )
        metrics_rows.extend(m_rows)
        conf_rows.extend(c_rows)

    if not args.skip_logistic:
        m_rows, c_rows, model_info = run_logistic_models(
            df=df,
            config=config,
            experiments=args.experiments,
            feature_sets=args.feature_sets,
            args=args,
            output_dir=output_dir,
        )
        metrics_rows.extend(m_rows)
        conf_rows.extend(c_rows)
        model_rows.extend(model_info)

    metrics_df = pd.DataFrame(metrics_rows)
    conf_df = pd.DataFrame(conf_rows)
    model_info_df = pd.DataFrame(model_rows)

    # Sort output for readability.
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["experiment_id", "model_name", "feature_set", "split"]
        ).reset_index(drop=True)

    if not conf_df.empty:
        conf_df = conf_df.sort_values(
            ["experiment_id", "model_name", "feature_set", "split"]
        ).reset_index(drop=True)

    best_df = build_best_summary(metrics_df)

    write_csv(output_dir / "baseline_metrics.csv", metrics_df, args.overwrite)
    write_csv(output_dir / "baseline_confusion_matrices.csv", conf_df, args.overwrite)
    write_csv(output_dir / "baseline_validation_ranking_top10.csv", best_df, args.overwrite)
    write_csv(output_dir / "baseline_model_info.csv", model_info_df, args.overwrite)

    summary = make_run_summary(
        args=args,
        final_path=final_path,
        config_path=config_path,
        output_dir=output_dir,
        df=df,
        metrics_df=metrics_df,
        conf_df=conf_df,
        model_info_df=model_info_df,
        elapsed_seconds=time.time() - t0,
    )
    write_json(output_dir / "baseline_run_summary.json", summary, args.overwrite)

    if args.progress:
        print("\n[summary] Validation ranking:")
        if not best_df.empty:
            print(
                best_df[
                    [
                        "experiment_id",
                        "model_name",
                        "feature_set",
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                        "roc_auc",
                        "pr_auc",
                    ]
                ].head(20).to_string(index=False)
            )
        else:
            print("(empty)")

    print("=" * 80)
    print(f"[{SCRIPT_NAME}] DONE")
    print(f"Metrics: {output_dir / 'baseline_metrics.csv'}")
    print(f"Confusion matrices: {output_dir / 'baseline_confusion_matrices.csv'}")
    print(f"Run summary: {output_dir / 'baseline_run_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        raise
