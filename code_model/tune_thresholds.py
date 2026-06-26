
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10.4 — Tune decision thresholds for trained tree/boosting models.

Project:
    SIMC NYC — Semantics-Aware Explainable Machine Learning for Urban Service
    Demand Forecasting in Smart Cities.

Purpose:
    Tune classification thresholds after model training.

Why this step matters:
    XGBoost can have strong ranking performance, e.g. good PR-AUC/ROC-AUC, while
    the default 0.5 threshold may produce too many false positives or too few
    positive predictions. For imbalanced abnormal-demand detection, threshold
    tuning must be performed on the validation set and then evaluated once on
    the test set.

Correct protocol:
    1. Train model on training period.
    2. Predict probabilities on validation period.
    3. Select threshold using validation only.
    4. Apply selected threshold to validation and test.
    5. Report tuned test metrics.

This script supports:
    - Global threshold tuning.
    - Several threshold-selection strategies:
        default_0_5
        best_f1
        best_f2
        best_balanced_accuracy
        best_precision_at_min_recall
        best_recall_at_min_precision
    - Category/year/period diagnostics for the selected threshold.
    - Chunked scoring to reduce RAM use.

Expected inputs:
    data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz
    data/processed/model_ready/model_config.json
    data/processed/model_results/tree_models_gpu_xgb_final_full_ablation/
        model_xgboost_E1_main_2015_2025_full_without_covid_period_features.joblib
        preprocessor_xgboost_E1_main_2015_2025_full_without_covid_period_features.joblib

Main outputs:
    data/processed/model_results/threshold_tuning/
        threshold_grid_validation.csv
        threshold_selection_summary.csv
        threshold_tuned_metrics.csv
        threshold_tuned_confusion_matrices.csv
        threshold_tuned_metrics_by_category.csv
        threshold_tuned_metrics_by_year.csv
        threshold_tuned_metrics_by_period.csv
        threshold_tuning_predictions_validation_test.csv.gz  [optional]
        threshold_tuning_run_summary.json

Recommended command:
    .\\.venv\\Scripts\\python.exe .\\code_model\\tune_thresholds.py --overwrite --progress ^
        --tree-output-dir data/processed/model_results/tree_models_gpu_xgb_final_full_ablation ^
        --experiment E1_main_2015_2025 ^
        --feature-set full_without_covid_period_features ^
        --model-name xgboost

Notes:
    - The script never tunes on test.
    - The main paper threshold should usually be best_f1 or a recall-constrained
      threshold depending on the operational objective.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


try:
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ModuleNotFoundError as exc:
    print(
        "\n[tune_thresholds] Missing required package.\n"
        "Install into project .venv using:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn joblib\n",
        file=sys.stderr,
    )
    raise exc


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SCRIPT_NAME = "tune_thresholds"

TARGET_COL = "abnormal_increase_next_week"
READY_COL = "final_train_ready_flag"
SPLIT_COL = "time_split"
YEAR_COL = "year"
WEEK_COL = "week_start"
TARGET_WEEK_COL = "target_week"
TARGET_YEAR_COL = "target_year"

DEFAULT_FINAL_REL = Path("data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz")
DEFAULT_CONFIG_REL = Path("data/processed/model_ready/model_config.json")
DEFAULT_TREE_OUTPUT_DIR_REL = Path("data/processed/model_results/tree_models_gpu_xgb_final_full_ablation")
DEFAULT_OUTPUT_DIR_REL = Path("data/processed/model_results/threshold_tuning")

DEFAULT_EXPERIMENT = "E1_main_2015_2025"
DEFAULT_FEATURE_SET = "full_without_covid_period_features"
DEFAULT_MODEL_NAME = "xgboost"

SPLITS_TO_SCORE = ["validation", "test"]

ID_OUTPUT_COLS = [
    "nta2020",
    "ntaname",
    "boroname",
    "week_start",
    "target_week",
    "target_year",
    "year",
    "period_type",
    "time_split",
    "complaint_category",
    "complaint_category_label",
    "complaint_count",
    TARGET_COL,
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune decision thresholds for trained NYC 311 abnormal-demand models."
    )

    parser.add_argument("--final-dataset", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tree-output-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--feature-set", type=str, default=DEFAULT_FEATURE_SET)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)

    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit joblib model path.")
    parser.add_argument("--preprocessor-path", type=str, default=None, help="Optional explicit preprocessor metadata path.")

    parser.add_argument("--chunksize", type=int, default=150_000, help="Rows per chunk for scoring.")
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--quantile-thresholds", type=int, default=200, help="Number of score quantile thresholds to add.")
    parser.add_argument("--min-recall", type=float, default=0.70, help="Constraint for best_precision_at_min_recall.")
    parser.add_argument("--min-precision", type=float, default=0.25, help="Constraint for best_recall_at_min_precision.")
    parser.add_argument("--primary-strategy", type=str, default="best_f1",
                        choices=[
                            "best_f1",
                            "best_f2",
                            "best_balanced_accuracy",
                            "best_precision_at_min_recall",
                            "best_recall_at_min_precision",
                            "default_0_5",
                        ])
    parser.add_argument("--save-predictions", action="store_true", help="Save validation/test y_score and selected predictions.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Path / I/O utilities
# ---------------------------------------------------------------------

def get_project_root() -> Path:
    # Expected location: SIMC_PROJECT/code_model/tune_thresholds.py
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


def write_json(path: Path, data: dict, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=to_jsonable)


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_model_paths(
    tree_output_dir: Path,
    experiment: str,
    feature_set: str,
    model_name: str,
    explicit_model_path: Optional[str],
    explicit_preprocessor_path: Optional[str],
) -> Tuple[Path, Path]:
    if explicit_model_path:
        model_path = Path(explicit_model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
    else:
        model_path = tree_output_dir / f"model_{model_name}_{experiment}_{feature_set}.joblib"

    if explicit_preprocessor_path:
        pre_path = Path(explicit_preprocessor_path)
        if not pre_path.is_absolute():
            pre_path = Path.cwd() / pre_path
    else:
        pre_path = tree_output_dir / f"preprocessor_{model_name}_{experiment}_{feature_set}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "You need to train with --save-models first."
        )
    if not pre_path.exists():
        raise FileNotFoundError(
            f"Preprocessor metadata file not found: {pre_path}\n"
            "You need to train with --save-models first."
        )

    return model_path, pre_path


# ---------------------------------------------------------------------
# Dataset / experiment utilities
# ---------------------------------------------------------------------

def normalize_flag(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype("int8")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype("int8")
    lower = s.astype(str).str.strip().str.lower()
    return lower.isin({"1", "true", "t", "yes", "y"}).astype("int8")


def infer_core_fields(df: pd.DataFrame) -> pd.DataFrame:
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

    if "period_type" not in df.columns and YEAR_COL in df.columns:
        y = pd.to_numeric(df[YEAR_COL], errors="coerce")
        period = pd.Series("unknown", index=df.index, dtype="object")
        period.loc[y.between(2015, 2019, inclusive="both")] = "pre_covid"
        period.loc[y.between(2020, 2021, inclusive="both")] = "covid_disruption"
        period.loc[y.between(2022, 2025, inclusive="both")] = "post_covid"
        df["period_type"] = period

    if READY_COL in df.columns:
        df[READY_COL] = normalize_flag(df[READY_COL])
    else:
        df[READY_COL] = 1

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
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


def required_columns_from_preprocessor(pre_meta: dict) -> List[str]:
    features = list(pre_meta.get("numeric_features", [])) + list(pre_meta.get("categorical_features", []))
    required = features + ID_OUTPUT_COLS + [READY_COL, SPLIT_COL, YEAR_COL]
    # preserve order unique
    seen = set()
    out = []
    for c in required:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def actual_usecols(final_path: Path, requested: List[str]) -> Tuple[List[str], List[str]]:
    header = pd.read_csv(final_path, nrows=0)
    available = set(header.columns)
    usecols = [c for c in requested if c in available]
    missing = [c for c in requested if c not in available]
    return usecols, missing


class MetadataPreprocessor:
    """
    Transform final dataset chunks using the metadata saved by train_tree_models.py.

    The saved metadata contains:
      numeric_features
      categorical_features
      numeric_medians
      category_maps
      feature_names
    """

    def __init__(self, metadata: dict):
        self.numeric_features = list(metadata.get("numeric_features", []))
        self.categorical_features = list(metadata.get("categorical_features", []))
        self.numeric_medians = dict(metadata.get("numeric_medians", {}))
        self.category_maps = dict(metadata.get("category_maps", {}))
        self.feature_names = list(metadata.get("feature_names", self.numeric_features + self.categorical_features))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.numeric_features:
            num = pd.DataFrame(index=df.index)
            for col in self.numeric_features:
                med = float(self.numeric_medians.get(col, 0.0))
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    num[col] = s.fillna(med).astype("float32")
                else:
                    num[col] = np.float32(med)
            parts.append(num)

        if self.categorical_features:
            cat = pd.DataFrame(index=df.index)
            for col in self.categorical_features:
                cmap = self.category_maps.get(col, {})
                if col in df.columns:
                    s = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
                    cat[col] = s.map(cmap).fillna(-1).astype("int16")
                else:
                    cat[col] = np.int16(-1)
            parts.append(cat)

        if not parts:
            return pd.DataFrame(index=df.index)

        out = pd.concat(parts, axis=1)
        return out[self.feature_names]


def predict_score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
        return np.asarray(score, dtype=float)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return np.asarray(score, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def score_validation_test_chunks(
    *,
    final_path: Path,
    usecols: List[str],
    model,
    preprocessor: MetadataPreprocessor,
    experiment: str,
    chunksize: int,
    progress: bool,
) -> pd.DataFrame:
    parts = []
    rows_seen = 0
    rows_kept = 0

    chunks = pd.read_csv(final_path, usecols=usecols, chunksize=chunksize, low_memory=False)

    for i, chunk in enumerate(chunks, start=1):
        rows_seen += len(chunk)
        chunk = infer_core_fields(chunk)

        keep_mask = pd.Series(False, index=chunk.index)
        for split in SPLITS_TO_SCORE:
            keep_mask |= experiment_mask(chunk, experiment, split)

        sub = chunk.loc[keep_mask].copy()
        if len(sub) == 0:
            if progress:
                print(f"[score] chunk={i:,}, rows_seen={rows_seen:,}, kept={rows_kept:,}")
            continue

        X = preprocessor.transform(sub)
        scores = predict_score(model, X)

        keep_cols = [c for c in ID_OUTPUT_COLS if c in sub.columns]
        out = sub[keep_cols].copy()
        out["y_true"] = pd.to_numeric(sub[TARGET_COL], errors="coerce").astype(int).values
        out["y_score"] = scores

        parts.append(out)
        rows_kept += len(out)

        if progress:
            print(
                f"[score] chunk={i:,}, rows_seen={rows_seen:,}, kept={rows_kept:,}, "
                f"last_scores_mean={np.nanmean(scores):.4f}",
                flush=True,
            )

    if not parts:
        raise RuntimeError("No validation/test rows were scored. Check experiment and split fields.")

    pred_df = pd.concat(parts, ignore_index=True)
    return pred_df


# ---------------------------------------------------------------------
# Metrics and threshold tuning
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


def evaluate_at_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float,
) -> Tuple[dict, dict]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_score_arr = np.nan_to_num(y_score_arr, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred = (y_score_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    metrics = {
        "rows": int(len(y_true_arr)),
        "positive_rows": int(y_true_arr.sum()),
        "positive_share": float(y_true_arr.mean()) if len(y_true_arr) else np.nan,
        "threshold": float(threshold),
        "predicted_positive_rows": int(y_pred.sum()),
        "predicted_positive_share": float(y_pred.mean()) if len(y_pred) else np.nan,
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": float(specificity) if not pd.isna(specificity) else np.nan,
        "npv": float(npv) if not pd.isna(npv) else np.nan,
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true_arr, y_pred, beta=2.0, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true_arr, y_score_arr),
        "pr_auc": safe_pr_auc(y_true_arr, y_score_arr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    conf = {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, conf


def build_threshold_candidates(
    scores: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    quantile_thresholds: int,
) -> np.ndarray:
    base = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    base = base[(base >= 0) & (base <= 1)]

    extra = np.array([0.5])

    if quantile_thresholds and quantile_thresholds > 0:
        qs = np.linspace(0.01, 0.99, quantile_thresholds)
        qvals = np.quantile(scores[np.isfinite(scores)], qs)
        extra = np.concatenate([extra, qvals])

    thresholds = np.unique(np.round(np.concatenate([base, extra]), 6))
    thresholds = thresholds[(thresholds >= 0) & (thresholds <= 1)]
    return thresholds


def build_threshold_grid(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m, _ = evaluate_at_threshold(y_true, y_score, float(t))
        rows.append(m)
    df = pd.DataFrame(rows)
    return df.sort_values("threshold").reset_index(drop=True)


def pick_thresholds(
    grid: pd.DataFrame,
    min_recall: float,
    min_precision: float,
) -> pd.DataFrame:
    rows = []

    def add_strategy(strategy: str, row: pd.Series, note: str):
        rows.append(
            {
                "strategy": strategy,
                "selected_threshold": float(row["threshold"]),
                "validation_precision": float(row["precision"]),
                "validation_recall": float(row["recall"]),
                "validation_f1": float(row["f1"]),
                "validation_f2": float(row["f2"]),
                "validation_balanced_accuracy": float(row["balanced_accuracy"]),
                "validation_pr_auc": float(row["pr_auc"]),
                "validation_roc_auc": float(row["roc_auc"]),
                "validation_predicted_positive_share": float(row["predicted_positive_share"]),
                "note": note,
            }
        )

    # default 0.5: choose closest if exact not present.
    idx_default = (grid["threshold"] - 0.5).abs().idxmin()
    add_strategy("default_0_5", grid.loc[idx_default], "Closest available threshold to 0.5.")

    idx_f1 = grid["f1"].idxmax()
    add_strategy("best_f1", grid.loc[idx_f1], "Maximizes F1 on validation.")

    idx_f2 = grid["f2"].idxmax()
    add_strategy("best_f2", grid.loc[idx_f2], "Maximizes F2 on validation; recall-oriented.")

    idx_ba = grid["balanced_accuracy"].idxmax()
    add_strategy("best_balanced_accuracy", grid.loc[idx_ba], "Maximizes balanced accuracy on validation.")

    feasible_recall = grid[grid["recall"] >= min_recall].copy()
    if len(feasible_recall):
        idx = feasible_recall.sort_values(
            ["precision", "f1", "balanced_accuracy"],
            ascending=[False, False, False],
        ).index[0]
        add_strategy(
            "best_precision_at_min_recall",
            grid.loc[idx],
            f"Maximizes precision subject to recall >= {min_recall}.",
        )
    else:
        add_strategy(
            "best_precision_at_min_recall",
            grid.loc[idx_f1],
            f"No threshold reached recall >= {min_recall}; fallback to best_f1.",
        )

    feasible_precision = grid[grid["precision"] >= min_precision].copy()
    if len(feasible_precision):
        idx = feasible_precision.sort_values(
            ["recall", "f1", "balanced_accuracy"],
            ascending=[False, False, False],
        ).index[0]
        add_strategy(
            "best_recall_at_min_precision",
            grid.loc[idx],
            f"Maximizes recall subject to precision >= {min_precision}.",
        )
    else:
        add_strategy(
            "best_recall_at_min_precision",
            grid.loc[idx_f1],
            f"No threshold reached precision >= {min_precision}; fallback to best_f1.",
        )

    out = pd.DataFrame(rows)
    return out


def metrics_for_selected_thresholds(
    pred_df: pd.DataFrame,
    selections: pd.DataFrame,
    experiment: str,
    model_name: str,
    feature_set: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    conf_rows = []

    for sel in selections.itertuples(index=False):
        strategy = sel.strategy
        threshold = float(sel.selected_threshold)

        for split in SPLITS_TO_SCORE:
            sub = pred_df[pred_df[SPLIT_COL] == split]
            if len(sub) == 0:
                continue

            m, c = evaluate_at_threshold(sub["y_true"].values, sub["y_score"].values, threshold)

            common = {
                "experiment_id": experiment,
                "model_name": model_name,
                "feature_set": feature_set,
                "split": split,
                "threshold_strategy": strategy,
            }

            metric_rows.append({**common, **m})
            conf_rows.append({**common, **c})

    return pd.DataFrame(metric_rows), pd.DataFrame(conf_rows)


def group_metrics(
    pred_df: pd.DataFrame,
    threshold: float,
    group_col: str,
    experiment: str,
    model_name: str,
    feature_set: str,
    strategy: str,
) -> pd.DataFrame:
    if group_col not in pred_df.columns:
        return pd.DataFrame()

    rows = []
    for (split, group_val), sub in pred_df.groupby([SPLIT_COL, group_col], dropna=False):
        if len(sub) == 0:
            continue
        # Need at least one class? Still compute hard metrics; AUC may become NaN.
        m, _ = evaluate_at_threshold(sub["y_true"].values, sub["y_score"].values, threshold)
        rows.append(
            {
                "experiment_id": experiment,
                "model_name": model_name,
                "feature_set": feature_set,
                "threshold_strategy": strategy,
                "split": split,
                group_col: group_val,
                **m,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    project_root = get_project_root()
    final_path = resolve_path(project_root, args.final_dataset, DEFAULT_FINAL_REL)
    config_path = resolve_path(project_root, args.config, DEFAULT_CONFIG_REL)
    tree_output_dir = resolve_path(project_root, args.tree_output_dir, DEFAULT_TREE_OUTPUT_DIR_REL)
    output_dir = resolve_path(project_root, args.output_dir, DEFAULT_OUTPUT_DIR_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not final_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {final_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found: {config_path}")

    config = load_json(config_path)
    if args.feature_set not in config["feature_sets"]:
        raise ValueError(f"Feature set not found in model_config.json: {args.feature_set}")

    model_path, preprocessor_path = find_model_paths(
        tree_output_dir=tree_output_dir,
        experiment=args.experiment,
        feature_set=args.feature_set,
        model_name=args.model_name,
        explicit_model_path=args.model_path,
        explicit_preprocessor_path=args.preprocessor_path,
    )

    if args.progress:
        print(f"[load] model: {model_path}")
        print(f"[load] preprocessor: {preprocessor_path}")

    model = joblib.load(model_path)
    pre_meta = joblib.load(preprocessor_path)
    preprocessor = MetadataPreprocessor(pre_meta)

    requested_cols = required_columns_from_preprocessor(pre_meta)
    usecols, missing_cols = actual_usecols(final_path, requested_cols)

    if args.progress:
        print(f"[data] usecols={len(usecols):,}, missing_requested={len(missing_cols):,}")
        if missing_cols:
            print(f"[data] missing requested columns: {missing_cols}")

    pred_df = score_validation_test_chunks(
        final_path=final_path,
        usecols=usecols,
        model=model,
        preprocessor=preprocessor,
        experiment=args.experiment,
        chunksize=args.chunksize,
        progress=args.progress,
    )

    if args.progress:
        print(f"[score] total scored validation/test rows={len(pred_df):,}")
        print(pred_df.groupby(SPLIT_COL)["y_true"].agg(["count", "mean"]).to_string())

    val = pred_df[pred_df[SPLIT_COL] == "validation"].copy()
    test = pred_df[pred_df[SPLIT_COL] == "test"].copy()
    if len(val) == 0:
        raise RuntimeError("No validation rows found for threshold tuning.")
    if len(test) == 0:
        print("[warning] No test rows found. Threshold tuning will only report validation.", file=sys.stderr)

    y_val = val["y_true"].values.astype(int)
    s_val = val["y_score"].values.astype(float)

    thresholds = build_threshold_candidates(
        scores=s_val,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        quantile_thresholds=args.quantile_thresholds,
    )

    if args.progress:
        print(f"[tune] threshold candidates={len(thresholds):,}")

    grid = build_threshold_grid(y_val, s_val, thresholds)
    selections = pick_thresholds(
        grid=grid,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
    )

    tuned_metrics, tuned_conf = metrics_for_selected_thresholds(
        pred_df=pred_df,
        selections=selections,
        experiment=args.experiment,
        model_name=args.model_name,
        feature_set=args.feature_set,
    )

    # Primary strategy diagnostics by category/year/period.
    primary_row = selections[selections["strategy"] == args.primary_strategy]
    if len(primary_row) == 0:
        raise ValueError(f"Primary strategy not found: {args.primary_strategy}")
    primary_threshold = float(primary_row.iloc[0]["selected_threshold"])

    by_category = group_metrics(
        pred_df=pred_df,
        threshold=primary_threshold,
        group_col="complaint_category",
        experiment=args.experiment,
        model_name=args.model_name,
        feature_set=args.feature_set,
        strategy=args.primary_strategy,
    )
    by_year = group_metrics(
        pred_df=pred_df,
        threshold=primary_threshold,
        group_col="year",
        experiment=args.experiment,
        model_name=args.model_name,
        feature_set=args.feature_set,
        strategy=args.primary_strategy,
    )
    by_period = group_metrics(
        pred_df=pred_df,
        threshold=primary_threshold,
        group_col="period_type",
        experiment=args.experiment,
        model_name=args.model_name,
        feature_set=args.feature_set,
        strategy=args.primary_strategy,
    )

    # Add selected predictions for primary strategy and optional constrained strategies.
    pred_df["y_pred_primary"] = (pred_df["y_score"].values >= primary_threshold).astype(int)
    pred_df["primary_threshold"] = primary_threshold
    pred_df["primary_strategy"] = args.primary_strategy

    # Write outputs.
    write_csv(output_dir / "threshold_grid_validation.csv", grid, args.overwrite)
    write_csv(output_dir / "threshold_selection_summary.csv", selections, args.overwrite)
    write_csv(output_dir / "threshold_tuned_metrics.csv", tuned_metrics, args.overwrite)
    write_csv(output_dir / "threshold_tuned_confusion_matrices.csv", tuned_conf, args.overwrite)
    write_csv(output_dir / "threshold_tuned_metrics_by_category.csv", by_category, args.overwrite)
    write_csv(output_dir / "threshold_tuned_metrics_by_year.csv", by_year, args.overwrite)
    write_csv(output_dir / "threshold_tuned_metrics_by_period.csv", by_period, args.overwrite)

    if args.save_predictions:
        pred_path = output_dir / "threshold_tuning_predictions_validation_test.csv.gz"
        ensure_output_path(pred_path, args.overwrite)
        pred_df.to_csv(pred_path, index=False, compression="gzip")

    # Compact summary.
    top_grid = grid.sort_values(["f1", "balanced_accuracy", "precision"], ascending=False).head(10)

    def metrics_record(strategy: str, split: str) -> Optional[dict]:
        sub = tuned_metrics[(tuned_metrics["threshold_strategy"] == strategy) & (tuned_metrics["split"] == split)]
        if len(sub) == 0:
            return None
        return sub.iloc[0].to_dict()

    summary = {
        "script": SCRIPT_NAME,
        "status": "done",
        "final_dataset": str(final_path),
        "config": str(config_path),
        "tree_output_dir": str(tree_output_dir),
        "output_dir": str(output_dir),
        "experiment": args.experiment,
        "feature_set": args.feature_set,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "scored_rows": int(len(pred_df)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "validation_positive_share": float(val["y_true"].mean()) if len(val) else None,
        "test_positive_share": float(test["y_true"].mean()) if len(test) else None,
        "threshold_candidates": int(len(thresholds)),
        "threshold_min": float(args.threshold_min),
        "threshold_max": float(args.threshold_max),
        "threshold_step": float(args.threshold_step),
        "min_recall_constraint": float(args.min_recall),
        "min_precision_constraint": float(args.min_precision),
        "primary_strategy": args.primary_strategy,
        "primary_threshold": float(primary_threshold),
        "primary_validation_metrics": metrics_record(args.primary_strategy, "validation"),
        "primary_test_metrics": metrics_record(args.primary_strategy, "test"),
        "default_0_5_validation_metrics": metrics_record("default_0_5", "validation"),
        "default_0_5_test_metrics": metrics_record("default_0_5", "test"),
        "selection_summary": selections.to_dict(orient="records"),
        "top_validation_f1_thresholds": top_grid.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - t0, 3),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }

    write_json(output_dir / "threshold_tuning_run_summary.json", summary, args.overwrite)

    if args.progress:
        print("\n[threshold selection]")
        print(selections.to_string(index=False))
        print("\n[tuned metrics]")
        cols = [
            "threshold_strategy",
            "split",
            "threshold",
            "precision",
            "recall",
            "f1",
            "f2",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "predicted_positive_share",
        ]
        print(tuned_metrics[cols].to_string(index=False))

    print("=" * 80)
    print(f"[{SCRIPT_NAME}] DONE")
    print(f"Selection summary: {output_dir / 'threshold_selection_summary.csv'}")
    print(f"Tuned metrics: {output_dir / 'threshold_tuned_metrics.csv'}")
    print(f"Run summary: {output_dir / 'threshold_tuning_run_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        raise
