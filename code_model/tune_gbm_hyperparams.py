
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10.5A — Hyperparameter + class-weight tuning for LightGBM/XGBoost.

Project:
    SIMC NYC — Semantics-Aware Explainable Machine Learning for Urban Service
    Demand Forecasting in Smart Cities.

Purpose:
    Improve model performance after Step 10.4 threshold tuning by searching:
      - LightGBM/XGBoost hyperparameters
      - positive-class weighting intensity
      - validation-based decision threshold

Correct protocol:
    - Train on train period only.
    - Tune model/threshold using validation period only.
    - Report test period once for selected candidates.
    - Never tune on test.

Default recommended run:
    .\\.venv\\Scripts\\python.exe .\\code_model\\tune_gbm_hyperparams.py --overwrite --progress ^
        --experiments E1_main_2015_2025 ^
        --feature-set full_without_covid_period_features ^
        --models lightgbm xgboost ^
        --preset quick ^
        --pos-weight-modes none sqrt half balanced ^
        --use-gpu ^
        --output-dir data/processed/model_results/gbm_hyperparam_tuning_quick

More thorough run:
    .\\.venv\\Scripts\\python.exe .\\code_model\\tune_gbm_hyperparams.py --overwrite --progress ^
        --experiments E1_main_2015_2025 ^
        --feature-set full_without_covid_period_features ^
        --models lightgbm xgboost ^
        --preset balanced ^
        --pos-weight-modes none p2 p3 sqrt half balanced ^
        --use-gpu ^
        --output-dir data/processed/model_results/gbm_hyperparam_tuning_balanced

Outputs:
    gbm_tuning_ranking.csv
    gbm_tuning_threshold_metrics.csv
    gbm_tuning_confusion_matrices.csv
    gbm_tuning_candidate_info.csv
    gbm_tuning_feature_importance_top.csv
    gbm_tuning_errors.csv
    gbm_tuning_skipped_models.csv
    gbm_tuning_run_summary.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
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
        "\n[tune_gbm_hyperparams] Missing required package.\n"
        "Install into project .venv:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn joblib\n",
        file=sys.stderr,
    )
    raise exc

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


SCRIPT_NAME = "tune_gbm_hyperparams"

TARGET_COL = "abnormal_increase_next_week"
READY_COL = "final_train_ready_flag"
SPLIT_COL = "time_split"
YEAR_COL = "year"
WEEK_COL = "week_start"
TARGET_WEEK_COL = "target_week"
TARGET_YEAR_COL = "target_year"

DEFAULT_FINAL_REL = Path("data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz")
DEFAULT_CONFIG_REL = Path("data/processed/model_ready/model_config.json")
DEFAULT_OUTPUT_DIR_REL = Path("data/processed/model_results/gbm_hyperparam_tuning")

DEFAULT_EXPERIMENTS = ["E1_main_2015_2025"]
DEFAULT_FEATURE_SET = "full_without_covid_period_features"
DEFAULT_MODELS = ["lightgbm", "xgboost"]

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
        description="Tune LightGBM/XGBoost hyperparameters, class weights, and thresholds."
    )

    parser.add_argument("--final-dataset", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--feature-set", type=str, default=DEFAULT_FEATURE_SET)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Allowed: lightgbm xgboost hist_gradient_boosting or all.")
    parser.add_argument("--preset", type=str, default="quick", choices=["quick", "balanced", "wide"],
                        help="Controls size of hyperparameter grid.")

    parser.add_argument("--pos-weight-modes", nargs="+", default=["none", "sqrt", "half", "balanced"],
                        help=(
                            "Positive-class weight modes: none, sqrt, half, balanced, "
                            "or p2/p3/p5/numeric values like 2, 3.5."
                        ))

    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="Optional stratified training sample per candidate. 0 means full train.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for LightGBM/XGBoost. Sklearn HGB remains CPU.")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--xgboost-gpu-mode", type=str, default="modern", choices=["modern", "legacy"])
    parser.add_argument("--lightgbm-gpu-platform-id", type=int, default=-1)

    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--quantile-thresholds", type=int, default=200)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-precision", type=float, default=0.25)
    parser.add_argument("--primary-strategy", type=str, default="best_f1",
                        choices=[
                            "best_f1",
                            "best_f2",
                            "best_balanced_accuracy",
                            "best_precision_at_min_recall",
                            "best_recall_at_min_precision",
                        ])

    parser.add_argument("--save-models", action="store_true",
                        help="Save every fitted candidate model. Can use disk space.")
    parser.add_argument("--save-predictions-for-best", action="store_true",
                        help="Save validation/test predictions for the best validation-ranked candidate.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Path / I/O utilities
# ---------------------------------------------------------------------

def get_project_root() -> Path:
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


# ---------------------------------------------------------------------
# Dataset and preprocessing
# ---------------------------------------------------------------------

def unique_preserve_order(cols: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


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
        return ready & target_ok & (~y.isin([2020, 2021])) & (time_split == split)

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

    raise ValueError(f"Unknown experiment: {experiment_id}")


def load_dataset(final_path: Path, config: dict, feature_set: str, progress: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if feature_set not in config["feature_sets"]:
        raise ValueError(f"Feature set not found in model_config.json: {feature_set}")

    features = list(config["feature_sets"][feature_set]["features"])
    categorical_all = set(config.get("categorical_features", []))
    categorical_features = [c for c in features if c in categorical_all]
    numeric_features = [c for c in features if c not in categorical_all]

    requested = unique_preserve_order(features + ID_OUTPUT_COLS + [READY_COL, SPLIT_COL, YEAR_COL])

    header = pd.read_csv(final_path, nrows=0)
    available = set(header.columns)
    usecols = [c for c in requested if c in available]
    missing = [c for c in requested if c not in available]

    if progress:
        print(f"[load] final dataset: {final_path}")
        print(f"[load] requested columns={len(requested):,}, usecols={len(usecols):,}")
        if missing:
            print(f"[load] missing columns ignored: {missing}")

    df = pd.read_csv(final_path, usecols=usecols, low_memory=False)
    df = infer_core_fields(df)

    present_features = [c for c in features if c in df.columns]
    categorical_features = [c for c in present_features if c in categorical_all]
    numeric_features = [c for c in present_features if c not in categorical_all]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in categorical_features:
        df[col] = df[col].astype("object")

    if progress:
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(
            f"[load] rows={len(df):,}, columns={len(df.columns):,}, "
            f"features={len(present_features):,}, numeric={len(numeric_features):,}, "
            f"categorical={len(categorical_features):,}, memory={mem_mb:,.1f} MB"
        )

    return df, numeric_features, categorical_features


class FastTabularPreprocessor:
    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.numeric_medians_: Dict[str, float] = {}
        self.category_levels_: Dict[str, List[str]] = {}
        self.one_hot_feature_names_: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = list(self.numeric_features)

    def fit(self, df: pd.DataFrame) -> "FastTabularPreprocessor":
        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = s.median()
            if pd.isna(med):
                med = 0.0
            self.numeric_medians_[col] = float(med)

        for col in self.categorical_features:
            s = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
            levels = sorted(s.unique().tolist())
            self.category_levels_[col] = levels
            self.one_hot_feature_names_[col] = [f"{col}={level}" for level in levels]

        self.feature_names_ = (
            list(self.numeric_features)
            + [name for col in self.categorical_features for name in self.one_hot_feature_names_.get(col, [])]
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.numeric_features:
            numeric_data = {}
            for col in self.numeric_features:
                med = self.numeric_medians_.get(col, 0.0)
                numeric_data[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(med)
                    .astype("float32")
                    .to_numpy()
                )
            parts.append(pd.DataFrame(numeric_data, index=df.index))

        if self.categorical_features:
            cat_parts = []
            for col in self.categorical_features:
                levels = self.category_levels_.get(col, [])
                names = self.one_hot_feature_names_.get(col, [])
                s = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
                if not levels:
                    continue
                dummies = pd.get_dummies(s, dtype="int8")
                dummies = dummies.reindex(columns=levels, fill_value=0)
                dummies.columns = names
                cat_parts.append(dummies)
            if cat_parts:
                parts.append(pd.concat(cat_parts, axis=1))

        if not parts:
            return pd.DataFrame(index=df.index)

        out = pd.concat(parts, axis=1)
        return out[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def to_metadata(self) -> dict:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "numeric_medians": self.numeric_medians_,
            "category_levels": self.category_levels_,
            "one_hot_feature_names": self.one_hot_feature_names_,
            "categorical_encoding": "train_fitted_one_hot",
            "feature_names": self.feature_names_,
        }


def stratified_sample_indices(y: pd.Series, max_rows: int, random_state: int) -> pd.Index:
    if not max_rows or max_rows <= 0 or max_rows >= len(y):
        return y.index

    tmp = pd.DataFrame({"y": y.astype(int)}, index=y.index)
    sampled = []
    for cls, g in tmp.groupby("y"):
        n_take = max(1, int(round(max_rows * len(g) / len(tmp))))
        n_take = min(n_take, len(g))
        sampled.append(g.sample(n=n_take, random_state=random_state))
    out = pd.concat(sampled).index

    if len(out) > max_rows:
        out = pd.Index(pd.Series(out).sample(n=max_rows, random_state=random_state).values)
    return out


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


def evaluate_at_threshold(y_true: Iterable[int], y_score: Iterable[float], threshold: float) -> Tuple[dict, dict]:
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
    conf = {"threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics, conf


def build_threshold_candidates(scores: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    base = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    base = base[(base >= 0) & (base <= 1)]
    extra = np.array([0.5])

    if args.quantile_thresholds and args.quantile_thresholds > 0:
        qs = np.linspace(0.01, 0.99, args.quantile_thresholds)
        valid = scores[np.isfinite(scores)]
        if len(valid):
            extra = np.concatenate([extra, np.quantile(valid, qs)])

    thresholds = np.unique(np.round(np.concatenate([base, extra]), 6))
    thresholds = thresholds[(thresholds >= 0) & (thresholds <= 1)]
    return thresholds


def threshold_grid(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m, _ = evaluate_at_threshold(y_true, y_score, float(t))
        rows.append(m)
    return pd.DataFrame(rows)


def pick_thresholds(grid: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []

    def add(strategy: str, row: pd.Series, note: str):
        rows.append({
            "threshold_strategy": strategy,
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
        })

    add("default_0_5", grid.loc[(grid["threshold"] - 0.5).abs().idxmin()], "Closest to 0.5.")
    add("best_f1", grid.loc[grid["f1"].idxmax()], "Max validation F1.")
    add("best_f2", grid.loc[grid["f2"].idxmax()], "Max validation F2.")
    add("best_balanced_accuracy", grid.loc[grid["balanced_accuracy"].idxmax()], "Max validation balanced accuracy.")

    feasible = grid[grid["recall"] >= args.min_recall]
    if len(feasible):
        idx = feasible.sort_values(["precision", "f1", "balanced_accuracy"], ascending=[False, False, False]).index[0]
        add("best_precision_at_min_recall", grid.loc[idx], f"Max precision with recall >= {args.min_recall}.")
    else:
        add("best_precision_at_min_recall", grid.loc[grid["f1"].idxmax()], "Constraint infeasible; fallback best_f1.")

    feasible = grid[grid["precision"] >= args.min_precision]
    if len(feasible):
        idx = feasible.sort_values(["recall", "f1", "balanced_accuracy"], ascending=[False, False, False]).index[0]
        add("best_recall_at_min_precision", grid.loc[idx], f"Max recall with precision >= {args.min_precision}.")
    else:
        add("best_recall_at_min_precision", grid.loc[grid["f1"].idxmax()], "Constraint infeasible; fallback best_f1.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------

def normalize_models(models: List[str]) -> List[str]:
    aliases = {
        "lgbm": "lightgbm",
        "light_gbm": "lightgbm",
        "xgb": "xgboost",
        "hgb": "hist_gradient_boosting",
        "all": "all",
    }
    out = []
    for m in models:
        m = aliases.get(m.lower(), m.lower())
        if m == "all":
            out.extend(["lightgbm", "xgboost", "hist_gradient_boosting"])
        else:
            out.append(m)
    valid = {"lightgbm", "xgboost", "hist_gradient_boosting"}
    bad = [m for m in out if m not in valid]
    if bad:
        raise ValueError(f"Unknown models: {bad}. Valid: {sorted(valid)}")
    return unique_preserve_order(out)


def resolve_pos_weight(mode: str, y_train: pd.Series) -> Tuple[str, float]:
    mode_raw = str(mode).strip().lower()
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    balanced = (neg / pos) if pos else 1.0

    if mode_raw in {"none", "1", "1.0", "unweighted"}:
        return "none", 1.0
    if mode_raw == "balanced":
        return "balanced", float(balanced)
    if mode_raw == "sqrt":
        return "sqrt", float(math.sqrt(balanced))
    if mode_raw == "half":
        return "half", float(max(1.0, balanced * 0.5))
    if mode_raw.startswith("p"):
        val = float(mode_raw[1:])
        return mode_raw, float(val)

    val = float(mode_raw)
    return f"p{val:g}", float(val)


def base_param_grid(model_name: str, preset: str) -> List[dict]:
    if model_name == "lightgbm":
        if preset == "quick":
            return [
                {
                    "param_id": "lgbm_current_depth8_leaf63_lr005",
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "max_depth": 8,
                    "min_child_samples": 50,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                },
                {
                    "param_id": "lgbm_regularized_depth6_leaf31_lr003",
                    "n_estimators": 800,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "max_depth": 6,
                    "min_child_samples": 100,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 3.0,
                },
            ]
        if preset == "balanced":
            return [
                {
                    "param_id": "lgbm_current_depth8_leaf63_lr005",
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "max_depth": 8,
                    "min_child_samples": 50,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                },
                {
                    "param_id": "lgbm_regularized_depth6_leaf31_lr003",
                    "n_estimators": 800,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "max_depth": 6,
                    "min_child_samples": 100,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 3.0,
                },
                {
                    "param_id": "lgbm_deeper_depth10_leaf127_lr003",
                    "n_estimators": 800,
                    "learning_rate": 0.03,
                    "num_leaves": 127,
                    "max_depth": 10,
                    "min_child_samples": 80,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 2.0,
                },
            ]
        return [
            {
                "param_id": "lgbm_current_depth8_leaf63_lr005",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
            {
                "param_id": "lgbm_regularized_depth6_leaf31_lr003",
                "n_estimators": 800,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 100,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
            },
            {
                "param_id": "lgbm_deeper_depth10_leaf127_lr003",
                "n_estimators": 1000,
                "learning_rate": 0.03,
                "num_leaves": 127,
                "max_depth": 10,
                "min_child_samples": 80,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
            },
            {
                "param_id": "lgbm_small_leaf15_lr003",
                "n_estimators": 900,
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 5,
                "min_child_samples": 150,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 5.0,
            },
        ]

    if model_name == "xgboost":
        if preset == "quick":
            return [
                {
                    "param_id": "xgb_current_depth8_lr005",
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "max_depth": 8,
                    "min_child_weight": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                },
                {
                    "param_id": "xgb_regularized_depth6_lr003",
                    "n_estimators": 800,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "min_child_weight": 50,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 3.0,
                },
            ]
        if preset == "balanced":
            return [
                {
                    "param_id": "xgb_current_depth8_lr005",
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "max_depth": 8,
                    "min_child_weight": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                },
                {
                    "param_id": "xgb_regularized_depth6_lr003",
                    "n_estimators": 800,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "min_child_weight": 50,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 3.0,
                },
                {
                    "param_id": "xgb_shallow_depth5_lr003",
                    "n_estimators": 900,
                    "learning_rate": 0.03,
                    "max_depth": 5,
                    "min_child_weight": 80,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_lambda": 5.0,
                },
            ]
        return [
            {
                "param_id": "xgb_current_depth8_lr005",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 8,
                "min_child_weight": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
            {
                "param_id": "xgb_regularized_depth6_lr003",
                "n_estimators": 800,
                "learning_rate": 0.03,
                "max_depth": 6,
                "min_child_weight": 50,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
            },
            {
                "param_id": "xgb_shallow_depth5_lr003",
                "n_estimators": 900,
                "learning_rate": 0.03,
                "max_depth": 5,
                "min_child_weight": 80,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 5.0,
            },
            {
                "param_id": "xgb_deep_depth10_lr003",
                "n_estimators": 700,
                "learning_rate": 0.03,
                "max_depth": 10,
                "min_child_weight": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
            },
        ]

    if model_name == "hist_gradient_boosting":
        if preset == "quick":
            return [
                {"param_id": "hgb_iter300_leaf31_lr005", "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 0.1},
            ]
        return [
            {"param_id": "hgb_iter300_leaf31_lr005", "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 0.1},
            {"param_id": "hgb_iter500_leaf31_lr003", "max_iter": 500, "learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 0.3},
            {"param_id": "hgb_iter500_leaf63_lr003", "max_iter": 500, "learning_rate": 0.03, "max_leaf_nodes": 63, "l2_regularization": 0.3},
        ]

    raise ValueError(f"Unknown model: {model_name}")


def available_models(models: List[str]) -> Tuple[List[str], List[dict]]:
    out = []
    skipped = []
    for m in models:
        if m == "lightgbm" and not HAS_LIGHTGBM:
            skipped.append({"model_name": m, "reason": "lightgbm_not_installed", "install": ".\\.venv\\Scripts\\python.exe -m pip install lightgbm"})
            continue
        if m == "xgboost" and not HAS_XGBOOST:
            skipped.append({"model_name": m, "reason": "xgboost_not_installed", "install": ".\\.venv\\Scripts\\python.exe -m pip install xgboost"})
            continue
        out.append(m)
    return out, skipped


def make_model(model_name: str, params: dict, pos_weight: float, args: argparse.Namespace):
    if model_name == "lightgbm":
        model_params = dict(
            objective="binary",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            max_depth=int(params["max_depth"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            class_weight={0: 1.0, 1: float(pos_weight)},
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=-1,
        )
        if args.use_gpu:
            model_params["device_type"] = "gpu"
            model_params["gpu_device_id"] = int(args.gpu_device_id)
            if int(args.lightgbm_gpu_platform_id) >= 0:
                model_params["gpu_platform_id"] = int(args.lightgbm_gpu_platform_id)
        return LGBMClassifier(**model_params)

    if model_name == "xgboost":
        model_params = dict(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            scale_pos_weight=float(pos_weight),
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        if args.use_gpu:
            if args.xgboost_gpu_mode == "legacy":
                model_params["tree_method"] = "gpu_hist"
                model_params["gpu_id"] = int(args.gpu_device_id)
            else:
                model_params["tree_method"] = "hist"
                model_params["device"] = f"cuda:{int(args.gpu_device_id)}"
        else:
            model_params["tree_method"] = "hist"
        return XGBClassifier(**model_params)

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=int(params["max_iter"]),
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            early_stopping=True,
            validation_fraction=0.1,
            random_state=args.random_state,
        )

    raise ValueError(f"Unknown model: {model_name}")


def sample_weight_for_hgb(y: pd.Series, pos_weight: float) -> np.ndarray:
    y_arr = np.asarray(y, dtype=int)
    w = np.ones(len(y_arr), dtype="float32")
    w[y_arr == 1] = float(pos_weight)
    return w


def predict_score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def feature_importance(model, model_name: str, feature_names: List[str], top_n: int = 50) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    try:
        imp = np.asarray(model.feature_importances_, dtype=float)
    except Exception:
        return pd.DataFrame()

    n = min(len(imp), len(feature_names))
    df = pd.DataFrame({
        "feature": feature_names[:n],
        "importance": imp[:n],
    })
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    df["importance_rank"] = np.arange(1, len(df) + 1)
    df["model_name"] = model_name
    return df


# ---------------------------------------------------------------------
# Candidate run
# ---------------------------------------------------------------------

def get_split(df: pd.DataFrame, experiment_id: str, split: str, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mask = experiment_mask(df, experiment_id, split)
    return df.loc[mask, feature_cols].copy(), df.loc[mask, TARGET_COL].astype(int), mask


def run_candidate(
    *,
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    experiment_id: str,
    model_name: str,
    params: dict,
    pos_weight_mode_input: str,
    args: argparse.Namespace,
    output_dir: Path,
    candidate_index: int,
) -> Tuple[List[dict], List[dict], dict, pd.DataFrame, Optional[pd.DataFrame]]:
    feature_cols = numeric_features + categorical_features

    X_train_raw, y_train, train_mask = get_split(df, experiment_id, "train", feature_cols)
    if len(y_train) == 0:
        raise ValueError(f"No train rows for {experiment_id}")

    train_rows_before_sample = len(y_train)
    sampled_idx = stratified_sample_indices(y_train, args.max_train_rows, args.random_state)
    X_train_raw = X_train_raw.loc[sampled_idx]
    y_train = y_train.loc[sampled_idx]

    pos_weight_label, pos_weight = resolve_pos_weight(pos_weight_mode_input, y_train)

    candidate_id = f"{candidate_index:03d}_{experiment_id}_{model_name}_{params['param_id']}_pw_{pos_weight_label}"
    candidate_id = candidate_id.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

    if args.progress:
        print(
            f"[candidate] {candidate_id}\n"
            f"  train_rows={len(y_train):,}/{train_rows_before_sample:,}, "
            f"pos_share={y_train.mean():.4f}, pos_weight={pos_weight:.4f}"
        )

    pre = FastTabularPreprocessor(numeric_features, categorical_features)

    t_pre = time.time()
    X_train = pre.fit_transform(X_train_raw)
    preprocess_seconds = time.time() - t_pre
    del X_train_raw
    gc.collect()

    model = make_model(model_name, params, pos_weight, args)

    fit_seconds = np.nan
    warnings_preview = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t_fit = time.time()
        if model_name == "hist_gradient_boosting":
            model.fit(X_train, y_train, sample_weight=sample_weight_for_hgb(y_train, pos_weight))
        else:
            model.fit(X_train, y_train)
        fit_seconds = time.time() - t_fit
        warnings_preview = " | ".join(str(w.message) for w in caught[:5])

    # Score validation and test.
    pred_parts = []
    split_rows = {}
    for split in ["validation", "test"]:
        X_raw, y_true, mask = get_split(df, experiment_id, split, feature_cols)
        X_split = pre.transform(X_raw)
        y_score = predict_score(model, X_split)

        keep_cols = [c for c in ID_OUTPUT_COLS if c in df.columns]
        pred = df.loc[mask, keep_cols].copy()
        pred["split"] = split
        pred["y_true"] = y_true.values.astype(int)
        pred["y_score"] = y_score
        pred_parts.append(pred)
        split_rows[split] = len(pred)

        del X_raw, X_split, y_true
        gc.collect()

    pred_df = pd.concat(pred_parts, ignore_index=True)

    val = pred_df[pred_df["split"] == "validation"]
    y_val = val["y_true"].values.astype(int)
    s_val = val["y_score"].values.astype(float)

    thresholds = build_threshold_candidates(s_val, args)
    grid = threshold_grid(y_val, s_val, thresholds)
    selections = pick_thresholds(grid, args)

    metric_rows = []
    conf_rows = []

    for sel in selections.itertuples(index=False):
        strategy = sel.threshold_strategy
        threshold = float(sel.selected_threshold)

        for split in ["validation", "test"]:
            sub = pred_df[pred_df["split"] == split]
            if len(sub) == 0:
                continue
            m, c = evaluate_at_threshold(sub["y_true"].values, sub["y_score"].values, threshold)
            common = {
                "candidate_id": candidate_id,
                "experiment_id": experiment_id,
                "model_name": model_name,
                "param_id": params["param_id"],
                "feature_set": args.feature_set,
                "pos_weight_mode": pos_weight_label,
                "pos_weight": float(pos_weight),
                "split": split,
                "threshold_strategy": strategy,
            }
            metric_rows.append({**common, **m})
            conf_rows.append({**common, **c})

    primary_sel = selections[selections["threshold_strategy"] == args.primary_strategy].iloc[0]
    val_primary = [r for r in metric_rows if r["split"] == "validation" and r["threshold_strategy"] == args.primary_strategy][0]
    test_primary = [r for r in metric_rows if r["split"] == "test" and r["threshold_strategy"] == args.primary_strategy][0]

    info = {
        "candidate_id": candidate_id,
        "experiment_id": experiment_id,
        "model_name": model_name,
        "param_id": params["param_id"],
        "feature_set": args.feature_set,
        "pos_weight_mode": pos_weight_label,
        "pos_weight": float(pos_weight),
        "train_rows": int(len(y_train)),
        "train_rows_before_sample": int(train_rows_before_sample),
        "validation_rows": int(split_rows.get("validation", 0)),
        "test_rows": int(split_rows.get("test", 0)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "selected_feature_count": int(len(feature_cols)),
        "primary_strategy": args.primary_strategy,
        "selected_threshold": float(primary_sel["selected_threshold"]),
        "validation_f1": float(val_primary["f1"]),
        "validation_f2": float(val_primary["f2"]),
        "validation_precision": float(val_primary["precision"]),
        "validation_recall": float(val_primary["recall"]),
        "validation_balanced_accuracy": float(val_primary["balanced_accuracy"]),
        "validation_pr_auc": float(val_primary["pr_auc"]),
        "validation_roc_auc": float(val_primary["roc_auc"]),
        "validation_predicted_positive_share": float(val_primary["predicted_positive_share"]),
        "test_f1": float(test_primary["f1"]),
        "test_f2": float(test_primary["f2"]),
        "test_precision": float(test_primary["precision"]),
        "test_recall": float(test_primary["recall"]),
        "test_balanced_accuracy": float(test_primary["balanced_accuracy"]),
        "test_pr_auc": float(test_primary["pr_auc"]),
        "test_roc_auc": float(test_primary["roc_auc"]),
        "test_predicted_positive_share": float(test_primary["predicted_positive_share"]),
        "preprocess_seconds": float(preprocess_seconds),
        "fit_seconds": float(fit_seconds),
        "warnings_count": int(len(caught)),
        "warnings_preview": warnings_preview,
        "params_json": json.dumps(params, ensure_ascii=False),
    }

    imp = feature_importance(model, model_name, pre.feature_names_, top_n=50)
    if not imp.empty:
        imp.insert(0, "candidate_id", candidate_id)
        imp.insert(1, "experiment_id", experiment_id)
        imp.insert(2, "param_id", params["param_id"])
        imp.insert(3, "pos_weight_mode", pos_weight_label)

    # Optional model saving.
    if args.save_models:
        model_path = output_dir / "models" / f"model_{candidate_id}.joblib"
        pre_path = output_dir / "models" / f"preprocessor_{candidate_id}.joblib"
        ensure_output_path(model_path, args.overwrite)
        ensure_output_path(pre_path, args.overwrite)
        joblib.dump(model, model_path)
        joblib.dump(pre.to_metadata(), pre_path)
        info["model_path"] = str(model_path)
        info["preprocessor_path"] = str(pre_path)

    # Save predictions only later for best candidate. Return pred_df to caller optionally only if requested? To save memory, not.
    pred_for_possible_best = pred_df if args.save_predictions_for_best else None

    del X_train, y_train, model
    gc.collect()

    return metric_rows, conf_rows, info, imp, pred_for_possible_best


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    project_root = get_project_root()
    final_path = resolve_path(project_root, args.final_dataset, DEFAULT_FINAL_REL)
    config_path = resolve_path(project_root, args.config, DEFAULT_CONFIG_REL)
    output_dir = resolve_path(project_root, args.output_dir, DEFAULT_OUTPUT_DIR_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not final_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {final_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found: {config_path}")

    config = load_json(config_path)
    models_requested = normalize_models(args.models)
    models_to_run, skipped_models = available_models(models_requested)

    if not models_to_run:
        raise RuntimeError("No available models to run. Install lightgbm/xgboost if needed.")

    if args.progress:
        print(f"[setup] models requested={models_requested}")
        print(f"[setup] models to run={models_to_run}")
        if skipped_models:
            print(f"[setup] skipped={skipped_models}")

    df, numeric_features, categorical_features = load_dataset(final_path, config, args.feature_set, args.progress)

    all_metric_rows = []
    all_conf_rows = []
    candidate_infos = []
    all_imp = []
    errors = []

    # Prepare candidate count.
    candidates = []
    for exp in args.experiments:
        for model_name in models_to_run:
            for params in base_param_grid(model_name, args.preset):
                for pw_mode in args.pos_weight_modes:
                    candidates.append((exp, model_name, params, pw_mode))

    if args.progress:
        print(f"[setup] total candidates={len(candidates):,}")
        print(f"[setup] output_dir={output_dir}")

    best_pred_df = None
    best_candidate_id = None
    best_val_f1 = -np.inf

    for i, (exp, model_name, params, pw_mode) in enumerate(candidates, start=1):
        if args.progress:
            print("=" * 90)
            print(f"[job {i}/{len(candidates)}] exp={exp}, model={model_name}, param={params['param_id']}, pos_weight={pw_mode}")
            print("=" * 90, flush=True)

        try:
            metric_rows, conf_rows, info, imp, pred_df = run_candidate(
                df=df,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                experiment_id=exp,
                model_name=model_name,
                params=params,
                pos_weight_mode_input=pw_mode,
                args=args,
                output_dir=output_dir,
                candidate_index=i,
            )

            all_metric_rows.extend(metric_rows)
            all_conf_rows.extend(conf_rows)
            candidate_infos.append(info)
            if imp is not None and not imp.empty:
                all_imp.append(imp)

            if args.save_predictions_for_best:
                if info["validation_f1"] > best_val_f1:
                    best_val_f1 = info["validation_f1"]
                    best_candidate_id = info["candidate_id"]
                    best_pred_df = pred_df.copy() if pred_df is not None else None

            if args.progress:
                print(
                    f"[done] {info['candidate_id']} | "
                    f"val_f1={info['validation_f1']:.4f}, val_pr_auc={info['validation_pr_auc']:.4f}, "
                    f"test_f1={info['test_f1']:.4f}, test_pr_auc={info['test_pr_auc']:.4f}, "
                    f"threshold={info['selected_threshold']:.6f}, fit={info['fit_seconds']:.1f}s",
                    flush=True,
                )

        except Exception as exc:
            err = {
                "job_index": i,
                "experiment_id": exp,
                "model_name": model_name,
                "param_id": params.get("param_id"),
                "pos_weight_mode": pw_mode,
                "error": repr(exc),
            }
            errors.append(err)
            print(f"[warning] Candidate failed but tuning continues: {err}", file=sys.stderr)

    metrics_df = pd.DataFrame(all_metric_rows)
    conf_df = pd.DataFrame(all_conf_rows)
    info_df = pd.DataFrame(candidate_infos)
    errors_df = pd.DataFrame(errors)
    skipped_df = pd.DataFrame(skipped_models)

    if all_imp:
        imp_df = pd.concat(all_imp, ignore_index=True)
    else:
        imp_df = pd.DataFrame()

    if not info_df.empty:
        ranking_df = info_df.sort_values(
            ["validation_f1", "validation_pr_auc", "validation_balanced_accuracy", "test_f1"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        ranking_df.insert(0, "validation_rank", np.arange(1, len(ranking_df) + 1))
    else:
        ranking_df = pd.DataFrame()

    write_csv(output_dir / "gbm_tuning_threshold_metrics.csv", metrics_df, args.overwrite)
    write_csv(output_dir / "gbm_tuning_confusion_matrices.csv", conf_df, args.overwrite)
    write_csv(output_dir / "gbm_tuning_candidate_info.csv", info_df, args.overwrite)
    write_csv(output_dir / "gbm_tuning_ranking.csv", ranking_df, args.overwrite)
    write_csv(output_dir / "gbm_tuning_feature_importance_top.csv", imp_df, args.overwrite)
    write_csv(output_dir / "gbm_tuning_errors.csv", errors_df, args.overwrite)
    write_csv(output_dir / "gbm_tuning_skipped_models.csv", skipped_df, args.overwrite)

    if args.save_predictions_for_best and best_pred_df is not None:
        pred_path = output_dir / f"predictions_best_{best_candidate_id}.csv.gz"
        ensure_output_path(pred_path, args.overwrite)
        best_pred_df.to_csv(pred_path, index=False, compression="gzip")

    summary = {
        "script": SCRIPT_NAME,
        "status": "done",
        "final_dataset": str(final_path),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "experiments": args.experiments,
        "feature_set": args.feature_set,
        "models_requested": models_requested,
        "models_run": models_to_run,
        "skipped_models": skipped_models,
        "preset": args.preset,
        "pos_weight_modes": args.pos_weight_modes,
        "use_gpu": bool(args.use_gpu),
        "gpu_device_id": int(args.gpu_device_id),
        "max_train_rows": int(args.max_train_rows),
        "primary_strategy": args.primary_strategy,
        "total_candidates_planned": int(len(candidates)),
        "successful_candidates": int(len(info_df)),
        "failed_candidates": int(len(errors_df)),
        "metrics_rows": int(len(metrics_df)),
        "candidate_info_rows": int(len(info_df)),
        "best_candidate_by_validation": ranking_df.head(1).to_dict(orient="records")[0] if not ranking_df.empty else None,
        "top_10_candidates": ranking_df.head(10).to_dict(orient="records") if not ranking_df.empty else [],
        "elapsed_seconds": round(time.time() - t0, 3),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }

    write_json(output_dir / "gbm_tuning_run_summary.json", summary, args.overwrite)

    if args.progress:
        print("\n[ranking top 20]")
        if not ranking_df.empty:
            cols = [
                "validation_rank",
                "candidate_id",
                "model_name",
                "param_id",
                "pos_weight_mode",
                "pos_weight",
                "selected_threshold",
                "validation_f1",
                "validation_precision",
                "validation_recall",
                "validation_pr_auc",
                "test_f1",
                "test_precision",
                "test_recall",
                "test_pr_auc",
                "fit_seconds",
            ]
            print(ranking_df[cols].head(20).to_string(index=False))
        else:
            print("(empty)")

    print("=" * 90)
    print(f"[{SCRIPT_NAME}] DONE")
    print(f"Ranking: {output_dir / 'gbm_tuning_ranking.csv'}")
    print(f"Candidate info: {output_dir / 'gbm_tuning_candidate_info.csv'}")
    print(f"Threshold metrics: {output_dir / 'gbm_tuning_threshold_metrics.csv'}")
    print(f"Run summary: {output_dir / 'gbm_tuning_run_summary.json'}")
    if errors:
        print(f"Some candidates failed. See: {output_dir / 'gbm_tuning_errors.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        raise
