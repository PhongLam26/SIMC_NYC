
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10.3 — Train tree / boosting models for NYC 311 abnormal service-demand forecasting.

Project:
    SIMC NYC — Semantics-Aware Explainable Machine Learning for Urban Service
    Demand Forecasting in Smart Cities.

Purpose:
    Train stronger non-linear models after Step 10.2 baselines.

Supported models:
    - lightgbm              optional, requires lightgbm
    - xgboost               optional, requires xgboost
    - hist_gradient_boosting sklearn built-in
    - random_forest         sklearn built-in
    - extra_trees           sklearn built-in

Design goals:
    1. Use the same leakage-safe model_config.json from Step 10.1.
    2. Use chronological splits, never random train/test splits.
    3. Support E1/E2/E3/E4 experiments.
    4. Support feature-set ablation.
    5. Handle class imbalance using sample weights / class weights.
    6. Save validation/test metrics and feature importances.
    7. Allow laptop-friendly sampling before full training.

Recommended workflow on 16GB RAM laptop:

    A. First, run broad comparison on a sample:
       .\\.venv\\Scripts\\python.exe .\\code_model\\train_tree_models.py --overwrite --progress ^
           --experiments E1_main_2015_2025 ^
           --feature-sets historical_only historical_calendar_weather full_without_covid_period_features ^
           --models lightgbm xgboost hist_gradient_boosting extra_trees random_forest ^
           --max-train-rows 300000

    B. Then run full training for the best 1–2 models, for example:
       .\\.venv\\Scripts\\python.exe .\\code_model\\train_tree_models.py --overwrite --progress ^
           --experiments E1_main_2015_2025 E2_without_covid_years ^
           --feature-sets historical_calendar_weather full_without_covid_period_features ^
           --models lightgbm hist_gradient_boosting ^
           --save-models

Optional installs:
    .\\.venv\\Scripts\\python.exe -m pip install lightgbm xgboost

Outputs:
    data/processed/model_results/tree_models/tree_model_metrics.csv
    data/processed/model_results/tree_models/tree_model_confusion_matrices.csv
    data/processed/model_results/tree_models/tree_model_validation_ranking.csv
    data/processed/model_results/tree_models/tree_model_feature_importance.csv
    data/processed/model_results/tree_models/tree_model_info.csv
    data/processed/model_results/tree_models/tree_model_run_summary.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Required sklearn imports
# ---------------------------------------------------------------------

try:
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
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
except ModuleNotFoundError as exc:
    print(
        "\n[train_tree_models] Missing required package.\n"
        "Install into project .venv using:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn joblib\n",
        file=sys.stderr,
    )
    raise exc


# Optional model packages
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


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SCRIPT_NAME = "train_tree_models"

TARGET_COL = "abnormal_increase_next_week"
READY_COL = "final_train_ready_flag"
SPLIT_COL = "time_split"
YEAR_COL = "year"
WEEK_COL = "week_start"
TARGET_WEEK_COL = "target_week"
TARGET_YEAR_COL = "target_year"

DEFAULT_FINAL_REL = Path("data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz")
DEFAULT_CONFIG_REL = Path("data/processed/model_ready/model_config.json")
DEFAULT_OUTPUT_DIR_REL = Path("data/processed/model_results/tree_models")

DEFAULT_EXPERIMENTS = ["E1_main_2015_2025"]
DEFAULT_FEATURE_SETS = ["historical_only", "historical_calendar_weather", "full_without_covid_period_features"]
DEFAULT_MODELS = ["lightgbm", "xgboost", "hist_gradient_boosting", "extra_trees", "random_forest"]

SKLEARN_TREE_MODELS = {"hist_gradient_boosting", "random_forest", "extra_trees"}
OPTIONAL_MODELS = {"lightgbm", "xgboost"}
ALL_MODELS = ["lightgbm", "xgboost", "hist_gradient_boosting", "extra_trees", "random_forest"]

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
        description="Train tree/boosting models for NYC 311 abnormal service-demand forecasting."
    )

    parser.add_argument("--final-dataset", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiments to run, e.g. E1_main_2015_2025 E2_without_covid_years.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=DEFAULT_FEATURE_SETS,
        help="Feature sets from model_config.json.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models: lightgbm xgboost hist_gradient_boosting extra_trees random_forest or all.",
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help=(
            "Optional stratified training sample size per experiment/model/feature-set. "
            "0 means use all training rows."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    # Model-size controls. Defaults are moderate for a laptop.
    parser.add_argument("--n-estimators", type=int, default=500, help="For LightGBM/XGBoost/RF/ExtraTrees.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="For boosting models.")
    parser.add_argument("--max-depth", type=int, default=8, help="For XGBoost/RF/ExtraTrees. Use 0 for unlimited in RF/ET.")
    parser.add_argument("--num-leaves", type=int, default=63, help="LightGBM num_leaves.")
    parser.add_argument("--min-samples-leaf", type=int, default=20, help="RF/ExtraTrees min_samples_leaf.")
    parser.add_argument("--hgb-max-iter", type=int, default=300, help="HistGradientBoosting max_iter.")
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)

    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save fitted model and preprocessing metadata. Can use disk space.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save validation/test predictions. Can create large files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Path / I/O utilities
# ---------------------------------------------------------------------

def get_project_root() -> Path:
    # Expected location: SIMC_PROJECT/code_model/train_tree_models.py
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
# Model availability and validation
# ---------------------------------------------------------------------

def normalize_model_list(models: List[str]) -> List[str]:
    out = []
    for m in models:
        if m.lower() == "all":
            out.extend(ALL_MODELS)
        else:
            out.append(m.lower())

    normalized = []
    for m in out:
        aliases = {
            "lgbm": "lightgbm",
            "light_gbm": "lightgbm",
            "xgb": "xgboost",
            "hgb": "hist_gradient_boosting",
            "histgb": "hist_gradient_boosting",
            "rf": "random_forest",
            "et": "extra_trees",
            "extratrees": "extra_trees",
        }
        m2 = aliases.get(m, m)
        if m2 not in ALL_MODELS:
            raise ValueError(f"Unknown model: {m}. Allowed: {ALL_MODELS} or all")
        if m2 not in normalized:
            normalized.append(m2)
    return normalized


def available_models(models: List[str]) -> Tuple[List[str], List[dict]]:
    available = []
    skipped = []

    for m in models:
        if m == "lightgbm" and not HAS_LIGHTGBM:
            skipped.append({
                "model_name": m,
                "reason": "lightgbm_not_installed",
                "install_command": ".\\.venv\\Scripts\\python.exe -m pip install lightgbm",
            })
            continue
        if m == "xgboost" and not HAS_XGBOOST:
            skipped.append({
                "model_name": m,
                "reason": "xgboost_not_installed",
                "install_command": ".\\.venv\\Scripts\\python.exe -m pip install xgboost",
            })
            continue
        available.append(m)

    return available, skipped


# ---------------------------------------------------------------------
# Dataset loading and preprocessing
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
    required = []
    for fs in feature_sets:
        required.extend(config["feature_sets"][fs]["features"])

    required.extend(
        ID_OUTPUT_COLS
        + [
            READY_COL,
            SPLIT_COL,
            YEAR_COL,
        ]
    )
    return unique_preserve_order(required)


def load_dataset(final_path: Path, usecols: List[str], progress: bool = False) -> pd.DataFrame:
    if not final_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {final_path}")

    header = pd.read_csv(final_path, nrows=0)
    available = set(header.columns)
    actual_usecols = [c for c in usecols if c in available]
    missing = [c for c in usecols if c not in available]

    if progress:
        print(f"[load] final dataset: {final_path}")
        print(f"[load] requested columns={len(usecols):,}, available used={len(actual_usecols):,}")
        if missing:
            print(f"[load] missing requested/optional columns ignored: {missing}")

    df = pd.read_csv(final_path, usecols=actual_usecols, low_memory=False)

    if progress:
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"[load] loaded rows={len(df):,}, columns={len(df.columns):,}, memory={mem_mb:,.1f} MB")

    return df


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


def clean_raw_feature_values(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
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


def split_feature_types(features: List[str], config: dict, df_columns: Iterable[str]) -> Tuple[List[str], List[str]]:
    df_cols = set(df_columns)
    categorical_all = set(config.get("categorical_features", []))
    present_features = [f for f in features if f in df_cols]
    categorical = [f for f in present_features if f in categorical_all]
    numeric = [f for f in present_features if f not in categorical]
    return numeric, categorical


def stratified_sample_indices(
    y: pd.Series,
    max_rows: int,
    random_state: int,
) -> pd.Index:
    if not max_rows or max_rows <= 0 or max_rows >= len(y):
        return y.index

    tmp = pd.DataFrame({"y": y.astype(int)}, index=y.index)
    rng = np.random.RandomState(random_state)

    sampled_parts = []
    for cls, g in tmp.groupby("y"):
        n_cls = len(g)
        n_take = max(1, int(round(max_rows * n_cls / len(tmp))))
        n_take = min(n_take, n_cls)
        sampled_parts.append(g.sample(n=n_take, random_state=random_state))

    out = pd.concat(sampled_parts).index
    if len(out) > max_rows:
        out = pd.Index(pd.Series(out).sample(n=max_rows, random_state=random_state).values)

    return out


class FastTabularPreprocessor:
    """
    Lightweight preprocessing for tree models.

    - Numeric features: median imputation, float32.
    - Categorical features: train-set ordinal mapping, missing=-1, unknown=-1.
    - No one-hot encoding to keep memory lower on laptop.
    - This is acceptable for tree/boosting baselines and keeps original feature names.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.numeric_medians_: Dict[str, float] = {}
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.feature_names_: List[str] = self.numeric_features + self.categorical_features

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
            self.category_maps_[col] = {v: i for i, v in enumerate(levels)}

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.numeric_features:
            num = pd.DataFrame(index=df.index)
            for col in self.numeric_features:
                med = self.numeric_medians_.get(col, 0.0)
                num[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(med)
                    .astype("float32")
                )
            parts.append(num)

        if self.categorical_features:
            cat = pd.DataFrame(index=df.index)
            for col in self.categorical_features:
                cmap = self.category_maps_.get(col, {})
                s = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
                cat[col] = s.map(cmap).fillna(-1).astype("int16")
            parts.append(cat)

        if not parts:
            return pd.DataFrame(index=df.index)

        out = pd.concat(parts, axis=1)
        return out[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def to_metadata(self) -> dict:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "numeric_medians": self.numeric_medians_,
            "category_maps": self.category_maps_,
            "feature_names": self.feature_names_,
        }


def compute_sample_weight(y: pd.Series) -> np.ndarray:
    """
    Balanced binary sample weights:
        class 0 weight = n / (2*n0)
        class 1 weight = n / (2*n1)
    """
    y_arr = np.asarray(y, dtype=int)
    n = len(y_arr)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())

    weights = np.ones(n, dtype="float32")
    if n_pos > 0 and n_neg > 0:
        weights[y_arr == 1] = n / (2.0 * n_pos)
        weights[y_arr == 0] = n / (2.0 * n_neg)
    return weights


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
    threshold: float = 0.5,
) -> Tuple[dict, dict]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if y_score is None:
        y_score_arr = y_pred_arr.astype(float)
    else:
        y_score_arr = np.asarray(y_score, dtype=float)

    y_score_arr = np.nan_to_num(y_score_arr, nan=0.0, posinf=1.0, neginf=0.0)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

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
# Model builders
# ---------------------------------------------------------------------

def make_model(
    model_name: str,
    args: argparse.Namespace,
    y_train: pd.Series,
):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0

    if model_name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise RuntimeError("LightGBM is not installed.")
        return LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=-1,
        )

    if model_name == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost is not installed.")
        max_depth = args.max_depth if args.max_depth and args.max_depth > 0 else 8
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )

    if model_name == "hist_gradient_boosting":
        # Uses sample_weight for imbalance; robust to numeric/categorical ordinal encoding.
        return HistGradientBoostingClassifier(
            max_iter=args.hgb_max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=args.random_state,
        )

    if model_name == "random_forest":
        max_depth = None if args.max_depth == 0 else args.max_depth
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=0,
        )

    if model_name == "extra_trees":
        max_depth = None if args.max_depth == 0 else args.max_depth
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=0,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def fit_model(model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: np.ndarray):
    """
    Fit with appropriate weighting.
    """
    if model_name in {"lightgbm", "hist_gradient_boosting"}:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    elif model_name == "xgboost":
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    else:
        # RF/ExtraTrees already use class_weight; adding sample_weight can double-correct.
        model.fit(X_train, y_train)
    return model


def predict_score(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        try:
            score = model.predict_proba(X)[:, 1]
            return np.asarray(y_pred).astype(int), np.asarray(score, dtype=float)
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            score = model.decision_function(X)
            return np.asarray(y_pred).astype(int), np.asarray(score, dtype=float)
        except Exception:
            pass

    return np.asarray(y_pred).astype(int), np.asarray(y_pred, dtype=float)


def get_feature_importance(model, model_name: str, feature_names: List[str]) -> pd.DataFrame:
    importance = None
    importance_type = ""

    if hasattr(model, "feature_importances_"):
        try:
            importance = np.asarray(model.feature_importances_, dtype=float)
            importance_type = "feature_importances_"
        except Exception:
            importance = None

    if importance is None and model_name == "hist_gradient_boosting":
        # HGB has no built-in feature_importances_. Permutation importance is expensive,
        # so leave it to SHAP/permutation step later.
        return pd.DataFrame(
            columns=["model_name", "feature", "importance", "importance_type", "rank"]
        )

    if importance is None:
        return pd.DataFrame(
            columns=["model_name", "feature", "importance", "importance_type", "rank"]
        )

    n = min(len(importance), len(feature_names))
    df_imp = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": feature_names[:n],
            "importance": importance[:n],
            "importance_type": importance_type,
        }
    )
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["rank"] = np.arange(1, len(df_imp) + 1)
    return df_imp


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def get_split_raw(
    df: pd.DataFrame,
    experiment_id: str,
    split: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mask = experiment_mask(df, experiment_id, split)
    X = df.loc[mask, feature_cols].copy()
    y = df.loc[mask, TARGET_COL].astype(int)
    return X, y, mask


def train_one_model(
    *,
    df: pd.DataFrame,
    config: dict,
    experiment_id: str,
    feature_set: str,
    model_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[List[dict], List[dict], List[dict], pd.DataFrame]:
    feature_set_map = config["feature_sets"]
    features = feature_set_map[feature_set]["features"]
    numeric_features, categorical_features = split_feature_types(features, config, df.columns)
    feature_cols = numeric_features + categorical_features

    X_train_raw, y_train, train_mask = get_split_raw(df, experiment_id, "train", feature_cols)
    if len(y_train) == 0:
        raise ValueError(f"No train rows for {experiment_id}/{feature_set}")

    # Optional stratified downsample for laptop experiments.
    train_rows_before_sample = len(y_train)
    sampled_idx = stratified_sample_indices(y_train, args.max_train_rows, args.random_state)
    X_train_raw = X_train_raw.loc[sampled_idx]
    y_train = y_train.loc[sampled_idx]

    if args.progress:
        sample_note = (
            f"sampled_train={len(y_train):,}/{train_rows_before_sample:,}"
            if len(y_train) != train_rows_before_sample
            else f"train_rows={len(y_train):,}"
        )
        print(
            f"[train] model={model_name}, exp={experiment_id}, fs={feature_set}, "
            f"features={len(feature_cols):,}, numeric={len(numeric_features):,}, "
            f"categorical={len(categorical_features):,}, {sample_note}, "
            f"pos_share={y_train.mean():.4f}",
            flush=True,
        )

    pre = FastTabularPreprocessor(numeric_features, categorical_features)

    t_pre = time.time()
    X_train = pre.fit_transform(X_train_raw)
    preprocess_seconds = time.time() - t_pre

    del X_train_raw
    gc.collect()

    sample_weight = compute_sample_weight(y_train)

    model = make_model(model_name, args, y_train)

    t_fit = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit_model(model, model_name, X_train, y_train, sample_weight)
    fit_seconds = time.time() - t_fit

    warning_messages = [str(w.message) for w in caught[:10]]

    metrics_rows: List[dict] = []
    conf_rows: List[dict] = []
    model_rows: List[dict] = []

    for split in SPLITS:
        X_raw, y_true, mask = get_split_raw(df, experiment_id, split, feature_cols)
        if len(y_true) == 0:
            continue

        t_transform = time.time()
        X_split = pre.transform(X_raw)
        transform_seconds = time.time() - t_transform

        y_pred, y_score = predict_score(model, X_split)

        m, c = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            experiment_id=experiment_id,
            split=split,
            model_name=model_name,
            feature_set=feature_set,
            train_rows=len(y_train),
            threshold=0.5,
        )
        m.update(
            {
                "selected_feature_count": int(len(feature_cols)),
                "numeric_feature_count": int(len(numeric_features)),
                "categorical_feature_count": int(len(categorical_features)),
                "train_rows_before_sample": int(train_rows_before_sample),
                "max_train_rows": int(args.max_train_rows),
                "preprocess_seconds": float(preprocess_seconds),
                "fit_seconds": float(fit_seconds),
                "transform_seconds": float(transform_seconds),
                "warnings_count": int(len(caught)),
            }
        )

        metrics_rows.append(m)
        conf_rows.append(c)

        if args.save_predictions and split in {"validation", "test"}:
            pred_cols = [c for c in ID_OUTPUT_COLS if c in df.columns]
            pred_df = df.loc[mask, pred_cols].copy()
            pred_df["y_true"] = y_true.values
            pred_df["y_pred"] = y_pred
            pred_df["y_score"] = y_score
            pred_path = output_dir / f"predictions_{model_name}_{experiment_id}_{feature_set}_{split}.csv.gz"
            ensure_output_path(pred_path, args.overwrite)
            pred_df.to_csv(pred_path, index=False, compression="gzip")

        del X_raw, X_split
        gc.collect()

    imp_df = get_feature_importance(model, model_name, pre.feature_names_)
    if not imp_df.empty:
        imp_df.insert(0, "experiment_id", experiment_id)
        imp_df.insert(1, "feature_set", feature_set)

    model_path = ""
    preprocessor_path = ""
    if args.save_models:
        model_path_obj = output_dir / f"model_{model_name}_{experiment_id}_{feature_set}.joblib"
        pre_path_obj = output_dir / f"preprocessor_{model_name}_{experiment_id}_{feature_set}.joblib"
        ensure_output_path(model_path_obj, args.overwrite)
        ensure_output_path(pre_path_obj, args.overwrite)
        joblib.dump(model, model_path_obj)
        joblib.dump(pre.to_metadata(), pre_path_obj)
        model_path = str(model_path_obj)
        preprocessor_path = str(pre_path_obj)

    model_rows.append(
        {
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "model_name": model_name,
            "train_rows": int(len(y_train)),
            "train_rows_before_sample": int(train_rows_before_sample),
            "selected_feature_count": int(len(feature_cols)),
            "numeric_feature_count": int(len(numeric_features)),
            "categorical_feature_count": int(len(categorical_features)),
            "preprocess_seconds": float(preprocess_seconds),
            "fit_seconds": float(fit_seconds),
            "warnings_count": int(len(caught)),
            "warning_messages_preview": " | ".join(warning_messages),
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
        }
    )

    del X_train, y_train, model
    gc.collect()

    return metrics_rows, conf_rows, model_rows, imp_df


def build_validation_ranking(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    val = metrics_df[metrics_df["split"] == "validation"].copy()
    if val.empty:
        return pd.DataFrame()
    val = val.sort_values(
        ["pr_auc", "f1", "balanced_accuracy", "roc_auc"],
        ascending=[False, False, False, False],
    )
    val["validation_rank"] = np.arange(1, len(val) + 1)
    return val


def make_run_summary(
    *,
    args: argparse.Namespace,
    final_path: Path,
    config_path: Path,
    output_dir: Path,
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    model_info_df: pd.DataFrame,
    skipped_models: List[dict],
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
        "requested_models": args.models,
        "max_train_rows": int(args.max_train_rows),
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "num_leaves": int(args.num_leaves),
        "hgb_max_iter": int(args.hgb_max_iter),
        "save_models": bool(args.save_models),
        "metrics_rows": int(len(metrics_df)),
        "model_info_rows": int(len(model_info_df)),
        "skipped_models": skipped_models,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_minutes": round(elapsed_seconds / 60, 3),
    }

    ranking = build_validation_ranking(metrics_df)
    if not ranking.empty:
        summary["top_validation_models"] = ranking[
            [
                "validation_rank",
                "experiment_id",
                "model_name",
                "feature_set",
                "rows",
                "train_rows",
                "positive_share",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
                "fit_seconds",
            ]
        ].head(20).to_dict(orient="records")

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    args.models = normalize_model_list(args.models)
    models_to_run, skipped_models = available_models(args.models)

    if not models_to_run:
        raise RuntimeError(
            "No requested models are available. "
            "Install optional packages if needed:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install lightgbm xgboost"
        )

    project_root = get_project_root()
    final_path = resolve_path(project_root, args.final_dataset, DEFAULT_FINAL_REL)
    config_path = resolve_path(project_root, args.config, DEFAULT_CONFIG_REL)
    output_dir = resolve_path(project_root, args.output_dir, DEFAULT_OUTPUT_DIR_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    # Validate experiment and feature set names.
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
    df = infer_core_fields(df)
    df = clean_raw_feature_values(df, all_feature_cols)

    if args.progress:
        print(f"[models] requested={args.models}")
        print(f"[models] available_to_run={models_to_run}")
        if skipped_models:
            print(f"[models] skipped={skipped_models}")
        print(f"[data] target positive share={df[TARGET_COL].mean(skipna=True):.4f}")

    metrics_rows: List[dict] = []
    conf_rows: List[dict] = []
    model_rows: List[dict] = []
    importance_parts: List[pd.DataFrame] = []
    error_rows: List[dict] = []

    total_jobs = len(args.experiments) * len(args.feature_sets) * len(models_to_run)
    job_i = 0

    for exp_id in args.experiments:
        for fs in args.feature_sets:
            for model_name in models_to_run:
                job_i += 1
                if args.progress:
                    print("=" * 80)
                    print(f"[job {job_i}/{total_jobs}] exp={exp_id}, feature_set={fs}, model={model_name}")
                    print("=" * 80, flush=True)

                try:
                    m_rows, c_rows, info_rows, imp_df = train_one_model(
                        df=df,
                        config=config,
                        experiment_id=exp_id,
                        feature_set=fs,
                        model_name=model_name,
                        args=args,
                        output_dir=output_dir,
                    )
                    metrics_rows.extend(m_rows)
                    conf_rows.extend(c_rows)
                    model_rows.extend(info_rows)
                    if imp_df is not None and not imp_df.empty:
                        importance_parts.append(imp_df)

                except Exception as exc:
                    err = {
                        "experiment_id": exp_id,
                        "feature_set": fs,
                        "model_name": model_name,
                        "error": repr(exc),
                    }
                    error_rows.append(err)
                    print(f"[warning] job failed but script will continue: {err}", file=sys.stderr)

    metrics_df = pd.DataFrame(metrics_rows)
    conf_df = pd.DataFrame(conf_rows)
    model_info_df = pd.DataFrame(model_rows)
    errors_df = pd.DataFrame(error_rows)

    if importance_parts:
        feature_importance_df = pd.concat(importance_parts, ignore_index=True)
    else:
        feature_importance_df = pd.DataFrame(
            columns=["experiment_id", "feature_set", "model_name", "feature", "importance", "importance_type", "rank"]
        )

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["experiment_id", "model_name", "feature_set", "split"]).reset_index(drop=True)
    if not conf_df.empty:
        conf_df = conf_df.sort_values(["experiment_id", "model_name", "feature_set", "split"]).reset_index(drop=True)
    if not model_info_df.empty:
        model_info_df = model_info_df.sort_values(["experiment_id", "feature_set", "model_name"]).reset_index(drop=True)

    ranking_df = build_validation_ranking(metrics_df)

    write_csv(output_dir / "tree_model_metrics.csv", metrics_df, args.overwrite)
    write_csv(output_dir / "tree_model_confusion_matrices.csv", conf_df, args.overwrite)
    write_csv(output_dir / "tree_model_validation_ranking.csv", ranking_df, args.overwrite)
    write_csv(output_dir / "tree_model_feature_importance.csv", feature_importance_df, args.overwrite)
    write_csv(output_dir / "tree_model_info.csv", model_info_df, args.overwrite)
    write_csv(output_dir / "tree_model_errors.csv", errors_df, args.overwrite)

    skipped_df = pd.DataFrame(skipped_models)
    write_csv(output_dir / "tree_model_skipped_models.csv", skipped_df, args.overwrite)

    summary = make_run_summary(
        args=args,
        final_path=final_path,
        config_path=config_path,
        output_dir=output_dir,
        df=df,
        metrics_df=metrics_df,
        model_info_df=model_info_df,
        skipped_models=skipped_models,
        elapsed_seconds=time.time() - t0,
    )
    write_json(output_dir / "tree_model_run_summary.json", summary, args.overwrite)

    if args.progress:
        print("\n[summary] Validation ranking:")
        if not ranking_df.empty:
            cols = [
                "validation_rank",
                "experiment_id",
                "model_name",
                "feature_set",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
                "fit_seconds",
            ]
            print(ranking_df[cols].head(30).to_string(index=False))
        else:
            print("(empty)")

    print("=" * 80)
    print(f"[{SCRIPT_NAME}] DONE")
    print(f"Metrics: {output_dir / 'tree_model_metrics.csv'}")
    print(f"Validation ranking: {output_dir / 'tree_model_validation_ranking.csv'}")
    print(f"Feature importance: {output_dir / 'tree_model_feature_importance.csv'}")
    print(f"Run summary: {output_dir / 'tree_model_run_summary.json'}")
    if error_rows:
        print(f"Some jobs failed. See: {output_dir / 'tree_model_errors.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        raise
