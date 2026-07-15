#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10.5B/C — Ensemble + global/per-category threshold tuning.

Project:
    SIMC NYC — Semantics-Aware Explainable Machine Learning for Urban Service
    Demand Forecasting in Smart Cities.

Purpose:
    Compare final decision-layer strategies after LightGBM/XGBoost tuning:

      1. Load or rebuild best LightGBM score.
      2. Load or rebuild best XGBoost score.
      3. Build ensemble score.
      4. Tune one global threshold on validation.
      5. Tune complaint-category-specific thresholds on validation.
      6. Compare:
           - LightGBM global threshold
           - XGBoost global threshold
           - Ensemble global threshold
           - LightGBM per-category threshold
           - XGBoost per-category threshold
           - Ensemble per-category threshold

Correct protocol:
    - Train/rebuild models on train period only.
    - Select thresholds on validation period only.
    - Report test performance once using the selected validation thresholds.
    - Do not tune on test.

Recommended command:
    .\\.venv\\Scripts\\python.exe .\\code_model\\ensemble_and_category_thresholds.py --overwrite --progress ^
        --lgbm-tuning-dir data/processed/model_results/gbm_hyperparam_tuning_lgbm_balanced_gpu ^
        --xgb-tuning-dir data/processed/model_results/gbm_hyperparam_tuning_xgb_balanced_cuda ^
        --xgboost-use-gpu ^
        --ensemble-weights 0.5 ^
        --output-dir data/processed/model_results/ensemble_category_thresholds

If LightGBM GPU is already configured and you want to try it:
    add --lightgbm-use-gpu

Outputs:
    final_model_comparison.csv
    final_model_comparison_long.csv
    threshold_selection_summary.csv
    category_thresholds.csv
    metrics_by_category.csv
    confusion_matrices.csv
    score_summary.csv
    ensemble_threshold_run_summary.json
    optional: scored_validation_test.csv.gz with --save-scores
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
        "\n[ensemble_and_category_thresholds] Missing scikit-learn.\n"
        "Install into project .venv:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn\n",
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


SCRIPT_NAME = "ensemble_and_category_thresholds"

TARGET_COL = "abnormal_increase_next_week"
READY_COL = "final_train_ready_flag"
SPLIT_COL = "time_split"
YEAR_COL = "year"
WEEK_COL = "week_start"
TARGET_WEEK_COL = "target_week"
TARGET_YEAR_COL = "target_year"
CATEGORY_COL = "complaint_category"

DEFAULT_FINAL_REL = Path("data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz")
DEFAULT_CONFIG_REL = Path("data/processed/model_ready/model_config.json")
DEFAULT_LGBM_DIR_REL = Path("data/processed/model_results/gbm_hyperparam_tuning_lgbm_balanced_gpu")
DEFAULT_XGB_DIR_REL = Path("data/processed/model_results/gbm_hyperparam_tuning_xgb_balanced_cuda")
DEFAULT_OUTPUT_DIR_REL = Path("data/processed/model_results/ensemble_category_thresholds")

DEFAULT_EXPERIMENT = "E1_main_2015_2025"
DEFAULT_FEATURE_SET = "full_without_covid_period_features"

ID_COLS = [
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
        description="Build LightGBM/XGBoost ensemble and tune global/per-category thresholds."
    )

    parser.add_argument("--final-dataset", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--lgbm-tuning-dir", type=str, default=None)
    parser.add_argument("--xgb-tuning-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--feature-set", type=str, default=DEFAULT_FEATURE_SET)
    parser.add_argument("--ranking-row", type=int, default=1,
                        help="Use this validation rank from each gbm_tuning_ranking.csv. Default rank 1.")

    parser.add_argument("--ensemble-weights", nargs="+", type=float, default=[0.5],
                        help=(
                            "Weight for LightGBM in ensemble_score = w*lgbm + (1-w)*xgb. "
                            "Default 0.5. You may pass multiple values, e.g. 0.25 0.5 0.75."
                        ))

    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--quantile-thresholds", type=int, default=200)

    parser.add_argument("--primary-strategy", type=str, default="best_f1",
                        choices=[
                            "best_f1",
                            "best_f2",
                            "best_balanced_accuracy",
                            "best_precision_at_min_recall",
                            "best_recall_at_min_precision",
                        ])
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-precision", type=float, default=0.25)
    parser.add_argument("--min-category-positive", type=int, default=30,
                        help="Fallback to global threshold for categories with too few validation positives.")

    parser.add_argument("--xgboost-use-gpu", action="store_true",
                        help="Use NVIDIA CUDA for XGBoost rebuild.")
    parser.add_argument("--lightgbm-use-gpu", action="store_true",
                        help="Use LightGBM OpenCL GPU for LightGBM rebuild. On Windows this may choose Intel iGPU.")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--xgboost-gpu-mode", type=str, default="modern", choices=["modern", "legacy"])
    parser.add_argument("--lightgbm-gpu-platform-id", type=int, default=-1)

    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="Optional stratified training sample for rebuilding models. 0 means full train.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--save-scores", action="store_true",
                        help="Save scored validation/test rows to scored_validation_test.csv.gz.")
    parser.add_argument("--force-rebuild-scores", action="store_true",
                        help="Ignore saved prediction files in tuning dirs and rebuild model scores.")
    parser.add_argument("--analysis-type", type=str, default="unspecified",
                        choices=["unspecified", "prospective", "retrospective_context"],
                        help="Analysis protocol identifier written to output metadata.")
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
        raise FileExistsError(f"Output already exists. Use --overwrite to replace: {path}")


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


def unique_preserve_order(cols: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ---------------------------------------------------------------------
# Dataset / preprocessing
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


def load_dataset(
    final_path: Path,
    config: dict,
    feature_set: str,
    progress: bool,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if feature_set not in config["feature_sets"]:
        raise ValueError(f"Feature set not found in model_config.json: {feature_set}")

    features = list(config["feature_sets"][feature_set]["features"])
    categorical_all = set(config.get("categorical_features", []))

    requested = unique_preserve_order(features + ID_COLS + [READY_COL, SPLIT_COL, YEAR_COL])
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

    if CATEGORY_COL not in df.columns:
        raise ValueError(f"Required category column not found: {CATEGORY_COL}")

    if progress:
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(
            f"[load] rows={len(df):,}, cols={len(df.columns):,}, "
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
        medians = {}
        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = s.median()
            if pd.isna(med):
                med = 0.0
            medians[col] = float(med)
        self.numeric_medians_ = medians

        category_levels = {}
        for col in self.categorical_features:
            s = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
            levels = sorted(s.unique().tolist())
            category_levels[col] = levels
        self.category_levels_ = category_levels
        self.one_hot_feature_names_ = {
            col: [f"{col}={level}" for level in levels]
            for col, levels in self.category_levels_.items()
        }
        self.feature_names_ = (
            list(self.numeric_features)
            + [name for col in self.categorical_features for name in self.one_hot_feature_names_.get(col, [])]
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.numeric_features:
            # Build all numeric columns in one dictionary to avoid pandas fragmentation warnings.
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


def get_split(
    df: pd.DataFrame,
    experiment_id: str,
    split: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mask = experiment_mask(df, experiment_id, split)
    return df.loc[mask, feature_cols].copy(), df.loc[mask, TARGET_COL].astype(int), mask


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
# Ranking / model rebuild
# ---------------------------------------------------------------------

def load_best_candidate(tuning_dir: Path, ranking_row: int, expected_model: str) -> dict:
    ranking_path = tuning_dir / "gbm_tuning_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing ranking file: {ranking_path}")

    ranking = pd.read_csv(ranking_path)
    if ranking.empty:
        raise ValueError(f"Ranking file is empty: {ranking_path}")

    if "validation_rank" in ranking.columns:
        row = ranking[ranking["validation_rank"].astype(int) == int(ranking_row)]
        if len(row) == 0:
            row = ranking.sort_values("validation_rank").head(1)
    else:
        row = ranking.head(ranking_row).tail(1)

    rec = row.iloc[0].to_dict()
    model_name = str(rec.get("model_name", "")).lower()
    if model_name != expected_model:
        raise ValueError(f"Expected {expected_model}, but ranking row is {model_name} in {ranking_path}")

    params = json.loads(rec["params_json"]) if isinstance(rec.get("params_json"), str) else {}
    rec["params"] = params
    return rec


def find_existing_prediction_file(tuning_dir: Path, candidate_id: str) -> Optional[Path]:
    # Created by tune_gbm_hyperparams.py when --save-predictions-for-best is used.
    candidates = sorted(tuning_dir.glob(f"predictions_best_{candidate_id}.csv.gz"))
    if candidates:
        return candidates[0]

    # Fallback: any predictions_best file if exact candidate not found.
    candidates = sorted(tuning_dir.glob("predictions_best_*.csv.gz"))
    if candidates:
        return candidates[0]

    return None


def score_column_from_existing_predictions(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None

    required = {"split", "y_true", "y_score", CATEGORY_COL}
    if not required.issubset(df.columns):
        return None

    keep = [c for c in ID_COLS if c in df.columns]
    out = df[keep + ["split", "y_true", "y_score"]].copy()
    return out


def make_lgbm_model(params: dict, pos_weight: float, args: argparse.Namespace):
    if not HAS_LIGHTGBM:
        raise ModuleNotFoundError(
            "lightgbm is not installed. Install: .\\.venv\\Scripts\\python.exe -m pip install lightgbm"
        )

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
    if args.lightgbm_use_gpu:
        model_params["device_type"] = "gpu"
        model_params["gpu_device_id"] = int(args.gpu_device_id)
        if int(args.lightgbm_gpu_platform_id) >= 0:
            model_params["gpu_platform_id"] = int(args.lightgbm_gpu_platform_id)

    return LGBMClassifier(**model_params)


def make_xgb_model(params: dict, pos_weight: float, args: argparse.Namespace):
    if not HAS_XGBOOST:
        raise ModuleNotFoundError(
            "xgboost is not installed. Install: .\\.venv\\Scripts\\python.exe -m pip install xgboost"
        )

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

    if args.xgboost_use_gpu:
        if args.xgboost_gpu_mode == "legacy":
            model_params["tree_method"] = "gpu_hist"
            model_params["gpu_id"] = int(args.gpu_device_id)
        else:
            model_params["tree_method"] = "hist"
            model_params["device"] = f"cuda:{int(args.gpu_device_id)}"
    else:
        model_params["tree_method"] = "hist"

    return XGBClassifier(**model_params)


def predict_score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def rebuild_scores_for_model(
    *,
    model_name: str,
    best: dict,
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    args: argparse.Namespace,
    progress: bool,
) -> Tuple[pd.DataFrame, dict]:
    feature_cols = numeric_features + categorical_features
    params = best["params"]
    pos_weight = float(best.get("pos_weight", 1.0))
    candidate_id = str(best["candidate_id"])

    X_train_raw, y_train, _ = get_split(df, args.experiment, "train", feature_cols)
    train_rows_full = len(y_train)
    sampled_idx = stratified_sample_indices(y_train, args.max_train_rows, args.random_state)
    X_train_raw = X_train_raw.loc[sampled_idx]
    y_train = y_train.loc[sampled_idx]

    if progress:
        print(
            f"[rebuild] {model_name} candidate={candidate_id}\n"
            f"          train_rows={len(y_train):,}/{train_rows_full:,}, "
            f"pos_share={y_train.mean():.4f}, pos_weight={pos_weight:.4f}"
        )

    pre = FastTabularPreprocessor(numeric_features, categorical_features)
    t_pre = time.time()
    X_train = pre.fit_transform(X_train_raw)
    preprocess_seconds = time.time() - t_pre

    if model_name == "lightgbm":
        model = make_lgbm_model(params, pos_weight, args)
    elif model_name == "xgboost":
        model = make_xgb_model(params, pos_weight, args)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t_fit = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - t_fit

    pred_parts = []
    for split in ["validation", "test"]:
        X_raw, y_true, mask = get_split(df, args.experiment, split, feature_cols)
        X_split = pre.transform(X_raw)
        scores = predict_score(model, X_split)

        keep = [c for c in ID_COLS if c in df.columns]
        pred = df.loc[mask, keep].copy()
        pred["split"] = split
        pred["y_true"] = y_true.values.astype(int)
        pred["y_score"] = scores.astype(float)
        pred_parts.append(pred)

        del X_raw, X_split, y_true
        gc.collect()

    pred_df = pd.concat(pred_parts, ignore_index=True)

    info = {
        "candidate_id": candidate_id,
        "model_name": model_name,
        "param_id": best.get("param_id"),
        "pos_weight_mode": best.get("pos_weight_mode"),
        "pos_weight": pos_weight,
        "params": params,
        "train_rows": int(len(y_train)),
        "train_rows_full": int(train_rows_full),
        "input_feature_count": int(len(feature_cols)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "input_feature_names": feature_cols,
        "model_feature_count": int(len(pre.feature_names_)),
        "model_feature_names": list(pre.feature_names_),
        "preprocess_seconds": float(preprocess_seconds),
        "fit_seconds": float(fit_seconds),
        "warnings_count": int(len(caught)),
        "warnings_preview": " | ".join(str(w.message) for w in caught[:5]),
        "score_source": "rebuilt_from_best_candidate",
    }

    del X_train_raw, X_train, y_train, model
    gc.collect()

    return pred_df, info


def load_or_rebuild_scores(
    *,
    model_name: str,
    tuning_dir: Path,
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    args: argparse.Namespace,
    progress: bool,
) -> Tuple[pd.DataFrame, dict]:
    best = load_best_candidate(tuning_dir, args.ranking_row, expected_model=model_name)
    candidate_id = str(best["candidate_id"])

    pred_path = None if args.force_rebuild_scores else find_existing_prediction_file(tuning_dir, candidate_id)
    if pred_path is not None:
        pred = score_column_from_existing_predictions(pred_path)
        if pred is not None:
            if progress:
                print(f"[load-score] {model_name}: loaded existing predictions: {pred_path}")
            info = {
                "candidate_id": candidate_id,
                "model_name": model_name,
                "param_id": best.get("param_id"),
                "pos_weight_mode": best.get("pos_weight_mode"),
                "pos_weight": float(best.get("pos_weight", 1.0)),
        "params": best.get("params", {}),
        "input_feature_count": int(len(numeric_features) + len(categorical_features)),
        "input_feature_names": list(numeric_features) + list(categorical_features),
        "score_source": str(pred_path),
            }
            return pred, info

    if progress:
        print(f"[load-score] {model_name}: no existing prediction file found; rebuilding model scores.")

    return rebuild_scores_for_model(
        model_name=model_name,
        best=best,
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        args=args,
        progress=progress,
    )


# ---------------------------------------------------------------------
# Metrics and thresholds
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


def evaluate_predictions(y_true: Iterable[int], y_score: Iterable[float], y_pred: Iterable[int]) -> Tuple[dict, dict]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_score_arr = np.nan_to_num(y_score_arr, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred_arr = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    metrics = {
        "rows": int(len(y_true_arr)),
        "positive_rows": int(y_true_arr.sum()),
        "positive_share": float(y_true_arr.mean()) if len(y_true_arr) else np.nan,
        "predicted_positive_rows": int(y_pred_arr.sum()),
        "predicted_positive_share": float(y_pred_arr.mean()) if len(y_pred_arr) else np.nan,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "specificity": float(specificity) if not pd.isna(specificity) else np.nan,
        "npv": float(npv) if not pd.isna(npv) else np.nan,
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f2": float(fbeta_score(y_true_arr, y_pred_arr, beta=2.0, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true_arr, y_score_arr),
        "pr_auc": safe_pr_auc(y_true_arr, y_score_arr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    conf = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics, conf


def evaluate_at_threshold(y_true: Iterable[int], y_score: Iterable[float], threshold: float) -> Tuple[dict, dict]:
    score = np.asarray(y_score, dtype=float)
    pred = (score >= threshold).astype(int)
    metrics, conf = evaluate_predictions(y_true, score, pred)
    metrics["threshold"] = float(threshold)
    conf["threshold"] = float(threshold)
    return metrics, conf


def build_threshold_candidates(scores: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    base = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    base = base[(base >= 0) & (base <= 1)]
    extra = np.array([0.5])

    if args.quantile_thresholds and args.quantile_thresholds > 0:
        valid = scores[np.isfinite(scores)]
        if len(valid):
            qs = np.linspace(0.01, 0.99, args.quantile_thresholds)
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


def pick_threshold(grid: pd.DataFrame, strategy: str, args: argparse.Namespace) -> Tuple[float, str]:
    if grid.empty:
        return 0.5, "empty_grid_fallback_0_5"

    if strategy == "best_f1":
        row = grid.loc[grid["f1"].idxmax()]
        return float(row["threshold"]), "Max validation F1."

    if strategy == "best_f2":
        row = grid.loc[grid["f2"].idxmax()]
        return float(row["threshold"]), "Max validation F2."

    if strategy == "best_balanced_accuracy":
        row = grid.loc[grid["balanced_accuracy"].idxmax()]
        return float(row["threshold"]), "Max validation balanced accuracy."

    if strategy == "best_precision_at_min_recall":
        feasible = grid[grid["recall"] >= args.min_recall]
        if len(feasible):
            row = feasible.sort_values(["precision", "f1", "balanced_accuracy"], ascending=[False, False, False]).iloc[0]
            return float(row["threshold"]), f"Max precision subject to recall >= {args.min_recall}."
        row = grid.loc[grid["f1"].idxmax()]
        return float(row["threshold"]), "Constraint infeasible; fallback best_f1."

    if strategy == "best_recall_at_min_precision":
        feasible = grid[grid["precision"] >= args.min_precision]
        if len(feasible):
            row = feasible.sort_values(["recall", "f1", "balanced_accuracy"], ascending=[False, False, False]).iloc[0]
            return float(row["threshold"]), f"Max recall subject to precision >= {args.min_precision}."
        row = grid.loc[grid["f1"].idxmax()]
        return float(row["threshold"]), "Constraint infeasible; fallback best_f1."

    raise ValueError(f"Unknown threshold strategy: {strategy}")


def tune_global_threshold(scored: pd.DataFrame, score_col: str, args: argparse.Namespace) -> Tuple[float, pd.DataFrame, dict]:
    val = scored[scored["split"] == "validation"]
    y = val["y_true"].values.astype(int)
    s = val[score_col].values.astype(float)
    thresholds = build_threshold_candidates(s, args)
    grid = threshold_grid(y, s, thresholds)
    selected, note = pick_threshold(grid, args.primary_strategy, args)
    selected_metrics, _ = evaluate_at_threshold(y, s, selected)
    selected_metrics["threshold_note"] = note
    return selected, grid, selected_metrics


def tune_category_thresholds(
    scored: pd.DataFrame,
    score_col: str,
    global_threshold: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    val = scored[scored["split"] == "validation"].copy()
    rows = []

    categories = sorted(val[CATEGORY_COL].dropna().astype(str).unique().tolist())

    for cat in categories:
        sub = val[val[CATEGORY_COL].astype(str) == cat]
        y = sub["y_true"].values.astype(int)
        s = sub[score_col].values.astype(float)
        positives = int(y.sum())
        rows_count = int(len(y))

        if rows_count == 0 or len(np.unique(y)) < 2 or positives < args.min_category_positive:
            threshold = float(global_threshold)
            note = (
                f"Fallback to global threshold; rows={rows_count}, "
                f"positives={positives}, min_positive={args.min_category_positive}."
            )
            m, _ = evaluate_at_threshold(y, s, threshold) if rows_count else ({}, {})
        else:
            thresholds = build_threshold_candidates(s, args)
            grid = threshold_grid(y, s, thresholds)
            threshold, note = pick_threshold(grid, args.primary_strategy, args)
            m, _ = evaluate_at_threshold(y, s, threshold)

        row = {
            "complaint_category": cat,
            "threshold": float(threshold),
            "threshold_strategy": args.primary_strategy,
            "note": note,
            "validation_rows": rows_count,
            "validation_positive_rows": positives,
        }
        if m:
            row.update({
                "validation_f1": m["f1"],
                "validation_precision": m["precision"],
                "validation_recall": m["recall"],
                "validation_pr_auc": m["pr_auc"],
                "validation_roc_auc": m["roc_auc"],
                "validation_predicted_positive_share": m["predicted_positive_share"],
            })
        rows.append(row)

    return pd.DataFrame(rows)


def apply_category_thresholds(scored: pd.DataFrame, score_col: str, threshold_df: pd.DataFrame, fallback: float) -> np.ndarray:
    thresholds = {
        str(row["complaint_category"]): float(row["threshold"])
        for _, row in threshold_df.iterrows()
    }
    cats = scored[CATEGORY_COL].astype(str)
    t = cats.map(thresholds).fillna(float(fallback)).astype(float).to_numpy()
    s = scored[score_col].astype(float).to_numpy()
    return (s >= t).astype(int)


def summarize_score_distribution(scored: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    rows = []
    for score_col in score_cols:
        for split, sub in scored.groupby("split"):
            s = sub[score_col].astype(float)
            rows.append({
                "score_col": score_col,
                "split": split,
                "rows": int(len(sub)),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q01": float(s.quantile(0.01)),
                "q05": float(s.quantile(0.05)),
                "q25": float(s.quantile(0.25)),
                "median": float(s.quantile(0.50)),
                "q75": float(s.quantile(0.75)),
                "q95": float(s.quantile(0.95)),
                "q99": float(s.quantile(0.99)),
                "max": float(s.max()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Build comparison
# ---------------------------------------------------------------------

def combine_scores(lgbm_pred: pd.DataFrame, xgb_pred: pd.DataFrame) -> pd.DataFrame:
    # Preserve row order from LightGBM predictions. Because both predictions are
    # generated from the same final dataset/masks, row count and key order should match.
    if len(lgbm_pred) != len(xgb_pred):
        raise ValueError(f"Prediction row mismatch: lgbm={len(lgbm_pred)}, xgb={len(xgb_pred)}")

    key_cols = [c for c in ["split", "nta2020", WEEK_COL, CATEGORY_COL, "y_true"] if c in lgbm_pred.columns and c in xgb_pred.columns]
    for col in key_cols:
        left = lgbm_pred[col].astype(str).to_numpy()
        right = xgb_pred[col].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(
                f"Prediction order/key mismatch on column {col}. "
                "Regenerate predictions from the same final dataset and experiment."
            )

    scored = lgbm_pred.copy()
    scored = scored.drop(columns=["y_score"], errors="ignore")
    scored["lgbm_score"] = lgbm_pred["y_score"].astype(float).to_numpy()
    scored["xgb_score"] = xgb_pred["y_score"].astype(float).to_numpy()

    if "y_true" not in scored.columns:
        scored["y_true"] = lgbm_pred["y_true"].astype(int).to_numpy()

    return scored


def add_ensemble_scores(scored: pd.DataFrame, weights: List[float]) -> List[str]:
    score_cols = ["lgbm_score", "xgb_score"]

    clean_weights = []
    for w in weights:
        if w < 0 or w > 1:
            raise ValueError(f"Ensemble weight must be between 0 and 1: {w}")
        clean_weights.append(float(w))

    for w in clean_weights:
        suffix = f"{w:.3f}".replace(".", "p")
        col = f"ensemble_lgbm_w{suffix}_score"
        scored[col] = w * scored["lgbm_score"].astype(float) + (1.0 - w) * scored["xgb_score"].astype(float)
        score_cols.append(col)

    return score_cols


def nice_model_label(score_col: str) -> str:
    if score_col == "lgbm_score":
        return "Tuned LightGBM"
    if score_col == "xgb_score":
        return "Tuned XGBoost"
    if score_col.startswith("ensemble_"):
        middle = score_col.replace("_score", "")
        w_part = middle.split("lgbm_w")[-1].replace("p", ".")
        try:
            w = float(w_part)
            return f"Ensemble LGBM({w:.2f}) + XGB({1-w:.2f})"
        except Exception:
            return "Ensemble LightGBM + XGBoost"
    return score_col


def evaluate_method(
    *,
    scored: pd.DataFrame,
    score_col: str,
    threshold_mode: str,
    global_threshold: float,
    category_thresholds: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> Tuple[List[dict], List[dict], List[dict]]:
    metric_rows = []
    conf_rows = []
    by_category_rows = []

    for split in ["validation", "test"]:
        sub = scored[scored["split"] == split].copy()
        y = sub["y_true"].values.astype(int)
        s = sub[score_col].values.astype(float)

        if threshold_mode == "global":
            y_pred = (s >= global_threshold).astype(int)
            threshold_value = float(global_threshold)
        elif threshold_mode == "per_category":
            if category_thresholds is None:
                raise ValueError("category_thresholds required for per_category mode")
            y_pred = apply_category_thresholds(sub, score_col, category_thresholds, fallback=global_threshold)
            threshold_value = np.nan
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

        m, c = evaluate_predictions(y, s, y_pred)
        method_id = f"{score_col.replace('_score', '')}_{threshold_mode}"
        common = {
            "method_id": method_id,
            "model_label": nice_model_label(score_col),
            "score_col": score_col,
            "threshold_mode": threshold_mode,
            "threshold_strategy": args.primary_strategy,
            "split": split,
            "global_threshold": threshold_value,
        }
        metric_rows.append({**common, **m})
        conf_rows.append({**common, **c})

        for cat, g in sub.assign(y_pred=y_pred).groupby(CATEGORY_COL):
            ym = g["y_true"].values.astype(int)
            sm = g[score_col].values.astype(float)
            pm = g["y_pred"].values.astype(int)
            mm, cc = evaluate_predictions(ym, sm, pm)
            by_category_rows.append({
                **common,
                "complaint_category": cat,
                **mm,
            })

    return metric_rows, conf_rows, by_category_rows


def build_wide_comparison(long_metrics: pd.DataFrame) -> pd.DataFrame:
    # One row per method_id with validation/test metrics in wide format.
    key_cols = [
        "method_id",
        "model_label",
        "score_col",
        "threshold_mode",
        "threshold_strategy",
    ]
    metric_cols = [
        "rows",
        "positive_rows",
        "positive_share",
        "predicted_positive_rows",
        "predicted_positive_share",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "npv",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "global_threshold",
    ]

    rows = []
    for method_id, g in long_metrics.groupby("method_id"):
        base = {c: g.iloc[0][c] for c in key_cols if c in g.columns}
        for split in ["validation", "test"]:
            sub = g[g["split"] == split]
            if sub.empty:
                continue
            rec = sub.iloc[0]
            prefix = "val" if split == "validation" else "test"
            for c in metric_cols:
                if c in rec.index:
                    base[f"{prefix}_{c}"] = rec[c]
        rows.append(base)

    out = pd.DataFrame(rows)
    if not out.empty:
        sort_cols = [c for c in ["val_f1", "val_pr_auc", "val_balanced_accuracy", "test_f1"] if c in out.columns]
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
        out.insert(0, "validation_rank", np.arange(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    project_root = get_project_root()
    final_path = resolve_path(project_root, args.final_dataset, DEFAULT_FINAL_REL)
    config_path = resolve_path(project_root, args.config, DEFAULT_CONFIG_REL)
    lgbm_dir = resolve_path(project_root, args.lgbm_tuning_dir, DEFAULT_LGBM_DIR_REL)
    xgb_dir = resolve_path(project_root, args.xgb_tuning_dir, DEFAULT_XGB_DIR_REL)
    output_dir = resolve_path(project_root, args.output_dir, DEFAULT_OUTPUT_DIR_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not final_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {final_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found: {config_path}")
    if not lgbm_dir.exists():
        raise FileNotFoundError(f"LightGBM tuning dir not found: {lgbm_dir}")
    if not xgb_dir.exists():
        raise FileNotFoundError(f"XGBoost tuning dir not found: {xgb_dir}")

    if args.progress:
        print(f"[setup] final_path={final_path}")
        print(f"[setup] config_path={config_path}")
        print(f"[setup] lgbm_dir={lgbm_dir}")
        print(f"[setup] xgb_dir={xgb_dir}")
        print(f"[setup] output_dir={output_dir}")

    config = load_json(config_path)
    df, numeric_features, categorical_features = load_dataset(final_path, config, args.feature_set, args.progress)

    lgbm_pred, lgbm_info = load_or_rebuild_scores(
        model_name="lightgbm",
        tuning_dir=lgbm_dir,
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        args=args,
        progress=args.progress,
    )

    xgb_pred, xgb_info = load_or_rebuild_scores(
        model_name="xgboost",
        tuning_dir=xgb_dir,
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        args=args,
        progress=args.progress,
    )

    scored = combine_scores(lgbm_pred, xgb_pred)
    score_cols = add_ensemble_scores(scored, args.ensemble_weights)

    # Make sure score columns are bounded valid probabilities.
    for col in score_cols:
        scored[col] = pd.to_numeric(scored[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    if args.progress:
        print(f"[scores] rows={len(scored):,}, score_cols={score_cols}")

    threshold_selection_rows = []
    all_grid_rows = []
    all_category_threshold_rows = []
    all_metric_rows = []
    all_conf_rows = []
    all_by_category_rows = []

    for score_col in score_cols:
        if args.progress:
            print("=" * 90)
            print(f"[threshold] score_col={score_col} ({nice_model_label(score_col)})")
            print("=" * 90)

        global_threshold, grid, selected_metrics = tune_global_threshold(scored, score_col, args)

        grid["score_col"] = score_col
        grid["model_label"] = nice_model_label(score_col)
        all_grid_rows.append(grid)

        threshold_selection_rows.append({
            "score_col": score_col,
            "model_label": nice_model_label(score_col),
            "threshold_mode": "global",
            "threshold_strategy": args.primary_strategy,
            "selected_threshold": global_threshold,
            "selection_note": selected_metrics.get("threshold_note", ""),
            "validation_f1": selected_metrics.get("f1"),
            "validation_precision": selected_metrics.get("precision"),
            "validation_recall": selected_metrics.get("recall"),
            "validation_f2": selected_metrics.get("f2"),
            "validation_balanced_accuracy": selected_metrics.get("balanced_accuracy"),
            "validation_pr_auc": selected_metrics.get("pr_auc"),
            "validation_roc_auc": selected_metrics.get("roc_auc"),
            "validation_predicted_positive_share": selected_metrics.get("predicted_positive_share"),
        })

        cat_thresholds = tune_category_thresholds(
            scored=scored,
            score_col=score_col,
            global_threshold=global_threshold,
            args=args,
        )
        cat_thresholds.insert(0, "score_col", score_col)
        cat_thresholds.insert(1, "model_label", nice_model_label(score_col))
        cat_thresholds.insert(2, "global_threshold_fallback", global_threshold)
        all_category_threshold_rows.append(cat_thresholds)

        # Evaluate global and per-category.
        for mode in ["global", "per_category"]:
            metric_rows, conf_rows, by_cat_rows = evaluate_method(
                scored=scored,
                score_col=score_col,
                threshold_mode=mode,
                global_threshold=global_threshold,
                category_thresholds=cat_thresholds,
                args=args,
            )
            all_metric_rows.extend(metric_rows)
            all_conf_rows.extend(conf_rows)
            all_by_category_rows.extend(by_cat_rows)

        if args.progress:
            print(
                f"[threshold] {score_col} global_threshold={global_threshold:.6f} | "
                f"val_f1={selected_metrics['f1']:.4f}, "
                f"val_precision={selected_metrics['precision']:.4f}, "
                f"val_recall={selected_metrics['recall']:.4f}, "
                f"val_pr_auc={selected_metrics['pr_auc']:.4f}"
            )

    threshold_selection_df = pd.DataFrame(threshold_selection_rows)
    threshold_grid_df = pd.concat(all_grid_rows, ignore_index=True) if all_grid_rows else pd.DataFrame()
    category_thresholds_df = pd.concat(all_category_threshold_rows, ignore_index=True) if all_category_threshold_rows else pd.DataFrame()
    long_metrics_df = pd.DataFrame(all_metric_rows)
    conf_df = pd.DataFrame(all_conf_rows)
    by_category_df = pd.DataFrame(all_by_category_rows)
    wide_comparison_df = build_wide_comparison(long_metrics_df)
    score_summary_df = summarize_score_distribution(scored, score_cols)

    # Output files.
    write_csv(output_dir / "final_model_comparison.csv", wide_comparison_df, args.overwrite)
    write_csv(output_dir / "final_model_comparison_long.csv", long_metrics_df, args.overwrite)
    write_csv(output_dir / "threshold_selection_summary.csv", threshold_selection_df, args.overwrite)
    write_csv(output_dir / "threshold_grid_validation.csv", threshold_grid_df, args.overwrite)
    write_csv(output_dir / "category_thresholds.csv", category_thresholds_df, args.overwrite)
    write_csv(output_dir / "metrics_by_category.csv", by_category_df, args.overwrite)
    write_csv(output_dir / "confusion_matrices.csv", conf_df, args.overwrite)
    write_csv(output_dir / "score_summary.csv", score_summary_df, args.overwrite)

    if args.save_scores:
        score_path = output_dir / "scored_validation_test.csv.gz"
        ensure_output_path(score_path, args.overwrite)
        scored.to_csv(score_path, index=False, compression="gzip")

    best_row = wide_comparison_df.head(1).to_dict(orient="records")[0] if not wide_comparison_df.empty else None

    summary = {
        "script": SCRIPT_NAME,
        "status": "done",
        "final_dataset": str(final_path),
        "config": str(config_path),
        "lgbm_tuning_dir": str(lgbm_dir),
        "xgb_tuning_dir": str(xgb_dir),
        "output_dir": str(output_dir),
        "experiment": args.experiment,
        "feature_set": args.feature_set,
        "analysis_type": args.analysis_type,
        "ranking_row": int(args.ranking_row),
        "primary_strategy": args.primary_strategy,
        "ensemble_weights": args.ensemble_weights,
        "score_columns": score_cols,
        "scored_rows": int(len(scored)),
        "validation_rows": int((scored["split"] == "validation").sum()),
        "test_rows": int((scored["split"] == "test").sum()),
        "validation_positive_share": float(scored.loc[scored["split"] == "validation", "y_true"].mean()),
        "test_positive_share": float(scored.loc[scored["split"] == "test", "y_true"].mean()),
        "lgbm_info": lgbm_info,
        "xgb_info": xgb_info,
        "best_method_by_validation": best_row,
        "top_methods_by_validation": wide_comparison_df.head(20).to_dict(orient="records") if not wide_comparison_df.empty else [],
        "xgboost_use_gpu": bool(args.xgboost_use_gpu),
        "lightgbm_use_gpu": bool(args.lightgbm_use_gpu),
        "max_train_rows": int(args.max_train_rows),
        "elapsed_seconds": round(time.time() - t0, 3),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json(output_dir / "ensemble_threshold_run_summary.json", summary, args.overwrite)

    if args.progress:
        print("\n[final ranking]")
        if not wide_comparison_df.empty:
            cols = [
                "validation_rank",
                "method_id",
                "model_label",
                "threshold_mode",
                "val_f1",
                "val_precision",
                "val_recall",
                "val_pr_auc",
                "test_f1",
                "test_precision",
                "test_recall",
                "test_pr_auc",
                "test_balanced_accuracy",
            ]
            print(wide_comparison_df[cols].to_string(index=False))
        else:
            print("(empty)")

    print("=" * 90)
    print(f"[{SCRIPT_NAME}] DONE")
    print(f"Final comparison: {output_dir / 'final_model_comparison.csv'}")
    print(f"Threshold summary: {output_dir / 'threshold_selection_summary.csv'}")
    print(f"Category thresholds: {output_dir / 'category_thresholds.csv'}")
    print(f"Run summary: {output_dir / 'ensemble_threshold_run_summary.json'}")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        raise
