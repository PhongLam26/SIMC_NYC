#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10.6 — SHAP explainability for the final SIMC NYC model pipeline.

Project:
    SIMC NYC — Semantics-Aware Explainable Machine Learning for Urban Service
    Demand Forecasting in Smart Cities.

Purpose:
    Produce paper-ready explainability artifacts after the final decision-layer
    experiment.

Recommended interpretation protocol:
    - The final predictor can be Ensemble LightGBM + XGBoost with per-category
      thresholds.
    - For explainability, use Tuned LightGBM as the explanation backbone because
      it is a strong component model and SHAP supports tree models directly.
    - Do not claim causality. SHAP explains model predictions/associations.

Main outputs:
    shap_global_importance.csv
    shap_group_importance.csv
    shap_importance_by_complaint_category.csv
    shap_local_cases.csv
    shap_local_case_feature_contributions.csv
    shap_summary_bar.png
    shap_beeswarm.png
    shap_dependence_top*.png
    shap_local_case_*.png
    shap_explainability_report.md
    shap_run_summary.json

Install dependency if needed:
    .\\.venv\\Scripts\\python.exe -m pip install shap

Recommended command:
    .\\.venv\\Scripts\\python.exe .\\code_model\\run_shap_explainability.py --overwrite --progress ^
        --lgbm-tuning-dir data/processed/model_results/gbm_hyperparam_tuning_lgbm_balanced_gpu ^
        --ensemble-dir data/processed/model_results/ensemble_category_thresholds ^
        --max-shap-rows 25000 ^
        --max-category-rows 4000 ^
        --local-cases-per-type 5 ^
        --output-dir data/processed/model_results/shap_explainability_lgbm

Faster smoke test:
    .\\.venv\\Scripts\\python.exe .\\code_model\\run_shap_explainability.py --overwrite --progress ^
        --max-shap-rows 5000 ^
        --max-category-rows 1000 ^
        --local-cases-per-type 3 ^
        --output-dir data/processed/model_results/shap_explainability_lgbm_smoke
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    print(
        "\n[run_shap_explainability] Missing matplotlib.\n"
        "Install into project .venv:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install matplotlib\n",
        file=sys.stderr,
    )
    raise exc

try:
    import shap
except ModuleNotFoundError as exc:
    print(
        "\n[run_shap_explainability] Missing SHAP package.\n"
        "Install into project .venv:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install shap\n",
        file=sys.stderr,
    )
    raise exc

try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    print(
        "\n[run_shap_explainability] Missing LightGBM package.\n"
        "Install into project .venv:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install lightgbm\n",
        file=sys.stderr,
    )
    raise exc

try:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        confusion_matrix,
    )
except ModuleNotFoundError as exc:
    print(
        "\n[run_shap_explainability] Missing scikit-learn.\n"
        "Install into project .venv:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn\n",
        file=sys.stderr,
    )
    raise exc


SCRIPT_NAME = "run_shap_explainability"

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
DEFAULT_ENSEMBLE_DIR_REL = Path("data/processed/model_results/ensemble_category_thresholds")
DEFAULT_OUTPUT_DIR_REL = Path("data/processed/model_results/shap_explainability_lgbm")

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
# CLI / IO
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP explainability for tuned LightGBM.")

    parser.add_argument("--final-dataset", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--lgbm-tuning-dir", type=str, default=None)
    parser.add_argument("--ensemble-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--experiment", type=str, default="E1_main_2015_2025")
    parser.add_argument("--feature-set", type=str, default="full_without_covid_period_features")
    parser.add_argument("--ranking-row", type=int, default=1)

    parser.add_argument("--max-shap-rows", type=int, default=25000,
                        help="Maximum test rows used for global SHAP.")
    parser.add_argument("--max-category-rows", type=int, default=4000,
                        help="Maximum rows per complaint_category for category-level SHAP summaries.")
    parser.add_argument("--local-cases-per-type", type=int, default=5,
                        help="Number of TP/FP/FN/TN local cases to explain.")
    parser.add_argument("--top-n-features", type=int, default=30)
    parser.add_argument("--top-n-dependence", type=int, default=8)
    parser.add_argument("--local-top-n", type=int, default=12)

    parser.add_argument("--threshold", type=float, default=None,
                        help="Override LightGBM threshold. Default uses selected_threshold from gbm_tuning_ranking.csv.")
    parser.add_argument("--sample-random-state", type=int, default=42)
    parser.add_argument("--analysis-type", type=str, default="unspecified",
                        choices=["unspecified", "prospective", "retrospective_context"],
                        help="Analysis protocol identifier written to output metadata.")

    parser.add_argument("--lightgbm-use-gpu", action="store_true",
                        help="Use LightGBM OpenCL GPU. On Windows this may pick Intel iGPU.")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--lightgbm-gpu-platform-id", type=int, default=-1)

    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")

    return parser.parse_args()


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


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    path.write_text(text, encoding="utf-8")


def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    if pd.isna(x):
        return None
    return x


def write_json(path: Path, data: dict, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=to_jsonable)


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


def stratified_sample(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int,
    strata_cols: List[str],
) -> pd.DataFrame:
    if not max_rows or max_rows <= 0 or max_rows >= len(df):
        return df.copy()

    if not strata_cols:
        return df.sample(n=max_rows, random_state=random_state).copy()

    tmp = df.copy()
    for col in strata_cols:
        if col not in tmp.columns:
            tmp[col] = "__NA__"
    tmp["_stratum"] = tmp[strata_cols].astype(str).agg("|".join, axis=1)

    sampled_parts = []
    for _, g in tmp.groupby("_stratum"):
        n_take = max(1, int(round(max_rows * len(g) / len(tmp))))
        n_take = min(n_take, len(g))
        sampled_parts.append(g.sample(n=n_take, random_state=random_state))

    out = pd.concat(sampled_parts, ignore_index=False)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state)
    return out.drop(columns=["_stratum"], errors="ignore").copy()


# ---------------------------------------------------------------------
# Model rebuild
# ---------------------------------------------------------------------

def load_best_lgbm_candidate(tuning_dir: Path, ranking_row: int) -> dict:
    ranking_path = tuning_dir / "gbm_tuning_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing LightGBM ranking file: {ranking_path}")

    ranking = pd.read_csv(ranking_path)
    if ranking.empty:
        raise ValueError(f"Empty ranking file: {ranking_path}")

    if "validation_rank" in ranking.columns:
        row = ranking[ranking["validation_rank"].astype(int) == int(ranking_row)]
        if row.empty:
            row = ranking.sort_values("validation_rank").head(1)
    else:
        row = ranking.head(ranking_row).tail(1)

    rec = row.iloc[0].to_dict()
    if str(rec.get("model_name", "")).lower() != "lightgbm":
        raise ValueError(f"Expected LightGBM candidate, got {rec.get('model_name')}")

    rec["params"] = json.loads(rec["params_json"]) if isinstance(rec.get("params_json"), str) else {}
    return rec


def make_lgbm_model(candidate: dict, args: argparse.Namespace) -> LGBMClassifier:
    params = candidate["params"]
    pos_weight = float(candidate.get("pos_weight", 1.0))

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
        random_state=args.sample_random_state,
        verbose=-1,
    )

    if args.lightgbm_use_gpu:
        model_params["device_type"] = "gpu"
        model_params["gpu_device_id"] = int(args.gpu_device_id)
        if int(args.lightgbm_gpu_platform_id) >= 0:
            model_params["gpu_platform_id"] = int(args.lightgbm_gpu_platform_id)

    return LGBMClassifier(**model_params)


def rebuild_lgbm(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    candidate: dict,
    args: argparse.Namespace,
) -> Tuple[LGBMClassifier, FastTabularPreprocessor, pd.DataFrame, pd.DataFrame, dict]:
    feature_cols = numeric_features + categorical_features

    X_train_raw, y_train, _ = get_split(df, args.experiment, "train", feature_cols)

    if args.progress:
        print(
            f"[train] LightGBM candidate={candidate.get('candidate_id')}\n"
            f"        param_id={candidate.get('param_id')}, pos_weight_mode={candidate.get('pos_weight_mode')}, "
            f"threshold={candidate.get('selected_threshold')}"
        )
        print(f"[train] rows={len(y_train):,}, pos_share={y_train.mean():.4f}")

    pre = FastTabularPreprocessor(numeric_features, categorical_features)

    t_pre = time.time()
    X_train = pre.fit_transform(X_train_raw)
    pre_seconds = time.time() - t_pre

    model = make_lgbm_model(candidate, args)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t_fit = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - t_fit

    # Build full validation/test scored frame.
    scored_parts = []
    for split in ["validation", "test"]:
        X_raw, y_true, mask = get_split(df, args.experiment, split, feature_cols)
        X_split = pre.transform(X_raw)
        scores = model.predict_proba(X_split)[:, 1]

        keep = [c for c in ID_COLS if c in df.columns]
        scored = df.loc[mask, keep].copy()
        scored["split"] = split
        scored["y_true"] = y_true.values.astype(int)
        scored["lgbm_score"] = scores.astype(float)
        scored_parts.append(scored)

        del X_raw, X_split, y_true
        gc.collect()

    scored_df = pd.concat(scored_parts, ignore_index=True)

    info = {
        "train_rows": int(len(y_train)),
        "train_positive_share": float(y_train.mean()),
        "preprocess_seconds": float(pre_seconds),
        "fit_seconds": float(fit_seconds),
        "warnings_count": int(len(caught)),
        "warnings_preview": " | ".join(str(w.message) for w in caught[:5]),
    }

    del X_train_raw, X_train, y_train
    gc.collect()

    return model, pre, scored_df, pd.DataFrame(), info


# ---------------------------------------------------------------------
# SHAP utilities
# ---------------------------------------------------------------------

def feature_group(feature: str) -> str:
    f = str(feature).lower()

    if f in {"complaint_count"}:
        return "current_demand"
    if f.startswith("complaint_category=") or f == "complaint_category":
        return "semantic_category"
    if f.startswith("boroname=") or f == "boroname" or "nta" in f or "boro" in f:
        return "spatial_context"
    if f.startswith("period_type=") or f == "period_type" or "covid" in f or "period" in f:
        return "period_context"
    if f.startswith("lag_") or f.startswith("rolling_") or f.startswith("ratio_to_") or f.startswith("diff_") or f.startswith("pct_change") or "history" in f:
        return "historical_temporal"
    if f.startswith("weather_") or f.startswith("temp_") or "prcp" in f or "snow" in f or "tmax" in f or "tmin" in f or "awnd" in f:
        return "weather"
    if f.startswith("osm_") or f.startswith("poi_") or "poi" in f:
        return "osm_poi_context"
    if f.startswith("pluto_") or "landuse" in f or "bldg" in f or "lot" in f:
        return "pluto_built_environment"
    if f in {"week_of_year", "month", "quarter", "year"} or "week" in f and "lag" not in f and "rolling" not in f:
        return "calendar"
    if "flag" in f or "missing" in f or "quality" in f:
        return "quality_flag"
    return "other"


def get_positive_class_shap_values(shap_values) -> np.ndarray:
    # shap.TreeExplainer output varies by SHAP/LightGBM version.
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # Common shape: (n_samples, n_features, n_outputs)
        if arr.shape[2] >= 2:
            return arr[:, :, 1]
        return arr[:, :, 0]
    return arr


def get_expected_value(explainer) -> float:
    ev = explainer.expected_value
    if isinstance(ev, list):
        if len(ev) == 2:
            return float(ev[1])
        return float(ev[0])
    arr = np.asarray(ev)
    if arr.size > 1:
        return float(arr.ravel()[-1])
    return float(arr.ravel()[0])


def compute_shap_values(model, X: pd.DataFrame, progress: bool) -> Tuple[np.ndarray, float, object]:
    if progress:
        print(f"[shap] computing SHAP for rows={len(X):,}, features={X.shape[1]:,}")

    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    values = get_positive_class_shap_values(raw)
    expected_value = get_expected_value(explainer)

    if progress:
        print(f"[shap] done in {(time.time() - t0):.1f}s, shape={values.shape}")

    return values, expected_value, explainer


def build_global_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    out = pd.DataFrame({
        "feature": feature_names,
        "feature_group": [feature_group(f) for f in feature_names],
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    })
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(0, "importance_rank", np.arange(1, len(out) + 1))
    total = out["mean_abs_shap"].sum()
    out["mean_abs_shap_share"] = out["mean_abs_shap"] / total if total else np.nan
    return out


def build_group_importance(global_imp: pd.DataFrame) -> pd.DataFrame:
    group = (
        global_imp.groupby("feature_group", as_index=False)
        .agg(
            mean_abs_shap_sum=("mean_abs_shap", "sum"),
            feature_count=("feature", "count"),
        )
        .sort_values("mean_abs_shap_sum", ascending=False)
        .reset_index(drop=True)
    )
    group.insert(0, "group_rank", np.arange(1, len(group) + 1))
    total = group["mean_abs_shap_sum"].sum()
    group["mean_abs_shap_share"] = group["mean_abs_shap_sum"] / total if total else np.nan
    return group


def build_category_importance(
    sample_meta: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int,
) -> pd.DataFrame:
    if CATEGORY_COL not in sample_meta.columns:
        return pd.DataFrame()

    rows = []
    for cat in sorted(sample_meta[CATEGORY_COL].astype(str).unique()):
        idx = np.flatnonzero(sample_meta[CATEGORY_COL].astype(str).to_numpy() == cat)
        if len(idx) == 0:
            continue
        mean_abs = np.abs(shap_values[idx, :]).mean(axis=0)
        temp = pd.DataFrame({
            "complaint_category": cat,
            "feature": feature_names,
            "feature_group": [feature_group(f) for f in feature_names],
            "mean_abs_shap": mean_abs,
            "rows": len(idx),
        })
        temp = temp.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
        temp.insert(1, "category_rank", np.arange(1, len(temp) + 1))
        rows.append(temp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def select_local_cases(scored_test: pd.DataFrame, threshold: float, cases_per_type: int, random_state: int) -> pd.DataFrame:
    df = scored_test.copy()
    df["pred"] = (df["lgbm_score"] >= threshold).astype(int)

    conditions = {
        "true_positive_high_score": (df["y_true"] == 1) & (df["pred"] == 1),
        "false_positive_high_score": (df["y_true"] == 0) & (df["pred"] == 1),
        "false_negative_low_score": (df["y_true"] == 1) & (df["pred"] == 0),
        "true_negative_low_score": (df["y_true"] == 0) & (df["pred"] == 0),
    }

    parts = []
    for case_type, mask in conditions.items():
        sub = df.loc[mask].copy()
        if sub.empty:
            continue

        if case_type in {"true_positive_high_score", "false_positive_high_score"}:
            sub = sub.sort_values("lgbm_score", ascending=False)
        elif case_type == "false_negative_low_score":
            sub = sub.sort_values("lgbm_score", ascending=False)
        else:
            sub = sub.sort_values("lgbm_score", ascending=True)

        sub = sub.head(cases_per_type).copy()
        sub["case_type"] = case_type
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=False)
    out = out.reset_index().rename(columns={"index": "original_index"})
    out.insert(0, "case_id", [f"case_{i+1:03d}" for i in range(len(out))])
    return out


def build_local_contributions(
    local_cases: pd.DataFrame,
    X_lookup: pd.DataFrame,
    shap_lookup: Dict[int, np.ndarray],
    feature_names: List[str],
    expected_value: float,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    case_rows = []
    contribution_rows = []

    for _, case in local_cases.iterrows():
        original_index = int(case["original_index"])
        if original_index not in shap_lookup:
            continue

        shap_row = shap_lookup[original_index]
        x_row = X_lookup.loc[original_index]

        order = np.argsort(np.abs(shap_row))[::-1][:top_n]
        top_pos = np.argsort(shap_row)[::-1][:top_n]
        top_neg = np.argsort(shap_row)[:top_n]

        case_rows.append({
            "case_id": case["case_id"],
            "case_type": case["case_type"],
            "original_index": original_index,
            "nta2020": case.get("nta2020", ""),
            "ntaname": case.get("ntaname", ""),
            "boroname": case.get("boroname", ""),
            "week_start": case.get("week_start", ""),
            "year": case.get("year", ""),
            "complaint_category": case.get("complaint_category", ""),
            "complaint_category_label": case.get("complaint_category_label", ""),
            "complaint_count": case.get("complaint_count", np.nan),
            "y_true": int(case["y_true"]),
            "prediction": int(case["pred"]),
            "lgbm_score": float(case["lgbm_score"]),
            "expected_value_log_odds": float(expected_value),
            "sum_shap_log_odds": float(np.sum(shap_row)),
            "top_positive_features": "; ".join(feature_names[i] for i in top_pos[:5]),
            "top_negative_features": "; ".join(feature_names[i] for i in top_neg[:5]),
        })

        for rank, i in enumerate(order, start=1):
            contribution_rows.append({
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "importance_rank": rank,
                "feature": feature_names[i],
                "feature_group": feature_group(feature_names[i]),
                "feature_value": x_row.iloc[i],
                "shap_value": shap_row[i],
                "abs_shap_value": abs(shap_row[i]),
            })

    return pd.DataFrame(case_rows), pd.DataFrame(contribution_rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_global_bar_plot(global_imp: pd.DataFrame, output_path: Path, top_n: int, overwrite: bool) -> None:
    ensure_output_path(output_path, overwrite)
    d = global_imp.head(top_n).iloc[::-1].copy()

    plt.figure(figsize=(9, max(6, 0.28 * len(d))))
    plt.barh(d["feature"], d["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"Top {len(d)} global SHAP features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_group_bar_plot(group_imp: pd.DataFrame, output_path: Path, overwrite: bool) -> None:
    ensure_output_path(output_path, overwrite)
    labels = {
        "historical_temporal": "Historical temporal",
        "semantic_category": "Service category",
        "calendar": "Calendar",
        "weather": "Weather",
        "current_demand": "Current demand",
        "spatial_context": "Spatial identifiers",
        "other": "Other",
    }
    d = group_imp.sort_values("mean_abs_shap_sum", ascending=True).copy()
    d["label"] = d["feature_group"].map(labels).fillna(d["feature_group"].str.replace("_", " ").str.title())
    d["share_pct"] = d["mean_abs_shap_sum"] / d["mean_abs_shap_sum"].sum() * 100.0

    plt.figure(figsize=(9, max(5, 0.35 * len(d))))
    plt.barh(d["label"], d["share_pct"])
    plt.xlabel("Share of summed mean absolute SHAP importance (%)")
    plt.ylabel("Feature group")
    plt.title("SHAP importance by feature group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_beeswarm_plot(shap_values: np.ndarray, X: pd.DataFrame, output_path: Path, top_n: int, overwrite: bool) -> None:
    ensure_output_path(output_path, overwrite)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=list(X.columns),
        max_display=top_n,
        show=False,
        plot_size=(10, 7),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_dependence_plots(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    global_imp: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    overwrite: bool,
) -> List[str]:
    paths = []
    top_features = global_imp["feature"].head(top_n).tolist()

    for rank, feature in enumerate(top_features, start=1):
        if feature not in X.columns:
            continue

        j = list(X.columns).index(feature)
        x = X[feature].astype(float)
        y = shap_values[:, j]

        # Downsample for plot clarity.
        plot_df = pd.DataFrame({"x": x, "shap": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(plot_df) > 8000:
            plot_df = plot_df.sample(n=8000, random_state=rank)

        path = output_dir / f"shap_dependence_top{rank:02d}_{safe_filename(feature)}.png"
        ensure_output_path(path, overwrite)

        plt.figure(figsize=(7.5, 5))
        plt.scatter(plot_df["x"], plot_df["shap"], s=5, alpha=0.35)
        plt.axhline(0, linewidth=1)
        plt.xlabel(feature)
        plt.ylabel("SHAP value")
        plt.title(f"SHAP dependence: {feature}")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(str(path))

    return paths


def safe_filename(s: str, max_len: int = 80) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:max_len] if out else "feature"


def save_local_case_plots(
    local_contrib: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    overwrite: bool,
) -> List[str]:
    paths = []
    if local_contrib.empty:
        return paths

    for case_id, g in local_contrib.groupby("case_id"):
        d = g.sort_values("abs_shap_value", ascending=False).head(top_n).copy()
        d = d.sort_values("shap_value", ascending=True)

        path = output_dir / f"shap_local_{safe_filename(case_id)}.png"
        ensure_output_path(path, overwrite)

        plt.figure(figsize=(9, max(5, 0.35 * len(d))))
        plt.barh(d["feature"], d["shap_value"])
        plt.axvline(0, linewidth=1)
        plt.xlabel("SHAP value")
        plt.ylabel("Feature")
        plt.title(f"Local SHAP contributions: {case_id}")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(str(path))

    return paths


# ---------------------------------------------------------------------
# Evaluation and report
# ---------------------------------------------------------------------

def evaluate_scores(scored: pd.DataFrame, threshold: float) -> dict:
    y = scored["y_true"].astype(int).to_numpy()
    s = scored["lgbm_score"].astype(float).to_numpy()
    p = (s >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_share": float(y.mean()),
        "threshold": float(threshold),
        "predicted_positive_rows": int(p.sum()),
        "predicted_positive_share": float(p.mean()),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def md_table(df: pd.DataFrame, columns: List[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No data available._\n"

    d = df[[c for c in columns if c in df.columns]].head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{int(v):,}" if not pd.isna(v) else "")
        else:
            d[c] = d[c].astype(str).replace("nan", "")

    header = "| " + " | ".join(d.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.to_numpy()]
    return "\n".join([header, sep] + rows) + "\n"


def build_report(
    *,
    candidate: dict,
    threshold: float,
    test_metrics: dict,
    global_imp: pd.DataFrame,
    group_imp: pd.DataFrame,
    category_imp: pd.DataFrame,
    local_cases: pd.DataFrame,
    output_files: Dict[str, str],
) -> str:
    lines = []
    lines.append("# SHAP explainability report — Tuned LightGBM\n")
    lines.append("This report explains the Tuned LightGBM component model used as the explanation backbone. The final decision system may use an ensemble and per-category thresholds; SHAP here should be interpreted as explaining a strong tree component of that decision system, not as causal evidence.\n")

    lines.append("## Model explained\n")
    lines.append(f"- Candidate: `{candidate.get('candidate_id')}`\n")
    lines.append(f"- Parameter set: `{candidate.get('param_id')}`\n")
    lines.append(f"- Positive-class weight mode: `{candidate.get('pos_weight_mode')}`\n")
    lines.append(f"- Threshold used for local case selection: `{threshold:.6f}`\n")
    lines.append(f"- Test F1 of LightGBM component at this threshold: `{test_metrics.get('f1', np.nan):.4f}`\n")
    lines.append(f"- Test precision: `{test_metrics.get('precision', np.nan):.4f}`\n")
    lines.append(f"- Test recall: `{test_metrics.get('recall', np.nan):.4f}`\n")
    lines.append(f"- Test PR-AUC: `{test_metrics.get('pr_auc', np.nan):.4f}`\n\n")

    lines.append("## Global SHAP importance\n")
    lines.append("Top features by mean absolute SHAP value:\n")
    lines.append(md_table(global_imp, ["importance_rank", "feature", "feature_group", "mean_abs_shap", "mean_abs_shap_share"], max_rows=20))

    lines.append("## Feature-group SHAP importance\n")
    lines.append(md_table(group_imp, ["group_rank", "feature_group", "feature_count", "mean_abs_shap_sum", "mean_abs_shap_share"], max_rows=20))

    lines.append("## Category-specific SHAP patterns\n")
    lines.append("The file `shap_importance_by_complaint_category.csv` contains the top features for each complaint category. Use this table to support the semantics-aware interpretation that different service categories rely on different signals.\n")
    if not category_imp.empty:
        top_cat = category_imp.groupby("complaint_category").head(5)
        lines.append(md_table(top_cat, ["complaint_category", "category_rank", "feature", "feature_group", "mean_abs_shap", "rows"], max_rows=60))

    lines.append("## Local explanations\n")
    lines.append("Local cases include true positives, false positives, false negatives, and true negatives selected from the test period. Use true positives as examples of correctly detected abnormal service-demand increases; use false positives/false negatives to discuss limitations.\n")
    lines.append(md_table(local_cases, ["case_id", "case_type", "week_start", "boroname", "ntaname", "complaint_category", "y_true", "prediction", "lgbm_score", "top_positive_features", "top_negative_features"], max_rows=40))

    lines.append("## Paper wording suggestions\n")
    lines.append(
        "Suggested wording: `Global SHAP analysis indicates that the model relies primarily on historical demand dynamics, seasonal/calendar signals, and selected contextual variables when estimating next-week abnormal service-demand risk. Category-specific SHAP summaries show that the relative importance of these factors varies across municipal service domains, supporting the semantics-aware formulation of the forecasting task.`\n\n"
    )
    lines.append(
        "Careful wording: `SHAP values explain how features contribute to model predictions; they should not be interpreted as causal effects of weather, POI density, or land-use conditions on actual urban problems.`\n\n"
    )

    lines.append("## Output files\n")
    for k, v in output_files.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    return "\n".join(lines)


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
    ensemble_dir = resolve_path(project_root, args.ensemble_dir, DEFAULT_ENSEMBLE_DIR_REL)
    output_dir = resolve_path(project_root, args.output_dir, DEFAULT_OUTPUT_DIR_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not final_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {final_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found: {config_path}")
    if not lgbm_dir.exists():
        raise FileNotFoundError(f"LightGBM tuning directory not found: {lgbm_dir}")

    if args.progress:
        print(f"[setup] final_path={final_path}")
        print(f"[setup] config_path={config_path}")
        print(f"[setup] lgbm_dir={lgbm_dir}")
        print(f"[setup] ensemble_dir={ensemble_dir}")
        print(f"[setup] output_dir={output_dir}")

    config = load_json(config_path)
    candidate = load_best_lgbm_candidate(lgbm_dir, args.ranking_row)
    threshold = float(args.threshold) if args.threshold is not None else float(candidate.get("selected_threshold", 0.5))

    df, numeric_features, categorical_features = load_dataset(final_path, config, args.feature_set, args.progress)
    model, pre, scored_df, _, train_info = rebuild_lgbm(df, numeric_features, categorical_features, candidate, args)

    test_scored = scored_df[scored_df["split"] == "test"].copy()
    val_scored = scored_df[scored_df["split"] == "validation"].copy()
    test_metrics = evaluate_scores(test_scored, threshold)
    val_metrics = evaluate_scores(val_scored, threshold)

    feature_cols = numeric_features + categorical_features

    # Sample test rows for global SHAP, stratified by target and category.
    sample_meta = stratified_sample(
        test_scored,
        max_rows=args.max_shap_rows,
        random_state=args.sample_random_state,
        strata_cols=["y_true", CATEGORY_COL],
    ).copy()

    # Transform sampled raw features.
    sample_indices = sample_meta.index
    X_sample_raw = df.loc[sample_indices, feature_cols].copy()
    X_sample = pre.transform(X_sample_raw)

    shap_values, expected_value, explainer = compute_shap_values(model, X_sample, args.progress)
    feature_names = list(X_sample.columns)

    global_imp = build_global_importance(shap_values, feature_names)
    group_imp = build_group_importance(global_imp)

    # Category-level importance with optional per-category cap.
    cat_sample_parts = []
    for cat, g in sample_meta.groupby(CATEGORY_COL):
        if len(g) > args.max_category_rows:
            cat_sample_parts.append(g.sample(n=args.max_category_rows, random_state=args.sample_random_state))
        else:
            cat_sample_parts.append(g)
    cat_meta = pd.concat(cat_sample_parts, ignore_index=False)
    cat_idx_positions = [list(sample_meta.index).index(idx) for idx in cat_meta.index if idx in set(sample_meta.index)]
    # More efficient mapping:
    pos_map = {idx: i for i, idx in enumerate(sample_meta.index)}
    cat_positions = [pos_map[idx] for idx in cat_meta.index if idx in pos_map]
    cat_values = shap_values[cat_positions, :]
    cat_meta_aligned = cat_meta.loc[[sample_meta.index[i] for i in cat_positions]].copy()
    category_imp = build_category_importance(cat_meta_aligned, cat_values, feature_names, top_n=args.top_n_features)

    # Local cases: compute SHAP for selected case rows if not already sampled.
    local_cases = select_local_cases(test_scored, threshold, args.local_cases_per_type, args.sample_random_state)
    if not local_cases.empty:
        local_original_indices = local_cases["original_index"].astype(int).tolist()
        X_local_raw = df.loc[local_original_indices, feature_cols].copy()
        X_local = pre.transform(X_local_raw)
        local_shap_values, _, _ = compute_shap_values(model, X_local, args.progress)
        shap_lookup = {
            int(idx): local_shap_values[i, :]
            for i, idx in enumerate(local_original_indices)
        }
        X_local.index = local_original_indices
        local_cases_table, local_contrib = build_local_contributions(
            local_cases=local_cases,
            X_lookup=X_local,
            shap_lookup=shap_lookup,
            feature_names=feature_names,
            expected_value=expected_value,
            top_n=args.local_top_n,
        )
    else:
        local_cases_table = pd.DataFrame()
        local_contrib = pd.DataFrame()

    # Save CSV outputs.
    write_csv(output_dir / "shap_global_importance.csv", global_imp, args.overwrite)
    write_csv(output_dir / "shap_group_importance.csv", group_imp, args.overwrite)
    write_csv(output_dir / "shap_importance_by_complaint_category.csv", category_imp, args.overwrite)
    write_csv(output_dir / "shap_sample_metadata.csv", sample_meta, args.overwrite)
    write_csv(output_dir / "shap_local_cases.csv", local_cases_table, args.overwrite)
    write_csv(output_dir / "shap_local_case_feature_contributions.csv", local_contrib, args.overwrite)

    # Save plots.
    plot_paths = {}
    save_global_bar_plot(global_imp, output_dir / "shap_summary_bar.png", args.top_n_features, args.overwrite)
    plot_paths["shap_summary_bar"] = str(output_dir / "shap_summary_bar.png")

    save_group_bar_plot(group_imp, output_dir / "shap_group_importance.png", args.overwrite)
    plot_paths["shap_group_importance"] = str(output_dir / "shap_group_importance.png")

    try:
        save_beeswarm_plot(shap_values, X_sample, output_dir / "shap_beeswarm.png", args.top_n_features, args.overwrite)
        plot_paths["shap_beeswarm"] = str(output_dir / "shap_beeswarm.png")
    except Exception as exc:
        plot_paths["shap_beeswarm_error"] = repr(exc)

    dependence_paths = save_dependence_plots(
        shap_values=shap_values,
        X=X_sample,
        global_imp=global_imp,
        output_dir=output_dir,
        top_n=args.top_n_dependence,
        overwrite=args.overwrite,
    )
    plot_paths["shap_dependence_plots"] = dependence_paths

    local_plot_paths = save_local_case_plots(
        local_contrib=local_contrib,
        output_dir=output_dir,
        top_n=args.local_top_n,
        overwrite=args.overwrite,
    )
    plot_paths["shap_local_case_plots"] = local_plot_paths

    output_files = {
        "global_importance": str(output_dir / "shap_global_importance.csv"),
        "group_importance": str(output_dir / "shap_group_importance.csv"),
        "category_importance": str(output_dir / "shap_importance_by_complaint_category.csv"),
        "local_cases": str(output_dir / "shap_local_cases.csv"),
        "local_contributions": str(output_dir / "shap_local_case_feature_contributions.csv"),
        "summary_bar": str(output_dir / "shap_summary_bar.png"),
        "group_bar": str(output_dir / "shap_group_importance.png"),
        "beeswarm": str(output_dir / "shap_beeswarm.png"),
        "report": str(output_dir / "shap_explainability_report.md"),
    }

    report = build_report(
        candidate=candidate,
        threshold=threshold,
        test_metrics=test_metrics,
        global_imp=global_imp,
        group_imp=group_imp,
        category_imp=category_imp,
        local_cases=local_cases_table,
        output_files=output_files,
    )
    write_text(output_dir / "shap_explainability_report.md", report, args.overwrite)

    summary = {
        "script": SCRIPT_NAME,
        "status": "done",
        "final_dataset": str(final_path),
        "config": str(config_path),
        "lgbm_tuning_dir": str(lgbm_dir),
        "ensemble_dir": str(ensemble_dir),
        "output_dir": str(output_dir),
        "experiment": args.experiment,
        "feature_set": args.feature_set,
        "analysis_type": args.analysis_type,
        "explained_model": "tuned_lightgbm",
        "candidate": {
            "candidate_id": candidate.get("candidate_id"),
            "param_id": candidate.get("param_id"),
            "pos_weight_mode": candidate.get("pos_weight_mode"),
            "pos_weight": candidate.get("pos_weight"),
            "selected_threshold": threshold,
            "params": candidate.get("params"),
        },
        "train_info": train_info,
        "validation_metrics_at_threshold": val_metrics,
        "test_metrics_at_threshold": test_metrics,
        "shap_sample_rows": int(len(sample_meta)),
        "shap_feature_count": int(len(feature_names)),
        "shap_feature_names": feature_names,
        "expected_value_log_odds": float(expected_value),
        "top_20_global_features": global_imp.head(20).to_dict(orient="records"),
        "feature_group_importance": group_imp.to_dict(orient="records"),
        "local_case_count": int(len(local_cases_table)),
        "plot_paths": plot_paths,
        "elapsed_seconds": round(time.time() - t0, 3),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json(output_dir / "shap_run_summary.json", summary, args.overwrite)

    if args.progress:
        print("\n[top 20 global features]")
        print(global_imp.head(20).to_string(index=False))
        print("\n[feature groups]")
        print(group_imp.to_string(index=False))
        print("\n[test metrics]")
        print(json.dumps(test_metrics, indent=2, ensure_ascii=False, default=to_jsonable))

    print("=" * 90)
    print(f"[{SCRIPT_NAME}] DONE")
    print(f"Report: {output_dir / 'shap_explainability_report.md'}")
    print(f"Global SHAP: {output_dir / 'shap_global_importance.csv'}")
    print(f"Group SHAP: {output_dir / 'shap_group_importance.csv'}")
    print(f"Category SHAP: {output_dir / 'shap_importance_by_complaint_category.csv'}")
    print(f"Run summary: {output_dir / 'shap_run_summary.json'}")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        raise
