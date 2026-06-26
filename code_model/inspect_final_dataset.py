
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10.1 — Inspect final dataset and define modeling feature/experiment config.

Project:
    SIMC NYC — Semantics-Aware Explainable Machine Learning for Urban Service
    Demand Forecasting in Smart Cities.

Purpose:
    This script does NOT train models. It prepares the modeling stage by:
      1. Inspecting the final merged dataset.
      2. Verifying target, split, and train-ready columns.
      3. Selecting safe feature columns and excluding leakage-prone columns.
      4. Defining feature groups and ablation feature sets.
      5. Defining experiment designs, including COVID-disruption robustness.
      6. Writing a machine-readable model_config.json for later scripts.

Expected input:
    data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz

Main outputs:
    data/processed/model_ready/model_config.json
    data/processed/_model_summaries/inspect_final_dataset_summary.json
    data/processed/_model_summaries/inspect_final_dataset_feature_columns.csv
    data/processed/_model_summaries/inspect_final_dataset_excluded_columns.csv
    data/processed/_model_summaries/inspect_final_dataset_feature_sets.json
    data/processed/_model_summaries/inspect_final_dataset_experiment_design.csv
    data/processed/_model_summaries/inspect_final_dataset_rows_by_split_period.csv
    data/processed/_model_summaries/inspect_final_dataset_target_balance_by_experiment.csv
    data/processed/_model_summaries/inspect_final_dataset_missing_selected_features.csv

Run from project root:
    .\\.venv\\Scripts\\python.exe .\\code_model\\inspect_final_dataset.py --progress

Notes:
    - Main target: abnormal_increase_next_week
    - Optional regression target: target_next_week_count
    - Main time split:
        train      <= 2022
        validation = 2023
        test       = 2024–2025
    - COVID periods:
        2015–2019 = pre_covid
        2020–2021 = covid_disruption
        2022–2025 = post_covid
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SCRIPT_NAME = "inspect_final_dataset"

TARGET_CLASSIFICATION = "abnormal_increase_next_week"
TARGET_REGRESSION = "target_next_week_count"

MAIN_READY_COL = "final_train_ready_flag"
TIME_SPLIT_COL = "time_split"
WEEK_COL = "week_start"
TARGET_WEEK_COL = "target_week"
TARGET_YEAR_COL = "target_year"

COVID_YEARS = {2020, 2021}
PRE_COVID_YEARS = set(range(2015, 2020))
POST_COVID_YEARS = set(range(2022, 2026))

DEFAULT_INPUT_REL = Path("data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz")
DEFAULT_MODEL_READY_DIR_REL = Path("data/processed/model_ready")
DEFAULT_SUMMARY_DIR_REL = Path("data/processed/_model_summaries")

# Model feature rules:
# - Do not include target columns or label-construction artifacts.
# - Do not include NTA ID in the main model, because OSM/PLUTO already capture
#   neighborhood context; using NTA ID can turn the model into a memorization
#   model rather than a general explainable framework.
# - Keep complaint_category and boroname as low-cardinality categorical context.
# - Keep period_type / is_covid_period for the main COVID-aware experiment.
# - Keep pluto_source_missing_flag and osm_zero_poi_flag because they describe
#   known data-source coverage conditions, not target leakage.

EXACT_EXCLUDE_REASONS: Dict[str, str] = {
    # Target / label construction
    TARGET_CLASSIFICATION: "target_classification",
    TARGET_REGRESSION: "target_regression",
    "abnormal_threshold_8w": "label_construction_artifact_excluded",

    # Dates and temporal keys
    "week_start": "date_key_not_model_feature",
    "week_end": "date_key_not_model_feature",
    "target_week": "target_date_key_not_model_feature",
    "target_year": "target_year_metadata_not_model_feature",
    "weather_week_end": "date_key_not_model_feature",

    # Spatial IDs and text metadata
    "nta2020": "high_cardinality_spatial_id_excluded_main_model",
    "ntaname": "text_metadata",
    "borocode": "id_metadata",
    "cdta2020": "id_metadata",
    "cdtaname": "text_metadata",
    "complaint_category_label": "duplicate_human_readable_label",
    "poi_dominant_semantic_category_label": "duplicate_human_readable_label",

    # Train filter / readiness flags, not predictors
    "is_train_ready_temporal": "row_filter_not_predictor",
    "has_min_history_12w": "row_filter_not_predictor",
    "has_target_next_week": "row_filter_not_predictor",
    "final_train_ready_flag": "row_filter_not_predictor",

    # Merge/debug keys: constant or quality diagnostics
    "weather_merge_key_present": "merge_diagnostic_not_predictor",
    "osm_merge_key_present": "merge_diagnostic_not_predictor",
    "pluto_merge_key_present": "merge_diagnostic_not_predictor",
    "external_features_merged_flag": "merge_diagnostic_not_predictor",

    # Weather coverage metadata, not actual exposure
    "weather_days_observed": "weather_coverage_metadata",
    "weather_days_missing_in_week": "weather_coverage_metadata",
    "weather_source_rows": "weather_coverage_metadata",
    "weather_complete_week_flag": "weather_coverage_metadata",
    "weather_min_days_ok_flag": "weather_coverage_metadata",
    "weather_missing_flag": "merge_or_coverage_diagnostic_not_predictor",
    "weather_incomplete_week_flag": "merge_or_coverage_diagnostic_not_predictor",

    # External table presence flags mostly constant in the final data
    "osm_missing_flag": "merge_diagnostic_not_predictor",
    "pluto_table_missing_flag": "merge_diagnostic_not_predictor",

    # Weather duplicate calendar fields; use main calendar instead.
    "weather_year": "duplicate_calendar_from_weather",
    "weather_month": "duplicate_calendar_from_weather",
    "weather_quarter": "duplicate_calendar_from_weather",
    "weather_week_of_year": "duplicate_calendar_from_weather",
    "weather_iso_year": "duplicate_calendar_from_weather",
}

KEEP_CATEGORICAL_FEATURES = {
    "complaint_category",
    "boroname",
    "period_type",
    "poi_dominant_semantic_category",
}

KEEP_QUALITY_FEATURES = {
    "pluto_source_missing_flag",
    "osm_zero_poi_flag",
}

# Columns that identify the row but can be useful in summaries.
SUMMARY_ID_COLS = [
    "nta2020",
    "ntaname",
    "boroname",
    "week_start",
    "target_week",
    "target_year",
    "complaint_category",
    "complaint_category_label",
    "year",
    "period_type",
    "time_split",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect final NYC service demand dataset and create modeling config."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=f"Final dataset path. Default: {DEFAULT_INPUT_REL}",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk for summary scans.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress while scanning chunks.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5000,
        help="Number of rows to sample for dtype/category preview.",
    )
    return parser.parse_args()


def get_project_root() -> Path:
    # Expected location: SIMC_PROJECT/code_model/inspect_final_dataset.py
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_to_json_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(path: Path, data: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=safe_to_json_value)


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")
    ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_bool_like(s: pd.Series) -> pd.Series:
    """
    Convert common bool/int/string flags to integer 0/1 while preserving missing as 0
    for filtering purposes.
    """
    if s.dtype == bool:
        return s.astype("int8")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype("int8")

    lowered = s.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"}).astype("int8")


def infer_time_split_from_target_week(target_week: pd.Series) -> pd.Series:
    dates = pd.to_datetime(target_week, errors="coerce")
    year = dates.dt.year
    split = pd.Series("unused", index=target_week.index, dtype="object")
    split.loc[year <= 2022] = "train"
    split.loc[year == 2023] = "validation"
    split.loc[year >= 2024] = "test"
    split.loc[dates.isna()] = "bad_date"
    return split


def infer_period_type_from_year(year: pd.Series) -> pd.Series:
    yr = pd.to_numeric(year, errors="coerce")
    out = pd.Series("unknown", index=year.index, dtype="object")
    out.loc[yr.between(2015, 2019, inclusive="both")] = "pre_covid"
    out.loc[yr.between(2020, 2021, inclusive="both")] = "covid_disruption"
    out.loc[yr.between(2022, 2025, inclusive="both")] = "post_covid"
    return out


def group_column(col: str) -> str:
    if col == "complaint_count":
        return "current_demand"

    if (
        col.startswith("lag_")
        or col.startswith("rolling_")
        or col.startswith("diff_")
        or col.startswith("pct_change_")
        or col.startswith("ratio_to_")
        or col == "history_weeks_available"
    ):
        return "temporal_demand"

    if col in {
        "year",
        "month",
        "quarter",
        "week_of_year",
        "iso_year",
        "is_year_start",
        "is_year_end",
        "period_type",
        "is_covid_period",
    }:
        return "calendar_period"

    if (
        col.startswith("weather_")
        or col.startswith("hot_day_")
        or col.startswith("cold_day_")
        or col.startswith("ice_day_")
        or col.startswith("very_cold_day_")
        or col.startswith("rain_day_")
        or col.startswith("moderate_rain_day_")
        or col.startswith("heavy_rain_day_")
        or col.startswith("very_heavy_rain_day_")
        or col.startswith("snow_day_")
        or col.startswith("heavy_snow_day_")
    ):
        return "weather"

    if col.startswith("poi_"):
        return "osm_poi"

    if col.startswith("pluto_"):
        return "pluto_landuse_built_environment"

    if col in KEEP_QUALITY_FEATURES:
        return "quality_flag"

    if col in {"complaint_category", "boroname"}:
        return "categorical_context"

    return "other"


def should_include_feature(col: str, sample_df: pd.DataFrame) -> Tuple[bool, str, str]:
    """
    Returns:
        include, reason, group
    """
    if col in EXACT_EXCLUDE_REASONS:
        return False, EXACT_EXCLUDE_REASONS[col], group_column(col)

    group = group_column(col)

    if col in KEEP_CATEGORICAL_FEATURES:
        return True, "included_categorical_context", group

    if col in KEEP_QUALITY_FEATURES:
        return True, "included_known_quality_flag", group

    if group == "other":
        return False, "unrecognized_or_metadata_column", group

    if col not in sample_df.columns:
        return False, "not_in_sample_df", group

    dtype = sample_df[col].dtype

    # Avoid arbitrary text columns unless explicitly allowed.
    if pd.api.types.is_object_dtype(dtype):
        return False, "object_text_column_not_in_allowed_categoricals", group

    return True, "included_numeric_feature", group


def unique_preserve_order(cols: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def make_feature_sets(feature_df: pd.DataFrame) -> Dict[str, List[str]]:
    by_group = defaultdict(list)
    for row in feature_df.itertuples(index=False):
        if bool(row.include):
            by_group[row.feature_group].append(row.column)

    categorical_core = [c for c in ["complaint_category", "boroname"] if c in by_group["categorical_context"]]
    period_categorical = [c for c in ["period_type"] if c in by_group["calendar_period"]]
    category_context = categorical_core + period_categorical

    current = by_group["current_demand"]
    temporal = by_group["temporal_demand"]
    calendar = by_group["calendar_period"]
    weather = by_group["weather"]
    osm = by_group["osm_poi"]
    pluto = by_group["pluto_landuse_built_environment"]
    quality = by_group["quality_flag"]

    historical_only = unique_preserve_order(categorical_core + current + temporal)

    historical_calendar_covid = unique_preserve_order(
        categorical_core + current + temporal + calendar
    )

    historical_calendar_no_covid_period = [
        c for c in historical_calendar_covid if c not in {"period_type", "is_covid_period"}
    ]

    plus_weather = unique_preserve_order(historical_calendar_covid + weather)

    plus_weather_osm = unique_preserve_order(plus_weather + osm)

    plus_weather_pluto = unique_preserve_order(plus_weather + pluto)

    full = unique_preserve_order(historical_calendar_covid + weather + osm + pluto + quality)

    full_without_covid_features = [
        c for c in full if c not in {"period_type", "is_covid_period"}
    ]

    full_without_osm = unique_preserve_order(historical_calendar_covid + weather + pluto + quality)

    full_without_pluto = unique_preserve_order(historical_calendar_covid + weather + osm + quality)

    external_only = unique_preserve_order(category_context + weather + osm + pluto + quality)

    return {
        "historical_only": historical_only,
        "historical_calendar_covid": historical_calendar_covid,
        "historical_calendar_no_covid_period": historical_calendar_no_covid_period,
        "historical_calendar_weather": plus_weather,
        "historical_calendar_weather_osm": plus_weather_osm,
        "historical_calendar_weather_pluto": plus_weather_pluto,
        "full": full,
        "full_without_covid_period_features": full_without_covid_features,
        "full_without_osm": full_without_osm,
        "full_without_pluto": full_without_pluto,
        "external_context_only": external_only,
    }


def make_experiment_design() -> pd.DataFrame:
    rows = [
        {
            "experiment_id": "E1_main_2015_2025",
            "purpose": "Main prospective forecasting model using all train-ready rows from normal, COVID-disruption, and post-COVID periods.",
            "train_rule": "final_train_ready_flag == 1 and time_split == 'train'",
            "validation_rule": "final_train_ready_flag == 1 and time_split == 'validation'",
            "test_rule": "final_train_ready_flag == 1 and time_split == 'test'",
            "recommended_feature_sets": "full; historical_only; historical_calendar_weather; ablations",
            "paper_use": "Main result",
        },
        {
            "experiment_id": "E2_without_covid_years",
            "purpose": "Robustness check excluding the 2020–2021 COVID-disruption years.",
            "train_rule": "final_train_ready_flag == 1 and time_split == 'train' and year not in [2020, 2021]",
            "validation_rule": "final_train_ready_flag == 1 and time_split == 'validation'",
            "test_rule": "final_train_ready_flag == 1 and time_split == 'test'",
            "recommended_feature_sets": "full; full_without_covid_period_features",
            "paper_use": "Robustness / COVID sensitivity",
        },
        {
            "experiment_id": "E3_covid_period_feature_ablation",
            "purpose": "Compare full model with and without explicit COVID-period features under the main 2015–2025 split.",
            "train_rule": "same_as_E1",
            "validation_rule": "same_as_E1",
            "test_rule": "same_as_E1",
            "recommended_feature_sets": "full vs full_without_covid_period_features",
            "paper_use": "COVID as urban-disruption feature analysis",
        },
        {
            "experiment_id": "E4_pre_to_post_generalization",
            "purpose": "Stress test: learn pre-COVID and early post-COVID structure without COVID-disruption years, then test on 2024–2025.",
            "train_rule": "final_train_ready_flag == 1 and year <= 2019",
            "validation_rule": "final_train_ready_flag == 1 and year in [2022, 2023]",
            "test_rule": "final_train_ready_flag == 1 and year >= 2024",
            "recommended_feature_sets": "full; historical_calendar_weather; historical_only",
            "paper_use": "Optional robustness if space allows",
        },
    ]
    return pd.DataFrame(rows)


def get_experiment_split_mask(df: pd.DataFrame, experiment_id: str, split_name: str) -> pd.Series:
    ready = normalize_bool_like(df[MAIN_READY_COL]) == 1

    year = pd.to_numeric(df["year"], errors="coerce")
    time_split = df[TIME_SPLIT_COL].astype(str)

    if experiment_id == "E1_main_2015_2025":
        return ready & (time_split == split_name)

    if experiment_id == "E2_without_covid_years":
        no_covid = ~year.isin([2020, 2021])
        return ready & no_covid & (time_split == split_name)

    if experiment_id == "E3_covid_period_feature_ablation":
        return ready & (time_split == split_name)

    if experiment_id == "E4_pre_to_post_generalization":
        if split_name == "train":
            return ready & (year <= 2019)
        if split_name == "validation":
            return ready & year.isin([2022, 2023])
        if split_name == "test":
            return ready & (year >= 2024)
        return pd.Series(False, index=df.index)

    raise ValueError(f"Unknown experiment_id: {experiment_id}")


def update_target_balance(
    accumulator: Dict[Tuple[str, str], Dict[str, float]],
    df: pd.DataFrame,
    experiment_ids: List[str],
) -> None:
    target = pd.to_numeric(df[TARGET_CLASSIFICATION], errors="coerce")

    for exp_id in experiment_ids:
        for split in ["train", "validation", "test"]:
            mask = get_experiment_split_mask(df, exp_id, split)
            n_rows = int(mask.sum())
            if n_rows == 0:
                continue

            y = target.loc[mask]
            valid = y.notna()
            pos = int((y.loc[valid] == 1).sum())
            neg = int((y.loc[valid] == 0).sum())
            nan = int((~valid).sum())

            key = (exp_id, split)
            acc = accumulator.setdefault(
                key,
                {
                    "rows": 0,
                    "valid_target_rows": 0,
                    "positive_rows": 0,
                    "negative_rows": 0,
                    "missing_target_rows": 0,
                },
            )
            acc["rows"] += n_rows
            acc["valid_target_rows"] += int(valid.sum())
            acc["positive_rows"] += pos
            acc["negative_rows"] += neg
            acc["missing_target_rows"] += nan


def update_group_counts(accumulator: Dict[Tuple[str, str, str], int], df: pd.DataFrame) -> None:
    needed = [MAIN_READY_COL, TIME_SPLIT_COL, "period_type", "year"]
    for c in needed:
        if c not in df.columns:
            return

    tmp = df[[MAIN_READY_COL, TIME_SPLIT_COL, "period_type", "year"]].copy()
    tmp["final_train_ready_flag"] = normalize_bool_like(tmp[MAIN_READY_COL])
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
    grouped = (
        tmp.groupby(["time_split", "period_type", "year", "final_train_ready_flag"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    for r in grouped.itertuples(index=False):
        key = (str(r.time_split), str(r.period_type), str(r.year), str(r.final_train_ready_flag))
        accumulator[key] += int(r.rows)


def update_missing_counts(
    missing_acc: Dict[str, int],
    selected_features: List[str],
    df: pd.DataFrame,
) -> None:
    present = [c for c in selected_features if c in df.columns]
    if not present:
        return
    miss = df[present].isna().sum()
    for col, val in miss.items():
        missing_acc[col] += int(val)


def update_category_preview(
    category_acc: Dict[str, set],
    df: pd.DataFrame,
    categorical_cols: List[str],
    max_levels: int = 100,
) -> None:
    for col in categorical_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(str).unique().tolist()
        store = category_acc.setdefault(col, set())
        for v in vals:
            if len(store) < max_levels:
                store.add(v)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    project_root = get_project_root()
    input_path = Path(args.input) if args.input else project_root / DEFAULT_INPUT_REL
    model_ready_dir = project_root / DEFAULT_MODEL_READY_DIR_REL
    summary_dir = project_root / DEFAULT_SUMMARY_DIR_REL

    if not input_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {input_path}")

    model_ready_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Read header and sample. This is intentionally lightweight.
    header = pd.read_csv(input_path, nrows=0)
    columns = header.columns.tolist()

    sample = pd.read_csv(input_path, nrows=args.sample_rows, low_memory=False)
    if WEEK_COL in sample.columns:
        sample[WEEK_COL] = pd.to_datetime(sample[WEEK_COL], errors="coerce")

    required = [WEEK_COL, TARGET_CLASSIFICATION, TARGET_REGRESSION, MAIN_READY_COL, TIME_SPLIT_COL, "year"]
    missing_required = [c for c in required if c not in columns]
    if missing_required:
        raise ValueError(f"Missing required final dataset columns: {missing_required}")

    # Select features.
    feature_rows = []
    excluded_rows = []
    for col in columns:
        include, reason, group = should_include_feature(col, sample)
        dtype = str(sample[col].dtype) if col in sample.columns else "unknown"

        row = {
            "column": col,
            "include": bool(include),
            "reason": reason,
            "feature_group": group,
            "dtype_sample": dtype,
            "is_categorical_feature": bool(col in KEEP_CATEGORICAL_FEATURES),
            "is_quality_feature": bool(col in KEEP_QUALITY_FEATURES),
        }
        if include:
            feature_rows.append(row)
        else:
            excluded_rows.append(row)

    feature_df = pd.DataFrame(feature_rows).sort_values(["feature_group", "column"]).reset_index(drop=True)
    excluded_df = pd.DataFrame(excluded_rows).sort_values(["reason", "column"]).reset_index(drop=True)

    selected_features = feature_df["column"].tolist()
    categorical_features = [c for c in selected_features if c in KEEP_CATEGORICAL_FEATURES]
    numeric_features = [c for c in selected_features if c not in categorical_features]

    feature_sets = make_feature_sets(feature_df)
    experiment_design = make_experiment_design()
    experiment_ids = experiment_design["experiment_id"].tolist()

    # Scan dataset in chunks for row counts, missing counts, target balance, and experiment counts.
    total_rows = 0
    min_week = None
    max_week = None
    unique_weeks = set()
    unique_ntas = set()
    unique_categories = set()

    target_balance_exp: Dict[Tuple[str, str], Dict[str, float]] = {}
    split_period_year_acc: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    missing_acc: Dict[str, int] = defaultdict(int)
    categorical_preview_acc: Dict[str, set] = {}

    chunks = pd.read_csv(input_path, chunksize=args.chunksize, low_memory=False)
    for i, chunk in enumerate(chunks, start=1):
        total_rows += len(chunk)

        # Robust fallback if period/time fields were altered.
        if WEEK_COL in chunk.columns:
            week_dates = pd.to_datetime(chunk[WEEK_COL], errors="coerce")
            cmin = week_dates.min()
            cmax = week_dates.max()
            if pd.notna(cmin):
                min_week = cmin if min_week is None else min(min_week, cmin)
            if pd.notna(cmax):
                max_week = cmax if max_week is None else max(max_week, cmax)
            # Unique week count is small, safe to keep as set.
            unique_weeks.update(week_dates.dropna().dt.strftime("%Y-%m-%d").unique().tolist())

        if "nta2020" in chunk.columns:
            unique_ntas.update(chunk["nta2020"].dropna().astype(str).unique().tolist())

        if "complaint_category" in chunk.columns:
            unique_categories.update(chunk["complaint_category"].dropna().astype(str).unique().tolist())

        if TARGET_WEEK_COL not in chunk.columns and WEEK_COL in chunk.columns:
            chunk[TARGET_WEEK_COL] = pd.to_datetime(chunk[WEEK_COL], errors="coerce") + pd.Timedelta(days=7)
        if TARGET_YEAR_COL not in chunk.columns and TARGET_WEEK_COL in chunk.columns:
            chunk[TARGET_YEAR_COL] = pd.to_datetime(chunk[TARGET_WEEK_COL], errors="coerce").dt.year
        if TIME_SPLIT_COL not in chunk.columns or chunk[TIME_SPLIT_COL].isna().all():
            split_key = chunk[TARGET_WEEK_COL] if TARGET_WEEK_COL in chunk.columns else chunk[WEEK_COL]
            chunk[TIME_SPLIT_COL] = infer_time_split_from_target_week(split_key)

        if "period_type" not in chunk.columns or chunk["period_type"].isna().all():
            chunk["period_type"] = infer_period_type_from_year(chunk["year"])

        update_target_balance(target_balance_exp, chunk, experiment_ids)
        update_group_counts(split_period_year_acc, chunk)
        update_missing_counts(missing_acc, selected_features, chunk)
        update_category_preview(categorical_preview_acc, chunk, categorical_features)

        if args.progress:
            print(
                f"[progress] chunks={i:,}, rows_seen={total_rows:,}, "
                f"weeks={len(unique_weeks):,}, ntas={len(unique_ntas):,}, "
                f"features={len(selected_features):,}",
                flush=True,
            )

    # Build summary tables.
    missing_df = pd.DataFrame(
        [
            {
                "column": col,
                "missing_rows": int(missing_acc.get(col, 0)),
                "missing_share": float(missing_acc.get(col, 0) / total_rows) if total_rows else np.nan,
                "feature_group": feature_df.loc[feature_df["column"] == col, "feature_group"].iloc[0],
                "is_categorical_feature": col in categorical_features,
            }
            for col in selected_features
        ]
    ).sort_values(["missing_share", "column"], ascending=[False, True])

    feature_group_counts = (
        feature_df.groupby("feature_group")
        .size()
        .reset_index(name="selected_feature_count")
        .sort_values("selected_feature_count", ascending=False)
    )

    rows_by_split_period = pd.DataFrame(
        [
            {
                "time_split": key[0],
                "period_type": key[1],
                "year": key[2],
                "final_train_ready_flag": key[3],
                "rows": val,
            }
            for key, val in split_period_year_acc.items()
        ]
    ).sort_values(["time_split", "year", "period_type", "final_train_ready_flag"])

    target_balance_rows = []
    for (exp_id, split), d in sorted(target_balance_exp.items()):
        valid = d["valid_target_rows"]
        pos = d["positive_rows"]
        neg = d["negative_rows"]
        target_balance_rows.append(
            {
                "experiment_id": exp_id,
                "split": split,
                "rows": int(d["rows"]),
                "valid_target_rows": int(valid),
                "positive_rows": int(pos),
                "negative_rows": int(neg),
                "missing_target_rows": int(d["missing_target_rows"]),
                "positive_share": float(pos / valid) if valid else np.nan,
            }
        )
    target_balance_df = pd.DataFrame(target_balance_rows)

    feature_sets_meta = {
        name: {
            "feature_count": len(cols),
            "features": cols,
        }
        for name, cols in feature_sets.items()
    }

    categorical_preview = {
        col: sorted(list(vals))
        for col, vals in categorical_preview_acc.items()
    }

    config = {
        "script": SCRIPT_NAME,
        "created_for_step": "10.1",
        "project_root": str(project_root),
        "input_file": str(input_path),
        "target_classification": TARGET_CLASSIFICATION,
        "target_regression_optional": TARGET_REGRESSION,
        "train_ready_filter": f"{MAIN_READY_COL} == 1",
        "main_time_split": {
            "train": "2015-01-01 <= target_week <= 2022-12-31",
            "validation": "2023-01-01 <= target_week <= 2023-12-31",
            "test": "2024-01-01 <= target_week <= 2025-12-31",
            "note": "Use chronological split by target_week; do not split by feature week or random split.",
        },
        "covid_periods": {
            "pre_covid": "2015-2019",
            "covid_disruption": "2020-2021",
            "post_covid": "2022-2025",
        },
        "selected_features": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_sets": feature_sets_meta,
        "experiment_ids": experiment_ids,
        "recommended_main_experiment": "E1_main_2015_2025",
        "recommended_main_feature_set": "full",
        "recommended_ablation_order": [
            "historical_only",
            "historical_calendar_covid",
            "historical_calendar_weather",
            "historical_calendar_weather_osm",
            "full",
            "full_without_covid_period_features",
            "full_without_osm",
            "full_without_pluto",
        ],
        "categorical_preview": categorical_preview,
        "leakage_policy": {
            "excluded_targets": [TARGET_CLASSIFICATION, TARGET_REGRESSION],
            "excluded_label_artifact": ["abnormal_threshold_8w"],
            "target_week_split": "feature_week = week_start; target_week = week_start + 7 days; time_split is assigned by target_week.",
            "excluded_high_cardinality_id_main_model": ["nta2020", "ntaname"],
            "excluded_filter_columns": ["final_train_ready_flag", "is_train_ready_temporal", "has_target_next_week"],
            "safe_current_week_feature": "complaint_count is allowed because row t predicts week t+1.",
        },
        "output_paths": {
            "model_config_json": str(model_ready_dir / "model_config.json"),
            "feature_columns_csv": str(summary_dir / "inspect_final_dataset_feature_columns.csv"),
            "excluded_columns_csv": str(summary_dir / "inspect_final_dataset_excluded_columns.csv"),
            "feature_sets_json": str(summary_dir / "inspect_final_dataset_feature_sets.json"),
            "experiment_design_csv": str(summary_dir / "inspect_final_dataset_experiment_design.csv"),
            "summary_json": str(summary_dir / "inspect_final_dataset_summary.json"),
        },
    }

    summary = {
        "script": SCRIPT_NAME,
        "status": "done",
        "input_file": str(input_path),
        "model_ready_dir": str(model_ready_dir),
        "summary_dir": str(summary_dir),
        "total_rows": int(total_rows),
        "selected_feature_count": int(len(selected_features)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "excluded_column_count": int(len(excluded_df)),
        "unique_weeks": int(len(unique_weeks)),
        "unique_nta2020": int(len(unique_ntas)),
        "unique_complaint_categories": int(len(unique_categories)),
        "min_week_start": min_week.strftime("%Y-%m-%d") if min_week is not None and pd.notna(min_week) else None,
        "max_week_start": max_week.strftime("%Y-%m-%d") if max_week is not None and pd.notna(max_week) else None,
        "covid_periods": config["covid_periods"],
        "main_time_split": config["main_time_split"],
        "feature_group_counts": {
            row.feature_group: int(row.selected_feature_count)
            for row in feature_group_counts.itertuples(index=False)
        },
        "feature_sets": {
            name: len(cols)
            for name, cols in feature_sets.items()
        },
        "experiments": experiment_design.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - t0, 3),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }

    # Write outputs.
    write_csv(summary_dir / "inspect_final_dataset_feature_columns.csv", feature_df, args.overwrite)
    write_csv(summary_dir / "inspect_final_dataset_excluded_columns.csv", excluded_df, args.overwrite)
    write_csv(summary_dir / "inspect_final_dataset_feature_group_counts.csv", feature_group_counts, args.overwrite)
    write_csv(summary_dir / "inspect_final_dataset_experiment_design.csv", experiment_design, args.overwrite)
    write_csv(summary_dir / "inspect_final_dataset_rows_by_split_period.csv", rows_by_split_period, args.overwrite)
    write_csv(summary_dir / "inspect_final_dataset_target_balance_by_experiment.csv", target_balance_df, args.overwrite)
    write_csv(summary_dir / "inspect_final_dataset_missing_selected_features.csv", missing_df, args.overwrite)

    write_json(summary_dir / "inspect_final_dataset_feature_sets.json", feature_sets_meta, args.overwrite)
    write_json(summary_dir / "inspect_final_dataset_summary.json", summary, args.overwrite)
    write_json(model_ready_dir / "model_config.json", config, args.overwrite)

    print("=" * 80)
    print(f"[{SCRIPT_NAME}] DONE")
    print(f"Input: {input_path}")
    print(f"Rows scanned: {total_rows:,}")
    print(f"Selected features: {len(selected_features):,}")
    print(f"Numeric features: {len(numeric_features):,}")
    print(f"Categorical features: {len(categorical_features):,}")
    print(f"Model config: {model_ready_dir / 'model_config.json'}")
    print(f"Summary dir: {summary_dir}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        raise
