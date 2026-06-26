#!/usr/bin/env python3
r"""
Step 9 — Build final NYC urban service-demand dataset

This script merges all feature tables produced by Steps 5–8 into one modeling table:

    NTA × week_start × complaint_category

Inputs:
    data/processed/feature_tables/nyc_311_weekly_temporal_features.csv.gz
    data/processed/feature_tables/noaa_weather_weekly_features.csv
    data/processed/feature_tables/osm_poi_nta_features.csv
    data/processed/feature_tables/pluto_nta_landuse_features.csv

Output:
    data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz

Design goals:
    - Keep the Step 5 temporal panel as the row spine.
    - Use left joins so no valid 311 temporal rows are dropped.
    - Require one-to-one dimension keys for weather, OSM, and PLUTO to avoid row explosion.
    - Merge in chunks to reduce memory pressure on laptops.
    - Add merge/missing/imputation flags so downstream modeling is transparent.
    - Generate summary files for paper reporting and debugging.

Run from project root:
    .\.venv\Scripts\python.exe .\code_processdata\build_final_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SCRIPT_NAME = "build_final_dataset"

DEFAULT_TEMPORAL_INPUT = Path("data/processed/feature_tables/nyc_311_weekly_temporal_features.csv.gz")
DEFAULT_WEATHER_INPUT = Path("data/processed/feature_tables/noaa_weather_weekly_features.csv")
DEFAULT_OSM_INPUT = Path("data/processed/feature_tables/osm_poi_nta_features.csv")
DEFAULT_PLUTO_INPUT = Path("data/processed/feature_tables/pluto_nta_landuse_features.csv")
DEFAULT_OUTPUT = Path("data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz")
DEFAULT_SUMMARY_DIR = Path("data/processed/_final_summaries")

CANONICAL_ID_COLS = [
    "nta2020",
    "ntaname",
    "boroname",
    "borocode",
    "cdta2020",
    "cdtaname",
    "week_start",
    "week_end",
    "target_week",
    "target_year",
    "complaint_category",
    "complaint_category_label",
]

TEMPORAL_REQUIRED_COLS = [
    "nta2020",
    "week_start",
    "complaint_category",
    "complaint_count",
    "target_next_week_count",
    "abnormal_increase_next_week",
]

WEATHER_REQUIRED_COLS = ["week_start"]
OSM_REQUIRED_COLS = ["nta2020"]
PLUTO_REQUIRED_COLS = ["nta2020"]

PREFERRED_TARGET_COLS = [
    "target_next_week_count",
    "abnormal_increase_next_week",
    "abnormal_threshold_8w",
]

TEMPORAL_READY_CANDIDATES = [
    "is_train_ready_temporal",
    "has_target_next_week",
    "has_min_history_12w",
]

SPLIT_TRAIN_END = pd.Timestamp("2022-12-31")
SPLIT_VAL_START = pd.Timestamp("2023-01-01")
SPLIT_VAL_END = pd.Timestamp("2023-12-31")
SPLIT_TEST_START = pd.Timestamp("2024-01-01")


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def project_root_from_script() -> Path:
    """Return project root assuming this script lives in code_processdata/."""
    return Path(__file__).resolve().parents[1]


def resolve_path(project_root: Path, path_like: Path | str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else project_root / p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def as_string_key(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def to_bool_int(series: pd.Series) -> pd.Series:
    """Convert common boolean/flag encodings to nullable Int64 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").round().astype("Int64")
    low = s.astype("string").str.strip().str.lower()
    mapped = low.map({
        "true": 1,
        "t": 1,
        "yes": 1,
        "y": 1,
        "1": 1,
        "false": 0,
        "f": 0,
        "no": 0,
        "n": 0,
        "0": 0,
    })
    return mapped.astype("Int64")


def date_to_monday_week_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt - pd.to_timedelta(dt.dt.dayofweek, unit="D")


def safe_json_dump(obj: dict, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            if math.isnan(float(o)):
                return None
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def read_header(path: Path) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def require_columns(df_or_cols: pd.DataFrame | Sequence[str], required: Sequence[str], label: str) -> None:
    cols = set(df_or_cols.columns if isinstance(df_or_cols, pd.DataFrame) else df_or_cols)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def duplicate_key_report(df: pd.DataFrame, key_cols: Sequence[str], label: str) -> Dict[str, object]:
    dup_mask = df.duplicated(list(key_cols), keep=False)
    duplicate_rows_involved = int(dup_mask.sum())
    duplicate_extra_rows = int(df.duplicated(list(key_cols), keep="first").sum())
    return {
        "table": label,
        "key_cols": list(key_cols),
        "rows": int(len(df)),
        "unique_keys": int(df[list(key_cols)].drop_duplicates().shape[0]),
        "duplicate_rows_involved": duplicate_rows_involved,
        "duplicate_extra_rows": duplicate_extra_rows,
    }


def fail_if_duplicate_keys(df: pd.DataFrame, key_cols: Sequence[str], label: str) -> None:
    report = duplicate_key_report(df, key_cols, label)
    if report["duplicate_extra_rows"] > 0:
        examples = df[df.duplicated(list(key_cols), keep=False)][list(key_cols)].head(10)
        raise ValueError(
            f"{label} has duplicate keys for {key_cols}; this would cause row explosion. "
            f"Duplicate extra rows: {report['duplicate_extra_rows']}. Examples:\n{examples}"
        )


def infer_time_split(target_week: pd.Series) -> pd.Series:
    dt = pd.to_datetime(target_week, errors="coerce")
    out = pd.Series("out_of_range", index=target_week.index, dtype="object")
    out.loc[dt <= SPLIT_TRAIN_END] = "train"
    out.loc[(dt >= SPLIT_VAL_START) & (dt <= SPLIT_VAL_END)] = "validation"
    out.loc[dt >= SPLIT_TEST_START] = "test"
    out.loc[dt.isna()] = "bad_date"
    return out


def classify_feature_column(col: str) -> str:
    """Assign columns to high-level groups for the feature reference file."""
    if col in CANONICAL_ID_COLS or col in ["time_split"]:
        return "id_metadata"
    if col in PREFERRED_TARGET_COLS:
        return "target"
    if col.startswith("weather_") or col.endswith("_day_count") or col in [
        "hot_day_30c_count",
        "hot_day_35c_count",
        "cold_day_0c_count",
        "rain_day_count",
        "snow_day_count",
        "heavy_rain_day_25mm_count",
    ]:
        return "weather"
    if col.startswith("poi_"):
        return "osm_poi"
    if col.startswith("pluto_") or col == "nta_area_km2":
        return "pluto_landuse_built_environment"
    if col.startswith("lag_") or col.startswith("rolling_") or col in [
        "diff_1w_count",
        "diff_4w_count",
        "pct_change_1w",
        "ratio_to_8w_mean",
        "ratio_to_12w_mean",
        "history_weeks_available",
    ]:
        return "temporal_demand"
    if col in [
        "year",
        "month",
        "quarter",
        "week_of_year",
        "iso_year",
        "period_type",
        "is_covid_period",
        "is_year_start",
        "is_year_end",
    ]:
        return "calendar_period"
    if col.endswith("_flag") or col.startswith("has_") or col.startswith("is_"):
        return "quality_flag"
    if col == "complaint_count":
        return "current_demand"
    return "other"


def is_likely_count_sum_density_col(col: str) -> bool:
    patterns = [
        "_count",
        "_sum",
        "_density",
        "total_",
        "lot_area_",
        "building_area",
        "parcel_density",
        "buildings",
        "units",
        "area",
    ]
    return any(p in col for p in patterns)


def is_likely_ratio_share_col(col: str) -> bool:
    return any(p in col for p in ["_ratio", "_share", "_hhi", "entropy", "mixed_use_index"])


def is_likely_average_intensity_col(col: str) -> bool:
    patterns = [
        "avg_",
        "mean",
        "median",
        "num_floors",
        "floor",
        "far",
        "building_area_to_lot_area_ratio",
    ]
    return any(p in col for p in patterns)


# -----------------------------------------------------------------------------
# Dimension table preparation
# -----------------------------------------------------------------------------


def prepare_weather(path: Path, strict: bool = True) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(path)
    require_columns(df, WEATHER_REQUIRED_COLS, "weather table")
    df = df.copy()
    df["week_start"] = date_to_monday_week_start(df["week_start"])
    bad_dates = int(df["week_start"].isna().sum())
    if strict and bad_dates:
        raise ValueError(f"weather table has {bad_dates} bad week_start values")
    df = df.dropna(subset=["week_start"])
    fail_if_duplicate_keys(df, ["week_start"], "weather table")

    # Avoid duplicate generic week_end if temporal table already has week_end.
    if "week_end" in df.columns:
        df = df.rename(columns={"week_end": "weather_week_end"})

    # Small indicator to identify successful merge.
    if "weather_merge_key_present" not in df.columns:
        df["weather_merge_key_present"] = 1

    info = {
        "rows": int(len(df)),
        "unique_week_start": int(df["week_start"].nunique()),
        "min_week_start": str(df["week_start"].min().date()) if len(df) else None,
        "max_week_start": str(df["week_start"].max().date()) if len(df) else None,
        "duplicate_key_report": duplicate_key_report(df, ["week_start"], "weather table"),
    }
    return df, info


def prepare_osm(path: Path, strict: bool = True) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(path)
    require_columns(df, OSM_REQUIRED_COLS, "OSM POI table")
    df = df.copy()
    df["nta2020"] = as_string_key(df["nta2020"])
    df = df.dropna(subset=["nta2020"])
    fail_if_duplicate_keys(df, ["nta2020"], "OSM POI table")

    # The temporal table is the canonical source for NTA names/borough labels.
    drop_identity = [c for c in ["ntaname", "boroname", "borocode", "cdta2020", "cdtaname"] if c in df.columns]
    df = df.drop(columns=drop_identity)

    if "osm_merge_key_present" not in df.columns:
        df["osm_merge_key_present"] = 1

    info = {
        "rows": int(len(df)),
        "unique_nta2020": int(df["nta2020"].nunique()),
        "duplicate_key_report": duplicate_key_report(df, ["nta2020"], "OSM POI table"),
        "has_nta_area_km2": "nta_area_km2" in df.columns,
        "poi_total_count_sum": float(pd.to_numeric(df.get("poi_total_count", pd.Series(dtype=float)), errors="coerce").sum()) if "poi_total_count" in df.columns else None,
    }
    return df, info


def prepare_pluto(path: Path, osm_df: pd.DataFrame, strict: bool = True) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    df = pd.read_csv(path)
    require_columns(df, PLUTO_REQUIRED_COLS, "PLUTO table")
    df = df.copy()
    df["nta2020"] = as_string_key(df["nta2020"])
    df = df.dropna(subset=["nta2020"])
    fail_if_duplicate_keys(df, ["nta2020"], "PLUTO table")

    # Keep temporal as canonical source for names.
    drop_identity = [c for c in ["ntaname", "boroname", "borocode", "cdta2020", "cdtaname"] if c in df.columns]
    df = df.drop(columns=drop_identity)

    area_consistency_rows = []
    if "nta_area_km2" in df.columns and "nta_area_km2" in osm_df.columns:
        area_cmp = osm_df[["nta2020", "nta_area_km2"]].merge(
            df[["nta2020", "nta_area_km2"]], on="nta2020", how="outer", suffixes=("_osm", "_pluto")
        )
        area_cmp["area_abs_diff_km2"] = (pd.to_numeric(area_cmp["nta_area_km2_osm"], errors="coerce") - pd.to_numeric(area_cmp["nta_area_km2_pluto"], errors="coerce")).abs()
        area_consistency_rows = area_cmp.to_dict("records")
        # Avoid duplicate area columns. OSM area is retained as canonical because Step 7 already uses it.
        df = df.drop(columns=["nta_area_km2"])
    elif "nta_area_km2" in df.columns and "nta_area_km2" not in osm_df.columns:
        # PLUTO will provide canonical area in this less common case.
        pass

    if "pluto_merge_key_present" not in df.columns:
        df["pluto_merge_key_present"] = 1

    # Detect PLUTO source-missing rows in the processed PLUTO table.
    anchor_candidates = [
        "pluto_lot_count",
        "pluto_parcel_count",
        "pluto_total_lot_area",
        "pluto_total_building_area",
        "pluto_total_units",
    ]
    anchors = [c for c in anchor_candidates if c in df.columns]
    if anchors:
        anchor_missing = df[anchors].isna().all(axis=1)
    else:
        pluto_cols = [c for c in df.columns if c.startswith("pluto_")]
        anchor_missing = df[pluto_cols].isna().all(axis=1) if pluto_cols else pd.Series(False, index=df.index)
    df["pluto_source_missing_flag"] = anchor_missing.astype("int8")

    info = {
        "rows": int(len(df)),
        "unique_nta2020": int(df["nta2020"].nunique()),
        "duplicate_key_report": duplicate_key_report(df, ["nta2020"], "PLUTO table"),
        "has_nta_area_km2_after_reconcile": "nta_area_km2" in df.columns,
        "pluto_source_missing_nta_count": int(df["pluto_source_missing_flag"].sum()),
        "pluto_source_missing_ntas": df.loc[df["pluto_source_missing_flag"] == 1, "nta2020"].astype(str).tolist(),
    }

    area_consistency = pd.DataFrame(area_consistency_rows)
    return df, info, area_consistency


def compute_imputation_values(weather: pd.DataFrame, osm: pd.DataFrame, pluto: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Precompute imputation defaults for merged exogenous features."""
    defaults: Dict[str, object] = {}
    rows = []

    def add_default(col: str, source: str, strategy: str, value: object) -> None:
        defaults[col] = value
        rows.append({"column": col, "source": source, "strategy": strategy, "value": value})

    # Weather: if a week ever fails to merge, use median for continuous/count weather features.
    for col in weather.columns:
        if col in ["week_start", "weather_merge_key_present"]:
            continue
        if pd.api.types.is_numeric_dtype(weather[col]):
            med = pd.to_numeric(weather[col], errors="coerce").median()
            if pd.isna(med):
                med = 0.0
            add_default(col, "weather", "median_if_missing_merge", float(med))
        elif col.startswith("weather_"):
            add_default(col, "weather", "unknown_if_missing_merge", "unknown")

    # OSM: missing NTA row means no observed OSM features; count/share/density defaults to zero.
    for col in osm.columns:
        if col in ["nta2020", "osm_merge_key_present"]:
            continue
        if pd.api.types.is_numeric_dtype(osm[col]):
            if col.startswith("poi_") or col == "nta_area_km2":
                add_default(col, "osm", "zero_for_missing_osm_or_zero_poi", 0.0)
        else:
            if col.startswith("poi_"):
                add_default(col, "osm", "none_for_missing_or_zero_poi", "none")

    # PLUTO: for the 2 NTA with missing PLUTO source, preserve flag and impute transparently.
    for col in pluto.columns:
        if col in ["nta2020", "pluto_merge_key_present", "pluto_source_missing_flag"]:
            continue
        if pd.api.types.is_numeric_dtype(pluto[col]):
            numeric = pd.to_numeric(pluto[col], errors="coerce")
            if is_likely_average_intensity_col(col):
                val = numeric.median()
                if pd.isna(val):
                    val = 0.0
                add_default(col, "pluto", "median_for_average_or_intensity", float(val))
            elif is_likely_ratio_share_col(col):
                add_default(col, "pluto", "zero_for_missing_ratio_share", 0.0)
            elif is_likely_count_sum_density_col(col):
                add_default(col, "pluto", "zero_for_missing_count_sum_density", 0.0)
            else:
                val = numeric.median()
                if pd.isna(val):
                    val = 0.0
                add_default(col, "pluto", "median_numeric_default", float(val))
        else:
            if col.startswith("pluto_"):
                add_default(col, "pluto", "unknown_categorical_default", "unknown")

    imputation_df = pd.DataFrame(rows)
    return defaults, imputation_df


# -----------------------------------------------------------------------------
# Chunk merge
# -----------------------------------------------------------------------------


def apply_defaults(df: pd.DataFrame, defaults: Dict[str, object]) -> pd.DataFrame:
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df


def standardize_temporal_chunk(chunk: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    chunk = chunk.copy()
    require_columns(chunk, TEMPORAL_REQUIRED_COLS, "temporal table chunk")
    chunk["nta2020"] = as_string_key(chunk["nta2020"])
    chunk["week_start"] = date_to_monday_week_start(chunk["week_start"])
    bad_week_start = int(chunk["week_start"].isna().sum())
    return chunk, bad_week_start


def add_quality_flags(
    df: pd.DataFrame,
    weather_cols: Sequence[str],
    osm_cols: Sequence[str],
    pluto_cols: Sequence[str],
) -> pd.DataFrame:
    # Merge-match flags
    df["weather_missing_flag"] = df.get("weather_merge_key_present", pd.Series(np.nan, index=df.index)).isna().astype("int8")
    df["osm_missing_flag"] = df.get("osm_merge_key_present", pd.Series(np.nan, index=df.index)).isna().astype("int8")
    df["pluto_table_missing_flag"] = df.get("pluto_merge_key_present", pd.Series(np.nan, index=df.index)).isna().astype("int8")

    # Source-level flags
    if "pluto_source_missing_flag" in df.columns:
        df["pluto_source_missing_flag"] = df["pluto_source_missing_flag"].fillna(1).astype("int8")
    else:
        df["pluto_source_missing_flag"] = df["pluto_table_missing_flag"].astype("int8")

    if "poi_total_count" in df.columns:
        df["osm_zero_poi_flag"] = (pd.to_numeric(df["poi_total_count"], errors="coerce").fillna(0) == 0).astype("int8")
    else:
        df["osm_zero_poi_flag"] = df["osm_missing_flag"].astype("int8")

    if "weather_complete_week_flag" in df.columns:
        wc = pd.to_numeric(df["weather_complete_week_flag"], errors="coerce")
        df["weather_incomplete_week_flag"] = (wc != 1).fillna(True).astype("int8")
    else:
        df["weather_incomplete_week_flag"] = df["weather_missing_flag"].astype("int8")

    # Merge-ready flag after external dimensions have matched.
    df["external_features_merged_flag"] = (
        (df["weather_missing_flag"] == 0)
        & (df["osm_missing_flag"] == 0)
        & (df["pluto_table_missing_flag"] == 0)
    ).astype("int8")

    # Temporal readiness.
    if "is_train_ready_temporal" in df.columns:
        temporal_ready = to_bool_int(df["is_train_ready_temporal"]).fillna(0).astype("int8")
    else:
        temporal_ready = pd.Series(1, index=df.index, dtype="int8")
        for col in ["target_next_week_count", "abnormal_increase_next_week"]:
            if col in df.columns:
                temporal_ready = (temporal_ready.astype(bool) & df[col].notna()).astype("int8")

    if "has_target_next_week" in df.columns:
        has_target = to_bool_int(df["has_target_next_week"]).fillna(0).astype("int8")
    else:
        has_target = df["target_next_week_count"].notna().astype("int8") if "target_next_week_count" in df.columns else pd.Series(0, index=df.index, dtype="int8")

    abnormal_ok = df["abnormal_increase_next_week"].notna().astype("int8") if "abnormal_increase_next_week" in df.columns else pd.Series(0, index=df.index, dtype="int8")
    target_ok = df["target_next_week_count"].notna().astype("int8") if "target_next_week_count" in df.columns else pd.Series(0, index=df.index, dtype="int8")

    df["final_train_ready_flag"] = (
        (temporal_ready == 1)
        & (has_target == 1)
        & (abnormal_ok == 1)
        & (target_ok == 1)
        & (df["external_features_merged_flag"] == 1)
    ).astype("int8")

    return df


def update_counter_from_series(counter: Counter, series: pd.Series) -> None:
    vc = series.value_counts(dropna=False)
    for k, v in vc.items():
        counter[str(k)] += int(v)


def increment_group_counter(counter: Counter, df: pd.DataFrame, group_cols: Sequence[str], value_col: Optional[str] = None) -> None:
    if not set(group_cols).issubset(df.columns):
        return
    if value_col and value_col in df.columns:
        grouped = df.groupby(list(group_cols), dropna=False)[value_col].sum()
    else:
        grouped = df.groupby(list(group_cols), dropna=False).size()
    for key, value in grouped.items():
        if not isinstance(key, tuple):
            key = (key,)
        counter[tuple(str(x) for x in key)] += int(value)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def build_final_dataset(args: argparse.Namespace) -> Dict[str, object]:
    start = time.time()
    project_root = resolve_path(Path.cwd(), args.project_root) if args.project_root else project_root_from_script()

    temporal_path = resolve_path(project_root, args.temporal_input)
    weather_path = resolve_path(project_root, args.weather_input)
    osm_path = resolve_path(project_root, args.osm_input)
    pluto_path = resolve_path(project_root, args.pluto_input)
    output_path = resolve_path(project_root, args.output)
    summary_dir = resolve_path(project_root, args.summary_dir)

    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for label, path in [
        ("temporal", temporal_path),
        ("weather", weather_path),
        ("osm", osm_path),
        ("pluto", pluto_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} input: {path}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    temporal_cols = read_header(temporal_path)
    require_columns(temporal_cols, TEMPORAL_REQUIRED_COLS, "temporal input")

    weather, weather_info = prepare_weather(weather_path, strict=args.strict)
    osm, osm_info = prepare_osm(osm_path, strict=args.strict)
    pluto, pluto_info, area_consistency = prepare_pluto(pluto_path, osm, strict=args.strict)

    # Save area consistency before merge, useful for confirming OSM/PLUTO share same NTA areas.
    area_consistency_path = summary_dir / f"{SCRIPT_NAME}_area_consistency.csv"
    if not area_consistency.empty:
        area_consistency.to_csv(area_consistency_path, index=False)
    else:
        pd.DataFrame([{"status": "not_applicable_or_only_one_area_source"}]).to_csv(area_consistency_path, index=False)

    defaults, imputation_df = compute_imputation_values(weather, osm, pluto)
    imputation_path = summary_dir / f"{SCRIPT_NAME}_imputation_defaults.csv"
    imputation_df.to_csv(imputation_path, index=False)

    weather_cols = [c for c in weather.columns if c != "week_start"]
    osm_cols = [c for c in osm.columns if c != "nta2020"]
    pluto_cols = [c for c in pluto.columns if c != "nta2020"]

    # Running summary accumulators.
    output_rows = 0
    input_rows_temporal = 0
    bad_temporal_week_start_rows = 0
    output_total_complaint_count = 0.0
    week_set: set[str] = set()
    nta_set: set[str] = set()
    category_set: set[str] = set()
    output_columns: Optional[List[str]] = None
    missing_counts: Counter = Counter()
    dtype_examples: Dict[str, str] = {}

    flag_counters: Dict[str, Counter] = defaultdict(Counter)
    time_split_counter: Counter = Counter()
    target_counter: Counter = Counter()
    target_by_split: Counter = Counter()
    target_by_category: Counter = Counter()
    target_by_year: Counter = Counter()
    target_by_split_category: Counter = Counter()
    rows_by_split: Counter = Counter()
    rows_by_category: Counter = Counter()
    rows_by_year: Counter = Counter()

    # Keep only unique missing NTA/week values for merge diagnostics.
    missing_weather_weeks: set[str] = set()
    missing_osm_ntas: set[str] = set()
    missing_pluto_table_ntas: set[str] = set()
    pluto_source_missing_ntas_seen: set[str] = set()
    osm_zero_poi_ntas_seen: set[str] = set()

    first_chunk = True
    chunk_index = 0

    # Open gzip output once to safely append chunk CSVs.
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as out_f:
        reader = pd.read_csv(temporal_path, chunksize=args.chunksize, low_memory=False)
        for chunk in reader:
            chunk_index += 1
            raw_chunk_rows = int(len(chunk))
            input_rows_temporal += raw_chunk_rows

            chunk, bad_dates = standardize_temporal_chunk(chunk)
            bad_temporal_week_start_rows += bad_dates
            if bad_dates:
                chunk = chunk.dropna(subset=["week_start"])

            chunk["nta2020"] = as_string_key(chunk["nta2020"])

            merged = chunk.merge(weather, on="week_start", how="left", validate="many_to_one")
            merged["target_week"] = merged["week_start"] + pd.Timedelta(days=7)
            if "has_target_next_week" in merged.columns:
                has_target_for_week = to_bool_int(merged["has_target_next_week"]).fillna(0).astype("int8") == 1
                merged.loc[~has_target_for_week, "target_week"] = pd.NaT
            merged["target_year"] = pd.to_datetime(merged["target_week"], errors="coerce").dt.year.astype("Int16")
            merged = merged.merge(osm, on="nta2020", how="left", validate="many_to_one")
            merged = merged.merge(pluto, on="nta2020", how="left", validate="many_to_one")

            merged = add_quality_flags(merged, weather_cols, osm_cols, pluto_cols)
            merged = apply_defaults(merged, defaults)
            merged["time_split"] = infer_time_split(merged["target_week"])

            # Convert merge-key-present indicators after flags/defaults.
            for col in ["weather_merge_key_present", "osm_merge_key_present", "pluto_merge_key_present"]:
                if col in merged.columns:
                    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype("int8")

            # Normalize flag columns.
            for col in [c for c in merged.columns if c.endswith("_flag") or c.startswith("has_") or c.startswith("is_")]:
                # Avoid converting non-binary semantic columns accidentally.
                if col in merged.columns and col not in ["is_covid_period", "is_year_start", "is_year_end", "is_train_ready_temporal"]:
                    if pd.api.types.is_numeric_dtype(merged[col]) or pd.api.types.is_bool_dtype(merged[col]):
                        # Keep floats if they are not binary-like.
                        non_na = pd.to_numeric(merged[col], errors="coerce").dropna()
                        if len(non_na) == 0 or set(non_na.unique()).issubset({0, 1}):
                            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype("int8")

            # Summary accumulators.
            output_rows += int(len(merged))
            if "complaint_count" in merged.columns:
                output_total_complaint_count += float(pd.to_numeric(merged["complaint_count"], errors="coerce").fillna(0).sum())

            week_set.update(pd.to_datetime(merged["week_start"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().astype(str).unique())
            nta_set.update(merged["nta2020"].astype(str).dropna().unique())
            category_set.update(merged["complaint_category"].astype(str).dropna().unique())

            if output_columns is None:
                output_columns = list(merged.columns)
                for col in output_columns:
                    dtype_examples[col] = str(merged[col].dtype)
            else:
                # Defensive check: every chunk should have identical columns.
                if list(merged.columns) != output_columns:
                    raise RuntimeError("Merged chunk columns changed across chunks; check input schemas.")

            missing_counts.update({col: int(merged[col].isna().sum()) for col in merged.columns})

            for flag_col in [
                "weather_missing_flag",
                "weather_incomplete_week_flag",
                "osm_missing_flag",
                "osm_zero_poi_flag",
                "pluto_table_missing_flag",
                "pluto_source_missing_flag",
                "external_features_merged_flag",
                "final_train_ready_flag",
            ]:
                if flag_col in merged.columns:
                    update_counter_from_series(flag_counters[flag_col], merged[flag_col])

            update_counter_from_series(time_split_counter, merged["time_split"])
            if "abnormal_increase_next_week" in merged.columns:
                update_counter_from_series(target_counter, merged["abnormal_increase_next_week"])
                increment_group_counter(target_by_split, merged, ["time_split", "abnormal_increase_next_week"])
                increment_group_counter(target_by_category, merged, ["complaint_category", "abnormal_increase_next_week"])
                if "target_year" in merged.columns:
                    increment_group_counter(target_by_year, merged, ["target_year", "abnormal_increase_next_week"])
                increment_group_counter(target_by_split_category, merged, ["time_split", "complaint_category", "abnormal_increase_next_week"])

            increment_group_counter(rows_by_split, merged, ["time_split"])
            increment_group_counter(rows_by_category, merged, ["complaint_category"])
            if "year" in merged.columns:
                increment_group_counter(rows_by_year, merged, ["year"])

            if "weather_missing_flag" in merged.columns:
                miss_w = merged.loc[merged["weather_missing_flag"] == 1, "week_start"]
                missing_weather_weeks.update(pd.to_datetime(miss_w, errors="coerce").dt.strftime("%Y-%m-%d").dropna().astype(str).unique())
            if "osm_missing_flag" in merged.columns:
                missing_osm_ntas.update(merged.loc[merged["osm_missing_flag"] == 1, "nta2020"].astype(str).unique())
            if "pluto_table_missing_flag" in merged.columns:
                missing_pluto_table_ntas.update(merged.loc[merged["pluto_table_missing_flag"] == 1, "nta2020"].astype(str).unique())
            if "pluto_source_missing_flag" in merged.columns:
                pluto_source_missing_ntas_seen.update(merged.loc[merged["pluto_source_missing_flag"] == 1, "nta2020"].astype(str).unique())
            if "osm_zero_poi_flag" in merged.columns:
                osm_zero_poi_ntas_seen.update(merged.loc[merged["osm_zero_poi_flag"] == 1, "nta2020"].astype(str).unique())

            # Store dates as ISO strings for portable CSV output.
            for date_col in ["week_start", "week_end", "target_week", "weather_week_end"]:
                if date_col in merged.columns:
                    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

            merged.to_csv(out_f, index=False, header=first_chunk)
            first_chunk = False

            if args.progress and chunk_index % args.progress_every == 0:
                print(
                    f"[progress] chunks={chunk_index}, rows_out={output_rows:,}, "
                    f"weeks={len(week_set)}, ntas={len(nta_set)}, categories={len(category_set)}",
                    flush=True,
                )

    # Build summary files.
    elapsed = time.time() - start

    if output_columns is None:
        output_columns = []

    missing_df = pd.DataFrame([
        {
            "column": col,
            "missing_count": int(missing_counts.get(col, 0)),
            "missing_share": float(missing_counts.get(col, 0) / output_rows) if output_rows else None,
            "dtype_example": dtype_examples.get(col),
            "feature_group": classify_feature_column(col),
        }
        for col in output_columns
    ]).sort_values(["missing_count", "column"], ascending=[False, True])
    missing_path = summary_dir / f"{SCRIPT_NAME}_missing_values.csv"
    missing_df.to_csv(missing_path, index=False)

    feature_groups_df = pd.DataFrame([
        {"column": col, "feature_group": classify_feature_column(col), "dtype_example": dtype_examples.get(col)}
        for col in output_columns
    ])
    feature_groups_path = summary_dir / f"{SCRIPT_NAME}_feature_groups.csv"
    feature_groups_df.to_csv(feature_groups_path, index=False)

    merge_coverage_rows = [
        {
            "source": "weather_weekly",
            "merge_key": "week_start",
            "dimension_rows": weather_info["rows"],
            "dimension_unique_keys": weather_info["unique_week_start"],
            "final_rows_missing_merge": int(flag_counters["weather_missing_flag"].get("1", 0)),
            "unique_missing_keys": len(missing_weather_weeks),
            "missing_keys_sample": "; ".join(sorted(list(missing_weather_weeks))[:20]),
        },
        {
            "source": "osm_poi_nta",
            "merge_key": "nta2020",
            "dimension_rows": osm_info["rows"],
            "dimension_unique_keys": osm_info["unique_nta2020"],
            "final_rows_missing_merge": int(flag_counters["osm_missing_flag"].get("1", 0)),
            "unique_missing_keys": len(missing_osm_ntas),
            "missing_keys_sample": "; ".join(sorted(list(missing_osm_ntas))[:20]),
        },
        {
            "source": "pluto_nta_landuse",
            "merge_key": "nta2020",
            "dimension_rows": pluto_info["rows"],
            "dimension_unique_keys": pluto_info["unique_nta2020"],
            "final_rows_missing_merge": int(flag_counters["pluto_table_missing_flag"].get("1", 0)),
            "unique_missing_keys": len(missing_pluto_table_ntas),
            "missing_keys_sample": "; ".join(sorted(list(missing_pluto_table_ntas))[:20]),
        },
        {
            "source": "pluto_source_coverage",
            "merge_key": "nta2020",
            "dimension_rows": pluto_info["rows"],
            "dimension_unique_keys": pluto_info["unique_nta2020"],
            "final_rows_missing_merge": int(flag_counters["pluto_source_missing_flag"].get("1", 0)),
            "unique_missing_keys": len(pluto_source_missing_ntas_seen),
            "missing_keys_sample": "; ".join(sorted(list(pluto_source_missing_ntas_seen))[:20]),
        },
        {
            "source": "osm_zero_poi_coverage",
            "merge_key": "nta2020",
            "dimension_rows": osm_info["rows"],
            "dimension_unique_keys": osm_info["unique_nta2020"],
            "final_rows_missing_merge": int(flag_counters["osm_zero_poi_flag"].get("1", 0)),
            "unique_missing_keys": len(osm_zero_poi_ntas_seen),
            "missing_keys_sample": "; ".join(sorted(list(osm_zero_poi_ntas_seen))[:20]),
        },
    ]
    merge_coverage_path = summary_dir / f"{SCRIPT_NAME}_merge_coverage.csv"
    pd.DataFrame(merge_coverage_rows).to_csv(merge_coverage_path, index=False)

    def counter_to_df(counter: Counter, cols: Sequence[str], value_name: str = "rows") -> pd.DataFrame:
        rows = []
        for key, value in counter.items():
            if isinstance(key, tuple):
                row = {c: key[i] if i < len(key) else None for i, c in enumerate(cols)}
            else:
                row = {cols[0]: key}
            row[value_name] = int(value)
            rows.append(row)
        return pd.DataFrame(rows)

    train_ready_path = summary_dir / f"{SCRIPT_NAME}_train_ready_summary.csv"
    train_ready_rows = []
    for flag_col, counter in flag_counters.items():
        total = sum(counter.values())
        for value, count in counter.items():
            train_ready_rows.append({
                "flag": flag_col,
                "value": value,
                "rows": int(count),
                "share": float(count / total) if total else None,
            })
    pd.DataFrame(train_ready_rows).to_csv(train_ready_path, index=False)

    target_by_split_path = summary_dir / f"{SCRIPT_NAME}_target_balance_by_split.csv"
    counter_to_df(target_by_split, ["time_split", "abnormal_increase_next_week"]).to_csv(target_by_split_path, index=False)

    target_by_category_path = summary_dir / f"{SCRIPT_NAME}_target_balance_by_category.csv"
    counter_to_df(target_by_category, ["complaint_category", "abnormal_increase_next_week"]).to_csv(target_by_category_path, index=False)

    target_by_year_path = summary_dir / f"{SCRIPT_NAME}_target_balance_by_year.csv"
    counter_to_df(target_by_year, ["target_year", "abnormal_increase_next_week"]).to_csv(target_by_year_path, index=False)

    target_by_split_category_path = summary_dir / f"{SCRIPT_NAME}_target_balance_by_split_category.csv"
    counter_to_df(target_by_split_category, ["time_split", "complaint_category", "abnormal_increase_next_week"]).to_csv(target_by_split_category_path, index=False)

    rows_by_split_path = summary_dir / f"{SCRIPT_NAME}_rows_by_split.csv"
    counter_to_df(rows_by_split, ["time_split"]).to_csv(rows_by_split_path, index=False)

    # Lightweight final profile by feature group.
    feature_group_counts = feature_groups_df["feature_group"].value_counts().rename_axis("feature_group").reset_index(name="column_count")
    feature_group_counts_path = summary_dir / f"{SCRIPT_NAME}_feature_group_counts.csv"
    feature_group_counts.to_csv(feature_group_counts_path, index=False)

    # Summary JSON.
    output_file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else None
    final_train_ready_rows = int(flag_counters["final_train_ready_flag"].get("1", 0))
    external_merged_rows = int(flag_counters["external_features_merged_flag"].get("1", 0))
    weather_missing_rows = int(flag_counters["weather_missing_flag"].get("1", 0))
    osm_missing_rows = int(flag_counters["osm_missing_flag"].get("1", 0))
    pluto_table_missing_rows = int(flag_counters["pluto_table_missing_flag"].get("1", 0))
    pluto_source_missing_rows = int(flag_counters["pluto_source_missing_flag"].get("1", 0))
    osm_zero_poi_rows = int(flag_counters["osm_zero_poi_flag"].get("1", 0))

    summary = {
        "script": SCRIPT_NAME,
        "status": "done",
        "project_root": str(project_root),
        "inputs": {
            "temporal": str(temporal_path),
            "weather": str(weather_path),
            "osm": str(osm_path),
            "pluto": str(pluto_path),
        },
        "output_file": str(output_path),
        "output_file_size_mb": round(output_file_size_mb, 3) if output_file_size_mb is not None else None,
        "summary_dir": str(summary_dir),
        "chunksize": int(args.chunksize),
        "input_rows_temporal": int(input_rows_temporal),
        "bad_temporal_week_start_rows": int(bad_temporal_week_start_rows),
        "output_rows": int(output_rows),
        "row_count_preserved": bool(output_rows == input_rows_temporal - bad_temporal_week_start_rows),
        "output_columns_count": int(len(output_columns)),
        "output_columns": output_columns,
        "output_total_complaint_count": int(round(output_total_complaint_count)),
        "output_unique_nta2020": int(len(nta_set)),
        "output_unique_weeks": int(len(week_set)),
        "output_unique_categories": int(len(category_set)),
        "output_min_week_start": min(week_set) if week_set else None,
        "output_max_week_start": max(week_set) if week_set else None,
        "weather_info": weather_info,
        "osm_info": osm_info,
        "pluto_info": pluto_info,
        "external_merge_coverage": {
            "external_merged_rows": external_merged_rows,
            "external_merged_share": float(external_merged_rows / output_rows) if output_rows else None,
            "weather_missing_rows": weather_missing_rows,
            "weather_missing_unique_weeks": len(missing_weather_weeks),
            "osm_missing_rows": osm_missing_rows,
            "osm_missing_unique_ntas": len(missing_osm_ntas),
            "pluto_table_missing_rows": pluto_table_missing_rows,
            "pluto_table_missing_unique_ntas": len(missing_pluto_table_ntas),
            "pluto_source_missing_rows": pluto_source_missing_rows,
            "pluto_source_missing_unique_ntas": len(pluto_source_missing_ntas_seen),
            "osm_zero_poi_rows": osm_zero_poi_rows,
            "osm_zero_poi_unique_ntas": len(osm_zero_poi_ntas_seen),
        },
        "time_split_counts": dict(time_split_counter),
        "time_split_policy": "Rows are assigned by target_week = week_start + 7 days, not by feature week.",
        "target_balance": dict(target_counter),
        "final_train_ready_rows": final_train_ready_rows,
        "final_train_ready_share": float(final_train_ready_rows / output_rows) if output_rows else None,
        "feature_group_counts": dict(zip(feature_group_counts["feature_group"], feature_group_counts["column_count"])),
        "summary_files": {
            "summary_json": str(summary_dir / f"{SCRIPT_NAME}_summary.json"),
            "missing_values": str(missing_path),
            "feature_groups": str(feature_groups_path),
            "feature_group_counts": str(feature_group_counts_path),
            "merge_coverage": str(merge_coverage_path),
            "train_ready_summary": str(train_ready_path),
            "target_balance_by_split": str(target_by_split_path),
            "target_balance_by_category": str(target_by_category_path),
            "target_balance_by_year": str(target_by_year_path),
            "target_balance_by_split_category": str(target_by_split_category_path),
            "rows_by_split": str(rows_by_split_path),
            "imputation_defaults": str(imputation_path),
            "area_consistency": str(area_consistency_path),
        },
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_minutes": round(elapsed / 60, 3),
    }

    summary_path = summary_dir / f"{SCRIPT_NAME}_summary.json"
    safe_json_dump(summary, summary_path)

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final NYC urban service-demand modeling dataset.")
    parser.add_argument("--project-root", type=Path, default=None, help="Project root. Defaults to parent of code_processdata/.")
    parser.add_argument("--temporal-input", type=Path, default=DEFAULT_TEMPORAL_INPUT)
    parser.add_argument("--weather-input", type=Path, default=DEFAULT_WEATHER_INPUT)
    parser.add_argument("--osm-input", type=Path, default=DEFAULT_OSM_INPUT)
    parser.add_argument("--pluto-input", type=Path, default=DEFAULT_PLUTO_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--chunksize", type=int, default=200_000, help="Rows per temporal chunk.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    parser.add_argument("--strict", action="store_true", help="Fail on bad dates in dimension tables.")
    parser.add_argument("--progress", action="store_true", help="Print progress after every N chunks.")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_final_dataset(args)
    except Exception as exc:
        print(f"[ERROR] {SCRIPT_NAME} failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "script": summary["script"],
        "status": summary["status"],
        "output_file": summary["output_file"],
        "output_rows": summary["output_rows"],
        "output_columns_count": summary["output_columns_count"],
        "row_count_preserved": summary["row_count_preserved"],
        "final_train_ready_rows": summary["final_train_ready_rows"],
        "external_merge_coverage": summary["external_merge_coverage"],
        "elapsed_minutes": summary["elapsed_minutes"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
