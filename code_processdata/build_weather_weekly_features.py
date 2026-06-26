#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
build_weather_weekly_features.py

Step 6 for the SIMC NYC pipeline.

FIXED VERSION: avoids detecting derived indicator columns such as rain_day/snow_day as continuous PRCP/SNOW measurements.

Purpose
-------
Aggregate NOAA daily NYC weather data into weekly weather features that can be
merged into the final NTA x week_start x complaint_category panel.

Design principles
-----------------
1. Use PROJECT_ROOT = Path(__file__).resolve().parents[1], not Path.cwd().
2. Keep raw data untouched under data/raw/.
3. Output model-ready feature tables under data/processed/feature_tables/.
4. Write diagnostics under data/processed/_feature_summaries/.
5. Make column detection robust because NOAA files may be exported with either
   canonical GHCN names (DATE, TMAX, PRCP, ...) or friendlier names.
6. Aggregate to Monday-based week_start to match the weekly 311 panel.

Expected input
--------------
data/raw/noaa_weather/noaa_nyc_daily_weather_2015_2025.csv.gz

Expected output
---------------
data/processed/feature_tables/noaa_weather_weekly_features.csv

Main merge key
--------------
week_start

Example command from project root
---------------------------------
.\.venv\Scripts\python.exe .\code_processdata\build_weather_weekly_features.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "noaa_weather"
    / "noaa_nyc_daily_weather_2015_2025.csv.gz"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_tables"
    / "noaa_weather_weekly_features.csv"
)
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "data" / "processed" / "_feature_summaries"

SCRIPT_NAME = "build_weather_weekly_features"


# -----------------------------------------------------------------------------
# Constants and column detection settings
# -----------------------------------------------------------------------------
COMMON_MISSING_SENTINELS = {
    -9999,
    -9999.0,
    9999,
    9999.0,
    99999,
    99999.0,
    999999,
    999999.0,
}

DATE_CANDIDATES = [
    "date",
    "DATE",
    "datetime",
    "DATETIME",
    "day",
    "DAY",
    "time",
    "TIME",
]

# Regexes intentionally target column names, not values.
# They avoid common NOAA metadata columns such as *_ATTRIBUTES, MFLAG, QFLAG, SFLAG.
WEATHER_FAMILY_REGEX: Dict[str, List[str]] = {
    "tmax": [
        r"(^|[^a-z0-9])tmax($|[^a-z0-9])",
        r"max[_\s\-]*temp",
        r"temp[_\s\-]*max",
        r"maximum[_\s\-]*temperature",
    ],
    "tmin": [
        r"(^|[^a-z0-9])tmin($|[^a-z0-9])",
        r"min[_\s\-]*temp",
        r"temp[_\s\-]*min",
        r"minimum[_\s\-]*temperature",
    ],
    "tavg": [
        r"(^|[^a-z0-9])tavg($|[^a-z0-9])",
        r"avg[_\s\-]*temp",
        r"temp[_\s\-]*avg",
        r"mean[_\s\-]*temp",
        r"average[_\s\-]*temperature",
    ],
    "prcp": [
        r"(^|[^a-z0-9])prcp($|[^a-z0-9])",
        r"precip",
        r"precipitation",
        r"rainfall",
        r"(^|[^a-z0-9])rain($|[^a-z0-9])",
    ],
    "snow": [
        r"(^|[^a-z0-9])snow($|[^a-z0-9])",
        r"snowfall",
    ],
    "snwd": [
        r"(^|[^a-z0-9])snwd($|[^a-z0-9])",
        r"snow[_\s\-]*depth",
    ],
    "awnd": [
        r"(^|[^a-z0-9])awnd($|[^a-z0-9])",
        r"avg[_\s\-]*wind",
        r"average[_\s\-]*wind",
        r"wind[_\s\-]*speed",
        r"(^|[^a-z0-9])wspd($|[^a-z0-9])",
    ],
}

EXCLUDE_COLUMN_REGEX = re.compile(
    r"flag|mflag|qflag|sflag|attribute|attributes|quality|station|name|id$|identifier|source|datatype|units?",
    flags=re.IGNORECASE,
)

DISPLAY_FAMILY = {
    "tmax": "maximum temperature",
    "tmin": "minimum temperature",
    "tavg": "average temperature",
    "prcp": "precipitation",
    "snow": "snowfall",
    "snwd": "snow depth",
    "awnd": "average wind speed",
}


# Prefer exact raw NOAA/friendly weather columns before broad regex matching.
# This is important because the NOAA daily file may already contain derived
# binary indicators such as rain_day, heavy_rain_day_25mm, snow_day, etc.
# Those columns must NOT be mixed into continuous PRCP/SNOW aggregation.
CANONICAL_FAMILY_COLUMNS: Dict[str, List[str]] = {
    "tmax": ["TMAX", "tmax", "max_temp", "maximum_temperature"],
    "tmin": ["TMIN", "tmin", "min_temp", "minimum_temperature"],
    "tavg": ["TAVG", "TAVG_est", "tavg", "tavg_est", "avg_temp", "mean_temp"],
    "prcp": ["PRCP", "prcp", "precipitation", "rainfall"],
    "snow": ["SNOW", "snow", "snowfall"],
    "snwd": ["SNWD", "snwd", "snow_depth"],
    "awnd": ["AWND", "awnd", "wind_speed", "avg_wind_speed", "average_wind_speed"],
}

# Exclude already-derived event/indicator columns from raw weather-family detection.
# Examples that must be ignored for continuous families:
# rain_day, heavy_rain_day_25mm, snow_day, hot_day_30c, cold_day_0c.
DERIVED_WEATHER_FEATURE_REGEX = re.compile(
    r"""
    (^|[_\-\s])(
        day|days|count|counts|flag|indicator|event|events|
        hot|cold|ice|rain|snow|heavy|moderate|very
    )([_\-\s]|$)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)



@dataclass
class UnitDecision:
    family: str
    requested_unit: str
    inferred_unit: str
    conversion: str
    n_valid: int
    min_value_before: Optional[float]
    p50_before: Optional[float]
    p99_before: Optional[float]
    max_value_before: Optional[float]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[INFO] {message}")


def normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def safe_float(x: object) -> Optional[float]:
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def to_numeric_clean(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not s.empty:
        s = s.mask(s.isin(COMMON_MISSING_SENTINELS))
        # Keep plausible zeros and negative temperatures. Remove extreme placeholders.
        s = s.mask(s >= 9990)
        s = s.mask(s <= -9990)
    return s


def infer_date_column(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in df.columns:
            fail(f"Requested --date-col '{explicit}' was not found. Available columns: {list(df.columns)}")
        return explicit

    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c

    # Flexible fallback: first column whose normalized name is date-like.
    for c in df.columns:
        nc = normalize_col_name(c)
        if nc in {"date", "datetime", "day", "time", "observation_date"}:
            return c
        if nc.endswith("_date") or nc.startswith("date_"):
            return c

    fail("Could not infer date column. Use --date-col to specify it explicitly.")
    raise AssertionError("unreachable")


def is_candidate_numeric_column(df: pd.DataFrame, col: str, min_valid_ratio: float = 0.01) -> bool:
    s = to_numeric_clean(df[col])
    if len(s) == 0:
        return False
    return float(s.notna().mean()) >= min_valid_ratio


def detect_family_columns(df: pd.DataFrame, family: str) -> List[str]:
    """Detect raw weather columns for a family.

    Detection is intentionally conservative:
    1. Prefer exact canonical raw columns such as PRCP, SNOW, TMAX.
    2. Only if no canonical column exists, use broader regex matching.
    3. Never use already-derived event/indicator columns such as rain_day,
       heavy_rain_day_25mm, snow_day, hot_day_30c, etc. as continuous weather
       measurements.
    """
    out: List[str] = []

    # Step 1: exact canonical/friendly names first.
    canonical_names = CANONICAL_FAMILY_COLUMNS.get(family, [])
    canonical_lookup = {normalize_col_name(c): c for c in canonical_names}

    for col in df.columns:
        col_str = str(col)
        col_norm = normalize_col_name(col_str)
        if col_norm in canonical_lookup and is_candidate_numeric_column(df, col):
            out.append(col)

    if out:
        return list(dict.fromkeys(out))

    # Step 2: fallback regex matching, but exclude derived indicator columns.
    patterns = WEATHER_FAMILY_REGEX[family]
    for col in df.columns:
        col_str = str(col)
        col_low = col_str.lower()
        col_norm = normalize_col_name(col_str)

        if EXCLUDE_COLUMN_REGEX.search(col_low):
            continue

        if DERIVED_WEATHER_FEATURE_REGEX.search(col_low) or DERIVED_WEATHER_FEATURE_REGEX.search(col_norm):
            continue

        matched = False
        for pat in patterns:
            if re.search(pat, col_low, flags=re.IGNORECASE) or re.search(pat, col_norm, flags=re.IGNORECASE):
                matched = True
                break

        if matched and is_candidate_numeric_column(df, col):
            out.append(col)

    return list(dict.fromkeys(out))


def series_stats_before_conversion(values: pd.DataFrame) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    arr = values.to_numpy(dtype=float, copy=False).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0, None, None, None, None
    return (
        int(arr.size),
        safe_float(np.min(arr)),
        safe_float(np.quantile(arr, 0.50)),
        safe_float(np.quantile(arr, 0.99)),
        safe_float(np.max(arr)),
    )


def normalize_temperature(values: pd.DataFrame, requested_unit: str, family: str) -> Tuple[pd.DataFrame, UnitDecision]:
    n_valid, min_v, p50, p99, max_v = series_stats_before_conversion(values)
    unit = requested_unit

    if requested_unit == "auto":
        if n_valid == 0:
            unit = "unknown"
        elif max_v is not None and max_v > 80:
            # NOAA GHCN daily TMAX/TMIN/TAVG are commonly stored as tenths of deg C.
            unit = "tenths_c"
        elif p99 is not None and p99 > 45:
            # Friendly exports sometimes use Fahrenheit.
            unit = "fahrenheit"
        else:
            unit = "celsius"

    if unit == "tenths_c":
        converted = values / 10.0
        conversion = "divide_by_10_to_celsius"
    elif unit == "fahrenheit":
        converted = (values - 32.0) * 5.0 / 9.0
        conversion = "fahrenheit_to_celsius"
    elif unit == "celsius":
        converted = values.copy()
        conversion = "none_celsius"
    elif unit == "unknown":
        converted = values.copy()
        conversion = "none_unknown_no_valid_values"
    else:
        fail(f"Unsupported temperature unit: {requested_unit}")

    decision = UnitDecision(
        family=family,
        requested_unit=requested_unit,
        inferred_unit=unit,
        conversion=conversion,
        n_valid=n_valid,
        min_value_before=min_v,
        p50_before=p50,
        p99_before=p99,
        max_value_before=max_v,
    )
    return converted, decision


def normalize_depth_mm(values: pd.DataFrame, requested_unit: str, family: str) -> Tuple[pd.DataFrame, UnitDecision]:
    n_valid, min_v, p50, p99, max_v = series_stats_before_conversion(values)
    unit = requested_unit

    if requested_unit == "auto":
        if n_valid == 0:
            unit = "unknown"
        elif p99 is not None and p99 <= 20 and max_v is not None and max_v <= 80:
            # Could be either mm in a dry climate or inches. For NYC daily rainfall/snow,
            # preserving as mm is the safer default unless the user explicitly passes inches.
            unit = "mm"
        elif p99 is not None and p99 > 300:
            # NOAA GHCN PRCP/SNOW/SNWD are commonly tenths of mm.
            unit = "tenths_mm"
        elif max_v is not None and max_v > 600:
            unit = "tenths_mm"
        else:
            unit = "mm"

    if unit == "tenths_mm":
        converted = values / 10.0
        conversion = "divide_by_10_to_mm"
    elif unit == "inches":
        converted = values * 25.4
        conversion = "inches_to_mm"
    elif unit == "mm":
        converted = values.copy()
        conversion = "none_mm"
    elif unit == "unknown":
        converted = values.copy()
        conversion = "none_unknown_no_valid_values"
    else:
        fail(f"Unsupported depth unit for {family}: {requested_unit}")

    decision = UnitDecision(
        family=family,
        requested_unit=request_unit_for_summary(requested_unit),
        inferred_unit=unit,
        conversion=conversion,
        n_valid=n_valid,
        min_value_before=min_v,
        p50_before=p50,
        p99_before=p99,
        max_value_before=max_v,
    )
    return converted, decision


def request_unit_for_summary(unit: str) -> str:
    return unit


def normalize_wind(values: pd.DataFrame, requested_unit: str, family: str = "awnd") -> Tuple[pd.DataFrame, UnitDecision]:
    n_valid, min_v, p50, p99, max_v = series_stats_before_conversion(values)
    unit = requested_unit

    if requested_unit == "auto":
        if n_valid == 0:
            unit = "unknown"
        elif p99 is not None and p99 > 50:
            # NOAA GHCN AWND is commonly tenths of m/s.
            unit = "tenths_mps"
        else:
            unit = "mps"

    if unit == "tenths_mps":
        converted = values / 10.0
        conversion = "divide_by_10_to_mps"
    elif unit == "mph":
        converted = values * 0.44704
        conversion = "mph_to_mps"
    elif unit == "mps":
        converted = values.copy()
        conversion = "none_mps"
    elif unit == "unknown":
        converted = values.copy()
        conversion = "none_unknown_no_valid_values"
    else:
        fail(f"Unsupported wind unit: {requested_unit}")

    decision = UnitDecision(
        family=family,
        requested_unit=requested_unit,
        inferred_unit=unit,
        conversion=conversion,
        n_valid=n_valid,
        min_value_before=min_v,
        p50_before=p50,
        p99_before=p99,
        max_value_before=max_v,
    )
    return converted, decision


def normalize_family_values(
    raw_df: pd.DataFrame,
    family: str,
    cols: List[str],
    temperature_unit: str,
    precipitation_unit: str,
    snow_unit: str,
    wind_unit: str,
) -> Tuple[pd.DataFrame, UnitDecision]:
    values = pd.DataFrame(index=raw_df.index)
    for c in cols:
        values[c] = to_numeric_clean(raw_df[c])

    if family in {"tmax", "tmin", "tavg"}:
        return normalize_temperature(values, temperature_unit, family)
    if family == "prcp":
        return normalize_depth_mm(values, precipitation_unit, family)
    if family in {"snow", "snwd"}:
        return normalize_depth_mm(values, snow_unit, family)
    if family == "awnd":
        return normalize_wind(values, wind_unit, family)

    fail(f"Unknown weather family: {family}")
    raise AssertionError("unreachable")


def monday_week_start(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates, errors="coerce")
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def count_condition(s: pd.Series, condition) -> float:
    # If the whole series is missing, return NaN rather than 0 to avoid falsely
    # implying that an unavailable measurement had no events.
    if s.notna().sum() == 0:
        return np.nan
    return float(condition(s).sum())


def safe_sum(s: pd.Series) -> float:
    if s.notna().sum() == 0:
        return np.nan
    return float(s.sum(skipna=True))


def safe_mean(s: pd.Series) -> float:
    if s.notna().sum() == 0:
        return np.nan
    return float(s.mean(skipna=True))


def safe_min(s: pd.Series) -> float:
    if s.notna().sum() == 0:
        return np.nan
    return float(s.min(skipna=True))


def safe_max(s: pd.Series) -> float:
    if s.notna().sum() == 0:
        return np.nan
    return float(s.max(skipna=True))


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------
def build_daily_city_weather(
    raw_df: pd.DataFrame,
    date_col: str,
    detected_cols: Dict[str, List[str]],
    temperature_unit: str,
    precipitation_unit: str,
    snow_unit: str,
    wind_unit: str,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], List[Dict[str, object]]]:
    """Build one city-level daily weather row per date.

    The function supports both wide daily files and long station-date files:
    - Wide case: one row per date, multiple weather columns.
    - Long case: multiple rows per date, e.g. station-date rows.

    For each family, row-level values are computed by averaging detected columns
    in that row. Then rows are grouped by date to produce a city-level daily mean,
    min, max, and observation coverage.
    """
    work = pd.DataFrame(index=raw_df.index)
    work["date"] = pd.to_datetime(raw_df[date_col], errors="coerce").dt.normalize()

    unit_decisions: List[Dict[str, object]] = []
    detection_rows: List[Dict[str, object]] = []

    for family in WEATHER_FAMILY_REGEX:
        cols = detected_cols.get(family, [])
        detection_rows.append(
            {
                "family": family,
                "display_name": DISPLAY_FAMILY.get(family, family),
                "detected_column_count": len(cols),
                "detected_columns": " | ".join(map(str, cols)),
            }
        )

        if not cols:
            continue

        normalized_values, decision = normalize_family_values(
            raw_df=raw_df,
            family=family,
            cols=cols,
            temperature_unit=temperature_unit,
            precipitation_unit=precipitation_unit,
            snow_unit=snow_unit,
            wind_unit=wind_unit,
        )
        unit_decisions.append(decision.__dict__)

        # Row-level aggregation across detected columns. This supports files with
        # multiple stations/versions of the same element in separate columns.
        work[f"row_{family}_mean"] = normalized_values.mean(axis=1, skipna=True)
        work[f"row_{family}_min"] = normalized_values.min(axis=1, skipna=True)
        work[f"row_{family}_max"] = normalized_values.max(axis=1, skipna=True)
        work[f"row_{family}_obs_cols"] = normalized_values.notna().sum(axis=1)

    valid = work.dropna(subset=["date"]).copy()
    if valid.empty:
        fail("No valid dates after parsing the NOAA date column.")

    agg_spec: Dict[str, Tuple[str, object]] = {
        "source_rows_for_date": ("date", "size"),
    }

    for family in WEATHER_FAMILY_REGEX:
        if f"row_{family}_mean" not in valid.columns:
            continue

        agg_spec[f"daily_{family}_mean"] = (f"row_{family}_mean", "mean")
        agg_spec[f"daily_{family}_min"] = (f"row_{family}_min", "min")
        agg_spec[f"daily_{family}_max"] = (f"row_{family}_max", "max")
        agg_spec[f"daily_{family}_nonmissing_rows"] = (f"row_{family}_mean", lambda s: int(s.notna().sum()))
        agg_spec[f"daily_{family}_obs_cols_total"] = (f"row_{family}_obs_cols", "sum")

    daily = valid.groupby("date", as_index=False).agg(**agg_spec)

    # Derive tavg if not provided but tmax/tmin are present.
    if "daily_tavg_mean" not in daily.columns and {"daily_tmax_mean", "daily_tmin_mean"}.issubset(daily.columns):
        daily["daily_tavg_mean"] = (daily["daily_tmax_mean"] + daily["daily_tmin_mean"]) / 2.0
        daily["daily_tavg_min"] = np.nan
        daily["daily_tavg_max"] = np.nan
        daily["daily_tavg_nonmissing_rows"] = np.where(
            daily[["daily_tmax_mean", "daily_tmin_mean"]].notna().all(axis=1), 1, 0
        )
        daily["daily_tavg_obs_cols_total"] = np.nan
        unit_decisions.append(
            {
                "family": "tavg",
                "requested_unit": "derived",
                "inferred_unit": "celsius",
                "conversion": "derived_from_tmax_tmin_mean",
                "n_valid": int(daily["daily_tavg_mean"].notna().sum()),
                "min_value_before": None,
                "p50_before": None,
                "p99_before": None,
                "max_value_before": None,
            }
        )

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["week_start"] = monday_week_start(daily["date"])

    return daily, detection_rows, unit_decisions


def add_missing_columns(df: pd.DataFrame, columns: Iterable[str], fill_value=np.nan) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def aggregate_weekly_weather(
    daily: pd.DataFrame,
    min_days_per_week: int,
    densify_weeks: bool,
) -> pd.DataFrame:
    """Aggregate one-row-per-day city weather data to Monday weekly features."""
    grouped = daily.groupby("week_start", as_index=False)

    weekly = grouped.agg(
        weather_days_observed=("date", "nunique"),
        weather_source_rows=("source_rows_for_date", "sum"),
        weather_tmax_mean=("daily_tmax_mean", safe_mean) if "daily_tmax_mean" in daily else ("date", lambda s: np.nan),
        weather_tmax_max=("daily_tmax_mean", safe_max) if "daily_tmax_mean" in daily else ("date", lambda s: np.nan),
        weather_tmax_min=("daily_tmax_mean", safe_min) if "daily_tmax_mean" in daily else ("date", lambda s: np.nan),
        weather_tmax_station_max=("daily_tmax_max", safe_max) if "daily_tmax_max" in daily else ("date", lambda s: np.nan),
        weather_tmin_mean=("daily_tmin_mean", safe_mean) if "daily_tmin_mean" in daily else ("date", lambda s: np.nan),
        weather_tmin_min=("daily_tmin_mean", safe_min) if "daily_tmin_mean" in daily else ("date", lambda s: np.nan),
        weather_tmin_max=("daily_tmin_mean", safe_max) if "daily_tmin_mean" in daily else ("date", lambda s: np.nan),
        weather_tavg_mean=("daily_tavg_mean", safe_mean) if "daily_tavg_mean" in daily else ("date", lambda s: np.nan),
        weather_prcp_sum=("daily_prcp_mean", safe_sum) if "daily_prcp_mean" in daily else ("date", lambda s: np.nan),
        weather_prcp_mean=("daily_prcp_mean", safe_mean) if "daily_prcp_mean" in daily else ("date", lambda s: np.nan),
        weather_prcp_max=("daily_prcp_mean", safe_max) if "daily_prcp_mean" in daily else ("date", lambda s: np.nan),
        weather_prcp_station_max=("daily_prcp_max", safe_max) if "daily_prcp_max" in daily else ("date", lambda s: np.nan),
        weather_snow_sum=("daily_snow_mean", safe_sum) if "daily_snow_mean" in daily else ("date", lambda s: np.nan),
        weather_snow_mean=("daily_snow_mean", safe_mean) if "daily_snow_mean" in daily else ("date", lambda s: np.nan),
        weather_snow_max=("daily_snow_mean", safe_max) if "daily_snow_mean" in daily else ("date", lambda s: np.nan),
        weather_snwd_mean=("daily_snwd_mean", safe_mean) if "daily_snwd_mean" in daily else ("date", lambda s: np.nan),
        weather_snwd_max=("daily_snwd_mean", safe_max) if "daily_snwd_mean" in daily else ("date", lambda s: np.nan),
        weather_awnd_mean=("daily_awnd_mean", safe_mean) if "daily_awnd_mean" in daily else ("date", lambda s: np.nan),
        weather_awnd_max=("daily_awnd_mean", safe_max) if "daily_awnd_mean" in daily else ("date", lambda s: np.nan),
    )

    # Event-day counts. Use groupby.apply for clear NaN semantics.
    event_parts: List[pd.DataFrame] = []
    event_grouped = daily.groupby("week_start")

    if "daily_tmax_mean" in daily.columns:
        part = event_grouped["daily_tmax_mean"].apply(lambda s: count_condition(s, lambda x: x >= 30.0)).reset_index(name="hot_day_30c_count")
        event_parts.append(part)
        part = event_grouped["daily_tmax_mean"].apply(lambda s: count_condition(s, lambda x: x >= 35.0)).reset_index(name="hot_day_35c_count")
        event_parts.append(part)
        part = event_grouped["daily_tmax_mean"].apply(lambda s: count_condition(s, lambda x: x <= 0.0)).reset_index(name="ice_day_tmax_0c_count")
        event_parts.append(part)

    if "daily_tmin_mean" in daily.columns:
        part = event_grouped["daily_tmin_mean"].apply(lambda s: count_condition(s, lambda x: x <= 0.0)).reset_index(name="cold_day_0c_count")
        event_parts.append(part)
        part = event_grouped["daily_tmin_mean"].apply(lambda s: count_condition(s, lambda x: x <= -5.0)).reset_index(name="very_cold_day_minus5c_count")
        event_parts.append(part)

    if "daily_prcp_mean" in daily.columns:
        part = event_grouped["daily_prcp_mean"].apply(lambda s: count_condition(s, lambda x: x >= 1.0)).reset_index(name="rain_day_count")
        event_parts.append(part)
        part = event_grouped["daily_prcp_mean"].apply(lambda s: count_condition(s, lambda x: x >= 10.0)).reset_index(name="moderate_rain_day_10mm_count")
        event_parts.append(part)
        part = event_grouped["daily_prcp_mean"].apply(lambda s: count_condition(s, lambda x: x >= 25.0)).reset_index(name="heavy_rain_day_25mm_count")
        event_parts.append(part)
        part = event_grouped["daily_prcp_mean"].apply(lambda s: count_condition(s, lambda x: x >= 50.0)).reset_index(name="very_heavy_rain_day_50mm_count")
        event_parts.append(part)

    if "daily_snow_mean" in daily.columns:
        part = event_grouped["daily_snow_mean"].apply(lambda s: count_condition(s, lambda x: x > 0.0)).reset_index(name="snow_day_count")
        event_parts.append(part)
        part = event_grouped["daily_snow_mean"].apply(lambda s: count_condition(s, lambda x: x >= 25.0)).reset_index(name="heavy_snow_day_25mm_count")
        event_parts.append(part)

    for part in event_parts:
        weekly = weekly.merge(part, on="week_start", how="left")

    expected_event_cols = [
        "hot_day_30c_count",
        "hot_day_35c_count",
        "ice_day_tmax_0c_count",
        "cold_day_0c_count",
        "very_cold_day_minus5c_count",
        "rain_day_count",
        "moderate_rain_day_10mm_count",
        "heavy_rain_day_25mm_count",
        "very_heavy_rain_day_50mm_count",
        "snow_day_count",
        "heavy_snow_day_25mm_count",
    ]
    weekly = add_missing_columns(weekly, expected_event_cols, fill_value=np.nan)

    # Coverage and calendar metadata.
    weekly["week_start"] = pd.to_datetime(weekly["week_start"]).dt.normalize()
    weekly["week_end"] = weekly["week_start"] + pd.Timedelta(days=6)
    weekly["weather_complete_week_flag"] = (weekly["weather_days_observed"] >= 7).astype(int)
    weekly["weather_min_days_ok_flag"] = (weekly["weather_days_observed"] >= int(min_days_per_week)).astype(int)
    weekly["weather_days_missing_in_week"] = (7 - weekly["weather_days_observed"]).clip(lower=0)

    # Optional densification catches missing weeks even if the daily file has date gaps.
    if densify_weeks and not weekly.empty:
        min_week = weekly["week_start"].min()
        max_week = weekly["week_start"].max()
        all_weeks = pd.DataFrame({"week_start": pd.date_range(min_week, max_week, freq="W-MON")})
        weekly = all_weeks.merge(weekly, on="week_start", how="left")
        weekly["week_end"] = weekly["week_start"] + pd.Timedelta(days=6)
        weekly["weather_days_observed"] = weekly["weather_days_observed"].fillna(0).astype(int)
        weekly["weather_source_rows"] = weekly["weather_source_rows"].fillna(0).astype(int)
        weekly["weather_complete_week_flag"] = (weekly["weather_days_observed"] >= 7).astype(int)
        weekly["weather_min_days_ok_flag"] = (weekly["weather_days_observed"] >= int(min_days_per_week)).astype(int)
        weekly["weather_days_missing_in_week"] = (7 - weekly["weather_days_observed"]).clip(lower=0).astype(int)

    iso = weekly["week_start"].dt.isocalendar()
    weekly["weather_year"] = weekly["week_start"].dt.year.astype("Int64")
    weekly["weather_month"] = weekly["week_start"].dt.month.astype("Int64")
    weekly["weather_quarter"] = weekly["week_start"].dt.quarter.astype("Int64")
    weekly["weather_week_of_year"] = iso.week.astype("Int64")
    weekly["weather_iso_year"] = iso.year.astype("Int64")

    # Stable column order for downstream merges and paper tables.
    ordered_cols = [
        "week_start",
        "week_end",
        "weather_days_observed",
        "weather_days_missing_in_week",
        "weather_complete_week_flag",
        "weather_min_days_ok_flag",
        "weather_source_rows",
        "weather_tmax_mean",
        "weather_tmax_max",
        "weather_tmax_min",
        "weather_tmax_station_max",
        "weather_tmin_mean",
        "weather_tmin_min",
        "weather_tmin_max",
        "weather_tavg_mean",
        "weather_prcp_sum",
        "weather_prcp_mean",
        "weather_prcp_max",
        "weather_prcp_station_max",
        "weather_snow_sum",
        "weather_snow_mean",
        "weather_snow_max",
        "weather_snwd_mean",
        "weather_snwd_max",
        "weather_awnd_mean",
        "weather_awnd_max",
        "hot_day_30c_count",
        "hot_day_35c_count",
        "ice_day_tmax_0c_count",
        "cold_day_0c_count",
        "very_cold_day_minus5c_count",
        "rain_day_count",
        "moderate_rain_day_10mm_count",
        "heavy_rain_day_25mm_count",
        "very_heavy_rain_day_50mm_count",
        "snow_day_count",
        "heavy_snow_day_25mm_count",
        "weather_year",
        "weather_month",
        "weather_quarter",
        "weather_week_of_year",
        "weather_iso_year",
    ]
    weekly = add_missing_columns(weekly, ordered_cols, fill_value=np.nan)
    weekly = weekly[ordered_cols].sort_values("week_start").reset_index(drop=True)

    return weekly


def make_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing,
                "missing_share": (missing / n) if n else np.nan,
                "dtype": str(df[col].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True])


def make_weekly_coverage_report(weekly: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "week_start",
        "week_end",
        "weather_days_observed",
        "weather_days_missing_in_week",
        "weather_complete_week_flag",
        "weather_min_days_ok_flag",
        "weather_source_rows",
    ]
    return weekly[keep].copy()


def make_weather_profile(weekly: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in weekly.columns if c.startswith("weather_") or c.endswith("_count")]
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(weekly[c])]
    rows = []
    for c in numeric_cols:
        s = weekly[c]
        rows.append(
            {
                "feature": c,
                "nonmissing_count": int(s.notna().sum()),
                "missing_count": int(s.isna().sum()),
                "mean": safe_float(s.mean(skipna=True)),
                "std": safe_float(s.std(skipna=True)),
                "min": safe_float(s.min(skipna=True)),
                "p01": safe_float(s.quantile(0.01)),
                "p05": safe_float(s.quantile(0.05)),
                "p50": safe_float(s.quantile(0.50)),
                "p95": safe_float(s.quantile(0.95)),
                "p99": safe_float(s.quantile(0.99)),
                "max": safe_float(s.max(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    weekly: pd.DataFrame,
    summary: Dict[str, object],
    summary_dir: Path,
    output_file: Path,
    detection_rows: List[Dict[str, object]],
    unit_decisions: List[Dict[str, object]],
) -> None:
    ensure_parent(output_file)
    summary_dir.mkdir(parents=True, exist_ok=True)

    weekly.to_csv(output_file, index=False)

    summary_json = summary_dir / f"{SCRIPT_NAME}_summary.json"
    missing_values_csv = summary_dir / f"{SCRIPT_NAME}_missing_values.csv"
    column_detection_csv = summary_dir / f"{SCRIPT_NAME}_column_detection.csv"
    unit_decisions_csv = summary_dir / f"{SCRIPT_NAME}_unit_decisions.csv"
    weekly_coverage_csv = summary_dir / f"{SCRIPT_NAME}_weekly_coverage.csv"
    weather_profile_csv = summary_dir / f"{SCRIPT_NAME}_weather_profile.csv"

    summary["summary_files"] = {
        "summary_json": str(summary_json),
        "missing_values": str(missing_values_csv),
        "column_detection": str(column_detection_csv),
        "unit_decisions": str(unit_decisions_csv),
        "weekly_coverage": str(weekly_coverage_csv),
        "weather_profile": str(weather_profile_csv),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    make_missing_values_report(weekly).to_csv(missing_values_csv, index=False)
    pd.DataFrame(detection_rows).to_csv(column_detection_csv, index=False)
    pd.DataFrame(unit_decisions).to_csv(unit_decisions_csv, index=False)
    make_weekly_coverage_report(weekly).to_csv(weekly_coverage_csv, index=False)
    make_weather_profile(weekly).to_csv(weather_profile_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate NOAA daily NYC weather data to Monday weekly weather features."
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--date-col", type=str, default=None, help="Explicit date column name if auto-detection fails.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    parser.add_argument(
        "--temperature-unit",
        choices=["auto", "celsius", "tenths_c", "fahrenheit"],
        default="auto",
        help="Unit for TMAX/TMIN/TAVG columns. NOAA GHCN usually uses tenths_c.",
    )
    parser.add_argument(
        "--precipitation-unit",
        choices=["auto", "mm", "tenths_mm", "inches"],
        default="auto",
        help="Unit for PRCP columns. NOAA GHCN usually uses tenths_mm.",
    )
    parser.add_argument(
        "--snow-unit",
        choices=["auto", "mm", "tenths_mm", "inches"],
        default="auto",
        help="Unit for SNOW/SNWD columns. NOAA GHCN usually uses tenths_mm.",
    )
    parser.add_argument(
        "--wind-unit",
        choices=["auto", "mps", "tenths_mps", "mph"],
        default="auto",
        help="Unit for AWND/wind columns. NOAA GHCN usually uses tenths_mps.",
    )
    parser.add_argument(
        "--min-days-per-week",
        type=int,
        default=4,
        help="Minimum observed daily records for weather_min_days_ok_flag. Default: 4.",
    )
    parser.add_argument(
        "--no-densify-weeks",
        action="store_true",
        help="Do not add missing Monday weeks between min and max week_start.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    input_file: Path = args.input_file
    output_file: Path = args.output_file
    summary_dir: Path = args.summary_dir

    if not input_file.exists():
        fail(f"Input file not found: {input_file}")

    if output_file.exists() and not args.overwrite:
        fail(f"Output file already exists: {output_file}. Use --overwrite to replace it.")

    info(f"Reading NOAA daily weather file: {input_file}")
    raw_df = pd.read_csv(input_file, low_memory=False)
    if raw_df.empty:
        fail("Input file is empty.")

    date_col = infer_date_column(raw_df, args.date_col)
    info(f"Using date column: {date_col}")

    parsed_dates = pd.to_datetime(raw_df[date_col], errors="coerce")
    bad_date_rows = int(parsed_dates.isna().sum())

    detected_cols: Dict[str, List[str]] = {
        family: detect_family_columns(raw_df, family) for family in WEATHER_FAMILY_REGEX
    }
    detected_nonempty = {k: v for k, v in detected_cols.items() if v}
    if not detected_nonempty:
        fail(
            "No weather measurement columns detected. "
            "Expected columns such as TMAX, TMIN, PRCP, SNOW, SNWD, AWND or similar names."
        )

    info("Detected weather columns:")
    for family, cols in detected_cols.items():
        info(f"  {family}: {len(cols)} column(s)" + (f" -> {cols}" if cols else ""))

    daily, detection_rows, unit_decisions = build_daily_city_weather(
        raw_df=raw_df,
        date_col=date_col,
        detected_cols=detected_cols,
        temperature_unit=args.temperature_unit,
        precipitation_unit=args.precipitation_unit,
        snow_unit=args.snow_unit,
        wind_unit=args.wind_unit,
    )

    weekly = aggregate_weekly_weather(
        daily=daily,
        min_days_per_week=args.min_days_per_week,
        densify_weeks=not args.no_densify_weeks,
    )

    duplicate_week_start_rows = int(weekly.duplicated(subset=["week_start"]).sum())
    if duplicate_week_start_rows > 0:
        fail(f"Duplicate week_start rows after aggregation: {duplicate_week_start_rows}")

    if weekly.empty:
        fail("Weekly weather output is empty.")

    feature_cols = [c for c in weekly.columns if c not in {"week_start", "week_end"}]
    weather_value_cols = [c for c in feature_cols if c.startswith("weather_") or c.endswith("_count")]
    all_missing_features = [c for c in weather_value_cols if weekly[c].isna().all()]

    elapsed = time.time() - t0
    summary: Dict[str, object] = {
        "script": SCRIPT_NAME,
        "status": "done",
        "input_file": str(input_file),
        "output_file": str(output_file),
        "summary_dir": str(summary_dir),
        "input_rows_raw": int(len(raw_df)),
        "input_columns_raw": list(map(str, raw_df.columns)),
        "date_col": date_col,
        "bad_date_rows": bad_date_rows,
        "valid_daily_dates": int(daily["date"].nunique()),
        "daily_rows_after_city_aggregation": int(len(daily)),
        "min_daily_date": str(daily["date"].min().date()),
        "max_daily_date": str(daily["date"].max().date()),
        "output_rows": int(len(weekly)),
        "output_columns": list(map(str, weekly.columns)),
        "output_min_week_start": str(weekly["week_start"].min().date()),
        "output_max_week_start": str(weekly["week_start"].max().date()),
        "duplicate_week_start_rows": duplicate_week_start_rows,
        "min_days_per_week": int(args.min_days_per_week),
        "densify_weeks": bool(not args.no_densify_weeks),
        "complete_week_count": int(weekly["weather_complete_week_flag"].sum()),
        "incomplete_week_count": int((weekly["weather_complete_week_flag"] == 0).sum()),
        "min_days_ok_week_count": int(weekly["weather_min_days_ok_flag"].sum()),
        "weather_days_observed_min": int(weekly["weather_days_observed"].min()),
        "weather_days_observed_max": int(weekly["weather_days_observed"].max()),
        "detected_columns": detected_cols,
        "unit_decisions": unit_decisions,
        "all_missing_weather_features": all_missing_features,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_minutes": round(elapsed / 60.0, 3),
    }

    write_outputs(
        weekly=weekly,
        summary=summary,
        summary_dir=summary_dir,
        output_file=output_file,
        detection_rows=detection_rows,
        unit_decisions=unit_decisions,
    )

    info("Done.")
    info(f"Output: {output_file}")
    info(f"Summary directory: {summary_dir}")
    info(f"Weekly rows: {len(weekly):,}")
    info(f"Date range: {weekly['week_start'].min().date()} to {weekly['week_start'].max().date()}")

    if all_missing_features:
        warn(
            "Some weather feature columns are all missing because the corresponding raw variables "
            f"were not detected: {all_missing_features}"
        )


if __name__ == "__main__":
    main()
