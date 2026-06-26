r"""
build_311_temporal_features.py

Step 5 for the SIMC NYC data pipeline.

Purpose
-------
Build leakage-safe temporal features from the dense weekly NYC 311 panel:

    NTA x week_start x complaint_category

Main input
----------
    data/processed/weekly_311/nyc_311_weekly_by_nta_category.csv.gz

Main output
-----------
    data/processed/feature_tables/nyc_311_weekly_temporal_features.csv.gz

Diagnostics
-----------
    data/processed/_feature_summaries/build_311_temporal_features_summary.json
    data/processed/_feature_summaries/build_311_temporal_features_missing_values.csv
    data/processed/_feature_summaries/build_311_temporal_features_target_balance_by_category.csv
    data/processed/_feature_summaries/build_311_temporal_features_target_balance_by_year.csv
    data/processed/_feature_summaries/build_311_temporal_features_target_balance_by_category_year.csv
    data/processed/_feature_summaries/build_311_temporal_features_category_profile.csv
    data/processed/_feature_summaries/build_311_temporal_features_group_gaps.csv

Design notes
------------
1. All lag/rolling features are computed separately by:
       nta2020 x complaint_category

2. Rolling features use shift(1) before rolling, so they only use information
   available before the current week. This prevents leakage.

3. target_next_week_count uses shift(-1) within each NTA-category series.

4. abnormal_increase_next_week is defined as:
       target_next_week_count > rolling_8w_mean + threshold_multiplier * rolling_8w_std

5. The first weeks in each series naturally contain missing lag/rolling values.
   The last week naturally has missing target_next_week_count.

Recommended command from project root
-------------------------------------
    .\.venv\Scripts\python.exe .\code_processdata\build_311_temporal_features.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_NAME = "build_311_temporal_features"

REQUIRED_COLUMNS = {
    "nta2020",
    "week_start",
    "complaint_category",
    "complaint_count",
}

OPTIONAL_ID_COLUMNS = [
    "ntaname",
    "boroname",
]

GROUP_COLS = ["nta2020", "complaint_category"]
KEY_COLS = ["nta2020", "week_start", "complaint_category"]

LAG_WEEKS = [1, 2, 4, 8, 12, 52]
ROLLING_WINDOWS = [4, 8, 12]

CATEGORY_LABELS = {
    "noise": "Noise & Disturbance",
    "parking_traffic": "Parking, Traffic & Mobility",
    "sanitation": "Sanitation & Waste",
    "housing": "Housing & Building Conditions",
    "water_sewer": "Water & Sewer",
    "infrastructure": "Street Infrastructure & Public Works",
    "environment": "Environmental Health & Green Space",
    "public_safety": "Public Safety & Regulatory Enforcement",
    "other": "Other / Administrative & Unclassified",
}


class PipelineError(RuntimeError):
    """Raised for expected pipeline/data validation errors."""


def project_root_from_script() -> Path:
    """Return SIMC_PROJECT root when this script is stored in code_processdata/."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_script()

    parser = argparse.ArgumentParser(
        description=(
            "Build lag, rolling, calendar, and next-week target features from "
            "the dense weekly NYC 311 NTA-category panel."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "weekly_311" / "nyc_311_weekly_by_nta_category.csv.gz",
        help="Input dense weekly 311 panel CSV/CSV.GZ.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "feature_tables" / "nyc_311_weekly_temporal_features.csv.gz",
        help="Output temporal feature table CSV.GZ.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=root / "data" / "processed" / "_feature_summaries",
        help="Folder for diagnostic summary files.",
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=1.5,
        help=(
            "Multiplier used in abnormal target: target_next_week_count > "
            "rolling_8w_mean + multiplier * rolling_8w_std."
        ),
    )
    parser.add_argument(
        "--min-history-weeks",
        type=int,
        default=12,
        help=(
            "Minimum number of previous weeks recommended for train-ready rows. "
            "This does not drop rows; it creates has_min_history_XXw and "
            "is_train_ready_temporal flags."
        ),
    )
    parser.add_argument(
        "--rolling-min-periods",
        type=str,
        default="window",
        choices=["window", "1"],
        help=(
            "Minimum periods for rolling windows. Use 'window' for stricter, "
            "leakage-safe complete-history features; use '1' for early partial windows."
        ),
    )
    parser.add_argument(
        "--no-densify-check",
        action="store_true",
        help=(
            "Skip reindexing to the full NTA x week x category grid. By default, "
            "the script safely densifies/checks the input and fills missing weekly "
            "cells with complaint_count=0."
        ),
    )
    parser.add_argument(
        "--strict-duplicates",
        action="store_true",
        help=(
            "Fail if duplicate nta2020-week_start-complaint_category keys are found. "
            "By default, duplicates are aggregated by summing complaint_count."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output and summary files.",
    )

    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{SCRIPT_NAME}] {message}", flush=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_allowed(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise PipelineError(
            f"Output already exists and --overwrite was not provided: {path}"
        )


def read_weekly_panel(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise PipelineError(f"Input file not found: {input_path}")

    log(f"Reading input: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise PipelineError(
            "Input is missing required columns: " + ", ".join(missing)
        )

    return df


def clean_weekly_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize dtypes, dates, counts, and duplicate keys."""
    summary: dict = {}

    summary["input_rows_raw"] = int(len(df))
    summary["input_columns_raw"] = list(df.columns)

    # Keep useful columns while preserving any extra columns at the end.
    ordered_cols = []
    for col in ["nta2020", "ntaname", "boroname", "week_start", "complaint_category", "complaint_count"]:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra_cols].copy()

    # Standardize textual keys.
    df["nta2020"] = df["nta2020"].astype("string").str.strip()
    df["complaint_category"] = df["complaint_category"].astype("string").str.strip()
    for col in OPTIONAL_ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Dates.
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    bad_date_rows = int(df["week_start"].isna().sum())
    summary["bad_week_start_rows"] = bad_date_rows
    if bad_date_rows > 0:
        log(f"Dropping rows with bad week_start: {bad_date_rows:,}")
        df = df.loc[df["week_start"].notna()].copy()

    # Counts.
    raw_count_na = int(df["complaint_count"].isna().sum())
    df["complaint_count"] = pd.to_numeric(df["complaint_count"], errors="coerce")
    count_na_after_numeric = int(df["complaint_count"].isna().sum())
    summary["complaint_count_missing_before_numeric"] = raw_count_na
    summary["complaint_count_missing_or_bad_after_numeric"] = count_na_after_numeric
    if count_na_after_numeric > 0:
        log(f"Filling missing/bad complaint_count with 0: {count_na_after_numeric:,}")
        df["complaint_count"] = df["complaint_count"].fillna(0)

    negative_count_rows = int((df["complaint_count"] < 0).sum())
    summary["negative_complaint_count_rows"] = negative_count_rows
    if negative_count_rows > 0:
        raise PipelineError(
            f"Found negative complaint_count rows: {negative_count_rows:,}. "
            "Please inspect the input before building temporal features."
        )

    # Key completeness.
    missing_key_rows = int(df[KEY_COLS].isna().any(axis=1).sum())
    summary["missing_key_rows"] = missing_key_rows
    if missing_key_rows > 0:
        log(f"Dropping rows with missing key columns: {missing_key_rows:,}")
        df = df.loc[~df[KEY_COLS].isna().any(axis=1)].copy()

    # Duplicate key check.
    duplicate_rows = int(df.duplicated(KEY_COLS, keep=False).sum())
    duplicate_keys = int(df.duplicated(KEY_COLS).sum())
    summary["duplicate_key_extra_rows"] = duplicate_keys
    summary["duplicate_key_rows_involved"] = duplicate_rows

    if duplicate_keys > 0:
        # Metadata columns: first non-null value.
        log(
            f"Found duplicate keys; aggregating by {KEY_COLS}: "
            f"extra duplicate rows={duplicate_keys:,}, involved rows={duplicate_rows:,}"
        )

        agg_spec: dict[str, str] = {"complaint_count": "sum"}
        for col in OPTIONAL_ID_COLUMNS:
            if col in df.columns:
                agg_spec[col] = "first"

        # Preserve only columns with clear aggregation semantics.
        df = (
            df.groupby(KEY_COLS, as_index=False, dropna=False)
            .agg(agg_spec)
        )

    # Complaint counts should be integer-like, but keep int64 after aggregation.
    rounded = np.rint(df["complaint_count"].to_numpy(dtype=float))
    if not np.allclose(df["complaint_count"].to_numpy(dtype=float), rounded, equal_nan=True):
        raise PipelineError("complaint_count contains non-integer values after cleaning.")
    df["complaint_count"] = rounded.astype("int64")

    df = df.sort_values(KEY_COLS).reset_index(drop=True)

    summary["rows_after_cleaning"] = int(len(df))
    summary["total_complaint_count_after_cleaning"] = int(df["complaint_count"].sum())
    summary["unique_nta2020"] = int(df["nta2020"].nunique())
    summary["unique_complaint_categories"] = int(df["complaint_category"].nunique())
    summary["unique_weeks"] = int(df["week_start"].nunique())
    summary["min_week_start"] = str(df["week_start"].min().date()) if len(df) else None
    summary["max_week_start"] = str(df["week_start"].max().date()) if len(df) else None

    return df, summary


def fail_on_duplicates_if_requested(summary: dict, strict: bool) -> None:
    duplicate_keys = int(summary.get("duplicate_key_extra_rows", 0))
    if strict and duplicate_keys > 0:
        raise PipelineError(
            f"Duplicate keys found ({duplicate_keys:,}) and --strict-duplicates was provided."
        )


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return pd.NA
    return non_null.iloc[0]


def densify_weekly_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ensure full NTA x complaint_category x weekly grid from global min to max week."""
    summary: dict = {}

    nta_values = sorted(df["nta2020"].dropna().unique())
    category_values = sorted(df["complaint_category"].dropna().unique())
    min_week = df["week_start"].min()
    max_week = df["week_start"].max()

    all_weeks = pd.date_range(min_week, max_week, freq="W-MON")
    expected_rows = len(nta_values) * len(category_values) * len(all_weeks)

    summary["densify_unique_nta2020"] = int(len(nta_values))
    summary["densify_unique_complaint_categories"] = int(len(category_values))
    summary["densify_unique_weeks"] = int(len(all_weeks))
    summary["densify_expected_rows"] = int(expected_rows)
    summary["densify_rows_before"] = int(len(df))

    if len(df) == expected_rows and df[KEY_COLS].duplicated().sum() == 0:
        # Still check gaps later, but no need to reindex.
        summary["densify_added_rows"] = 0
        summary["densify_performed"] = False
        return df.sort_values(KEY_COLS).reset_index(drop=True), summary

    log(
        "Densifying weekly grid to full NTA x week_start x complaint_category "
        f"panel: expected rows={expected_rows:,}"
    )

    full_index = pd.MultiIndex.from_product(
        [nta_values, all_weeks, category_values],
        names=KEY_COLS,
    )

    base = (
        df.set_index(KEY_COLS)
        .sort_index()
        .reindex(full_index)
        .reset_index()
    )

    base["complaint_count"] = base["complaint_count"].fillna(0).astype("int64")

    # Fill stable NTA metadata after reindexing.
    for col in OPTIONAL_ID_COLUMNS:
        if col in df.columns:
            lookup = (
                df[["nta2020", col]]
                .dropna()
                .groupby("nta2020", as_index=True)[col]
                .agg(first_non_null)
            )
            base[col] = base["nta2020"].map(lookup)

    # Put columns in a clean order.
    ordered = ["nta2020"]
    ordered += [col for col in OPTIONAL_ID_COLUMNS if col in base.columns]
    ordered += ["week_start", "complaint_category", "complaint_count"]
    remaining = [c for c in base.columns if c not in ordered]
    base = base[ordered + remaining]

    summary["densify_added_rows"] = int(len(base) - len(df))
    summary["densify_rows_after"] = int(len(base))
    summary["densify_performed"] = True
    summary["total_complaint_count_after_densify"] = int(base["complaint_count"].sum())

    return base.sort_values(KEY_COLS).reset_index(drop=True), summary


def find_group_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return groups where consecutive week_start values are not 7 days apart."""
    work = df.sort_values(GROUP_COLS + ["week_start"]).copy()
    work["prev_week_start"] = work.groupby(GROUP_COLS, sort=False)["week_start"].shift(1)
    work["gap_days"] = (work["week_start"] - work["prev_week_start"]).dt.days

    gaps = work.loc[work["gap_days"].notna() & (work["gap_days"] != 7), GROUP_COLS + ["prev_week_start", "week_start", "gap_days"]]
    if gaps.empty:
        return pd.DataFrame(columns=GROUP_COLS + ["prev_week_start", "week_start", "gap_days"])

    return gaps.sort_values(GROUP_COLS + ["week_start"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    iso = df["week_start"].dt.isocalendar()
    df["year"] = df["week_start"].dt.year.astype("int16")
    df["month"] = df["week_start"].dt.month.astype("int8")
    df["quarter"] = df["week_start"].dt.quarter.astype("int8")
    df["week_of_year"] = iso.week.astype("int16")
    df["iso_year"] = iso.year.astype("int16")
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)

    # Useful flags for seasonality and service planning.
    df["is_year_start"] = (df["week_of_year"] <= 2).astype("int8")
    df["is_year_end"] = (df["week_of_year"] >= 51).astype("int8")

    # Context requested by the project checkpoint.
    # Keep calendar year as-is; the 2014-12-29 week is still treated as pre-COVID.
    conditions = [
        df["year"] <= 2019,
        df["year"].between(2020, 2021, inclusive="both"),
        df["year"] >= 2022,
    ]
    choices = ["pre_covid", "covid_disruption", "post_covid"]
    df["period_type"] = np.select(conditions, choices, default="unknown")
    df["is_covid_period"] = (df["period_type"] == "covid_disruption").astype("int8")

    return df


def add_temporal_features(
    df: pd.DataFrame,
    threshold_multiplier: float,
    min_history_weeks: int,
    rolling_min_periods_mode: str,
) -> tuple[pd.DataFrame, dict]:
    df = df.sort_values(GROUP_COLS + ["week_start"]).reset_index(drop=True).copy()
    summary: dict = {}

    group = df.groupby(GROUP_COLS, sort=False, observed=True)

    # Number of previous rows/weeks available in the current series.
    df["history_weeks_available"] = group.cumcount().astype("int16")

    # Lag features.
    lag_cols = []
    for lag in LAG_WEEKS:
        col = f"lag_{lag}w_count"
        df[col] = group["complaint_count"].shift(lag)
        lag_cols.append(col)

    # Shifted rolling features: all leakage-safe.
    rolling_cols = []
    for window in ROLLING_WINDOWS:
        min_periods = window if rolling_min_periods_mode == "window" else 1

        mean_col = f"rolling_{window}w_mean"
        std_col = f"rolling_{window}w_std"
        sum_col = f"rolling_{window}w_sum"

        df[mean_col] = group["complaint_count"].transform(
            lambda s, w=window, mp=min_periods: s.shift(1).rolling(window=w, min_periods=mp).mean()
        )
        df[std_col] = group["complaint_count"].transform(
            lambda s, w=window, mp=min_periods: s.shift(1).rolling(window=w, min_periods=mp).std()
        )
        df[sum_col] = group["complaint_count"].transform(
            lambda s, w=window, mp=min_periods: s.shift(1).rolling(window=w, min_periods=mp).sum()
        )

        rolling_cols.extend([mean_col, std_col, sum_col])

    # Extra momentum features. These are safe because they only use current/past observed count.
    # A row at week t is used to predict week t+1, so complaint_count(t) is known.
    df["diff_1w_count"] = df["complaint_count"] - df["lag_1w_count"]
    df["diff_4w_count"] = df["complaint_count"] - df["lag_4w_count"]

    df["pct_change_1w"] = np.where(
        df["lag_1w_count"] > 0,
        df["diff_1w_count"] / df["lag_1w_count"],
        np.nan,
    )
    df["ratio_to_8w_mean"] = np.where(
        df["rolling_8w_mean"] > 0,
        df["complaint_count"] / df["rolling_8w_mean"],
        np.nan,
    )
    df["ratio_to_12w_mean"] = np.where(
        df["rolling_12w_mean"] > 0,
        df["complaint_count"] / df["rolling_12w_mean"],
        np.nan,
    )

    # Targets.
    df["target_next_week_count"] = group["complaint_count"].shift(-1)

    threshold = df["rolling_8w_mean"] + threshold_multiplier * df["rolling_8w_std"]
    label_valid = (
        df["target_next_week_count"].notna()
        & df["rolling_8w_mean"].notna()
        & df["rolling_8w_std"].notna()
    )
    abnormal = (df["target_next_week_count"] > threshold).astype("Int64")
    abnormal = abnormal.where(label_valid, pd.NA)
    df["abnormal_increase_next_week"] = abnormal
    df["abnormal_threshold_8w"] = threshold

    history_flag_col = f"has_min_history_{min_history_weeks}w"
    df[history_flag_col] = (df["history_weeks_available"] >= min_history_weeks).astype("int8")
    df["has_target_next_week"] = df["target_next_week_count"].notna().astype("int8")
    df["is_train_ready_temporal"] = (
        (df[history_flag_col] == 1)
        & (df["has_target_next_week"] == 1)
        & df["abnormal_increase_next_week"].notna()
    ).astype("int8")

    # Compact numeric dtypes where safe.
    for col in lag_cols + rolling_cols + [
        "diff_1w_count",
        "diff_4w_count",
        "pct_change_1w",
        "ratio_to_8w_mean",
        "ratio_to_12w_mean",
        "target_next_week_count",
        "abnormal_threshold_8w",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    summary["lag_columns"] = lag_cols
    summary["rolling_columns"] = rolling_cols
    summary["target_columns"] = [
        "target_next_week_count",
        "abnormal_increase_next_week",
        "abnormal_threshold_8w",
    ]
    summary["threshold_multiplier"] = threshold_multiplier
    summary["min_history_weeks"] = min_history_weeks
    summary["rolling_min_periods_mode"] = rolling_min_periods_mode
    summary["train_ready_rows"] = int(df["is_train_ready_temporal"].sum())

    return df, summary


def add_category_display_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["complaint_category_label"] = df["complaint_category"].map(CATEGORY_LABELS).fillna(df["complaint_category"])
    return df


def build_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_rows": missing,
                "missing_share": float(missing / total) if total else np.nan,
                "dtype": str(df[col].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rows", "column"], ascending=[False, True])


def build_target_balance(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    group_cols = list(group_cols)
    valid = df.loc[df["abnormal_increase_next_week"].notna()].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "rows_with_valid_label",
                "positive_abnormal_rows",
                "negative_normal_rows",
                "positive_share",
                "target_next_week_count_mean",
                "target_next_week_count_median",
            ]
        )

    valid["abnormal_increase_next_week"] = valid["abnormal_increase_next_week"].astype("int8")

    out = (
        valid.groupby(group_cols, dropna=False)
        .agg(
            rows_with_valid_label=("abnormal_increase_next_week", "size"),
            positive_abnormal_rows=("abnormal_increase_next_week", "sum"),
            target_next_week_count_mean=("target_next_week_count", "mean"),
            target_next_week_count_median=("target_next_week_count", "median"),
        )
        .reset_index()
    )
    out["negative_normal_rows"] = out["rows_with_valid_label"] - out["positive_abnormal_rows"]
    out["positive_share"] = out["positive_abnormal_rows"] / out["rows_with_valid_label"]

    # Clean column order.
    ordered = group_cols + [
        "rows_with_valid_label",
        "positive_abnormal_rows",
        "negative_normal_rows",
        "positive_share",
        "target_next_week_count_mean",
        "target_next_week_count_median",
    ]
    return out[ordered]


def build_category_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby(["complaint_category", "complaint_category_label"], dropna=False)
        .agg(
            rows=("complaint_count", "size"),
            total_complaint_count=("complaint_count", "sum"),
            mean_weekly_count=("complaint_count", "mean"),
            median_weekly_count=("complaint_count", "median"),
            max_weekly_count=("complaint_count", "max"),
            ntas=("nta2020", "nunique"),
            weeks=("week_start", "nunique"),
            train_ready_rows=("is_train_ready_temporal", "sum"),
        )
        .reset_index()
        .sort_values("total_complaint_count", ascending=False)
    )

    total = profile["total_complaint_count"].sum()
    profile["complaint_count_share"] = profile["total_complaint_count"] / total if total else np.nan
    return profile


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    ensure_output_allowed(path, overwrite)
    ensure_parent(path)
    df.to_csv(path, index=False)
    log(f"Wrote: {path}")


def write_json(data: dict, path: Path, overwrite: bool) -> None:
    ensure_output_allowed(path, overwrite)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log(f"Wrote: {path}")


def write_outputs(df: pd.DataFrame, args: argparse.Namespace, summary: dict, group_gaps: pd.DataFrame) -> None:
    ensure_output_allowed(args.output, args.overwrite)
    ensure_parent(args.output)

    log(f"Writing main feature table: {args.output}")
    df.to_csv(args.output, index=False, compression="gzip")

    summary_path = args.summary_dir / f"{SCRIPT_NAME}_summary.json"
    missing_path = args.summary_dir / f"{SCRIPT_NAME}_missing_values.csv"
    by_category_path = args.summary_dir / f"{SCRIPT_NAME}_target_balance_by_category.csv"
    by_year_path = args.summary_dir / f"{SCRIPT_NAME}_target_balance_by_year.csv"
    by_category_year_path = args.summary_dir / f"{SCRIPT_NAME}_target_balance_by_category_year.csv"
    category_profile_path = args.summary_dir / f"{SCRIPT_NAME}_category_profile.csv"
    gaps_path = args.summary_dir / f"{SCRIPT_NAME}_group_gaps.csv"

    # Summary CSVs.
    missing = build_missing_values_summary(df)
    balance_category = build_target_balance(df, ["complaint_category", "complaint_category_label"])
    balance_year = build_target_balance(df, ["year"])
    balance_category_year = build_target_balance(df, ["complaint_category", "complaint_category_label", "year"])
    category_profile = build_category_profile(df)

    write_csv(missing, missing_path, args.overwrite)
    write_csv(balance_category, by_category_path, args.overwrite)
    write_csv(balance_year, by_year_path, args.overwrite)
    write_csv(balance_category_year, by_category_year_path, args.overwrite)
    write_csv(category_profile, category_profile_path, args.overwrite)
    write_csv(group_gaps, gaps_path, args.overwrite)

    # Add selected diagnostics to JSON.
    valid_label = df["abnormal_increase_next_week"].notna()
    positive = int(df.loc[valid_label, "abnormal_increase_next_week"].astype("int8").sum()) if valid_label.any() else 0
    valid_n = int(valid_label.sum())

    summary.update(
        {
            "status": "done",
            "output_file": str(args.output),
            "summary_dir": str(args.summary_dir),
            "output_rows": int(len(df)),
            "output_columns": list(df.columns),
            "output_total_complaint_count": int(df["complaint_count"].sum()),
            "output_unique_nta2020": int(df["nta2020"].nunique()),
            "output_unique_categories": int(df["complaint_category"].nunique()),
            "output_unique_weeks": int(df["week_start"].nunique()),
            "output_min_week_start": str(df["week_start"].min().date()) if len(df) else None,
            "output_max_week_start": str(df["week_start"].max().date()) if len(df) else None,
            "duplicate_keys_after_processing": int(df.duplicated(KEY_COLS).sum()),
            "group_gap_rows": int(len(group_gaps)),
            "target_valid_label_rows": valid_n,
            "target_positive_abnormal_rows": positive,
            "target_negative_normal_rows": int(valid_n - positive),
            "target_positive_share": float(positive / valid_n) if valid_n else None,
            "summary_files": {
                "summary_json": str(summary_path),
                "missing_values": str(missing_path),
                "target_balance_by_category": str(by_category_path),
                "target_balance_by_year": str(by_year_path),
                "target_balance_by_category_year": str(by_category_year_path),
                "category_profile": str(category_profile_path),
                "group_gaps": str(gaps_path),
            },
        }
    )

    write_json(summary, summary_path, args.overwrite)


def main() -> int:
    start = time.time()
    args = parse_args()

    try:
        ensure_output_allowed(args.output, args.overwrite)
        args.summary_dir.mkdir(parents=True, exist_ok=True)

        summary: dict = {
            "script": SCRIPT_NAME,
            "input_file": str(args.input),
            "threshold_multiplier": args.threshold_multiplier,
            "min_history_weeks": args.min_history_weeks,
            "rolling_min_periods": args.rolling_min_periods,
            "densify_check_enabled": not args.no_densify_check,
            "strict_duplicates": bool(args.strict_duplicates),
        }

        df = read_weekly_panel(args.input)
        df, clean_summary = clean_weekly_panel(df)
        summary.update(clean_summary)
        fail_on_duplicates_if_requested(summary, args.strict_duplicates)

        if not args.no_densify_check:
            df, densify_summary = densify_weekly_grid(df)
            summary.update(densify_summary)

        group_gaps = find_group_gaps(df)
        if len(group_gaps) > 0:
            log(
                f"WARNING: Found non-7-day week gaps in grouped series: {len(group_gaps):,}. "
                "See group_gaps summary CSV."
            )
        else:
            log("Group gap check passed: all consecutive rows are 7 days apart within NTA-category series.")

        df = add_category_display_label(df)
        df = add_calendar_features(df)
        df, temporal_summary = add_temporal_features(
            df=df,
            threshold_multiplier=args.threshold_multiplier,
            min_history_weeks=args.min_history_weeks,
            rolling_min_periods_mode=args.rolling_min_periods,
        )
        summary.update(temporal_summary)

        # Put target/features in a readable order.
        leading_cols = [
            "nta2020",
            "ntaname" if "ntaname" in df.columns else None,
            "boroname" if "boroname" in df.columns else None,
            "week_start",
            "week_end",
            "complaint_category",
            "complaint_category_label",
            "complaint_count",
            "target_next_week_count",
            "abnormal_increase_next_week",
            "abnormal_threshold_8w",
            "is_train_ready_temporal",
        ]
        leading_cols = [c for c in leading_cols if c is not None and c in df.columns]
        remaining_cols = [c for c in df.columns if c not in leading_cols]
        df = df[leading_cols + remaining_cols]

        elapsed = time.time() - start
        summary["elapsed_seconds"] = round(elapsed, 3)
        summary["elapsed_minutes"] = round(elapsed / 60, 3)

        write_outputs(df, args, summary, group_gaps)

        log("DONE")
        log(f"Rows: {len(df):,}")
        log(f"Train-ready temporal rows: {int(df['is_train_ready_temporal'].sum()):,}")
        log(f"Elapsed minutes: {elapsed / 60:.3f}")
        return 0

    except PipelineError as exc:
        log(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        log("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
