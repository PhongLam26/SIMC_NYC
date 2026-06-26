#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate NYC 311 requests joined to NTA into weekly NTA × complaint_category counts.

Pipeline step:
    Step 3. Semantic grouping complaint_type
    Step 4. Aggregate weekly 311

Input:
    data/processed/nyc_311_with_nta/*_with_nta.csv.gz

Outputs:
    data/processed/weekly_311/nyc_311_weekly_by_nta_category.csv.gz
    data/processed/weekly_311/nyc_311_weekly_by_nta_category_observed.csv.gz
    data/processed/_aggregate_summaries/aggregate_weekly_311_summary.csv
    data/processed/_aggregate_summaries/aggregate_weekly_311_file_summary.csv
    data/processed/_aggregate_summaries/weekly_311_category_counts.csv
    data/processed/_aggregate_summaries/weekly_311_other_complaint_type_top200.csv
    data/processed/_aggregate_summaries/weekly_311_complaint_type_category_counts.csv

Run from project root:
    python code_processdata/aggregate_weekly_311.py --limit-files 5 --chunksize 100000 --overwrite
    python code_processdata/aggregate_weekly_311.py --chunksize 200000 --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CATEGORIES: List[str] = [
    "noise",
    "parking_traffic",
    "sanitation",
    "housing",
    "water_sewer",
    "infrastructure",
    "environment",
    "public_safety",
    "other",
]

# Patterns are intentionally broad because NYC 311 complaint_type labels vary by year.
# Category priority is controlled by the order in categorize_complaints().
CATEGORY_PATTERNS: Dict[str, str] = {
    "noise": r"\b(noise|loud|music|talking|banging|party|jackhammer|after\s*hours)\b",
    "housing": r"\b(heat/hot\s*water|hot\s*water|heating|heat|plumbing|electric|door/window|window|flooring|stairs|paint/plaster|paint|plaster|appliance|elevator|building|maintenance|water\s*leak|leak|unsanitary\s*condition|safety|gas|boiler)\b",
    "sanitation": r"\b(sanitation|dirty|garbage|trash|litter|recycling|missed\s*collection|bulk|derelict\s*bicycle|street\s*sweeping|sweeping|rodent|rat|mouse|mice|pest|mold|illegal\s*dumping|dumping)\b",
    "water_sewer": r"\b(sewer|water\s*system|water\s*quality|catch\s*basin|flooding|hydrant|drinking\s*water|wastewater)\b",
    "infrastructure": r"\b(street\s*condition|sidewalk\s*condition|street\s*light|highway\s*condition|bridge\s*condition|curb|pothole|roadway|road\s*condition|traffic\s*signal|street\s*light\s*condition|streetlight)\b",
    "parking_traffic": r"\b(illegal\s*parking|blocked\s*driveway|traffic|street\s*sign|parking\s*meter|derelict\s*vehicle|abandoned\s*vehicle|taxi|for\s*hire\s*vehicle|bike\s*lane|bus\s*stop|bus\s*lane|vehicle|parking)\b",
    "environment": r"\b(tree|damaged\s*tree|dead/dying\s*tree|air\s*quality|asbestos|lead|industrial\s*waste|hazardous|chemical|environment|plant|parks|green|noise\s*from\s*park)\b",
    "public_safety": r"\b(homeless|animal|dog|illegal\s*fireworks|fireworks|drinking|drug\s*activity|panhandling|vending|encampment|harassment|graffiti|non-emergency\s*police|police|fire\s*safety)\b",
}

# This order is chosen to avoid common false matches.
# Example: "Noise - Street/Sidewalk" should be noise, not infrastructure.
CATEGORY_ORDER: List[str] = [
    "noise",
    "housing",
    "sanitation",
    "water_sewer",
    "infrastructure",
    "parking_traffic",
    "environment",
    "public_safety",
]


ColumnMap = Dict[str, Optional[str]]
AggKey = Tuple[str, str, str]  # nta2020, week_start YYYY-MM-DD, complaint_category
TypeCatKey = Tuple[str, str]   # complaint_type, complaint_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate joined NYC 311 × NTA files to weekly NTA/category counts."
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed/nyc_311_with_nta",
        help="Directory containing *_with_nta.csv.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/weekly_311",
        help="Directory for weekly 311 outputs.",
    )
    parser.add_argument(
        "--summary-dir",
        default="data/processed/_aggregate_summaries",
        help="Directory for diagnostic summaries.",
    )
    parser.add_argument(
        "--pattern",
        default="*_with_nta.csv.gz",
        help="Glob pattern for joined input files.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk. Use 100000 if laptop lags; 300000 if stable.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Process only first N files for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive created_date filter, e.g. 2015-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive created_date filter, e.g. 2025-12-31.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N files.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path.cwd().resolve()


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def pick_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {normalize_col_name(c): c for c in columns}
    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized:
            return normalized[key]
    return None


def detect_columns(path: Path) -> ColumnMap:
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)

    colmap: ColumnMap = {
        "created_date": pick_column(cols, ["created_date", "created date", "createddate"]),
        "complaint_type": pick_column(cols, ["complaint_type", "complaint type", "complainttype"]),
        "descriptor": pick_column(cols, ["descriptor", "incident_descriptor"]),
        "nta2020": pick_column(cols, ["nta2020", "nta", "nta_code", "ntacode"]),
        "ntaname": pick_column(cols, ["ntaname", "nta_name", "nta name"]),
        "boroname": pick_column(cols, ["boroname", "boro_name", "borough", "borough_name"]),
        "unique_key": pick_column(cols, ["unique_key", "unique key", "uniquekey"]),
    }
    return colmap


def selected_usecols(colmap: ColumnMap) -> List[str]:
    required_keys = ["created_date", "complaint_type", "nta2020"]
    missing = [k for k in required_keys if not colmap.get(k)]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return [c for c in colmap.values() if c is not None]


def parse_datetime_series(s: pd.Series) -> pd.Series:
    # pandas >= 2 supports format="mixed"; older versions do not.
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, errors="coerce")


def monday_week_start(dt: pd.Series) -> pd.Series:
    dates = dt.dt.floor("D")
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def categorize_complaints(complaint_type: pd.Series, descriptor: Optional[pd.Series] = None) -> pd.Series:
    ctype = complaint_type.fillna("").astype(str)
    if descriptor is None:
        text = ctype
    else:
        text = ctype + " " + descriptor.fillna("").astype(str)

    text = text.str.lower()
    out = pd.Series(np.full(len(text), "other", dtype=object), index=text.index)

    unmatched = pd.Series(True, index=text.index)
    for category in CATEGORY_ORDER:
        pattern = CATEGORY_PATTERNS[category]
        mask = unmatched & text.str.contains(pattern, regex=True, na=False)
        out.loc[mask] = category
        unmatched.loc[mask] = False
    return out


def update_counter_from_grouped(
    agg_counter: defaultdict,
    grouped: pd.DataFrame,
) -> None:
    for row in grouped.itertuples(index=False):
        key = (str(row.nta2020), str(row.week_start), str(row.complaint_category))
        agg_counter[key] += int(row.complaint_count)


def update_counter_from_series(counter: Counter, s: pd.Series) -> None:
    vc = s.fillna("__MISSING__").astype(str).value_counts(dropna=False)
    counter.update(vc.to_dict())


def process_file(
    path: Path,
    chunksize: int,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    agg_counter: defaultdict,
    other_type_counter: Counter,
    type_category_counter: Counter,
    nta_meta: Dict[str, Dict[str, str]],
) -> Dict[str, object]:
    t0 = time.time()
    file_summary: Dict[str, object] = {
        "file": str(path),
        "status": "started",
        "rows_read": 0,
        "rows_valid_date": 0,
        "rows_valid_nta": 0,
        "rows_used": 0,
        "rows_bad_date": 0,
        "rows_missing_nta": 0,
        "unique_weeks": 0,
        "unique_ntas": 0,
        "elapsed_sec": 0.0,
        "error": "",
    }

    try:
        colmap = detect_columns(path)
        usecols = selected_usecols(colmap)
    except Exception as e:
        file_summary.update({"status": "skipped", "error": str(e), "elapsed_sec": round(time.time() - t0, 3)})
        return file_summary

    created_col = colmap["created_date"]
    complaint_col = colmap["complaint_type"]
    descriptor_col = colmap.get("descriptor")
    nta_col = colmap["nta2020"]
    ntaname_col = colmap.get("ntaname")
    boroname_col = colmap.get("boroname")

    file_weeks = set()
    file_ntas = set()

    try:
        reader = pd.read_csv(
            path,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            n_read = len(chunk)
            file_summary["rows_read"] += n_read
            if n_read == 0:
                continue

            dt = parse_datetime_series(chunk[created_col])
            valid_date = dt.notna()
            file_summary["rows_valid_date"] += int(valid_date.sum())
            file_summary["rows_bad_date"] += int((~valid_date).sum())

            if start_date is not None:
                valid_date &= dt >= start_date
            if end_date is not None:
                # Include the whole end-date day if user passes YYYY-MM-DD.
                valid_date &= dt <= end_date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

            nta = chunk[nta_col].fillna("").astype(str).str.strip()
            valid_nta = nta.ne("") & nta.str.lower().ne("nan")
            file_summary["rows_valid_nta"] += int(valid_nta.sum())
            file_summary["rows_missing_nta"] += int((~valid_nta).sum())

            mask = valid_date & valid_nta
            if not mask.any():
                continue

            work = pd.DataFrame({
                "nta2020": nta.loc[mask].values,
                "created_dt": dt.loc[mask].values,
                "complaint_type": chunk.loc[mask, complaint_col].fillna("__MISSING__").astype(str).values,
            })
            if descriptor_col is not None:
                descriptor = chunk.loc[mask, descriptor_col].fillna("").astype(str).reset_index(drop=True)
            else:
                descriptor = None

            if ntaname_col is not None or boroname_col is not None:
                meta_chunk = pd.DataFrame({"nta2020": work["nta2020"]})
                if ntaname_col is not None:
                    meta_chunk["ntaname"] = chunk.loc[mask, ntaname_col].fillna("").astype(str).values
                if boroname_col is not None:
                    meta_chunk["boroname"] = chunk.loc[mask, boroname_col].fillna("").astype(str).values

                for meta_row in meta_chunk.drop_duplicates("nta2020").itertuples(index=False):
                    code = str(meta_row.nta2020)
                    if code not in nta_meta:
                        nta_meta[code] = {}
                    if hasattr(meta_row, "ntaname") and getattr(meta_row, "ntaname"):
                        nta_meta[code]["ntaname"] = getattr(meta_row, "ntaname")
                    if hasattr(meta_row, "boroname") and getattr(meta_row, "boroname"):
                        nta_meta[code]["boroname"] = getattr(meta_row, "boroname")

            work["week_start"] = monday_week_start(pd.to_datetime(work["created_dt"])).dt.strftime("%Y-%m-%d")
            work["complaint_category"] = categorize_complaints(work["complaint_type"], descriptor)
            work["complaint_count"] = 1

            file_summary["rows_used"] += len(work)
            file_weeks.update(work["week_start"].unique().tolist())
            file_ntas.update(work["nta2020"].unique().tolist())

            grouped = (
                work.groupby(["nta2020", "week_start", "complaint_category"], observed=True)["complaint_count"]
                .sum()
                .reset_index()
            )
            update_counter_from_grouped(agg_counter, grouped)

            # Diagnostics for semantic mapping.
            other_mask = work["complaint_category"].eq("other")
            if other_mask.any():
                update_counter_from_series(other_type_counter, work.loc[other_mask, "complaint_type"])

            type_cat = work.groupby(["complaint_type", "complaint_category"], observed=True).size()
            type_category_counter.update({(str(k[0]), str(k[1])): int(v) for k, v in type_cat.items()})

        file_summary.update({
            "status": "done",
            "unique_weeks": len(file_weeks),
            "unique_ntas": len(file_ntas),
            "elapsed_sec": round(time.time() - t0, 3),
        })
        return file_summary

    except Exception as e:
        file_summary.update({
            "status": "error",
            "error": repr(e),
            "elapsed_sec": round(time.time() - t0, 3),
        })
        return file_summary


def agg_counter_to_df(agg_counter: defaultdict) -> pd.DataFrame:
    if not agg_counter:
        return pd.DataFrame(columns=["nta2020", "week_start", "complaint_category", "complaint_count"])
    rows = [
        (nta, week, cat, count)
        for (nta, week, cat), count in agg_counter.items()
    ]
    df = pd.DataFrame(rows, columns=["nta2020", "week_start", "complaint_category", "complaint_count"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["complaint_count"] = df["complaint_count"].astype("int64")
    return df


def add_nta_meta(df: pd.DataFrame, nta_meta: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    if df.empty or not nta_meta:
        return df
    meta = pd.DataFrame([
        {
            "nta2020": code,
            "ntaname": vals.get("ntaname", ""),
            "boroname": vals.get("boroname", ""),
        }
        for code, vals in nta_meta.items()
    ])
    out = df.merge(meta, on="nta2020", how="left")
    first_cols = ["nta2020"]
    if "ntaname" in out.columns:
        first_cols.append("ntaname")
    if "boroname" in out.columns:
        first_cols.append("boroname")
    remaining = [c for c in out.columns if c not in first_cols]
    return out[first_cols + remaining]


def make_dense_weekly(observed: pd.DataFrame, nta_meta: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    if observed.empty:
        return observed.copy()

    min_week = observed["week_start"].min()
    max_week = observed["week_start"].max()
    weeks = pd.date_range(min_week, max_week, freq="W-MON")
    ntas = sorted(observed["nta2020"].dropna().astype(str).unique().tolist())

    base = pd.MultiIndex.from_product(
        [ntas, weeks, CATEGORIES],
        names=["nta2020", "week_start", "complaint_category"],
    ).to_frame(index=False)

    dense = base.merge(
        observed[["nta2020", "week_start", "complaint_category", "complaint_count"]],
        on=["nta2020", "week_start", "complaint_category"],
        how="left",
    )
    dense["complaint_count"] = dense["complaint_count"].fillna(0).astype("int64")
    dense["year"] = dense["week_start"].dt.year.astype("int16")
    dense["month"] = dense["week_start"].dt.month.astype("int8")
    dense["iso_week"] = dense["week_start"].dt.isocalendar().week.astype("int16")

    dense = add_nta_meta(dense, nta_meta)
    return dense.sort_values(["nta2020", "week_start", "complaint_category"]).reset_index(drop=True)


def write_type_category_counts(counter: Counter, path: Path) -> None:
    rows = [(ctype, cat, count) for (ctype, cat), count in counter.items()]
    df = pd.DataFrame(rows, columns=["complaint_type", "complaint_category", "complaint_count"])
    if not df.empty:
        df = df.sort_values("complaint_count", ascending=False)
    df.to_csv(path, index=False)


def check_output_paths(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        msg = "Output files already exist. Use --overwrite to regenerate:\n" + "\n".join(str(p) for p in existing)
        fail(msg)


def main() -> None:
    args = parse_args()
    root = project_root()

    input_dir = (root / args.input_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    summary_dir = (root / args.summary_dir).resolve()
    ensure_dirs(output_dir, summary_dir)

    if not input_dir.exists():
        fail(f"Input directory not found: {input_dir}")

    observed_out = output_dir / "nyc_311_weekly_by_nta_category_observed.csv.gz"
    dense_out = output_dir / "nyc_311_weekly_by_nta_category.csv.gz"
    summary_out = summary_dir / "aggregate_weekly_311_summary.csv"
    file_summary_out = summary_dir / "aggregate_weekly_311_file_summary.csv"
    category_counts_out = summary_dir / "weekly_311_category_counts.csv"
    other_top_out = summary_dir / "weekly_311_other_complaint_type_top200.csv"
    type_cat_out = summary_dir / "weekly_311_complaint_type_category_counts.csv"
    overview_json_out = summary_dir / "aggregate_weekly_311_overview.json"

    check_output_paths(
        [observed_out, dense_out, summary_out, file_summary_out, category_counts_out, other_top_out, type_cat_out, overview_json_out],
        args.overwrite,
    )

    files = sorted(input_dir.glob(args.pattern))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        fail(f"No input files found in {input_dir} with pattern {args.pattern}")

    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    print("=" * 90)
    print("Aggregate weekly NYC 311 by NTA/category")
    print(f"Input dir      : {input_dir}")
    print(f"Files          : {len(files)}")
    print(f"Chunksize      : {args.chunksize:,}")
    print(f"Output dir     : {output_dir}")
    print(f"Summary dir    : {summary_dir}")
    print("=" * 90)

    t_all = time.time()
    agg_counter: defaultdict = defaultdict(int)
    other_type_counter: Counter = Counter()
    type_category_counter: Counter = Counter()
    nta_meta: Dict[str, Dict[str, str]] = {}
    file_summaries: List[Dict[str, object]] = []

    for i, path in enumerate(files, start=1):
        fs = process_file(
            path=path,
            chunksize=args.chunksize,
            start_date=start_date,
            end_date=end_date,
            agg_counter=agg_counter,
            other_type_counter=other_type_counter,
            type_category_counter=type_category_counter,
            nta_meta=nta_meta,
        )
        file_summaries.append(fs)

        if i == 1 or i % args.progress_every == 0 or i == len(files):
            used = sum(int(x.get("rows_used", 0)) for x in file_summaries)
            elapsed = time.time() - t_all
            print(
                f"[{i:>4}/{len(files)}] status={fs['status']:<6} "
                f"rows_used_total={used:,} elapsed_min={elapsed/60:.2f} last={path.name}"
            )

    file_summary_df = pd.DataFrame(file_summaries)
    file_summary_df.to_csv(file_summary_out, index=False)

    observed = agg_counter_to_df(agg_counter)
    if observed.empty:
        fail("No rows were aggregated. Check input schema, dates, and nta2020 values.")

    observed = add_nta_meta(observed, nta_meta)
    sort_cols = [c for c in ["nta2020", "week_start", "complaint_category"] if c in observed.columns]
    observed = observed.sort_values(sort_cols).reset_index(drop=True)
    observed.to_csv(observed_out, index=False, compression="gzip")

    # Category totals based on observed rows, before dense zero filling.
    category_counts = (
        observed.groupby("complaint_category", observed=True)["complaint_count"]
        .sum()
        .reindex(CATEGORIES, fill_value=0)
        .reset_index()
        .rename(columns={"complaint_count": "total_complaint_count"})
    )
    total_count = category_counts["total_complaint_count"].sum()
    category_counts["share"] = np.where(total_count > 0, category_counts["total_complaint_count"] / total_count, 0.0)
    category_counts.to_csv(category_counts_out, index=False)

    dense = make_dense_weekly(
        observed[["nta2020", "week_start", "complaint_category", "complaint_count"]].copy(),
        nta_meta=nta_meta,
    )
    dense.to_csv(dense_out, index=False, compression="gzip")

    other_top = pd.DataFrame(
        other_type_counter.most_common(200),
        columns=["complaint_type", "complaint_count"],
    )
    other_top.to_csv(other_top_out, index=False)

    write_type_category_counts(type_category_counter, type_cat_out)

    elapsed_all = time.time() - t_all
    overview = {
        "status": "done",
        "files_requested": len(files),
        "files_done": int((file_summary_df["status"] == "done").sum()) if not file_summary_df.empty else 0,
        "files_skipped": int((file_summary_df["status"] == "skipped").sum()) if not file_summary_df.empty else 0,
        "files_error": int((file_summary_df["status"] == "error").sum()) if not file_summary_df.empty else 0,
        "rows_read": int(file_summary_df["rows_read"].sum()) if "rows_read" in file_summary_df else 0,
        "rows_used": int(file_summary_df["rows_used"].sum()) if "rows_used" in file_summary_df else 0,
        "bad_date_rows": int(file_summary_df["rows_bad_date"].sum()) if "rows_bad_date" in file_summary_df else 0,
        "missing_nta_rows": int(file_summary_df["rows_missing_nta"].sum()) if "rows_missing_nta" in file_summary_df else 0,
        "observed_rows": int(len(observed)),
        "dense_rows": int(len(dense)),
        "unique_ntas": int(dense["nta2020"].nunique()) if not dense.empty else 0,
        "unique_weeks": int(dense["week_start"].nunique()) if not dense.empty else 0,
        "categories": CATEGORIES,
        "min_week_start": str(pd.to_datetime(dense["week_start"]).min().date()) if not dense.empty else None,
        "max_week_start": str(pd.to_datetime(dense["week_start"]).max().date()) if not dense.empty else None,
        "total_complaint_count_observed": int(observed["complaint_count"].sum()),
        "elapsed_seconds": round(elapsed_all, 3),
        "elapsed_minutes": round(elapsed_all / 60, 3),
        "outputs": {
            "dense_weekly": str(dense_out),
            "observed_weekly": str(observed_out),
            "summary": str(summary_out),
            "file_summary": str(file_summary_out),
            "category_counts": str(category_counts_out),
            "other_top200": str(other_top_out),
            "type_category_counts": str(type_cat_out),
        },
    }

    pd.DataFrame([overview]).drop(columns=["categories", "outputs"]).to_csv(summary_out, index=False)
    with open(overview_json_out, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)

    print("\nDONE")
    print(f"Observed weekly output : {observed_out}")
    print(f"Dense weekly output    : {dense_out}")
    print(f"Summary CSV            : {summary_out}")
    print(f"Other top-200 CSV      : {other_top_out}")
    print("\nQuality-check next:")
    print("1) Open weekly_311_category_counts.csv and check whether 'other' dominates.")
    print("2) Open weekly_311_other_complaint_type_top200.csv; update mapping if major types are in 'other'.")
    print("3) Confirm week_start is Monday and complaint_count is non-negative integer.")


if __name__ == "__main__":
    main()
