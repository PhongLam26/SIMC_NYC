"""
Aggregate NYC 311 requests joined to NTA into weekly NTA × complaint_category counts.

Run from project root:
    .\.venv\Scripts\python.exe .\code_processdata\aggregate_weekly_311.py --chunksize 200000 --overwrite

Run from code_processdata:
    ..\.venv\Scripts\python.exe .\aggregate_weekly_311.py --chunksize 200000 --overwrite

Folder rule:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
This script never uses Path.cwd() to build data paths.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_VERSION = "aggregate_weekly_311_fixed_v2_type_only_mapping"

# ---------------------------------------------------------------------
# Fixed project paths: same style as join_311_to_nta.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "nyc_311_with_nta"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "weekly_311"
SUMMARY_DIR = PROJECT_ROOT / "data" / "processed" / "_aggregate_summaries"

DENSE_WEEKLY_PATH = OUTPUT_DIR / "nyc_311_weekly_by_nta_category.csv.gz"
OBSERVED_WEEKLY_PATH = OUTPUT_DIR / "nyc_311_weekly_by_nta_category_observed.csv.gz"

SUMMARY_CSV_PATH = SUMMARY_DIR / "aggregate_weekly_311_summary.csv"
SUMMARY_JSON_PATH = SUMMARY_DIR / "aggregate_weekly_311_summary.json"
OVERVIEW_JSON_PATH = SUMMARY_DIR / "aggregate_weekly_311_overview.json"
FILE_SUMMARY_PATH = SUMMARY_DIR / "aggregate_weekly_311_file_summary.csv"
CATEGORY_COUNTS_PATH = SUMMARY_DIR / "weekly_311_category_counts.csv"
OTHER_TOP200_PATH = SUMMARY_DIR / "weekly_311_other_complaint_type_top200.csv"
TYPE_CATEGORY_COUNTS_PATH = SUMMARY_DIR / "weekly_311_complaint_type_category_counts.csv"

# Old versions of this script created partial files. This clean-up only removes
# the known partial folder under the correct weekly_311 output directory.
PARTIAL_DIR = OUTPUT_DIR / "_partial_file_aggregates"

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

CATEGORY_LABELS: Dict[str, str] = {
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

INPUT_REQUIRED = ["created_date", "complaint_type", "nta2020"]
INPUT_OPTIONAL = ["ntaname", "boroname", "borocode", "cdta2020", "cdtaname"]


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def norm_text(value) -> str:
    """Normalize complaint_type for stable exact matching."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------
# Semantic category mapping
# Important rule:
#     Category is decided by complaint_type only.
#     Descriptor is NOT used, because it caused bad splits such as:
#     Illegal Parking -> water_sewer when descriptor contains hydrant.
# ---------------------------------------------------------------------
EXACT_CATEGORY: Dict[str, str] = {}

def add_exact(category: str, names: Iterable[str]) -> None:
    for name in names:
        EXACT_CATEGORY[norm_text(name)] = category


# Noise & disturbance
add_exact("noise", [
    "Noise", "Noise - Residential", "Noise - Street/Sidewalk", "Noise - Commercial",
    "Noise - Vehicle", "Noise - Park", "Noise - House of Worship",
    "Noise - Helicopter", "Helicopter", "Collection Truck Noise",
    "Loud Music/Party", "Loud Talking", "Banging/Pounding",
])

# Parking, traffic & mobility
add_exact("parking_traffic", [
    "Illegal Parking", "Blocked Driveway", "Derelict Vehicles",
    "Traffic", "Traffic Signal Condition", "Traffic Signal",
    "Broken Muni Meter", "Street Sign - Damaged", "Street Sign - Missing",
    "Street Sign - Dangling", "Highway Sign - Damaged", "Highway Sign - Missing",
    "Highway Sign - Dangling", "Bike/Roller/Skate Chronic",
    "Abandoned Bike", "Derelict Bicycle", "Bike Rack Condition",
    "Bus Stop Shelter Complaint", "Bus Stop Shelter Placement",
    "Taxi Complaint", "Taxi Report", "For Hire Vehicle Complaint",
    "For Hire Vehicle Report", "FHV Licensee Complaint",
    "Green Taxi Complaint", "E-Scooter", "E-Bike", "Request Changes - A.S.P.",
    "Lost Property",
])

# Sanitation & waste
add_exact("sanitation", [
    "Request Large Bulky Item Collection", "Dirty Conditions",
    "Missed Collection", "Missed Collection (All Materials)", "Sanitation Condition",
    "Illegal Dumping", "Electronics Waste Appointment",
    "Residential Disposal Complaint", "Commercial Disposal Complaint",
    "Dumpster Complaint", "Overflowing Litter Baskets", "Litter Basket / Request",
    "Litter Basket Complaint", "Dead Animal", "Derelict Vehicle",
    "Recycling Enforcement", "Sweeping/Missed", "Sweeping/Inadequate",
    "Sweeping/Missed-Inadequate", "DSNY Spillage", "Lot Condition",
    "General Sanitation", "Graffiti", "Graffiti Removal",
    "Unsanitary Pigeon Condition", "Wood Pile Remaining",
    "Illegal Animal Kept as Pet", "Animal-Abuse", "Animal Abuse",
])

# Housing & building conditions
add_exact("housing", [
    "HEAT/HOT WATER", "Heat/Hot Water", "Heating", "HEATING",
    "UNSANITARY CONDITION", "PAINT/PLASTER", "PLUMBING", "DOOR/WINDOW",
    "WATER LEAK", "GENERAL", "General Construction/Plumbing", "FLOORING/STAIRS",
    "APPLIANCE", "ELECTRIC", "Electrical", "SAFETY", "OUTSIDE BUILDING",
    "ELEVATOR", "Elevator", "Boiler", "Building/Use", "Building Marshals office",
    "Building Condition", "Building Maintenance", "Maintenance or Facility",
    "Mold", "Asbestos", "Lead", "Indoor Air Quality", "Non-Residential Heat",
    "Window Guard", "Peeling Paint", "HPD Literature Request",
    "Special Projects Inspection Team (SPIT)", "Construction",
    "COVID-19 Non-essential Construction", "Scaffold Safety",
    "Cranes and Derricks", "BEST/Site Safety",
])

# Water & sewer
add_exact("water_sewer", [
    "Water System", "Sewer", "Water Conservation", "Water Quality",
    "Indoor Sewage", "Water Drainage", "Drinking Water", "Bottled Water",
    "Hydrant", "Street Flooding", "Catch Basin Clogged/Flooding",
    "Root/Sewer/Sidewalk Condition", "Beach/Pool/Sauna Complaint",
])

# Street infrastructure & public works
add_exact("infrastructure", [
    "Street Condition", "Street Light Condition", "Sidewalk Condition",
    "Curb Condition", "Highway Condition", "Bridge Condition",
    "Tunnel Condition", "Public Toilet", "Snow", "Snow or Ice", "Snow Removal",
    "Obstruction", "Damaged Tree", "Dead/Dying Tree", "Overgrown Tree/Branches",
    "New Tree Request", "Plant", "Street Sweeping Complaint",
    "Municipal Parking Facility", "Found Property",
])

# Environmental health & green space
add_exact("environment", [
    "Air Quality", "Air", "Rodent", "Standing Water", "Vacant Lot",
    "Mosquitoes", "Poison Ivy", "Harboring Bees/Wasps", "Illegal Tree Damage",
    "Plant", "Unsanitary Animal Pvt Property", "Animal Facility - No Permit",
    "Industrial Waste", "Hazardous Materials", "Hazmat Storage/Use",
    "Calorie Labeling", "Drinking", "Open Flame Permit",
])

# Public safety & regulatory enforcement
add_exact("public_safety", [
    "Non-Emergency Police Matter", "Consumer Complaint", "Food Establishment",
    "Food Poisoning", "Vendor Enforcement", "Mobile Food Vendor",
    "Other Enforcement", "Real Time Enforcement", "Violation of Park Rules",
    "Outdoor Dining", "Illegal Posting", "Smoking", "Smoking or Vaping",
    "Urinating in Public", "Day Care", "NonCompliance with Phased Reopening",
    "Vaccine Mandate Non-Compliance", "Mass Gathering Complaint",
    "Sustainability Enforcement", "Investigations and Discipline (IAD)",
    "Illegal Fireworks", "Homeless Person Assistance", "Homeless Encampment",
    "Panhandling", "Drinking", "Tattooing", "Public Payphone Complaint",
    "Disorderly Youth", "Senior Center Complaint", "School Maintenance",
    "Public Assembly - Temporary", "Fire Alarm - Replacement",
    "Emergency Response Team (ERT)",
])


# Fallback patterns on complaint_type only. Use non-capturing groups to avoid
# pandas regex warnings if reused vectorized in the future.
FALLBACK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bnoise\b|loud|helicopter"), "noise"),
    (re.compile(r"parking|driveway|vehicle|traffic|meter|taxi|fhv|e-?bike|e-?scooter|bike|skate|bus stop|street sign|highway sign"), "parking_traffic"),
    (re.compile(r"sanitation|collection|litter|dump|dumpster|recycl|disposal|garbage|rubbish|trash|dirty|dead animal|graffiti|pigeon|wood pile"), "sanitation"),
    (re.compile(r"heat|hot water|paint|plaster|plumb|window|door|floor|stairs|appliance|electric|elevator|boiler|building|mold|asbestos|lead|scaffold|construction|housing|hpd|maintenance"), "housing"),
    (re.compile(r"water system|sewer|water conservation|water quality|drainage|hydrant|flood|catch basin|bottled water|indoor sewage"), "water_sewer"),
    (re.compile(r"street condition|street light|sidewalk|curb|bridge|tunnel|highway|snow|ice|obstruction|tree|public toilet|public works"), "infrastructure"),
    (re.compile(r"air quality|rodent|standing water|mosquito|vacant lot|poison ivy|bees|wasps|hazard|industrial waste|animal facility"), "environment"),
    (re.compile(r"police|consumer|food|vendor|enforcement|park rules|smoking|vaping|urinating|day care|vaccine|gathering|fireworks|homeless|panhandling|public assembly|safety|discipline"), "public_safety"),
]


def categorize_complaint_type(value) -> str:
    text = norm_text(value)
    if not text:
        return "other"

    exact = EXACT_CATEGORY.get(text)
    if exact:
        return exact

    for pattern, category in FALLBACK_PATTERNS:
        if pattern.search(text):
            return category

    return "other"


# Backward-compatible alias so old debug commands that look for
# categorize_complaints will still resolve to the corrected type-only mapper.
def categorize_complaints(value) -> str:
    return categorize_complaint_type(value)


def list_input_files(limit_files: Optional[int] = None) -> List[Path]:
    files = sorted(INPUT_DIR.glob("*_with_nta.csv.gz"))
    if limit_files is not None:
        files = files[:limit_files]
    return files


def get_usecols(file_path: Path) -> List[str]:
    sample = pd.read_csv(file_path, nrows=5, low_memory=False)
    sample = normalize_columns(sample)
    cols = list(sample.columns)

    missing = [c for c in INPUT_REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"Input file missing required columns {missing}: {file_path}")

    usecols = [c for c in INPUT_REQUIRED + INPUT_OPTIONAL if c in cols]
    return usecols


def read_joined_chunks(file_path: Path, chunksize: int, usecols: List[str]):
    reader = pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        yield normalize_columns(chunk)


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        compression={"method": "gzip", "compresslevel": 1},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log("=" * 90)
    log("AGGREGATE WEEKLY NYC 311 BY NTA AND SEMANTIC CATEGORY")
    log("=" * 90)
    log(f"Script path:  {Path(__file__).resolve()}")
    log(f"Script version: {SCRIPT_VERSION}")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Input dir:    {INPUT_DIR}")
    log(f"Output dir:   {OUTPUT_DIR}")
    log(f"Summary dir:  {SUMMARY_DIR}")
    log(f"Chunksize:    {args.chunksize:,}")
    log("=" * 90)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}\n"
            "Expected joined files in data/processed/nyc_311_with_nta."
        )

    files = list_input_files(args.limit_files)
    if not files:
        raise FileNotFoundError(f"No *_with_nta.csv.gz files found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = [
        DENSE_WEEKLY_PATH,
        OBSERVED_WEEKLY_PATH,
        SUMMARY_CSV_PATH,
        SUMMARY_JSON_PATH,
        OVERVIEW_JSON_PATH,
        FILE_SUMMARY_PATH,
        CATEGORY_COUNTS_PATH,
        OTHER_TOP200_PATH,
        TYPE_CATEGORY_COUNTS_PATH,
    ]

    existing_outputs = [p for p in output_paths if p.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Output files already exist. Re-run with --overwrite.\n"
            + "\n".join(str(p) for p in existing_outputs[:10])
        )

    if args.overwrite:
        for p in output_paths:
            if p.exists():
                p.unlink()
        if PARTIAL_DIR.exists():
            shutil.rmtree(PARTIAL_DIR)

    log(f"Files to process: {len(files):,}")

    # Counters keep memory small and avoid creating many partial files.
    weekly_counter: Counter = Counter()
    type_counter: Counter = Counter()
    nta_meta: Dict[str, Dict[str, object]] = {}

    total_rows_read = 0
    total_rows_used = 0
    total_bad_date_rows = 0
    total_missing_nta_rows = 0
    file_summaries = []

    global_start = time.time()

    for i, fp in enumerate(files, start=1):
        file_start = time.time()
        status = "done"
        error = ""

        file_rows_read = 0
        file_rows_used = 0
        file_bad_date = 0
        file_missing_nta = 0
        chunk_count = 0

        try:
            usecols = get_usecols(fp)

            for chunk in read_joined_chunks(fp, args.chunksize, usecols):
                chunk_count += 1
                raw_n = len(chunk)
                file_rows_read += raw_n
                total_rows_read += raw_n

                # Clean date
                created = pd.to_datetime(chunk["created_date"], errors="coerce")
                bad_date_mask = created.isna()
                bad_date_n = int(bad_date_mask.sum())

                # Clean NTA
                nta_series = chunk["nta2020"].astype("string").str.strip()
                missing_nta_mask = nta_series.isna() | (nta_series == "")
                missing_nta_n = int(missing_nta_mask.sum())

                valid_mask = (~bad_date_mask) & (~missing_nta_mask)

                file_bad_date += bad_date_n
                file_missing_nta += missing_nta_n
                total_bad_date_rows += bad_date_n
                total_missing_nta_rows += missing_nta_n

                if not valid_mask.any():
                    continue

                work = chunk.loc[valid_mask].copy()
                work["nta2020"] = work["nta2020"].astype(str).str.strip()
                work["_created_dt"] = created.loc[valid_mask]
                work["week_start"] = (
                    work["_created_dt"]
                    .dt.to_period("W-SUN")
                    .dt.start_time
                    .dt.date
                    .astype(str)
                )

                # Map complaint_type -> category by complaint_type only.
                unique_types = pd.Series(work["complaint_type"].dropna().unique())
                local_type_map = {t: categorize_complaint_type(t) for t in unique_types}
                work["complaint_category"] = work["complaint_type"].map(local_type_map).fillna("other")

                # Keep latest metadata for each NTA.
                meta_cols = [c for c in INPUT_OPTIONAL if c in work.columns]
                if meta_cols:
                    meta_df = work[["nta2020"] + meta_cols].drop_duplicates("nta2020")
                    for row in meta_df.itertuples(index=False):
                        nta = getattr(row, "nta2020")
                        if nta not in nta_meta:
                            nta_meta[nta] = {}
                        for c in meta_cols:
                            val = getattr(row, c)
                            if pd.notna(val) and str(val).strip() != "":
                                nta_meta[nta][c] = val

                group = (
                    work.groupby(["nta2020", "week_start", "complaint_category"], observed=True)
                    .size()
                    .reset_index(name="complaint_count")
                )
                for row in group.itertuples(index=False):
                    weekly_counter[(row.nta2020, row.week_start, row.complaint_category)] += int(row.complaint_count)

                type_group = (
                    work.groupby(["complaint_type", "complaint_category"], observed=True)
                    .size()
                    .reset_index(name="complaint_count")
                )
                for row in type_group.itertuples(index=False):
                    type_counter[(str(row.complaint_type), row.complaint_category)] += int(row.complaint_count)

                used_n = int(len(work))
                file_rows_used += used_n
                total_rows_used += used_n

                del chunk, work, group, type_group
                gc.collect()

        except Exception as exc:
            status = "error"
            error = str(exc)

        elapsed = time.time() - file_start
        file_summaries.append({
            "file": safe_relative(fp),
            "status": status,
            "chunksize": args.chunksize,
            "chunk_count": chunk_count,
            "total_rows_read": file_rows_read,
            "total_rows_after_clean": file_rows_used,
            "bad_date_rows": file_bad_date,
            "missing_nta_rows": file_missing_nta,
            "elapsed_seconds": round(elapsed, 3),
            "error": error,
        })

        log(
            f"[{i:>4}/{len(files)}] status={status} "
            f"rows_used_total={total_rows_used:,} "
            f"elapsed_min={(time.time() - global_start) / 60:.2f} "
            f"last={fp.name}"
        )

    # Build observed weekly output
    observed_records = [
        {
            "nta2020": k[0],
            "week_start": k[1],
            "complaint_category": k[2],
            "complaint_count": v,
        }
        for k, v in weekly_counter.items()
    ]
    observed = pd.DataFrame(observed_records)

    if observed.empty:
        raise RuntimeError("No observed weekly records were produced.")

    observed["complaint_count"] = observed["complaint_count"].astype("int64")
    observed = observed.sort_values(["nta2020", "week_start", "complaint_category"]).reset_index(drop=True)

    # NTA metadata table
    nta_rows = []
    for nta in sorted(observed["nta2020"].unique()):
        row = {"nta2020": nta}
        row.update(nta_meta.get(nta, {}))
        nta_rows.append(row)
    nta_df = pd.DataFrame(nta_rows)

    if len(nta_df) > 0:
        observed = observed.merge(nta_df, on="nta2020", how="left")
        front = ["nta2020"] + [c for c in INPUT_OPTIONAL if c in observed.columns] + [
            "week_start", "complaint_category", "complaint_count"
        ]
        observed = observed[front]

    # Dense panel: every NTA x week_start x category
    min_week = observed["week_start"].min()
    max_week = observed["week_start"].max()
    weeks = pd.date_range(min_week, max_week, freq="W-MON").date.astype(str)
    ntas = sorted(observed["nta2020"].unique())

    dense_index = pd.MultiIndex.from_product(
        [ntas, weeks, CATEGORIES],
        names=["nta2020", "week_start", "complaint_category"],
    )
    dense = pd.DataFrame(index=dense_index).reset_index()
    dense = dense.merge(
        observed[["nta2020", "week_start", "complaint_category", "complaint_count"]],
        on=["nta2020", "week_start", "complaint_category"],
        how="left",
    )
    dense["complaint_count"] = dense["complaint_count"].fillna(0).astype("int64")

    if len(nta_df) > 0:
        dense = dense.merge(nta_df, on="nta2020", how="left")
        front = ["nta2020"] + [c for c in INPUT_OPTIONAL if c in dense.columns] + [
            "week_start", "complaint_category", "complaint_count"
        ]
        dense = dense[front]

    # Complaint type category counts
    type_records = [
        {
            "complaint_type": k[0],
            "complaint_category": k[1],
            "complaint_count": v,
        }
        for k, v in type_counter.items()
    ]
    type_counts = pd.DataFrame(type_records)
    type_counts["complaint_count"] = type_counts["complaint_count"].astype("int64")
    type_counts = type_counts.sort_values("complaint_count", ascending=False).reset_index(drop=True)

    # Category counts
    category_counts = (
        observed.groupby("complaint_category", observed=True)["complaint_count"]
        .sum()
        .reindex(CATEGORIES, fill_value=0)
        .reset_index()
        .rename(columns={"complaint_count": "total_complaint_count"})
    )
    total_observed_count = int(category_counts["total_complaint_count"].sum())
    category_counts["share"] = category_counts["total_complaint_count"] / total_observed_count

    # Other top 200
    other_top200 = (
        type_counts[type_counts["complaint_category"] == "other"]
        .groupby("complaint_type", observed=True)["complaint_count"]
        .sum()
        .sort_values(ascending=False)
        .head(200)
        .reset_index()
    )

    elapsed_seconds = round(time.time() - global_start, 3)
    summary = {
        "status": "done" if all(s["status"] == "done" for s in file_summaries) else "done_with_errors",
        "files_requested": len(files),
        "files_done": int(sum(s["status"] == "done" for s in file_summaries)),
        "files_skipped": 0,
        "files_error": int(sum(s["status"] == "error" for s in file_summaries)),
        "rows_read": int(total_rows_read),
        "rows_used": int(total_rows_used),
        "bad_date_rows": int(total_bad_date_rows),
        "missing_nta_rows": int(total_missing_nta_rows),
        "observed_rows": int(len(observed)),
        "dense_rows": int(len(dense)),
        "unique_ntas": int(len(ntas)),
        "unique_weeks": int(len(weeks)),
        "categories": CATEGORIES,
        "category_labels": CATEGORY_LABELS,
        "min_week_start": str(min_week),
        "max_week_start": str(max_week),
        "total_complaint_count_observed": total_observed_count,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": round(elapsed_seconds / 60, 3),
        "outputs": {
            "dense_weekly": str(DENSE_WEEKLY_PATH),
            "observed_weekly": str(OBSERVED_WEEKLY_PATH),
            "summary": str(SUMMARY_CSV_PATH),
            "file_summary": str(FILE_SUMMARY_PATH),
            "category_counts": str(CATEGORY_COUNTS_PATH),
            "other_top200": str(OTHER_TOP200_PATH),
            "type_category_counts": str(TYPE_CATEGORY_COUNTS_PATH),
        },
    }

    # Write outputs last, so old files are not replaced by partial results if crash occurs.
    write_csv_gz(dense, DENSE_WEEKLY_PATH)
    write_csv_gz(observed, OBSERVED_WEEKLY_PATH)
    pd.DataFrame(file_summaries).to_csv(FILE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    category_counts.to_csv(CATEGORY_COUNTS_PATH, index=False, encoding="utf-8-sig")
    other_top200.to_csv(OTHER_TOP200_PATH, index=False, encoding="utf-8-sig")
    type_counts.to_csv(TYPE_CATEGORY_COUNTS_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).drop(columns=["categories", "category_labels", "outputs"], errors="ignore").to_csv(
        SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig"
    )

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(OVERVIEW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("")
    log("=" * 90)
    log("ALL DONE")
    log("=" * 90)
    log(f"Files done:      {summary['files_done']:,}/{summary['files_requested']:,}")
    log(f"Rows read:       {summary['rows_read']:,}")
    log(f"Rows used:       {summary['rows_used']:,}")
    log(f"Missing NTA:     {summary['missing_nta_rows']:,}")
    log(f"Observed rows:   {summary['observed_rows']:,}")
    log(f"Dense rows:      {summary['dense_rows']:,}")
    log(f"Weeks:           {summary['unique_weeks']:,}")
    log(f"NTAs:            {summary['unique_ntas']:,}")
    log(f"Elapsed:         {summary['elapsed_minutes']:.2f} minutes")
    log("")
    log("Category distribution:")
    for row in category_counts.itertuples(index=False):
        log(
            f"  {row.complaint_category:<16} "
            f"{int(row.total_complaint_count):>12,} "
            f"{row.share:>8.2%}   {CATEGORY_LABELS.get(row.complaint_category, '')}"
        )


if __name__ == "__main__":
    main()
