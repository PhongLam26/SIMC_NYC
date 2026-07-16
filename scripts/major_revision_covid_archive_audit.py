from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_model_audits import (  # noqa: E402
    COUNT_TARGET_COL,
    FORMULA_ALIGNED_REMOVE,
    SPLIT_COL,
    TARGET_COL,
    best_lgbm_params,
    load_dataset,
    load_feature_sets,
    select_threshold,
    write_csv,
    write_json,
)
from major_revision_rolling_origin import assign_fold_split  # noqa: E402
from major_revision_target_selection import encode_splits, make_targets, md_table, score_metrics  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/covid_archive"
ROOT_REPORT = ROOT / "covid_archive_report.md"
SCHEMA_REPORT = ROOT / "data/processed/_schema_checks/raw_schema_check_report.csv"
FILE_SUMMARY = ROOT / "data/processed/_aggregate_summaries/aggregate_weekly_311_file_summary.csv"
OBSERVED_WEEKLY = ROOT / "data/processed/weekly_311/nyc_311_weekly_by_nta_category_observed.csv.gz"
FINAL_DATASET = ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz"
RAW_311_DIR = ROOT / "data/raw/nyc_311"
PULL_SCRIPT = ROOT / "code_pulldata/pull_nyc_311.py"

FINAL_FOLD = {"fold_id": "final_style_2025", "train_end_year": 2023, "validation_year": 2024, "test_year": 2025}
COVID_TRAIN_YEARS = {2020, 2021}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit COVID/archive boundary issues for the major revision.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--target", default="T2_min_count_3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-raw-types", action="store_true")
    parser.add_argument("--raw-chunksize", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def year_month_from_path(path: str) -> tuple[int | None, int | None]:
    m = re.search(r"nyc_311_(\d{4})_(\d{2})", str(path))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def normalize_columns(text: str) -> tuple[str, ...]:
    return tuple(sorted(c.strip() for c in str(text).split("|") if c.strip()))


def schema_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"unique_key", "created_date", "closed_date", "complaint_type", "borough", "latitude", "longitude"}
    rows = []
    for fp in sorted(RAW_311_DIR.glob("nyc_311_*.csv.gz")):
        if "download_summary" in fp.name:
            continue
        year, month = year_month_from_path(fp.name)
        if year is None:
            continue
        try:
            cols = pd.read_csv(fp, nrows=0).columns.tolist()
            missing = sorted(required - set(cols))
            rows.append(
                {
                    "dataset": "nyc_311",
                    "file": str(fp.relative_to(ROOT)),
                    "status": "OK",
                    "year": year,
                    "month": month,
                    "num_columns": len(cols),
                    "required_missing": ", ".join(missing),
                    "columns": " | ".join(cols),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": "nyc_311",
                    "file": str(fp.relative_to(ROOT)),
                    "status": "ERROR",
                    "year": year,
                    "month": month,
                    "num_columns": np.nan,
                    "required_missing": "",
                    "columns": "",
                    "error": str(exc),
                }
            )
    nyc = pd.DataFrame(rows)
    nyc["column_set"] = nyc["columns"].map(normalize_columns)
    nyc["ordered_columns"] = nyc["columns"].astype(str)
    nyc["has_required_missing"] = nyc["required_missing"].fillna("").astype(str).str.strip().ne("")
    nyc["period"] = np.where(nyc["year"].le(2019), "2015-2019", "2020-2025")

    rows = []
    for period, g in nyc.groupby("period", sort=True):
        rows.append(
            {
                "period": period,
                "files": int(len(g)),
                "status_ok_files": int(g["status"].eq("OK").sum()),
                "files_with_required_missing": int(g["has_required_missing"].sum()),
                "unique_ordered_column_sequences": int(g["ordered_columns"].nunique()),
                "unique_unordered_column_sets": int(g["column_set"].nunique()),
                "min_year": int(g["year"].min()),
                "max_year": int(g["year"].max()),
            }
        )
    return nyc, pd.DataFrame(rows)


def file_rows_by_year() -> pd.DataFrame:
    files = pd.read_csv(FILE_SUMMARY)
    years_months = files["file"].map(year_month_from_path)
    files["year"] = [ym[0] for ym in years_months]
    files = files[files["year"].notna()].copy()
    group = (
        files.groupby("year", as_index=False)
        .agg(
            files=("file", "count"),
            rows_read=("total_rows_read", "sum"),
            rows_after_clean=("total_rows_after_clean", "sum"),
            bad_date_rows=("bad_date_rows", "sum"),
            missing_nta_rows=("missing_nta_rows", "sum"),
        )
        .sort_values("year")
    )
    group["period"] = np.where(group["year"].le(2019), "2015-2019", "2020-2025")
    group["missing_nta_share"] = group["missing_nta_rows"] / group["rows_read"].replace(0, np.nan)
    return group


def final_yearly_diagnostics() -> pd.DataFrame:
    cols = [
        "target_year",
        "is_covid_period",
        "final_train_ready_flag",
        TARGET_COL,
        COUNT_TARGET_COL,
        "rolling_8w_mean",
    ]
    header = pd.read_csv(FINAL_DATASET, nrows=0).columns
    df = pd.read_csv(FINAL_DATASET, usecols=[c for c in cols if c in header], low_memory=False)
    df = df[df["final_train_ready_flag"].eq(1) & df[TARGET_COL].notna() & df[COUNT_TARGET_COL].notna()].copy()
    df["target_year"] = df["target_year"].astype(int)
    df["t0_event"] = df[TARGET_COL].astype(int)
    df["t2_event"] = (df["t0_event"].eq(1) & df[COUNT_TARGET_COL].ge(3)).astype(int)
    rows = []
    for year, g in df.groupby("target_year", sort=True):
        rows.append(
            {
                "target_year": int(year),
                "rows": int(len(g)),
                "covid_period_rows": int(g["is_covid_period"].eq(1).sum()),
                "covid_period_share": float(g["is_covid_period"].eq(1).mean()),
                "t0_positive_rows": int(g["t0_event"].sum()),
                "t0_positive_share": float(g["t0_event"].mean()),
                "t2_positive_rows": int(g["t2_event"].sum()),
                "t2_positive_share": float(g["t2_event"].mean()),
                "mean_target_next_week_count": float(g[COUNT_TARGET_COL].mean()),
                "median_target_next_week_count": float(g[COUNT_TARGET_COL].median()),
                "positive_mu8w_lt_1_share_t0": float(g.loc[g["t0_event"].eq(1), "rolling_8w_mean"].lt(1).mean()),
                "positive_mu8w_lt_1_share_t2": float(g.loc[g["t2_event"].eq(1), "rolling_8w_mean"].lt(1).mean()),
            }
        )
    return pd.DataFrame(rows)


def category_year_counts() -> pd.DataFrame:
    weekly = pd.read_csv(OBSERVED_WEEKLY, usecols=["week_start", "complaint_category", "complaint_count"], parse_dates=["week_start"])
    weekly["year"] = weekly["week_start"].dt.year
    out = (
        weekly.groupby(["year", "complaint_category"], as_index=False)["complaint_count"]
        .sum()
        .sort_values(["year", "complaint_category"])
    )
    out["year_total"] = out.groupby("year")["complaint_count"].transform("sum")
    out["category_share"] = out["complaint_count"] / out["year_total"].replace(0, np.nan)
    return out


def scan_complaint_types(chunksize: int, progress: bool) -> pd.DataFrame:
    files = sorted(p for p in RAW_311_DIR.glob("nyc_311_*.csv.gz") if "download_summary" not in p.name)
    counts: Counter[tuple[int, str]] = Counter()
    for i, fp in enumerate(files, start=1):
        year, _ = year_month_from_path(fp.name)
        if year is None:
            continue
        if progress and (i == 1 or i % 50 == 0 or i == len(files)):
            print(f"[complaint-types] {i}/{len(files)} {fp.name}")
        for chunk in pd.read_csv(fp, usecols=["complaint_type"], chunksize=chunksize, low_memory=False):
            s = chunk["complaint_type"].fillna("__MISSING__").astype(str)
            for complaint_type, n in s.value_counts().items():
                counts[(year, complaint_type)] += int(n)
    rows = [
        {"year": year, "complaint_type": complaint_type, "rows": rows}
        for (year, complaint_type), rows in counts.items()
    ]
    out = pd.DataFrame(rows).sort_values(["year", "rows"], ascending=[True, False])
    out["period"] = np.where(out["year"].le(2019), "2015-2019", "2020-2025")
    return out


def complaint_type_overlap(type_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if type_year.empty:
        return pd.DataFrame(), pd.DataFrame()
    period_counts = (
        type_year.groupby(["period", "complaint_type"], as_index=False)["rows"].sum()
        .pivot(index="complaint_type", columns="period", values="rows")
        .fillna(0)
        .reset_index()
    )
    for col in ["2015-2019", "2020-2025"]:
        if col not in period_counts:
            period_counts[col] = 0
    period_counts["presence"] = np.select(
        [
            period_counts["2015-2019"].gt(0) & period_counts["2020-2025"].gt(0),
            period_counts["2015-2019"].gt(0),
            period_counts["2020-2025"].gt(0),
        ],
        ["both", "2015-2019_only", "2020-2025_only"],
        default="none",
    )
    summary = (
        period_counts.groupby("presence", as_index=False)
        .agg(complaint_types=("complaint_type", "count"), rows_2015_2019=("2015-2019", "sum"), rows_2020_2025=("2020-2025", "sum"))
        .sort_values("presence")
    )
    return period_counts.sort_values(["presence", "2020-2025", "2015-2019"], ascending=[True, False, False]), summary


def fit_final_style(features: list[str], categorical: list[str], target: str, seed: int, exclude_covid_train: bool, progress: bool) -> dict[str, object]:
    required_features = sorted(set(features + ["rolling_8w_mean"]))
    df = load_dataset({"features": required_features})
    frame = make_targets(df)[target]
    frame = assign_fold_split(frame, FINAL_FOLD)
    if exclude_covid_train:
        mask = frame[SPLIT_COL].eq("train") & frame["target_year"].astype(int).isin(COVID_TRAIN_YEARS)
        frame.loc[mask, SPLIT_COL] = "unused"
        frame = frame[frame[SPLIT_COL].isin(["train", "validation", "test"])].copy()
    split_counts = frame[SPLIT_COL].value_counts().to_dict()
    if progress:
        label = "exclude_2020_2021_train" if exclude_covid_train else "reference_train"
        print(f"[covid-sensitivity] {label}: {split_counts}")

    X_train, y_train, X_val, y_val, X_test, y_test, model_features = encode_splits(frame, features, categorical)
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    params = best_lgbm_params()
    model = LGBMClassifier(
        objective="binary",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=neg / pos if pos else 1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    threshold, val_f1 = select_threshold(y_val, val_scores)
    rec = score_metrics(y_test, test_scores, threshold)
    rec.update(
        {
            "config": "exclude_2020_2021_from_training" if exclude_covid_train else "reference_train_through_2023",
            "target_definition": target,
            "train_rows": int(len(y_train)),
            "train_positive_rows": int(y_train.sum()),
            "validation_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "validation_selected_f1": float(val_f1),
            "threshold": float(threshold),
            "raw_feature_count": int(len(features)),
            "model_feature_count": int(len(model_features)),
            "fit_seconds": float(fit_seconds),
            "seed": int(seed),
            "removed_train_years": "2020,2021" if exclude_covid_train else "",
            "removed_formula_aligned_features": ", ".join(FORMULA_ALIGNED_REMOVE),
        }
    )
    return rec


def sensitivity(args: argparse.Namespace) -> pd.DataFrame:
    feature_sets, categorical = load_feature_sets()
    features = feature_sets["B_no_8w_formula_features"]
    rows = [
        fit_final_style(features, categorical, args.target, args.seed, False, args.progress),
        fit_final_style(features, categorical, args.target, args.seed, True, args.progress),
    ]
    out = pd.DataFrame(rows)
    base = out[out["config"].eq("reference_train_through_2023")].iloc[0]
    for metric in ["pr_auc", "precision_at_5pct", "f1", "precision", "recall", "alert_rate"]:
        out[f"delta_vs_reference_{metric}"] = out[metric] - float(base[metric])
    return out


def write_report(
    out_dir: Path,
    schema_summary: pd.DataFrame,
    file_years: pd.DataFrame,
    yearly: pd.DataFrame,
    category_counts: pd.DataFrame,
    type_overlap_summary: pd.DataFrame,
    sensitivity_rows: pd.DataFrame,
    scan_raw_types: bool,
) -> None:
    pull_text = PULL_SCRIPT.read_text(encoding="utf-8", errors="ignore") if PULL_SCRIPT.exists() else ""
    pull_dataset_match = re.search(r'DATASET_ID\s*=\s*"([^"]+)"', pull_text)
    pull_dataset_id = pull_dataset_match.group(1) if pull_dataset_match else "not_found"

    cat_focus = category_counts[category_counts["year"].isin([2019, 2020, 2021, 2024, 2025])].copy()
    cat_focus = cat_focus.sort_values(["year", "complaint_count"], ascending=[True, False])
    year_focus = yearly[yearly["target_year"].isin([2019, 2020, 2021, 2022, 2024, 2025])].copy()

    lines = [
        "# COVID and Archive Boundary Audit",
        "",
        "This audit addresses reviewer P2-2 using only data already present in the workspace. It checks raw 311 schema consistency, yearly/COVID target composition, category drift, and a final-style training sensitivity that excludes 2020-2021 from the training window.",
        "",
        "## Provenance Guardrail",
        "",
        f"- `code_pulldata/pull_nyc_311.py` currently declares `DATASET_ID = {pull_dataset_id}`.",
        "- The final modeling panel does not retain a row-level `source_dataset` or archive identifier.",
        "- Therefore, this audit can compare 2015-2019 versus 2020-2025 periods and schema consistency, but it cannot prove row-level provenance from `76ig-c548` versus `erm2-nwe9` without regenerating data with a retained source-id column.",
        "- The manuscript should describe this as an archive/vintage limitation unless source provenance is regenerated and retained.",
        "",
        "## Raw 311 Schema Summary",
        "",
        md_table(schema_summary, ["period", "files", "status_ok_files", "files_with_required_missing", "unique_ordered_column_sequences", "unique_unordered_column_sets", "min_year", "max_year"]),
        "",
        "## Processed File Rows by Year",
        "",
        md_table(file_years, ["year", "period", "files", "rows_read", "rows_after_clean", "bad_date_rows", "missing_nta_rows", "missing_nta_share"], max_rows=20),
        "",
        "## Yearly Target/COVID Diagnostics",
        "",
        md_table(year_focus, ["target_year", "rows", "covid_period_share", "t0_positive_share", "t2_positive_share", "mean_target_next_week_count", "positive_mu8w_lt_1_share_t0", "positive_mu8w_lt_1_share_t2"], max_rows=20),
        "",
        "## Category Mix Around COVID and Final Holdout",
        "",
        md_table(cat_focus, ["year", "complaint_category", "complaint_count", "category_share"], max_rows=60),
        "",
        "## Complaint-Type Overlap",
        "",
    ]
    if scan_raw_types and not type_overlap_summary.empty:
        lines.extend(
            [
                md_table(type_overlap_summary, ["presence", "complaint_types", "rows_2015_2019", "rows_2020_2025"]),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Raw complaint-type scanning was not requested. Re-run with `--scan-raw-types` to produce `complaint_type_year_counts.csv` and overlap diagnostics.",
                "",
            ]
        )
    lines.extend(
        [
            "## Excluding 2020-2021 From Final-Style Training",
            "",
            md_table(sensitivity_rows, ["config", "train_rows", "train_positive_rows", "pr_auc", "precision_at_5pct", "f1", "precision", "recall", "alert_rate", "threshold", "delta_vs_reference_pr_auc", "delta_vs_reference_precision_at_5pct", "delta_vs_reference_f1"]),
            "",
            "## Interpretation Guardrails",
            "",
            "- This is a sensitivity check, not a final model-selection rule. It uses one seed and the current T2 no-shortcut LightGBM candidate.",
            "- 2020 and 2021 are visibly different in target prevalence and request volume, so the manuscript should discuss regime shift rather than treating the training period as homogeneous.",
            "- Exact archive-boundary provenance is not retained in the modeling rows; any final paper claim about `76ig-c548` versus `erm2-nwe9` must be phrased as source-data/vintage limitation unless the pipeline is regenerated with a source-id field.",
            "",
        ]
    )
    report = "\n".join(lines)
    (out_dir / "covid_archive_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    schema_detail, schema_summary = schema_audit()
    file_years = file_rows_by_year()
    yearly = final_yearly_diagnostics()
    categories = category_year_counts()

    type_year = pd.DataFrame()
    type_overlap = pd.DataFrame()
    type_overlap_summary = pd.DataFrame()
    if args.scan_raw_types:
        type_year = scan_complaint_types(args.raw_chunksize, args.progress)
        type_overlap, type_overlap_summary = complaint_type_overlap(type_year)

    sens = sensitivity(args)

    write_csv(out_dir / "raw_311_schema_detail.csv", schema_detail.drop(columns=["column_set"]), args.overwrite)
    write_csv(out_dir / "raw_311_schema_summary.csv", schema_summary, args.overwrite)
    write_csv(out_dir / "raw_311_file_rows_by_year.csv", file_years, args.overwrite)
    write_csv(out_dir / "final_yearly_target_diagnostics.csv", yearly, args.overwrite)
    write_csv(out_dir / "category_year_counts.csv", categories, args.overwrite)
    if args.scan_raw_types:
        write_csv(out_dir / "complaint_type_year_counts.csv", type_year, args.overwrite)
        write_csv(out_dir / "complaint_type_period_overlap.csv", type_overlap, args.overwrite)
        write_csv(out_dir / "complaint_type_period_overlap_summary.csv", type_overlap_summary, args.overwrite)
    write_csv(out_dir / "covid_exclusion_sensitivity.csv", sens, args.overwrite)
    write_report(out_dir, schema_summary, file_years, yearly, categories, type_overlap_summary, sens, args.scan_raw_types)
    write_json(
        out_dir / "covid_archive_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_covid_archive_audit.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "target": args.target,
            "seed": args.seed,
            "scan_raw_types": bool(args.scan_raw_types),
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
