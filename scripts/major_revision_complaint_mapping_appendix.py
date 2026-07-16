from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/processed/model_results/major_revision/complaint_mapping"
ROOT_REPORT = ROOT / "complaint_mapping_appendix_report.md"
TYPE_COUNTS = ROOT / "data/processed/_aggregate_summaries/weekly_311_complaint_type_category_counts.csv"
OTHER_TOP = ROOT / "data/processed/_aggregate_summaries/weekly_311_other_complaint_type_top200.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complaint-type mapping appendix artifacts.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {path}")
    df.to_csv(path, index=False, encoding="utf-8")


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    d = df.loc[:, cols].head(max_rows).copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        elif pd.api.types.is_integer_dtype(d[col]):
            d[col] = d[col].map(lambda x: f"{int(x)}")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(TYPE_COUNTS)
    mapping["complaint_count"] = pd.to_numeric(mapping["complaint_count"], errors="coerce").fillna(0).astype(int)
    total = int(mapping["complaint_count"].sum())
    mapping = mapping.sort_values(["complaint_category", "complaint_count", "complaint_type"], ascending=[True, False, True]).copy()
    mapping["share_of_all_requests"] = mapping["complaint_count"] / total if total else np.nan
    mapping["rank_within_category"] = mapping.groupby("complaint_category")["complaint_count"].rank(method="first", ascending=False).astype(int)

    category_summary = (
        mapping.groupby("complaint_category", as_index=False)
        .agg(
            complaint_types=("complaint_type", "nunique"),
            complaint_count=("complaint_count", "sum"),
        )
        .sort_values("complaint_count", ascending=False)
    )
    category_summary["share_of_all_requests"] = category_summary["complaint_count"] / total if total else np.nan

    other = mapping[mapping["complaint_category"].eq("other")].copy()
    other["share_of_other_requests"] = other["complaint_count"] / other["complaint_count"].sum() if len(other) else np.nan
    other = other.sort_values("complaint_count", ascending=False)

    top_other_existing = pd.read_csv(OTHER_TOP) if OTHER_TOP.exists() else other[["complaint_type", "complaint_count"]].copy()
    top_other_existing["complaint_count"] = pd.to_numeric(top_other_existing["complaint_count"], errors="coerce").fillna(0).astype(int)
    top_other_existing["share_of_other_requests"] = top_other_existing["complaint_count"] / other["complaint_count"].sum() if len(other) else np.nan

    return mapping, category_summary, other, top_other_existing


def write_report(out_dir: Path, mapping: pd.DataFrame, category_summary: pd.DataFrame, other: pd.DataFrame, top_other: pd.DataFrame) -> None:
    total = int(mapping["complaint_count"].sum())
    other_total = int(other["complaint_count"].sum()) if len(other) else 0
    lines = [
        "# Complaint Mapping Appendix Report",
        "",
        "This report generates reviewer-facing artifacts for P2-10: the mapping from original NYC 311 complaint types to the nine analysis categories and the composition of the `other` category.",
        "",
        "## Overall Category Summary",
        "",
        md_table(category_summary, ["complaint_category", "complaint_types", "complaint_count", "share_of_all_requests"]),
        "",
        "## Other Category Composition",
        "",
        f"- Total mapped requests: {total:,}.",
        f"- `other` contains {other['complaint_type'].nunique() if len(other) else 0} complaint types and {other_total:,} requests ({other_total / total:.4%} of all mapped requests).",
        "- The rows below are the largest `other` complaint types by request count.",
        "",
        md_table(top_other, ["complaint_type", "complaint_count", "share_of_other_requests"], max_rows=30),
        "",
        "## Appendix Files",
        "",
        "- `complaint_type_category_mapping_appendix.csv`: all complaint types, assigned category, counts, and shares.",
        "- `complaint_category_summary.csv`: category-level type counts and request shares.",
        "- `other_category_composition.csv`: all complaint types mapped to `other`.",
        "- `other_category_top200.csv`: existing top-200 `other` composition table with shares.",
        "",
        "## Guardrails",
        "",
        "- This pass documents the current deterministic mapping; it does not complete a sensitivity analysis with alternate groupings.",
        "- The manuscript should avoid implying that `other` is semantically homogeneous.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "complaint_mapping_appendix_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    mapping, category_summary, other, top_other = build_outputs()
    write_csv(out_dir / "complaint_type_category_mapping_appendix.csv", mapping, args.overwrite)
    write_csv(out_dir / "complaint_category_summary.csv", category_summary, args.overwrite)
    write_csv(out_dir / "other_category_composition.csv", other, args.overwrite)
    write_csv(out_dir / "other_category_top200.csv", top_other, args.overwrite)
    write_report(out_dir, mapping, category_summary, other, top_other)
    summary = {
        "status": "done",
        "script": "major_revision_complaint_mapping_appendix.py",
        "elapsed_seconds": round(time.time() - start, 3),
        "total_complaint_types": int(mapping["complaint_type"].nunique()),
        "total_requests": int(mapping["complaint_count"].sum()),
        "other_complaint_types": int(other["complaint_type"].nunique()),
        "other_requests": int(other["complaint_count"].sum()),
        "other_request_share": float(other["complaint_count"].sum() / mapping["complaint_count"].sum()),
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "complaint_mapping_appendix_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
