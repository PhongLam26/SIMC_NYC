from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_model_audits import precision_at_k, write_csv  # noqa: E402
from major_revision_target_selection import md_table  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/disparity_workload"
ROOT_REPORT = ROOT / "disparity_workload_report.md"
PREDICTIONS = ROOT / "data/processed/model_results/major_revision/bootstrap/bootstrap_prediction_rows.csv.gz"
FINAL_DATASET = ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz"
KEYS = ["target_week", "nta2020", "complaint_category"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit alert workload and group performance using existing prediction rows.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--target", default="T2_min_count_3")
    parser.add_argument("--score-col", default="platt_score")
    parser.add_argument("--threshold-col", default="platt_threshold")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(target: str) -> pd.DataFrame:
    preds = pd.read_csv(PREDICTIONS, parse_dates=["target_week"])
    preds = preds[preds["target_definition"].eq(target)].copy()
    cols = KEYS + ["boroname", "rolling_8w_mean", "target_next_week_count"]
    final = pd.read_csv(FINAL_DATASET, usecols=cols, parse_dates=["target_week"], low_memory=False)
    final = final.drop_duplicates(KEYS)
    out = preds.merge(final, on=KEYS, how="left", validate="many_to_one")
    if out["boroname"].isna().any() or out["rolling_8w_mean"].isna().any():
        missing = int(out["boroname"].isna().sum() + out["rolling_8w_mean"].isna().sum())
        raise ValueError(f"Missing joined context values: {missing}")
    out["volume_decile"] = pd.qcut(
        out["rolling_8w_mean"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    return out


def group_metrics(df: pd.DataFrame, group_col: str, score_col: str, threshold_col: str) -> pd.DataFrame:
    total_rows = len(df)
    total_alerts = int((df[score_col] >= df[threshold_col]).sum())
    rows = []
    for group, g in df.groupby(group_col, observed=True, sort=True):
        y = g["y_true"].astype(int).to_numpy()
        score = g[score_col].to_numpy(dtype=float)
        pred = score >= g[threshold_col].to_numpy(dtype=float)
        rows.append(
            {
                "group_type": group_col,
                "group": str(group),
                "rows": int(len(g)),
                "row_share": float(len(g) / total_rows) if total_rows else np.nan,
                "positive_rows": int(y.sum()),
                "positive_share": float(y.mean()) if len(y) else np.nan,
                "alerts": int(pred.sum()),
                "alert_share": float(pred.sum() / total_alerts) if total_alerts else np.nan,
                "alert_rate": float(pred.mean()) if len(pred) else np.nan,
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
                "precision_at_1pct": precision_at_k(y, score, 0.01),
                "precision_at_5pct": precision_at_k(y, score, 0.05),
                "mean_score": float(np.mean(score)),
                "mean_rolling_8w_mean": float(g["rolling_8w_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def weekly_workload(df: pd.DataFrame, score_col: str, threshold_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["alert"] = d[score_col].ge(d[threshold_col]).astype(int)
    overall = (
        d.groupby("target_week", as_index=False)
        .agg(rows=("y_true", "size"), positives=("y_true", "sum"), alerts=("alert", "sum"))
        .sort_values("target_week")
    )
    overall["alert_rate"] = overall["alerts"] / overall["rows"]
    by_borough = (
        d.groupby(["target_week", "boroname"], as_index=False)
        .agg(rows=("y_true", "size"), positives=("y_true", "sum"), alerts=("alert", "sum"))
        .sort_values(["target_week", "boroname"])
    )
    by_borough["alert_rate"] = by_borough["alerts"] / by_borough["rows"]
    return overall, by_borough


def workload_summary(overall: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "weeks": int(len(overall)),
                "mean_alerts_per_week": float(overall["alerts"].mean()),
                "median_alerts_per_week": float(overall["alerts"].median()),
                "min_alerts_per_week": int(overall["alerts"].min()),
                "max_alerts_per_week": int(overall["alerts"].max()),
                "mean_positives_per_week": float(overall["positives"].mean()),
                "mean_alert_rate": float(overall["alert_rate"].mean()),
            }
        ]
    )


def write_report(out_dir: Path, group: pd.DataFrame, workload: pd.DataFrame, args: argparse.Namespace) -> None:
    borough = group[group["group_type"].eq("boroname")].sort_values("alert_rate", ascending=False)
    category = group[group["group_type"].eq("complaint_category")].sort_values("alert_rate", ascending=False)
    decile = group[group["group_type"].eq("volume_decile")].sort_values("group")
    lines = [
        "# Disparity and Workload Audit",
        "",
        f"This audit uses `{args.score_col}` prediction rows for `{args.target}` from the final-style 2025 bootstrap artifact. It evaluates observable group workload only; no socioeconomic or demographic data are used.",
        "",
        "## Weekly Workload Summary",
        "",
        md_table(workload, ["weeks", "mean_alerts_per_week", "median_alerts_per_week", "min_alerts_per_week", "max_alerts_per_week", "mean_positives_per_week", "mean_alert_rate"]),
        "",
        "## Borough Metrics",
        "",
        md_table(borough, ["group", "rows", "positive_share", "alerts", "alert_rate", "precision", "recall", "f1", "pr_auc", "precision_at_5pct"]),
        "",
        "## Complaint Category Metrics",
        "",
        md_table(category, ["group", "rows", "positive_share", "alerts", "alert_rate", "precision", "recall", "f1", "pr_auc", "precision_at_5pct"], max_rows=20),
        "",
        "## Historical-Volume Decile Metrics",
        "",
        md_table(decile, ["group", "rows", "mean_rolling_8w_mean", "positive_share", "alerts", "alert_rate", "precision", "recall", "f1", "pr_auc", "precision_at_5pct"], max_rows=20),
        "",
        "## Guardrails",
        "",
        "- This is not a socioeconomic fairness audit because ACS/Census demographic data are outside the no-new-external-data boundary for this revision.",
        "- Borough and volume-decile diagnostics are still useful for detecting workload concentration and performance heterogeneity in observable groups.",
        "- Final manuscript fairness language should explicitly defer socioeconomic fairness to Future Work rather than implying it has been completed.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "disparity_workload_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows = load_rows(args.target)
    groups = pd.concat(
        [
            group_metrics(rows, "boroname", args.score_col, args.threshold_col),
            group_metrics(rows, "complaint_category", args.score_col, args.threshold_col),
            group_metrics(rows, "volume_decile", args.score_col, args.threshold_col),
        ],
        ignore_index=True,
    )
    weekly, weekly_borough = weekly_workload(rows, args.score_col, args.threshold_col)
    summary = workload_summary(weekly)
    write_csv(out_dir / "group_disparity_metrics.csv", groups, args.overwrite)
    write_csv(out_dir / "weekly_workload.csv", weekly, args.overwrite)
    write_csv(out_dir / "weekly_workload_by_borough.csv", weekly_borough, args.overwrite)
    write_csv(out_dir / "weekly_workload_summary.csv", summary, args.overwrite)
    write_report(out_dir, groups, summary, args)
    payload = {
        "status": "done",
        "script": "major_revision_disparity_workload_audit.py",
        "elapsed_seconds": round(time.time() - start, 3),
        "target": args.target,
        "score_col": args.score_col,
        "threshold_col": args.threshold_col,
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "disparity_workload_run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
