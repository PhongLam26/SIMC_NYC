from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_DATASET = ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz"
PREDICTIONS = ROOT / "data/processed/model_results/major_revision/bootstrap/bootstrap_prediction_rows.csv.gz"
OUT_DIR = ROOT / "data/processed/model_results/major_revision/error_analysis"
ROOT_REPORT = ROOT / "error_severity_report.md"

KEYS = ["target_week", "nta2020", "complaint_category"]
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze recall by realized event severity for major revision.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--predictions", default=str(PREDICTIONS))
    parser.add_argument("--calibration-method", default="platt", choices=["uncalibrated", "platt", "isotonic"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {path}")
    df.to_csv(path, index=False, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_predictions(path: Path, method: str) -> pd.DataFrame:
    pred = pd.read_csv(path)
    pred["target_week"] = pd.to_datetime(pred["target_week"]).dt.strftime("%Y-%m-%d")
    pred["score"] = pred[f"{method}_score"]
    pred["threshold"] = pred[f"{method}_threshold"]
    pred["alert"] = pred["score"].ge(pred["threshold"]).astype(int)
    return pred


def load_context() -> pd.DataFrame:
    usecols = [
        "target_week",
        "nta2020",
        "complaint_category",
        "target_next_week_count",
        "rolling_8w_mean",
        "rolling_8w_std",
        "abnormal_threshold_8w",
    ]
    df = pd.read_csv(FINAL_DATASET, usecols=usecols, parse_dates=["target_week"])
    df["target_week"] = df["target_week"].dt.strftime("%Y-%m-%d")
    return df.drop_duplicates(KEYS)


def add_severity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sigma = out["rolling_8w_std"].fillna(0).clip(lower=0) + EPS
    mu = out["rolling_8w_mean"].fillna(0).clip(lower=0)
    count = out["target_next_week_count"].fillna(0).clip(lower=0)
    out["realized_z"] = (count - mu) / sigma
    out["absolute_increase"] = count - mu
    out["ratio_increase"] = count / (mu + EPS)
    out["target_count"] = count
    out["z_bin"] = pd.cut(
        out["realized_z"],
        bins=[-np.inf, 1.5, 2.0, 3.0, np.inf],
        labels=["z_lt_1.5", "z_1.5_to_2", "z_2_to_3", "z_ge_3"],
        right=False,
    ).astype(str)
    out["absolute_increase_bin"] = pd.cut(
        out["absolute_increase"],
        bins=[-np.inf, 1.0, 2.0, 4.0, 10.0, np.inf],
        labels=["lt_1", "plus_1", "plus_2_to_3", "plus_4_to_9", "plus_10_plus"],
        right=False,
    ).astype(str)
    out["target_count_bin"] = pd.cut(
        out["target_count"],
        bins=[-np.inf, 1.5, 2.5, 3.5, 9.5, np.inf],
        labels=["count_0_or_1", "count_2", "count_3", "count_4_to_9", "count_10_plus"],
        right=False,
    ).astype(str)
    return out


def summarize_positive_bins(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    rows = []
    positives = frame[frame["y_true"].eq(1)].copy()
    target_positive_counts = positives.groupby("target_definition").size().to_dict()
    for (target, label), g in positives.groupby(["target_definition", dimension], sort=True):
        all_positive = max(1, int(target_positive_counts.get(target, len(g))))
        tp = int(g["alert"].sum())
        fn = int(len(g) - tp)
        rows.append(
            {
                "target_definition": target,
                "dimension": dimension,
                "bin": label,
                "positive_rows": int(len(g)),
                "share_of_target_positives": float(len(g) / all_positive),
                "true_positive_rows": tp,
                "false_negative_rows": fn,
                "recall": float(tp / len(g)) if len(g) else np.nan,
                "median_score": float(g["score"].median()),
                "median_target_count": float(g["target_count"].median()),
                "median_realized_z": float(g["realized_z"].replace([np.inf, -np.inf], np.nan).median()),
                "median_absolute_increase": float(g["absolute_increase"].median()),
                "median_ratio_increase": float(g["ratio_increase"].replace([np.inf, -np.inf], np.nan).median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_overall(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, g in frame.groupby("target_definition", sort=True):
        y = g["y_true"].astype(int)
        alert = g["alert"].astype(int)
        positives = g[y.eq(1)]
        tp = int(((y == 1) & (alert == 1)).sum())
        fp = int(((y == 0) & (alert == 1)).sum())
        fn = int(((y == 1) & (alert == 0)).sum())
        rows.append(
            {
                "target_definition": target,
                "rows": int(len(g)),
                "positive_rows": int(y.sum()),
                "alert_rows": int(alert.sum()),
                "true_positive_rows": tp,
                "false_positive_rows": fp,
                "false_negative_rows": fn,
                "precision": float(tp / (tp + fp)) if tp + fp else np.nan,
                "recall": float(tp / (tp + fn)) if tp + fn else np.nan,
                "median_positive_score": float(positives["score"].median()) if len(positives) else np.nan,
                "median_fn_score": float(g[(y == 1) & (alert == 0)]["score"].median()) if fn else np.nan,
                "median_tp_score": float(g[(y == 1) & (alert == 1)]["score"].median()) if tp else np.nan,
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    d = df[[c for c in columns if c in df.columns]].head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            d[c] = d[c].astype(str)
    return "\n".join(
        [
            "| " + " | ".join(d.columns) + " |",
            "| " + " | ".join(["---"] * len(d.columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in d.to_numpy(dtype=str)],
        ]
    )


def write_report(out_dir: Path, overall: pd.DataFrame, severity: pd.DataFrame, method: str) -> str:
    z = severity[severity["dimension"].eq("z_bin")].copy()
    abs_bins = severity[severity["dimension"].eq("absolute_increase_bin")].copy()
    count_bins = severity[severity["dimension"].eq("target_count_bin")].copy()
    lines = [
        "# Error Severity Analysis",
        "",
        f"This report analyzes final-style 2025 errors using `{method}` scores from the no-shortcut LightGBM bootstrap prediction rows.",
        "",
        "Realized exceedance magnitude is computed as `z = (count_t+1 - mu_8w) / (sigma_8w + epsilon)`. Because near-zero sigma can inflate z, the report also includes absolute increase and target-count bands.",
        "",
        "## Overall Error Counts",
        "",
        md_table(
            overall,
            [
                "target_definition",
                "rows",
                "positive_rows",
                "alert_rows",
                "true_positive_rows",
                "false_positive_rows",
                "false_negative_rows",
                "precision",
                "recall",
                "median_positive_score",
                "median_fn_score",
                "median_tp_score",
            ],
        ),
        "",
        "## Recall by Realized z Severity",
        "",
        md_table(
            z,
            [
                "target_definition",
                "bin",
                "positive_rows",
                "share_of_target_positives",
                "true_positive_rows",
                "false_negative_rows",
                "recall",
                "median_score",
                "median_target_count",
                "median_realized_z",
            ],
            max_rows=80,
        ),
        "",
        "## Recall by Absolute Increase",
        "",
        md_table(
            abs_bins,
            [
                "target_definition",
                "bin",
                "positive_rows",
                "true_positive_rows",
                "false_negative_rows",
                "recall",
                "median_score",
                "median_absolute_increase",
                "median_ratio_increase",
            ],
            max_rows=80,
        ),
        "",
        "## Recall by Target Count",
        "",
        md_table(
            count_bins,
            [
                "target_definition",
                "bin",
                "positive_rows",
                "true_positive_rows",
                "false_negative_rows",
                "recall",
                "median_score",
                "median_target_count",
            ],
            max_rows=80,
        ),
        "",
        "## Guardrails",
        "",
        "- Severity bins are computed only for positive rows; precision is reported in the overall table, not within positive-only bins.",
        "- T0 and T2 use different positive labels, so compare their severity profiles descriptively rather than as a paired model comparison.",
        "- Near-zero sigma can make z very large; use the absolute-increase and target-count bands alongside z bins.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "error_severity_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path
    pred = load_predictions(pred_path, args.calibration_method)
    context = load_context()
    merged = pred.merge(context, on=KEYS, how="left", validate="many_to_one")
    if merged["target_next_week_count"].isna().any():
        missing = int(merged["target_next_week_count"].isna().sum())
        raise ValueError(f"Missing severity context for {missing} prediction rows")
    frame = add_severity(merged)
    overall = summarize_overall(frame)
    severity = pd.concat(
        [
            summarize_positive_bins(frame, "z_bin"),
            summarize_positive_bins(frame, "absolute_increase_bin"),
            summarize_positive_bins(frame, "target_count_bin"),
        ],
        ignore_index=True,
    )

    write_csv(out_dir / "error_severity_analysis.csv", severity, args.overwrite)
    write_csv(out_dir / "error_severity_overall.csv", overall, args.overwrite)
    write_report(out_dir, overall, severity, args.calibration_method)
    write_json(
        out_dir / "error_severity_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_error_severity.py",
            "prediction_rows": int(len(frame)),
            "targets": sorted(frame["target_definition"].unique()),
            "calibration_method": args.calibration_method,
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
