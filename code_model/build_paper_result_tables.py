#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 10.5D — Build paper-ready result tables.

Project:
    SIMC NYC — Semantics-Aware Explainable Machine Learning for Urban Service
    Demand Forecasting in Smart Cities.

Purpose:
    Consolidate modeling outputs into clean CSV/Markdown tables that can be
    copied into the paper:

      - Final decision-layer comparison
      - Best final model summary
      - Confusion matrix summary
      - Per-category performance table
      - Per-category threshold table
      - Hyperparameter tuning best rows
      - Baseline/model comparison if baseline files are available
      - Suggested result-section wording

Recommended command:
    .\\.venv\\Scripts\\python.exe .\\code_model\\build_paper_result_tables.py --overwrite --progress ^
        --ensemble-dir data/processed/model_results/ensemble_category_thresholds ^
        --lgbm-tuning-dir data/processed/model_results/gbm_hyperparam_tuning_lgbm_balanced_gpu ^
        --xgb-tuning-dir data/processed/model_results/gbm_hyperparam_tuning_xgb_balanced_cuda ^
        --baselines-dir data/processed/model_results/baselines ^
        --output-dir data/processed/model_results/paper_tables

If you ran the ensemble weight search:
    change --ensemble-dir to:
        data/processed/model_results/ensemble_category_thresholds_weight_search

Outputs:
    paper_table_01_final_model_comparison.csv
    paper_table_02_final_model_summary.csv
    paper_table_03_final_model_confusion.csv
    paper_table_04_category_performance.csv
    paper_table_05_category_thresholds.csv
    paper_table_06_hyperparameter_best_models.csv
    paper_table_07_baseline_candidates.csv
    paper_results_tables.md
    paper_results_key_takeaways.md
    paper_result_tables_run_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


SCRIPT_NAME = "build_paper_result_tables"

DEFAULT_ENSEMBLE_DIR_REL = Path("data/processed/model_results/ensemble_category_thresholds")
DEFAULT_LGBM_DIR_REL = Path("data/processed/model_results/gbm_hyperparam_tuning_lgbm_balanced_gpu")
DEFAULT_XGB_DIR_REL = Path("data/processed/model_results/gbm_hyperparam_tuning_xgb_balanced_cuda")
DEFAULT_BASELINES_DIR_REL = Path("data/processed/model_results/baselines")
DEFAULT_OUTPUT_DIR_REL = Path("data/processed/model_results/paper_tables")


# ---------------------------------------------------------------------
# CLI / utilities
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready result tables from SIMC NYC model outputs.")

    parser.add_argument("--ensemble-dir", type=str, default=None)
    parser.add_argument("--lgbm-tuning-dir", type=str, default=None)
    parser.add_argument("--xgb-tuning-dir", type=str, default=None)
    parser.add_argument("--baselines-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--final-rank", type=int, default=1,
                        help="Use this validation_rank from final_model_comparison.csv as the selected final method.")
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")

    return parser.parse_args()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(project_root: Path, maybe_path: Optional[str], default_rel: Path) -> Path:
    if maybe_path:
        p = Path(maybe_path)
        return p if p.is_absolute() else project_root / p
    return project_root / default_rel


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")


def read_csv_if_exists(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def read_json_if_exists(path: Path, required: bool = False) -> dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    path.write_text(text, encoding="utf-8")


def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    if pd.isna(x):
        return None
    return x


def write_json(path: Path, data: dict, overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=to_jsonable)


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def round_numeric(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def fmt(x, digits: int = 4) -> str:
    val = safe_float(x)
    if pd.isna(val):
        return ""
    return f"{val:.{digits}f}"


def pct(x, digits: int = 1) -> str:
    val = safe_float(x)
    if pd.isna(val):
        return ""
    return f"{100.0 * val:.{digits}f}%"


def integer(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{int(round(float(x))):,}"
    except Exception:
        return ""


def md_table(df: pd.DataFrame, columns: List[str], rename: Optional[Dict[str, str]] = None, max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "_No data available._\n"

    d = df.copy()
    if max_rows:
        d = d.head(max_rows)

    cols = [c for c in columns if c in d.columns]
    d = d[cols]

    if rename:
        d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})

    # Convert all values to compact strings.
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{int(v):,}" if not pd.isna(v) else "")
        else:
            d[c] = d[c].astype(str).replace("nan", "")

    header = "| " + " | ".join(d.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.to_numpy()]
    return "\n".join([header, sep] + rows) + "\n"


# ---------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------

def load_ensemble_outputs(ensemble_dir: Path) -> dict:
    return {
        "comparison": read_csv_if_exists(ensemble_dir / "final_model_comparison.csv", required=True),
        "comparison_long": read_csv_if_exists(ensemble_dir / "final_model_comparison_long.csv", required=False),
        "threshold_selection": read_csv_if_exists(ensemble_dir / "threshold_selection_summary.csv", required=False),
        "category_thresholds": read_csv_if_exists(ensemble_dir / "category_thresholds.csv", required=False),
        "metrics_by_category": read_csv_if_exists(ensemble_dir / "metrics_by_category.csv", required=False),
        "confusion": read_csv_if_exists(ensemble_dir / "confusion_matrices.csv", required=False),
        "score_summary": read_csv_if_exists(ensemble_dir / "score_summary.csv", required=False),
        "scored": read_csv_if_exists(ensemble_dir / "scored_validation_test.csv.gz", required=False),
        "summary": read_json_if_exists(ensemble_dir / "ensemble_threshold_run_summary.json", required=False),
    }


def select_final_method(comparison: pd.DataFrame, final_rank: int) -> pd.Series:
    if comparison.empty:
        raise ValueError("final_model_comparison.csv is empty.")

    if "validation_rank" in comparison.columns:
        row = comparison[comparison["validation_rank"].astype(int) == int(final_rank)]
        if row.empty:
            row = comparison.sort_values("validation_rank").head(1)
    else:
        row = comparison.head(final_rank).tail(1)

    return row.iloc[0]


def build_final_comparison_table(comparison: pd.DataFrame, digits: int) -> pd.DataFrame:
    cols = [
        "validation_rank",
        "method_id",
        "model_label",
        "threshold_mode",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_pr_auc",
        "val_roc_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pr_auc",
        "test_roc_auc",
        "test_balanced_accuracy",
        "test_predicted_positive_share",
        "test_tp",
        "test_fp",
        "test_fn",
        "test_tn",
    ]
    present = [c for c in cols if c in comparison.columns]
    out = comparison[present].copy()

    if "validation_rank" in out.columns:
        out = out.sort_values("validation_rank").reset_index(drop=True)

    return round_numeric(out, digits)


def build_final_summary_table(final_row: pd.Series, digits: int) -> pd.DataFrame:
    fields = [
        ("Selected method", "method_id"),
        ("Model label", "model_label"),
        ("Threshold mode", "threshold_mode"),
        ("Threshold strategy", "threshold_strategy"),
        ("Validation F1", "val_f1"),
        ("Validation precision", "val_precision"),
        ("Validation recall", "val_recall"),
        ("Validation PR-AUC", "val_pr_auc"),
        ("Validation ROC-AUC", "val_roc_auc"),
        ("Test F1", "test_f1"),
        ("Test precision", "test_precision"),
        ("Test recall", "test_recall"),
        ("Test PR-AUC", "test_pr_auc"),
        ("Test ROC-AUC", "test_roc_auc"),
        ("Test balanced accuracy", "test_balanced_accuracy"),
        ("Test predicted positive share", "test_predicted_positive_share"),
        ("Test true positives", "test_tp"),
        ("Test false positives", "test_fp"),
        ("Test false negatives", "test_fn"),
        ("Test true negatives", "test_tn"),
    ]

    rows = []
    for label, key in fields:
        if key not in final_row.index:
            continue
        val = final_row[key]
        if isinstance(val, float):
            val = round(float(val), digits) if not pd.isna(val) else ""
        rows.append({"item": label, "value": val})

    return pd.DataFrame(rows)


def build_confusion_table(final_row: pd.Series, digits: int) -> pd.DataFrame:
    rows = []
    for split_prefix, split_name in [("val", "validation"), ("test", "test")]:
        rows.append({
            "split": split_name,
            "rows": final_row.get(f"{split_prefix}_rows", np.nan),
            "positive_rows": final_row.get(f"{split_prefix}_positive_rows", np.nan),
            "predicted_positive_rows": final_row.get(f"{split_prefix}_predicted_positive_rows", np.nan),
            "tn": final_row.get(f"{split_prefix}_tn", np.nan),
            "fp": final_row.get(f"{split_prefix}_fp", np.nan),
            "fn": final_row.get(f"{split_prefix}_fn", np.nan),
            "tp": final_row.get(f"{split_prefix}_tp", np.nan),
            "precision": final_row.get(f"{split_prefix}_precision", np.nan),
            "recall": final_row.get(f"{split_prefix}_recall", np.nan),
            "f1": final_row.get(f"{split_prefix}_f1", np.nan),
            "pr_auc": final_row.get(f"{split_prefix}_pr_auc", np.nan),
        })
    return round_numeric(pd.DataFrame(rows), digits)


def build_lift_at_k_table(scored: pd.DataFrame, final_row: pd.Series, final_score_col: str, digits: int) -> pd.DataFrame:
    if scored.empty or final_score_col not in scored.columns or "y_true" not in scored.columns:
        return pd.DataFrame()

    df = scored.copy()
    split_col = "split" if "split" in df.columns else "time_split" if "time_split" in df.columns else None
    if split_col:
        df = df[df[split_col].astype(str).str.lower() == "test"]
    if df.empty:
        return pd.DataFrame()

    df = df[[final_score_col, "y_true"]].dropna()
    if df.empty:
        return pd.DataFrame()

    df["y_true"] = df["y_true"].astype(int)
    total_rows = int(len(df))
    total_pos = int(df["y_true"].sum())
    base_rate = total_pos / total_rows if total_rows else np.nan
    ranked = df.sort_values(final_score_col, ascending=False).reset_index(drop=True)

    rows = []
    for frac, label in [(0.01, "Top 1% predicted risk"), (0.05, "Top 5% predicted risk"), (0.10, "Top 10% predicted risk")]:
        n = max(1, int(np.ceil(total_rows * frac)))
        sub = ranked.head(n)
        tp = int(sub["y_true"].sum())
        precision = tp / n if n else np.nan
        recall = tp / total_pos if total_pos else np.nan
        lift = precision / base_rate if base_rate and not pd.isna(base_rate) else np.nan
        rows.append({
            "risk_set": label,
            "rows_or_alerts": n,
            "true_positives": tp,
            "precision": precision,
            "recall": recall,
            "lift_over_base_rate": lift,
            "base_rate": base_rate,
        })

    alert_rows = safe_float(final_row.get("test_predicted_positive_rows", np.nan))
    alert_tp = safe_float(final_row.get("test_tp", np.nan))
    alert_precision = safe_float(final_row.get("test_precision", np.nan))
    alert_recall = safe_float(final_row.get("test_recall", np.nan))
    rows.append({
        "risk_set": "All category-threshold alerts",
        "rows_or_alerts": alert_rows,
        "true_positives": alert_tp,
        "precision": alert_precision,
        "recall": alert_recall,
        "lift_over_base_rate": alert_precision / base_rate if base_rate and not pd.isna(base_rate) else np.nan,
        "base_rate": base_rate,
    })

    return round_numeric(pd.DataFrame(rows), digits)


def build_category_performance_table(metrics_by_category: pd.DataFrame, final_method_id: str, digits: int) -> pd.DataFrame:
    if metrics_by_category.empty:
        return pd.DataFrame()

    df = metrics_by_category.copy()
    if "method_id" in df.columns:
        df = df[df["method_id"] == final_method_id]
    if "split" in df.columns:
        df = df[df["split"] == "test"]

    cols = [
        "complaint_category",
        "rows",
        "positive_rows",
        "positive_share",
        "predicted_positive_rows",
        "predicted_positive_share",
        "f1",
        "precision",
        "recall",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()

    if "f1" in out.columns:
        out = out.sort_values("f1", ascending=False).reset_index(drop=True)

    return round_numeric(out, digits)


def build_category_threshold_table(category_thresholds: pd.DataFrame, final_score_col: str, digits: int) -> pd.DataFrame:
    if category_thresholds.empty:
        return pd.DataFrame()

    df = category_thresholds.copy()
    if "score_col" in df.columns:
        df = df[df["score_col"] == final_score_col]

    cols = [
        "complaint_category",
        "threshold",
        "global_threshold_fallback",
        "validation_rows",
        "validation_positive_rows",
        "validation_f1",
        "validation_precision",
        "validation_recall",
        "validation_pr_auc",
        "validation_roc_auc",
        "note",
    ]
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    if "complaint_category" in out.columns:
        out = out.sort_values("complaint_category").reset_index(drop=True)

    return round_numeric(out, digits)


def build_hyperparam_best_table(lgbm_dir: Path, xgb_dir: Path, digits: int) -> pd.DataFrame:
    rows = []

    for label, path in [
        ("Tuned LightGBM", lgbm_dir / "gbm_tuning_ranking.csv"),
        ("Tuned XGBoost", xgb_dir / "gbm_tuning_ranking.csv"),
    ]:
        df = read_csv_if_exists(path, required=False)
        if df.empty:
            continue

        if "validation_rank" in df.columns:
            best = df.sort_values("validation_rank").iloc[0]
        else:
            best = df.iloc[0]

        row = {
            "model_label": label,
            "candidate_id": best.get("candidate_id", ""),
            "param_id": best.get("param_id", ""),
            "pos_weight_mode": best.get("pos_weight_mode", ""),
            "pos_weight": best.get("pos_weight", np.nan),
            "selected_threshold": best.get("selected_threshold", np.nan),
            "validation_f1": best.get("validation_f1", np.nan),
            "validation_precision": best.get("validation_precision", np.nan),
            "validation_recall": best.get("validation_recall", np.nan),
            "validation_pr_auc": best.get("validation_pr_auc", np.nan),
            "validation_roc_auc": best.get("validation_roc_auc", np.nan),
            "test_f1": best.get("test_f1", np.nan),
            "test_precision": best.get("test_precision", np.nan),
            "test_recall": best.get("test_recall", np.nan),
            "test_pr_auc": best.get("test_pr_auc", np.nan),
            "test_roc_auc": best.get("test_roc_auc", np.nan),
            "fit_seconds": best.get("fit_seconds", np.nan),
            "source_file": str(path),
        }
        rows.append(row)

    return round_numeric(pd.DataFrame(rows), digits)


def find_baseline_metrics_files(baselines_dir: Path) -> List[Path]:
    if not baselines_dir.exists():
        return []
    candidates = []
    for p in sorted(baselines_dir.rglob("*.csv")):
        name = p.name.lower()
        if "metric" in name or "result" in name:
            candidates.append(p)
    return candidates


def build_baseline_candidates_table(baselines_dir: Path, digits: int) -> pd.DataFrame:
    files = find_baseline_metrics_files(baselines_dir)
    rows = []

    metric_cols_required = {"f1", "precision", "recall"}
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue

        lower_map = {c.lower(): c for c in df.columns}
        if not metric_cols_required.issubset(set(lower_map)):
            continue

        # Try to select validation rows first, then test rows.
        d = df.copy()
        if "split" in d.columns:
            d = d[d["split"].astype(str).isin(["validation", "test"])]
        if "experiment_id" in d.columns:
            d = d[d["experiment_id"].astype(str).str.contains("E1", na=False)]
        if "feature_set" in d.columns:
            # Keep all feature sets; sorting will pick strong baselines at top.
            pass

        # Keep plausible model identifiers.
        id_cols = [c for c in ["experiment_id", "model_name", "baseline_name", "feature_set", "split", "threshold_strategy"] if c in d.columns]
        cols = id_cols + [c for c in ["f1", "precision", "recall", "pr_auc", "roc_auc", "balanced_accuracy", "accuracy"] if c in d.columns]
        if not cols:
            continue

        d = d[cols].copy()
        d["source_file"] = str(path)
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    sort_cols = [c for c in ["split", "f1", "pr_auc"] if c in out.columns]
    if "f1" in out.columns:
        out = out.sort_values("f1", ascending=False).reset_index(drop=True)

    return round_numeric(out, digits)


# ---------------------------------------------------------------------
# Markdown narrative
# ---------------------------------------------------------------------

def build_markdown_tables(
    *,
    final_comparison: pd.DataFrame,
    final_summary: pd.DataFrame,
    confusion: pd.DataFrame,
    lift_at_k: pd.DataFrame,
    category_perf: pd.DataFrame,
    category_thresholds: pd.DataFrame,
    hyperparam_best: pd.DataFrame,
    baselines: pd.DataFrame,
    final_row: pd.Series,
    digits: int,
) -> str:
    method = str(final_row.get("model_label", final_row.get("method_id", "selected model")))
    threshold_mode = str(final_row.get("threshold_mode", ""))
    test_f1 = fmt(final_row.get("test_f1"), digits)
    test_precision = fmt(final_row.get("test_precision"), digits)
    test_recall = fmt(final_row.get("test_recall"), digits)
    test_pr_auc = fmt(final_row.get("test_pr_auc"), digits)
    val_f1 = fmt(final_row.get("val_f1"), digits)
    val_pr_auc = fmt(final_row.get("val_pr_auc"), digits)

    lines = []
    lines.append("# SIMC NYC — Paper-ready model result tables\n")
    lines.append("Generated by `build_paper_result_tables.py`.\n")
    lines.append("These tables are intended for drafting the Results and Experiments sections. Do not paste every table into the final paper; select based on page limits.\n")

    lines.append("## Selected final model\n")
    lines.append(
        f"The selected final decision strategy is **{method}** with **{threshold_mode}** thresholding. "
        f"It achieved validation F1 = **{val_f1}**, validation PR-AUC = **{val_pr_auc}**, "
        f"future-test F1 = **{test_f1}**, test precision = **{test_precision}**, "
        f"test recall = **{test_recall}**, and test PR-AUC = **{test_pr_auc}**.\n"
    )

    lines.append("### Table 1. Final model/decision-layer comparison\n")
    lines.append(md_table(
        final_comparison,
        columns=[
            "validation_rank", "model_label", "threshold_mode",
            "val_f1", "val_precision", "val_recall", "val_pr_auc",
            "test_f1", "test_precision", "test_recall", "test_pr_auc",
            "test_balanced_accuracy",
        ],
        rename={
            "validation_rank": "Rank",
            "model_label": "Model",
            "threshold_mode": "Threshold",
            "val_f1": "Val F1",
            "val_precision": "Val Precision",
            "val_recall": "Val Recall",
            "val_pr_auc": "Val PR-AUC",
            "test_f1": "Test F1",
            "test_precision": "Test Precision",
            "test_recall": "Test Recall",
            "test_pr_auc": "Test PR-AUC",
            "test_balanced_accuracy": "Test Bal. Acc.",
        },
        max_rows=12,
    ))

    lines.append("### Table 2. Selected final model summary\n")
    lines.append(md_table(final_summary, columns=["item", "value"], rename={"item": "Item", "value": "Value"}))

    lines.append("### Table 3. Confusion matrix summary\n")
    lines.append(md_table(
        confusion,
        columns=["split", "rows", "positive_rows", "predicted_positive_rows", "tn", "fp", "fn", "tp", "precision", "recall", "f1", "pr_auc"],
        rename={
            "split": "Split",
            "rows": "Rows",
            "positive_rows": "Actual positives",
            "predicted_positive_rows": "Predicted positives",
            "tn": "TN",
            "fp": "FP",
            "fn": "FN",
            "tp": "TP",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "pr_auc": "PR-AUC",
        },
    ))

    lines.append("### Table 4. Precision and lift at top-ranked risk sets\n")
    if lift_at_k.empty:
        lines.append("_No lift-at-K table was generated because the final scored validation/test file was not available._\n")
    else:
        lines.append(md_table(
            lift_at_k,
            columns=["risk_set", "rows_or_alerts", "true_positives", "precision", "recall", "lift_over_base_rate"],
            rename={
                "risk_set": "Risk set",
                "rows_or_alerts": "Rows / alerts",
                "true_positives": "True positives",
                "precision": "Precision",
                "recall": "Recall",
                "lift_over_base_rate": "Lift over base rate",
            },
        ))

    lines.append("### Table 5. Final model performance by complaint category\n")
    lines.append(md_table(
        category_perf,
        columns=["complaint_category", "positive_rows", "predicted_positive_rows", "f1", "precision", "recall", "pr_auc", "roc_auc", "balanced_accuracy"],
        rename={
            "complaint_category": "Category",
            "positive_rows": "Actual positives",
            "predicted_positive_rows": "Predicted positives",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
            "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC",
            "balanced_accuracy": "Bal. Acc.",
        },
    ))

    lines.append("### Table 6. Category-specific thresholds\n")
    lines.append(md_table(
        category_thresholds,
        columns=["complaint_category", "threshold", "validation_f1", "validation_precision", "validation_recall", "validation_pr_auc"],
        rename={
            "complaint_category": "Category",
            "threshold": "Threshold",
            "validation_f1": "Val F1",
            "validation_precision": "Val Precision",
            "validation_recall": "Val Recall",
            "validation_pr_auc": "Val PR-AUC",
        },
    ))

    lines.append("### Table 7. Best tuned single models\n")
    lines.append(md_table(
        hyperparam_best,
        columns=[
            "model_label", "param_id", "pos_weight_mode", "selected_threshold",
            "validation_f1", "validation_pr_auc", "test_f1", "test_precision", "test_recall", "test_pr_auc", "fit_seconds",
        ],
        rename={
            "model_label": "Model",
            "param_id": "Parameter set",
            "pos_weight_mode": "Pos. weight",
            "selected_threshold": "Threshold",
            "validation_f1": "Val F1",
            "validation_pr_auc": "Val PR-AUC",
            "test_f1": "Test F1",
            "test_precision": "Test Precision",
            "test_recall": "Test Recall",
            "test_pr_auc": "Test PR-AUC",
            "fit_seconds": "Fit seconds",
        },
    ))

    lines.append("### Table 8. Baseline candidates found in baseline result folder\n")
    if baselines.empty:
        lines.append("_No baseline candidate table was generated because no compatible baseline metrics CSV was detected. This is not fatal; use the original baseline output files directly if needed._\n")
    else:
        lines.append(md_table(
            baselines,
            columns=[c for c in ["model_name", "baseline_name", "feature_set", "split", "f1", "precision", "recall", "pr_auc", "roc_auc", "balanced_accuracy"] if c in baselines.columns],
            rename={
                "model_name": "Model",
                "baseline_name": "Baseline",
                "feature_set": "Feature set",
                "split": "Split",
                "f1": "F1",
                "precision": "Precision",
                "recall": "Recall",
                "pr_auc": "PR-AUC",
                "roc_auc": "ROC-AUC",
                "balanced_accuracy": "Bal. Acc.",
            },
            max_rows=20,
        ))

    lines.append("## Suggested Results-section wording\n")
    lines.append(
        "After hyperparameter tuning, the best LightGBM and XGBoost models obtained similar validation and future-test performance, indicating that gradient-boosted tree models provide stable improvements for next-week abnormal service-demand forecasting. "
        "We therefore evaluated an ensemble decision layer that averages the LightGBM and XGBoost probability scores and compared both global and complaint-category-specific thresholds selected exclusively on the validation period.\n"
    )
    lines.append(
        f"The best validation-ranked decision strategy was {method} with {threshold_mode} thresholding. "
        f"On the held-out 2024–2025 test period, this strategy achieved F1 = {test_f1}, precision = {test_precision}, recall = {test_recall}, and PR-AUC = {test_pr_auc}. "
        "The category-specific thresholding result supports the semantics-aware framing of the task: different municipal service categories exhibit different score distributions and benefit from category-aware conversion of risk scores into operational alerts.\n"
    )
    lines.append(
        "Because the abnormal-increase class is imbalanced, we interpret performance primarily using F1, precision, recall, PR-AUC, and balanced accuracy rather than raw accuracy. "
        "The confusion matrix should be reported to make the alert trade-off transparent: higher recall improves surge detection, while higher precision reduces false alerts.\n"
    )

    lines.append("## Suggested Discussion points\n")
    lines.append(
        "- The improvement from ensemble and per-category thresholding is modest in absolute F1, but it is methodologically meaningful because it improves the decision layer without changing the chronological train/validation/test protocol.\n"
        "- Per-category thresholds are aligned with the paper's semantic formulation: noise, housing, sanitation, traffic, and infrastructure complaints have different baseline dynamics and alert costs.\n"
        "- Category-level results should be discussed as evidence that urban service-demand forecasting is heterogeneous across municipal service domains.\n"
        "- Avoid claiming causality. The model identifies predictive associations between historical demand, context features, and abnormal reported demand.\n"
    )

    return "\n".join(lines)


def build_key_takeaways_markdown(final_row: pd.Series, category_perf: pd.DataFrame, hyperparam_best: pd.DataFrame, digits: int) -> str:
    method = str(final_row.get("model_label", final_row.get("method_id", "selected model")))
    threshold_mode = str(final_row.get("threshold_mode", ""))
    test_f1 = fmt(final_row.get("test_f1"), digits)
    test_precision = fmt(final_row.get("test_precision"), digits)
    test_recall = fmt(final_row.get("test_recall"), digits)
    test_pr_auc = fmt(final_row.get("test_pr_auc"), digits)
    test_acc = fmt(final_row.get("test_accuracy"), digits)

    lines = []
    lines.append("# Key takeaways for paper writing\n")

    lines.append("## Most important claims to use\n")
    lines.append(
        f"1. **Final model:** {method} with {threshold_mode} thresholding is the selected final method because it ranks first by validation F1.\n"
        f"2. **Future-test performance:** test F1 = {test_f1}, precision = {test_precision}, recall = {test_recall}, PR-AUC = {test_pr_auc}.\n"
        "3. **Semantics-aware decision layer:** category-specific thresholds improve the validation-ranked decision strategy and fit the paper's semantic service-demand framing.\n"
        "4. **Modeling protocol:** all thresholds are selected on the validation period and evaluated on the held-out future test period.\n"
        "5. **Imbalance-aware metrics:** report F1 and PR-AUC prominently; raw accuracy should be secondary.\n"
    )

    lines.append("## Plain-language interpretation\n")
    lines.append(
        f"On the future test period, the selected model has precision {test_precision} and recall {test_recall}. "
        "This means the system catches about half of true abnormal demand increases, while roughly three out of ten alerts correspond to true abnormal increases. "
        "This is more informative than saying only that the model has high accuracy, because the abnormal class is relatively rare.\n"
    )

    if not category_perf.empty and "f1" in category_perf.columns:
        best_cat = category_perf.sort_values("f1", ascending=False).head(3)
        worst_cat = category_perf.sort_values("f1", ascending=True).head(3)

        lines.append("## Strongest categories by test F1\n")
        for _, r in best_cat.iterrows():
            lines.append(
                f"- {r.get('complaint_category')}: F1={fmt(r.get('f1'), digits)}, "
                f"precision={fmt(r.get('precision'), digits)}, recall={fmt(r.get('recall'), digits)}"
            )
        lines.append("")

        lines.append("## Weakest categories by test F1\n")
        for _, r in worst_cat.iterrows():
            lines.append(
                f"- {r.get('complaint_category')}: F1={fmt(r.get('f1'), digits)}, "
                f"precision={fmt(r.get('precision'), digits)}, recall={fmt(r.get('recall'), digits)}"
            )
        lines.append("")

    lines.append("## What not to overclaim\n")
    lines.append(
        "- Do not claim that 311 requests represent all real urban problems; they represent reported service demand.\n"
        "- Do not claim SHAP/feature importance proves causality.\n"
        "- Do not claim per-category thresholds are universally optimal; they are validation-selected for this forecasting setup.\n"
        "- Do not use random split results as the main claim if chronological split results are available.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    project_root = get_project_root()

    ensemble_dir = resolve_path(project_root, args.ensemble_dir, DEFAULT_ENSEMBLE_DIR_REL)
    lgbm_dir = resolve_path(project_root, args.lgbm_tuning_dir, DEFAULT_LGBM_DIR_REL)
    xgb_dir = resolve_path(project_root, args.xgb_tuning_dir, DEFAULT_XGB_DIR_REL)
    baselines_dir = resolve_path(project_root, args.baselines_dir, DEFAULT_BASELINES_DIR_REL)
    output_dir = resolve_path(project_root, args.output_dir, DEFAULT_OUTPUT_DIR_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.progress:
        print(f"[setup] ensemble_dir={ensemble_dir}")
        print(f"[setup] lgbm_dir={lgbm_dir}")
        print(f"[setup] xgb_dir={xgb_dir}")
        print(f"[setup] baselines_dir={baselines_dir}")
        print(f"[setup] output_dir={output_dir}")

    ensemble = load_ensemble_outputs(ensemble_dir)
    comparison = ensemble["comparison"]
    final_row = select_final_method(comparison, args.final_rank)

    final_method_id = str(final_row.get("method_id", ""))
    final_score_col = str(final_row.get("score_col", ""))

    final_comparison = build_final_comparison_table(comparison, args.digits)
    final_summary = build_final_summary_table(final_row, args.digits)
    confusion = build_confusion_table(final_row, args.digits)
    lift_at_k = build_lift_at_k_table(ensemble["scored"], final_row, final_score_col, args.digits)
    category_perf = build_category_performance_table(ensemble["metrics_by_category"], final_method_id, args.digits)
    category_thresholds = build_category_threshold_table(ensemble["category_thresholds"], final_score_col, args.digits)
    hyperparam_best = build_hyperparam_best_table(lgbm_dir, xgb_dir, args.digits)
    baselines = build_baseline_candidates_table(baselines_dir, args.digits)

    write_csv(output_dir / "paper_table_01_final_model_comparison.csv", final_comparison, args.overwrite)
    write_csv(output_dir / "paper_table_02_final_model_summary.csv", final_summary, args.overwrite)
    write_csv(output_dir / "paper_table_03_final_model_confusion.csv", confusion, args.overwrite)
    write_csv(output_dir / "paper_table_04_category_performance.csv", category_perf, args.overwrite)
    write_csv(output_dir / "paper_table_05_category_thresholds.csv", category_thresholds, args.overwrite)
    write_csv(output_dir / "paper_table_06_hyperparameter_best_models.csv", hyperparam_best, args.overwrite)
    write_csv(output_dir / "paper_table_07_baseline_candidates.csv", baselines, args.overwrite)
    write_csv(output_dir / "paper_table_08_lift_at_k.csv", lift_at_k, args.overwrite)

    md = build_markdown_tables(
        final_comparison=final_comparison,
        final_summary=final_summary,
        confusion=confusion,
        lift_at_k=lift_at_k,
        category_perf=category_perf,
        category_thresholds=category_thresholds,
        hyperparam_best=hyperparam_best,
        baselines=baselines,
        final_row=final_row,
        digits=args.digits,
    )
    takeaways = build_key_takeaways_markdown(
        final_row=final_row,
        category_perf=category_perf,
        hyperparam_best=hyperparam_best,
        digits=args.digits,
    )

    write_text(output_dir / "paper_results_tables.md", md, args.overwrite)
    write_text(output_dir / "paper_results_key_takeaways.md", takeaways, args.overwrite)

    summary = {
        "script": SCRIPT_NAME,
        "status": "done",
        "ensemble_dir": str(ensemble_dir),
        "lgbm_tuning_dir": str(lgbm_dir),
        "xgb_tuning_dir": str(xgb_dir),
        "baselines_dir": str(baselines_dir),
        "output_dir": str(output_dir),
        "final_rank": int(args.final_rank),
        "final_method_id": final_method_id,
        "final_model_label": str(final_row.get("model_label", "")),
        "final_threshold_mode": str(final_row.get("threshold_mode", "")),
        "final_score_col": final_score_col,
        "final_validation_f1": safe_float(final_row.get("val_f1", np.nan)),
        "final_validation_pr_auc": safe_float(final_row.get("val_pr_auc", np.nan)),
        "final_test_f1": safe_float(final_row.get("test_f1", np.nan)),
        "final_test_precision": safe_float(final_row.get("test_precision", np.nan)),
        "final_test_recall": safe_float(final_row.get("test_recall", np.nan)),
        "final_test_pr_auc": safe_float(final_row.get("test_pr_auc", np.nan)),
        "final_test_roc_auc": safe_float(final_row.get("test_roc_auc", np.nan)),
        "final_test_tp": safe_float(final_row.get("test_tp", np.nan)),
        "final_test_fp": safe_float(final_row.get("test_fp", np.nan)),
        "final_test_fn": safe_float(final_row.get("test_fn", np.nan)),
        "final_test_tn": safe_float(final_row.get("test_tn", np.nan)),
        "table_rows": {
            "final_comparison": int(len(final_comparison)),
            "final_summary": int(len(final_summary)),
            "confusion": int(len(confusion)),
            "lift_at_k": int(len(lift_at_k)),
            "category_performance": int(len(category_perf)),
            "category_thresholds": int(len(category_thresholds)),
            "hyperparam_best": int(len(hyperparam_best)),
            "baseline_candidates": int(len(baselines)),
        },
        "outputs": {
            "paper_table_01_final_model_comparison": str(output_dir / "paper_table_01_final_model_comparison.csv"),
            "paper_table_02_final_model_summary": str(output_dir / "paper_table_02_final_model_summary.csv"),
            "paper_table_03_final_model_confusion": str(output_dir / "paper_table_03_final_model_confusion.csv"),
            "paper_table_04_category_performance": str(output_dir / "paper_table_04_category_performance.csv"),
            "paper_table_05_category_thresholds": str(output_dir / "paper_table_05_category_thresholds.csv"),
            "paper_table_06_hyperparameter_best_models": str(output_dir / "paper_table_06_hyperparameter_best_models.csv"),
            "paper_table_07_baseline_candidates": str(output_dir / "paper_table_07_baseline_candidates.csv"),
            "paper_table_08_lift_at_k": str(output_dir / "paper_table_08_lift_at_k.csv"),
            "paper_results_tables_md": str(output_dir / "paper_results_tables.md"),
            "paper_results_key_takeaways_md": str(output_dir / "paper_results_key_takeaways.md"),
        },
    }
    write_json(output_dir / "paper_result_tables_run_summary.json", summary, args.overwrite)

    if args.progress:
        print("\n[final selected method]")
        print(final_summary.to_string(index=False))
        print("\n[final comparison]")
        keep = [c for c in ["validation_rank", "model_label", "threshold_mode", "val_f1", "val_pr_auc", "test_f1", "test_precision", "test_recall", "test_pr_auc"] if c in final_comparison.columns]
        print(final_comparison[keep].to_string(index=False))
        print(f"\n[output] {output_dir}")

    print("=" * 90)
    print(f"[{SCRIPT_NAME}] DONE")
    print(f"Main Markdown: {output_dir / 'paper_results_tables.md'}")
    print(f"Key takeaways: {output_dir / 'paper_results_key_takeaways.md'}")
    print(f"Run summary: {output_dir / 'paper_result_tables_run_summary.json'}")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        raise
