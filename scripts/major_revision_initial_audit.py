from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
FINAL_DATASET = ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz"
SCORED = ROOT / "data/processed/model_results/prospective/ensemble/scored_validation_test.csv.gz"
THRESHOLDS = ROOT / "data/processed/model_results/prospective/ensemble/category_thresholds.csv"
OUT_DIR = ROOT / "data/processed/model_results/major_revision/initial_audit"

SCORE_COL = "ensemble_lgbm_w0p500_score"
TARGET_COL = "y_true"
KEYS = ["nta2020", "target_week", "complaint_category"]


def _safe_ap(y_true: pd.Series, score: pd.Series) -> float:
    if y_true.nunique(dropna=True) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def _classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame[TARGET_COL].astype(int)
    pred = frame["alert"].astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": _safe_ap(y, frame[SCORE_COL]),
        "alert_rate": float(pred.mean()),
    }


def _precision_at_k(frame: pd.DataFrame, k_share: float) -> float:
    if frame.empty:
        return np.nan
    k = max(1, int(np.ceil(len(frame) * k_share)))
    top = frame.nlargest(k, SCORE_COL)
    return float(top[TARGET_COL].mean())


def _read_scored_with_features() -> pd.DataFrame:
    scored = pd.read_csv(SCORED, parse_dates=["target_week", "week_start"])
    thresholds = pd.read_csv(THRESHOLDS)
    thresholds = thresholds[thresholds["score_col"].eq(SCORE_COL)]
    threshold_map = thresholds.set_index("complaint_category")["threshold"].to_dict()

    usecols = [
        "nta2020",
        "target_week",
        "complaint_category",
        "target_next_week_count",
        "rolling_8w_mean",
        "rolling_8w_std",
        "ratio_to_8w_mean",
        "rolling_12w_mean",
        "complaint_count",
        "history_weeks_available",
        "final_train_ready_flag",
        "time_split",
    ]
    final = pd.read_csv(FINAL_DATASET, usecols=usecols, parse_dates=["target_week"])
    merged = scored.merge(
        final.drop_duplicates(KEYS),
        on=KEYS,
        how="left",
        suffixes=("", "_final"),
        validate="many_to_one",
    )
    merged["selected_threshold"] = merged["complaint_category"].map(threshold_map)
    merged["alert"] = merged[SCORE_COL] >= merged["selected_threshold"]
    return merged


def build_sparse_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df[df["split"].eq("test")].copy()
    positives = test[test[TARGET_COL].eq(1)].copy()

    rows: list[dict[str, object]] = []
    rows.append({"metric": "test_rows", "value": len(test), "note": "Rows in held-out 2024-2025 scored set."})
    rows.append({"metric": "test_positive_rows", "value": int(positives.shape[0]), "note": "Observed abnormal rows."})
    rows.append({"metric": "test_positive_share", "value": float(test[TARGET_COL].mean()), "note": "Observed abnormal prevalence."})

    for threshold in [0.25, 0.5, 1.0, 2.0]:
        share = float((positives["rolling_8w_mean"] < threshold).mean())
        count = int((positives["rolling_8w_mean"] < threshold).sum())
        rows.append(
            {
                "metric": f"positive_rows_with_rolling_8w_mean_lt_{threshold:g}",
                "value": count,
                "share_of_test_positives": share,
                "note": "Reviewer P1-2 sparse-baseline diagnostic.",
            }
        )

    for label, mask in {
        "target_count_eq_1": positives["target_next_week_count"].eq(1),
        "target_count_eq_2": positives["target_next_week_count"].eq(2),
        "target_count_eq_3": positives["target_next_week_count"].eq(3),
        "target_count_ge_4": positives["target_next_week_count"].ge(4),
    }.items():
        rows.append(
            {
                "metric": label,
                "value": int(mask.sum()),
                "share_of_test_positives": float(mask.mean()),
                "note": "Composition of positive target rows.",
            }
        )

    category_rows = []
    for cat, g in test.groupby("complaint_category", sort=True):
        pos = g[g[TARGET_COL].eq(1)]
        category_rows.append(
            {
                "complaint_category": cat,
                "rows": len(g),
                "positive_rows": int(pos.shape[0]),
                "positive_share": float(g[TARGET_COL].mean()),
                "rolling_8w_mean_zero_share": float(g["rolling_8w_mean"].eq(0).mean()),
                "rolling_8w_mean_lt_1_share": float(g["rolling_8w_mean"].lt(1).mean()),
                "positive_rows_rolling_8w_mean_lt_1": int(pos["rolling_8w_mean"].lt(1).sum()),
                "positive_share_from_rolling_8w_mean_lt_1": float(pos["rolling_8w_mean"].lt(1).mean()) if len(pos) else np.nan,
                "mean_rolling_8w_mean": float(g["rolling_8w_mean"].mean()),
                "mean_target_next_week_count": float(g["target_next_week_count"].mean()),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(category_rows)


def build_volume_deciles(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"].eq("test")].copy()
    test["volume_decile"] = pd.qcut(
        test["rolling_8w_mean"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )

    rows = []
    for decile, g in test.groupby("volume_decile", observed=True):
        metrics = _classification_metrics(g)
        rows.append(
            {
                "volume_decile": decile,
                "rows": len(g),
                "rolling_8w_mean_min": float(g["rolling_8w_mean"].min()),
                "rolling_8w_mean_median": float(g["rolling_8w_mean"].median()),
                "rolling_8w_mean_max": float(g["rolling_8w_mean"].max()),
                "positive_rows": int(g[TARGET_COL].sum()),
                "positive_share": float(g[TARGET_COL].mean()),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "pr_auc": metrics["pr_auc"],
                "alert_rate": metrics["alert_rate"],
                "precision_at_1pct": _precision_at_k(g, 0.01),
                "precision_at_5pct": _precision_at_k(g, 0.05),
            }
        )
    return pd.DataFrame(rows)


def build_capacity_table(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"].eq("test")].copy().sort_values(SCORE_COL, ascending=False)
    base = float(test[TARGET_COL].mean())
    rows = []
    for label, share in [("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10)]:
        k = max(1, int(np.ceil(len(test) * share)))
        top = test.head(k)
        precision = float(top[TARGET_COL].mean())
        recall = float(top[TARGET_COL].sum() / test[TARGET_COL].sum())
        rows.append(
            {
                "risk_set": label,
                "rows": k,
                "true_positives": int(top[TARGET_COL].sum()),
                "precision": precision,
                "recall": recall,
                "lift_over_base_rate": precision / base if base else np.nan,
                "base_rate": base,
            }
        )
    alerts = test[test["alert"]]
    rows.append(
        {
            "risk_set": "category_threshold_alerts",
            "rows": len(alerts),
            "true_positives": int(alerts[TARGET_COL].sum()),
            "precision": float(alerts[TARGET_COL].mean()),
            "recall": float(alerts[TARGET_COL].sum() / test[TARGET_COL].sum()),
            "lift_over_base_rate": float(alerts[TARGET_COL].mean() / base),
            "base_rate": base,
        }
    )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype(str)
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in text.to_numpy(dtype=str)]
    return "\n".join([header, sep, *body])


def write_report(
    sparse: pd.DataFrame,
    category: pd.DataFrame,
    deciles: pd.DataFrame,
    capacity: pd.DataFrame,
) -> None:
    lines = [
        "# Major Revision Initial Sparse-Panel Audit",
        "",
        "This audit is a reproducible first-pass diagnostic for reviewer P1-2. It does not resolve the target-shortcut critique by itself; it quantifies how much of the current held-out positive class comes from low historical-volume rows.",
        "",
        "## Key Sparse Diagnostics",
        "",
        _markdown_table(sparse),
        "",
        "## Capacity Ranking",
        "",
        _markdown_table(capacity),
        "",
        "## Performance by Historical-Volume Decile",
        "",
        _markdown_table(deciles),
        "",
        "## Category Sparse Profile",
        "",
        _markdown_table(category),
        "",
        "## Interpretation Guardrails",
        "",
        "- These diagnostics describe the current submitted target and decision layer only.",
        "- Because the current target uses an 8-week rolling baseline, rows with low `rolling_8w_mean` need target-definition sensitivity checks before claims are made.",
        "- Follow-up scripts should test sparse-aware targets and no-shortcut feature sets before the manuscript is rewritten.",
        "",
    ]
    (OUT_DIR / "sparse_panel_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _read_scored_with_features()
    sparse, category = build_sparse_diagnostics(df)
    deciles = build_volume_deciles(df)
    capacity = build_capacity_table(df)

    sparse.to_csv(OUT_DIR / "sparse_panel_diagnostics.csv", index=False)
    category.to_csv(OUT_DIR / "sparse_panel_category_profile.csv", index=False)
    deciles.to_csv(OUT_DIR / "performance_by_volume_decile.csv", index=False)
    capacity.to_csv(OUT_DIR / "capacity_precision_at_k.csv", index=False)

    write_report(sparse, category, deciles, capacity)

    metadata = {
        "final_dataset": str(FINAL_DATASET.relative_to(ROOT)),
        "scored_file": str(SCORED.relative_to(ROOT)),
        "threshold_file": str(THRESHOLDS.relative_to(ROOT)),
        "score_col": SCORE_COL,
        "rows_scored": int(df.shape[0]),
        "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
    }
    (OUT_DIR / "initial_audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
