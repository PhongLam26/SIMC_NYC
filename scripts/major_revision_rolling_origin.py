from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_model_audits import (  # noqa: E402
    FORMULA_ALIGNED_REMOVE,
    SPLIT_COL,
    best_lgbm_params,
    load_dataset,
    load_feature_sets,
    select_threshold,
    write_csv,
    write_json,
)
from major_revision_target_selection import (  # noqa: E402
    TARGET_DEFINITIONS,
    encode_splits,
    make_targets,
    md_table,
    score_metrics,
)


OUT_DIR = ROOT / "data/processed/model_results/major_revision/backtests"
ROOT_REPORT = ROOT / "rolling_origin_report.md"


FOLDS = [
    {"fold_id": "fold_2021", "train_end_year": 2019, "validation_year": 2020, "test_year": 2021},
    {"fold_id": "fold_2022", "train_end_year": 2020, "validation_year": 2021, "test_year": 2022},
    {"fold_id": "fold_2023", "train_end_year": 2021, "validation_year": 2022, "test_year": 2023},
    {"fold_id": "fold_2024", "train_end_year": 2022, "validation_year": 2023, "test_year": 2024},
    {"fold_id": "final_style_2025", "train_end_year": 2023, "validation_year": 2024, "test_year": 2025},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanding-window rolling-origin backtests for the major revision.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--targets",
        default="T0_current_reference,T1_min_count_2,T2_min_count_3,T3_mu8w_ge_1_eligible",
        help="Comma-separated target definitions to evaluate.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def assign_fold_split(frame: pd.DataFrame, fold: dict[str, int | str]) -> pd.DataFrame:
    d = frame.copy()
    year = d["target_year"].astype(int)
    split = np.full(len(d), "unused", dtype=object)
    split[year <= int(fold["train_end_year"])] = "train"
    split[year == int(fold["validation_year"])] = "validation"
    split[year == int(fold["test_year"])] = "test"
    d[SPLIT_COL] = split
    return d[d[SPLIT_COL].isin(["train", "validation", "test"])].copy()


def train_fold(
    target_name: str,
    frame: pd.DataFrame,
    fold: dict[str, int | str],
    features: list[str],
    categorical: list[str],
    random_state: int,
    progress: bool,
) -> list[dict[str, object]]:
    d = assign_fold_split(frame, fold)
    split_counts = d[SPLIT_COL].value_counts().to_dict()
    if progress:
        print(f"[rolling] {target_name} {fold['fold_id']}: {split_counts}")
    required = {"train", "validation", "test"}
    if not required.issubset(split_counts):
        raise ValueError(f"Missing split rows for {target_name} {fold['fold_id']}: {split_counts}")

    X_train, y_train, X_val, y_val, X_test, y_test, model_features = encode_splits(d, features, categorical)
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    scale_pos_weight = neg / pos if pos else 1.0
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
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    threshold, val_f1_at_threshold = select_threshold(y_val, val_scores)

    rows = []
    for split_name, y, scores in [("validation", y_val, val_scores), ("test", y_test, test_scores)]:
        rec = score_metrics(y, scores, threshold)
        rec.update(
            {
                "fold_id": fold["fold_id"],
                "target_definition": target_name,
                "description": TARGET_DEFINITIONS[target_name]["description"],
                "split": split_name,
                "train_end_year": fold["train_end_year"],
                "validation_year": fold["validation_year"],
                "test_year": fold["test_year"],
                "model_name": "lightgbm_no_8w_formula_features",
                "raw_feature_count": len(features),
                "model_feature_count": len(model_features),
                "validation_selected_f1": val_f1_at_threshold,
                "fit_seconds": fit_seconds,
                "removed_formula_aligned_features": ", ".join(FORMULA_ALIGNED_REMOVE),
            }
        )
        rows.append(rec)
    return rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "positive_share",
        "pr_auc",
        "precision_at_1pct",
        "precision_at_5pct",
        "precision_at_10pct",
        "f1",
        "precision",
        "recall",
        "alert_rate",
        "threshold",
    ]
    test = metrics[metrics["split"].eq("test")].copy()
    rows = []
    for target, g in test.groupby("target_definition", sort=True):
        row = {"target_definition": target, "folds": int(g["fold_id"].nunique())}
        for field in fields:
            row[f"{field}_mean"] = float(g[field].mean())
            row[f"{field}_std"] = float(g[field].std(ddof=1)) if len(g) > 1 else 0.0
            row[f"{field}_min"] = float(g[field].min())
            row[f"{field}_max"] = float(g[field].max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, metrics: pd.DataFrame, summary: pd.DataFrame) -> str:
    test = metrics[metrics["split"].eq("test")].copy()
    val = metrics[metrics["split"].eq("validation")].copy()
    per_year = test.sort_values(["target_definition", "test_year"])
    lines = [
        "# Rolling-Origin Backtest Report",
        "",
        "This report runs expanding-window backtests using the existing data only. All rows use the no-shortcut LightGBM feature set, with formula-aligned 8-week predictors removed.",
        "",
        "Thresholds are selected on the validation year within each fold and applied unchanged to that fold's test year. No test year is used to select thresholds, target definitions, or feature sets in this script.",
        "",
        "## Fold Design",
        "",
        md_table(pd.DataFrame(FOLDS), ["fold_id", "train_end_year", "validation_year", "test_year"]),
        "",
        "## Test-Year Summary Across Folds",
        "",
        md_table(
            summary,
            [
                "target_definition",
                "folds",
                "pr_auc_mean",
                "pr_auc_std",
                "precision_at_5pct_mean",
                "precision_at_5pct_std",
                "f1_mean",
                "f1_std",
                "positive_share_mean",
                "positive_share_std",
                "threshold_mean",
                "threshold_std",
            ],
        ),
        "",
        "## Per-Test-Year Metrics",
        "",
        md_table(
            per_year,
            [
                "target_definition",
                "fold_id",
                "test_year",
                "rows",
                "positive_share",
                "pr_auc",
                "precision_at_1pct",
                "precision_at_5pct",
                "f1",
                "precision",
                "recall",
                "threshold",
                "alert_rate",
            ],
            max_rows=120,
        ),
        "",
        "## Validation-Year Metrics",
        "",
        md_table(
            val.sort_values(["target_definition", "validation_year"]),
            [
                "target_definition",
                "fold_id",
                "validation_year",
                "rows",
                "positive_share",
                "pr_auc",
                "precision_at_5pct",
                "f1",
                "threshold",
                "alert_rate",
            ],
            max_rows=120,
        ),
        "",
        "## Interpretation Guardrails",
        "",
        "- T3 is a restricted-risk-set target and is not directly comparable to T0/T1/T2 because it excludes low-baseline rows.",
        "- This pass does not include multiple seeds, calibration, bootstrap intervals, or COVID-exclusion sensitivity.",
        "- Final model and target selection remain open until paired uncertainty and calibration evidence are added.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "rolling_origin_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    requested_targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = sorted(set(requested_targets) - set(TARGET_DEFINITIONS))
    if unknown:
        raise ValueError(f"Unknown target definitions: {unknown}")

    feature_sets, categorical = load_feature_sets()
    features = feature_sets["B_no_8w_formula_features"]
    required_features = sorted(set(features + ["rolling_8w_mean"]))
    df = load_dataset({"B_no_8w_formula_features": required_features})
    candidates = make_targets(df)

    metric_rows = []
    for target_name in requested_targets:
        frame = candidates[target_name]
        for fold in FOLDS:
            metric_rows.extend(
                train_fold(
                    target_name=target_name,
                    frame=frame,
                    fold=fold,
                    features=features,
                    categorical=categorical,
                    random_state=args.random_state,
                    progress=args.progress,
                )
            )

    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics)
    write_csv(out_dir / "rolling_origin_results.csv", metrics, args.overwrite)
    write_csv(out_dir / "rolling_origin_summary.csv", summary, args.overwrite)
    write_report(out_dir, metrics, summary)
    write_json(
        out_dir / "rolling_origin_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_rolling_origin.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "targets": requested_targets,
            "folds": FOLDS,
            "feature_set": "B_no_8w_formula_features",
            "removed_formula_aligned_features": FORMULA_ALIGNED_REMOVE,
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
