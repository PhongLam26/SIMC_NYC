from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_calibration import apply_platt, fit_platt  # noqa: E402
from major_revision_model_audits import (  # noqa: E402
    FORMULA_ALIGNED_REMOVE,
    SPLIT_COL,
    best_lgbm_params,
    load_dataset,
    load_feature_sets,
    precision_at_k,
    select_threshold,
    write_csv,
    write_json,
)
from major_revision_rolling_origin import assign_fold_split  # noqa: E402
from major_revision_target_selection import TARGET_DEFINITIONS, encode_splits, make_targets, md_table  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/seeds"
ROOT_REPORT = ROOT / "seed_stability_report.md"
SEEDS = [42, 123, 2026, 3407, 7777]
FINAL_FOLD = {"fold_id": "final_style_2025", "train_end_year": 2023, "validation_year": 2024, "test_year": 2025}
DEFAULT_TARGETS = "T0_current_reference,T2_min_count_3"
METRIC_FIELDS = [
    "pr_auc",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "precision_at_1pct",
    "precision_at_5pct",
    "precision_at_10pct",
    "brier",
    "log_loss",
    "threshold",
    "alert_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five-seed stability checks for final-style major-revision candidates.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    pred = score >= threshold
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_share": float(y.mean()),
        "threshold": float(threshold),
        "predicted_positive_rows": int(pred.sum()),
        "alert_rate": float(pred.mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "precision_at_1pct": precision_at_k(y, score, 0.01),
        "precision_at_5pct": precision_at_k(y, score, 0.05),
        "precision_at_10pct": precision_at_k(y, score, 0.10),
        "brier": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
    }


def run_one_seed(
    target_name: str,
    frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    seed: int,
    progress: bool,
) -> list[dict[str, object]]:
    d = assign_fold_split(frame, FINAL_FOLD)
    if progress:
        print(f"[seed] {target_name} seed={seed}: {d[SPLIT_COL].value_counts().to_dict()}")
    X_train, y_train, X_val, y_val, X_test, y_test, model_features = encode_splits(d, features, categorical)
    params = best_lgbm_params()
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    scale_pos_weight = neg / pos if pos else 1.0
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
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start
    val_raw = model.predict_proba(X_val)[:, 1]
    test_raw = model.predict_proba(X_test)[:, 1]
    raw_threshold, _ = select_threshold(y_val, val_raw)
    platt = fit_platt(val_raw, y_val.to_numpy(dtype=int))
    val_platt = apply_platt(platt, val_raw)
    test_platt = apply_platt(platt, test_raw)
    platt_threshold = float(apply_platt(platt, np.array([raw_threshold]))[0])

    rows = []
    for calibration_method, split, y, score, threshold in [
        ("uncalibrated", "validation", y_val.to_numpy(dtype=int), val_raw, raw_threshold),
        ("uncalibrated", "test", y_test.to_numpy(dtype=int), test_raw, raw_threshold),
        ("platt", "validation", y_val.to_numpy(dtype=int), val_platt, platt_threshold),
        ("platt", "test", y_test.to_numpy(dtype=int), test_platt, platt_threshold),
    ]:
        rec = metrics(y, score, threshold)
        rec.update(
            {
                "target_definition": target_name,
                "description": TARGET_DEFINITIONS[target_name]["description"],
                "fold_id": FINAL_FOLD["fold_id"],
                "train_end_year": FINAL_FOLD["train_end_year"],
                "validation_year": FINAL_FOLD["validation_year"],
                "test_year": FINAL_FOLD["test_year"],
                "seed": seed,
                "split": split,
                "calibration_method": calibration_method,
                "model_name": "lightgbm_no_8w_formula_features",
                "raw_feature_count": len(features),
                "model_feature_count": len(model_features),
                "fit_seconds": fit_seconds,
                "removed_formula_aligned_features": ", ".join(FORMULA_ALIGNED_REMOVE),
            }
        )
        rows.append(rec)
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = results[results["split"].eq("test")].copy()
    for (target, method), g in test.groupby(["target_definition", "calibration_method"], sort=True):
        row = {"target_definition": target, "calibration_method": method, "seeds": int(g["seed"].nunique())}
        for metric in METRIC_FIELDS:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
            row[f"{metric}_min"] = float(g[metric].min())
            row[f"{metric}_max"] = float(g[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, results: pd.DataFrame, summary: pd.DataFrame) -> str:
    test = results[results["split"].eq("test")].copy()
    platt_test = test[test["calibration_method"].eq("platt")].sort_values(["target_definition", "seed"])
    lines = [
        "# Seed Stability Report",
        "",
        "This report evaluates five stochastic seeds for final-style 2025 no-shortcut LightGBM candidates. Thresholds and Platt calibrators are fitted only on validation 2024 and evaluated on test 2025.",
        "",
        "Seeds: 42, 123, 2026, 3407, 7777.",
        "",
        "## Test Summary Across Seeds",
        "",
        md_table(
            summary.sort_values(["target_definition", "calibration_method"]),
            [
                "target_definition",
                "calibration_method",
                "seeds",
                "pr_auc_mean",
                "pr_auc_std",
                "precision_at_5pct_mean",
                "precision_at_5pct_std",
                "f1_mean",
                "f1_std",
                "brier_mean",
                "brier_std",
            ],
            max_rows=80,
        ),
        "",
        "## Per-Seed Platt-Calibrated Test Metrics",
        "",
        md_table(
            platt_test,
            [
                "target_definition",
                "seed",
                "pr_auc",
                "precision_at_1pct",
                "precision_at_5pct",
                "f1",
                "precision",
                "recall",
                "brier",
                "threshold",
                "alert_rate",
            ],
            max_rows=80,
        ),
        "",
        "## Guardrails",
        "",
        "- This is a final-style 2025 seed-stability pass, not a full rolling-origin multi-seed run for every fold.",
        "- T0 and T2 use different target labels, so small metric differences should not be interpreted as a paired model comparison.",
        "- The manuscript should report rounded metrics and uncertainty, not overclaim 0.001-scale differences.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "seed_stability_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = sorted(set(targets) - set(TARGET_DEFINITIONS))
    if unknown:
        raise ValueError(f"Unknown target definitions: {unknown}")

    feature_sets, categorical = load_feature_sets()
    features = feature_sets["B_no_8w_formula_features"]
    required_features = sorted(set(features + ["rolling_8w_mean"]))
    df = load_dataset({"B_no_8w_formula_features": required_features})
    candidates = make_targets(df)

    rows = []
    for target in targets:
        for seed in seeds:
            rows.extend(run_one_seed(target, candidates[target], features, categorical, seed, args.progress))

    results = pd.DataFrame(rows)
    summary = summarize(results)
    write_csv(out_dir / "seed_stability_results.csv", results, args.overwrite)
    write_csv(out_dir / "seed_stability_summary.csv", summary, args.overwrite)
    write_report(out_dir, results, summary)
    write_json(
        out_dir / "seed_stability_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_seed_stability.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "fold": FINAL_FOLD,
            "seeds": seeds,
            "targets": targets,
            "feature_set": "B_no_8w_formula_features",
            "calibration_methods": ["uncalibrated", "platt"],
            "removed_formula_aligned_features": FORMULA_ALIGNED_REMOVE,
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
