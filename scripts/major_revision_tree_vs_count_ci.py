from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_count_extensions import (  # noqa: E402
    FINAL_FOLD,
    formula_event_from_count,
    load_final_style,
    target_event_from_name,
    transform_count_features,
)
from major_revision_model_audits import COUNT_TARGET_COL, SPLIT_COL, THRESHOLD_COL, precision_at_k, write_csv, write_json  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/bootstrap"
TREE_PRED_PATH = OUT_DIR / "bootstrap_prediction_rows.csv.gz"
TARGET = "T2_min_count_3"
METRICS = ["pr_auc", "precision_at_5pct", "f1", "precision", "recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap CI for final tree model versus count baselines.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=20260716)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def fit_count_predictions(args: argparse.Namespace) -> pd.DataFrame:
    frame, _features, numeric, categorical = load_final_style(TARGET)
    train = frame[frame[SPLIT_COL].eq("train")].copy()
    val = frame[frame[SPLIT_COL].eq("validation")].copy()
    test = frame[frame[SPLIT_COL].eq("test")].copy()
    (x_train, _x_val, x_test), cat_idx, _ = transform_count_features(train, [train, val, test], numeric, categorical)
    y_count = train[COUNT_TARGET_COL].to_numpy(dtype=float)
    if args.progress:
        print(f"[tree-vs-count] train={len(train)} val={len(val)} test={len(test)} features={x_train.shape[1]}")

    hgb = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        categorical_features=cat_idx if cat_idx else None,
        random_state=args.seed,
    )
    start = time.time()
    hgb.fit(x_train, y_count)
    hgb_fit_seconds = time.time() - start
    hgb_test = np.clip(hgb.predict(x_test), 0, None)

    occ = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        categorical_features=cat_idx if cat_idx else None,
        random_state=args.seed,
    )
    count = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        categorical_features=cat_idx if cat_idx else None,
        random_state=args.seed,
    )
    positive = y_count > 0
    start = time.time()
    occ.fit(x_train, (y_count > 0).astype(int))
    count.fit(x_train[positive], y_count[positive])
    hurdle_fit_seconds = time.time() - start
    hurdle_test = np.clip(occ.predict_proba(x_test)[:, 1] * np.clip(count.predict(x_test), 0, None), 0, None)

    out = pd.DataFrame(
        {
            "target_week": test["target_week"].astype(str).to_numpy(),
            "nta2020": test["nta2020"].astype(str).to_numpy(),
            "complaint_category": test["complaint_category"].astype(str).to_numpy(),
            "target_threshold": test[THRESHOLD_COL].to_numpy(dtype=float),
            "target_next_week_count": test[COUNT_TARGET_COL].to_numpy(dtype=float),
            "y_true_count_check": target_event_from_name(test, TARGET),
            "hgb_poisson_count_score": hgb_test,
            "hgb_poisson_formula_event": formula_event_from_count(test, hgb_test, TARGET).astype(int),
            "hurdle_count_score": hurdle_test,
            "hurdle_formula_event": formula_event_from_count(test, hurdle_test, TARGET).astype(int),
        }
    )
    out.attrs["hgb_fit_seconds"] = hgb_fit_seconds
    out.attrs["hurdle_fit_seconds"] = hurdle_fit_seconds
    return out


def load_tree_predictions() -> pd.DataFrame:
    tree = pd.read_csv(TREE_PRED_PATH)
    tree = tree[(tree["target_definition"].eq(TARGET)) & (tree["fold_id"].eq("final_style_2025"))].copy()
    keep = [
        "target_week",
        "nta2020",
        "complaint_category",
        "y_true",
        "platt_score",
        "platt_threshold",
    ]
    return tree[keep]


def metric_values(y: np.ndarray, score: np.ndarray, pred_event: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred_event = np.asarray(pred_event, dtype=bool)
    return {
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "precision_at_5pct": precision_at_k(y, score, 0.05),
        "f1": float(f1_score(y, pred_event, zero_division=0)),
        "precision": float(precision_score(y, pred_event, zero_division=0)),
        "recall": float(recall_score(y, pred_event, zero_division=0)),
    }


def paired_bootstrap(merged: pd.DataFrame, challenger: str, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    y = merged["y_true"].to_numpy(dtype=int)
    tree_score = merged["platt_score"].to_numpy(dtype=float)
    tree_pred = tree_score >= float(merged["platt_threshold"].iloc[0])
    if challenger == "hgb_poisson_formula":
        count_score = merged["hgb_poisson_count_score"].to_numpy(dtype=float)
        count_pred = merged["hgb_poisson_formula_event"].to_numpy(dtype=bool)
    elif challenger == "hurdle_formula":
        count_score = merged["hurdle_count_score"].to_numpy(dtype=float)
        count_pred = merged["hurdle_formula_event"].to_numpy(dtype=bool)
    else:
        raise ValueError(challenger)

    labels = merged["nta2020"].astype(str) + "|" + merged["complaint_category"].astype(str)
    clusters = np.array(sorted(labels.unique()))
    label_arr = labels.to_numpy()
    index_by_cluster = {c: np.flatnonzero(label_arr == c) for c in clusters}
    tree_obs = metric_values(y, tree_score, tree_pred)
    count_obs = metric_values(y, count_score, count_pred)
    observed = {m: tree_obs[m] - count_obs[m] for m in METRICS}

    reps = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in sampled])
        tv = metric_values(y[idx], tree_score[idx], tree_pred[idx])
        cv = metric_values(y[idx], count_score[idx], count_pred[idx])
        reps.append({m: tv[m] - cv[m] for m in METRICS})
    reps_df = pd.DataFrame(reps)

    rows = []
    for metric in METRICS:
        vals = reps_df[metric].to_numpy(dtype=float)
        rows.append(
            {
                "target_definition": TARGET,
                "challenger": "platt_lightgbm",
                "baseline": challenger,
                "fold_id": "final_style_2025",
                "cluster_unit": "nta_category",
                "clusters": int(len(clusters)),
                "n_bootstrap": int(n_bootstrap),
                "metric": metric,
                "challenger_observed": tree_obs[metric],
                "baseline_observed": count_obs[metric],
                "observed_difference": observed[metric],
                "boot_mean_difference": float(np.nanmean(vals)),
                "ci_lower": float(np.nanpercentile(vals, 2.5)),
                "ci_upper": float(np.nanpercentile(vals, 97.5)),
                "win_proportion": float(np.nanmean(vals > 0)),
                "ci_includes_zero": bool(np.nanpercentile(vals, 2.5) <= 0 <= np.nanpercentile(vals, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def write_report(out_dir: Path, ci: pd.DataFrame, fit_info: dict[str, float]) -> None:
    rows = []
    for _, r in ci.iterrows():
        rows.append(
            {
                "baseline": r["baseline"],
                "metric": r["metric"],
                "tree": f"{r['challenger_observed']:.4f}",
                "count": f"{r['baseline_observed']:.4f}",
                "diff": f"{r['observed_difference']:.4f}",
                "95% CI": f"{r['ci_lower']:.4f} to {r['ci_upper']:.4f}",
                "includes zero": str(bool(r["ci_includes_zero"])),
            }
        )
    report = ["# Tree versus Count Paired Bootstrap Report", ""]
    report.append("This report compares the final-style Platt-calibrated no-shortcut LightGBM against row-level count-baseline scores on the same held-out 2025 rows.")
    report.append("")
    report.append(f"- HGB Poisson fit seconds: {fit_info['hgb_fit_seconds']:.2f}")
    report.append(f"- Hurdle count fit seconds: {fit_info['hurdle_fit_seconds']:.2f}")
    report.append("- Bootstrap unit: NTA-category clusters.")
    report.append("")
    report.extend(["| baseline | metric | tree | count | diff | 95% CI | includes zero |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in rows:
        report.append("| " + " | ".join(row[c] for c in ["baseline", "metric", "tree", "count", "diff", "95% CI", "includes zero"]) + " |")
    report.append("")
    report.append("Guardrail: PR-AUC and precision@5% compare ranking scores. F1/precision/recall compare the validation-selected tree threshold against the count-model formula-threshold event decision.")
    text = "\n".join(report)
    (out_dir / "tree_vs_count_paired_ci_report.md").write_text(text, encoding="utf-8")
    (ROOT / "tree_vs_count_paired_ci_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    count_pred = fit_count_predictions(args)
    fit_info = {
        "hgb_fit_seconds": float(count_pred.attrs.get("hgb_fit_seconds", np.nan)),
        "hurdle_fit_seconds": float(count_pred.attrs.get("hurdle_fit_seconds", np.nan)),
    }
    tree = load_tree_predictions()
    merged = tree.merge(count_pred, on=["target_week", "nta2020", "complaint_category"], how="inner", validate="one_to_one")
    if len(merged) != len(tree) or not np.array_equal(merged["y_true"].to_numpy(dtype=int), merged["y_true_count_check"].to_numpy(dtype=int)):
        raise RuntimeError(f"Prediction merge failed or target mismatch: merged={len(merged)} tree={len(tree)}")
    pred_path = out_dir / "tree_vs_count_prediction_rows.csv.gz"
    merged.to_csv(pred_path, index=False, compression="gzip")

    rng = np.random.default_rng(args.random_state)
    ci = pd.concat(
        [
            paired_bootstrap(merged, "hgb_poisson_formula", rng, args.n_bootstrap),
            paired_bootstrap(merged, "hurdle_formula", rng, args.n_bootstrap),
        ],
        ignore_index=True,
    )
    write_csv(out_dir / "tree_vs_count_paired_ci.csv", ci, args.overwrite)
    write_report(out_dir, ci, fit_info)
    write_json(
        out_dir / "tree_vs_count_paired_ci_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_tree_vs_count_ci.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "target": TARGET,
            "fold": FINAL_FOLD,
            "n_bootstrap": args.n_bootstrap,
            "cluster_unit": "nta_category",
            "prediction_rows": int(len(merged)),
            "hgb_fit_seconds": fit_info["hgb_fit_seconds"],
            "hurdle_fit_seconds": fit_info["hurdle_fit_seconds"],
            "outputs": [
                "tree_vs_count_prediction_rows.csv.gz",
                "tree_vs_count_paired_ci.csv",
                "tree_vs_count_paired_ci_report.md",
                "tree_vs_count_paired_ci_run_summary.json",
            ],
        },
    )
    print(f"Wrote {out_dir / 'tree_vs_count_paired_ci.csv'}")
    print(f"Wrote {out_dir / 'tree_vs_count_paired_ci_report.md'}")


if __name__ == "__main__":
    main()
