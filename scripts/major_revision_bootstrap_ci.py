from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_calibration import apply_platt, clip_prob, fit_isotonic, fit_platt  # noqa: E402
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


OUT_DIR = ROOT / "data/processed/model_results/major_revision/bootstrap"
ROOT_REPORT = ROOT / "bootstrap_report.md"

FINAL_FOLD = {"fold_id": "final_style_2025", "train_end_year": 2023, "validation_year": 2024, "test_year": 2025}
DEFAULT_TARGETS = "T0_current_reference,T2_min_count_3"
METRICS = ["pr_auc", "f1", "precision", "recall", "precision_at_1pct", "precision_at_5pct", "lift_at_1pct", "brier", "log_loss"]
LOWER_IS_BETTER = {"brier", "log_loss"}


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(base_seed + int(digest[:8], 16) % 1_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cluster bootstrap confidence intervals for final-style 2025 predictions.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--cluster-unit", choices=["nta_category", "nta", "target_week"], default="nta_category")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def fit_predictions_for_target(
    target_name: str,
    frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    random_state: int,
    progress: bool,
) -> pd.DataFrame:
    d = assign_fold_split(frame, FINAL_FOLD)
    if progress:
        print(f"[bootstrap] train scores for {target_name}: {d[SPLIT_COL].value_counts().to_dict()}")
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
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    val_raw = model.predict_proba(X_val)[:, 1]
    test_raw = model.predict_proba(X_test)[:, 1]
    raw_threshold, _ = select_threshold(y_val, val_raw)
    platt = fit_platt(val_raw, y_val.to_numpy(dtype=int))
    iso = fit_isotonic(val_raw, y_val.to_numpy(dtype=int))
    platt_test = apply_platt(platt, test_raw)
    iso_test = iso.predict(clip_prob(test_raw))
    test_rows = d[d[SPLIT_COL].eq("test")].copy()
    out = pd.DataFrame(
        {
            "target_definition": target_name,
            "fold_id": FINAL_FOLD["fold_id"],
            "target_year": test_rows["target_year"].astype(int).to_numpy(),
            "target_week": test_rows["target_week"].astype(str).to_numpy(),
            "nta2020": test_rows["nta2020"].astype(str).to_numpy(),
            "complaint_category": test_rows["complaint_category"].astype(str).to_numpy(),
            "y_true": y_test.to_numpy(dtype=int),
            "uncalibrated_score": test_raw,
            "platt_score": platt_test,
            "isotonic_score": iso_test,
            "uncalibrated_threshold": raw_threshold,
            "platt_threshold": float(apply_platt(platt, np.array([raw_threshold]))[0]),
            "isotonic_threshold": float(iso.predict(np.array([raw_threshold]))[0]),
            "model_feature_count": len(model_features),
        }
    )
    return out


def metric_values(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = clip_prob(score)
    pred = score >= threshold
    base = float(y.mean()) if len(y) else np.nan
    p1 = precision_at_k(y, score, 0.01)
    return {
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision_at_1pct": p1,
        "precision_at_5pct": precision_at_k(y, score, 0.05),
        "lift_at_1pct": float(p1 / base) if base else np.nan,
        "brier": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
    }


def cluster_labels(df: pd.DataFrame, unit: str) -> pd.Series:
    if unit == "nta_category":
        return df["nta2020"].astype(str) + "|" + df["complaint_category"].astype(str)
    if unit == "nta":
        return df["nta2020"].astype(str)
    return df["target_week"].astype(str)


def bootstrap_one_prediction_set(
    pred: pd.DataFrame,
    method: str,
    cluster_unit: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], pd.DataFrame]:
    score_col = f"{method}_score"
    threshold = float(pred[f"{method}_threshold"].iloc[0])
    y = pred["y_true"].to_numpy(dtype=int)
    score = pred[score_col].to_numpy(dtype=float)
    observed = metric_values(y, score, threshold)

    labels = cluster_labels(pred, cluster_unit)
    clusters = np.array(sorted(labels.unique()))
    index_by_cluster = {c: np.flatnonzero(labels.to_numpy() == c) for c in clusters}
    rows = []
    for b in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in sampled_clusters])
        vals = metric_values(y[idx], score[idx], threshold)
        vals["bootstrap_id"] = b
        rows.append(vals)
    reps = pd.DataFrame(rows)
    records = []
    for metric in METRICS:
        values = reps[metric].to_numpy(dtype=float)
        records.append(
            {
                "target_definition": str(pred["target_definition"].iloc[0]),
                "calibration_method": method,
                "fold_id": str(pred["fold_id"].iloc[0]),
                "cluster_unit": cluster_unit,
                "clusters": int(len(clusters)),
                "n_bootstrap": int(n_bootstrap),
                "metric": metric,
                "observed": observed[metric],
                "boot_mean": float(np.nanmean(values)),
                "ci_lower": float(np.nanpercentile(values, 2.5)),
                "ci_upper": float(np.nanpercentile(values, 97.5)),
                "boot_std": float(np.nanstd(values, ddof=1)),
            }
        )
    return observed, pd.DataFrame(records)


def paired_differences(
    pred: pd.DataFrame,
    challenger: str,
    baseline: str,
    cluster_unit: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    y = pred["y_true"].to_numpy(dtype=int)
    labels = cluster_labels(pred, cluster_unit)
    clusters = np.array(sorted(labels.unique()))
    index_by_cluster = {c: np.flatnonzero(labels.to_numpy() == c) for c in clusters}
    base_score = pred[f"{baseline}_score"].to_numpy(dtype=float)
    base_threshold = float(pred[f"{baseline}_threshold"].iloc[0])
    chal_score = pred[f"{challenger}_score"].to_numpy(dtype=float)
    chal_threshold = float(pred[f"{challenger}_threshold"].iloc[0])
    base_obs = metric_values(y, base_score, base_threshold)
    chal_obs = metric_values(y, chal_score, chal_threshold)

    diff_rows = []
    for b in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in sampled_clusters])
        base_vals = metric_values(y[idx], base_score[idx], base_threshold)
        chal_vals = metric_values(y[idx], chal_score[idx], chal_threshold)
        row = {"bootstrap_id": b}
        for metric in METRICS:
            row[metric] = chal_vals[metric] - base_vals[metric]
        diff_rows.append(row)
    reps = pd.DataFrame(diff_rows)
    records = []
    for metric in METRICS:
        values = reps[metric].to_numpy(dtype=float)
        observed_diff = chal_obs[metric] - base_obs[metric]
        if metric in LOWER_IS_BETTER:
            win_proportion = float(np.nanmean(values < 0))
            favorable_direction = "negative"
        else:
            win_proportion = float(np.nanmean(values > 0))
            favorable_direction = "positive"
        records.append(
            {
                "target_definition": str(pred["target_definition"].iloc[0]),
                "challenger": challenger,
                "baseline": baseline,
                "fold_id": str(pred["fold_id"].iloc[0]),
                "cluster_unit": cluster_unit,
                "clusters": int(len(clusters)),
                "n_bootstrap": int(n_bootstrap),
                "metric": metric,
                "observed_difference": observed_diff,
                "boot_mean_difference": float(np.nanmean(values)),
                "ci_lower": float(np.nanpercentile(values, 2.5)),
                "ci_upper": float(np.nanpercentile(values, 97.5)),
                "favorable_direction": favorable_direction,
                "win_proportion": win_proportion,
                "ci_includes_zero": bool(np.nanpercentile(values, 2.5) <= 0 <= np.nanpercentile(values, 97.5)),
            }
        )
    return pd.DataFrame(records)


def md_metric_table(df: pd.DataFrame, metrics: list[str]) -> str:
    sub = df[df["metric"].isin(metrics)].copy()
    sub["ci"] = sub.apply(lambda r: f"{r['observed']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]", axis=1)
    return md_table(sub, ["target_definition", "calibration_method", "metric", "ci", "clusters", "n_bootstrap"], max_rows=120)


def write_report(out_dir: Path, ci: pd.DataFrame, diff: pd.DataFrame, args: argparse.Namespace) -> str:
    key_ci = ci[
        ci["calibration_method"].eq("platt")
        & ci["target_definition"].isin(["T0_current_reference", "T2_min_count_3"])
    ].copy()
    key_diff = diff[
        diff["challenger"].eq("platt")
        & diff["baseline"].eq("uncalibrated")
        & diff["metric"].isin(["brier", "log_loss", "pr_auc", "precision_at_5pct", "f1"])
    ].copy()
    key_diff["difference_ci"] = key_diff.apply(
        lambda r: f"{r['observed_difference']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]", axis=1
    )
    lines = [
        "# Bootstrap Confidence Interval Report",
        "",
        "This report provides a first cluster-bootstrap uncertainty pass for final-style 2025 predictions. It uses existing data only and samples clusters with replacement rather than individual rows.",
        "",
        f"- Fold: `{FINAL_FOLD['fold_id']}` with train through {FINAL_FOLD['train_end_year']}, validation {FINAL_FOLD['validation_year']}, and test {FINAL_FOLD['test_year']}.",
        f"- Cluster unit: `{args.cluster_unit}`.",
        f"- Bootstrap replicates: {args.n_bootstrap}.",
        "- Targets evaluated: T0 current reference and T2 minimum-count target.",
        "- Models: no-shortcut LightGBM with uncalibrated, Platt, and isotonic scores.",
        "",
        "## Main 95% CIs for Platt-Calibrated Scores",
        "",
        md_metric_table(key_ci, ["pr_auc", "f1", "precision", "recall", "precision_at_1pct", "precision_at_5pct", "lift_at_1pct", "brier", "log_loss"]),
        "",
        "## Paired Calibration Differences",
        "",
        "Differences are challenger minus baseline within the same target and cluster sample. Negative Brier/log-loss differences favor calibration; positive PR-AUC, precision@k, and F1 differences favor calibration.",
        "",
        md_table(
            key_diff,
            ["target_definition", "challenger", "baseline", "metric", "difference_ci", "favorable_direction", "win_proportion", "ci_includes_zero"],
            max_rows=80,
        ),
        "",
        "## Guardrails",
        "",
        "- T0 and T2 have different labels, so this report does not present a paired T2-minus-T0 difference as a model superiority claim.",
        "- This is a final-style 2025 uncertainty pass, not yet the full Table 4/Table 5 CI package for every final manuscript row.",
        "- Multiple-seed uncertainty remains open.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "bootstrap_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = sorted(set(targets) - set(TARGET_DEFINITIONS))
    if unknown:
        raise ValueError(f"Unknown target definitions: {unknown}")

    feature_sets, categorical = load_feature_sets()
    features = feature_sets["B_no_8w_formula_features"]
    required_features = sorted(set(features + ["rolling_8w_mean"]))
    df = load_dataset({"B_no_8w_formula_features": required_features})
    candidates = make_targets(df)

    predictions = []
    for target in targets:
        predictions.append(
            fit_predictions_for_target(
                target_name=target,
                frame=candidates[target],
                features=features,
                categorical=categorical,
                random_state=args.random_state,
                progress=args.progress,
            )
        )
    pred = pd.concat(predictions, ignore_index=True)
    write_csv(out_dir / "bootstrap_prediction_rows.csv.gz", pred, args.overwrite)

    ci_tables = []
    diff_tables = []
    for target, g in pred.groupby("target_definition", sort=True):
        for method in ["uncalibrated", "platt", "isotonic"]:
            if args.progress:
                print(f"[bootstrap] CI {target} {method}")
            rng = np.random.default_rng(stable_seed(args.bootstrap_seed, target, method))
            _, ci = bootstrap_one_prediction_set(g, method, args.cluster_unit, args.n_bootstrap, rng)
            ci_tables.append(ci)
        for challenger in ["platt", "isotonic"]:
            if args.progress:
                print(f"[bootstrap] paired {target} {challenger} vs uncalibrated")
            rng = np.random.default_rng(stable_seed(args.bootstrap_seed, target, challenger, "paired"))
            diff_tables.append(paired_differences(g, challenger, "uncalibrated", args.cluster_unit, args.n_bootstrap, rng))

    ci_df = pd.concat(ci_tables, ignore_index=True)
    diff_df = pd.concat(diff_tables, ignore_index=True)
    write_csv(out_dir / "bootstrap_ci_results.csv", ci_df, args.overwrite)
    write_csv(out_dir / "paired_model_difference_ci.csv", diff_df, args.overwrite)
    write_report(out_dir, ci_df, diff_df, args)
    write_json(
        out_dir / "bootstrap_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_bootstrap_ci.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "fold": FINAL_FOLD,
            "targets": targets,
            "n_bootstrap": args.n_bootstrap,
            "cluster_unit": args.cluster_unit,
            "bootstrap_seed": args.bootstrap_seed,
            "feature_set": "B_no_8w_formula_features",
            "removed_formula_aligned_features": FORMULA_ALIGNED_REMOVE,
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
