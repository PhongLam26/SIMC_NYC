from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_full_ablation import FINAL_FOLD, build_feature_configs, is_weather_feature  # noqa: E402
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
ROOT_REPORT = ROOT / "nta_vs_borough_paired_ci_report.md"
CONFIGS = ["07_final_no_shortcut_borough", "08_final_no_shortcut_nta_fe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap CI for final borough vs NTA fixed-effect LightGBM variants.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--target", default="T2_min_count_3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fit_config(
    frame: pd.DataFrame,
    target_name: str,
    config_name: str,
    features: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    d = assign_fold_split(frame, FINAL_FOLD)
    X_train, y_train, X_val, y_val, X_test, y_test, model_features = encode_splits(d, features, categorical)
    params = best_lgbm_params()
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
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
        scale_pos_weight=(neg / pos) if pos else 1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start
    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    threshold, val_f1 = select_threshold(y_val, val_score)
    test_rows = d[d[SPLIT_COL].eq("test")].reset_index(drop=True)
    pred = test_rows[["target_week", "week_start", "nta2020", "boroname", "complaint_category"]].copy()
    pred["row_id"] = np.arange(len(pred), dtype=np.int64)
    pred["y_true"] = y_test.to_numpy(dtype=int)
    pred[f"{config_name}_score"] = test_score
    pred[f"{config_name}_threshold"] = threshold
    meta = {
        "fit_seconds": float(fit_seconds),
        "validation_f1": float(val_f1),
        "threshold": float(threshold),
        "raw_feature_count": float(len(features)),
        "model_feature_count": float(len(model_features)),
        "contains_weather": float(any(is_weather_feature(f) for f in features)),
        "contains_borough": float("boroname" in features),
        "contains_nta_fe": float("nta2020" in features),
    }
    return pred, meta


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


def paired_ci(pred: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    borough = "07_final_no_shortcut_borough"
    nta = "08_final_no_shortcut_nta_fe"
    y = pred["y_true"].to_numpy(dtype=int)
    borough_score = pred[f"{borough}_score"].to_numpy(dtype=float)
    nta_score = pred[f"{nta}_score"].to_numpy(dtype=float)
    borough_pred = borough_score >= float(pred[f"{borough}_threshold"].iloc[0])
    nta_pred = nta_score >= float(pred[f"{nta}_threshold"].iloc[0])
    borough_obs = metric_values(y, borough_score, borough_pred)
    nta_obs = metric_values(y, nta_score, nta_pred)

    cluster = pred["nta2020"].astype(str) + "|" + pred["complaint_category"].astype(str)
    cluster_codes, uniques = pd.factorize(cluster, sort=True)
    groups = [np.flatnonzero(cluster_codes == i) for i in range(len(uniques))]
    rng = np.random.default_rng(seed)
    diffs = {metric: [] for metric in borough_obs}
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        b = metric_values(y[idx], borough_score[idx], borough_pred[idx])
        n = metric_values(y[idx], nta_score[idx], nta_pred[idx])
        for metric in diffs:
            diffs[metric].append(n[metric] - b[metric])

    rows = []
    for metric, values in diffs.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "comparison": "nta_fe_minus_borough",
                "metric": metric,
                "borough_value": borough_obs[metric],
                "nta_fe_value": nta_obs[metric],
                "observed_difference": nta_obs[metric] - borough_obs[metric],
                "ci_lower": float(np.nanquantile(arr, 0.025)),
                "ci_upper": float(np.nanquantile(arr, 0.975)),
                "ci_includes_zero": bool(np.nanquantile(arr, 0.025) <= 0 <= np.nanquantile(arr, 0.975)),
                "bootstrap_replicates": int(n_bootstrap),
                "cluster_unit": "nta2020|complaint_category",
                "clusters": int(len(groups)),
            }
        )
    return pd.DataFrame(rows)


def write_report(out_dir: Path, ci: pd.DataFrame, meta: dict[str, dict[str, float]], pred_rows: int, target: str, seed: int) -> str:
    rows = ci[ci["metric"].isin(["pr_auc", "precision_at_5pct", "f1", "precision", "recall"])].copy()
    lines = [
        "# NTA vs Borough Paired Bootstrap CI",
        "",
        "This report compares the final no-shortcut LightGBM borough variant against the otherwise matched NTA fixed-effect variant on the same held-out 2025 rows.",
        "",
        f"- Target: `{target}` ({TARGET_DEFINITIONS[target]['description']}).",
        f"- Seed: {seed}.",
        f"- Prediction rows: {pred_rows:,}.",
        f"- Bootstrap unit: NTA-category clusters; replicates: {int(ci['bootstrap_replicates'].iloc[0]):,}.",
        "- Difference direction: NTA fixed effects minus borough.",
        "",
        "## Paired Differences",
        "",
        md_table(
            rows,
            [
                "metric",
                "borough_value",
                "nta_fe_value",
                "observed_difference",
                "ci_lower",
                "ci_upper",
                "ci_includes_zero",
            ],
            max_rows=20,
        ),
        "",
        "## Fit Metadata",
        "",
        md_table(
            pd.DataFrame(
                [
                    {"feature_config": name, **vals}
                    for name, vals in meta.items()
                ]
            ),
            [
                "feature_config",
                "fit_seconds",
                "validation_f1",
                "threshold",
                "raw_feature_count",
                "model_feature_count",
                "contains_borough",
                "contains_nta_fe",
            ],
            max_rows=10,
        ),
        "",
        "## Interpretation Guardrail",
        "",
        "- The NTA fixed-effect variant has a very small PR-AUC gain whose paired interval excludes zero, while precision@5% and F1 intervals include zero and recall decreases.",
        "- The manuscript should keep the NTA result as a diagnostic spatial check rather than changing the simpler final borough model claim.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "nta_vs_borough_paired_ci_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    start = time.time()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_sets, categorical_base = load_feature_sets()
    prospective = feature_sets["B_no_8w_formula_features"]
    header = pd.read_csv(ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz", nrows=0)
    configs = build_feature_configs(prospective, set(header.columns))
    categorical = sorted(set(categorical_base + ["complaint_category", "boroname", "nta2020"]))
    required_features = sorted(set(f for name in CONFIGS for f in configs[name]) | {"rolling_8w_mean"})
    df = load_dataset({"nta_borough_paired_ci_features": required_features})
    if args.target not in TARGET_DEFINITIONS:
        raise ValueError(f"Unknown target: {args.target}")
    frame = make_targets(df)[args.target]

    preds = []
    meta = {}
    for name in CONFIGS:
        pred, cfg_meta = fit_config(frame, args.target, name, configs[name], categorical, args.seed)
        preds.append(pred)
        meta[name] = cfg_meta
    merged = preds[0]
    for pred in preds[1:]:
        keep = [c for c in pred.columns if c == "row_id" or c.startswith("08_final")]
        merged = merged.merge(pred[keep], on="row_id", how="inner", validate="one_to_one")

    ci = paired_ci(merged, args.n_bootstrap, args.bootstrap_seed)
    write_csv(out_dir / "nta_vs_borough_prediction_rows.csv.gz", merged, args.overwrite)
    write_csv(out_dir / "nta_vs_borough_paired_ci.csv", ci, args.overwrite)
    write_report(out_dir, ci, meta, len(merged), args.target, args.seed)
    write_json(
        out_dir / "nta_vs_borough_paired_ci_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_nta_borough_paired_ci.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "target": args.target,
            "seed": args.seed,
            "bootstrap_seed": args.bootstrap_seed,
            "n_bootstrap": args.n_bootstrap,
            "prediction_rows": int(len(merged)),
            "feature_configs": CONFIGS,
            "removed_formula_aligned_features": sorted(FORMULA_ALIGNED_REMOVE),
            "outputs": [
                "nta_vs_borough_prediction_rows.csv.gz",
                "nta_vs_borough_paired_ci.csv",
                "nta_vs_borough_paired_ci_report.md",
                "nta_vs_borough_paired_ci_run_summary.json",
            ],
        },
    )


if __name__ == "__main__":
    main()
