from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


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
from major_revision_rolling_origin import assign_fold_split  # noqa: E402
from major_revision_target_selection import TARGET_DEFINITIONS, encode_splits, make_targets, md_table  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/explainability"
ROOT_REPORT = ROOT / "final_model_explainability_report.md"
FINAL_FOLD = {"fold_id": "final_style_2025", "train_end_year": 2023, "validation_year": 2024, "test_year": 2025}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final-style SHAP explainability for the major revision candidate.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--target", default="T2_min_count_3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shap-rows", type=int, default=8000)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--local-top-n", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def positive_class_values(raw) -> np.ndarray:
    if isinstance(raw, list):
        return np.asarray(raw[1] if len(raw) > 1 else raw[0])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def positive_expected_value(explainer) -> float:
    val = explainer.expected_value
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val)
        return float(arr[1] if arr.size > 1 else arr[0])
    return float(val)


def feature_group(feature: str) -> str:
    if feature.startswith("complaint_category"):
        return "service_category"
    if feature.startswith("boroname"):
        return "borough"
    if feature.startswith("nta2020"):
        return "nta_fixed_effect"
    if feature.startswith("weather_") or feature in {
        "hot_day_30c_count",
        "hot_day_35c_count",
        "cold_day_0c_count",
        "ice_day_tmax_0c_count",
        "very_cold_day_minus5c_count",
        "rain_day_count",
        "moderate_rain_day_10mm_count",
        "heavy_rain_day_25mm_count",
        "very_heavy_rain_day_50mm_count",
        "snow_day_count",
        "heavy_snow_day_25mm_count",
    }:
        return "feature_week_weather"
    if feature in {"year", "month", "quarter", "week_of_year", "iso_year", "is_year_start", "is_year_end"}:
        return "calendar"
    if feature == "complaint_count":
        return "current_count"
    if feature.startswith("lag_") or feature.startswith("rolling_") or feature.startswith("diff_") or feature.startswith("pct_change") or feature.startswith("ratio_to_") or feature == "history_weeks_available":
        return "shifted_history"
    return "other"


def train_model(args: argparse.Namespace):
    feature_sets, categorical = load_feature_sets()
    features = feature_sets["B_no_8w_formula_features"]
    required = sorted(set(features + ["rolling_8w_mean", "rolling_8w_std", "target_next_week_count"]))
    df = load_dataset({"features": required})
    frame = make_targets(df)[args.target]
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
        scale_pos_weight=neg / pos if pos else 1.0,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    threshold, val_f1 = select_threshold(y_val, val_scores)
    test_rows = d[d[SPLIT_COL].eq("test")].copy()
    scored = test_rows[
        [
            "target_week",
            "nta2020",
            "ntaname",
            "boroname",
            "complaint_category",
            "complaint_category_label",
            "rolling_8w_mean",
            "rolling_8w_std",
            "target_next_week_count",
        ]
    ].reset_index(drop=False).rename(columns={"index": "original_index"})
    scored["y_true"] = y_test.to_numpy(dtype=int)
    scored["score"] = test_scores
    scored["threshold"] = threshold
    scored["prediction"] = (test_scores >= threshold).astype(int)
    scored["margin_over_threshold"] = scored["score"] - threshold
    scored["z_exceedance"] = (
        (scored["target_next_week_count"].astype(float) - scored["rolling_8w_mean"].astype(float))
        / (scored["rolling_8w_std"].astype(float).replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan)
    metrics = {
        "validation_selected_threshold": float(threshold),
        "validation_selected_f1": float(val_f1),
        "test_pr_auc": float(average_precision_score(y_test, test_scores)),
        "test_f1": float(f1_score(y_test, scored["prediction"], zero_division=0)),
        "test_precision": float(precision_score(y_test, scored["prediction"], zero_division=0)),
        "test_recall": float(recall_score(y_test, scored["prediction"], zero_division=0)),
        "fit_seconds": float(fit_seconds),
        "raw_feature_count": len(features),
        "model_feature_count": len(model_features),
    }
    return d, X_test, model, model_features, scored, metrics


def sample_for_shap(X_test: pd.DataFrame, scored: pd.DataFrame, max_rows: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(X_test) <= max_rows:
        return X_test.copy(), scored.copy()
    # Stratify lightly by y/pred so rare FN/FP patterns survive in global explanations.
    temp = scored.assign(stratum=scored["y_true"].astype(str) + scored["prediction"].astype(str))
    take = []
    per = max(1, max_rows // max(1, temp["stratum"].nunique()))
    for _, g in temp.groupby("stratum", sort=True):
        n = min(len(g), per)
        take.extend(g.sample(n=n, random_state=seed).index.tolist())
    if len(take) < max_rows:
        rest = temp.drop(index=take)
        take.extend(rest.sample(n=max_rows - len(take), random_state=seed).index.tolist())
    take = sorted(take[:max_rows])
    return X_test.iloc[take].copy(), scored.iloc[take].copy()


def compute_shap(model, X: pd.DataFrame, progress: bool) -> tuple[np.ndarray, float, shap.TreeExplainer]:
    if progress:
        print(f"[explainability] SHAP rows={len(X):,}, features={X.shape[1]:,}")
    start = time.time()
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    vals = positive_class_values(raw)
    expected = positive_expected_value(explainer)
    if progress:
        print(f"[explainability] SHAP done in {time.time() - start:.1f}s")
    return vals, expected, explainer


def global_importance(values: np.ndarray, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = pd.DataFrame(
        {
            "feature": features,
            "feature_group": [feature_group(f) for f in features],
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_signed_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    total = float(imp["mean_abs_shap"].sum())
    imp["mean_abs_shap_share"] = imp["mean_abs_shap"] / total if total else np.nan
    imp["importance_rank"] = np.arange(1, len(imp) + 1)
    group = (
        imp.groupby("feature_group", as_index=False)
        .agg(mean_abs_shap_sum=("mean_abs_shap", "sum"), feature_count=("feature", "count"))
        .sort_values("mean_abs_shap_sum", ascending=False)
    )
    group["mean_abs_shap_share"] = group["mean_abs_shap_sum"] / group["mean_abs_shap_sum"].sum()
    group["group_rank"] = np.arange(1, len(group) + 1)
    return imp, group


def select_cases(scored: pd.DataFrame) -> pd.DataFrame:
    tp = scored[(scored["y_true"].eq(1)) & (scored["prediction"].eq(1))].sort_values("score", ascending=False).head(1)
    fp = scored[(scored["y_true"].eq(0)) & (scored["prediction"].eq(1))].sort_values("score", ascending=False).head(1)
    fn_pool = scored[(scored["y_true"].eq(1)) & (scored["prediction"].eq(0))].copy()
    fn = fn_pool.sort_values(["z_exceedance", "target_next_week_count"], ascending=False, na_position="last").head(1)
    cases = []
    for label, frame in [("tp_high_confidence", tp), ("fp_high_confidence", fp), ("fn_severe_missed", fn)]:
        if frame.empty:
            continue
        row = frame.iloc[0].copy()
        row["case_type"] = label
        row["case_id"] = label.replace("_high_confidence", "").replace("_severe_missed", "")
        cases.append(row)
    return pd.DataFrame(cases)


def local_contrib(cases: pd.DataFrame, X_local: pd.DataFrame, values: np.ndarray, top_n: int) -> pd.DataFrame:
    rows = []
    features = list(X_local.columns)
    for i, (_, case) in enumerate(cases.iterrows()):
        shap_row = values[i]
        order = np.argsort(np.abs(shap_row))[::-1][:top_n]
        for rank, pos in enumerate(order, start=1):
            rows.append(
                {
                    "case_id": case["case_id"],
                    "case_type": case["case_type"],
                    "rank": rank,
                    "feature": features[pos],
                    "feature_group": feature_group(features[pos]),
                    "feature_value": X_local.iloc[i, pos],
                    "shap_value": shap_row[pos],
                    "abs_shap_value": abs(shap_row[pos]),
                }
            )
    return pd.DataFrame(rows)


def save_beeswarm(values: np.ndarray, X: pd.DataFrame, out: Path, top_n: int) -> None:
    plt.figure()
    shap.summary_plot(values, X, max_display=top_n, show=False, plot_size=(8, 8))
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def save_waterfall(case_id: str, explanation: shap.Explanation, out_dir: Path, top_n: int) -> str:
    path = out_dir / f"shap_local_{case_id}.pdf"
    plt.figure()
    shap.plots.waterfall(explanation, max_display=top_n, show=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)


def write_case_report(cases: pd.DataFrame, contrib: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Local Case Selection Report",
        "",
        "Cases are selected by pre-specified rules from the final-style 2025 T2 no-shortcut LightGBM candidate:",
        "",
        "- TP: highest-score true positive.",
        "- FP: highest-score false positive.",
        "- FN: missed positive with largest realized z-exceedance, breaking ties by next-week count.",
        "",
        md_table(
            cases,
            [
                "case_id",
                "case_type",
                "target_week",
                "nta2020",
                "ntaname",
                "boroname",
                "complaint_category",
                "y_true",
                "prediction",
                "score",
                "threshold",
                "target_next_week_count",
                "rolling_8w_mean",
                "z_exceedance",
            ],
        ),
        "",
        "Top local contributions are saved in `shap_local_case_feature_contributions.csv`. These explanations describe model score contributions, not causal effects.",
        "",
    ]
    (out_dir / "local_case_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(out_dir: Path, metrics: dict[str, object], global_imp: pd.DataFrame, group_imp: pd.DataFrame, cases: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Final-Style Explainability Report",
        "",
        f"This pass explains the actual single LightGBM score model used for the current `{args.target}` no-shortcut candidate on the final-style 2025 fold. The explanation model and score model are the same fitted LightGBM; no ensemble explanation proxy is used.",
        "",
        "## Model and Decision Context",
        "",
        md_table(pd.DataFrame([metrics]), ["validation_selected_threshold", "test_pr_auc", "test_f1", "test_precision", "test_recall", "raw_feature_count", "model_feature_count", "fit_seconds"]),
        "",
        "Formula-aligned 8-week target-construction predictors are removed: `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean`.",
        "",
        "## SHAP Group Importance",
        "",
        md_table(group_imp, ["group_rank", "feature_group", "feature_count", "mean_abs_shap_sum", "mean_abs_shap_share"], max_rows=20),
        "",
        "## Top Features",
        "",
        md_table(global_imp, ["importance_rank", "feature", "feature_group", "mean_abs_shap", "mean_abs_shap_share"], max_rows=25),
        "",
        "## Local Cases",
        "",
        md_table(cases, ["case_id", "case_type", "target_week", "boroname", "ntaname", "complaint_category", "y_true", "prediction", "score", "target_next_week_count", "z_exceedance"]),
        "",
        "## Output Figures",
        "",
        "- `shap_beeswarm.pdf`",
        "- `shap_local_tp.pdf`",
        "- `shap_local_fp.pdf`",
        "- `shap_local_fn.pdf`",
        "",
        "## Guardrails",
        "",
        "- SHAP values explain fitted LightGBM score contributions, not causal effects.",
        "- This pass supports the current candidate model; final manuscript claims still need final target/model freeze.",
        "- Calibration is monotonic and fitted on validation, so these SHAP values explain the underlying score ranking used by the calibrated decision layer.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "final_model_explainability_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if args.progress:
        print("[explainability] train current final-style candidate")
    _, X_test, model, model_features, scored, metrics = train_model(args)
    X_sample, sample_meta = sample_for_shap(X_test, scored, args.max_shap_rows, args.seed)
    shap_values, expected_value, _ = compute_shap(model, X_sample, args.progress)
    global_imp, group_imp = global_importance(shap_values, model_features)
    save_beeswarm(shap_values, X_sample, out_dir / "shap_beeswarm.pdf", args.top_n)

    cases = select_cases(scored)
    X_local = X_test.iloc[cases.index.tolist()].copy()
    local_values, local_expected, _ = compute_shap(model, X_local, args.progress)
    contrib = local_contrib(cases, X_local, local_values, args.local_top_n)
    for i, (_, case) in enumerate(cases.iterrows()):
        exp = shap.Explanation(
            values=local_values[i],
            base_values=local_expected,
            data=X_local.iloc[i].to_numpy(),
            feature_names=list(X_local.columns),
        )
        save_waterfall(str(case["case_id"]), exp, out_dir, args.local_top_n)

    write_csv(out_dir / "shap_global_importance.csv", global_imp, args.overwrite)
    write_csv(out_dir / "shap_group_importance.csv", group_imp, args.overwrite)
    write_csv(out_dir / "shap_sample_metadata.csv", sample_meta, args.overwrite)
    write_csv(out_dir / "shap_local_cases.csv", cases, args.overwrite)
    write_csv(out_dir / "shap_local_case_feature_contributions.csv", contrib, args.overwrite)
    write_case_report(cases, contrib, out_dir)
    write_report(out_dir, metrics, global_imp, group_imp, cases, args)
    write_json(
        out_dir / "explainability_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_explainability.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "target": args.target,
            "seed": args.seed,
            "fold": FINAL_FOLD,
            "shap_rows": int(len(X_sample)),
            "local_cases": int(len(cases)),
            "expected_value": expected_value,
            "local_expected_value": local_expected,
            "metrics": metrics,
            "removed_formula_aligned_features": FORMULA_ALIGNED_REMOVE,
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
