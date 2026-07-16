from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

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
from major_revision_rolling_origin import FOLDS, assign_fold_split  # noqa: E402
from major_revision_target_selection import (  # noqa: E402
    TARGET_DEFINITIONS,
    encode_splits,
    make_targets,
    md_table,
)


OUT_DIR = ROOT / "data/processed/model_results/major_revision/calibration"
ROOT_REPORT = ROOT / "calibration_report.md"

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fold-specific calibration audits for the major revision.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--targets",
        default="T0_current_reference,T1_min_count_2,T2_min_count_3,T3_mu8w_ge_1_eligible",
        help="Comma-separated target definitions to evaluate.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def clip_prob(scores: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)


def fit_platt(val_scores: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(clip_prob(val_scores).reshape(-1, 1), y_val)
    return model


def apply_platt(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(clip_prob(scores).reshape(-1, 1))[:, 1]


def fit_isotonic(val_scores: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(clip_prob(val_scores), y_val)
    return model


def calibration_bins(y: np.ndarray, prob: np.ndarray, n_bins: int) -> pd.DataFrame:
    p = clip_prob(prob)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            rows.append(
                {
                    "bin_id": i + 1,
                    "bin_lower": lo,
                    "bin_upper": hi,
                    "rows": 0,
                    "mean_score": np.nan,
                    "observed_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        mean_score = float(p[mask].mean())
        observed = float(y[mask].mean())
        rows.append(
            {
                "bin_id": i + 1,
                "bin_lower": lo,
                "bin_upper": hi,
                "rows": int(mask.sum()),
                "mean_score": mean_score,
                "observed_rate": observed,
                "abs_gap": abs(mean_score - observed),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int) -> tuple[float, float]:
    bins = calibration_bins(y, prob, n_bins)
    total = float(bins["rows"].sum())
    if total == 0:
        return np.nan, np.nan
    nonempty = bins[bins["rows"].gt(0)].copy()
    ece = float((nonempty["rows"] / total * nonempty["abs_gap"]).sum())
    mce = float(nonempty["abs_gap"].max())
    return ece, mce


def calibration_slope_intercept(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    p = clip_prob(prob)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def metric_row(y: np.ndarray, scores: np.ndarray, threshold: float, n_bins: int) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = clip_prob(scores)
    pred = p >= threshold
    ece, mce = expected_calibration_error(y, p, n_bins)
    slope, intercept = calibration_slope_intercept(y, p)
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
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "precision_at_1pct": precision_at_k(y, p, 0.01),
        "precision_at_5pct": precision_at_k(y, p, 0.05),
        "precision_at_10pct": precision_at_k(y, p, 0.10),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": ece,
        "mce": mce,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def train_scores_for_fold(
    target_name: str,
    frame: pd.DataFrame,
    fold: dict[str, int | str],
    features: list[str],
    categorical: list[str],
    random_state: int,
    progress: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    d = assign_fold_split(frame, fold)
    if progress:
        print(f"[calibration] {target_name} {fold['fold_id']}: {d[SPLIT_COL].value_counts().to_dict()}")
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
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - fit_start
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    threshold, _ = select_threshold(y_val, val_scores)
    split_frame = pd.DataFrame(
        {
            "split": ["validation", "test"],
            "rows": [len(y_val), len(y_test)],
            "fit_seconds": [fit_seconds, fit_seconds],
            "threshold": [threshold, threshold],
            "model_feature_count": [len(model_features), len(model_features)],
        }
    )
    payload = {
        "y_validation": y_val.to_numpy(dtype=int),
        "score_validation": val_scores,
        "y_test": y_test.to_numpy(dtype=int),
        "score_test": test_scores,
    }
    return split_frame, payload


def run_calibration(
    candidates: dict[str, pd.DataFrame],
    targets: list[str],
    features: list[str],
    categorical: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    bin_rows = []
    prediction_rows = []
    for target_name in targets:
        frame = candidates[target_name]
        for fold in FOLDS:
            split_frame, payload = train_scores_for_fold(
                target_name=target_name,
                frame=frame,
                fold=fold,
                features=features,
                categorical=categorical,
                random_state=args.random_state,
                progress=args.progress,
            )
            y_val = payload["y_validation"]
            val_raw = payload["score_validation"]
            y_test = payload["y_test"]
            test_raw = payload["score_test"]
            threshold = float(split_frame.loc[split_frame["split"].eq("validation"), "threshold"].iloc[0])

            platt = fit_platt(val_raw, y_val)
            iso = fit_isotonic(val_raw, y_val)
            method_scores = {
                "uncalibrated": {
                    "validation": val_raw,
                    "test": test_raw,
                    "threshold_validation": threshold,
                },
                "platt": {
                    "validation": apply_platt(platt, val_raw),
                    "test": apply_platt(platt, test_raw),
                    "threshold_validation": float(apply_platt(platt, np.array([threshold]))[0]),
                },
                "isotonic": {
                    "validation": iso.predict(clip_prob(val_raw)),
                    "test": iso.predict(clip_prob(test_raw)),
                    "threshold_validation": float(iso.predict(np.array([threshold]))[0]),
                },
            }

            for method, scores_by_split in method_scores.items():
                for split, y in [("validation", y_val), ("test", y_test)]:
                    scores = scores_by_split[split]
                    rec = metric_row(y, scores, float(scores_by_split["threshold_validation"]), args.n_bins)
                    rec.update(
                        {
                            "target_definition": target_name,
                            "description": TARGET_DEFINITIONS[target_name]["description"],
                            "fold_id": fold["fold_id"],
                            "train_end_year": fold["train_end_year"],
                            "validation_year": fold["validation_year"],
                            "test_year": fold["test_year"],
                            "split": split,
                            "calibration_method": method,
                            "calibrator_fit_split": "validation" if method != "uncalibrated" else "none",
                            "model_name": "lightgbm_no_8w_formula_features",
                            "removed_formula_aligned_features": ", ".join(FORMULA_ALIGNED_REMOVE),
                            "raw_feature_count": len(features),
                            "model_feature_count": int(split_frame["model_feature_count"].iloc[0]),
                            "fit_seconds": float(split_frame["fit_seconds"].iloc[0]),
                        }
                    )
                    metric_rows.append(rec)
                    bins = calibration_bins(y, scores, args.n_bins)
                    bins.insert(0, "calibration_method", method)
                    bins.insert(0, "split", split)
                    bins.insert(0, "test_year", fold["test_year"])
                    bins.insert(0, "validation_year", fold["validation_year"])
                    bins.insert(0, "fold_id", fold["fold_id"])
                    bins.insert(0, "target_definition", target_name)
                    bin_rows.append(bins)

            if args.save_predictions:
                pred = pd.DataFrame(
                    {
                        "target_definition": target_name,
                        "fold_id": fold["fold_id"],
                        "test_year": fold["test_year"],
                        "y_true": y_test,
                        "uncalibrated_score": test_raw,
                        "platt_score": method_scores["platt"]["test"],
                        "isotonic_score": method_scores["isotonic"]["test"],
                    }
                )
                prediction_rows.append(pred)

    metrics = pd.DataFrame(metric_rows)
    bins = pd.concat(bin_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return metrics, bins, predictions


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"].eq("test")].copy()
    fields = ["brier", "log_loss", "ece", "mce", "pr_auc", "precision_at_5pct", "f1", "threshold"]
    rows = []
    for (target, method), g in test.groupby(["target_definition", "calibration_method"], sort=True):
        row = {"target_definition": target, "calibration_method": method, "folds": int(g["fold_id"].nunique())}
        for field in fields:
            row[f"{field}_mean"] = float(g[field].mean())
            row[f"{field}_std"] = float(g[field].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_reliability(bins: pd.DataFrame, out_path: Path) -> None:
    final_bins = bins[
        bins["fold_id"].eq("final_style_2025")
        & bins["split"].eq("test")
        & bins["target_definition"].isin(["T0_current_reference", "T2_min_count_3"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    titles = {
        "T0_current_reference": "T0 original target",
        "T2_min_count_3": "T2 minimum-count target",
    }
    for ax, target in zip(axes, ["T0_current_reference", "T2_min_count_3"]):
        sub_target = final_bins[final_bins["target_definition"].eq(target)]
        ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1.0, label="perfect")
        for method, style in [("uncalibrated", "o-"), ("platt", "s-"), ("isotonic", "^-")]:
            sub = sub_target[sub_target["calibration_method"].eq(method) & sub_target["rows"].gt(0)]
            ax.plot(sub["mean_score"], sub["observed_rate"], style, linewidth=1.2, markersize=4, label=method)
        ax.set_title(titles[target])
        ax.set_xlabel("Mean predicted probability")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Observed event rate")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Reliability diagrams, held-out 2025 evaluation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_report(out_dir: Path, metrics: pd.DataFrame, summary: pd.DataFrame) -> str:
    test_summary = summary.sort_values(["target_definition", "brier_mean"])
    final_test = metrics[
        metrics["split"].eq("test")
        & metrics["fold_id"].eq("final_style_2025")
        & metrics["target_definition"].isin(["T0_current_reference", "T2_min_count_3"])
    ].sort_values(["target_definition", "brier"])
    lines = [
        "# Calibration Report",
        "",
        "This report evaluates fold-specific calibration for the no-shortcut LightGBM models. Platt and isotonic calibrators are fit only on each fold's validation year, then evaluated unchanged on that fold's test year.",
        "",
        "Calibration methods:",
        "",
        "- `uncalibrated`: raw LightGBM probabilities.",
        "- `platt`: logistic calibration fit on validation scores.",
        "- `isotonic`: isotonic regression fit on validation scores.",
        "",
        "## Mean Test Metrics Across Rolling-Origin Folds",
        "",
        md_table(
            test_summary,
            [
                "target_definition",
                "calibration_method",
                "folds",
                "brier_mean",
                "brier_std",
                "ece_mean",
                "ece_std",
                "log_loss_mean",
                "pr_auc_mean",
                "precision_at_5pct_mean",
                "f1_mean",
            ],
            max_rows=80,
        ),
        "",
        "## Final-Style 2025 Test Fold",
        "",
        md_table(
            final_test,
            [
                "target_definition",
                "calibration_method",
                "brier",
                "ece",
                "log_loss",
                "calibration_slope",
                "calibration_intercept",
                "pr_auc",
                "precision_at_5pct",
                "f1",
            ],
            max_rows=80,
        ),
        "",
        "## Reliability Diagram",
        "",
        "- PDF: `data/processed/model_results/major_revision/calibration/reliability_diagram.pdf`",
        "- Reliability bins: `data/processed/model_results/major_revision/calibration/reliability_bins.csv`",
        "",
        "## Interpretation Guardrails",
        "",
        "- Calibration should be judged on Brier score, log loss, ECE, and reliability curves, not PR-AUC alone.",
        "- Platt and isotonic are monotone transformations for most score ranges, so ranking metrics can remain nearly unchanged; do not claim ranking improvement unless PR-AUC or precision@k changes.",
        "- Isotonic can overfit in sparse score regions; final method selection still needs uncertainty intervals and final target/model freeze.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "calibration_report.md").write_text(report, encoding="utf-8")
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

    metrics, bins, predictions = run_calibration(candidates, targets, features, categorical, args)
    summary = summarize(metrics)
    write_csv(out_dir / "calibration_results.csv", metrics, args.overwrite)
    write_csv(out_dir / "calibration_summary.csv", summary, args.overwrite)
    write_csv(out_dir / "reliability_bins.csv", bins, args.overwrite)
    if args.save_predictions and not predictions.empty:
        write_csv(out_dir / "calibration_test_predictions.csv.gz", predictions, args.overwrite)
    plot_reliability(bins, out_dir / "reliability_diagram.pdf")
    write_report(out_dir, metrics, summary)
    write_json(
        out_dir / "calibration_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_calibration.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "targets": targets,
            "folds": FOLDS,
            "n_bins": args.n_bins,
            "calibration_methods": ["uncalibrated", "platt", "isotonic"],
            "calibrator_fit_split": "validation",
            "feature_set": "B_no_8w_formula_features",
            "removed_formula_aligned_features": FORMULA_ALIGNED_REMOVE,
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
