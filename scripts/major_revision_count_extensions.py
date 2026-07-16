from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, mean_absolute_error, mean_poisson_deviance, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_model_audits import (  # noqa: E402
    COUNT_TARGET_COL,
    FORMULA_ALIGNED_REMOVE,
    SPLIT_COL,
    TARGET_COL,
    THRESHOLD_COL,
    count_baseline_features,
    load_dataset,
    precision_at_k,
    select_threshold,
    write_csv,
    write_json,
)
from major_revision_rolling_origin import assign_fold_split  # noqa: E402
from major_revision_target_selection import TARGET_DEFINITIONS, make_targets, md_table  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/count_extensions"
ROOT_REPORT = ROOT / "count_model_extension_report.md"
FINAL_FOLD = {"fold_id": "final_style_2025", "train_end_year": 2023, "validation_year": 2024, "test_year": 2025}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run additional count-model baselines for the major revision.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--target", default="T2_min_count_3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def safe_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(mean_poisson_deviance(y_true, np.clip(y_pred, 1e-9, None)))
    except Exception:
        return np.nan


def load_final_style(target: str) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    features = [f for f in count_baseline_features(include_nta=False) if f != "nta2020"]
    required = sorted(set(features + ["rolling_8w_mean"]))
    df = load_dataset({"count_features": required})
    if target not in TARGET_DEFINITIONS:
        raise ValueError(f"Unknown target definition: {target}")
    frame = make_targets(df)[target]
    frame = assign_fold_split(frame, FINAL_FOLD)
    categorical = [f for f in ["complaint_category", "boroname"] if f in features]
    numeric = [f for f in features if f not in categorical]
    return frame, features, numeric, categorical


def transform_count_features(
    train: pd.DataFrame,
    frames: list[pd.DataFrame],
    numeric: list[str],
    categorical: list[str],
) -> tuple[list[np.ndarray], list[int], OrdinalEncoder | None]:
    numeric_frames = []
    log_cols = [
        c
        for c in numeric
        if c.endswith("_count")
        or c.endswith("_mean")
        or c.endswith("_std")
        or c in {"complaint_count", "history_weeks_available"}
    ]
    medians = {}
    for col in numeric:
        s = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians[col] = 0.0 if pd.isna(s.median()) else float(s.median())
    for frame in frames:
        cols = {}
        for col in numeric:
            s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if col in log_cols:
                s = np.log1p(s.clip(lower=0))
            cols[col] = s.fillna(medians[col]).astype(float)
        numeric_frames.append(pd.DataFrame(cols, index=frame.index))

    encoder: OrdinalEncoder | None = None
    cat_arrays = []
    categorical_indices: list[int] = []
    if categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan)
        train_cats = train[categorical].astype("object").where(train[categorical].notna(), "__MISSING__").astype(str)
        encoder.fit(train_cats)
        for frame in frames:
            cats = frame[categorical].astype("object").where(frame[categorical].notna(), "__MISSING__").astype(str)
            cat_arrays.append(encoder.transform(cats))
        categorical_indices = list(range(len(numeric), len(numeric) + len(categorical)))

    transformed = []
    for i, num in enumerate(numeric_frames):
        if categorical:
            transformed.append(np.hstack([num.to_numpy(dtype=float), cat_arrays[i]]))
        else:
            transformed.append(num.to_numpy(dtype=float))
    return transformed, categorical_indices, encoder


def target_event_from_name(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == "T2_min_count_3":
        return (frame[TARGET_COL].astype(int).eq(1) & frame[COUNT_TARGET_COL].ge(3)).to_numpy(dtype=int)
    if target == "T1_min_count_2":
        return (frame[TARGET_COL].astype(int).eq(1) & frame[COUNT_TARGET_COL].ge(2)).to_numpy(dtype=int)
    return frame[TARGET_COL].astype(int).to_numpy()


def formula_event_from_count(frame: pd.DataFrame, pred_count: np.ndarray, target: str) -> np.ndarray:
    event = pred_count > frame[THRESHOLD_COL].to_numpy(dtype=float)
    if target == "T2_min_count_3":
        event = event & (pred_count >= 3)
    elif target == "T1_min_count_2":
        event = event & (pred_count >= 2)
    return event


def event_metrics(
    model_name: str,
    decision_mode: str,
    split: str,
    frame: pd.DataFrame,
    y_event: np.ndarray,
    y_count: np.ndarray,
    pred_count: np.ndarray,
    event_pred: np.ndarray,
    threshold: float,
    fit_seconds: float,
    feature_count: int,
) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_event, event_pred, labels=[0, 1]).ravel()
    return {
        "model_name": model_name,
        "decision_mode": decision_mode,
        "split": split,
        "rows": int(len(frame)),
        "positive_rows": int(y_event.sum()),
        "positive_share": float(y_event.mean()) if len(y_event) else np.nan,
        "feature_count": int(feature_count),
        "fit_seconds": float(fit_seconds),
        "count_mae": float(mean_absolute_error(y_count, pred_count)),
        "mean_observed_count": float(np.mean(y_count)),
        "mean_predicted_count": float(np.mean(pred_count)),
        "poisson_deviance": safe_poisson_deviance(y_count, pred_count),
        "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
        "alert_rate": float(np.mean(event_pred)),
        "precision": float(precision_score(y_event, event_pred, zero_division=0)),
        "recall": float(recall_score(y_event, event_pred, zero_division=0)),
        "f1": float(f1_score(y_event, event_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_event, pred_count)) if len(np.unique(y_event)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y_event, pred_count)) if len(np.unique(y_event)) > 1 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_at_1pct": precision_at_k(y_event, pred_count, 0.01),
        "precision_at_5pct": precision_at_k(y_event, pred_count, 0.05),
        "precision_at_10pct": precision_at_k(y_event, pred_count, 0.10),
    }


def evaluate_count_model(
    model_name: str,
    target: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    fit_seconds: float,
    feature_count: int,
) -> list[dict[str, object]]:
    rows = []
    y_val = target_event_from_name(val, target)
    y_test = target_event_from_name(test, target)
    y_count_val = val[COUNT_TARGET_COL].to_numpy(dtype=float)
    y_count_test = test[COUNT_TARGET_COL].to_numpy(dtype=float)
    validation_threshold, _ = select_threshold(pd.Series(y_val), pred_val)
    for split_name, frame, y_event, y_count, pred_count in [
        ("validation", val, y_val, y_count_val, pred_val),
        ("test", test, y_test, y_count_test, pred_test),
    ]:
        rows.append(
            event_metrics(
                model_name,
                "formula_threshold",
                split_name,
                frame,
                y_event,
                y_count,
                pred_count,
                formula_event_from_count(frame, pred_count, target),
                np.nan,
                fit_seconds,
                feature_count,
            )
        )
        rows.append(
            event_metrics(
                model_name,
                "validation_score_threshold",
                split_name,
                frame,
                y_event,
                y_count,
                pred_count,
                pred_count >= validation_threshold,
                validation_threshold,
                fit_seconds,
                feature_count,
            )
        )
    return rows


def run_hgb_poisson(args: argparse.Namespace, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, numeric: list[str], categorical: list[str]) -> list[dict[str, object]]:
    (X_train, X_val, X_test), cat_idx, _ = transform_count_features(train, [train, val, test], numeric, categorical)
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        categorical_features=cat_idx if cat_idx else None,
        random_state=args.seed,
    )
    start = time.time()
    model.fit(X_train, train[COUNT_TARGET_COL].to_numpy(dtype=float))
    fit_seconds = time.time() - start
    pred_val = np.clip(model.predict(X_val), 0, None)
    pred_test = np.clip(model.predict(X_test), 0, None)
    return evaluate_count_model(
        "hist_gradient_boosting_poisson_count",
        args.target,
        train,
        val,
        test,
        pred_val,
        pred_test,
        fit_seconds,
        len(numeric) + len(categorical),
    )


def run_hurdle(args: argparse.Namespace, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, numeric: list[str], categorical: list[str]) -> list[dict[str, object]]:
    (X_train, X_val, X_test), cat_idx, _ = transform_count_features(train, [train, val, test], numeric, categorical)
    y_occ = train[COUNT_TARGET_COL].gt(0).astype(int).to_numpy()
    y_count = train[COUNT_TARGET_COL].to_numpy(dtype=float)
    positive = y_count > 0
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
    start = time.time()
    occ.fit(X_train, y_occ)
    count.fit(X_train[positive], y_count[positive])
    fit_seconds = time.time() - start

    def predict_expected(X: np.ndarray) -> np.ndarray:
        p_occ = occ.predict_proba(X)[:, 1]
        pos_count = np.clip(count.predict(X), 0, None)
        return np.clip(p_occ * pos_count, 0, None)

    pred_val = predict_expected(X_val)
    pred_test = predict_expected(X_test)
    return evaluate_count_model(
        "hurdle_hgb_occurrence_poisson_positive_count",
        args.target,
        train,
        val,
        test,
        pred_val,
        pred_test,
        fit_seconds,
        len(numeric) + len(categorical),
    )


def negative_binomial_status() -> pd.DataFrame:
    available = importlib.util.find_spec("statsmodels") is not None
    return pd.DataFrame(
        [
            {
                "model_name": "negative_binomial_glm",
                "status": "available_not_run" if available else "blocked_missing_statsmodels",
                "reason": "statsmodels is installed" if available else "statsmodels is not installed in the local environment; sklearn has no Negative Binomial GLM.",
                "next_action": "Run NB GLM on reduced and NTA FE formulations." if available else "Document blocker or install statsmodels before final NB attempt.",
            }
        ]
    )


def write_report(out_dir: Path, metrics: pd.DataFrame, nb_status: pd.DataFrame, args: argparse.Namespace) -> None:
    test = metrics[metrics["split"].eq("test")].sort_values(["model_name", "decision_mode"])
    lines = [
        "# Count Model Extension Report",
        "",
        f"This report adds overdispersed/tree and hurdle-style count baselines for `{args.target}` on the final-style 2025 fold. It complements the earlier Poisson and Poisson + NTA fixed-effect audit.",
        "",
        "## Negative Binomial Status",
        "",
        md_table(nb_status, ["model_name", "status", "reason", "next_action"]),
        "",
        "## Held-Out 2025 Test Metrics",
        "",
        md_table(
            test,
            [
                "model_name",
                "decision_mode",
                "count_mae",
                "mean_predicted_count",
                "poisson_deviance",
                "pr_auc",
                "precision_at_5pct",
                "f1",
                "precision",
                "recall",
                "alert_rate",
            ],
        ),
        "",
        "## Guardrails",
        "",
        "- These models predict `target_next_week_count` and are converted to the candidate abnormal-event target using either the original count-threshold formula or a validation-selected score threshold.",
        "- The hurdle model is a practical two-stage baseline, not a full statistical hurdle/negative-binomial model.",
        "- The final manuscript should compare count baselines against the frozen final classifier with paired uncertainty before making a strong dominance claim.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "count_model_extension_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    frame, features, numeric, categorical = load_final_style(args.target)
    train = frame[frame[SPLIT_COL].eq("train")].copy()
    val = frame[frame[SPLIT_COL].eq("validation")].copy()
    test = frame[frame[SPLIT_COL].eq("test")].copy()
    if args.progress:
        print(f"[count-ext] splits: train={len(train)} validation={len(val)} test={len(test)}")
        print(f"[count-ext] features: numeric={len(numeric)} categorical={categorical}")

    rows = []
    rows.extend(run_hgb_poisson(args, train, val, test, numeric, categorical))
    rows.extend(run_hurdle(args, train, val, test, numeric, categorical))
    metrics = pd.DataFrame(rows)
    nb_status = negative_binomial_status()
    write_csv(out_dir / "count_model_extension_results.csv", metrics, args.overwrite)
    write_csv(out_dir / "negative_binomial_status.csv", nb_status, args.overwrite)
    write_report(out_dir, metrics, nb_status, args)
    write_json(
        out_dir / "count_model_extension_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_count_extensions.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "target": args.target,
            "seed": args.seed,
            "fold": FINAL_FOLD,
            "features": features,
            "categorical_features": categorical,
            "removed_formula_aligned_features": FORMULA_ALIGNED_REMOVE,
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
