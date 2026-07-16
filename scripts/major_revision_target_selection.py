from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_model_audits import (  # noqa: E402
    COUNT_TARGET_COL,
    FORMULA_ALIGNED_REMOVE,
    SPLIT_COL,
    TARGET_COL,
    THRESHOLD_COL,
    best_lgbm_params,
    load_dataset,
    load_feature_sets,
    precision_at_k,
    select_threshold,
    write_csv,
    write_json,
)


OUT_DIR = ROOT / "data/processed/model_results/major_revision/target_selection"
ROOT_REPORT = ROOT / "target_definition_selection_report.md"


TARGET_DEFINITIONS = {
    "T0_current_reference": {
        "description": "count_t+1 > mu_8w + 1.5 sigma_8w",
        "min_count": None,
        "min_mu8w": None,
    },
    "T1_min_count_2": {
        "description": "T0 AND count_t+1 >= 2",
        "min_count": 2,
        "min_mu8w": None,
    },
    "T2_min_count_3": {
        "description": "T0 AND count_t+1 >= 3",
        "min_count": 3,
        "min_mu8w": None,
    },
    "T3_mu8w_ge_1_eligible": {
        "description": "Evaluate only rows with mu_8w >= 1, target remains T0 within eligible rows",
        "min_count": None,
        "min_mu8w": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sparse-aware target definitions for the major revision.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def make_targets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base_event = df[TARGET_COL].astype(int).eq(1).to_numpy()
    out: dict[str, pd.DataFrame] = {}
    for name, spec in TARGET_DEFINITIONS.items():
        d = df.copy()
        if spec["min_mu8w"] is not None:
            d = d[d["rolling_8w_mean"].ge(float(spec["min_mu8w"]))].copy()
            event = d[TARGET_COL].astype(int).eq(1).to_numpy()
        else:
            event = base_event.copy()
        if spec["min_count"] is not None:
            event = event & df[COUNT_TARGET_COL].ge(int(spec["min_count"])).to_numpy()
        if spec["min_mu8w"] is None:
            d = d.copy()
        d["candidate_target"] = event.astype(int)
        d["target_definition"] = name
        out[name] = d
    return out


def encode_splits(
    df: pd.DataFrame,
    features: list[str],
    categorical_all: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    cat_features = [f for f in features if f in categorical_all]
    num_features = [f for f in features if f not in cat_features]

    train = df[df[SPLIT_COL].eq("train")].copy()
    val = df[df[SPLIT_COL].eq("validation")].copy()
    test = df[df[SPLIT_COL].eq("test")].copy()

    medians = {}
    for col in num_features:
        s = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians[col] = 0.0 if pd.isna(s.median()) else float(s.median())

    def transform(raw: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if num_features:
            num = pd.DataFrame(
                {
                    col: pd.to_numeric(raw[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(medians[col])
                    for col in num_features
                },
                index=raw.index,
            )
            parts.append(num)
        for col in cat_features:
            cat = raw[col].fillna("missing").astype(str)
            dummies = pd.get_dummies(cat, prefix=col, dtype=float)
            parts.append(dummies)
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=raw.index)

    X_train = transform(train)
    X_val = transform(val).reindex(columns=X_train.columns, fill_value=0)
    X_test = transform(test).reindex(columns=X_train.columns, fill_value=0)
    return (
        X_train,
        train["candidate_target"].astype(int),
        X_val,
        val["candidate_target"].astype(int),
        X_test,
        test["candidate_target"].astype(int),
        list(X_train.columns),
    )


def score_metrics(y: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_arr = y.to_numpy(dtype=int)
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_arr)),
        "positive_rows": int(y_arr.sum()),
        "positive_share": float(y_arr.mean()) if len(y_arr) else np.nan,
        "threshold": float(threshold),
        "predicted_positive_rows": int(pred.sum()),
        "alert_rate": float(pred.mean()) if len(pred) else np.nan,
        "precision": float(precision_score(y_arr, pred, zero_division=0)),
        "recall": float(recall_score(y_arr, pred, zero_division=0)),
        "f1": float(f1_score(y_arr, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_arr, scores)) if len(np.unique(y_arr)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y_arr, scores)) if len(np.unique(y_arr)) > 1 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_at_1pct": precision_at_k(y_arr, scores, 0.01),
        "precision_at_5pct": precision_at_k(y_arr, scores, 0.05),
        "precision_at_10pct": precision_at_k(y_arr, scores, 0.10),
    }


def composition_rows(full_df: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    full_split_rows = full_df.groupby(SPLIT_COL).size().to_dict()
    for name, d in candidates.items():
        for split, g in d.groupby(SPLIT_COL, sort=True):
            pos = g[g["candidate_target"].eq(1)]
            rows.append(
                {
                    "target_definition": name,
                    "description": TARGET_DEFINITIONS[name]["description"],
                    "split": split,
                    "rows": int(len(g)),
                    "full_split_rows": int(full_split_rows.get(split, len(g))),
                    "excluded_rows": int(full_split_rows.get(split, len(g)) - len(g)),
                    "positive_rows": int(pos.shape[0]),
                    "positive_share": float(g["candidate_target"].mean()) if len(g) else np.nan,
                    "positive_mu8w_lt_0p25": int(pos["rolling_8w_mean"].lt(0.25).sum()),
                    "positive_mu8w_lt_0p5": int(pos["rolling_8w_mean"].lt(0.5).sum()),
                    "positive_mu8w_lt_1": int(pos["rolling_8w_mean"].lt(1).sum()),
                    "positive_mu8w_lt_2": int(pos["rolling_8w_mean"].lt(2).sum()),
                    "positive_share_mu8w_lt_1": float(pos["rolling_8w_mean"].lt(1).mean()) if len(pos) else np.nan,
                    "positive_count_eq_1": int(pos[COUNT_TARGET_COL].eq(1).sum()),
                    "positive_count_eq_2": int(pos[COUNT_TARGET_COL].eq(2).sum()),
                    "positive_count_eq_3": int(pos[COUNT_TARGET_COL].eq(3).sum()),
                    "positive_count_ge_4": int(pos[COUNT_TARGET_COL].ge(4).sum()),
                    "positive_share_count_ge_4": float(pos[COUNT_TARGET_COL].ge(4).mean()) if len(pos) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def decile_rows(target_name: str, split: str, d: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = d[d[SPLIT_COL].eq(split)].copy()
    frame["score"] = scores
    frame["pred"] = frame["score"].ge(threshold).astype(int)
    frame["volume_decile"] = pd.qcut(
        frame["rolling_8w_mean"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    rows = []
    for decile, g in frame.groupby("volume_decile", observed=True):
        y = g["candidate_target"].astype(int).to_numpy()
        pred = g["pred"].astype(int).to_numpy()
        scores_g = g["score"].to_numpy(dtype=float)
        rows.append(
            {
                "target_definition": target_name,
                "split": split,
                "volume_decile": str(decile),
                "rows": int(len(g)),
                "rolling_8w_mean_min": float(g["rolling_8w_mean"].min()),
                "rolling_8w_mean_median": float(g["rolling_8w_mean"].median()),
                "rolling_8w_mean_max": float(g["rolling_8w_mean"].max()),
                "positive_rows": int(y.sum()),
                "positive_share": float(y.mean()),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "pr_auc": float(average_precision_score(y, scores_g)) if len(np.unique(y)) > 1 else np.nan,
                "alert_rate": float(pred.mean()),
                "precision_at_1pct": precision_at_k(y, scores_g, 0.01),
                "precision_at_5pct": precision_at_k(y, scores_g, 0.05),
            }
        )
    return pd.DataFrame(rows)


def train_targets(
    candidates: dict[str, pd.DataFrame],
    features: list[str],
    categorical: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = best_lgbm_params()
    metrics = []
    deciles = []
    for name, d in candidates.items():
        if args.progress:
            print(f"[target] {name}: rows={len(d):,}")
        start = time.time()
        X_train, y_train, X_val, y_val, X_test, y_test, model_features = encode_splits(d, features, categorical)
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
            random_state=args.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        fit_start = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - fit_start
        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold, val_f1_at_threshold = select_threshold(y_val, val_scores)
        for split, y, scores in [("validation", y_val, val_scores), ("test", y_test, test_scores)]:
            rec = score_metrics(y, scores, threshold)
            rec.update(
                {
                    "target_definition": name,
                    "description": TARGET_DEFINITIONS[name]["description"],
                    "model_name": "lightgbm_no_8w_formula_features",
                    "split": split,
                    "raw_feature_count": len(features),
                    "model_feature_count": len(model_features),
                    "validation_selected_f1": val_f1_at_threshold,
                    "fit_seconds": fit_seconds,
                    "elapsed_seconds": time.time() - start,
                    "removed_formula_aligned_features": ", ".join(FORMULA_ALIGNED_REMOVE),
                }
            )
            metrics.append(rec)
        deciles.append(decile_rows(name, "validation", d, val_scores, threshold))
        deciles.append(decile_rows(name, "test", d, test_scores, threshold))
    return pd.DataFrame(metrics), pd.concat(deciles, ignore_index=True)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    d = df[[c for c in columns if c in df.columns]].head(max_rows).copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            d[col] = d[col].astype(str)
    return "\n".join(
        [
            "| " + " | ".join(d.columns) + " |",
            "| " + " | ".join(["---"] * len(d.columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in d.to_numpy(dtype=str)],
        ]
    )


def write_report(out_dir: Path, composition: pd.DataFrame, metrics: pd.DataFrame, deciles: pd.DataFrame) -> str:
    val = metrics[metrics["split"].eq("validation")].sort_values("pr_auc", ascending=False)
    test = metrics[metrics["split"].eq("test")]
    comp_test = composition[composition["split"].eq("test")]
    report = "\n".join(
        [
            "# Target Definition Selection Report",
            "",
            "This report evaluates sparse-aware target definitions using the existing data only. The later revision freeze selects T2 by construct validity plus validation/backtest evidence; the held-out diagnostics below are retained as supporting checks, not as the sole selection rule.",
            "",
            "All model rows below use the no-shortcut LightGBM feature set with `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean` removed.",
            "",
            "## Candidate Definitions",
            "",
            md_table(
                pd.DataFrame(
                    [
                        {"target_definition": k, "description": v["description"]}
                        for k, v in TARGET_DEFINITIONS.items()
                    ]
                ),
                ["target_definition", "description"],
            ),
            "",
            "## Test-Period Target Composition",
            "",
            md_table(
                comp_test,
                [
                    "target_definition",
                    "rows",
                    "excluded_rows",
                    "positive_rows",
                    "positive_share",
                    "positive_share_mu8w_lt_1",
                    "positive_count_eq_1",
                    "positive_count_eq_2",
                    "positive_count_eq_3",
                    "positive_count_ge_4",
                    "positive_share_count_ge_4",
                ],
            ),
            "",
            "## Validation Metrics for Selection",
            "",
            md_table(
                val,
                [
                    "target_definition",
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
            ),
            "",
            "## Held-Out Test Diagnostics",
            "",
            md_table(
                test,
                [
                    "target_definition",
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
            ),
            "",
            "## Volume-Decile Diagnostics",
            "",
            "Detailed validation and test decile results are saved in `target_definition_decile_results.csv`. These results are required before any target can be defended in the manuscript because sparse low-volume rows behave differently from high-volume rows.",
            "",
            "## Interpretation",
            "",
            "- T1 and T2 reduce one-call positives by construction; this improves construct validity for sparse cells but changes the event being forecast.",
            "- T3 removes low-baseline rows from the risk set, so it is not directly comparable to T0/T1/T2 and may exclude low-volume neighborhoods or categories.",
            "- T4 hurdle-style modeling is addressed as count/hurdle baseline evidence in the count-model extension artifacts rather than as a separate final binary target label.",
            "- The final manuscript selects T2 because it reduces one-call/two-call sparse-cell artifacts while retaining the full dense-panel risk set; this choice is documented in `pre_registered_revision_selection_rule.md` and should not be re-selected by held-out test metrics.",
            "",
        ]
    )
    (out_dir / "target_definition_selection_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    feature_sets, categorical = load_feature_sets()
    features = feature_sets["B_no_8w_formula_features"]
    required_features = sorted(set(features + ["rolling_8w_mean"]))
    df = load_dataset({"B_no_8w_formula_features": required_features})
    candidates = make_targets(df)
    composition = composition_rows(df, candidates)
    metrics, deciles = train_targets(candidates, features, categorical, args)

    write_csv(out_dir / "target_definition_composition.csv", composition, args.overwrite)
    write_csv(out_dir / "target_definition_results.csv", metrics, args.overwrite)
    write_csv(out_dir / "target_definition_decile_results.csv", deciles, args.overwrite)
    write_report(out_dir, composition, metrics, deciles)
    write_json(
        out_dir / "target_definition_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_target_selection.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "rows_loaded": int(len(df)),
            "target_definitions": TARGET_DEFINITIONS,
            "feature_set": "B_no_8w_formula_features",
            "removed_formula_aligned_features": FORMULA_ALIGNED_REMOVE,
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
