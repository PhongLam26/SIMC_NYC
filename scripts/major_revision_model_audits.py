from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_poisson_deviance,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover
    raise RuntimeError("lightgbm is required for this audit script") from exc

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


ROOT = Path(__file__).resolve().parents[1]
FINAL_DATASET = ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz"
MODEL_CONFIG = ROOT / "data/processed/model_ready/model_config.json"
LGBM_SUMMARY = ROOT / "data/processed/model_results/prospective/tuning_lgbm/gbm_tuning_run_summary.json"
OUT_DIR = ROOT / "data/processed/model_results/major_revision/model_audits"

TARGET_COL = "abnormal_increase_next_week"
COUNT_TARGET_COL = "target_next_week_count"
THRESHOLD_COL = "abnormal_threshold_8w"
READY_COL = "final_train_ready_flag"
SPLIT_COL = "time_split"
SCORE_COL = "score"

FORMULA_ALIGNED_REMOVE = [
    "rolling_8w_mean",
    "rolling_8w_std",
    "rolling_8w_sum",
    "ratio_to_8w_mean",
]

ID_COLS = [
    "nta2020",
    "ntaname",
    "boroname",
    "week_start",
    "target_week",
    "target_year",
    "year",
    "period_type",
    "time_split",
    "complaint_category",
    "complaint_category_label",
    "complaint_count",
    TARGET_COL,
    COUNT_TARGET_COL,
    THRESHOLD_COL,
    READY_COL,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run major-revision target shortcut and count baseline audits.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--max-shap-rows", type=int, default=6000)
    parser.add_argument("--poisson-max-iter", type=int, default=120)
    parser.add_argument("--skip-lgbm", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-poisson", action="store_true")
    parser.add_argument("--save-count-predictions", action="store_true")
    parser.add_argument("--poisson-alpha", type=float, default=1.0)
    parser.add_argument(
        "--count-event-target",
        choices=["T0_current_reference", "T1_min_count_2", "T2_min_count_3"],
        default="T2_min_count_3",
        help="Event label used when evaluating count-model predictions.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=jsonable), encoding="utf-8")


def jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
    return obj


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {path}")
    df.to_csv(path, index=False, encoding="utf-8")


def load_feature_sets() -> tuple[dict[str, list[str]], list[str]]:
    config = read_json(MODEL_CONFIG)
    prospective = list(config["feature_sets"]["prospective_forecast_available"]["features"])
    categorical = list(config.get("categorical_features", []))
    no_shortcut = [f for f in prospective if f not in FORMULA_ALIGNED_REMOVE]
    reduced = [
        f
        for f in no_shortcut
        if f
        not in {
            "diff_1w_count",
            "diff_4w_count",
            "pct_change_1w",
            "ratio_to_12w_mean",
            "rolling_12w_sum",
            "rolling_4w_sum",
        }
    ]
    return {
        "A_current_prospective": prospective,
        "B_no_8w_formula_features": no_shortcut,
        "C_reduced_nonformula_history": reduced,
    }, categorical


def load_dataset(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    requested = sorted(set(ID_COLS + [f for features in feature_sets.values() for f in features]))
    header = pd.read_csv(FINAL_DATASET, nrows=0)
    usecols = [c for c in requested if c in set(header.columns)]
    df = pd.read_csv(FINAL_DATASET, usecols=usecols, low_memory=False, parse_dates=["target_week", "week_start"])
    df = df[df[READY_COL].eq(1) & df[TARGET_COL].notna() & df[COUNT_TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df[COUNT_TARGET_COL] = pd.to_numeric(df[COUNT_TARGET_COL], errors="coerce").clip(lower=0)
    df[THRESHOLD_COL] = pd.to_numeric(df[THRESHOLD_COL], errors="coerce")
    for col in df.columns:
        if col not in {"nta2020", "ntaname", "boroname", "week_start", "target_week", "period_type", "time_split", "complaint_category", "complaint_category_label"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_xy(df: pd.DataFrame, features: list[str], split: str) -> tuple[pd.DataFrame, pd.Series]:
    d = df[df[SPLIT_COL].eq(split)].copy()
    X = d[features].copy()
    y = d[TARGET_COL].astype(int)
    return X, y


def encode_train_valid_test(
    df: pd.DataFrame,
    features: list[str],
    categorical_all: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    cat_features = [f for f in features if f in categorical_all]
    num_features = [f for f in features if f not in cat_features]

    X_train_raw, y_train = split_xy(df, features, "train")
    X_val_raw, y_val = split_xy(df, features, "validation")
    X_test_raw, y_test = split_xy(df, features, "test")

    medians = {}
    for col in num_features:
        s = pd.to_numeric(X_train_raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians[col] = 0.0 if pd.isna(s.median()) else float(s.median())

    def transform(raw: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if num_features:
            parts.append(
                pd.DataFrame(
                    {
                        col: pd.to_numeric(raw[col], errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(medians[col])
                        .astype("float32")
                        for col in num_features
                    },
                    index=raw.index,
                )
            )
        for col in cat_features:
            train_levels = sorted(X_train_raw[col].astype("object").where(X_train_raw[col].notna(), "__MISSING__").astype(str).unique())
            s = raw[col].astype("object").where(raw[col].notna(), "__MISSING__").astype(str)
            dummies = pd.get_dummies(s, dtype="int8").reindex(columns=train_levels, fill_value=0)
            dummies.columns = [f"{col}={level}" for level in train_levels]
            parts.append(dummies)
        out = pd.concat(parts, axis=1)
        return out

    X_train = transform(X_train_raw)
    X_val = transform(X_val_raw).reindex(columns=X_train.columns, fill_value=0)
    X_test = transform(X_test_raw).reindex(columns=X_train.columns, fill_value=0)
    return X_train, y_train, X_val, y_val, X_test, y_test, list(X_train.columns)


def best_lgbm_params() -> dict:
    summary = read_json(LGBM_SUMMARY)
    best = summary["best_candidate_by_validation"]
    params = json.loads(best["params_json"])
    return {
        "n_estimators": int(params.get("n_estimators", 800)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "num_leaves": int(params.get("num_leaves", 31)),
        "max_depth": int(params.get("max_depth", 6)),
        "min_child_samples": int(params.get("min_child_samples", 100)),
        "subsample": float(params.get("subsample", 0.85)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
        "reg_lambda": float(params.get("reg_lambda", 3.0)),
    }


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(finite, np.linspace(0.01, 0.99, 99))]))
    return grid


def select_threshold(y: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    y_arr = y.to_numpy(dtype=int)
    for t in threshold_grid(scores):
        f1 = f1_score(y_arr, scores >= t, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def score_metrics(y: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_arr = y.to_numpy(dtype=int)
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_arr)),
        "positive_rows": int(y_arr.sum()),
        "positive_share": float(y_arr.mean()),
        "threshold": float(threshold),
        "predicted_positive_rows": int(pred.sum()),
        "alert_rate": float(pred.mean()),
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


def precision_at_k(y: np.ndarray, scores: np.ndarray, share: float) -> float:
    k = max(1, int(np.ceil(len(y) * share)))
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(y[idx]))


def train_lgbm_variants(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    categorical: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    params = best_lgbm_params()
    rows = []
    feature_rows = []
    shap_payload = {}

    for feature_set_name, features in feature_sets.items():
        t0 = time.time()
        if args.progress:
            print(f"[lgbm] {feature_set_name}: {len(features)} raw features")
        X_train, y_train, X_val, y_val, X_test, y_test, model_features = encode_train_valid_test(df, features, categorical)
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
                    "audit_family": "target_shortcut",
                    "model_name": "lightgbm",
                    "feature_set": feature_set_name,
                    "split": split,
                    "raw_feature_count": len(features),
                    "model_feature_count": len(model_features),
                    "fit_seconds": fit_seconds,
                    "elapsed_seconds": time.time() - t0,
                    "validation_selected_f1": val_f1_at_threshold,
                    "removed_formula_aligned_features": ", ".join([f for f in FORMULA_ALIGNED_REMOVE if f not in features]),
                }
            )
            rows.append(rec)

        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame(
                {
                    "feature_set": feature_set_name,
                    "feature": model_features,
                    "gain_importance": model.feature_importances_,
                }
            ).sort_values("gain_importance", ascending=False)
            imp["importance_rank"] = np.arange(1, len(imp) + 1)
            feature_rows.append(imp.head(50))

        if not args.skip_shap and shap is not None:
            shap_payload[feature_set_name] = compute_shap_summary(
                model=model,
                X_test=X_test,
                y_test=y_test,
                feature_set_name=feature_set_name,
                max_rows=args.max_shap_rows,
                random_state=args.random_state,
                out_dir=out_dir,
                overwrite=args.overwrite,
            )

    metrics_df = pd.DataFrame(rows)
    imp_df = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    return metrics_df, imp_df, shap_payload


def compute_shap_summary(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_set_name: str,
    max_rows: int,
    random_state: int,
    out_dir: Path,
    overwrite: bool,
) -> dict:
    sample = X_test.copy()
    sample["__y"] = y_test.values
    if max_rows and len(sample) > max_rows:
        parts = []
        for _, g in sample.groupby("__y"):
            n_take = max(1, int(round(max_rows * len(g) / len(sample))))
            n_take = min(n_take, len(g))
            parts.append(g.sample(n=n_take, random_state=random_state))
        sample = pd.concat(parts)
        if len(sample) > max_rows:
            sample = sample.sample(n=max_rows, random_state=random_state)
    y_sample = sample.pop("__y")
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[-1]
    imp = pd.DataFrame(
        {
            "feature_set": feature_set_name,
            "feature": sample.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_signed_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    imp["importance_rank"] = np.arange(1, len(imp) + 1)
    total = imp["mean_abs_shap"].sum()
    imp["mean_abs_shap_share"] = imp["mean_abs_shap"] / total if total else np.nan
    path = out_dir / f"shap_global_{feature_set_name}.csv"
    write_csv(path, imp, overwrite)
    return {
        "path": str(path.relative_to(ROOT)),
        "sample_rows": int(len(sample)),
        "positive_share": float(np.mean(y_sample)),
        "top_20": imp.head(20).to_dict(orient="records"),
    }


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def count_baseline_features(include_nta: bool) -> list[str]:
    features = [
        "complaint_category",
        "boroname",
        "complaint_count",
        "lag_1w_count",
        "lag_2w_count",
        "lag_4w_count",
        "lag_8w_count",
        "lag_12w_count",
        "lag_52w_count",
        "rolling_4w_mean",
        "rolling_4w_std",
        "rolling_12w_mean",
        "rolling_12w_std",
        "history_weeks_available",
        "month",
        "quarter",
        "week_of_year",
        "year",
    ]
    if include_nta:
        features.append("nta2020")
    return features


def stabilize_count_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame[features].copy()
    log_cols = [
        c
        for c in features
        if c.endswith("_count")
        or c.endswith("_mean")
        or c.endswith("_std")
        or c in {"complaint_count", "history_weeks_available"}
    ]
    for col in log_cols:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
            out[col] = np.log1p(values)
    return out


def fit_poisson_baseline(
    df: pd.DataFrame,
    categorical_all: list[str],
    include_nta: bool,
    args: argparse.Namespace,
) -> tuple[list[dict], pd.DataFrame]:
    features = [f for f in count_baseline_features(include_nta) if f in df.columns]
    categorical = [f for f in features if f in set(categorical_all + ["nta2020"])]
    numeric = [f for f in features if f not in categorical]

    # Count baselines use the final-style chronological protocol rather than
    # the older diagnostic `time_split`, whose test partition pools 2024--2025.
    train = df[df["target_year"].le(2023)].copy()
    val = df[df["target_year"].eq(2024)].copy()
    test = df[df["target_year"].eq(2025)].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Final-style Poisson split requires train through 2023, validation 2024, and test 2025 rows.")

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
    pre = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=1.0,
    )
    model = PoissonRegressor(alpha=args.poisson_alpha, max_iter=args.poisson_max_iter)
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    t0 = time.time()
    pipe.fit(stabilize_count_features(train, features), train[COUNT_TARGET_COL])
    fit_seconds = time.time() - t0

    def event_label(frame: pd.DataFrame) -> np.ndarray:
        label = frame[TARGET_COL].astype(int).to_numpy(dtype=int)
        if args.count_event_target == "T1_min_count_2":
            label = label & frame[COUNT_TARGET_COL].ge(2).to_numpy(dtype=bool)
        elif args.count_event_target == "T2_min_count_3":
            label = label & frame[COUNT_TARGET_COL].ge(3).to_numpy(dtype=bool)
        return label.astype(int)

    def formula_event(frame: pd.DataFrame, predicted_count: np.ndarray) -> np.ndarray:
        event = predicted_count > frame[THRESHOLD_COL].to_numpy(dtype=float)
        if args.count_event_target == "T1_min_count_2":
            event &= predicted_count >= 2
        elif args.count_event_target == "T2_min_count_3":
            event &= predicted_count >= 3
        return event

    pred_val = np.clip(pipe.predict(stabilize_count_features(val, features)), 0, None)
    y_val_event = event_label(val)
    validation_threshold, _ = select_threshold(pd.Series(y_val_event), pred_val)

    rows = []
    pred_frames = []
    model_name = "poisson_regressor_nta_fe" if include_nta else "poisson_regressor_no_nta"
    for split_name, d in [("validation", val), ("test", test)]:
        pred_count = np.clip(pipe.predict(stabilize_count_features(d, features)), 0, None)
        y_event = event_label(d)
        y_count = d[COUNT_TARGET_COL].to_numpy(dtype=float)
        formula_decision = formula_event(d, pred_count)
        val_event = pred_count >= validation_threshold
        for mode, event_pred, hard_threshold in [
            ("formula_threshold", formula_decision, np.nan),
            ("validation_score_threshold", val_event, validation_threshold),
        ]:
            event_scores = pred_count
            tn, fp, fn, tp = confusion_matrix(y_event, event_pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "audit_family": "count_baseline",
                    "target_definition": args.count_event_target,
                    "model_name": model_name,
                    "decision_mode": mode,
                    "split": split_name,
                    "rows": len(d),
                    "positive_rows": int(y_event.sum()),
                    "positive_share": float(y_event.mean()),
                    "feature_count": len(features),
                    "categorical_features": ", ".join(categorical),
                    "fit_seconds": fit_seconds,
                    "poisson_alpha": args.poisson_alpha,
                    "poisson_max_iter": args.poisson_max_iter,
                    "poisson_n_iter": int(getattr(pipe.named_steps["model"], "n_iter_", -1)),
                    "count_mae": float(mean_absolute_error(y_count, pred_count)),
                    "mean_observed_count": float(np.mean(y_count)),
                    "mean_predicted_count": float(np.mean(pred_count)),
                    "poisson_deviance": safe_poisson_deviance(y_count, pred_count),
                    "threshold": float(hard_threshold) if np.isfinite(hard_threshold) else np.nan,
                    "alert_rate": float(np.mean(event_pred)),
                    "precision": float(precision_score(y_event, event_pred, zero_division=0)),
                    "recall": float(recall_score(y_event, event_pred, zero_division=0)),
                    "f1": float(f1_score(y_event, event_pred, zero_division=0)),
                    "pr_auc": float(average_precision_score(y_event, event_scores)) if len(np.unique(y_event)) > 1 else np.nan,
                    "roc_auc": float(roc_auc_score(y_event, event_scores)) if len(np.unique(y_event)) > 1 else np.nan,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "precision_at_1pct": precision_at_k(y_event, event_scores, 0.01),
                    "precision_at_5pct": precision_at_k(y_event, event_scores, 0.05),
                    "precision_at_10pct": precision_at_k(y_event, event_scores, 0.10),
                }
            )
        pred_frame = d[ID_COLS].copy()
        pred_frame["split"] = split_name
        pred_frame["target_definition"] = args.count_event_target
        pred_frame["event_label"] = y_event
        pred_frame["model_name"] = model_name
        pred_frame["predicted_count"] = pred_count
        pred_frames.append(pred_frame)
    return rows, pd.concat(pred_frames, ignore_index=True)


def safe_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(mean_poisson_deviance(y_true, np.clip(y_pred, 1e-9, None)))
    except Exception:
        return np.nan


def add_delta_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    shortcut = metrics[metrics["audit_family"].eq("target_shortcut")].copy()
    for split, split_df in shortcut.groupby("split"):
        base = split_df[split_df["feature_set"].eq("A_current_prospective")]
        if base.empty:
            continue
        base_vals = base.iloc[0]
        mask = metrics["audit_family"].eq("target_shortcut") & metrics["split"].eq(split)
        for col in ["f1", "pr_auc", "precision", "recall", "precision_at_5pct"]:
            metrics.loc[mask, f"delta_vs_current_{col}"] = metrics.loc[mask, col] - float(base_vals[col])
    return metrics


def make_report(
    lgbm_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    shap_payload: dict[str, object],
    count_metrics: pd.DataFrame,
    out_dir: Path,
) -> str:
    lines = [
        "# Target Shortcut and Count Baseline Audit",
        "",
        "This report is generated from actual model runs for the major revision. It is still an audit artifact, not the final model-selection report.",
        "",
        "## LightGBM Target-Shortcut Configurations",
        "",
        md_table(
            lgbm_metrics[lgbm_metrics["split"].eq("test")],
            [
                "feature_set",
                "raw_feature_count",
                "model_feature_count",
                "threshold",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
                "precision_at_5pct",
                "delta_vs_current_f1",
                "delta_vs_current_pr_auc",
            ],
        ),
        "",
        "Formula-aligned features removed in the no-shortcut configuration: `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean`.",
        "",
        "## Top Gain Importances",
        "",
        md_table(feature_importance.groupby("feature_set").head(12), ["feature_set", "importance_rank", "feature", "gain_importance"]),
        "",
        "## SHAP Global Importance Files",
        "",
    ]
    if shap_payload:
        for name, payload in shap_payload.items():
            lines.append(f"- `{name}`: `{payload['path']}`, sample rows = {payload['sample_rows']}")
    else:
        lines.append("- SHAP was skipped or unavailable.")
    lines.extend(
        [
            "",
            "## Count Baselines",
            "",
            md_table(
                count_metrics[count_metrics["split"].eq("test")] if not count_metrics.empty and "split" in count_metrics.columns else count_metrics,
                [
                    "model_name",
                    "decision_mode",
                    "feature_count",
                    "count_mae",
                    "poisson_deviance",
                    "mean_observed_count",
                    "mean_predicted_count",
                    "precision",
                    "recall",
                    "f1",
                    "pr_auc",
                    "precision_at_5pct",
                ],
            ),
            "",
            "## Guardrails",
            "",
            "- These runs do not use OSM/PLUTO features.",
            "- The no-shortcut model removes direct 8-week target-formula predictors but still predicts the current submitted target.",
        "- Count baselines predict `target_next_week_count`; event metrics are evaluated on the stated event target with a formula-threshold conversion and a validation-selected score threshold.",
            "- Final target/model selection still requires rolling-origin validation, uncertainty intervals, and target-definition sensitivity.",
            "",
        ]
    )
    report = "\n".join(lines)
    (out_dir / "model_audit_report.md").write_text(report, encoding="utf-8")
    return report


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    d = df[[c for c in columns if c in df.columns]].head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            d[c] = d[c].astype(str)
    return "\n".join(
        [
            "| " + " | ".join(d.columns) + " |",
            "| " + " | ".join(["---"] * len(d.columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in d.to_numpy(dtype=str)],
        ]
    )


def main() -> None:
    args = parse_args()
    t_start = time.time()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_sets, categorical = load_feature_sets()
    df = load_dataset(feature_sets)
    if args.progress:
        print(f"[data] rows={len(df):,}; splits={df[SPLIT_COL].value_counts().to_dict()}")

    if args.skip_lgbm:
        lgbm_metrics_path = out_dir / "target_shortcut_results.csv"
        feature_importance_path = out_dir / "target_shortcut_feature_importance.csv"
        if not lgbm_metrics_path.exists() or not feature_importance_path.exists():
            raise FileNotFoundError("--skip-lgbm requires existing target shortcut output files")
        lgbm_metrics = pd.read_csv(lgbm_metrics_path)
        feature_importance = pd.read_csv(feature_importance_path)
        shap_payload = {}
        for path in out_dir.glob("shap_global_*.csv"):
            name = path.stem.replace("shap_global_", "")
            shap_payload[name] = {"path": str(path.relative_to(ROOT)), "sample_rows": "existing"}
    else:
        lgbm_metrics, feature_importance, shap_payload = train_lgbm_variants(df, feature_sets, categorical, args, out_dir)
        lgbm_metrics = add_delta_columns(lgbm_metrics)
        write_csv(out_dir / "target_shortcut_results.csv", lgbm_metrics, args.overwrite)
        write_csv(out_dir / "target_shortcut_feature_importance.csv", feature_importance, args.overwrite)

    count_predictions = []
    if args.skip_poisson:
        count_metrics_path = out_dir / "count_model_baseline_results.csv"
        if not count_metrics_path.exists():
            raise FileNotFoundError("--skip-poisson requires existing count baseline output files")
        count_metrics = pd.read_csv(count_metrics_path)
    else:
        count_rows = []
        for include_nta in [False, True]:
            if args.progress:
                print(f"[poisson] include_nta={include_nta}")
            rows, preds = fit_poisson_baseline(df, categorical, include_nta, args)
            count_rows.extend(rows)
            count_predictions.append(preds)
        count_metrics = pd.DataFrame(count_rows)
        write_csv(out_dir / "count_model_baseline_results.csv", count_metrics, args.overwrite)
        count_predictions_df = pd.concat(count_predictions, ignore_index=True) if count_predictions else pd.DataFrame()
        if args.save_count_predictions and not count_predictions_df.empty:
            write_csv(out_dir / "count_model_predictions_validation_test.csv.gz", count_predictions_df, args.overwrite)

    make_report(lgbm_metrics, feature_importance, shap_payload, count_metrics, out_dir)
    run_summary = {
        "status": "done",
        "script": "major_revision_model_audits.py",
        "rows_loaded": int(len(df)),
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "formula_aligned_removed": FORMULA_ALIGNED_REMOVE,
        "count_event_target": args.count_event_target,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "elapsed_seconds": round(time.time() - t_start, 3),
    }
    write_json(out_dir / "model_audit_run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, default=jsonable))


if __name__ == "__main__":
    main()
