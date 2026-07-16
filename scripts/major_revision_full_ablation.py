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
from major_revision_rolling_origin import assign_fold_split  # noqa: E402
from major_revision_target_selection import TARGET_DEFINITIONS, encode_splits, make_targets, md_table  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/ablations"
ROOT_REPORT = ROOT / "full_training_ablation_report.md"
FINAL_FOLD = {"fold_id": "final_style_2025", "train_end_year": 2023, "validation_year": 2024, "test_year": 2025}


CALENDAR_FEATURES = ["year", "month", "quarter", "week_of_year", "iso_year", "is_year_start", "is_year_end"]
HISTORY_LAG_FEATURES = [
    "complaint_category",
    "history_weeks_available",
    "lag_1w_count",
    "lag_2w_count",
    "lag_4w_count",
    "lag_8w_count",
    "lag_12w_count",
    "lag_52w_count",
]
HISTORY_SUMMARY_FEATURES = [
    "complaint_count",
    "diff_1w_count",
    "diff_4w_count",
    "pct_change_1w",
    "ratio_to_12w_mean",
    "rolling_4w_mean",
    "rolling_4w_std",
    "rolling_4w_sum",
    "rolling_12w_mean",
    "rolling_12w_std",
    "rolling_12w_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-training ablation and weather checks for the major revision.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--target", default="T2_min_count_3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def unique_existing(features: list[str], available: set[str]) -> list[str]:
    seen = set()
    out = []
    for f in features:
        if f in available and f not in seen:
            out.append(f)
            seen.add(f)
    return out


def is_weather_feature(name: str) -> bool:
    return (
        name.startswith("weather_")
        or name.endswith("_day_count")
        or name in {"rain_day_count", "snow_day_count", "hot_day_30c_count", "hot_day_35c_count"}
    )


def build_feature_configs(prospective: list[str], available: set[str]) -> dict[str, list[str]]:
    no_shortcut = [f for f in prospective if f not in FORMULA_ALIGNED_REMOVE]
    weather = [f for f in no_shortcut if is_weather_feature(f)]
    base_lags = HISTORY_LAG_FEATURES
    hist_current = base_lags + ["complaint_count"]
    hist_calendar = hist_current + CALENDAR_FEATURES
    hist_calendar_weather = hist_calendar + weather
    return {
        "01_history_lags_only": unique_existing(base_lags, available),
        "02_history_current_count": unique_existing(hist_current, available),
        "03_history_calendar": unique_existing(hist_calendar, available),
        "04_history_calendar_weather": unique_existing(hist_calendar_weather, available),
        "05_history_calendar_weather_borough": unique_existing(hist_calendar_weather + ["boroname"], available),
        "06_history_calendar_weather_nta_fe": unique_existing(hist_calendar_weather + ["nta2020"], available),
        "07_final_no_shortcut_borough": unique_existing(no_shortcut, available),
        "08_final_no_shortcut_nta_fe": unique_existing(no_shortcut + ["nta2020"], available),
        "09_calendar_only": unique_existing(["complaint_category"] + CALENDAR_FEATURES, available),
        "10_weather_only": unique_existing(["complaint_category"] + weather, available),
        "11_calendar_weather_only": unique_existing(["complaint_category"] + CALENDAR_FEATURES + weather, available),
    }


def score_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
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
    }


def run_config(
    target_name: str,
    frame: pd.DataFrame,
    config_name: str,
    features: list[str],
    categorical: list[str],
    seed: int,
    progress: bool,
) -> list[dict[str, object]]:
    d = assign_fold_split(frame, FINAL_FOLD)
    if progress:
        print(f"[ablation] {config_name}: {len(features)} raw features; splits={d[SPLIT_COL].value_counts().to_dict()}")
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
    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    threshold, val_f1 = select_threshold(y_val, val_score)
    rows = []
    for split, y, score in [
        ("validation", y_val.to_numpy(dtype=int), val_score),
        ("test", y_test.to_numpy(dtype=int), test_score),
    ]:
        rec = score_metrics(y, score, threshold)
        rec.update(
            {
                "target_definition": target_name,
                "description": TARGET_DEFINITIONS[target_name]["description"],
                "feature_config": config_name,
                "split": split,
                "seed": seed,
                "raw_feature_count": len(features),
                "model_feature_count": len(model_features),
                "fit_seconds": fit_seconds,
                "validation_selected_f1": val_f1,
                "contains_weather": bool(any(is_weather_feature(f) for f in features)),
                "contains_borough": "boroname" in features,
                "contains_nta_fe": "nta2020" in features,
                "contains_osm_pluto": bool(any(f.startswith("poi_") or f.startswith("osm_") or f.startswith("pluto_") for f in features)),
                "removed_formula_aligned_features": ", ".join(FORMULA_ALIGNED_REMOVE),
            }
        )
        rows.append(rec)
    return rows


def add_delta_columns(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    for split, g in out.groupby("split"):
        base = g[g["feature_config"].eq("03_history_calendar")]
        final = g[g["feature_config"].eq("07_final_no_shortcut_borough")]
        nta = g[g["feature_config"].eq("08_final_no_shortcut_nta_fe")]
        if not base.empty:
            for col in ["pr_auc", "f1", "precision_at_5pct"]:
                out.loc[out["split"].eq(split), f"delta_vs_history_calendar_{col}"] = out.loc[out["split"].eq(split), col] - float(base[col].iloc[0])
        if not final.empty:
            for col in ["pr_auc", "f1", "precision_at_5pct"]:
                out.loc[out["split"].eq(split), f"delta_vs_final_borough_{col}"] = out.loc[out["split"].eq(split), col] - float(final[col].iloc[0])
        if not nta.empty:
            for col in ["pr_auc", "f1", "precision_at_5pct"]:
                out.loc[out["split"].eq(split), f"delta_vs_final_nta_{col}"] = out.loc[out["split"].eq(split), col] - float(nta[col].iloc[0])
    return out


def write_report(out_dir: Path, results: pd.DataFrame, target: str, seed: int) -> str:
    val = results[results["split"].eq("validation")].sort_values("pr_auc", ascending=False)
    test = results[results["split"].eq("test")]
    weather_rows = val[val["feature_config"].isin(["03_history_calendar", "04_history_calendar_weather", "09_calendar_only", "10_weather_only", "11_calendar_weather_only"])]
    lines = [
        "# Full-Training Ablation Report",
        "",
        f"This report runs a full-row final-style ablation for `{target}` with seed `{seed}`. It replaces the old compact 300k protocol for this pass, but it is not yet a five-seed ablation table.",
        "",
        "All configurations exclude OSM/PLUTO and remove formula-aligned 8-week shortcut features.",
        "",
        "## Validation Metrics for Selection",
        "",
        md_table(
            val,
            [
                "feature_config",
                "raw_feature_count",
                "model_feature_count",
                "contains_weather",
                "contains_borough",
                "contains_nta_fe",
                "pr_auc",
                "precision_at_5pct",
                "f1",
                "precision",
                "recall",
                "threshold",
                "fit_seconds",
            ],
            max_rows=80,
        ),
        "",
        "## Held-Out 2025 Diagnostics",
        "",
        md_table(
            test.sort_values("feature_config"),
            [
                "feature_config",
                "pr_auc",
                "precision_at_5pct",
                "f1",
                "precision",
                "recall",
                "threshold",
                "alert_rate",
            ],
            max_rows=80,
        ),
        "",
        "## Weather-Specific Validation Rows",
        "",
        md_table(
            weather_rows.sort_values("feature_config"),
            [
                "feature_config",
                "pr_auc",
                "precision_at_5pct",
                "f1",
                "delta_vs_history_calendar_pr_auc",
                "delta_vs_history_calendar_precision_at_5pct",
                "delta_vs_history_calendar_f1",
            ],
            max_rows=80,
        ),
        "",
        "## Guardrails",
        "",
        "- Feature selection should use validation/backtest evidence, not 2025 test metrics.",
        "- Weather here is city-level feature-week observed Central Park exposure, not NTA-level weather and not t+1 forecast weather.",
        "- Historical NWS forecast archives and spatial weather grids remain Future Work because they require external data not collected in this revision.",
        "- This pass uses one seed for all rows; five-seed ablation for final/key rows remains open.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "full_training_ablation_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    feature_sets, categorical_base = load_feature_sets()
    prospective = feature_sets["B_no_8w_formula_features"]
    header = pd.read_csv(ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz", nrows=0)
    available = set(header.columns)
    configs = build_feature_configs(prospective, available)
    categorical = sorted(set(categorical_base + ["complaint_category", "boroname", "nta2020"]))
    required_features = sorted(set(f for features in configs.values() for f in features) | {"rolling_8w_mean"})
    df = load_dataset({"ablation_features": required_features})
    if args.target not in TARGET_DEFINITIONS:
        raise ValueError(f"Unknown target: {args.target}")
    frame = make_targets(df)[args.target]
    rows = []
    for config_name, features in configs.items():
        rows.extend(run_config(args.target, frame, config_name, features, categorical, args.seed, args.progress))
    results = add_delta_columns(pd.DataFrame(rows))
    write_csv(out_dir / "full_training_ablation_results.csv", results, args.overwrite)
    feature_manifest = pd.DataFrame(
        [
            {
                "feature_config": name,
                "raw_feature_count": len(features),
                "features": ", ".join(features),
                "contains_weather": any(is_weather_feature(f) for f in features),
                "contains_borough": "boroname" in features,
                "contains_nta_fe": "nta2020" in features,
            }
            for name, features in configs.items()
        ]
    )
    write_csv(out_dir / "full_training_ablation_feature_manifest.csv", feature_manifest, args.overwrite)
    write_report(out_dir, results, args.target, args.seed)
    write_json(
        out_dir / "full_training_ablation_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_full_ablation.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "target": args.target,
            "fold": FINAL_FOLD,
            "seed": args.seed,
            "feature_configs": list(configs.keys()),
            "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        },
    )


if __name__ == "__main__":
    main()
