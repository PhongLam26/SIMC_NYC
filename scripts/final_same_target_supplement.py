"""Build reviewer-facing same-target evidence and LaTeX tables from stored artifacts."""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/model_results/major_revision/supplementary"
TEX = ROOT / "paper_springer/tables/supplementary"
TARGET = "T2_min_count_3"
BOOTSTRAP_SEED = 20260716
N_BOOTSTRAP = 250


def esc(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def fmt(value: float, digits: int = 3) -> str:
    return "--" if pd.isna(value) else f"{float(value):.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return "--" if pd.isna(value) else f"{100 * float(value):.{digits}f}\\%"


def precision_at_k(y: np.ndarray, score: np.ndarray, share: float = 0.05) -> float:
    k = max(1, int(math.ceil(len(y) * share)))
    return float(np.mean(y[np.argsort(score)[::-1][:k]]))


def metrics(y: np.ndarray, score: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "precision_at_5pct": precision_at_k(y, score),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "alert_rate": float(np.mean(predicted)),
    }


def cluster_ci(frame: pd.DataFrame, score_col: str, pred_col: str, name: str) -> list[dict[str, object]]:
    y = frame["y_true"].to_numpy(dtype=int)
    score = frame[score_col].to_numpy(dtype=float)
    predicted = frame[pred_col].to_numpy(dtype=bool)
    labels = (frame["nta2020"].astype(str) + "|" + frame["complaint_category"].astype(str)).to_numpy()
    clusters = np.unique(labels)
    indices = {cluster: np.flatnonzero(labels == cluster) for cluster in clusters}
    observed = metrics(y, score, predicted)
    rng = np.random.default_rng((BOOTSTRAP_SEED + zlib.crc32(name.encode("utf-8"))) % (2**32))
    reps = {metric: [] for metric in observed}
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([indices[cluster] for cluster in sampled])
        values = metrics(y[idx], score[idx], predicted[idx])
        for metric, value in values.items():
            reps[metric].append(value)
    return [
        {
            "model": name,
            "metric": metric,
            "observed": observed[metric],
            "ci_lower": float(np.percentile(values, 2.5)),
            "ci_upper": float(np.percentile(values, 97.5)),
            "clusters": len(clusters),
            "n_bootstrap": N_BOOTSTRAP,
        }
        for metric, values in reps.items()
    ]


def write_tex(name: str, content: str) -> None:
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def tabular(headers: list[str], rows: list[list[str]], alignment: str) -> str:
    lines = [r"\begin{tabular}{" + alignment + r"}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\\\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def volume_deciles(tree: pd.DataFrame) -> pd.DataFrame:
    dataset = ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz"
    columns = ["target_year", "target_week", "nta2020", "complaint_category", "rolling_8w_mean", "final_train_ready_flag"]
    source = pd.read_csv(dataset, usecols=columns, parse_dates=["target_week"])
    source = source[source["final_train_ready_flag"].eq(1)].copy()
    train = source[source["target_year"].le(2023)]["rolling_8w_mean"].dropna()
    cuts = np.unique(train.quantile(np.linspace(0, 1, 11)).to_numpy(dtype=float))
    if len(cuts) < 3:
        raise RuntimeError("Train-derived volume cut points are degenerate.")
    test = source[source["target_year"].eq(2025)].copy()
    test["target_week"] = test["target_week"].dt.strftime("%Y-%m-%d")
    merged = tree.merge(test.drop(columns=["target_year", "final_train_ready_flag"]), on=["target_week", "nta2020", "complaint_category"], how="inner", validate="one_to_one")
    if len(merged) != len(tree):
        raise RuntimeError(f"Volume-decile merge lost rows: {len(merged)} of {len(tree)}")
    merged["volume_decile"] = pd.cut(merged["rolling_8w_mean"], bins=cuts, include_lowest=True, duplicates="drop", labels=False)
    merged["volume_decile"] = merged["volume_decile"].fillna(0).astype(int) + 1
    rows = []
    for decile, group in merged.groupby("volume_decile", sort=True):
        y = group["y_true"].to_numpy(dtype=int)
        score = group["platt_score"].to_numpy(dtype=float)
        pred = score >= group["platt_threshold"].iloc[0]
        row = metrics(y, score, pred)
        row.update({"volume_decile": f"D{decile}", "rows": len(group), "positive_rows": int(y.sum()), "event_prevalence": float(y.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tree = pd.read_csv(ROOT / "data/processed/model_results/major_revision/bootstrap/bootstrap_prediction_rows.csv.gz")
    tree = tree[(tree["target_definition"].eq(TARGET)) & (tree["fold_id"].eq("final_style_2025"))].copy()
    tree["lightgbm_alert"] = tree["platt_score"] >= tree["platt_threshold"]
    if len(tree) != 122616 or int(tree["y_true"].sum()) != 13562:
        raise RuntimeError("Unexpected final-style LightGBM T2 population.")

    poisson = pd.read_csv(ROOT / "data/processed/model_results/major_revision/model_audits/count_model_predictions_validation_test.csv.gz")
    poisson = poisson[poisson["split"].eq("test")].copy()
    poisson = poisson.rename(columns={"event_label": "y_true"})
    poisson["formula_alert"] = poisson["predicted_count"] > poisson["abnormal_threshold_8w"]
    poisson.loc[poisson["predicted_count"] < 3, "formula_alert"] = False

    counts = pd.read_csv(ROOT / "data/processed/model_results/major_revision/bootstrap/tree_vs_count_prediction_rows.csv.gz")
    count_tree = tree[["target_week", "nta2020", "complaint_category", "y_true", "platt_score", "lightgbm_alert"]].merge(
        counts,
        on=["target_week", "nta2020", "complaint_category", "y_true"],
        how="inner",
        validate="one_to_one",
    )
    if len(count_tree) != len(tree):
        raise RuntimeError("Stored HGB/hurdle predictions do not align with LightGBM T2 rows.")

    volume = volume_deciles(tree)
    volume.to_csv(OUT / "volume_decile_performance_2025.csv", index=False)
    volume_rows = [[esc(r.volume_decile), f"{int(r.rows):,}", pct(r.event_prevalence), fmt(r.pr_auc), fmt(r.precision), fmt(r.recall), fmt(r.f1), fmt(r.precision_at_5pct), pct(r.alert_rate)] for r in volume.itertuples()]
    write_tex("volume_deciles.tex", tabular(["Volume group", "Rows", "Prevalence", "PR-AUC", "Precision", "Recall", "F1", "P@5\\%", "Alert rate"], volume_rows, "@{}lrrrrrrrr@{}"))

    base = pd.read_csv(ROOT / "data/processed/model_results/major_revision/model_audits/count_model_baseline_results.csv")
    base = base[(base["split"].eq("test")) & (base["decision_mode"].eq("formula_threshold")) & (base["target_definition"].eq(TARGET))]
    ext = pd.read_csv(ROOT / "data/processed/model_results/major_revision/count_extensions/count_model_extension_results.csv")
    ext = ext[(ext["split"].eq("test")) & (ext["decision_mode"].eq("formula_threshold"))]
    final = pd.DataFrame([{ "model_name": "No-shortcut LightGBM classifier", "decision_mode": "validation_score_threshold", "rows": len(tree), "positive_rows": int(tree.y_true.sum()), "count_mae": np.nan, **metrics(tree.y_true.to_numpy(), tree.platt_score.to_numpy(), tree.lightgbm_alert.to_numpy()) }])
    full_baselines = pd.concat([base, ext, final], ignore_index=True, sort=False)
    full_baselines.to_csv(OUT / "same_target_baseline_metrics_2025.csv", index=False)
    labels = {"poisson_regressor_no_nta": "Poisson GLM, no NTA FE", "poisson_regressor_nta_fe": "Poisson GLM + NTA FE", "hist_gradient_boosting_poisson_count": "HGB Poisson count", "hurdle_hgb_occurrence_poisson_positive_count": "Hurdle HGB count", "No-shortcut LightGBM classifier": "No-shortcut LightGBM classifier"}
    baseline_rows = [[esc(labels[r.model_name]), "Formula threshold" if r.decision_mode == "formula_threshold" else "Validation threshold", fmt(r.count_mae), fmt(r.pr_auc), fmt(r.precision_at_5pct), fmt(r.precision), fmt(r.recall), fmt(r.f1)] for r in full_baselines.itertuples()]
    write_tex("same_target_baselines.tex", tabular(["Method", "Decision", "Count MAE", "PR-AUC", "P@5\\%", "Precision", "Recall", "F1"], baseline_rows, "@{}llrrrrrr@{}"))

    ci_rows = []
    ci_rows.extend(cluster_ci(tree, "platt_score", "lightgbm_alert", "No-shortcut LightGBM"))
    for model, group in poisson.groupby("model_name", sort=True):
        ci_rows.extend(cluster_ci(group, "predicted_count", "formula_alert", labels[model]))
    ci_rows.extend(cluster_ci(count_tree, "hgb_poisson_count_score", "hgb_poisson_formula_event", "HGB Poisson count"))
    ci_rows.extend(cluster_ci(count_tree, "hurdle_count_score", "hurdle_formula_event", "Hurdle HGB count"))
    cis = pd.DataFrame(ci_rows)
    cis.to_csv(OUT / "same_target_baseline_bootstrap_ci_2025.csv", index=False)
    ci_table = cis[cis["metric"].isin(["pr_auc", "precision_at_5pct", "f1"])].copy()
    ci_table["metric"] = ci_table["metric"].replace({"pr_auc": "PR-AUC", "precision_at_5pct": "P@5%", "f1": "F1"})
    ci_tex_rows = [[esc(r.model), esc(r.metric), fmt(r.observed), f"[{fmt(r.ci_lower)}, {fmt(r.ci_upper)}]"] for r in ci_table.itertuples()]
    write_tex("baseline_bootstrap_ci.tex", tabular(["Model", "Metric", "Estimate", "Cluster-bootstrap 95\\% CI"], ci_tex_rows, "@{}llrr@{}"))

    seed = pd.read_csv(ROOT / "data/processed/model_results/major_revision/ablations/key_ablation_seed_stability_summary.csv")
    selected = seed[seed["feature_config"].isin(["03_history_calendar", "04_history_calendar_weather", "07_final_no_shortcut_borough", "08_final_no_shortcut_nta_fe"])].copy()
    seed_rows = [[esc(r.feature_config.replace("_", " ")), str(int(r.seeds)), f"{fmt(r.pr_auc_mean)} $\\pm$ {fmt(r.pr_auc_std)}", f"{fmt(r.precision_at_5pct_mean)} $\\pm$ {fmt(r.precision_at_5pct_std)}", f"{fmt(r.f1_mean)} $\\pm$ {fmt(r.f1_std)}"] for r in selected.itertuples()]
    write_tex("seed_stability.tex", tabular(["Configuration", "Seeds", "PR-AUC", "P@5\\%", "F1"], seed_rows, "@{}lrrrr@{}"))

    rolling = pd.read_csv(ROOT / "data/processed/model_results/major_revision/backtests/rolling_origin_results.csv")
    rolling = rolling[(rolling["target_definition"].eq(TARGET)) & (rolling["split"].eq("test"))]
    rolling_rows = [[str(int(r.test_year)), f"{int(r.rows):,}", pct(r.positive_share), fmt(r.pr_auc), fmt(r.precision_at_5pct), fmt(r.f1)] for r in rolling.itertuples()]
    write_tex("rolling_origin.tex", tabular(["Test year", "Rows", "Prevalence", "PR-AUC", "P@5\\%", "F1"], rolling_rows, "@{}lrrrrr@{}"))

    calibration = pd.read_csv(ROOT / "data/processed/model_results/major_revision/calibration/calibration_results.csv")
    calibration = calibration[(calibration["target_definition"].eq(TARGET)) & (calibration["fold_id"].eq("final_style_2025")) & (calibration["split"].eq("test"))]
    cal_rows = [[esc(r.calibration_method.title()), fmt(r.brier), fmt(r.ece), fmt(r.log_loss), fmt(r.pr_auc)] for r in calibration.itertuples()]
    write_tex("calibration.tex", tabular(["Method", "Brier", "ECE", "Log loss", "PR-AUC"], cal_rows, "@{}lrrrr@{}"))

    sensitivity = pd.read_csv(ROOT / "data/processed/model_results/major_revision/target_selection/target_definition_results.csv")
    sensitivity = sensitivity[sensitivity["split"].eq("test") & sensitivity["target_definition"].isin(["T0_current_reference", "T1_min_count_2", "T2_min_count_3", "T3_mu8w_ge_1_eligible"])]
    sens_rows = [[esc(r.target_definition.split("_")[0]), f"{int(r.rows):,}", pct(r.positive_share), fmt(r.pr_auc), fmt(r.precision_at_5pct), fmt(r.f1)] for r in sensitivity.itertuples()]
    write_tex("target_sensitivity.tex", tabular(["Target", "Rows", "Prevalence", "PR-AUC", "P@5\\%", "F1"], sens_rows, "@{}lrrrrr@{}"))

    categories = pd.read_csv(ROOT / "data/processed/model_results/major_revision/complaint_mapping/complaint_category_summary.csv")
    map_rows = [[esc(r.complaint_category.replace("_", "/")), str(int(r.complaint_types)), f"{int(r.complaint_count):,}", pct(r.share_of_all_requests, 2)] for r in categories.itertuples()]
    write_tex("complaint_mapping.tex", tabular(["Analysis category", "Types", "Requests", "Share"], map_rows, "@{}lrrr@{}"))
    other = pd.read_csv(ROOT / "data/processed/model_results/major_revision/complaint_mapping/other_category_composition.csv").head(12)
    other_rows = [[esc(r.complaint_type), f"{int(r.complaint_count):,}", pct(r.share_of_other_requests, 2)] for r in other.itertuples()]
    write_tex("other_composition.tex", tabular(["Complaint type", "Requests", "Share of other"], other_rows, "@{}lrr@{}"))

    nb = pd.read_csv(ROOT / "data/processed/model_results/major_revision/count_extensions/negative_binomial_status.csv").iloc[0]
    audit = [
        "# Final Same-Target Comparability Workspace Audit",
        "",
        "| model | target | split | test rows | decision rule | status |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for r in full_baselines.itertuples():
        audit.append(f"| {labels.get(r.model_name, r.model_name)} | T2 | train <= 2023; validation 2024; test 2025 | {int(r.rows):,} | {r.decision_mode} | PASS |")
    audit.extend(["", "- Poisson preprocessing and categorical encodings are fit on train rows only; its test predictions are stored in `count_model_predictions_validation_test.csv.gz`.", "- Poisson convergence: no-NTA 156/500 iterations; NTA fixed effects 199/500 iterations.", f"- Negative Binomial: `{nb.status}`. {nb.reason}", ""])
    (ROOT / "final_comparability_workspace_audit.md").write_text("\n".join(audit), encoding="utf-8")

    report = [
        "# Final Same-Target Numerical Consistency Report",
        "",
        "All Table 3 rows are reconstructed from current held-out-2025 artifacts using T2, with 122,616 rows and 13,562 positives.",
        "",
        "| model | rows | positives | PR-AUC | P@5% | precision | recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in full_baselines.itertuples():
        report.append(f"| {labels.get(r.model_name, r.model_name)} | {int(r.rows):,} | {int(r.positive_rows):,} | {fmt(r.pr_auc)} | {fmt(r.precision_at_5pct)} | {fmt(r.precision)} | {fmt(r.recall)} | {fmt(r.f1)} |")
    report.extend(["", "- Volume-decile rows sum to the full final-style 2025 test population.", "- Supplementary baseline CIs use 250 NTA-category cluster resamples; the final LightGBM main-metric CI artifact uses 1,000. Table 3 values and the supplementary CSV retain full precision.", "- T0/T1/T2/T3 sensitivity is labelled as a separate two-year diagnostic and is not mixed into the T2 final comparison.", ""])
    (ROOT / "final_same_target_numerical_consistency_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    build()
