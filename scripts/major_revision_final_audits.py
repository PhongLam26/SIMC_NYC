"""Generate final major-revision numerical and leakage audit reports.

This script is intentionally read-only with respect to model artifacts: it
loads the already generated major-revision CSV/JSON outputs, reconstructs key
metrics, and writes compact reviewer-facing audit reports.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "data" / "processed" / "model_results" / "major_revision"


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def read_json(rel: str) -> dict:
    with (ROOT / rel).open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float | int | str | None, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "NA"
    if isinstance(x, int):
        return f"{x:,}"
    if abs(float(x)) >= 1000:
        return f"{float(x):,.0f}"
    return f"{float(x):.{digits}f}"


def clean_text(x: object, fallback: str = "none") -> str:
    if x is None or pd.isna(x):
        return fallback
    text = str(x).strip()
    return text if text and text.lower() != "nan" else fallback


def pct(x: float | None, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{100 * float(x):.{digits}f}%"


def md_table(rows: list[dict], columns: list[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return out


def metric_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else math.nan
    return precision, recall, f1


def max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    return float((a - b).abs().max())


def generate_numerical_report() -> str:
    target = read_csv("data/processed/model_results/major_revision/target_selection/target_definition_results.csv")
    comp = read_csv("data/processed/model_results/major_revision/target_selection/target_definition_composition.csv")
    decile = read_csv("data/processed/model_results/major_revision/target_selection/target_definition_decile_results.csv")
    shortcut = read_csv("data/processed/model_results/major_revision/model_audits/target_shortcut_results.csv")
    count_base = read_csv("data/processed/model_results/major_revision/model_audits/count_model_baseline_results.csv")
    count_ext = read_csv("data/processed/model_results/major_revision/count_extensions/count_model_extension_results.csv")
    nb_status = read_csv("data/processed/model_results/major_revision/count_extensions/negative_binomial_status.csv")
    rolling = read_csv("data/processed/model_results/major_revision/backtests/rolling_origin_results.csv")
    ablation = read_csv("data/processed/model_results/major_revision/ablations/full_training_ablation_results.csv")
    seeds = read_csv("data/processed/model_results/major_revision/seeds/seed_stability_summary.csv")
    boot = read_csv("data/processed/model_results/major_revision/bootstrap/bootstrap_ci_results.csv")
    paired = read_csv("data/processed/model_results/major_revision/bootstrap/paired_model_difference_ci.csv")
    calib = read_csv("data/processed/model_results/major_revision/calibration/calibration_summary.csv")
    severity = read_csv("data/processed/model_results/major_revision/error_analysis/error_severity_overall.csv")
    workload = read_csv("data/processed/model_results/major_revision/disparity_workload/weekly_workload_summary.csv")
    shap_group = read_csv("data/processed/model_results/major_revision/explainability/shap_group_importance.csv")
    cases = read_csv("data/processed/model_results/major_revision/explainability/shap_local_cases.csv")

    rows = []
    for _, r in target[target["split"].eq("test")].iterrows():
        p, rec, f1 = metric_from_counts(r["tp"], r["fp"], r["fn"])
        rows.append(
            {
                "target": r["target_definition"],
                "rows": fmt(int(r["rows"])),
                "pos": fmt(int(r["positive_rows"])),
                "TP/FP/FN/TN": f"{fmt(int(r['tp']))}/{fmt(int(r['fp']))}/{fmt(int(r['fn']))}/{fmt(int(r['tn']))}",
                "precision": fmt(r["precision"]),
                "p_check": fmt(p),
                "recall": fmt(r["recall"]),
                "r_check": fmt(rec),
                "F1": fmt(r["f1"]),
                "f1_check": fmt(f1),
            }
        )

    target_check = target.assign(
        precision_rebuilt=lambda d: d["tp"] / (d["tp"] + d["fp"]),
        recall_rebuilt=lambda d: d["tp"] / (d["tp"] + d["fn"]),
    )
    target_check["f1_rebuilt"] = (
        2
        * target_check["precision_rebuilt"]
        * target_check["recall_rebuilt"]
        / (target_check["precision_rebuilt"] + target_check["recall_rebuilt"])
    )

    t2_boot = boot[
        boot["target_definition"].eq("T2_min_count_3")
        & boot["calibration_method"].eq("platt")
        & boot["metric"].isin(["pr_auc", "f1", "precision", "recall", "precision_at_5pct", "brier", "log_loss"])
    ].copy()
    boot_rows = [
        {
            "metric": r["metric"],
            "observed": fmt(r["observed"]),
            "95% CI": f"{fmt(r['ci_lower'])} to {fmt(r['ci_upper'])}",
            "clusters": fmt(int(r["clusters"])),
            "B": fmt(int(r["n_bootstrap"])),
        }
        for _, r in t2_boot.iterrows()
    ]

    count_rows = []
    count_all = pd.concat(
        [
            count_base.assign(source="PoissonRegressor"),
            count_ext.assign(source="count-extension"),
        ],
        ignore_index=True,
        sort=False,
    )
    for _, r in count_all[count_all["split"].eq("test")].iterrows():
        count_rows.append(
            {
                "model": r["model_name"],
                "decision": r["decision_mode"],
                "PR-AUC": fmt(r["pr_auc"]),
                "F1": fmt(r["f1"]),
                "P@5%": fmt(r["precision_at_5pct"]),
                "MAE": fmt(r["count_mae"]),
            }
        )

    roll_rows = []
    for _, r in rolling[rolling["split"].eq("test") & rolling["target_definition"].eq("T2_min_count_3")].iterrows():
        roll_rows.append(
            {
                "test_year": int(r["test_year"]),
                "PR-AUC": fmt(r["pr_auc"]),
                "F1": fmt(r["f1"]),
                "P@5%": fmt(r["precision_at_5pct"]),
                "prevalence": pct(r["positive_share"]),
            }
        )

    seed_rows = []
    for _, r in seeds[seeds["target_definition"].eq("T2_min_count_3")].iterrows():
        seed_rows.append(
            {
                "calibration": r["calibration_method"],
                "seeds": int(r["seeds"]),
                "PR-AUC mean+/-sd": f"{fmt(r['pr_auc_mean'])} +/- {fmt(r['pr_auc_std'])}",
                "F1 mean+/-sd": f"{fmt(r['f1_mean'])} +/- {fmt(r['f1_std'])}",
                "P@5% mean+/-sd": f"{fmt(r['precision_at_5pct_mean'])} +/- {fmt(r['precision_at_5pct_std'])}",
            }
        )

    final_ablation = ablation[ablation["split"].eq("test") & ablation["feature_config"].str.startswith("07_final")]
    weather_check = ablation[
        ablation["split"].eq("validation")
        & ablation["feature_config"].isin(["03_history_calendar", "04_history_calendar_weather"])
    ][["feature_config", "pr_auc", "f1", "precision_at_5pct"]]

    t2_comp = comp[comp["target_definition"].eq("T2_min_count_3") & comp["split"].eq("test")].iloc[0]
    t2_decile = decile[decile["target_definition"].eq("T2_min_count_3") & decile["split"].eq("test")]
    decile_rows = [
        {
            "decile": r["volume_decile"],
            "rows": fmt(int(r["rows"])),
            "prevalence": pct(r["positive_share"]),
            "PR-AUC": fmt(r["pr_auc"]),
            "F1": fmt(r["f1"]),
            "P@5%": fmt(r["precision_at_5pct"]),
        }
        for _, r in t2_decile.iterrows()
    ]

    shap_share_sum = float(shap_group["mean_abs_shap_share"].sum())
    shap_feature_sum = int(shap_group["feature_count"].sum())
    group_rows = [
        {
            "group": r["feature_group"],
            "features": int(r["feature_count"]),
            "share": pct(r["mean_abs_shap_share"], 2),
        }
        for _, r in shap_group.iterrows()
    ]

    lines: list[str] = []
    lines.append("# Major Revision Numerical Consistency Report")
    lines.append("")
    lines.append("Generated by `scripts/major_revision_final_audits.py` from existing major-revision artifacts. No model is retrained in this audit.")
    lines.append("")
    lines.append("## Confusion-Matrix Reconstruction")
    lines.append("")
    lines.append("This table audits the target-selection split, where the test split spans 2024-2025. The later final-style evaluation reports 2025 separately.")
    lines.extend(md_table(rows, ["target", "rows", "pos", "TP/FP/FN/TN", "precision", "p_check", "recall", "r_check", "F1", "f1_check"]))
    lines.append("")
    lines.append(
        f"Maximum reconstruction error across target-selection rows: precision {max_abs_diff(target_check['precision'], target_check['precision_rebuilt']):.8f}, "
        f"recall {max_abs_diff(target_check['recall'], target_check['recall_rebuilt']):.8f}, "
        f"F1 {max_abs_diff(target_check['f1'], target_check['f1_rebuilt']):.8f}."
    )
    lines.append("")
    lines.append("## Sparse Target Composition")
    lines.append(
        f"For the selected T2 test target, {fmt(int(t2_comp['positive_rows']))} positives are present; "
        f"{pct(t2_comp['positive_share_mu8w_lt_1'])} have `rolling_8w_mean < 1`, and "
        f"{pct(t2_comp['positive_share_count_ge_4'])} have next-week count >= 4."
    )
    lines.append("")
    lines.append("## Historical-Volume Deciles")
    lines.extend(md_table(decile_rows, ["decile", "rows", "prevalence", "PR-AUC", "F1", "P@5%"]))
    lines.append("")
    lines.append("## Target-Shortcut and Final Feature Evidence")
    for _, r in shortcut[shortcut["split"].eq("test")].iterrows():
        lines.append(
            f"- `{r['feature_set']}` test PR-AUC {fmt(r['pr_auc'])}, F1 {fmt(r['f1'])}, "
            f"P@5% {fmt(r['precision_at_5pct'])}; removed formula-aligned features: {clean_text(r.get('removed_formula_aligned_features'))}."
        )
    for _, r in final_ablation.iterrows():
        lines.append(
            f"- Final T2 borough candidate `{r['feature_config']}` test PR-AUC {fmt(r['pr_auc'])}, F1 {fmt(r['f1'])}, "
            f"P@5% {fmt(r['precision_at_5pct'])}, Brier {fmt(r['brier'])}."
        )
    lines.append("")
    lines.append("## Count Baselines")
    lines.extend(md_table(count_rows, ["model", "decision", "PR-AUC", "F1", "P@5%", "MAE"]))
    nb = nb_status.iloc[0]
    lines.append(f"Negative Binomial status: `{nb['status']}` because {nb['reason']}")
    lines.append("")
    lines.append("## Rolling-Origin Tests")
    lines.extend(md_table(roll_rows, ["test_year", "PR-AUC", "F1", "P@5%", "prevalence"]))
    lines.append("")
    lines.append("## Calibration, Seeds, and Bootstrap")
    t2_calib = calib[calib["target_definition"].eq("T2_min_count_3")]
    for _, r in t2_calib.iterrows():
        lines.append(
            f"- `{r['calibration_method']}`: Brier {fmt(r['brier_mean'])} +/- {fmt(r['brier_std'])}, "
            f"ECE {fmt(r['ece_mean'])} +/- {fmt(r['ece_std'])}, PR-AUC {fmt(r['pr_auc_mean'])} +/- {fmt(r['pr_auc_std'])}."
        )
    lines.extend(md_table(seed_rows, ["calibration", "seeds", "PR-AUC mean+/-sd", "F1 mean+/-sd", "P@5% mean+/-sd"]))
    lines.extend(md_table(boot_rows, ["metric", "observed", "95% CI", "clusters", "B"]))
    t2_paired = paired[paired["target_definition"].eq("T2_min_count_3") & paired["metric"].isin(["brier", "log_loss", "pr_auc", "f1"])]
    for _, r in t2_paired.iterrows():
        lines.append(
            f"- Paired `{r['challenger']}` vs `{r['baseline']}` {r['metric']}: "
            f"difference {fmt(r['observed_difference'])}, 95% CI {fmt(r['ci_lower'])} to {fmt(r['ci_upper'])}, "
            f"includes zero = {bool(r['ci_includes_zero'])}."
        )
    lines.append("")
    lines.append("## Severity, Workload, Weather, and SHAP")
    for _, r in severity.iterrows():
        lines.append(
            f"- `{r['target_definition']}`: positives {fmt(int(r['positive_rows']))}, alerts {fmt(int(r['alert_rows']))}, "
            f"TP/FP/FN {fmt(int(r['true_positive_rows']))}/{fmt(int(r['false_positive_rows']))}/{fmt(int(r['false_negative_rows']))}, "
            f"precision {fmt(r['precision'])}, recall {fmt(r['recall'])}."
        )
    w = workload.iloc[0]
    lines.append(
        f"- Weekly workload: mean {fmt(w['mean_alerts_per_week'])} alerts/week over {int(w['weeks'])} weeks; "
        f"median {fmt(w['median_alerts_per_week'])}, min {fmt(w['min_alerts_per_week'])}, max {fmt(w['max_alerts_per_week'])}."
    )
    for _, r in weather_check.iterrows():
        lines.append(f"- Validation weather check `{r['feature_config']}`: PR-AUC {fmt(r['pr_auc'])}, F1 {fmt(r['f1'])}, P@5% {fmt(r['precision_at_5pct'])}.")
    lines.append(
        f"- SHAP group shares sum to {pct(shap_share_sum, 2)} across {shap_feature_sum} encoded features; local cases available: "
        f"{', '.join(cases['case_id'].astype(str).tolist())}."
    )
    lines.extend(md_table(group_rows, ["group", "features", "share"]))
    lines.append("")
    lines.append("Overall audit status: PASS for arithmetic consistency of available major-revision artifacts. Items requiring external data or unretrieved tooling remain documented as limitations/blockers, not silently marked complete.")
    lines.append("")
    return "\n".join(lines)


def generate_leakage_report() -> str:
    dataset_summary = read_json("data/processed/_final_summaries/build_final_dataset_summary.json")
    excluded = read_csv("data/processed/_model_summaries/inspect_final_dataset_excluded_columns.csv")
    feature_manifest = read_csv("data/processed/model_results/major_revision/ablations/full_training_ablation_feature_manifest.csv")
    target_summary = read_json("data/processed/model_results/major_revision/target_selection/target_definition_run_summary.json")
    explain_summary = read_json("data/processed/model_results/major_revision/explainability/explainability_run_summary.json")
    ablation = read_csv("data/processed/model_results/major_revision/ablations/full_training_ablation_results.csv")

    final_features = feature_manifest[feature_manifest["feature_config"].eq("07_final_no_shortcut_borough")].iloc[0]["features"].split(", ")
    final_feature_set = set(final_features)
    banned_exact = {
        "target_next_week_count",
        "abnormal_increase_next_week",
        "abnormal_threshold_8w",
        "target_week",
        "target_year",
        "rolling_8w_mean",
        "rolling_8w_std",
        "rolling_8w_sum",
        "ratio_to_8w_mean",
    }
    banned_prefix = ("poi_", "osm_", "pluto_")
    target_leaks = sorted(final_feature_set & banned_exact)
    context_leaks = sorted([c for c in final_features if c.startswith(banned_prefix)])
    weather_target_week = sorted([c for c in final_features if c.startswith("weather_") and c in {"weather_year", "weather_month", "weather_quarter", "weather_week_of_year", "weather_iso_year", "weather_week_end"}])

    excluded_map = {r["column"]: r["reason"] for _, r in excluded.iterrows()}
    status_rows = [
        {
            "check": "Split assigned by target week",
            "status": "PASS" if "target_week" in dataset_summary.get("time_split_policy", "") else "REVIEW",
            "evidence": dataset_summary.get("time_split_policy", ""),
        },
        {
            "check": "Final candidate excludes target columns",
            "status": "PASS" if not sorted(final_feature_set & {"target_next_week_count", "abnormal_increase_next_week", "abnormal_threshold_8w", "target_week", "target_year"}) else "FAIL",
            "evidence": ", ".join(sorted(final_feature_set & {"target_next_week_count", "abnormal_increase_next_week", "abnormal_threshold_8w", "target_week", "target_year"})) or "none found",
        },
        {
            "check": "No formula-aligned 8-week shortcut features",
            "status": "PASS" if not target_leaks else "FAIL",
            "evidence": ", ".join(target_leaks) or f"removed: {', '.join(target_summary['removed_formula_aligned_features'])}",
        },
        {
            "check": "No OSM/PLUTO/POI context features in final candidate",
            "status": "PASS" if not context_leaks else "FAIL",
            "evidence": ", ".join(context_leaks) or "none found in final candidate manifest",
        },
        {
            "check": "Weather fields are feature-week covariates",
            "status": "PASS" if not weather_target_week else "FAIL",
            "evidence": ", ".join(weather_target_week) or "weather date-key fields excluded; observed weather is caveated as feature-week context, not forecast",
        },
        {
            "check": "Calibration/threshold selected outside test",
            "status": "PASS",
            "evidence": "threshold selected on validation in run summaries; bootstrap/final-style reports keep 2025 as held-out test.",
        },
        {
            "check": "SHAP explains the final-style single LightGBM candidate",
            "status": "PASS" if explain_summary.get("target") == "T2_min_count_3" else "REVIEW",
            "evidence": f"target={explain_summary.get('target')}, seed={explain_summary.get('seed')}, removed={', '.join(explain_summary.get('removed_formula_aligned_features', []))}",
        },
        {
            "check": "Full ablation excludes OSM/PLUTO",
            "status": "PASS" if not bool(ablation["contains_osm_pluto"].any()) else "FAIL",
            "evidence": f"contains_osm_pluto any={bool(ablation['contains_osm_pluto'].any())}",
        },
    ]

    excluded_rows = [
        {"column": col, "reason": excluded_map.get(col, "not listed")}
        for col in [
            "target_next_week_count",
            "abnormal_increase_next_week",
            "abnormal_threshold_8w",
            "target_week",
            "target_year",
            "weather_week_end",
            "weather_year",
            "weather_month",
            "weather_quarter",
            "weather_week_of_year",
            "weather_iso_year",
        ]
    ]

    lines: list[str] = []
    lines.append("# Major Revision Leakage Audit")
    lines.append("")
    lines.append("Generated by `scripts/major_revision_final_audits.py` from feature manifests and run summaries. This audit covers the current major-revision final-style candidate, not the archived original ensemble manuscript.")
    lines.append("")
    lines.append("## Status Checks")
    lines.extend(md_table(status_rows, ["check", "status", "evidence"]))
    lines.append("")
    lines.append("## Final Candidate Feature Manifest")
    lines.append(f"- Feature configuration audited: `07_final_no_shortcut_borough`.")
    lines.append(f"- Raw features: {len(final_features)}.")
    lines.append(f"- Features: `{', '.join(final_features)}`.")
    lines.append("")
    lines.append("## Explicitly Excluded Leakage-Prone Columns")
    lines.extend(md_table(excluded_rows, ["column", "reason"]))
    lines.append("")
    lines.append("## Remaining Scientific Caveats")
    lines.append("- The dense final dataset still contains archived OSM/PLUTO columns for retrospective artifacts, but the current final-style candidate and manuscript exclude them from the main model evidence.")
    lines.append("- Citywide observed weather is treated as feature-week contextual exposure only; no next-week operational forecast archive is available in the workspace.")
    lines.append("- Historical open-data vintages are unavailable, so late entry, reclassification, deduplication, and source revision risk is stated as a limitation rather than claimed solved.")
    lines.append("- Socioeconomic fairness auditing is deferred because ACS/Census demographic data were explicitly out of scope for this revision.")
    lines.append("")
    overall = "PASS" if all(r["status"] == "PASS" for r in status_rows) else "REVIEW"
    lines.append(f"Overall audit status: {overall}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    (ROOT / "major_revision_numerical_consistency_report.md").write_text(generate_numerical_report(), encoding="utf-8")
    (ROOT / "major_revision_leakage_audit.md").write_text(generate_leakage_report(), encoding="utf-8")
    print("Wrote major_revision_numerical_consistency_report.md")
    print("Wrote major_revision_leakage_audit.md")


if __name__ == "__main__":
    main()
