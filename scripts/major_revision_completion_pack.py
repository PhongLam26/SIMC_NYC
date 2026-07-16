"""Build reviewer-facing completion status documents for the major revision.

The goal is not to declare completion by default. The generated files separate
artifact-backed answers from remaining open/deferred items so the revision can
be audited against the user's checklist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def fmt(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def table(rows: list[dict], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return out


def final_answers() -> str:
    comp = csv("data/processed/model_results/major_revision/target_selection/target_definition_composition.csv")
    decile = csv("data/processed/model_results/major_revision/target_selection/target_definition_decile_results.csv")
    shortcut = csv("data/processed/model_results/major_revision/model_audits/target_shortcut_results.csv")
    count_base = csv("data/processed/model_results/major_revision/model_audits/count_model_baseline_results.csv")
    count_ext = csv("data/processed/model_results/major_revision/count_extensions/count_model_extension_results.csv")
    nb = csv("data/processed/model_results/major_revision/count_extensions/negative_binomial_status.csv")
    boot = csv("data/processed/model_results/major_revision/bootstrap/bootstrap_ci_results.csv")
    paired = csv("data/processed/model_results/major_revision/bootstrap/paired_model_difference_ci.csv")
    rolling = csv("data/processed/model_results/major_revision/backtests/rolling_origin_results.csv")
    calib = csv("data/processed/model_results/major_revision/calibration/calibration_results.csv")
    severity = csv("data/processed/model_results/major_revision/error_analysis/error_severity_analysis.csv")
    workload = csv("data/processed/model_results/major_revision/disparity_workload/weekly_workload_summary.csv")
    ablation = csv("data/processed/model_results/major_revision/ablations/full_training_ablation_results.csv")
    shap = csv("data/processed/model_results/major_revision/explainability/shap_group_importance.csv")
    other = csv("data/processed/model_results/major_revision/complaint_mapping/other_category_composition.csv")
    covid = csv("data/processed/model_results/major_revision/covid_archive/covid_exclusion_sensitivity.csv")

    t2_comp = comp[(comp.target_definition == "T2_min_count_3") & (comp.split == "test")].iloc[0]
    t0_comp = comp[(comp.target_definition == "T0_current_reference") & (comp.split == "test")].iloc[0]
    t2_dec = decile[(decile.target_definition == "T2_min_count_3") & (decile.split == "test")]
    dec_rows = [
        {
            "decile": r.volume_decile,
            "prevalence": pct(r.positive_share),
            "PR-AUC": fmt(r.pr_auc),
            "F1": fmt(r.f1),
            "P@5%": fmt(r.precision_at_5pct),
        }
        for _, r in t2_dec.iterrows()
    ]

    shortcut_rows = []
    for _, r in shortcut[shortcut.split == "test"].iterrows():
        shortcut_rows.append(
            {
                "config": r.feature_set,
                "PR-AUC": fmt(r.pr_auc),
                "F1": fmt(r.f1),
                "P@5%": fmt(r.precision_at_5pct),
                "delta PR-AUC": fmt(r.delta_vs_current_pr_auc),
            }
        )

    count_all = pd.concat([count_base, count_ext], ignore_index=True, sort=False)
    count_rows = []
    for _, r in count_all[count_all.split == "test"].iterrows():
        count_rows.append(
            {
                "model": r.model_name,
                "decision": r.decision_mode,
                "PR-AUC": fmt(r.pr_auc),
                "F1": fmt(r.f1),
                "P@5%": fmt(r.precision_at_5pct),
                "MAE": fmt(r.count_mae),
            }
        )

    ci_rows = []
    for _, r in boot[
        (boot.target_definition == "T2_min_count_3")
        & (boot.calibration_method == "platt")
        & boot.metric.isin(["pr_auc", "f1", "precision", "recall", "precision_at_1pct", "precision_at_5pct", "brier", "log_loss"])
    ].iterrows():
        ci_rows.append(
            {
                "metric": r.metric,
                "value": fmt(r.observed),
                "95% CI": f"{fmt(r.ci_lower)} to {fmt(r.ci_upper)}",
            }
        )

    roll_rows = []
    for _, r in rolling[(rolling.target_definition == "T2_min_count_3") & (rolling.split == "test")].iterrows():
        roll_rows.append(
            {
                "test year": int(r.test_year),
                "PR-AUC": fmt(r.pr_auc),
                "F1": fmt(r.f1),
                "P@5%": fmt(r.precision_at_5pct),
                "prevalence": pct(r.positive_share),
            }
        )

    cal_rows = []
    for _, r in calib[
        (calib.target_definition == "T2_min_count_3")
        & (calib.fold_id == "final_style_2025")
        & (calib.split == "test")
    ].iterrows():
        cal_rows.append(
            {
                "method": r.calibration_method,
                "Brier": fmt(r.brier),
                "ECE": fmt(r.ece),
                "log loss": fmt(r.log_loss),
                "PR-AUC": fmt(r.pr_auc),
            }
        )

    sev_rows = []
    for _, r in severity[(severity.target_definition == "T2_min_count_3") & (severity.dimension == "z_bin")].iterrows():
        sev_rows.append(
            {
                "severity bin": r.bin,
                "positives": int(r.positive_rows),
                "recall": fmt(r.recall),
                "share": pct(r.share_of_target_positives),
            }
        )

    final = ablation[(ablation.feature_config == "07_final_no_shortcut_borough") & (ablation.split == "test")].iloc[0]
    hgb = count_ext[
        (count_ext.model_name == "hist_gradient_boosting_poisson_count")
        & (count_ext.decision_mode == "formula_threshold")
        & (count_ext.split == "test")
    ].iloc[0]
    hurdle = count_ext[
        (count_ext.model_name == "hurdle_hgb_occurrence_poisson_positive_count")
        & (count_ext.decision_mode == "formula_threshold")
        & (count_ext.split == "test")
    ].iloc[0]
    w = workload.iloc[0]
    weather_calendar = ablation[(ablation.feature_config == "03_history_calendar") & (ablation.split == "validation")].iloc[0]
    weather_added = ablation[(ablation.feature_config == "04_history_calendar_weather") & (ablation.split == "validation")].iloc[0]
    borough = ablation[(ablation.feature_config == "07_final_no_shortcut_borough") & (ablation.split == "validation")].iloc[0]
    nta = ablation[(ablation.feature_config == "08_final_no_shortcut_nta_fe") & (ablation.split == "validation")].iloc[0]
    covid_ref = covid[covid.config == "reference_train_through_2023"].iloc[0]
    covid_ex = covid[covid.config == "exclude_2020_2021_from_training"].iloc[0]

    top_other = ", ".join(
        f"{r.complaint_type} ({pct(r.share_of_other_requests)})" for _, r in other.head(8).iterrows()
    )
    shap_rows = ", ".join(
        f"{r.feature_group} {pct(r.mean_abs_shap_share, 2)}" for _, r in shap.iterrows()
    )

    lines: list[str] = [
        "# Major Revision Final Required Answers",
        "",
        "Generated by `scripts/major_revision_completion_pack.py` from current major-revision artifacts. `OPEN` means the answer is not yet fully supported by the current artifacts and should not be claimed as complete.",
        "",
        "1. Low-baseline positive rows.",
        f"- T2 test positives with `rolling_8w_mean < 1`: {pct(t2_comp.positive_share_mu8w_lt_1, 2)} ({int(t2_comp.positive_mu8w_lt_1):,}/{int(t2_comp.positive_rows):,}).",
        f"- Original T0 comparison: {pct(t0_comp.positive_share_mu8w_lt_1, 2)} ({int(t0_comp.positive_mu8w_lt_1):,}/{int(t0_comp.positive_rows):,}).",
        "",
        "2. Historical-volume decile performance.",
    ]
    lines.extend(table(dec_rows, ["decile", "prevalence", "PR-AUC", "F1", "P@5%"]))
    lines.extend(
        [
            "",
            "3. Formula-aligned 8-week feature removal.",
        ]
    )
    lines.extend(table(shortcut_rows, ["config", "PR-AUC", "F1", "P@5%", "delta PR-AUC"]))
    lines.append(f"- Final SHAP after removal: {shap_rows}.")
    lines.append("")
    lines.append("4. Poisson/NB/NTA fixed-effect baselines.")
    lines.extend(table(count_rows, ["model", "decision", "PR-AUC", "F1", "P@5%", "MAE"]))
    lines.append(f"- Negative Binomial: {nb.iloc[0].status}; reason: {nb.iloc[0].reason}")
    lines.append("")
    lines.append("5. Does the final tree model beat count baselines, and what is the CI difference?")
    lines.append(
        f"- Point estimate: final LightGBM PR-AUC {fmt(final.pr_auc)} and P@5% {fmt(final.precision_at_5pct)} beat HGB Poisson PR-AUC {fmt(hgb.pr_auc)} / P@5% {fmt(hgb.precision_at_5pct)} and hurdle HGB PR-AUC {fmt(hurdle.pr_auc)} / P@5% {fmt(hurdle.precision_at_5pct)}."
    )
    lines.append("- OPEN: a paired tree-vs-count bootstrap CI has not yet been generated because current count-baseline artifacts do not save row-level count-model scores.")
    lines.append("")
    lines.append("6. Main 95% CIs.")
    lines.extend(table(ci_rows, ["metric", "value", "95% CI"]))
    lines.append("")
    lines.append("7. Are model differences statistically distinguishable?")
    for _, r in paired[(paired.target_definition == "T2_min_count_3") & paired.metric.isin(["brier", "log_loss", "pr_auc", "f1"])].iterrows():
        lines.append(f"- {r.challenger} vs {r.baseline}, {r.metric}: diff {fmt(r.observed_difference)}, CI {fmt(r.ci_lower)} to {fmt(r.ci_upper)}, includes zero={bool(r.ci_includes_zero)}.")
    lines.append("- OPEN: tree-vs-count and NTA-vs-borough paired CIs are not yet available.")
    lines.append("")
    lines.append("8. Rolling-origin metrics by year.")
    lines.extend(table(roll_rows, ["test year", "PR-AUC", "F1", "P@5%", "prevalence"]))
    lines.append("")
    lines.append("9. 2024 versus 2025.")
    y2024 = rolling[(rolling.target_definition == "T2_min_count_3") & (rolling.split == "test") & (rolling.test_year == 2024)].iloc[0]
    y2025 = rolling[(rolling.target_definition == "T2_min_count_3") & (rolling.split == "test") & (rolling.test_year == 2025)].iloc[0]
    lines.append(f"- 2024: PR-AUC {fmt(y2024.pr_auc)}, F1 {fmt(y2024.f1)}, P@5% {fmt(y2024.precision_at_5pct)}, prevalence {pct(y2024.positive_share)}.")
    lines.append(f"- 2025: PR-AUC {fmt(y2025.pr_auc)}, F1 {fmt(y2025.f1)}, P@5% {fmt(y2025.precision_at_5pct)}, prevalence {pct(y2025.positive_share)}.")
    lines.append("")
    lines.append("10. Calibration before and after.")
    lines.extend(table(cal_rows, ["method", "Brier", "ECE", "log loss", "PR-AUC"]))
    lines.append("")
    lines.append("11. Recall by event severity.")
    lines.extend(table(sev_rows, ["severity bin", "positives", "recall", "share"]))
    lines.append("")
    lines.append("12. Precision@1%, 5%, and fixed weekly capacity.")
    lines.append(f"- Final 2025 precision@1% {fmt(final.precision_at_1pct)}, precision@5% {fmt(final.precision_at_5pct)}, validation-threshold alert rate {fmt(final.alert_rate)}.")
    lines.append("")
    lines.append("13. Average weekly alert workload.")
    lines.append(f"- Mean {fmt(w.mean_alerts_per_week)} alerts/week over {int(w.weeks)} weeks; median {fmt(w.median_alerts_per_week)}, min {int(w.min_alerts_per_week)}, max {int(w.max_alerts_per_week)}.")
    lines.append("")
    lines.append("14. Weather validation/backtest improvement.")
    lines.append(f"- History+calendar validation PR-AUC {fmt(weather_calendar.pr_auc)}; adding feature-week weather gives {fmt(weather_added.pr_auc)} (delta {fmt(weather_added.pr_auc - weather_calendar.pr_auc)}). Weather is not claimed as a core contribution.")
    lines.append("")
    lines.append("15. NTA fixed effects.")
    lines.append(f"- Borough final validation PR-AUC {fmt(borough.pr_auc)}; NTA fixed-effect variant {fmt(nta.pr_auc)} (delta {fmt(nta.pr_auc - borough.pr_auc)}). The gain is small and currently single-seed.")
    lines.append("")
    lines.append("16. What is in `other`?")
    lines.append(f"- `other` contains {other.complaint_type.nunique()} complaint types; largest contributors: {top_other}.")
    lines.append("")
    lines.append("17. Final model and why not selected by test.")
    lines.append("- Current draft freezes a single no-shortcut LightGBM selected by validation/backtest evidence, construct validity, simpler explanation, and the pre-registered tie-breaker favoring a single model over an ensemble.")
    lines.append("")
    lines.append("18. Final target and why.")
    lines.append("- T2 = shifted 8-week abnormal threshold plus `count_{t+1} >= 3`; selected for construct validity and validation/backtest support while retaining the full dense panel risk set.")
    lines.append("")
    lines.append("19. Final title and why.")
    lines.append("- `Explainable Early Warning for Next-Week Abnormal Reported 311 Demand`; it matches the binary abnormal reported-demand task and avoids unsupported `Category-Aware`/ensemble framing.")
    lines.append("")
    lines.append("20. Future-work/deferred external-data items.")
    lines.append("- Deferred because external data were explicitly out of scope: ACS/Census socioeconomic fairness, historical NWS forecasts, spatial weather grids, historical 311 vintages, historical OSM/PLUTO snapshots, agency staffing/cost functions, operational outcomes, and cross-city replication.")
    lines.append(f"- COVID sensitivity retained evidence: excluding 2020-2021 changes 2025 PR-AUC from {fmt(covid_ref.pr_auc)} to {fmt(covid_ex.pr_auc)} and P@5% from {fmt(covid_ref.precision_at_5pct)} to {fmt(covid_ex.precision_at_5pct)}.")
    lines.append("")
    return "\n".join(lines)


def completion_checklist() -> str:
    items = [
        ("Current submission archived safely", "PASS", "MAJOR_REVISION_BASELINE.md; archive/submitted_2026-07-16/."),
        ("Workspace audit completed", "PASS", "major_revision_workspace_audit.md."),
        ("OSM/PLUTO removed from main paper and final pipeline", "PARTIAL/PASS", "paper_overleaf/main.tex and major_revision_leakage_audit.md; dense dataset still stores archived columns."),
        ("Ensemble removed unless statistically justified", "PASS", "paper_overleaf/main.tex uses single LightGBM; pre_registered_revision_selection_rule.md."),
        ("Single final model selected without test", "PARTIAL/PASS", "pre_registered_revision_selection_rule.md; selection is documented, but full XGBoost multi-seed comparison is not expanded."),
        ("Target shortcut audit completed", "PASS", "data/processed/model_results/major_revision/model_audits/ and major_revision_numerical_consistency_report.md."),
        ("Sparse-panel diagnostics completed", "PASS", "data/processed/model_results/major_revision/initial_audit/."),
        ("Low-baseline positive share reported", "PASS", "major_revision_final_required_answers.md; paper_overleaf/main.tex."),
        ("Sparse-aware targets evaluated", "PARTIAL/PASS", "T0-T3 evaluated; T4 target formulation remains open as model baseline only."),
        ("Main target selected without test", "PARTIAL/PASS", "pre_registered_revision_selection_rule.md."),
        ("Poisson count baseline completed", "PASS", "count_model_baseline_results.csv."),
        ("NTA fixed-effect baseline completed", "PASS", "count_model_baseline_results.csv and full_training_ablation_results.csv."),
        ("Negative Binomial completed or documented blocker", "PASS", "negative_binomial_status.csv documents missing statsmodels blocker."),
        ("Full-training ablation completed", "PARTIAL/PASS", "full_training_ablation_results.csv full-row seed 42; five-seed ablation for all rows remains open."),
        ("At least 5 stochastic seeds completed", "PARTIAL/PASS", "seed_stability_results.csv for T0/T2 final-style LightGBM; not every stochastic candidate/ablation row."),
        ("Rolling-origin backtesting completed", "PASS", "rolling_origin_results.csv."),
        ("2024 and 2025 reported separately", "PASS", "rolling_origin_results.csv; paper_overleaf/main.tex."),
        ("COVID/archive boundary analysis completed", "PARTIAL/PASS", "covid_archive_report.md; row-level source-vintage IDs unavailable and discussed as limitation."),
        ("Data vintage limitation added", "PASS", "paper_overleaf/main.tex Discussion."),
        ("Platt/isotonic calibration completed", "PASS", "calibration_results.csv."),
        ("Reliability diagram generated", "PASS", "paper_overleaf/figures/reliability_diagram.pdf."),
        ("Brier/ECE reported", "PASS", "calibration_report.md; paper_overleaf/main.tex."),
        ("Cluster bootstrap CI completed", "PASS", "bootstrap_ci_results.csv."),
        ("Paired difference CI completed", "PARTIAL", "paired_model_difference_ci.csv covers calibration differences; tree-vs-count paired CI remains OPEN."),
        ("Precision@k/capacity analysis completed", "PASS", "capacity_precision_at_k.csv; major_revision_final_required_answers.md."),
        ("Weekly workload reported", "PASS", "weekly_workload_summary.csv; paper_overleaf/main.tex."),
        ("Error severity analysis completed", "PASS", "error_severity_analysis.csv; paper_overleaf/main.tex."),
        ("Weather full-data ablation completed", "PASS", "full_training_ablation_results.csv."),
        ("NTA spatial baseline completed", "PASS", "count baselines and NTA FE ablation exist."),
        ("Borough/volume disparity analysis completed", "PASS", "disparity_workload_report.md and decile outputs."),
        ("Socioeconomic fairness clearly deferred", "PASS", "paper_overleaf/main.tex Discussion."),
        ("Complaint mapping appendix generated", "PASS", "paper_overleaf/supplementary_complaint_mapping.pdf and CSVs."),
        ("Other category composition reported", "PASS", "other_category_composition.csv."),
        ("SHAP explains final model", "PASS", "final_model_explainability_report.md; no ensemble proxy."),
        ("Beeswarm generated", "PASS", "shap_beeswarm.pdf and paper_overleaf figure copy."),
        ("TP/FP/FN local cases generated", "PASS", "shap_local_tp/fp/fn.pdf and shap_local_cases.csv."),
        ("Old generic figures removed", "PASS", "paper_overleaf/main.tex uses reliability/SHAP/local figures; no old workflow figure."),
        ("Tables regenerated from artifacts", "PARTIAL/PASS", "main tables synced to artifacts; not every table has an automated generator script."),
        ("Title matches binary abnormal-demand task", "PASS", "paper_overleaf/main.tex title."),
        ("Abstract uses new frozen metrics", "PASS", "paper_overleaf/main.tex abstract."),
        ("Contributions rewritten", "PASS", "paper_overleaf/main.tex Introduction."),
        ("Discussion includes feedback loop and limitations", "PASS", "paper_overleaf/main.tex Discussion."),
        ("References cleaned and verified", "PASS", "paper_overleaf/references.bib and main.bbl cleaned."),
        ("Data/code statement accurate", "PASS", "paper_overleaf/main.tex includes GitHub URL and rebuild wording."),
        ("Numerical consistency audit PASS", "PASS", "major_revision_numerical_consistency_report.md."),
        ("Leakage audit PASS", "PASS", "major_revision_leakage_audit.md."),
        ("PDF <=15 pages", "PASS", "paper_overleaf/main.pdf is 13 A4 pages."),
        ("No stale old metrics", "PARTIAL/PASS", "paper_overleaf text audit clean for old framing; old paper_springer remains archived/submission source."),
        ("Review PDF visual QA PASS", "PARTIAL/PASS", "recent visual QA passed; rerender after any later LaTeX change."),
        ("SIMC submission compliance PASS", "PARTIAL", "old submission audit passed; final new SIMC submission PDF has not been rebuilt/frozen."),
        ("REVISION_REPORT maps all 38 issues", "PARTIAL", "REVISION_REPORT has many mappings; this checklist adds explicit final status but P/R item-by-item matrix can be expanded."),
        ("All external-data-dependent tasks marked Future Work", "PASS", "paper_overleaf/main.tex and final answers."),
        ("No unsupported claim remains", "PARTIAL/PASS", "main Overleaf text is conservative; final reviewer pass still recommended before goal completion."),
    ]
    rows = [{"item": item, "status": status, "evidence": evidence} for item, status, evidence in items]
    lines = [
        "# Major Revision Completion Checklist",
        "",
        "Generated by `scripts/major_revision_completion_pack.py`. This is a status audit, not an automatic completion declaration.",
        "",
    ]
    lines.extend(table(rows, ["item", "status", "evidence"]))
    lines.append("")
    lines.append("Open items before a strict completion claim:")
    lines.append("- Generate paired tree-vs-count bootstrap differences if reviewer-facing statistical dominance over count baselines is required.")
    lines.append("- Decide whether to expand five-seed evidence to all key ablation rows and any remaining stochastic candidate.")
    lines.append("- Build and audit the final SIMC submission PDF only after manuscript freeze.")
    lines.append("- Optionally expand REVISION_REPORT into a literal P1/P2/P3/R item-by-item matrix.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    (ROOT / "major_revision_final_required_answers.md").write_text(final_answers(), encoding="utf-8")
    (ROOT / "major_revision_completion_checklist.md").write_text(completion_checklist(), encoding="utf-8")
    print("Wrote major_revision_final_required_answers.md")
    print("Wrote major_revision_completion_checklist.md")


if __name__ == "__main__":
    main()
