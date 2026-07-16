from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from major_revision_full_ablation import build_feature_configs, run_config, write_report  # noqa: E402
from major_revision_model_audits import load_dataset, load_feature_sets, write_csv, write_json  # noqa: E402
from major_revision_target_selection import TARGET_DEFINITIONS, make_targets, md_table  # noqa: E402


OUT_DIR = ROOT / "data/processed/model_results/major_revision/ablations"
ROOT_REPORT = ROOT / "key_ablation_seed_stability_report.md"
KEY_CONFIGS = [
    "03_history_calendar",
    "04_history_calendar_weather",
    "05_history_calendar_weather_borough",
    "06_history_calendar_weather_nta_fe",
    "07_final_no_shortcut_borough",
    "08_final_no_shortcut_nta_fe",
]
SEEDS = [42, 123, 2026, 3407, 7777]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five-seed stability for key full-training ablation rows.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--target", default="T2_min_count_3")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"].eq("test")].copy()
    metrics = ["pr_auc", "precision_at_5pct", "f1", "precision", "recall", "alert_rate"]
    rows = []
    for cfg, g in test.groupby("feature_config", sort=False):
        row = {"feature_config": cfg, "seeds": int(g["seed"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1))
            row[f"{metric}_min"] = float(g[metric].min())
            row[f"{metric}_max"] = float(g[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_seed_report(out_dir: Path, results: pd.DataFrame, summary: pd.DataFrame, target: str, seeds: list[int]) -> None:
    val = results[results["split"].eq("validation")].sort_values(["feature_config", "seed"])
    test = results[results["split"].eq("test")].sort_values(["feature_config", "seed"])
    lines = [
        "# Key Ablation Seed-Stability Report",
        "",
        f"This report reruns the key full-training ablation rows for `{target}` across seeds {', '.join(map(str, seeds))}.",
        "",
        "Scope: the rows that support final feature-scope claims are rerun across five seeds. Auxiliary negative-control rows such as calendar-only/weather-only are not used for final claims and are left in the single-seed full-ablation artifact.",
        "",
        "## Held-Out 2025 Summary Across Seeds",
        "",
        md_table(
            summary,
            [
                "feature_config",
                "seeds",
                "pr_auc_mean",
                "pr_auc_std",
                "precision_at_5pct_mean",
                "precision_at_5pct_std",
                "f1_mean",
                "f1_std",
                "precision_mean",
                "recall_mean",
            ],
            max_rows=40,
        ),
        "",
        "## Per-Seed Validation Rows",
        "",
        md_table(
            val,
            ["feature_config", "seed", "pr_auc", "precision_at_5pct", "f1", "precision", "recall", "threshold"],
            max_rows=80,
        ),
        "",
        "## Per-Seed Test Rows",
        "",
        md_table(
            test,
            ["feature_config", "seed", "pr_auc", "precision_at_5pct", "f1", "precision", "recall", "threshold"],
            max_rows=80,
        ),
        "",
        "## Guardrails",
        "",
        "- These seeds quantify stability for key ablation evidence; final selection still follows the pre-registered validation/backtest rule.",
        "- Differences on the order of a few thousandths should be interpreted with the bootstrap and paired-CI evidence, not as standalone decisive improvements.",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "key_ablation_seed_stability_report.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")


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
    selected_configs = {name: configs[name] for name in KEY_CONFIGS}
    categorical = sorted(set(categorical_base + ["complaint_category", "boroname", "nta2020"]))
    required_features = sorted(set(f for features in selected_configs.values() for f in features) | {"rolling_8w_mean"})
    df = load_dataset({"key_ablation_seed_features": required_features})
    if args.target not in TARGET_DEFINITIONS:
        raise ValueError(f"Unknown target: {args.target}")
    frame = make_targets(df)[args.target]

    rows = []
    for seed in args.seeds:
        for config_name, features in selected_configs.items():
            rows.extend(run_config(args.target, frame, config_name, features, categorical, seed, args.progress))
    results = pd.DataFrame(rows)
    summary = summarize(results)
    write_csv(out_dir / "key_ablation_seed_stability_results.csv", results, args.overwrite)
    write_csv(out_dir / "key_ablation_seed_stability_summary.csv", summary, args.overwrite)
    write_seed_report(out_dir, results, summary, args.target, list(args.seeds))
    write_json(
        out_dir / "key_ablation_seed_stability_run_summary.json",
        {
            "status": "done",
            "script": "major_revision_key_ablation_seed_stability.py",
            "elapsed_seconds": round(time.time() - start, 3),
            "target": args.target,
            "seeds": list(args.seeds),
            "feature_configs": KEY_CONFIGS,
            "outputs": [
                "key_ablation_seed_stability_results.csv",
                "key_ablation_seed_stability_summary.csv",
                "key_ablation_seed_stability_report.md",
                "key_ablation_seed_stability_run_summary.json",
            ],
        },
    )


if __name__ == "__main__":
    main()
