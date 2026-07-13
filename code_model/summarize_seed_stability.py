#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Summarize seed-stability ensemble runs for the SIMC NYC paper.

Expected input layout:
    data/processed/model_results/seed_stability_onehot/seed_042/final_model_comparison.csv
    data/processed/model_results/seed_stability_onehot/seed_007/final_model_comparison.csv
    data/processed/model_results/seed_stability_onehot/seed_123/final_model_comparison.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = [
    "val_f1",
    "val_pr_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_pr_auc",
    "test_roc_auc",
    "test_balanced_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize seed-stability final model comparisons.")
    parser.add_argument("--input-root", default="data/processed/model_results/seed_stability_onehot")
    parser.add_argument("--method-id", default="ensemble_lgbm_w0p500_per_category")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_from_name(path: Path) -> int:
    m = re.search(r"(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    if not cols:
        return ""

    def fmt(x) -> str:
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        comparison_path = seed_dir / "final_model_comparison.csv"
        if not comparison_path.exists():
            continue
        df = pd.read_csv(comparison_path)
        row = df[df["method_id"].astype(str) == args.method_id]
        if row.empty:
            row = df.sort_values("validation_rank").head(1)
        rec = row.iloc[0].to_dict()
        rec["seed"] = seed_from_name(seed_dir)
        rec["source_dir"] = str(seed_dir)
        rows.append(rec)

    if not rows:
        raise FileNotFoundError(f"No seed final_model_comparison.csv files found under {root}")

    by_seed = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    keep = ["seed", "method_id", "model_label", "threshold_mode"] + [m for m in METRICS if m in by_seed.columns]
    by_seed = by_seed[keep + ["source_dir"]]

    summary_rows = []
    for metric in METRICS:
        if metric not in by_seed.columns:
            continue
        s = pd.to_numeric(by_seed[metric], errors="coerce")
        summary_rows.append({
            "metric": metric,
            "n_seeds": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
        })
    summary = pd.DataFrame(summary_rows)

    by_seed_path = output_dir / "seed_stability_by_seed.csv"
    summary_path = output_dir / "seed_stability_summary.csv"
    md_path = output_dir / "seed_stability_summary.md"
    if not args.overwrite:
        for p in [by_seed_path, summary_path, md_path]:
            if p.exists():
                raise FileExistsError(f"Output exists. Use --overwrite: {p}")

    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    lines = [
        "# Seed Stability Summary",
        "",
        f"Method summarized: `{args.method_id}`.",
        "",
        "## Mean ± Std",
        "",
    ]
    for _, r in summary.iterrows():
        lines.append(f"- `{r['metric']}`: {r['mean']:.4f} ± {r['std']:.4f} (range {r['min']:.4f}-{r['max']:.4f})")
    lines.extend(["", "## By Seed", "", markdown_table(by_seed)])
    write_text(md_path, "\n".join(lines) + "\n", args.overwrite)

    print(f"Wrote {by_seed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
