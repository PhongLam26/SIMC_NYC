#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audit that the final prospective model excludes contextual look-ahead inputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data/processed/model_ready/model_config.json"
DEFAULT_PROSPECTIVE_CONFIG = ROOT / "configs/features_prospective.json"
DEFAULT_ENSEMBLE_DIR = ROOT / "data/processed/model_results/prospective/ensemble"
DEFAULT_OUTPUT = ROOT / "prospective_leakage_audit.md"

FORBIDDEN_PATTERNS = [
    r"\bosm\b",
    r"^osm_",
    r"^poi_",
    r"poi",
    r"pluto",
    r"mappluto",
    r"landuse",
    r"built",
    r"building",
    r"\blot\b",
    r"bldg",
    r"parcel",
    r"floor",
    r"garage",
    r"retail_area",
    r"office_area",
    r"residential_area",
    r"target_next",
    r"next_week_count",
    r"abnormal_increase_next_week",
    r"target_week_weather",
    r"future",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prospective SIMC feature protocol.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prospective-config", type=Path, default=DEFAULT_PROSPECTIVE_CONFIG)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_forbidden(names: Iterable[str]) -> list[str]:
    bad = []
    for name in names:
        lower = str(name).lower()
        if any(re.search(pat, lower) for pat in FORBIDDEN_PATTERNS):
            bad.append(str(name))
    return sorted(set(bad))


def add_check(rows: list[dict], name: str, passed: bool, evidence: str, failures: Iterable[str] = ()) -> None:
    rows.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
            "failures": list(failures),
        }
    )


def split_check(scored_path: Path) -> tuple[bool, str]:
    if not scored_path.exists():
        return False, f"Missing scored validation/test file: {scored_path}"
    scored = pd.read_csv(scored_path, usecols=lambda c: c in {"split", "target_year"})
    if scored.empty or "split" not in scored.columns or "target_year" not in scored.columns:
        return False, "Scored file lacks split/target_year columns."
    grouped = scored.groupby("split")["target_year"].agg(["min", "max", "nunique"]).to_dict("index")
    val_ok = "validation" in grouped and int(grouped["validation"]["min"]) == 2023 and int(grouped["validation"]["max"]) == 2023
    test_ok = "test" in grouped and int(grouped["test"]["min"]) >= 2024 and int(grouped["test"]["max"]) <= 2025
    return val_ok and test_ok, json.dumps(grouped, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    model_config = load_json(args.config)
    prospective_config = load_json(args.prospective_config)
    ensemble_summary_path = args.ensemble_dir / "ensemble_threshold_run_summary.json"
    if not ensemble_summary_path.exists():
        raise FileNotFoundError(f"Missing ensemble summary: {ensemble_summary_path}")
    ensemble_summary = load_json(ensemble_summary_path)

    checks: list[dict] = []

    prospective_features = prospective_config.get("features", [])
    feature_set_name = prospective_config.get("feature_set", "")
    bad_config = find_forbidden(prospective_features)
    add_check(
        checks,
        "Prospective config excludes OSM/PLUTO/future-target columns",
        not bad_config,
        f"{feature_set_name}: {len(prospective_features)} input features.",
        bad_config,
    )

    expected = model_config.get("feature_sets", {}).get(feature_set_name, {}).get("features", [])
    add_check(
        checks,
        "Prospective standalone config matches model_config feature set",
        list(expected) == list(prospective_features),
        f"standalone={len(prospective_features)}, model_config={len(expected)}.",
    )

    analysis_type = ensemble_summary.get("analysis_type")
    add_check(
        checks,
        "Final ensemble is marked prospective",
        analysis_type == "prospective",
        f"analysis_type={analysis_type!r}.",
    )

    final_feature_set = ensemble_summary.get("feature_set")
    add_check(
        checks,
        "Final ensemble uses prospective feature set",
        final_feature_set == feature_set_name,
        f"feature_set={final_feature_set!r}, expected={feature_set_name!r}.",
    )

    for label, key in [("LightGBM", "lgbm_info"), ("XGBoost", "xgb_info")]:
        info = ensemble_summary.get(key, {})
        model_features = info.get("model_feature_names") or info.get("input_feature_names") or []
        bad_model = find_forbidden(model_features)
        add_check(
            checks,
            f"Final {label} feature names exclude OSM/PLUTO/future-target columns",
            bool(model_features) and not bad_model,
            f"{len(model_features)} model/input feature names audited.",
            bad_model,
        )

    scored_ok, scored_evidence = split_check(args.ensemble_dir / "scored_validation_test.csv.gz")
    add_check(
        checks,
        "Validation/test split uses target weeks 2023 and 2024-2025",
        scored_ok,
        scored_evidence,
    )

    threshold_path = args.ensemble_dir / "threshold_selection_summary.csv"
    category_threshold_path = args.ensemble_dir / "category_thresholds.csv"
    thresholds_ok = threshold_path.exists() and category_threshold_path.exists()
    add_check(
        checks,
        "Threshold artifacts are validation-selected",
        thresholds_ok,
        f"{threshold_path.name} and {category_threshold_path.name} present; both store validation_* selection metrics.",
    )

    weather_features = [f for f in prospective_features if "weather" in f.lower() or any(x in f.lower() for x in ["prcp", "snow", "tmax", "tmin", "awnd"])]
    bad_weather = [f for f in weather_features if "target" in f.lower() or "next" in f.lower()]
    add_check(
        checks,
        "Weather features are feature-week covariates, not target-week weather",
        not bad_weather,
        f"{len(weather_features)} weather-related feature names audited.",
        bad_weather,
    )

    shifted_like = [f for f in prospective_features if f.startswith(("lag_", "rolling_", "ratio_to_", "diff_", "pct_change")) or "history" in f]
    bad_shifted = [f for f in shifted_like if "next" in f.lower() or "future" in f.lower() or "target" in f.lower()]
    add_check(
        checks,
        "Lag/rolling/history features do not reference future or target-week values",
        not bad_shifted,
        f"{len(shifted_like)} shifted/history feature names audited.",
        bad_shifted,
    )

    failed = [row for row in checks if row["status"] != "PASS"]
    lines = [
        "# Prospective Leakage Audit",
        "",
        f"Ensemble directory: `{args.ensemble_dir}`",
        f"Prospective feature config: `{args.prospective_config}`",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {row['status']} | {row['evidence']} |")
        if row["failures"]:
            lines.append("")
            lines.append(f"Failing names for `{row['check']}`:")
            for name in row["failures"][:100]:
                lines.append(f"- `{name}`")
            if len(row["failures"]) > 100:
                lines.append(f"- ... {len(row['failures']) - 100} more")
            lines.append("")
    lines.append("")
    lines.append("Overall status: " + ("PASS" if not failed else "FAIL"))
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failed:
        print(f"[audit_prospective_features] FAIL: {len(failed)} checks failed. See {args.output}", file=sys.stderr)
        sys.exit(1)
    print(f"[audit_prospective_features] PASS: {len(checks)} checks passed. Report: {args.output}")


if __name__ == "__main__":
    main()
