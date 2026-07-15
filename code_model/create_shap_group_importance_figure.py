"""Create the paper Figure 3 SHAP group percentage chart.

This script uses existing prospective SHAP aggregation outputs. It does not
rerun SHAP or change model metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/model_results/prospective/shap/shap_group_importance.csv"
REPORT = ROOT / "data/processed/model_results/prospective/shap/shap_group_importance_figure_report.md"
OUTPUTS = [
    ROOT / "data/processed/model_results/prospective/shap/shap_group_importance.pdf",
    ROOT / "data/processed/model_results/prospective/shap/shap_group_importance.svg",
    ROOT / "data/processed/model_results/prospective/shap/shap_group_importance.png",
    ROOT / "paper_overleaf/figures/shap_group_importance.pdf",
    ROOT / "paper_overleaf/figures/shap_group_importance.svg",
    ROOT / "paper_overleaf/figures/shap_group_importance.png",
    ROOT / "paper_springer/figures/shap_group_importance.pdf",
    ROOT / "paper_springer/figures/shap_group_importance.svg",
    ROOT / "paper_springer/figures/shap_group_importance.png",
]

LABELS = {
    "historical_temporal": "Historical temporal",
    "semantic_category": "Service category",
    "calendar": "Calendar",
    "weather": "Weather",
    "current_demand": "Current demand",
    "spatial_context": "Spatial identifiers",
    "other": "Other",
}


def main() -> None:
    df = pd.read_csv(INPUT).copy()
    required = {"feature_group", "mean_abs_shap_sum", "feature_count", "mean_abs_shap_share"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required SHAP group columns: {sorted(missing)}")

    df["label"] = df["feature_group"].map(LABELS).fillna(df["feature_group"].str.replace("_", " ").str.title())
    df["share_pct"] = df["mean_abs_shap_sum"] / df["mean_abs_shap_sum"].sum() * 100.0
    total_pct = df["share_pct"].sum()
    if abs(total_pct - 100.0) > 1e-6:
        raise ValueError(f"SHAP group percentages sum to {total_pct:.8f}, expected 100.")
    if df["feature_group"].str.contains("osm|pluto", case=False, regex=True).any():
        raise ValueError("OSM/PLUTO group unexpectedly appears in prospective SHAP groups.")

    df = df.sort_values("share_pct", ascending=False).reset_index(drop=True)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.15))
    colors = ["#4C78A8"] + ["#9ECAE1"] * (len(df) - 1)
    bars = ax.barh(df["label"], df["share_pct"], color=colors, edgecolor="#3a3a3a", linewidth=0.35)
    ax.invert_yaxis()
    ax.set_xlabel("Share of summed mean absolute SHAP importance (%)")
    ax.set_xlim(0, max(82.0, df["share_pct"].max() + 7.0))
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="y", length=0)

    for bar, value in zip(bars, df["share_pct"]):
        ax.text(
            value + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#222222",
        )

    fig.tight_layout(pad=0.4)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=220)
    plt.close(fig)

    lines = [
        "# SHAP Group Importance Figure Report",
        "",
        f"Input: `{INPUT.relative_to(ROOT)}`",
        "",
        "| Display label | Internal group | Mean abs SHAP sum | Feature count | Share (%) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.label} | `{row.feature_group}` | {row.mean_abs_shap_sum:.6f} | "
            f"{int(row.feature_count)} | {row.share_pct:.2f} |"
        )
    lines.extend(["", f"Total share: {total_pct:.6f}%."])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
