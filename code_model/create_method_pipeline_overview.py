"""Create the revised Figure 1 prospective/retrospective workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [
    ROOT / "paper_overleaf/figures/method_pipeline_overview.pdf",
    ROOT / "paper_overleaf/figures/method_pipeline_overview.svg",
    ROOT / "paper_springer/figures/method_pipeline_overview.pdf",
    ROOT / "paper_springer/figures/method_pipeline_overview.svg",
]


def card(ax, xy, wh, edge, title, lines, badge=None, face="#fbfbfb"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top", fontsize=8.7, weight="bold", linespacing=1.03)
    top = y + h - 0.72
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 0.18 * i, line, ha="center", va="top", fontsize=7.0, color="#1f1f1f")
    if badge:
        bx, by, bw, bh = x + 0.20, y + 0.04, w - 0.40, 0.21
        ax.add_patch(
            FancyBboxPatch(
                (bx, by),
                bw,
                bh,
                boxstyle="round,pad=0.012,rounding_size=0.028",
                linewidth=1.0,
                edgecolor=edge,
                facecolor="white",
            )
        )
        ax.text(bx + bw / 2, by + bh / 2, badge, ha="center", va="center", fontsize=7.4, weight="bold", color=edge)


def arrow(ax, start, end, color="#5f5f5f", dashed=False, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.6,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.9)
    ax.axis("off")

    colors = {
        "blue": "#2f5f9f",
        "orange": "#c65d12",
        "green": "#3f7f3a",
        "purple": "#6a4c93",
        "gray": "#6a6a6a",
    }
    y = 1.06
    w = 2.04
    h = 1.62
    xs = [0.18, 2.75, 5.32, 7.89]

    card(
        ax,
        (xs[0], y),
        (w, h),
        colors["blue"],
        "Forecast-available\nsignals",
        ["311 history + calendar", "Weather at t + IDs"],
    )
    card(
        ax,
        (xs[1], y),
        (w, h),
        colors["orange"],
        "Leakage-controlled\npanel",
        ["NTA × Week × Category", "History at t → Event at t+1"],
        badge="t → t+1 design",
    )
    card(
        ax,
        (xs[2], y),
        (w, h),
        colors["green"],
        "Prospective\nevaluation",
        ["LightGBM + XGBoost", "2015–22 Train | 2023 Val.", "2024–25 Test"],
    )
    card(
        ax,
        (xs[3], y),
        (w, h),
        colors["purple"],
        "Category-calibrated\nalerts",
        ["Ensemble score →", "service threshold →", "risk alert", "SHAP review"],
        badge="validation only",
    )

    for i in range(3):
        arrow(ax, (xs[i] + w + 0.05, y + h / 2), (xs[i + 1] - 0.08, y + h / 2), lw=1.45)

    # Retrospective context branch: visually separate and intentionally not connected to alerts.
    branch_x = 2.18
    branch_y = 0.18
    branch_w = 5.86
    branch_h = 0.54
    ax.add_patch(
        FancyBboxPatch(
            (branch_x, branch_y),
            branch_w,
            branch_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.0,
            linestyle="--",
            edgecolor=colors["gray"],
            facecolor="#ffffff",
        )
    )
    ax.text(
        branch_x + branch_w / 2,
        branch_y + branch_h * 0.64,
        "OSM + PLUTO 2026: retrospective context only",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#2f2f2f",
        weight="bold",
    )
    ax.text(
        branch_x + branch_w / 2,
        branch_y + branch_h * 0.28,
        "excluded from prospective alerts",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#4a4a4a",
    )

    fig.tight_layout(pad=0.03)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
