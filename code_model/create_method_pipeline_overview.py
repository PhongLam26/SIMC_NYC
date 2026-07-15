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
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=8.2, weight="bold")
    top = y + h - 0.34
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 0.105 * i, line, ha="center", va="top", fontsize=6.8)
    if badge:
        bx, by, bw, bh = x + 0.11, y + 0.055, w - 0.22, 0.15
        ax.add_patch(
            FancyBboxPatch(
                (bx, by),
                bw,
                bh,
                boxstyle="round,pad=0.012,rounding_size=0.028",
                linewidth=0.9,
                edgecolor=edge,
                facecolor="white",
            )
        )
        ax.text(bx + bw / 2, by + bh / 2, badge, ha="center", va="center", fontsize=6.9, weight="bold", color=edge)


def arrow(ax, start, end, color="#5f5f5f", dashed=False, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
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
    fig, ax = plt.subplots(figsize=(7.25, 2.08))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")

    colors = {
        "blue": "#2f5f9f",
        "orange": "#c65d12",
        "green": "#3f7f3a",
        "purple": "#6a4c93",
        "gray": "#6a6a6a",
    }
    y = 0.82
    w = 2.03
    h = 1.15
    xs = [0.18, 2.78, 5.38, 7.78]

    card(
        ax,
        (xs[0], y),
        (w, h),
        colors["blue"],
        "1  Forecast-available signals",
        ["311 history", "Calendar", "Feature-week weather", "Service/geographic IDs"],
    )
    card(
        ax,
        (xs[1], y),
        (w, h),
        colors["orange"],
        "2  Leakage-controlled panel",
        ["NTA x Week x Category", "History at t -> Event at t+1", "shifted temporal features"],
        badge="t -> t+1 design",
    )
    card(
        ax,
        (xs[2], y),
        (w, h),
        colors["green"],
        "3  Prospective evaluation",
        ["LightGBM + XGBoost", "Train 2015-22 | Val 2023", "Test 2024-25"],
    )
    card(
        ax,
        (xs[3], y),
        (w, h),
        colors["purple"],
        "4  Category-calibrated alerts",
        ["Ensemble score ->", "service threshold -> risk alert", "SHAP-assisted review"],
        badge="validation only",
    )

    for i in range(3):
        arrow(ax, (xs[i] + w + 0.05, y + h / 2), (xs[i + 1] - 0.08, y + h / 2))

    # Retrospective context branch: visually separate and intentionally not connected to alerts.
    branch_y = 0.18
    branch_h = 0.34
    bx1, bx2, bx3 = 2.0, 4.25, 6.55
    bw = 1.75
    for x, text in [
        (bx1, "OSM + PLUTO 2026"),
        (bx2, "Retrospective\ncontext check"),
        (bx3, "Not used for\nprospective alerts"),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, branch_y),
                bw,
                branch_h,
                boxstyle="round,pad=0.015,rounding_size=0.035",
                linewidth=0.9,
                linestyle="--",
                edgecolor=colors["gray"],
                facecolor="#ffffff",
            )
        )
        ax.text(x + bw / 2, branch_y + branch_h / 2, text, ha="center", va="center", fontsize=6.7, color="#333333")
    arrow(ax, (bx1 + bw, branch_y + branch_h / 2), (bx2, branch_y + branch_h / 2), dashed=True, color=colors["gray"], lw=1.0)
    arrow(ax, (bx2 + bw, branch_y + branch_h / 2), (bx3, branch_y + branch_h / 2), dashed=True, color=colors["gray"], lw=1.0)
    ax.text(
        5.0,
        0.64,
        "Separated retrospective branch",
        ha="center",
        va="center",
        fontsize=6.2,
        color=colors["gray"],
    )

    fig.tight_layout(pad=0.03)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
