"""Generate the Figure 1 method-contribution overview.

Run from the repository root:

    python code_model/create_method_pipeline_overview.py

The script writes vector PDF/SVG assets for both manuscript folders and
temporary PNG previews for visual QA. It is deterministic and does not read
project data or use network resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRS = [
    ROOT / "paper_overleaf" / "figures",
    ROOT / "paper_springer" / "figures",
]
PREVIEW_DIR = ROOT / "tmp" / "figure1_workflow"

FIG_NAME = "method_pipeline_overview"
FIG_SIZE = (6.05, 1.75)
DPI = 220

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 10.0
BODY_SIZE = 8.6
SMALL_SIZE = 6.7

COLORS = {
    "data": {"edge": "#567CA7", "fill": "#EEF4FA", "bar": "#D7E5F2"},
    "panel": {"edge": "#B77B3A", "fill": "#FFF4E8", "bar": "#F2D9B9"},
    "eval": {"edge": "#5F8A64", "fill": "#EFF7EF", "bar": "#D9EAD8"},
    "decision": {"edge": "#71608F", "fill": "#F3F0F8", "bar": "#DDD6EB"},
    "ink": "#202020",
    "muted": "#555555",
    "arrow": "#5A5A5A",
    "soft": "#F7F7F7",
}

GRAY_COLORS = {
    "data": {"edge": "#666666", "fill": "#F4F4F4", "bar": "#E0E0E0"},
    "panel": {"edge": "#555555", "fill": "#F2F2F2", "bar": "#D4D4D4"},
    "eval": {"edge": "#6A6A6A", "fill": "#F5F5F5", "bar": "#E3E3E3"},
    "decision": {"edge": "#505050", "fill": "#F0F0F0", "bar": "#D8D8D8"},
    "ink": "#202020",
    "muted": "#555555",
    "arrow": "#5A5A5A",
    "soft": "#F7F7F7",
}


@dataclass(frozen=True)
class BoxSpec:
    key: str
    title: str
    x: float
    y: float
    w: float
    h: float
    emphasized: bool = False
    badge: str | None = None


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.0,
        }
    )


def add_round_box(ax: plt.Axes, spec: BoxSpec, palette: dict[str, dict[str, str] | str]) -> None:
    colors = palette[spec.key]
    assert isinstance(colors, dict)
    lw = 1.35 if spec.emphasized else 0.85
    patch = FancyBboxPatch(
        (spec.x, spec.y),
        spec.w,
        spec.h,
        boxstyle="round,pad=0.006,rounding_size=0.016",
        linewidth=lw,
        edgecolor=colors["edge"],
        facecolor=colors["fill"],
    )
    ax.add_patch(patch)
    ax.add_patch(
        Rectangle(
            (spec.x + 0.006, spec.y + spec.h - 0.115),
            spec.w - 0.012,
            0.105,
            linewidth=0,
            facecolor=colors["bar"],
        )
    )
    ax.text(
        spec.x + spec.w / 2,
        spec.y + spec.h - 0.062,
        spec.title,
        ha="center",
        va="center",
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=palette["ink"],
        linespacing=1.05,
    )
    if spec.badge:
        ax.text(
            spec.x + spec.w - 0.012,
            spec.y + spec.h + 0.036,
            spec.badge,
            ha="right",
            va="center",
            fontsize=SMALL_SIZE,
            fontweight="semibold",
            color=colors["edge"],
        )


def add_arrow(ax: plt.Axes, left: BoxSpec, right: BoxSpec, palette: dict[str, dict[str, str] | str]) -> None:
    y = left.y + left.h / 2
    arrow = FancyArrowPatch(
        (left.x + left.w + 0.008, y),
        (right.x - 0.008, y),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.1,
        color=palette["arrow"],
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def add_chip(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    edge: str,
    fill: str,
    fontsize: float = SMALL_SIZE,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.011",
            linewidth=0.7,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"])


def populate_data_box(ax: plt.Axes, spec: BoxSpec, palette: dict[str, dict[str, str] | str]) -> None:
    ax.text(
        spec.x + spec.w / 2,
        spec.y + 0.335,
        "311  |  Weather\nPOI  |  Land use",
        ha="center",
        va="center",
        fontsize=BODY_SIZE,
        color=palette["ink"],
        linespacing=1.1,
    )
    colors = palette[spec.key]
    assert isinstance(colors, dict)
    chip_w = (spec.w - 0.062) / 2
    for i, label in enumerate(["311", "Weather", "POI", "Land use"]):
        row = i // 2
        col = i % 2
        add_chip(
            ax,
            spec.x + 0.024 + col * (chip_w + 0.014),
            spec.y + 0.095 + (1 - row) * 0.082,
            chip_w,
            0.052,
            label,
            colors["edge"],
            "#FFFFFF",
        )


def populate_panel_box(ax: plt.Axes, spec: BoxSpec, palette: dict[str, dict[str, str] | str]) -> None:
    ax.text(
        spec.x + spec.w / 2,
        spec.y + 0.345,
        "NTA x Week x Category",
        ha="center",
        va="center",
        fontsize=BODY_SIZE,
        color=palette["ink"],
    )
    ax.text(
        spec.x + spec.w / 2,
        spec.y + 0.255,
        "History at t\n-> Event at t+1",
        ha="center",
        va="center",
        fontsize=BODY_SIZE,
        fontweight="semibold",
        color=palette["ink"],
        linespacing=1.1,
    )
    ax.text(
        spec.x + spec.w / 2,
        spec.y + 0.14,
        "shifted history only",
        ha="center",
        va="center",
        fontsize=SMALL_SIZE,
        color=palette["muted"],
    )


def populate_eval_box(ax: plt.Axes, spec: BoxSpec, palette: dict[str, dict[str, str] | str]) -> None:
    ax.text(
        spec.x + spec.w / 2,
        spec.y + 0.35,
        "LightGBM + XGBoost",
        ha="center",
        va="center",
        fontsize=BODY_SIZE,
        fontweight="semibold",
        color=palette["ink"],
    )
    colors = palette[spec.key]
    assert isinstance(colors, dict)
    bar_x = spec.x + 0.026
    bar_y = spec.y + 0.145
    bar_w = spec.w - 0.052
    bar_h = 0.066
    segments = [
        (0.00, 0.51, "2015-22", "Train", colors["bar"]),
        (0.51, 0.73, "2023", "Validation", "#FFFFFF"),
        (0.73, 1.00, "2024-25", "Future\ntest", "#FFFFFF"),
    ]
    for start, end, top, bottom, fill in segments:
        x = bar_x + bar_w * start
        w = bar_w * (end - start)
        ax.add_patch(Rectangle((x, bar_y), w, bar_h, linewidth=0.7, edgecolor=colors["edge"], facecolor=fill))
        ax.text(x + w / 2, bar_y + bar_h + 0.031, top, ha="center", va="bottom", fontsize=SMALL_SIZE, color=palette["ink"])
        ax.text(
            x + w / 2,
            bar_y - 0.025,
            bottom,
            ha="center",
            va="top",
            fontsize=SMALL_SIZE,
            color=palette["muted"],
            linespacing=0.9,
        )


def populate_decision_box(ax: plt.Axes, spec: BoxSpec, palette: dict[str, dict[str, str] | str]) -> None:
    ax.text(
        spec.x + spec.w / 2,
        spec.y + 0.36,
        "Validation-only\ncategory thresholds",
        ha="center",
        va="center",
        fontsize=BODY_SIZE,
        color=palette["ink"],
        linespacing=1.1,
    )
    ax.text(
        spec.x + spec.w / 2,
        spec.y + 0.265,
        "score -> threshold -> alert",
        ha="center",
        va="center",
        fontsize=SMALL_SIZE + 0.2,
        color=palette["ink"],
        linespacing=1.15,
    )
    colors = palette[spec.key]
    assert isinstance(colors, dict)
    add_chip(ax, spec.x + 0.028, spec.y + 0.105, 0.078, 0.056, "Risk alerts", colors["edge"], "#FFFFFF", 6.0)
    add_chip(ax, spec.x + spec.w - 0.108, spec.y + 0.105, 0.080, 0.056, "SHAP review", colors["edge"], "#FFFFFF", 6.0)


def render_figure(output_path: Path, palette: dict[str, dict[str, str] | str], fmt: str) -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    margin = 0.014
    gap = 0.019
    box_w = (1 - 2 * margin - 3 * gap) / 4
    y = 0.16
    h = 0.70
    boxes = [
        BoxSpec("data", "Multi-source\nurban signals", margin, y, box_w, h),
        BoxSpec("panel", "Leakage-safe\npanel", margin + (box_w + gap), y, box_w, h, True, "Key design 1"),
        BoxSpec("eval", "Prospective\nevaluation", margin + 2 * (box_w + gap), y, box_w, h),
        BoxSpec("decision", "Category\ncalibrated alerts", margin + 3 * (box_w + gap), y, box_w, h, True, "Key design 2"),
    ]

    for spec in boxes:
        add_round_box(ax, spec, palette)
    for left, right in zip(boxes, boxes[1:]):
        add_arrow(ax, left, right, palette)

    populate_data_box(ax, boxes[0], palette)
    populate_panel_box(ax, boxes[1], palette)
    populate_eval_box(ax, boxes[2], palette)
    populate_decision_box(ax, boxes[3], palette)

    fig.savefig(output_path, format=fmt, bbox_inches="tight", pad_inches=0.015, dpi=DPI)
    plt.close(fig)


def main() -> None:
    for figure_dir in FIGURE_DIRS:
        figure_dir.mkdir(parents=True, exist_ok=True)
        render_figure(figure_dir / f"{FIG_NAME}.pdf", COLORS, "pdf")
        render_figure(figure_dir / f"{FIG_NAME}.svg", COLORS, "svg")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    render_figure(PREVIEW_DIR / f"{FIG_NAME}_preview.png", COLORS, "png")
    render_figure(PREVIEW_DIR / f"{FIG_NAME}_grayscale_preview.png", GRAY_COLORS, "png")


if __name__ == "__main__":
    main()
