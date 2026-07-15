"""Create a compact top-feature SHAP figure for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/model_results/prospective/shap/shap_global_importance.csv"
OUTPUTS = [
    ROOT / "paper_springer/figures/shap_top_features.pdf",
    ROOT / "paper_overleaf/figures/shap_top_features.pdf",
]


def clean_label(name: str) -> str:
    return (
        name.replace("complaint_category=", "category: ")
        .replace("_", " ")
        .replace("weather ", "weather: ")
    )


def main() -> None:
    df = pd.read_csv(INPUT).head(15).copy()
    df["label"] = df["feature"].map(clean_label)
    df = df.iloc[::-1]

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    colors = ["#4C78A8" if g == "historical_temporal" else "#F58518" for g in df["feature_group"]]
    ax.barh(df["label"], df["mean_abs_shap"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Top individual features for the LightGBM explanation component")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
