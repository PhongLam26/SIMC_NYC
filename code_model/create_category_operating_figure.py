from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "model_results" / "paper_tables_target_week_fulltrain" / "paper_table_04_category_performance.csv"
OUTPUTS = [
    ROOT / "paper_springer" / "figures" / "category_operating_points.pdf",
    ROOT / "paper_overleaf" / "figures" / "category_operating_points.pdf",
]


def main() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"

    df = pd.read_csv(INPUT)
    df = df.sort_values("f1", ascending=True).reset_index(drop=True)
    labels = df["complaint_category"].str.replace("_", " ", regex=False)
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 3.95))
    ax.hlines(y, df["positive_share"], df["predicted_positive_share"], color="#b8b8b8", linewidth=1.6)
    ax.scatter(df["positive_share"], y, s=46, color="#4c78a8", label="Observed positive share", zorder=3)
    ax.scatter(df["predicted_positive_share"], y, s=46, color="#f58518", label="Alert share", zorder=3)

    for idx, row in df.iterrows():
        ax.text(
            max(row["positive_share"], row["predicted_positive_share"]) + 0.006,
            idx,
            f"P={row['precision']:.2f}, R={row['recall']:.2f}",
            va="center",
            fontsize=7.5,
            color="#333333",
        )

    ax.set_yticks(list(y), labels)
    ax.set_xlim(0.08, 0.27)
    ax.set_xlabel("Share of category-week-neighborhood rows in the held-out test period")
    ax.set_title("Category-specific operating points on the 2024-2025 test period", pad=8)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
