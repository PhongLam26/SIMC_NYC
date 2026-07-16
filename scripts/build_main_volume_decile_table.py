"""Build the main-manuscript historical-volume decile table from stored T2 predictions.

No model is fitted here. Deciles use only the feature-week historical reporting
volume that was available when each held-out row was scored; a deterministic
tie-breaker gives the required ten equally sized descriptive groups despite the
large mass of zero-valued histories.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
TARGET = "T2_min_count_3"


def precision_at_5pct(y: np.ndarray, score: np.ndarray) -> float:
    k = max(1, int(math.ceil(len(y) * 0.05)))
    return float(np.mean(y[np.argsort(score)[::-1][:k]]))


def main() -> None:
    predictions = pd.read_csv(ROOT / "data/processed/model_results/major_revision/bootstrap/bootstrap_prediction_rows.csv.gz")
    predictions = predictions[(predictions["target_definition"].eq(TARGET)) & (predictions["fold_id"].eq("final_style_2025"))].copy()
    predictions["target_week"] = pd.to_datetime(predictions["target_week"]).dt.strftime("%Y-%m-%d")

    dataset = ROOT / "data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz"
    source = pd.read_csv(
        dataset,
        usecols=["target_year", "target_week", "nta2020", "complaint_category", "rolling_8w_mean", "final_train_ready_flag"],
        parse_dates=["target_week"],
    )
    source = source[(source["target_year"].eq(2025)) & (source["final_train_ready_flag"].eq(1))].copy()
    source["target_week"] = source["target_week"].dt.strftime("%Y-%m-%d")

    frame = predictions.merge(
        source.drop(columns=["target_year", "final_train_ready_flag"]),
        on=["target_week", "nta2020", "complaint_category"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != 122_616 or int(frame["y_true"].sum()) != 13_562:
        raise RuntimeError("Unexpected final held-out T2 population after volume merge.")

    # Ties at zero are extensive. Ranking only by observed historical volume and
    # stable row identifiers avoids label use while retaining D1--D10.
    frame = frame.sort_values(
        ["rolling_8w_mean", "nta2020", "complaint_category", "target_week"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["volume_decile"] = np.floor(np.arange(len(frame)) * 10 / len(frame)).astype(int) + 1
    frame["alert"] = frame["platt_score"] >= frame["platt_threshold"]

    rows: list[dict[str, object]] = []
    for decile, group in frame.groupby("volume_decile", sort=True):
        y = group["y_true"].to_numpy(dtype=int)
        score = group["platt_score"].to_numpy(dtype=float)
        alert = group["alert"].to_numpy(dtype=bool)
        rows.append(
            {
                "volume_decile": f"D{decile}",
                "rows": len(group),
                "historical_volume_min": float(group["rolling_8w_mean"].min()),
                "historical_volume_median": float(group["rolling_8w_mean"].median()),
                "historical_volume_max": float(group["rolling_8w_mean"].max()),
                "event_prevalence": float(y.mean()),
                "pr_auc": float(average_precision_score(y, score)),
                "f1": float(f1_score(y, alert, zero_division=0)),
                "precision_at_5pct": precision_at_5pct(y, score),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 10 or int(result["rows"].sum()) != 122_616:
        raise RuntimeError("Expected ten complete descriptive volume deciles.")

    output = ROOT / "data/processed/model_results/major_revision/supplementary/main_volume_decile_performance_2025.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    table_rows = [
        f"{r.volume_decile} & {r.event_prevalence:.3f} & {r.pr_auc:.3f} & {r.f1:.3f} & {r.precision_at_5pct:.3f} \\\\"
        for r in result.itertuples()
    ]
    table = "\n".join(
        [
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Decile & Prevalence & PR-AUC & F1 & P@5\% \\",
            r"\midrule",
            *table_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    for paper in [ROOT / "paper_springer", ROOT / "paper_overleaf"]:
        path = paper / "tables/volume_deciles_main.tex"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table, encoding="utf-8")

    print(output.relative_to(ROOT))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
