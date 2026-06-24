from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HUMAN_RANK_COLUMNS = (
    "rank_1_most_preferable",
    "rank_2",
    "rank_3",
    "rank_4",
    "rank_5_least_preferable",
)

MODEL_RANK_COLUMNS = (
    "final_rank_1",
    "final_rank_2",
    "final_rank_3",
    "final_rank_4",
    "final_rank_5",
)


def compare_rankings(
    *,
    model_csv: Path,
    reviewer_csvs: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_rows = _read_csv(model_csv)
    model_by_key = {
        _key(row): row for row in model_rows if row.get("status") == "success"
    }
    alignment_rows: list[dict[str, Any]] = []
    for reviewer_csv in reviewer_csvs:
        for human_row in _read_csv(reviewer_csv):
            model_row = model_by_key.get(_key(human_row))
            if model_row is None:
                continue
            human_ranking = [human_row[column] for column in HUMAN_RANK_COLUMNS]
            model_ranking = [model_row[column] for column in MODEL_RANK_COLUMNS]
            alignment_rows.append(
                {
                    "reviewer_name": human_row.get("reviewer_name", ""),
                    "dataset": human_row["dataset"],
                    "record_id": human_row["record_id"],
                    "review_block": human_row["review_block"],
                    "human_ranking": " > ".join(human_ranking),
                    "model_ranking": " > ".join(model_ranking),
                    "human_top1": human_ranking[0],
                    "model_top1": model_ranking[0],
                    "top1_match": human_ranking[0] == model_ranking[0],
                    "full_rank_match": human_ranking == model_ranking,
                    "rank_distance_sum": _rank_distance_sum(
                        human_ranking, model_ranking
                    ),
                    "spearman": _spearman(human_ranking, model_ranking),
                }
            )

    alignment_path = output_dir / "qual_multi_agent_alignment_rows.csv"
    summary_path = output_dir / "qual_multi_agent_alignment_summary.csv"
    _write_csv(alignment_path, alignment_rows)
    _write_csv(summary_path, _summary_rows(alignment_rows))
    return {"alignment_rows": alignment_path, "summary": summary_path}


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["dataset"], row["record_id"], row["review_block"])


def _rank_distance_sum(human: list[str], model: list[str]) -> int:
    model_positions = {label: index for index, label in enumerate(model, start=1)}
    return sum(
        abs(index - model_positions.get(label, index))
        for index, label in enumerate(human, start=1)
    )


def _spearman(human: list[str], model: list[str]) -> float:
    n = len(human)
    model_positions = {label: index for index, label in enumerate(model, start=1)}
    d_squared = sum(
        (index - model_positions.get(label, index)) ** 2
        for index, label in enumerate(human, start=1)
    )
    return 1 - ((6 * d_squared) / (n * (n**2 - 1)))


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["review_block"])].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, review_block), group in sorted(groups.items()):
        summary.append(
            {
                "dataset": dataset,
                "review_block": review_block,
                "comparison_count": len(group),
                "top1_match_rate": mean(
                    1.0 if row["top1_match"] else 0.0 for row in group
                ),
                "full_rank_match_rate": mean(
                    1.0 if row["full_rank_match"] else 0.0 for row in group
                ),
                "mean_rank_distance_sum": mean(
                    float(row["rank_distance_sum"]) for row in group
                ),
                "mean_spearman": mean(float(row["spearman"]) for row in group),
            }
        )
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

