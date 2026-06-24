from __future__ import annotations

from typing import Any

from .schema import CANDIDATE_LABELS


def borda_aggregate(
    agent_rankings: list[dict[str, Any]],
    *,
    tiebreak_agent_id: str,
    candidate_labels: tuple[str, ...] = CANDIDATE_LABELS,
) -> dict[str, Any]:
    scores = {label: 0 for label in candidate_labels}
    points_by_rank = {
        index + 1: len(candidate_labels) - index
        for index in range(len(candidate_labels))
    }
    for item in agent_rankings:
        for index, label in enumerate(item["ranking"], start=1):
            scores[label] += points_by_rank[index]

    tiebreak_ranking = next(
        (
            item["ranking"]
            for item in agent_rankings
            if item["agent_id"] == tiebreak_agent_id
        ),
        list(candidate_labels),
    )
    tiebreak_position = {
        label: index for index, label in enumerate(tiebreak_ranking)
    }
    sorted_labels = sorted(
        candidate_labels,
        key=lambda label: (-scores[label], tiebreak_position.get(label, 999), label),
    )
    tiebreaks = _tiebreak_details(scores, sorted_labels, tiebreak_position)
    return {
        "ranking": sorted_labels,
        "scores": scores,
        "points_by_rank": points_by_rank,
        "tiebreak_agent_id": tiebreak_agent_id,
        "tiebreaks": tiebreaks,
        "tiebreak_applied": bool(tiebreaks),
    }


def _tiebreak_details(
    scores: dict[str, int],
    final_ranking: list[str],
    tiebreak_position: dict[str, int],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for score in sorted(set(scores.values()), reverse=True):
        tied = [label for label, value in scores.items() if value == score]
        if len(tied) < 2:
            continue
        details.append(
            {
                "score": score,
                "tied_candidates": sorted(tied),
                "applied_order": [
                    label for label in final_ranking if label in tied
                ],
                "tiebreak_positions": {
                    label: tiebreak_position.get(label) for label in tied
                },
            }
        )
    return details

