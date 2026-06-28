from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CANDIDATE_LABELS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True, slots=True)
class ReviewBlock:
    id: str
    title: str
    kind: str
    source_name: str | None = None
    question_index: int | None = None
    fields: tuple[str, ...] = ()


REVIEW_BLOCKS: tuple[ReviewBlock, ...] = (
    ReviewBlock(
        id="wrong_code",
        title="Wrong code",
        kind="code",
        source_name="wrong_code",
        fields=(
            "code_label",
            "actual_segment_quote",
            "why_plausible_for_wider_dataset",
            "why_unsupported_by_this_segment",
            "relation_to_research_questions",
            "category_boundary",
        ),
    ),
    ReviewBlock(
        id="descriptive_not_answering_research_question",
        title="Descriptive but not answering the research question",
        kind="code",
        source_name="descriptive_not_answering_research_question",
        fields=(
            "code_label",
            "evidence_quote",
            "surface_description",
            "why_true_of_segment",
            "why_not_useful_for_research_questions",
            "relation_to_research_questions",
            "category_boundary",
        ),
    ),
    ReviewBlock(
        id="too_broad_code",
        title="Too broad code",
        kind="code",
        source_name="too_broad_code",
        fields=(
            "code_label",
            "evidence_quote",
            "broad_relevance_to_research_questions",
            "specific_meaning_lost",
            "why_it_is_too_broad",
            "relation_to_research_questions",
            "category_boundary",
        ),
    ),
    ReviewBlock(
        id="useful_analytical_code",
        title="Useful analytical code",
        kind="code",
        source_name="useful_analytical_code",
        fields=(
            "code_label",
            "evidence_quote",
            "specific_analytical_insight",
            "why_it_is_useful",
            "relation_to_research_questions",
            "category_boundary",
        ),
    ),
    ReviewBlock(
        id="reflective_question_q1",
        title="Reflective question Q1",
        kind="reflective_question",
        question_index=0,
        fields=(
            "question",
            "question_type",
            "reflexive_dimension",
            "trigger_quote",
            "why_this_question_is_useful",
            "what_human_researcher_should_inspect",
            "risk_if_ignored",
            "confidence",
        ),
    ),
    ReviewBlock(
        id="reflective_question_q2",
        title="Reflective question Q2",
        kind="reflective_question",
        question_index=1,
        fields=(
            "question",
            "question_type",
            "reflexive_dimension",
            "trigger_quote",
            "why_this_question_is_useful",
            "what_human_researcher_should_inspect",
            "risk_if_ignored",
            "confidence",
        ),
    ),
    ReviewBlock(
        id="reflective_question_q3",
        title="Reflective question Q3",
        kind="reflective_question",
        question_index=2,
        fields=(
            "question",
            "question_type",
            "reflexive_dimension",
            "trigger_quote",
            "why_this_question_is_useful",
            "what_human_researcher_should_inspect",
            "risk_if_ignored",
            "confidence",
        ),
    ),
)

REVIEW_BLOCK_BY_ID = {block.id: block for block in REVIEW_BLOCKS}


def configured_review_blocks(block_ids: list[str] | None) -> tuple[ReviewBlock, ...]:
    if not block_ids:
        return REVIEW_BLOCKS
    unknown = [block_id for block_id in block_ids if block_id not in REVIEW_BLOCK_BY_ID]
    if unknown:
        raise ValueError(f"Unknown review block ids: {unknown}")
    return tuple(REVIEW_BLOCK_BY_ID[block_id] for block_id in block_ids)


def validate_ranking_payload(
    payload: dict[str, Any] | None,
    *,
    record_id: str,
    review_block: str,
    candidate_labels: tuple[str, ...] = CANDIDATE_LABELS,
) -> list[str]:
    errors: list[str] = []
    if payload is None:
        return ["No JSON object could be parsed."]

    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        errors.append("ranking must be a list.")
    else:
        normalized = [str(label) for label in ranking]
        if normalized != ranking:
            errors.append("ranking labels must be strings.")
        if len(normalized) != len(candidate_labels):
            errors.append(
                f"ranking must contain exactly {len(candidate_labels)} candidate labels."
            )
        if sorted(normalized) != sorted(candidate_labels):
            errors.append(
                f"ranking must be a complete permutation of {list(candidate_labels)}."
            )

    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("rationale must be a non-empty string.")

    output_record_id = payload.get("record_id")
    if output_record_id is not None and output_record_id != record_id:
        errors.append(
            f"record_id must be {record_id!r} when supplied, got {output_record_id!r}."
        )

    output_block = payload.get("review_block")
    if output_block is not None and output_block != review_block:
        errors.append(
            f"review_block must be {review_block!r} when supplied, got {output_block!r}."
        )

    return errors


def ranking_from_payload(payload: dict[str, Any]) -> list[str]:
    return [str(label) for label in payload["ranking"]]
