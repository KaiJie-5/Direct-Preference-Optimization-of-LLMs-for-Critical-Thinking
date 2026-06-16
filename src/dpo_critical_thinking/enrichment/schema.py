from __future__ import annotations

import json
import re
from typing import Any

from .data import DatasetRecord


SAMPLE_SCHEMA_VERSION = "segment_enrichment_sample_v1"

BASE_SAMPLE_REQUIRED_FIELDS = {
    "schema_version",
    "record_id",
    "codebook_version",
    "analysis_unit",
    "candidate_code_matches",
    "possible_new_codes",
    "code_quality_examples",
    "reflective_question_candidates",
    "quality_control",
}

SAMPLE_REQUIRED_FIELDS = BASE_SAMPLE_REQUIRED_FIELDS | {
    "research_question_relevance",
    "contrastive_judgement",
}

CODE_QUALITY_EXAMPLE_FIELDS = {
    "wrong_code",
    "descriptive_not_answering_research_question",
    "too_broad_code",
    "useful_analytical_code",
}

CODE_QUALITY_EXAMPLE_REQUIRED_STRING_FIELDS = {
    "wrong_code": {
        "code_label",
        "actual_segment_quote",
        "why_plausible_for_wider_dataset",
        "why_unsupported_by_this_segment",
        "relation_to_research_questions",
        "category_boundary",
    },
    "descriptive_not_answering_research_question": {
        "code_label",
        "evidence_quote",
        "surface_description",
        "why_true_of_segment",
        "why_not_useful_for_research_questions",
        "relation_to_research_questions",
        "category_boundary",
    },
    "too_broad_code": {
        "code_label",
        "evidence_quote",
        "broad_relevance_to_research_questions",
        "specific_meaning_lost",
        "why_it_is_too_broad",
        "relation_to_research_questions",
        "category_boundary",
    },
    "useful_analytical_code": {
        "code_label",
        "evidence_quote",
        "specific_analytical_insight",
        "why_it_is_useful",
        "relation_to_research_questions",
        "why_better_than_other_three",
        "category_boundary",
    },
}

CONTRASTIVE_JUDGEMENT_REQUIRED_STRING_FIELDS = {
    "wrong_vs_descriptive",
    "descriptive_vs_too_broad",
    "too_broad_vs_useful",
    "final_preference_reason",
}

REFLECTIVE_QUESTION_REQUIRED_STRING_FIELDS = {
    "question",
    "question_type",
    "reflexive_dimension",
    "trigger_quote",
    "why_this_question_is_useful",
    "what_human_researcher_should_inspect",
    "risk_if_ignored",
}


def split_response_sections(text: str) -> dict[str, str]:
    think_open = "<think>"
    think_close = "</think>"
    start = text.find(think_open)
    if start == -1:
        return {
            "reasoning_text": "",
            "reasoning_block": "",
            "json_text": text.strip(),
            "reasoning_parse_status": "no_think_block",
        }

    reasoning_start = start + len(think_open)
    end = text.find(think_close, reasoning_start)
    if end == -1:
        return {
            "reasoning_text": text[reasoning_start:].strip(),
            "reasoning_block": text[start:],
            "json_text": "",
            "reasoning_parse_status": "missing_close_think_tag",
        }

    close_end = end + len(think_close)
    return {
        "reasoning_text": text[reasoning_start:end].strip(),
        "reasoning_block": text[start:close_end],
        "json_text": text[close_end:].strip(),
        "reasoning_parse_status": "found_closed_think_block",
    }


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    sections = split_response_sections(text)
    candidates = []
    if sections["json_text"]:
        candidates.extend(_json_candidates(sections["json_text"]))
    candidates.extend(_json_candidates(text))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict):
            return parsed, None
        last_error = "Parsed JSON is not an object."

    return None, locals().get("last_error", "No JSON object found.")


def _json_candidates(text: str) -> list[str]:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced_match.group(1)] if fenced_match else []

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


def validate_segment_enrichment_sample(
    payload: dict[str, Any] | None,
    record: DatasetRecord,
    *,
    expected_codebook_version: str | None = None,
    strict_prompt_schema: bool = True,
) -> list[str]:
    if payload is None:
        return ["No JSON object could be parsed."]

    errors: list[str] = []
    required_fields = (
        SAMPLE_REQUIRED_FIELDS if strict_prompt_schema else BASE_SAMPLE_REQUIRED_FIELDS
    )
    missing = sorted(required_fields - set(payload))
    if missing:
        errors.append(f"Missing required fields: {missing}")

    if payload.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SAMPLE_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )

    if payload.get("record_id") != record.record_id:
        errors.append(
            f"record_id must be {record.record_id!r}, got {payload.get('record_id')!r}"
        )

    codebook_version = expected_codebook_version or record.metadata.get("codebook_version")
    if payload.get("codebook_version") != codebook_version:
        errors.append(
            "codebook_version must match the selected codebook, expected "
            f"{codebook_version!r}, got {payload.get('codebook_version')!r}"
        )

    analysis_unit = payload.get("analysis_unit")
    if not isinstance(analysis_unit, dict):
        errors.append("analysis_unit must be an object.")
    else:
        expected = {
            "interview_id": record.metadata.get("interview_id"),
            "segment_id": record.metadata.get("segment_id"),
            "speaker": record.metadata.get("speaker"),
        }
        for key, expected_value in expected.items():
            if analysis_unit.get(key) != expected_value:
                errors.append(
                    f"analysis_unit.{key} must be {expected_value!r}, "
                    f"got {analysis_unit.get(key)!r}"
                )
        if analysis_unit.get("target_text") != record.text:
            errors.append("analysis_unit.target_text must equal the segment text.")

    for field in [
        "candidate_code_matches",
        "possible_new_codes",
    ]:
        if not isinstance(payload.get(field), list):
            errors.append(f"{field} must be a list.")

    if strict_prompt_schema:
        _validate_research_question_relevance(payload, errors)
    _validate_code_quality_examples(payload, errors, strict=strict_prompt_schema)
    if strict_prompt_schema:
        _validate_contrastive_judgement(payload, errors)
        _validate_reflective_questions(payload, errors)

    if not isinstance(payload.get("quality_control"), dict):
        errors.append("quality_control must be an object.")

    return errors


def _validate_research_question_relevance(
    payload: dict[str, Any], errors: list[str]
) -> None:
    relevance = payload.get("research_question_relevance")
    if not isinstance(relevance, dict):
        errors.append("research_question_relevance must be an object.")
        return

    if not isinstance(relevance.get("relevant_research_questions"), list):
        errors.append(
            "research_question_relevance.relevant_research_questions must be a list."
        )
    if not isinstance(relevance.get("is_segment_analytically_useful"), bool):
        errors.append(
            "research_question_relevance.is_segment_analytically_useful must be a boolean."
        )
    _validate_string_fields(
        relevance,
        "research_question_relevance",
        {"segment_relevance_summary", "why_or_why_not"},
        errors,
    )


def _validate_code_quality_examples(
    payload: dict[str, Any], errors: list[str], *, strict: bool
) -> None:
    code_quality_examples = payload.get("code_quality_examples")
    if not isinstance(code_quality_examples, dict):
        errors.append("code_quality_examples must be an object.")
        return

    actual_fields = set(code_quality_examples)
    missing_examples = sorted(CODE_QUALITY_EXAMPLE_FIELDS - actual_fields)
    extra_examples = sorted(actual_fields - CODE_QUALITY_EXAMPLE_FIELDS)
    if missing_examples:
        errors.append(
            "code_quality_examples is missing required examples: "
            f"{missing_examples}"
        )
    if strict and extra_examples:
        errors.append(
            "code_quality_examples has unexpected examples: "
            f"{extra_examples}"
        )

    for field in sorted(CODE_QUALITY_EXAMPLE_FIELDS):
        example = code_quality_examples.get(field)
        if not isinstance(example, dict):
            errors.append(f"code_quality_examples.{field} must be an object.")
            continue
        if strict:
            _validate_string_fields(
                example,
                f"code_quality_examples.{field}",
                CODE_QUALITY_EXAMPLE_REQUIRED_STRING_FIELDS[field],
                errors,
            )


def _validate_contrastive_judgement(
    payload: dict[str, Any], errors: list[str]
) -> None:
    judgement = payload.get("contrastive_judgement")
    if not isinstance(judgement, dict):
        errors.append("contrastive_judgement must be an object.")
        return
    _validate_string_fields(
        judgement,
        "contrastive_judgement",
        CONTRASTIVE_JUDGEMENT_REQUIRED_STRING_FIELDS,
        errors,
    )


def _validate_reflective_questions(
    payload: dict[str, Any], errors: list[str]
) -> None:
    questions = payload.get("reflective_question_candidates")
    if not isinstance(questions, list):
        errors.append("reflective_question_candidates must be a list.")
        return
    if len(questions) != 3:
        errors.append("reflective_question_candidates must contain exactly 3 questions.")

    for index, question in enumerate(questions, start=1):
        path = f"reflective_question_candidates[{index - 1}]"
        expected_id = f"Q{index}"
        if not isinstance(question, dict):
            errors.append(f"{path} must be an object.")
            continue
        if question.get("question_id") != expected_id:
            errors.append(
                f"{path}.question_id must be {expected_id!r}, "
                f"got {question.get('question_id')!r}"
            )
        if question.get("linked_code_quality_example") != "useful_analytical_code":
            errors.append(
                f"{path}.linked_code_quality_example must be "
                "'useful_analytical_code'."
            )
        for field in ["linked_code_ids", "linked_provisional_code_ids"]:
            if not isinstance(question.get(field), list):
                errors.append(f"{path}.{field} must be a list.")
        if not isinstance(question.get("confidence"), int) or isinstance(
            question.get("confidence"), bool
        ):
            errors.append(f"{path}.confidence must be an integer.")
        _validate_string_fields(
            question,
            path,
            REFLECTIVE_QUESTION_REQUIRED_STRING_FIELDS,
            errors,
        )


def _validate_string_fields(
    payload: dict[str, Any],
    path: str,
    fields: set[str],
    errors: list[str],
) -> None:
    for field in sorted(fields):
        if field not in payload:
            errors.append(f"{path}.{field} is required.")
        elif not isinstance(payload[field], str):
            errors.append(f"{path}.{field} must be a string.")


def build_json_repair_prompt(
    *,
    original_prompt: str,
    invalid_output: str,
    errors: list[str],
) -> str:
    return (
        f"{original_prompt}\n\n"
        "Your previous response was not valid for the required JSON schema.\n"
        "Return only one corrected JSON object. Do not add markdown fences.\n\n"
        "Validation errors:\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n\nPrevious invalid response:\n"
        + invalid_output
    )
