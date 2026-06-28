from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .data import DatasetRecord


LEGACY_SAMPLE_SCHEMA_VERSION = "segment_enrichment_sample_v1"
SAMPLE_SCHEMA_VERSION = "segment_enrichment_sample_v2"
SUPPORTED_SAMPLE_SCHEMA_VERSIONS = {
    LEGACY_SAMPLE_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
}

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

V1_SAMPLE_REQUIRED_FIELDS = BASE_SAMPLE_REQUIRED_FIELDS | {
    "research_question_relevance",
    "contrastive_judgement",
}

V2_SAMPLE_REQUIRED_FIELDS = BASE_SAMPLE_REQUIRED_FIELDS | {
    "research_question_relevance",
}

CODE_QUALITY_EXAMPLE_FIELDS = {
    "wrong_code",
    "descriptive_not_answering_research_question",
    "too_broad_code",
    "useful_analytical_code",
}

V2_CODE_QUALITY_EXAMPLE_REQUIRED_STRING_FIELDS = {
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
        "category_boundary",
    },
}

V1_CODE_QUALITY_EXAMPLE_REQUIRED_STRING_FIELDS = {
    key: set(fields)
    for key, fields in V2_CODE_QUALITY_EXAMPLE_REQUIRED_STRING_FIELDS.items()
}
V1_CODE_QUALITY_EXAMPLE_REQUIRED_STRING_FIELDS["useful_analytical_code"].add(
    "why_better_than_other_three"
)

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

ANALYSIS_UNIT_V2_FIELDS = {
    "interview_id",
    "segment_id",
    "speaker",
    "target_text",
    "analysis_context_used",
    "analysis_context_scope",
    "context_warning",
}

RESEARCH_QUESTION_RELEVANCE_FIELDS = {
    "relevant_research_questions",
    "segment_relevance_summary",
    "is_segment_analytically_useful",
    "why_or_why_not",
}

CANDIDATE_CODE_MATCH_FIELDS = {
    "code_id",
    "code_label",
    "match_strength",
    "evidence_quote",
    "rationale",
    "confidence",
}

POSSIBLE_NEW_CODE_FIELDS = {
    "provisional_code_id",
    "provisional_code_label",
    "definition",
    "evidence_quote",
    "why_candidate_codes_do_not_fully_cover_it",
    "confidence",
}

REFLECTIVE_QUESTION_FIELDS = {
    "question_id",
    "question",
    "linked_code_ids",
    "linked_provisional_code_ids",
    "linked_code_quality_example",
    "question_type",
    "reflexive_dimension",
    "trigger_quote",
    "why_this_question_is_useful",
    "what_human_researcher_should_inspect",
    "risk_if_ignored",
    "confidence",
}

QUALITY_CONTROL_FIELDS = {
    "hallucination_risk",
    "over_generalisation_risk",
    "participant_voice_loss_risk",
    "needs_human_review",
    "review_reason",
    "overall_confidence",
}

MATCH_STRENGTHS = {"strong", "partial", "weak"}
RISK_LEVELS = {"low", "medium", "high"}
QUESTION_TYPES = {
    "automated_socrates",
    "devils_advocate",
    "participant_voice_check",
    "context_check",
    "methodological_check",
    "technology_check",
}
REFLEXIVE_DIMENSIONS = {
    "personal",
    "interpersonal",
    "methodological",
    "contextual",
    "technological",
}

NARRATIVE_FIELD_NAMES = {
    "segment_relevance_summary",
    "why_or_why_not",
    "rationale",
    "definition",
    "why_candidate_codes_do_not_fully_cover_it",
    "why_plausible_for_wider_dataset",
    "why_unsupported_by_this_segment",
    "relation_to_research_questions",
    "category_boundary",
    "surface_description",
    "why_true_of_segment",
    "why_not_useful_for_research_questions",
    "broad_relevance_to_research_questions",
    "specific_meaning_lost",
    "why_it_is_too_broad",
    "specific_analytical_insight",
    "why_it_is_useful",
    "question",
    "why_this_question_is_useful",
    "what_human_researcher_should_inspect",
    "risk_if_ignored",
    "review_reason",
}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]


def canonicalize_source_fields(
    payload: dict[str, Any] | None,
    record: DatasetRecord,
    *,
    expected_codebook_version: str | None,
    expected_context_scope: str,
    expected_schema_version: str = SAMPLE_SCHEMA_VERSION,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Inject fields owned by the runtime while retaining an audit trail."""
    if payload is None:
        return None, []

    canonical = deepcopy(payload)
    corrections: list[dict[str, Any]] = []
    expected_top_level = {
        "schema_version": expected_schema_version,
        "record_id": record.record_id,
        "codebook_version": expected_codebook_version,
    }
    for field, expected_value in expected_top_level.items():
        _set_canonical_field(canonical, field, expected_value, corrections)

    analysis_unit = canonical.get("analysis_unit")
    if isinstance(analysis_unit, dict):
        expected_analysis_unit = {
            "interview_id": record.metadata.get("interview_id"),
            "segment_id": record.metadata.get("segment_id"),
            "speaker": record.metadata.get("speaker"),
            "target_text": record.text,
        }
        if expected_schema_version == SAMPLE_SCHEMA_VERSION:
            expected_analysis_unit.update(
                {
                    "analysis_context_used": True,
                    "analysis_context_scope": expected_context_scope,
                }
            )
        for field, expected_value in expected_analysis_unit.items():
            _set_canonical_field(
                analysis_unit,
                f"analysis_unit.{field}",
                expected_value,
                corrections,
                storage_key=field,
            )

    _canonicalize_structured_quotes(canonical, record.text, corrections)

    return canonical, corrections


def _set_canonical_field(
    container: dict[str, Any],
    path: str,
    expected_value: Any,
    corrections: list[dict[str, Any]],
    *,
    storage_key: str | None = None,
) -> None:
    key = storage_key or path
    was_present = key in container
    model_value = container.get(key)
    if not was_present or model_value != expected_value:
        corrections.append(
            {
                "path": path,
                "was_present": was_present,
                "model_value": model_value,
                "canonical_value": expected_value,
            }
        )
        container[key] = expected_value


def _canonicalize_structured_quotes(
    payload: dict[str, Any],
    target_text: str,
    corrections: list[dict[str, Any]],
) -> None:
    for container, key, path in _structured_quote_slots(payload):
        model_value = container.get(key)
        if not isinstance(model_value, str) or model_value in target_text:
            continue
        restored = _unique_whitespace_only_source_match(model_value, target_text)
        if restored is None:
            continue
        corrections.append(
            {
                "path": path,
                "was_present": True,
                "model_value": model_value,
                "canonical_value": restored,
                "correction_type": "whitespace_only_quote_repair",
            }
        )
        container[key] = restored


def _structured_quote_slots(
    payload: dict[str, Any],
) -> list[tuple[dict[str, Any], str, str]]:
    slots: list[tuple[dict[str, Any], str, str]] = []
    for collection_name in ("candidate_code_matches", "possible_new_codes"):
        collection = payload.get(collection_name)
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if isinstance(item, dict) and "evidence_quote" in item:
                slots.append(
                    (
                        item,
                        "evidence_quote",
                        f"{collection_name}[{index}].evidence_quote",
                    )
                )

    code_quality_examples = payload.get("code_quality_examples")
    quote_fields = {
        "wrong_code": "actual_segment_quote",
        "descriptive_not_answering_research_question": "evidence_quote",
        "too_broad_code": "evidence_quote",
        "useful_analytical_code": "evidence_quote",
    }
    if isinstance(code_quality_examples, dict):
        for example_name, quote_field in quote_fields.items():
            example = code_quality_examples.get(example_name)
            if isinstance(example, dict) and quote_field in example:
                slots.append(
                    (
                        example,
                        quote_field,
                        f"code_quality_examples.{example_name}.{quote_field}",
                    )
                )

    questions = payload.get("reflective_question_candidates")
    if isinstance(questions, list):
        for index, question in enumerate(questions):
            if isinstance(question, dict) and "trigger_quote" in question:
                slots.append(
                    (
                        question,
                        "trigger_quote",
                        f"reflective_question_candidates[{index}].trigger_quote",
                    )
                )
    return slots


def _unique_whitespace_only_source_match(
    model_quote: str, target_text: str
) -> str | None:
    compact_quote = "".join(
        character for character in model_quote if not character.isspace()
    )
    if not compact_quote:
        return None

    source_characters = [
        (index, character)
        for index, character in enumerate(target_text)
        if not character.isspace()
    ]
    compact_source = "".join(character for _, character in source_characters)
    starts: list[int] = []
    search_from = 0
    while True:
        start = compact_source.find(compact_quote, search_from)
        if start == -1:
            break
        starts.append(start)
        if len(starts) > 1:
            return None
        search_from = start + 1

    if len(starts) != 1:
        return None
    start = starts[0]
    end = start + len(compact_quote) - 1
    source_start = source_characters[start][0]
    source_end = source_characters[end][0] + 1
    return target_text[source_start:source_end]


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
    candidates: list[str] = []
    if sections["json_text"]:
        candidates.extend(_json_candidates(sections["json_text"]))
    else:
        candidates.extend(_json_candidates(text))

    for candidate in candidates:
        parsed, parse_error = _parse_json_candidate(candidate)
        if parse_error:
            last_error = parse_error
            continue
        if isinstance(parsed, dict):
            return parsed, None
        last_error = "Parsed JSON is not an object."

    return None, locals().get("last_error", "No JSON object found.")


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        stripped,
        flags=re.DOTALL,
    )
    candidates = [stripped]
    if fenced_match:
        candidates.append(fenced_match.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        suffix = stripped[end + 1 :]
        if not suffix.strip():
            candidates.append(stripped[start : end + 1])
    return _dedupe_preserving_order(candidates)


def _parse_json_candidate(text: str) -> tuple[Any | None, str | None]:
    stripped = text.strip()
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError as exc:
        strict_error = str(exc)

    decoder = json.JSONDecoder()
    try:
        parsed, end_index = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None, strict_error

    remainder = stripped[end_index:].strip()
    if remainder and set(remainder) <= {"}"}:
        return parsed, None
    return None, strict_error


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def validate_segment_enrichment_sample(
    payload: dict[str, Any] | None,
    record: DatasetRecord,
    *,
    expected_codebook_version: str | None = None,
    expected_schema_version: str | None = None,
    expected_context_scope: str | None = None,
    strict_prompt_schema: bool = True,
) -> list[str]:
    return validate_segment_enrichment_sample_result(
        payload,
        record,
        expected_codebook_version=expected_codebook_version,
        expected_schema_version=expected_schema_version,
        expected_context_scope=expected_context_scope,
        strict_prompt_schema=strict_prompt_schema,
        allow_target_text_mismatch=False,
    ).errors


def validate_segment_enrichment_sample_result(
    payload: dict[str, Any] | None,
    record: DatasetRecord,
    *,
    expected_codebook_version: str | None = None,
    expected_schema_version: str | None = None,
    expected_context_scope: str | None = None,
    strict_prompt_schema: bool = True,
    allow_target_text_mismatch: bool = False,
) -> ValidationResult:
    if payload is None:
        return ValidationResult(errors=["No JSON object could be parsed."], warnings=[])

    errors: list[str] = []
    warnings: list[str] = []
    actual_schema_version = payload.get("schema_version")
    validation_schema_version = expected_schema_version or actual_schema_version
    if validation_schema_version not in SUPPORTED_SAMPLE_SCHEMA_VERSIONS:
        errors.append(
            "schema_version must be one of "
            f"{sorted(SUPPORTED_SAMPLE_SCHEMA_VERSIONS)}, "
            f"got {actual_schema_version!r}"
        )
        validation_schema_version = expected_schema_version or SAMPLE_SCHEMA_VERSION

    if expected_schema_version is not None and actual_schema_version != expected_schema_version:
        errors.append(
            "schema_version must be "
            f"{expected_schema_version!r}, got {actual_schema_version!r}"
        )

    is_v2 = validation_schema_version == SAMPLE_SCHEMA_VERSION
    if is_v2:
        required_fields = V2_SAMPLE_REQUIRED_FIELDS
    elif strict_prompt_schema:
        required_fields = V1_SAMPLE_REQUIRED_FIELDS
    else:
        required_fields = BASE_SAMPLE_REQUIRED_FIELDS
    missing = sorted(required_fields - set(payload))
    if missing:
        errors.append(f"Missing required fields: {missing}")
    if is_v2:
        _validate_exact_fields(payload, "sample", V2_SAMPLE_REQUIRED_FIELDS, errors)

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
            message = "analysis_unit.target_text differs from the segment text."
            if allow_target_text_mismatch:
                warnings.append(message)
            else:
                errors.append("analysis_unit.target_text must equal the segment text.")
        if is_v2:
            _validate_analysis_unit_v2(
                analysis_unit,
                expected_context_scope=expected_context_scope,
                errors=errors,
            )

    if is_v2:
        _validate_research_question_relevance(payload, errors, exact=True)
        candidate_ids = _validate_candidate_code_matches(payload, errors)
        provisional_ids = _validate_possible_new_codes(payload, errors)
        _validate_code_quality_examples(
            payload,
            errors,
            strict=True,
            schema_version=SAMPLE_SCHEMA_VERSION,
        )
        _validate_reflective_questions_v2(
            payload,
            errors,
            candidate_ids=candidate_ids,
            provisional_ids=provisional_ids,
        )
        _validate_quality_control_v2(payload, errors)
    else:
        for field in ["candidate_code_matches", "possible_new_codes"]:
            if not isinstance(payload.get(field), list):
                errors.append(f"{field} must be a list.")
        if strict_prompt_schema:
            _validate_research_question_relevance(payload, errors)
        _validate_code_quality_examples(
            payload,
            errors,
            strict=strict_prompt_schema,
            schema_version=LEGACY_SAMPLE_SCHEMA_VERSION,
        )
        if strict_prompt_schema:
            _validate_contrastive_judgement(payload, errors)
            _validate_reflective_questions(payload, errors)
        if not isinstance(payload.get("quality_control"), dict):
            errors.append("quality_control must be an object.")

    if is_v2:
        _validate_structured_quotes(payload, record.text, errors)
        collapsed_paths = _collapsed_narrative_paths(payload)
        if len(collapsed_paths) >= 3:
            errors.append(
                "Generated analytical prose appears to have broadly collapsed "
                "whitespace in fields: " + ", ".join(collapsed_paths[:8])
            )

    return ValidationResult(errors=errors, warnings=warnings)


def _validate_structured_quotes(
    payload: dict[str, Any], target_text: str, errors: list[str]
) -> None:
    for container, key, path in _structured_quote_slots(payload):
        quote = container.get(key)
        if isinstance(quote, str) and quote and quote in target_text:
            continue
        if isinstance(quote, str):
            errors.append(
                f"{path} must be an exact non-empty substring of the target segment."
            )


def _collapsed_narrative_paths(payload: dict[str, Any]) -> list[str]:
    collapsed: list[str] = []

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                item_path = f"{path}.{key}" if path else key
                if key in NARRATIVE_FIELD_NAMES and _looks_whitespace_collapsed(item):
                    collapsed.append(item_path)
                visit(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(payload)
    return collapsed


def _looks_whitespace_collapsed(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 24:
        return False
    if re.search(r"\s", value):
        return False
    return len(re.findall(r"[A-Za-z]", value)) >= 20


def _validate_research_question_relevance(
    payload: dict[str, Any], errors: list[str], *, exact: bool = False
) -> None:
    relevance = payload.get("research_question_relevance")
    if not isinstance(relevance, dict):
        errors.append("research_question_relevance must be an object.")
        return

    if exact:
        _validate_exact_fields(
            relevance,
            "research_question_relevance",
            RESEARCH_QUESTION_RELEVANCE_FIELDS,
            errors,
        )

    questions = relevance.get("relevant_research_questions")
    if not isinstance(questions, list):
        errors.append(
            "research_question_relevance.relevant_research_questions must be a list."
        )
    elif exact and any(not isinstance(question, str) for question in questions):
        errors.append(
            "research_question_relevance.relevant_research_questions "
            "must contain only strings."
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
    payload: dict[str, Any],
    errors: list[str],
    *,
    strict: bool,
    schema_version: str,
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
            expected_fields = (
                V2_CODE_QUALITY_EXAMPLE_REQUIRED_STRING_FIELDS[field]
                if schema_version == SAMPLE_SCHEMA_VERSION
                else V1_CODE_QUALITY_EXAMPLE_REQUIRED_STRING_FIELDS[field]
            )
            if schema_version == SAMPLE_SCHEMA_VERSION:
                _validate_exact_fields(
                    example,
                    f"code_quality_examples.{field}",
                    expected_fields,
                    errors,
                )
            _validate_string_fields(
                example,
                f"code_quality_examples.{field}",
                expected_fields,
                errors,
            )


def _validate_analysis_unit_v2(
    analysis_unit: dict[str, Any],
    *,
    expected_context_scope: str | None,
    errors: list[str],
) -> None:
    _validate_exact_fields(
        analysis_unit,
        "analysis_unit",
        ANALYSIS_UNIT_V2_FIELDS,
        errors,
    )
    _validate_string_fields(
        analysis_unit,
        "analysis_unit",
        {
            "interview_id",
            "segment_id",
            "speaker",
            "target_text",
            "analysis_context_scope",
            "context_warning",
        },
        errors,
    )
    if analysis_unit.get("analysis_context_used") is not True:
        errors.append("analysis_unit.analysis_context_used must be true.")
    if (
        expected_context_scope is not None
        and analysis_unit.get("analysis_context_scope") != expected_context_scope
    ):
        errors.append(
            "analysis_unit.analysis_context_scope must match the runtime context "
            f"scope {expected_context_scope!r}, got "
            f"{analysis_unit.get('analysis_context_scope')!r}."
        )


def _validate_candidate_code_matches(
    payload: dict[str, Any], errors: list[str]
) -> set[str]:
    matches = payload.get("candidate_code_matches")
    if not isinstance(matches, list):
        errors.append("candidate_code_matches must be a list.")
        return set()

    code_ids: set[str] = set()
    for index, match in enumerate(matches):
        path = f"candidate_code_matches[{index}]"
        if not isinstance(match, dict):
            errors.append(f"{path} must be an object.")
            continue
        _validate_exact_fields(match, path, CANDIDATE_CODE_MATCH_FIELDS, errors)
        _validate_string_fields(
            match,
            path,
            {"code_id", "code_label", "match_strength", "evidence_quote", "rationale"},
            errors,
        )
        _validate_enum(
            match.get("match_strength"),
            path,
            "match_strength",
            MATCH_STRENGTHS,
            errors,
        )
        _validate_confidence(match.get("confidence"), path, "confidence", errors)
        code_id = match.get("code_id")
        if isinstance(code_id, str):
            if code_id in code_ids:
                errors.append(f"{path}.code_id must be unique within candidate_code_matches.")
            code_ids.add(code_id)
    return code_ids


def _validate_possible_new_codes(
    payload: dict[str, Any], errors: list[str]
) -> set[str]:
    codes = payload.get("possible_new_codes")
    if not isinstance(codes, list):
        errors.append("possible_new_codes must be a list.")
        return set()

    provisional_ids: set[str] = set()
    string_fields = POSSIBLE_NEW_CODE_FIELDS - {"confidence"}
    for index, code in enumerate(codes):
        path = f"possible_new_codes[{index}]"
        if not isinstance(code, dict):
            errors.append(f"{path} must be an object.")
            continue
        _validate_exact_fields(code, path, POSSIBLE_NEW_CODE_FIELDS, errors)
        _validate_string_fields(code, path, string_fields, errors)
        _validate_confidence(code.get("confidence"), path, "confidence", errors)
        provisional_id = code.get("provisional_code_id")
        if isinstance(provisional_id, str):
            if provisional_id in provisional_ids:
                errors.append(
                    f"{path}.provisional_code_id must be unique within possible_new_codes."
                )
            provisional_ids.add(provisional_id)
    return provisional_ids


def _validate_reflective_questions_v2(
    payload: dict[str, Any],
    errors: list[str],
    *,
    candidate_ids: set[str],
    provisional_ids: set[str],
) -> None:
    questions = payload.get("reflective_question_candidates")
    if not isinstance(questions, list):
        errors.append("reflective_question_candidates must be a list.")
        return
    if len(questions) != 3:
        errors.append("reflective_question_candidates must contain exactly 3 questions.")

    for index, question in enumerate(questions, start=1):
        path = f"reflective_question_candidates[{index - 1}]"
        if not isinstance(question, dict):
            errors.append(f"{path} must be an object.")
            continue
        _validate_exact_fields(question, path, REFLECTIVE_QUESTION_FIELDS, errors)
        _validate_string_fields(
            question,
            path,
            REFLECTIVE_QUESTION_REQUIRED_STRING_FIELDS
            | {"question_id", "linked_code_quality_example"},
            errors,
        )
        expected_id = f"Q{index}"
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
        linked_code_ids = _validate_string_list(
            question.get("linked_code_ids"), path, "linked_code_ids", errors
        )
        linked_provisional_ids = _validate_string_list(
            question.get("linked_provisional_code_ids"),
            path,
            "linked_provisional_code_ids",
            errors,
        )
        unknown_codes = sorted(set(linked_code_ids) - candidate_ids)
        if unknown_codes:
            errors.append(f"{path}.linked_code_ids contains unknown ids: {unknown_codes}.")
        unknown_provisional = sorted(set(linked_provisional_ids) - provisional_ids)
        if unknown_provisional:
            errors.append(
                f"{path}.linked_provisional_code_ids contains unknown ids: "
                f"{unknown_provisional}."
            )
        _validate_enum(
            question.get("question_type"),
            path,
            "question_type",
            QUESTION_TYPES,
            errors,
        )
        _validate_enum(
            question.get("reflexive_dimension"),
            path,
            "reflexive_dimension",
            REFLEXIVE_DIMENSIONS,
            errors,
        )
        _validate_confidence(question.get("confidence"), path, "confidence", errors)


def _validate_quality_control_v2(
    payload: dict[str, Any], errors: list[str]
) -> None:
    quality = payload.get("quality_control")
    if not isinstance(quality, dict):
        errors.append("quality_control must be an object.")
        return
    _validate_exact_fields(quality, "quality_control", QUALITY_CONTROL_FIELDS, errors)
    _validate_string_fields(
        quality,
        "quality_control",
        {
            "hallucination_risk",
            "over_generalisation_risk",
            "participant_voice_loss_risk",
            "review_reason",
        },
        errors,
    )
    for field in [
        "hallucination_risk",
        "over_generalisation_risk",
        "participant_voice_loss_risk",
    ]:
        _validate_enum(quality.get(field), "quality_control", field, RISK_LEVELS, errors)
    if quality.get("needs_human_review") is not True:
        errors.append("quality_control.needs_human_review must be true.")
    _validate_confidence(
        quality.get("overall_confidence"),
        "quality_control",
        "overall_confidence",
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


def _validate_exact_fields(
    payload: dict[str, Any],
    path: str,
    expected_fields: set[str],
    errors: list[str],
) -> None:
    actual_fields = set(payload)
    missing = sorted(expected_fields - actual_fields)
    extra = sorted(actual_fields - expected_fields)
    if missing:
        errors.append(f"{path} is missing required fields: {missing}.")
    if extra:
        errors.append(f"{path} has unexpected fields: {extra}.")


def _validate_enum(
    value: Any,
    path: str,
    field: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    if not isinstance(value, str) or value not in allowed:
        errors.append(
            f"{path}.{field} must be one of {sorted(allowed)}, got {value!r}."
        )


def _validate_confidence(
    value: Any,
    path: str,
    field: str,
    errors: list[str],
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        errors.append(f"{path}.{field} must be an integer from 1 to 10.")


def _validate_string_list(
    value: Any,
    path: str,
    field: str,
    errors: list[str],
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{path}.{field} must be a list.")
        return []
    if any(not isinstance(item, str) for item in value):
        errors.append(f"{path}.{field} must contain only strings.")
        return []
    return value


def build_json_repair_prompt(
    *,
    original_prompt: str,
    invalid_output: str,
    parsed_output: dict[str, Any] | None,
    errors: list[str],
) -> str:
    if parsed_output is None:
        previous_candidate = invalid_output
        candidate_label = "Previous unparseable response"
    else:
        previous_candidate = json.dumps(parsed_output, ensure_ascii=False)
        candidate_label = "Previous parsed JSON candidate"
    return (
        f"{original_prompt}\n\n"
        "Your previous response was not valid for the required response format.\n"
        "Regenerate a concise <think>...</think> reasoning block followed by one "
        "corrected JSON object. Do not add markdown fences or text after the JSON.\n\n"
        "Validation errors:\n"
        + "\n".join(f"- {error}" for error in errors)
        + f"\n\n{candidate_label}:\n"
        + previous_candidate
    )
