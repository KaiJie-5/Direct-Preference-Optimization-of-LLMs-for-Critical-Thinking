from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from enrichment.schema import normalize_code_label, split_response_sections


CATEGORY_ORDER = (
    "wrong_code",
    "descriptive_not_answering_research_question",
    "too_broad_code",
    "useful_analytical_code",
)


@dataclass(frozen=True, slots=True)
class ReflectiveParseResult:
    model_parsed_output: dict[str, Any] | None
    parsed_output: dict[str, Any] | None
    sections: dict[str, str]
    errors: list[str]
    validation_warnings: list[str]
    json_extraction: dict[str, Any]
    canonical_corrections: list[dict[str, Any]]


def required_output(selected_codes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reflective_questions": [
            {
                "code": item["code"]["code_label"],
                "hint": item["hint"],
                "question": "<one specific open-ended reflective question>",
            }
            for item in selected_codes
        ]
    }


def parse_and_validate_response(
    text: str,
    selected_codes: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, str], list[str]]:
    result = parse_response_result(text, selected_codes)
    return result.parsed_output, result.sections, result.errors


def parse_response_result(
    text: str,
    selected_codes: list[dict[str, Any]],
) -> ReflectiveParseResult:
    sections = split_response_sections(text)
    errors: list[str] = []
    if not text.startswith("<think>"):
        errors.append("Response must begin with <think>.")
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        errors.append("Response must contain exactly one <think>...</think> block.")
    if sections["reasoning_parse_status"] != "found_closed_think_block":
        errors.append("Response must contain one closed <think>...</think> block.")
    model_parsed: dict[str, Any] | None = None
    validation_warnings: list[str] = []
    json_extraction = _json_extraction_audit("none")
    json_text = sections["json_text"]
    if not json_text:
        errors.append("Response must contain a JSON object after </think>.")
    else:
        candidate, parse_error, json_extraction, format_warning = (
            _parse_reflective_json_object(json_text)
        )
        if parse_error:
            errors.append(parse_error)
        else:
            model_parsed = candidate
        if format_warning:
            validation_warnings.append(format_warning)
    parsed, corrections = canonicalize_reflective_payload(model_parsed, selected_codes)
    if parsed is not None:
        errors.extend(validate_payload(parsed, selected_codes))
    return ReflectiveParseResult(
        model_parsed_output=model_parsed,
        parsed_output=parsed,
        sections=sections,
        errors=errors,
        validation_warnings=validation_warnings,
        json_extraction=json_extraction,
        canonical_corrections=corrections,
    )


def _parse_reflective_json_object(
    json_text: str,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any], str | None]:
    """Parse strict JSON or one exact enclosing JSON fence.

    The fenced fallback is intentionally narrower than the general enrichment
    extractor: surrounding prose, trailing content, unsupported fence labels,
    multiple fences, and malformed fenced JSON remain invalid.
    """

    stripped = json_text.strip()
    try:
        candidate = json.loads(stripped)
    except json.JSONDecodeError as strict_exc:
        fenced_match = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_match is None:
            return (
                None,
                "Invalid strict JSON after </think>: "
                f"{strict_exc}. Expected one JSON object with no surrounding prose, "
                "or one exact enclosing ```json ... ``` fence.",
                _json_extraction_audit("none"),
                None,
            )
        fenced_body = fenced_match.group("body").strip()
        try:
            candidate = json.loads(fenced_body)
        except json.JSONDecodeError as fenced_exc:
            return (
                None,
                "Invalid JSON inside the exact enclosing fence after </think>: "
                f"{fenced_exc}",
                _json_extraction_audit("none"),
                None,
            )
        if not isinstance(candidate, dict):
            return (
                None,
                "The final JSON value inside the exact enclosing fence must be an object.",
                _json_extraction_audit("none"),
                None,
            )
        return (
            candidate,
            None,
            _json_extraction_audit(
                "exact_single_fenced_json",
                recovered_from_format_deviation=True,
                valid_fenced_object_count=1,
            ),
            "Recovered one valid JSON object from an exact enclosing Markdown fence; "
            "the response did not follow the required JSON-only format.",
        )
    if not isinstance(candidate, dict):
        return (
            None,
            "The final JSON value must be an object.",
            _json_extraction_audit("none"),
            None,
        )
    return candidate, None, _json_extraction_audit("strict_json"), None


def _json_extraction_audit(
    method: str,
    *,
    recovered_from_format_deviation: bool = False,
    valid_fenced_object_count: int = 0,
) -> dict[str, Any]:
    return {
        "method": method,
        "recovered_from_format_deviation": recovered_from_format_deviation,
        "valid_fenced_object_count": valid_fenced_object_count,
    }


def canonicalize_reflective_payload(
    payload: dict[str, Any] | None,
    selected_codes: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if payload is None:
        return None, []
    canonical = deepcopy(payload)
    corrections: list[dict[str, Any]] = []
    questions = canonical.get("reflective_questions")
    if not isinstance(questions, list):
        return canonical, corrections
    for index, selected in enumerate(selected_codes):
        if index >= len(questions) or not isinstance(questions[index], dict):
            continue
        item = questions[index]
        expected_hint = selected.get("hint")
        if item.get("hint") != expected_hint:
            continue
        expected_code = normalize_code_label(str(selected["code"]["code_label"]))
        model_code = item.get("code")
        if (
            isinstance(model_code, str)
            and model_code != expected_code
            and _separator_insensitive(model_code) == _separator_insensitive(expected_code)
        ):
            _record_correction(
                corrections,
                path=f"reflective_questions[{index}].code",
                model_value=model_code,
                canonical_value=expected_code,
                correction_type="canonical_selected_code_label",
            )
            item["code"] = expected_code
        question = item.get("question")
        if isinstance(question, str):
            repaired = _repair_label_occurrence(question, expected_code, model_code)
            if repaired is not None and repaired != question:
                _record_correction(
                    corrections,
                    path=f"reflective_questions[{index}].question",
                    model_value=question,
                    canonical_value=repaired,
                    correction_type="collapsed_code_label_in_question",
                )
                item["question"] = repaired
    return canonical, corrections


def _separator_insensitive(value: str) -> str:
    return re.sub(r"[\s_]+", "", value).casefold()


def _repair_label_occurrence(
    question: str, expected_code: str, model_code: Any
) -> str | None:
    variants = {
        "".join(expected_code.split()),
        re.sub(r"\s+", "_", expected_code),
    }
    if isinstance(model_code, str) and model_code != expected_code:
        variants.add(model_code)
    variants.discard(expected_code)
    matches: list[tuple[int, str]] = []
    folded = question.casefold()
    for variant in sorted(variants, key=len, reverse=True):
        if not variant:
            continue
        start = folded.find(variant.casefold())
        if start != -1 and folded.find(variant.casefold(), start + 1) == -1:
            matches.append((start, question[start : start + len(variant)]))
    unique_positions = {(start, matched) for start, matched in matches}
    if len(unique_positions) != 1:
        return None
    start, matched = next(iter(unique_positions))
    outside_label = question[:start] + question[start + len(matched) :]
    if not re.search(r"\s", outside_label):
        return None
    return question[:start] + expected_code + question[start + len(matched) :]


def _record_correction(
    corrections: list[dict[str, Any]],
    *,
    path: str,
    model_value: str,
    canonical_value: str,
    correction_type: str,
) -> None:
    corrections.append(
        {
            "path": path,
            "was_present": True,
            "model_value": model_value,
            "canonical_value": canonical_value,
            "correction_type": correction_type,
        }
    )


def validate_payload(
    payload: dict[str, Any], selected_codes: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    if set(payload) != {"reflective_questions"}:
        errors.append("Output must contain exactly the field 'reflective_questions'.")
    questions = payload.get("reflective_questions")
    if not isinstance(questions, list):
        return [*errors, "reflective_questions must be a list."]
    if len(questions) != len(CATEGORY_ORDER):
        errors.append("reflective_questions must contain exactly four entries.")
    seen_questions: set[str] = set()
    for index, selected in enumerate(selected_codes):
        path = f"reflective_questions[{index}]"
        if index >= len(questions):
            errors.append(f"{path} is missing.")
            continue
        item = questions[index]
        if not isinstance(item, dict):
            errors.append(f"{path} must be an object.")
            continue
        if set(item) != {"code", "hint", "question"}:
            errors.append(f"{path} must contain exactly code, hint, and question.")
        expected_code = selected["code"]["code_label"]
        if item.get("code") != expected_code:
            errors.append(f"{path}.code must exactly match {expected_code!r}.")
        expected_hint = selected["hint"]
        if item.get("hint") != expected_hint:
            errors.append(f"{path}.hint must be {expected_hint!r}.")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{path}.question must be a non-empty string.")
        elif _looks_whitespace_collapsed(question):
            errors.append(f"{path}.question appears to have collapsed whitespace.")
        elif not question.strip().endswith("?") or question.count("?") != 1:
            errors.append(f"{path}.question must contain one final question mark.")
        else:
            normalized = " ".join(question.casefold().split())
            if normalized in seen_questions:
                errors.append(f"{path}.question must be distinct.")
            seen_questions.add(normalized)
    if len(questions) > len(selected_codes):
        errors.append("reflective_questions contains unexpected additional entries.")
    return errors


def _looks_whitespace_collapsed(value: str) -> bool:
    return (
        len(value) >= 24
        and not re.search(r"\s", value)
        and len(re.findall(r"[A-Za-z]", value)) >= 20
    )
