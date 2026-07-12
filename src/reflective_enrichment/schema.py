from __future__ import annotations

import json
from typing import Any

from enrichment.schema import split_response_sections


CATEGORY_ORDER = (
    "wrong_code",
    "descriptive_not_answering_research_question",
    "too_broad_code",
    "useful_analytical_code",
)


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
    sections = split_response_sections(text)
    errors: list[str] = []
    if not text.startswith("<think>"):
        errors.append("Response must begin with <think>.")
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        errors.append("Response must contain exactly one <think>...</think> block.")
    if sections["reasoning_parse_status"] != "found_closed_think_block":
        errors.append("Response must contain one closed <think>...</think> block.")
    parsed: dict[str, Any] | None = None
    json_text = sections["json_text"]
    if not json_text:
        errors.append("Response must contain a JSON object after </think>.")
    else:
        try:
            candidate = json.loads(json_text)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid strict JSON after </think>: {exc}")
        else:
            if isinstance(candidate, dict):
                parsed = candidate
            else:
                errors.append("The final JSON value must be an object.")
    if parsed is not None:
        errors.extend(validate_payload(parsed, selected_codes))
    return parsed, sections, errors


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
