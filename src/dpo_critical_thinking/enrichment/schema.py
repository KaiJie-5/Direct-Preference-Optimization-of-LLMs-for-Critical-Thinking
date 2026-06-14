from __future__ import annotations

import json
import re
from typing import Any

from .data import DatasetRecord


SAMPLE_SCHEMA_VERSION = "segment_enrichment_sample_v1"

SAMPLE_REQUIRED_FIELDS = {
    "schema_version",
    "record_id",
    "codebook_version",
    "analysis_unit",
    "candidate_code_matches",
    "possible_new_codes",
    "reflective_question_candidates",
    "quality_control",
}


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced_match.group(1)] if fenced_match else []

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

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


def validate_segment_enrichment_sample(
    payload: dict[str, Any] | None,
    record: DatasetRecord,
) -> list[str]:
    if payload is None:
        return ["No JSON object could be parsed."]

    errors: list[str] = []
    missing = sorted(SAMPLE_REQUIRED_FIELDS - set(payload))
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

    if payload.get("codebook_version") != record.metadata.get("codebook_version"):
        errors.append(
            "codebook_version must match the segment record, got "
            f"{payload.get('codebook_version')!r}"
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
        "reflective_question_candidates",
    ]:
        if not isinstance(payload.get(field), list):
            errors.append(f"{field} must be a list.")

    if not isinstance(payload.get("quality_control"), dict):
        errors.append("quality_control must be an object.")

    return errors


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
