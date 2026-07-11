from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .rtf import _analysis_tokens, _clear_evidence_text, _evidence_tokens


UKDA_4688_REVIEW_PROFILE = "ukda-4688"
UKDA_4688_REVIEW_POLICY = "ukda-4688-enrichment-review-v1"

REVIEW_DECISIONS = {"review", "keep", "exclude"}
REVIEW_REQUIRED_FIELDS = {
    "record_id",
    "text",
    "interviewer_question",
    "suggested_reasons",
    "clear_word_count",
    "decision",
}
_REVIEW_REASON_ORDER = (
    "very_short",
    "explicitly_unfinished",
    "unclear_or_damaged",
    "interview_management",
    "participant_question",
    "bare_evaluation",
)
_EVALUATION_WORDS = {
    "amazing",
    "awful",
    "bad",
    "best",
    "brilliant",
    "difficult",
    "easier",
    "easy",
    "fantastic",
    "fine",
    "flexible",
    "good",
    "great",
    "happy",
    "hard",
    "horrible",
    "incredible",
    "love",
    "lovely",
    "nice",
    "perfect",
    "romantic",
    "stressful",
    "terrible",
    "unhappy",
    "wonderful",
    "wonderfully",
    "worse",
    "worst",
}
_HARD_CONTINUATION_WORDS = {
    "although",
    "and",
    "because",
    "but",
    "or",
    "whereas",
}
_INCOMPLETE_END_WORDS = _HARD_CONTINUATION_WORDS | {
    "a",
    "an",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "is",
    "may",
    "might",
    "must",
    "my",
    "of",
    "our",
    "should",
    "the",
    "their",
    "to",
    "was",
    "were",
    "will",
    "with",
    "would",
    "your",
}
_TRAILING_UNCLEAR = re.compile(
    r"\[unclear(?::[^\]]*)?\]\s*[.!?]*\s*$", re.IGNORECASE | re.DOTALL
)
_INTERVIEW_MANAGEMENT = re.compile(
    r"(?:\bare you funded\b|"
    r"\bwhat (?:are you|will you) going to do with\b|"
    r"\bhow many responses\b|"
    r"\bresponse rate\b|"
    r"\bsee (?:the )?(?:research|results)\b|"
    r"\bopportunity to see\b|"
    r"\bwhat time is it\b|"
    r"\blate for (?:them|the interview)\b|"
    r"\bwhen do you go back\b|"
    r"\bhow long are you (?:here|around)\b|"
    r"\bquestions? (?:for|about) (?:me|the research)\b|"
    r"\btape (?:turned|is) off\b)",
    re.IGNORECASE,
)


def generate_target_review(
    *,
    profile: str,
    audit_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if profile != UKDA_4688_REVIEW_PROFILE:
        raise ValueError(
            f"Unknown target-review profile {profile!r}; available profiles: "
            f"{UKDA_4688_REVIEW_PROFILE}."
        )
    if not audit_path.is_file():
        raise FileNotFoundError(f"Target filter audit does not exist: {audit_path}")

    manifest_path = _manifest_path(output_path)
    _ensure_outputs_writable((output_path, manifest_path), overwrite=overwrite)
    retained_audit = _load_retained_audit(audit_path)

    review_records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for audit_record in retained_audit.values():
        suggested_reasons = _ukda_4688_review_reasons(audit_record)
        if not suggested_reasons:
            continue
        reason_counts.update(suggested_reasons)
        review_records.append(
            {
                "record_id": audit_record["candidate_record_id"],
                "text": audit_record["selected_text"],
                "interviewer_question": audit_record["interviewer_question"],
                "suggested_reasons": suggested_reasons,
                "clear_word_count": audit_record["clear_word_count"],
                "decision": "review",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, review_records)
    manifest = {
        "created_utc": _created_utc(),
        "profile": profile,
        "review_policy": UKDA_4688_REVIEW_POLICY,
        "source_audit_path": str(audit_path),
        "source_audit_sha256": _sha256_path(audit_path),
        "output_path": str(output_path),
        "output_sha256": _sha256_path(output_path),
        "retained_record_count": len(retained_audit),
        "candidate_count": len(review_records),
        "reason_counts": dict(reason_counts),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def approve_exclusions(
    *,
    review_path: Path,
    audit_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not review_path.is_file():
        raise FileNotFoundError(f"Exclusion review file does not exist: {review_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Target filter audit does not exist: {audit_path}")

    manifest_path = _manifest_path(output_path)
    _ensure_outputs_writable((output_path, manifest_path), overwrite=overwrite)
    retained_audit = _load_retained_audit(audit_path)
    review_records = _load_review_records(review_path, retained_audit)

    unresolved = [
        record["record_id"]
        for record in review_records
        if record["decision"] == "review"
    ]
    if unresolved:
        preview = ", ".join(unresolved[:10])
        suffix = "" if len(unresolved) <= 10 else f" (+{len(unresolved) - 10} more)"
        raise ValueError(
            "All exclusion review decisions must be resolved to 'keep' or "
            f"'exclude'; unresolved records: {preview}{suffix}."
        )

    approved_records = [
        {"record_id": record["record_id"], "text": record["text"]}
        for record in review_records
        if record["decision"] == "exclude"
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, approved_records)
    decision_counts = Counter(record["decision"] for record in review_records)
    manifest = {
        "created_utc": _created_utc(),
        "source_review_path": str(review_path),
        "source_review_sha256": _sha256_path(review_path),
        "source_audit_path": str(audit_path),
        "source_audit_sha256": _sha256_path(audit_path),
        "output_path": str(output_path),
        "output_sha256": _sha256_path(output_path),
        "review_record_count": len(review_records),
        "keep_count": decision_counts.get("keep", 0),
        "exclude_count": decision_counts.get("exclude", 0),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _ukda_4688_review_reasons(audit_record: dict[str, Any]) -> list[str]:
    text = audit_record["selected_text"]
    question = audit_record["interviewer_question"]
    clear_word_count = audit_record["clear_word_count"]
    total_word_count = audit_record["total_word_count"]
    target_turn_indexes = audit_record["target_turn_indexes"]
    reasons: set[str] = set()

    if clear_word_count <= 10:
        reasons.add("very_short")
    if _is_explicitly_unfinished(text):
        reasons.add("explicitly_unfinished")

    clear_ratio = clear_word_count / max(1, total_word_count)
    question_clear_words = len(_evidence_tokens(question)[1])
    if (
        clear_ratio < 0.5
        or (clear_ratio < 0.75 and clear_word_count < 20)
        or (question_clear_words == 0 and clear_word_count < 30)
    ):
        reasons.add("unclear_or_damaged")

    if _INTERVIEW_MANAGEMENT.search(text) or (
        clear_word_count <= 25 and _INTERVIEW_MANAGEMENT.search(question)
    ):
        reasons.add("interview_management")
    if (
        len(target_turn_indexes) == 1
        and clear_word_count <= 25
        and text.rstrip().endswith("?")
    ):
        reasons.add("participant_question")
    if _is_bare_evaluation(text, clear_word_count):
        reasons.add("bare_evaluation")

    return [reason for reason in _REVIEW_REASON_ORDER if reason in reasons]


def _is_explicitly_unfinished(text: str) -> bool:
    clear_text = _clear_evidence_text(text).rstrip()
    clear_tokens = _analysis_tokens(clear_text)
    if not clear_tokens or clear_text.endswith("?"):
        return False
    last_token = clear_tokens[-1]
    if last_token in _HARD_CONTINUATION_WORDS:
        return True
    if clear_text.endswith(("\u2014", "\u2013", "--")):
        return True

    original_ends_unclear = bool(_TRAILING_UNCLEAR.search(text))
    continuation_punctuation = clear_text.endswith(
        (",", ":", ";", "...", "\u2026")
    )
    return (original_ends_unclear or continuation_punctuation) and (
        last_token in _INCOMPLETE_END_WORDS
    )


def _is_bare_evaluation(text: str, clear_word_count: int) -> bool:
    if clear_word_count > 12:
        return False
    clear_text = _clear_evidence_text(text)
    if any(character.isdigit() for character in clear_text):
        return False
    return bool(set(_analysis_tokens(clear_text)) & _EVALUATION_WORDS)


def _load_retained_audit(audit_path: Path) -> dict[str, dict[str, Any]]:
    retained: dict[str, dict[str, Any]] = {}
    for line_index, payload in _read_jsonl(audit_path):
        required = {
            "candidate_record_id",
            "decision",
            "selected_text",
            "interviewer_question",
            "clear_word_count",
            "total_word_count",
            "target_turn_indexes",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                f"{audit_path}:{line_index} is missing audit fields: {missing}."
            )
        if payload["decision"] != "retained":
            continue
        record_id = payload["candidate_record_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(
                f"{audit_path}:{line_index} candidate_record_id must be non-empty."
            )
        if record_id in retained:
            raise ValueError(f"Duplicate retained audit record_id: {record_id}.")
        retained[record_id] = payload
    return retained


def _load_review_records(
    review_path: Path,
    retained_audit: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_index, payload in _read_jsonl(review_path):
        fields = set(payload)
        if fields != REVIEW_REQUIRED_FIELDS:
            missing = sorted(REVIEW_REQUIRED_FIELDS - fields)
            extra = sorted(fields - REVIEW_REQUIRED_FIELDS)
            raise ValueError(
                f"{review_path}:{line_index} must contain exactly the review "
                f"fields; missing={missing}, extra={extra}."
            )
        record_id = payload["record_id"]
        text = payload["text"]
        question = payload["interviewer_question"]
        decision = payload["decision"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"{review_path}:{line_index} record_id must be non-empty.")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate exclusion review record_id: {record_id}.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{review_path}:{line_index} text must be non-empty.")
        if not isinstance(question, str):
            raise ValueError(
                f"{review_path}:{line_index} interviewer_question must be a string."
            )
        if decision not in REVIEW_DECISIONS:
            raise ValueError(
                f"{review_path}:{line_index} decision must be one of "
                f"{sorted(REVIEW_DECISIONS)}."
            )
        reasons = payload["suggested_reasons"]
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise ValueError(
                f"{review_path}:{line_index} suggested_reasons must be a string list."
            )
        clear_word_count = payload["clear_word_count"]
        if isinstance(clear_word_count, bool) or not isinstance(clear_word_count, int):
            raise ValueError(
                f"{review_path}:{line_index} clear_word_count must be an integer."
            )

        audit_record = retained_audit.get(record_id)
        if audit_record is None:
            raise ValueError(
                f"{review_path}:{line_index} record_id is not retained in the "
                f"target audit: {record_id}."
            )
        if audit_record["selected_text"] != text:
            raise ValueError(
                f"{review_path}:{line_index} text does not match retained audit "
                f"record {record_id}."
            )
        if audit_record["interviewer_question"] != question:
            raise ValueError(
                f"{review_path}:{line_index} interviewer_question does not match "
                f"retained audit record {record_id}."
            )
        if audit_record["clear_word_count"] != clear_word_count:
            raise ValueError(
                f"{review_path}:{line_index} clear_word_count does not match "
                f"retained audit record {record_id}."
            )
        seen_ids.add(record_id)
        records.append(payload)
    return records


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_index}: {exc}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_index} must contain a JSON object.")
            records.append((line_index, payload))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _ensure_outputs_writable(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists; use --overwrite: " + ", ".join(existing)
        )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _created_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
