from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from dpo_critical_thinking.enrichment.cli import build_parser, main as enrich_main
from dpo_critical_thinking.enrichment.data import load_segment_records
from dpo_critical_thinking.enrichment.logging import RunLogger
from dpo_critical_thinking.enrichment.prompts import PromptTemplate
from dpo_critical_thinking.enrichment.schema import (
    parse_json_object,
    split_response_sections,
)
from dpo_critical_thinking.enrichment.strategies import run_self_consistency
from dpo_critical_thinking.enrichment.teachers import (
    GenerationOptions,
    GenerationResult,
    resolve_effective_max_new_tokens,
)
from dpo_critical_thinking.preprocessing.codebook import convert_xlsx_codebook
from dpo_critical_thinking.preprocessing.html import preprocess_html_dataset


class QueueTeacher:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        return GenerationResult(
            text=output,
            raw={"options": asdict(options)},
            rendered_prompt=prompt,
            elapsed_seconds=0.0,
        )

    def metadata(self) -> dict[str, Any]:
        return {"backend": "queue"}


def test_convert_xlsx_codebook_supports_both_observed_sheet_layouts(
    tmp_path: Path,
) -> None:
    workbook_path = tmp_path / "ExampleCodes.xlsx"
    _write_example_workbook(workbook_path)

    payload = convert_xlsx_codebook(
        input_xlsx=workbook_path,
        output_path=tmp_path / "example_codes_v1.json",
        codebook_id="example_codes",
        codebook_version="v1",
        description="Test codebook.",
    )

    codes = {code["code_id"]: code for code in payload["codes"]}
    assert "human_oversight" in codes
    assert "childfree_by_choice" in codes
    assert codes["human_oversight"]["example_reflective_questions"] == [
        "Could this be about accountability rather than trust?"
    ]
    assert codes["childfree_by_choice"]["source_sheets"] == ["Braun and Clarke"]


def test_preprocess_html_writes_raw_interviews_and_participant_turn_segments(
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "multi.html"
    html_path.write_text(_multi_interview_html(), encoding="utf-8")

    manifest = preprocess_html_dataset(
        input_path=html_path,
        raw_html_dir=tmp_path / "data" / "raw_html",
        segments_dir=tmp_path / "data" / "segments_jsonl",
        manifest_path=tmp_path / "data" / "preprocessing_manifest.json",
    )

    assert [item["interview_id"] for item in manifest["interviews"]] == ["INT01", "INT02"]
    assert manifest["interviews"][0]["participant_characteristics"] == {
        "gender": "Woman",
        "age": "38",
        "notes": "Uses solar panels.",
    }
    assert (tmp_path / "data" / "raw_html" / "INT01.html").exists()
    segments = _read_jsonl(tmp_path / "data" / "segments_jsonl" / "INT01_segments.jsonl")
    assert [segment["record_id"] for segment in segments] == ["INT01_SEG001", "INT01_SEG002"]
    assert segments[0]["speaker"] == "participant"
    assert segments[0]["previous_context"].startswith("Interviewer:")
    assert segments[0]["next_context"].startswith("Interviewer:")
    assert segments[0]["line_start"] is None
    assert segments[0]["paragraph_index_start"] == 2
    assert segments[0]["participant_characteristics"] == {
        "gender": "Woman",
        "age": "38",
        "notes": "Uses solar panels.",
    }
    assert "candidate_example_codes" not in segments[0]
    assert "codebook_id" not in segments[0]
    assert "codebook_version" not in segments[0]


def test_load_segment_records_accepts_file_or_directory(tmp_path: Path) -> None:
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    _write_jsonl(segments_dir / "INT01_segments.jsonl", [_segment("INT01", 1)])
    _write_jsonl(segments_dir / "INT02_segments.jsonl", [_segment("INT02", 1)])

    one_file = load_segment_records(segments_dir / "INT01_segments.jsonl")
    all_files = load_segment_records(segments_dir)

    assert [record.record_id for record in one_file] == ["INT01_SEG001"]
    assert [record.record_id for record in all_files] == ["INT01_SEG001", "INT02_SEG001"]
    assert all_files[0].to_prompt_vars()["candidate_example_codes_json"] == "[]"


def test_parse_json_object_prefers_content_after_think_block() -> None:
    payload = {"schema_version": "segment_enrichment_sample_v1", "ok": True}
    text = (
        "<think>\n"
        "Reasoning may mention a non-final object like {not valid json}.\n"
        "</think>\n"
        + json.dumps(payload)
    )

    sections = split_response_sections(text)
    parsed, parse_error = parse_json_object(text)

    assert sections["reasoning_text"] == "Reasoning may mention a non-final object like {not valid json}."
    assert sections["reasoning_parse_status"] == "found_closed_think_block"
    assert sections["json_text"] == json.dumps(payload)
    assert parsed == payload
    assert parse_error is None


def test_parse_json_object_still_accepts_output_without_think_block() -> None:
    payload = {"schema_version": "segment_enrichment_sample_v1", "ok": True}

    sections = split_response_sections(json.dumps(payload))
    parsed, parse_error = parse_json_object(json.dumps(payload))

    assert sections["reasoning_parse_status"] == "no_think_block"
    assert sections["reasoning_text"] == ""
    assert parsed == payload
    assert parse_error is None


def test_self_consistency_retries_until_sample_json_is_valid(tmp_path: Path) -> None:
    prompt_path = _write_prompt(tmp_path, "Prompt {record_id}: {candidate_example_codes_json}")
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    codebook = _codebook_payload()
    valid_json = json.dumps(_valid_sample(record, codebook_version="v1"))
    teacher = QueueTeacher(
        [
            "<think>\nInvalid reasoning with {braces}.\n</think>\nnot json",
            f"<think>\nGrounded reasoning.\n</think>\n{valid_json}",
        ]
    )

    enriched = run_self_consistency(
        record=record,
        teacher=teacher,
        prompt=PromptTemplate(prompt_path),
        prompt_vars={},
        codebook=codebook,
        generation_options=GenerationOptions(seed=10),
        num_samples=1,
        aggregation="scaffold",
        json_retry_attempts=1,
        logger=RunLogger(tmp_path / "logs"),
    )

    sample = enriched["samples"][0]
    assert sample["attempt_count"] == 2
    assert sample["final_parse_status"] == "valid"
    assert sample["reasoning_text"] == "Grounded reasoning."
    assert sample["reasoning_block"] == "<think>\nGrounded reasoning.\n</think>"
    assert sample["json_text"] == valid_json
    assert sample["reasoning_parse_status"] == "found_closed_think_block"
    assert sample["attempts"][0]["reasoning_text"] == "Invalid reasoning with {braces}."
    assert sample["attempts"][0]["reasoning_parse_status"] == "found_closed_think_block"
    assert sample["attempts"][1]["reasoning_text"] == "Grounded reasoning."
    assert sample["parsed_output"]["schema_version"] == "segment_enrichment_sample_v1"
    assert enriched["aggregation_status"] == "not_implemented_yet"


def test_enrichment_cli_default_max_new_tokens_is_deepseek_budget() -> None:
    args = build_parser().parse_args(
        [
            "--segments-path",
            "segments",
            "--output-dir",
            "outputs",
            "--strategy",
            "self_consistency",
            "--prompt-path",
            "prompt.txt",
        ]
    )

    assert args.max_new_tokens == 32768


def test_transformers_token_budget_clamps_to_context_window() -> None:
    budget = resolve_effective_max_new_tokens(
        prompt_token_count=90,
        requested_max_new_tokens=32,
        context_window=100,
    )

    assert budget["requested_max_new_tokens"] == 32
    assert budget["effective_max_new_tokens"] == 10
    assert budget["context_window"] == 100
    assert budget["token_budget_clamped"] is True

    with pytest.raises(ValueError, match="exceeds the model context window"):
        resolve_effective_max_new_tokens(
            prompt_token_count=100,
            requested_max_new_tokens=1,
            context_window=100,
        )


def test_enrichment_cli_writes_per_interview_output_directory(tmp_path: Path) -> None:
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    _write_jsonl(segments_dir / "INT01_segments.jsonl", [_segment("INT01", 1)])
    prompt_path = _write_prompt(tmp_path, "Prompt {record_id}: {segment_json}")
    codebook_path = _write_codebook(tmp_path)

    status = enrich_main(
        [
            "--segments-path",
            str(segments_dir),
            "--output-dir",
            str(tmp_path / "outputs" / "enrichment"),
            "--codebook-path",
            str(codebook_path),
            "--strategy",
            "self_consistency",
            "--prompt-path",
            str(prompt_path),
            "--teacher-backend",
            "dry-run",
            "--self-consistency-samples",
            "1",
            "--json-retry-attempts",
            "0",
        ]
    )

    assert status == 0
    assert (
        tmp_path
        / "outputs"
        / "enrichment"
        / "INT01_self_consistency"
        / "run_manifest.json"
    ).exists()


def test_enrichment_prompt_vars_use_runtime_codebook(tmp_path: Path) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    variables = record.to_prompt_vars(_codebook_payload())

    assert variables["codebook_id"] == "example_codes"
    assert variables["codebook_version"] == "v1"
    assert "human_oversight" in variables["candidate_example_codes_json"]
    assert "candidate_example_codes" not in variables["segment_json"]


def test_self_consistency_prompt_has_single_codebook_and_target_copy(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    prompt = PromptTemplate(Path("prompts/enrichment/self_consistency_placeholder.txt"))
    rendered = prompt.render(record.to_prompt_vars(_codebook_payload()))

    assert rendered.count('"code_id": "human_oversight"') == 1
    assert rendered.count(record.text) == 1
    assert "Copy the Target segment text exactly." in rendered


def test_runtime_codebook_overrides_legacy_embedded_codes(tmp_path: Path) -> None:
    legacy_record = _segment("INT01", 1, include_codebook=True)
    legacy_record["candidate_example_codes"] = [
        {
            "code_id": "legacy_code",
            "code_label": "legacy_code",
            "definition": None,
            "example_quotes": [],
            "example_reflective_questions": [],
            "source_sheets": ["legacy"],
        }
    ]
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [legacy_record]))[0]
    variables = record.to_prompt_vars(_codebook_payload())

    assert "human_oversight" in variables["candidate_example_codes_json"]
    assert "legacy_code" not in variables["candidate_example_codes_json"]
    assert "legacy_code" not in variables["segment_json"]


def test_enrichment_cli_accepts_legacy_embedded_codebook_without_runtime_path(
    tmp_path: Path,
) -> None:
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    _write_jsonl(
        segments_dir / "INT01_segments.jsonl",
        [_segment("INT01", 1, include_codebook=True)],
    )
    prompt_path = _write_prompt(
        tmp_path, "Prompt {codebook_version}: {candidate_example_codes_json}"
    )

    status = enrich_main(
        [
            "--segments-path",
            str(segments_dir),
            "--output-dir",
            str(tmp_path / "outputs" / "enrichment"),
            "--strategy",
            "self_consistency",
            "--prompt-path",
            str(prompt_path),
            "--teacher-backend",
            "dry-run",
            "--self-consistency-samples",
            "1",
            "--json-retry-attempts",
            "0",
        ]
    )

    assert status == 0


def _write_example_workbook(path: Path) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Contestable Camera Cars"
    sheet.append(["Code", "Quotes", "Example Questions"])
    sheet.append(["human_oversight", "AI can help, but someone checks.", None])
    sheet.append([None, "Someone still needs to review it.", "Could this be about accountability rather than trust?"])
    sheet = workbook.create_sheet("Braun and Clarke")
    sheet.append(["Quote", "Codes"])
    sheet.append(["I chose not to have children.", "Childfree by choice"])
    workbook.save(path)


def _write_codebook(tmp_path: Path) -> Path:
    path = tmp_path / "codebook.json"
    path.write_text(json.dumps(_codebook_payload()), encoding="utf-8")
    return path


def _codebook_payload() -> dict[str, Any]:
    return {
        "codebook_id": "example_codes",
        "codebook_version": "v1",
        "source_file": "test.xlsx",
        "description": "Test codebook.",
        "codes": [
            {
                "code_id": "human_oversight",
                "code_label": "human_oversight",
                "definition": "Need for human checking.",
                "example_quotes": ["AI can help, but someone checks."],
                "example_reflective_questions": [],
                "source_sheets": ["manual"],
            }
        ],
    }


def _multi_interview_html() -> str:
    return """
    <!DOCTYPE html>
    <html><body>
      <h2>P1</h2>
      <table>
        <tr><td>Gender</td><td>Woman</td><td>Age</td><td>38</td></tr>
        <tr><td>Notes</td><td>Uses solar panels.</td></tr>
      </table>
      <p class="interviewer"><strong>Interviewer</strong> Question one?</p>
      <p class="participant"><strong>Participant</strong> Answer one.</p>
      <p class="interviewer"><strong>Interviewer</strong> Question two?</p>
      <p class="participant"><strong>Participant</strong> Answer two.</p>
      <h2>P2</h2>
      <table>
        <tr><td>Gender</td><td>Man</td><td>Age</td><td>44</td></tr>
      </table>
      <p class="interviewer"><strong>Interviewer</strong> Another question?</p>
      <p class="participant"><strong>Participant</strong> Another answer.</p>
    </body></html>
    """


def _segment(
    interview_id: str, index: int, *, include_codebook: bool = False
) -> dict[str, Any]:
    segment_id = f"SEG{index:03d}"
    payload = {
        "record_id": f"{interview_id}_{segment_id}",
        "text": "AI can help, but someone still needs to check the result.",
        "interview_id": interview_id,
        "segment_id": segment_id,
        "speaker": "participant",
        "turn_index": 2,
        "paragraph_index_start": 2,
        "paragraph_index_end": 2,
        "line_start": None,
        "line_end": None,
        "previous_context": "Interviewer: Can AI help?",
        "next_context": "",
        "source_html_path": f"data/raw_html/{interview_id}.html",
    }
    if include_codebook:
        payload["codebook_id"] = "example_codes"
        payload["codebook_version"] = "v1"
        payload["candidate_example_codes"] = _codebook_payload()["codes"]
    return payload


def _valid_sample(record: Any, *, codebook_version: str) -> dict[str, Any]:
    return {
        "schema_version": "segment_enrichment_sample_v1",
        "record_id": record.record_id,
        "codebook_version": codebook_version,
        "analysis_unit": {
            "interview_id": record.metadata["interview_id"],
            "segment_id": record.metadata["segment_id"],
            "speaker": "participant",
            "target_text": record.text,
            "previous_context_used": True,
            "next_context_used": False,
            "context_warning": "",
        },
        "candidate_code_matches": [],
        "possible_new_codes": [],
        "reflective_question_candidates": [],
        "quality_control": {
            "hallucination_risk": "low",
            "over_generalisation_risk": "low",
            "participant_voice_loss_risk": "low",
            "needs_human_review": False,
            "review_reason": "",
            "overall_confidence": 8,
        },
    }


def _write_prompt(tmp_path: Path, text: str) -> Path:
    path = tmp_path / f"prompt_{len(list(tmp_path.glob('prompt_*')))}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
