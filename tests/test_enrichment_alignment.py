from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from enrichment.cli import build_parser, main as enrich_main
from enrichment.data import load_segment_records
from enrichment.logging import RunLogger
from enrichment.prompts import PromptTemplate
from enrichment.schema import (
    parse_json_object,
    split_response_sections,
    validate_segment_enrichment_sample,
    validate_segment_enrichment_sample_result,
)
from enrichment.strategies import (
    run_self_consistency,
    run_self_refine,
)
from enrichment.teachers import (
    GenerationOptions,
    GenerationResult,
    normalize_decoded_text,
    resolve_effective_max_new_tokens,
)
from preprocessing.codebook import convert_xlsx_codebook
from preprocessing.html import preprocess_html_dataset


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
    assert "interview_id_source" not in manifest
    assert "text_normalization" not in manifest
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
    assert segments[0]["interview_turns"] == [
        {
            "turn_index": 1,
            "speaker": "interviewer",
            "text": "Question one?",
            "paragraph_index": 1,
        },
        {
            "turn_index": 2,
            "speaker": "participant",
            "text": "Answer one.",
            "paragraph_index": 2,
        },
        {
            "turn_index": 3,
            "speaker": "interviewer",
            "text": "Question two?",
            "paragraph_index": 3,
        },
        {
            "turn_index": 4,
            "speaker": "participant",
            "text": "Answer two.",
            "paragraph_index": 4,
        },
    ]
    assert segments[1]["interview_turns"] == segments[0]["interview_turns"]
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


def test_preprocess_html_can_use_heading_ids_and_repair_mojibake(
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "sexual_health.html"
    html_path.write_text(_sexual_health_html_with_mojibake(), encoding="utf-8")

    manifest = preprocess_html_dataset(
        input_path=html_path,
        raw_html_dir=tmp_path / "data" / "raw_html",
        segments_dir=tmp_path / "data" / "segments_jsonl",
        manifest_path=tmp_path / "data" / "preprocessing_manifest.json",
        interview_id_source="heading",
        text_normalization="mojibake",
        dataset_id="transcripts-sexual_health",
        domain="sexual health services",
    )

    assert [item["interview_id"] for item in manifest["interviews"]] == ["P1", "P2"]
    assert manifest["interview_id_source"] == "heading"
    assert manifest["text_normalization"] == "mojibake"
    assert manifest["dataset_id"] == "transcripts-sexual_health"
    assert manifest["domain"] == "sexual health services"
    assert manifest["interviews"][0]["source_interview_label"] == "P1"
    assert manifest["interviews"][0]["participant_characteristics"] == {
        "gender": "Woman",
        "age": "23",
        "location": "Guildford",
    }
    assert (tmp_path / "data" / "raw_html" / "P1.html").exists()

    segments = _read_jsonl(tmp_path / "data" / "segments_jsonl" / "P1_segments.jsonl")
    assert segments[0]["record_id"] == "P1_SEG001"
    assert segments[0]["interview_id"] == "P1"
    assert segments[0]["source_interview_label"] == "P1"
    assert segments[0]["text_normalization"] == "mojibake"
    assert segments[0]["dataset_id"] == "transcripts-sexual_health"
    assert segments[0]["domain"] == "sexual health services"
    assert segments[0]["text"] == "I\u2019m using \u201conline booking\u201d\u2014privately."
    assert "\u00e2\u20ac" not in segments[0]["text"]


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


def test_context_scope_immediate_supports_legacy_segments(tmp_path: Path) -> None:
    record = load_segment_records(
        _write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)])
    )[0]

    variables = record.to_prompt_vars(context_scope="immediate")

    assert variables["analysis_context"] == (
        "Previous context:\nInterviewer: Can AI help?\n\nNext context:\n"
    )


def test_context_scope_full_interview_marks_target_and_excludes_turns_from_json(
    tmp_path: Path,
) -> None:
    record = load_segment_records(
        _write_jsonl(
            tmp_path / "segments.jsonl",
            [_segment("INT01", 1, include_interview_turns=True)],
        )
    )[0]

    variables = record.to_prompt_vars(context_scope="full_interview")

    assert variables["analysis_context"].splitlines() == [
        "Turn 1 | Interviewer: Can AI help?",
        "Turn 2 | Participant [TARGET SEGMENT SEG001]: "
        "AI can help, but someone still needs to check the result.",
        "Turn 3 | Interviewer: What are the risks?",
    ]
    assert "interview_turns" not in variables
    assert "interview_turns" not in variables["segment_json"]


def test_context_scope_full_interview_rejects_missing_or_mismatched_turns(
    tmp_path: Path,
) -> None:
    legacy = load_segment_records(
        _write_jsonl(tmp_path / "legacy.jsonl", [_segment("INT01", 1)])
    )[0]
    with pytest.raises(ValueError, match="Reprocess the source HTML"):
        legacy.analysis_context("full_interview")

    mismatched_payload = _segment("INT01", 1, include_interview_turns=True)
    mismatched_payload["interview_turns"][1]["text"] = "Different target text."
    mismatched = load_segment_records(
        _write_jsonl(tmp_path / "mismatched.jsonl", [mismatched_payload])
    )[0]
    with pytest.raises(ValueError, match="same participant text"):
        mismatched.analysis_context("full_interview")


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


def test_parse_json_object_accepts_only_trailing_extra_closing_braces() -> None:
    payload = {"schema_version": "segment_enrichment_sample_v1", "ok": True}

    parsed, parse_error = parse_json_object(json.dumps(payload) + "}}")

    assert parse_error is None
    assert parsed == payload


def test_parse_json_object_rejects_trailing_prose_and_concatenated_objects() -> None:
    payload = {"schema_version": "segment_enrichment_sample_v1", "ok": True}

    prose_parsed, prose_error = parse_json_object(json.dumps(payload) + " extra text")
    concat_parsed, concat_error = parse_json_object(
        json.dumps(payload) + json.dumps({"another": True})
    )

    assert prose_parsed is None
    assert prose_error is not None
    assert concat_parsed is None
    assert concat_error is not None


def test_normalize_decoded_text_repairs_byte_level_json_markers(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")
    encoded = (
        "<think>ĊReasoningĊ</think>ĊĊ"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        .replace("\n", "Ċ")
        .replace(" ", "Ġ")
    )

    normalization = normalize_decoded_text(encoded)
    parsed, parse_error = parse_json_object(normalization.text)

    assert normalization.normalized is True
    assert normalization.raw_text == encoded
    assert parse_error is None
    assert parsed is not None
    assert parsed["analysis_unit"]["target_text"] == record.text


def test_normalize_decoded_text_leaves_clean_text_unchanged() -> None:
    text = "<think>\nReasoning\n</think>\n\n{\"ok\": true}"

    normalization = normalize_decoded_text(text)

    assert normalization.text == text
    assert normalization.normalized is False
    assert normalization.raw_text is None


def test_self_consistency_retries_until_sample_json_is_valid(tmp_path: Path) -> None:
    prompt_path = _write_prompt(tmp_path, "Prompt {record_id}: {candidate_example_codes_json}")
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    codebook = _codebook_payload()
    valid_json = json.dumps(_valid_v2_sample(record, codebook_version="v1"))
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
    assert sample["parsed_output"]["schema_version"] == "segment_enrichment_sample_v2"
    assert "Regenerate a concise <think>...</think>" in teacher.prompts[1]
    assert enriched["aggregation_status"] == "not_implemented_yet"


def test_self_consistency_rejects_target_text_mismatch(
    tmp_path: Path,
) -> None:
    prompt_path = _write_prompt(tmp_path, "Prompt {record_id}: {candidate_example_codes_json}")
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    codebook = _codebook_payload()
    payload = _valid_v2_sample(record, codebook_version="v1")
    payload["analysis_unit"]["target_text"] = (
        "AI can help, but someone still needs to check the corrected result."
    )
    teacher = QueueTeacher([f"<think>\nReasoning.\n</think>\n{json.dumps(payload)}"])

    with pytest.raises(ValueError, match="target_text"):
        run_self_consistency(
            record=record,
            teacher=teacher,
            prompt=PromptTemplate(prompt_path),
            prompt_vars={},
            codebook=codebook,
            generation_options=GenerationOptions(seed=10),
            num_samples=1,
            aggregation="scaffold",
            json_retry_attempts=0,
            logger=RunLogger(tmp_path / "logs"),
        )


def test_self_consistency_rejects_missing_think_block(tmp_path: Path) -> None:
    prompt_path = _write_prompt(tmp_path, "Prompt {record_id}")
    record = load_segment_records(
        _write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)])
    )[0]
    payload = _valid_v2_sample(record, codebook_version="v1")
    teacher = QueueTeacher([json.dumps(payload)])

    with pytest.raises(ValueError, match="closed <think>"):
        run_self_consistency(
            record=record,
            teacher=teacher,
            prompt=PromptTemplate(prompt_path),
            prompt_vars={},
            codebook=_codebook_payload(),
            generation_options=GenerationOptions(),
            num_samples=1,
            aggregation="scaffold",
            json_retry_attempts=0,
            logger=RunLogger(tmp_path / "logs"),
        )


def test_self_consistency_receives_full_interview_context(tmp_path: Path) -> None:
    record = load_segment_records(
        _write_jsonl(
            tmp_path / "segments.jsonl",
            [_segment("INT01", 1, include_interview_turns=True)],
        )
    )[0]
    prompt_path = _write_prompt(tmp_path, "Context:\n{analysis_context}")
    valid_output = json.dumps(
        _valid_v2_sample(
            record,
            codebook_version="v1",
            context_scope="full_interview",
        )
    )
    teacher = QueueTeacher([f"<think>\nReasoning.\n</think>\n{valid_output}"])

    enriched = run_self_consistency(
        record=record,
        teacher=teacher,
        prompt=PromptTemplate(prompt_path),
        prompt_vars={},
        codebook=_codebook_payload(),
        generation_options=GenerationOptions(),
        num_samples=1,
        aggregation="scaffold",
        json_retry_attempts=0,
        logger=RunLogger(tmp_path / "logs"),
        context_scope="full_interview",
    )

    assert "[TARGET SEGMENT SEG001]" in teacher.prompts[0]
    assert enriched["context_scope"] == "full_interview"


def test_self_refine_receives_full_context_in_every_prompt(tmp_path: Path) -> None:
    record = load_segment_records(
        _write_jsonl(
            tmp_path / "segments.jsonl",
            [_segment("INT01", 1, include_interview_turns=True)],
        )
    )[0]
    initial_prompt = _write_prompt(tmp_path, "Initial:\n{analysis_context}")
    critique_prompt = _write_prompt(
        tmp_path,
        "Critique:\n{analysis_context}\n{current_answer}",
    )
    revision_prompt = _write_prompt(
        tmp_path,
        "Revision:\n{analysis_context}\n{current_answer}\n{feedback}",
    )
    valid_json = json.dumps(
        _valid_v2_sample(
            record,
            codebook_version="v1",
            context_scope="full_interview",
        )
    )
    valid_output = f"<think>\nReasoning.\n</think>\n{valid_json}"
    teacher = QueueTeacher(
        [
            valid_output,
            json.dumps({"needs_refinement": True, "feedback": "Revise it."}),
            valid_output,
        ]
    )

    enriched = run_self_refine(
        record=record,
        teacher=teacher,
        initial_prompt=PromptTemplate(initial_prompt),
        critique_prompt=PromptTemplate(critique_prompt),
        revision_prompt=PromptTemplate(revision_prompt),
        prompt_vars={},
        codebook=_codebook_payload(),
        generation_options=GenerationOptions(),
        refine_rounds=1,
        stop_parser="json",
        history_format="text",
        json_retry_attempts=0,
        logger=RunLogger(tmp_path / "refine_logs"),
        context_scope="full_interview",
    )

    assert len(teacher.prompts) == 3
    assert all("[TARGET SEGMENT SEG001]" in prompt for prompt in teacher.prompts)
    assert enriched["context_scope"] == "full_interview"


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


def test_enrichment_cli_accepts_repeated_research_questions() -> None:
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
            "--research-question",
            "How do participants discuss energy efficiency?",
            "--research-question",
            "How do participants describe smart technology use?",
        ]
    )

    assert args.research_question == [
        "How do participants discuss energy efficiency?",
        "How do participants describe smart technology use?",
    ]


def test_enrichment_cli_context_scope_defaults_to_immediate() -> None:
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

    assert args.context_scope == "immediate"


def test_full_interview_rejects_missing_prompt_placeholder_before_teacher_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments_path = _write_jsonl(
        tmp_path / "segments.jsonl",
        [_segment("INT01", 1, include_interview_turns=True)],
    )
    prompt_path = _write_prompt(
        tmp_path,
        "An escaped placeholder does not inject context: {{analysis_context}}",
    )
    codebook_path = _write_codebook(tmp_path)

    def unexpected_teacher_build(args: Any) -> Any:
        raise AssertionError("Teacher construction must happen after context validation.")

    monkeypatch.setattr(
        "enrichment.cli.build_teacher",
        unexpected_teacher_build,
    )

    with pytest.raises(ValueError, match=r"requires \{analysis_context\}"):
        enrich_main(
            [
                "--segments-path",
                str(segments_path),
                "--output-dir",
                str(tmp_path / "outputs"),
                "--codebook-path",
                str(codebook_path),
                "--strategy",
                "self_consistency",
                "--prompt-path",
                str(prompt_path),
                "--context-scope",
                "full_interview",
            ]
        )


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
    interview_dir = (
        tmp_path
        / "outputs"
        / "enrichment"
        / "INT01_self_consistency"
    )
    segment_json = interview_dir / "segments" / "INT01_SEG001.json"
    assert (interview_dir / "run_manifest.json").exists()
    assert (interview_dir / "events.jsonl").exists()
    assert not (interview_dir / "enriched_records.jsonl").exists()
    assert segment_json.exists()
    segment_payload = json.loads(segment_json.read_text(encoding="utf-8"))
    assert segment_payload["interview_id"] == "INT01"
    assert segment_payload["segment_id"] == "SEG001"
    assert segment_payload["context_scope"] == "immediate"
    manifest = json.loads(
        (interview_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["args"]["context_scope"] == "immediate"
    assert manifest["output_schema_version"] == "segment_enrichment_sample_v2"
    sample = segment_payload["samples"][0]
    assert "reasoning_text" in sample
    assert "attempts" not in sample
    assert "rendered_prompt" not in sample
    assert "raw_output_text" not in sample


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
    assert "Copy the Target segment text exactly into analysis_unit.target_text." in rendered
    assert '"schema_version": "segment_enrichment_sample_v2"' in rendered


def test_self_consistency_prompt_includes_research_questions(tmp_path: Path) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    prompt = PromptTemplate(Path("prompts/enrichment/self_consistency_placeholder.txt"))
    rendered = prompt.render(
        {
            **record.to_prompt_vars(_codebook_payload()),
            "research_questions": "1. How do participants discuss energy efficiency?",
        }
    )

    assert "Research questions:" in rendered
    assert "Research questions JSON:" not in rendered
    assert "How do participants discuss energy efficiency?" in rendered


def test_segment_enrichment_schema_requires_code_quality_examples(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")

    assert validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    ) == []

    del payload["code_quality_examples"]["useful_analytical_code"]
    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    )

    assert any("useful_analytical_code" in error for error in errors)


def test_v2_schema_accepts_redesigned_output_and_irrelevant_segment(
    tmp_path: Path,
) -> None:
    record = load_segment_records(
        _write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)])
    )[0]
    payload = _valid_v2_sample(record, codebook_version="v1")
    payload["research_question_relevance"] = {
        "relevant_research_questions": [],
        "segment_relevance_summary": "The segment does not answer the supplied questions.",
        "is_segment_analytically_useful": False,
        "why_or_why_not": "All four contrasts are retained without inventing relevance.",
    }

    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
        expected_schema_version="segment_enrichment_sample_v2",
        expected_context_scope="immediate",
    )

    assert errors == []


def test_v2_schema_rejects_old_fields_invalid_values_and_unknown_links(
    tmp_path: Path,
) -> None:
    record = load_segment_records(
        _write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)])
    )[0]
    payload = _valid_v2_sample(record, codebook_version="v1")
    payload["contrastive_judgement"] = {}
    payload["code_quality_examples"]["useful_analytical_code"][
        "why_better_than_other_three"
    ] = "Legacy field."
    payload["reflective_question_candidates"][0]["confidence"] = 0
    payload["reflective_question_candidates"][0]["linked_code_ids"] = ["missing"]
    payload["quality_control"]["hallucination_risk"] = "unknown"

    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
        expected_schema_version="segment_enrichment_sample_v2",
        expected_context_scope="immediate",
    )

    assert any("unexpected fields" in error for error in errors)
    assert any("integer from 1 to 10" in error for error in errors)
    assert any("unknown ids" in error for error in errors)
    assert any("hallucination_risk" in error for error in errors)

def test_segment_enrichment_result_warns_for_target_text_mismatch(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")
    payload["analysis_unit"]["target_text"] = "AI can help, but someone still needs to check the corrected result."

    strict_errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    )
    relaxed = validate_segment_enrichment_sample_result(
        payload,
        record,
        expected_codebook_version="v1",
        allow_target_text_mismatch=True,
    )

    assert strict_errors == ["analysis_unit.target_text must equal the segment text."]
    assert relaxed.errors == []
    assert relaxed.warnings == [
        "analysis_unit.target_text differs from the segment text."
    ]


def test_segment_enrichment_schema_requires_prompt_top_level_sections(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")
    del payload["research_question_relevance"]
    del payload["contrastive_judgement"]

    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    )

    assert any("research_question_relevance" in error for error in errors)
    assert any("contrastive_judgement" in error for error in errors)


def test_segment_enrichment_schema_rejects_extra_code_quality_category(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")
    payload["code_quality_examples"]["extra_category"] = {
        "code_label": "Unexpected category"
    }

    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    )

    assert any("unexpected examples" in error for error in errors)


def test_segment_enrichment_schema_requires_nested_code_quality_fields(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")
    del payload["code_quality_examples"]["wrong_code"]["actual_segment_quote"]

    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    )

    assert any("actual_segment_quote" in error for error in errors)


def test_segment_enrichment_schema_requires_three_reflective_questions(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")
    payload["reflective_question_candidates"] = payload[
        "reflective_question_candidates"
    ][:2]

    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    )

    assert any("exactly 3" in error for error in errors)


def test_segment_enrichment_schema_requires_ordered_reflective_question_ids(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")
    payload["reflective_question_candidates"][1]["question_id"] = "QX"

    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    )

    assert any("question_id must be 'Q2'" in error for error in errors)


def test_segment_enrichment_schema_requires_useful_linked_quality_example(
    tmp_path: Path,
) -> None:
    record = load_segment_records(_write_jsonl(tmp_path / "segments.jsonl", [_segment("INT01", 1)]))[0]
    payload = _valid_sample(record, codebook_version="v1")
    payload["reflective_question_candidates"][0][
        "linked_code_quality_example"
    ] = "too_broad_code"

    errors = validate_segment_enrichment_sample(
        payload,
        record,
        expected_codebook_version="v1",
    )

    assert any("linked_code_quality_example" in error for error in errors)


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


def test_sexual_health_slurm_uses_separate_dataset_paths() -> None:
    script = Path("submit_job_enrichment_self_consistency_sexual_health.slurm")
    text = script.read_text(encoding="utf-8")

    assert "transcripts-sexual_health-preprocessed/segments_jsonl" in text
    assert "transcripts-sexual_health-enriched" in text
    assert "transcripts_sexual_health_${STRATEGY}_deepseek_r1_distill_llama_70b" in text
    assert "transcripts-energy-preprocessed" not in text
    assert "transcripts-energy-enriched" not in text


@pytest.mark.parametrize(
    "script_name",
    [
        "submit_job_enrichment_self_consistency.slurm",
        "submit_job_enrichment_self_consistency_sexual_health.slurm",
    ],
)
def test_enrichment_slurm_exposes_context_scope(script_name: str) -> None:
    text = Path(script_name).read_text(encoding="utf-8")

    assert 'CONTEXT_SCOPE="full_interview"' in text
    assert '--context-scope "${CONTEXT_SCOPE}"' in text
    assert "set -eo pipefail" in text
    assert "set -euo pipefail" not in text
    assert 'require_directory "${SEGMENTS_PATH}"' in text
    assert 'require_file "${PROMPT_PATH}"' in text


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


def _sexual_health_html_with_mojibake() -> str:
    return """
    <!DOCTYPE html>
    <html><body>
      <h2>P1</h2>
      <table>
        <tr><td>Gender:</td><td>Woman</td></tr>
        <tr><td>Age:</td><td>23</td></tr>
        <tr><td>Location:</td><td>Guildford</td></tr>
      </table>
      <p class="interviewer"><strong>Interviewer</strong> How was booking?</p>
      <p class="participant"><strong>Participant:</strong> I\u00e2\u20ac\u2122m using \u00e2\u20ac\u0153online booking\u00e2\u20ac\u009d\u00e2\u20ac\u201dprivately.</p>
      <h2>P2</h2>
      <table>
        <tr><td>Gender:</td><td>Man</td></tr>
        <tr><td>Age:</td><td>19</td></tr>
        <tr><td>Location:</td><td>Portsmouth</td></tr>
      </table>
      <p class="interviewer"><strong>Interviewer</strong> What helped?</p>
      <p class="participant"><strong>Participant</strong> Reminders helped.</p>
    </body></html>
    """


def _segment(
    interview_id: str,
    index: int,
    *,
    include_codebook: bool = False,
    include_interview_turns: bool = False,
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
    if include_interview_turns:
        payload["interview_turns"] = [
            {
                "turn_index": 1,
                "speaker": "interviewer",
                "text": "Can AI help?",
                "paragraph_index": 1,
            },
            {
                "turn_index": 2,
                "speaker": "participant",
                "text": payload["text"],
                "paragraph_index": 2,
            },
            {
                "turn_index": 3,
                "speaker": "interviewer",
                "text": "What are the risks?",
                "paragraph_index": 3,
            },
        ]
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
        "research_question_relevance": {
            "relevant_research_questions": [
                "How do participants describe AI support?"
            ],
            "segment_relevance_summary": "The segment is useful for analyzing reliance on AI with human checking.",
            "is_segment_analytically_useful": True,
            "why_or_why_not": "It explains that AI may help but still needs human oversight.",
        },
        "candidate_code_matches": [],
        "possible_new_codes": [],
        "code_quality_examples": {
            "wrong_code": {
                "code_label": "No human checking",
                "actual_segment_quote": "someone still needs to check",
                "why_plausible_for_wider_dataset": "Some datasets about AI may include automation without review.",
                "why_unsupported_by_this_segment": "The segment explicitly says checking remains necessary.",
                "relation_to_research_questions": "It would misrepresent the role of oversight.",
                "category_boundary": "It sounds plausible but contradicts this segment's evidence.",
            },
            "descriptive_not_answering_research_question": {
                "code_label": "Mentions AI",
                "evidence_quote": "AI can help",
                "surface_description": "The participant mentions AI as potentially useful.",
                "why_true_of_segment": "The words directly refer to AI helping.",
                "why_not_useful_for_research_questions": "It names the topic without explaining the oversight relationship.",
                "relation_to_research_questions": "It does not explain oversight or use.",
                "category_boundary": "It is true but too surface-level to answer the research question.",
            },
            "too_broad_code": {
                "code_label": "Technology",
                "evidence_quote": "AI can help",
                "broad_relevance_to_research_questions": "AI is a technology relevant to the study.",
                "specific_meaning_lost": "The need for human checking is lost.",
                "why_it_is_too_broad": "It is too general for this meaning.",
                "relation_to_research_questions": "It loses the human-checking issue.",
                "category_boundary": "It is relevant but hides the specific oversight mechanism.",
            },
            "useful_analytical_code": {
                "code_label": "AI help remains conditional on human checking",
                "evidence_quote": "someone still needs to check the result",
                "specific_analytical_insight": "The participant frames AI as useful only when human review remains in place.",
                "why_it_is_useful": "It captures the need for checking AI output.",
                "relation_to_research_questions": "It helps analyze technology reliance.",
                "why_better_than_other_three": "It is supported, specific, and analytically relevant.",
                "category_boundary": "It captures the segment's specific oversight condition.",
            },
        },
        "contrastive_judgement": {
            "wrong_vs_descriptive": "The wrong code contradicts the segment, while the descriptive code is merely shallow.",
            "descriptive_vs_too_broad": "The descriptive code is off-aim, while the broad code is relevant but vague.",
            "too_broad_vs_useful": "The broad code loses the checking condition that the useful code preserves.",
            "final_preference_reason": "The useful analytical code best explains how AI help depends on human checking.",
        },
        "reflective_question_candidates": [
            {
                "question_id": "Q1",
                "question": "Does the phrase 'still needs to check' justify framing this as conditional AI use?",
                "linked_code_ids": [],
                "linked_provisional_code_ids": [],
                "linked_code_quality_example": "useful_analytical_code",
                "question_type": "devils_advocate",
                "reflexive_dimension": "methodological",
                "trigger_quote": "still needs to check",
                "why_this_question_is_useful": "It checks whether the interpretation over-reads the quote.",
                "what_human_researcher_should_inspect": "Whether checking is central or incidental.",
                "risk_if_ignored": "The code may overstate the participant's emphasis.",
                "confidence": 8,
            },
            {
                "question_id": "Q2",
                "question": "Does the code preserve the participant's balanced view that AI can help?",
                "linked_code_ids": [],
                "linked_provisional_code_ids": [],
                "linked_code_quality_example": "useful_analytical_code",
                "question_type": "participant_voice_check",
                "reflexive_dimension": "methodological",
                "trigger_quote": "AI can help",
                "why_this_question_is_useful": "It guards against turning caution into rejection.",
                "what_human_researcher_should_inspect": "Whether the participant is positive, cautious, or both.",
                "risk_if_ignored": "The analysis may lose participant nuance.",
                "confidence": 8,
            },
            {
                "question_id": "Q3",
                "question": "What context is needed to know who should check the result?",
                "linked_code_ids": [],
                "linked_provisional_code_ids": [],
                "linked_code_quality_example": "useful_analytical_code",
                "question_type": "context_check",
                "reflexive_dimension": "contextual",
                "trigger_quote": "someone still needs to check",
                "why_this_question_is_useful": "It identifies an unresolved contextual detail.",
                "what_human_researcher_should_inspect": "Nearby turns about who performs checking.",
                "risk_if_ignored": "The analyst may infer an actor not present in the segment.",
                "confidence": 8,
            },
        ],
        "quality_control": {
            "hallucination_risk": "low",
            "over_generalisation_risk": "low",
            "participant_voice_loss_risk": "low",
            "needs_human_review": False,
            "review_reason": "",
            "overall_confidence": 8,
        },
    }


def _valid_v2_sample(
    record: Any,
    *,
    codebook_version: str,
    context_scope: str = "immediate",
) -> dict[str, Any]:
    payload = _valid_sample(record, codebook_version=codebook_version)
    payload["schema_version"] = "segment_enrichment_sample_v2"
    payload["analysis_unit"] = {
        "interview_id": record.metadata["interview_id"],
        "segment_id": record.metadata["segment_id"],
        "speaker": "participant",
        "target_text": record.text,
        "analysis_context_used": True,
        "analysis_context_scope": context_scope,
        "context_warning": "",
    }
    del payload["contrastive_judgement"]
    del payload["code_quality_examples"]["useful_analytical_code"][
        "why_better_than_other_three"
    ]
    payload["quality_control"]["needs_human_review"] = True
    payload["quality_control"]["review_reason"] = "Manual review is required."
    return payload


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
