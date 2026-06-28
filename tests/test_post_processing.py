from __future__ import annotations

import json
from pathlib import Path

import pytest

from post_processing import collect_failed_outputs, main, repair_failed_outputs, write_aggregate


def test_collects_invalid_samples_and_raw_final_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "transcripts_energy_run"
    interview_dir = run_dir / "INT01_self_consistency"
    segment_path = interview_dir / "segments" / "INT01_SEG001.json"
    _write_json(
        segment_path,
        _segment_payload(
            samples=[
                _sample(1, "invalid", ["No JSON object could be parsed."]),
                _sample(2, "valid", []),
            ]
        ),
    )
    _write_jsonl(
        interview_dir / "events.jsonl",
        [
            _event(1, 1, "first raw output"),
            _event(1, 3, "final raw output"),
            _event(2, 1, "valid raw output"),
        ],
    )

    payload = collect_failed_outputs(enriched_dir=run_dir)

    assert payload["counts"]["segments_scanned"] == 1
    assert payload["counts"]["samples_scanned"] == 2
    assert payload["counts"]["invalid_samples"] == 1
    assert payload["counts"]["raw_attempts_matched"] == 1
    assert payload["counts"]["event_files_read"] == 1
    failed = payload["failed_samples"][0]
    assert failed["record_id"] == "INT01_SEG001"
    assert failed["sample_index"] == 1
    assert failed["validation_errors"] == ["No JSON object could be parsed."]
    assert failed["segment_path"] == failed["repair_target"]["relative_segment_path"]
    assert Path(failed["repair_target"]["absolute_segment_path"]) == segment_path.resolve()
    assert failed["repair_target"]["record_id"] == "INT01_SEG001"
    assert failed["repair_target"]["sample_index"] == 1
    assert failed["repair_target"]["sample_list_index"] == 0
    assert failed["repair_target"]["strategy"] == "self_consistency"
    assert len(failed["repair_target"]["original_sample_fingerprint"]) == 64
    assert failed["raw_final_attempt"]["attempt_index"] == 3
    assert failed["raw_final_attempt"]["raw_output_text"] == "final raw output"


def test_collects_from_parent_folder_containing_run_folders(tmp_path: Path) -> None:
    enriched_root = tmp_path / "transcripts-energy-enriched"
    segment_path = (
        enriched_root
        / "run_001"
        / "INT01_self_consistency"
        / "segments"
        / "INT01_SEG001.json"
    )
    _write_json(segment_path, _segment_payload(samples=[_sample(1, "invalid", ["bad"])]))

    payload = collect_failed_outputs(enriched_dir=enriched_root, include_raw=False)

    assert payload["counts"]["segment_files_found"] == 1
    assert payload["counts"]["segments_scanned"] == 1
    assert payload["counts"]["invalid_samples"] == 1
    assert Path(payload["failed_samples"][0]["segment_path"]) == Path(
        "run_001", "INT01_self_consistency", "segments", "INT01_SEG001.json"
    )


def test_both_scope_reports_segments_with_no_valid_samples(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json",
        _segment_payload(
            samples=[
                _sample(1, "invalid", ["bad one"]),
                _sample(2, "invalid", ["bad two"]),
            ]
        ),
    )
    _write_json(
        run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG002.json",
        _segment_payload(
            record_id="INT01_SEG002",
            segment_id="SEG002",
            samples=[
                _sample(1, "invalid", ["bad"]),
                _sample(2, "valid", []),
            ],
        ),
    )

    payload = collect_failed_outputs(enriched_dir=run_dir, scope="both", include_raw=False)

    assert payload["counts"]["segments_scanned"] == 2
    assert payload["counts"]["samples_scanned"] == 4
    assert payload["counts"]["invalid_samples"] == 3
    assert payload["counts"]["segments_with_no_valid_samples"] == 1
    assert [item["record_id"] for item in payload["segments_with_no_valid_samples"]] == [
        "INT01_SEG001"
    ]
    assert payload["segments_with_no_valid_samples"][0]["invalid_sample_indexes"] == [1, 2]


def test_repair_target_identifies_one_invalid_sample_among_five(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    segment_path = run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json"
    samples = [
        _sample(1, "valid", []),
        _sample(2, "valid", []),
        _sample(3, "invalid", ["bad third sample"]),
        _sample(4, "valid", []),
        _sample(5, "valid", []),
    ]
    _write_json(segment_path, _segment_payload(samples=samples))

    payload = collect_failed_outputs(enriched_dir=run_dir, include_raw=False)

    failed = payload["failed_samples"][0]
    assert failed["sample_index"] == 3
    assert failed["repair_target"]["sample_index"] == 3
    assert failed["repair_target"]["sample_list_index"] == 2
    assert failed["repair_target"]["relative_segment_path"] == failed["segment_path"]
    assert Path(failed["repair_target"]["absolute_segment_path"]) == segment_path.resolve()


def test_sample_fingerprint_is_stable_and_changes_with_sample_content(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    segment_path = run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json"
    sample = _sample(1, "invalid", ["bad"])
    _write_json(segment_path, _segment_payload(samples=[sample]))

    first = collect_failed_outputs(enriched_dir=run_dir, include_raw=False)
    second = collect_failed_outputs(enriched_dir=run_dir, include_raw=False)
    first_fingerprint = first["failed_samples"][0]["repair_target"][
        "original_sample_fingerprint"
    ]
    second_fingerprint = second["failed_samples"][0]["repair_target"][
        "original_sample_fingerprint"
    ]
    assert first_fingerprint == second_fingerprint

    changed_sample = {**sample, "reasoning_text": "Changed reasoning"}
    _write_json(segment_path, _segment_payload(samples=[changed_sample]))
    changed = collect_failed_outputs(enriched_dir=run_dir, include_raw=False)
    changed_fingerprint = changed["failed_samples"][0]["repair_target"][
        "original_sample_fingerprint"
    ]
    assert changed_fingerprint != first_fingerprint


def test_duplicate_sample_index_adds_warning(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json",
        _segment_payload(
            samples=[
                _sample(1, "valid", []),
                _sample(1, "invalid", ["duplicate bad"]),
            ]
        ),
    )

    payload = collect_failed_outputs(enriched_dir=run_dir, include_raw=False)

    assert any("Duplicate sample_index values" in warning for warning in payload["warnings"])
    assert payload["failed_samples"][0]["repair_target"]["sample_list_index"] == 1


def test_missing_sample_index_adds_repair_target_warning(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    sample = _sample(1, "invalid", ["missing index"])
    del sample["sample_index"]
    _write_json(
        run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json",
        _segment_payload(samples=[sample]),
    )

    payload = collect_failed_outputs(enriched_dir=run_dir, include_raw=False)

    repair_target = payload["failed_samples"][0]["repair_target"]
    assert repair_target["sample_index"] is None
    assert repair_target["sample_list_index"] == 0
    assert any("missing sample_index" in warning for warning in payload["warnings"])
    assert any("Repair target may be unsafe" in warning for warning in payload["warnings"])


def test_main_writes_default_output_and_protects_existing_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json",
        _segment_payload(samples=[_sample(1, "invalid", ["bad"])]),
    )

    assert (
        main(
            [
                "--process",
                "aggregate",
                "--enriched-dir",
                str(run_dir),
                "--no-include-raw",
            ]
        )
        == 0
    )
    output_path = run_dir / "failed_parsed_outputs.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["counts"]["invalid_samples"] == 1

    assert (
        main(
            [
                "--process",
                "aggregate",
                "--enriched-dir",
                str(run_dir),
                "--no-include-raw",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "--process",
                "aggregate",
                "--enriched-dir",
                str(run_dir),
                "--no-include-raw",
                "--overwrite",
            ]
        )
        == 0
    )


def test_repair_updates_one_invalid_sample_in_place(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    segment_path = run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json"
    original_samples = [
        _sample(1, "valid", []),
        _sample(2, "valid", []),
        _sample(
            3,
            "invalid",
            ["analysis_unit.target_text must equal the segment text."],
            parsed_output=_valid_parsed_output(target_text="Participant text corrected."),
        ),
        _sample(4, "valid", []),
        _sample(5, "valid", []),
    ]
    _write_json(segment_path, _segment_payload(samples=original_samples))
    failed_path = tmp_path / "failed.json"
    failed_payload = collect_failed_outputs(enriched_dir=run_dir, include_raw=False)
    _write_json(failed_path, failed_payload)

    report = repair_failed_outputs(failed_path=failed_path)

    repaired_segment = json.loads(segment_path.read_text(encoding="utf-8"))
    assert report["counts"]["repaired"] == 1
    assert report["counts"]["files_written"] == 1
    assert repaired_segment["samples"][0] == original_samples[0]
    assert repaired_segment["samples"][1] == original_samples[1]
    assert repaired_segment["samples"][3] == original_samples[3]
    assert repaired_segment["samples"][4] == original_samples[4]
    repaired_sample = repaired_segment["samples"][2]
    assert repaired_sample["final_parse_status"] == "valid"
    assert repaired_sample["validation_errors"] == []
    assert repaired_sample["validation_warnings"] == []
    assert (
        repaired_sample["parsed_output"]["analysis_unit"]["target_text"]
        == "Participant text."
    )
    assert (
        repaired_sample["repair_metadata"]["method"]
        == "canonical_source_field_injection"
    )
    assert repaired_sample["canonical_corrections"][0]["path"] == (
        "analysis_unit.target_text"
    )


def test_repair_batches_multiple_samples_in_one_segment_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    segment_path = run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json"
    _write_json(
        segment_path,
        _segment_payload(
            samples=[
                _sample(
                    1,
                    "invalid",
                    ["analysis_unit.target_text must equal the segment text."],
                    parsed_output=_valid_parsed_output(target_text="Corrected one."),
                ),
                _sample(2, "valid", []),
                _sample(
                    3,
                    "invalid",
                    ["analysis_unit.target_text must equal the segment text."],
                    parsed_output=_valid_parsed_output(target_text="Corrected three."),
                ),
            ]
        ),
    )
    failed_path = tmp_path / "failed.json"
    _write_json(failed_path, collect_failed_outputs(enriched_dir=run_dir, include_raw=False))

    report = repair_failed_outputs(failed_path=failed_path)

    repaired_segment = json.loads(segment_path.read_text(encoding="utf-8"))
    assert report["counts"]["repaired"] == 2
    assert report["counts"]["files_written"] == 1
    assert repaired_segment["samples"][0]["final_parse_status"] == "valid"
    assert repaired_segment["samples"][2]["final_parse_status"] == "valid"


def test_repair_skips_when_fingerprint_changed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    segment_path = run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json"
    sample = _sample(
        1,
        "invalid",
        ["analysis_unit.target_text must equal the segment text."],
        parsed_output=_valid_parsed_output(target_text="Corrected text."),
    )
    _write_json(segment_path, _segment_payload(samples=[sample]))
    failed_path = tmp_path / "failed.json"
    _write_json(failed_path, collect_failed_outputs(enriched_dir=run_dir, include_raw=False))
    changed = {**sample, "reasoning_text": "Changed after aggregation"}
    _write_json(segment_path, _segment_payload(samples=[changed]))

    report = repair_failed_outputs(failed_path=failed_path)

    current_segment = json.loads(segment_path.read_text(encoding="utf-8"))
    assert report["counts"]["skipped"] == 1
    assert report["counts"]["repaired"] == 0
    assert current_segment["samples"][0]["reasoning_text"] == "Changed after aggregation"


def test_repair_recovers_trailing_extra_brace_raw_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    interview_dir = run_dir / "INT01_self_consistency"
    segment_path = interview_dir / "segments" / "INT01_SEG001.json"
    _write_json(
        segment_path,
        _segment_payload(
            samples=[
                _sample(
                    1,
                    "invalid",
                    [
                        "Extra data: line 1 column 10 (char 9)",
                        "No JSON object could be parsed.",
                    ],
                )
            ]
        ),
    )
    parsed = _valid_parsed_output(target_text="Participant text corrected.")
    raw_output_text = "<think>\nReasoning\n</think>\n" + json.dumps(parsed) + "}"
    _write_jsonl(interview_dir / "events.jsonl", [_event(1, 1, raw_output_text)])
    failed_path = tmp_path / "failed.json"
    _write_json(failed_path, collect_failed_outputs(enriched_dir=run_dir))

    report = repair_failed_outputs(failed_path=failed_path)

    repaired_segment = json.loads(segment_path.read_text(encoding="utf-8"))
    assert report["counts"]["repaired"] == 1
    assert repaired_segment["samples"][0]["final_parse_status"] == "valid"
    assert (
        repaired_segment["samples"][0]["repair_metadata"]["method"]
        == "trailing_extra_brace_parse_recovery"
    )


def test_repair_reports_no_json_and_reflective_link_errors_unrepairable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    segment_path = run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json"
    parsed_with_link_error = _valid_parsed_output(target_text="Participant text.")
    parsed_with_link_error["reflective_question_candidates"][0][
        "linked_code_quality_example"
    ] = "too_broad_code"
    _write_json(
        segment_path,
        _segment_payload(
            samples=[
                _sample(1, "invalid", ["No JSON object could be parsed."]),
                _sample(
                    2,
                    "invalid",
                    [
                        "reflective_question_candidates[0].linked_code_quality_example "
                        "must be 'useful_analytical_code'."
                    ],
                    parsed_output=parsed_with_link_error,
                ),
            ]
        ),
    )
    failed_path = tmp_path / "failed.json"
    _write_json(failed_path, collect_failed_outputs(enriched_dir=run_dir, include_raw=False))

    report = repair_failed_outputs(failed_path=failed_path)

    current_segment = json.loads(segment_path.read_text(encoding="utf-8"))
    assert report["counts"]["unrepairable"] == 2
    assert report["counts"]["repaired"] == 0
    assert current_segment["samples"][0]["final_parse_status"] == "invalid"
    assert current_segment["samples"][1]["final_parse_status"] == "invalid"


def test_repair_dry_run_writes_report_but_not_segment(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    segment_path = run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json"
    _write_json(
        segment_path,
        _segment_payload(
            samples=[
                _sample(
                    1,
                    "invalid",
                    ["analysis_unit.target_text must equal the segment text."],
                    parsed_output=_valid_parsed_output(target_text="Corrected text."),
                )
            ]
        ),
    )
    failed_path = tmp_path / "failed.json"
    report_path = tmp_path / "repair_report.json"
    _write_json(failed_path, collect_failed_outputs(enriched_dir=run_dir, include_raw=False))

    report = repair_failed_outputs(
        failed_path=failed_path,
        report_path=report_path,
        dry_run=True,
    )

    current_segment = json.loads(segment_path.read_text(encoding="utf-8"))
    assert report_path.exists()
    assert report["counts"]["repaired"] == 1
    assert report["counts"]["files_written"] == 0
    assert current_segment["samples"][0]["final_parse_status"] == "invalid"


def test_main_runs_repair_with_failed_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    segment_path = run_dir / "INT01_self_consistency" / "segments" / "INT01_SEG001.json"
    _write_json(
        segment_path,
        _segment_payload(
            samples=[
                _sample(
                    1,
                    "invalid",
                    ["analysis_unit.target_text must equal the segment text."],
                    parsed_output=_valid_parsed_output(target_text="Corrected text."),
                )
            ]
        ),
    )
    failed_path = tmp_path / "failed.json"
    report_path = tmp_path / "report.json"
    _write_json(failed_path, collect_failed_outputs(enriched_dir=run_dir, include_raw=False))

    status = main(
        [
            "--process",
            "repair",
            "--failed-path",
            str(failed_path),
            "--report-path",
            str(report_path),
            "--dry-run",
        ]
    )

    assert status == 0
    assert report_path.exists()


def test_main_requires_failed_path_for_repair() -> None:
    assert main(["--process", "repair"]) == 1


def test_main_requires_enriched_dir_for_aggregate() -> None:
    assert main(["--process", "aggregate"]) == 1


def test_write_aggregate_requires_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "aggregate.json"
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_aggregate({"ok": True}, output_path)

    write_aggregate({"ok": True}, output_path, overwrite=True)
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}


def _segment_payload(
    *,
    samples: list[dict[str, object]],
    record_id: str = "INT01_SEG001",
    segment_id: str = "SEG001",
) -> dict[str, object]:
    return {
        "timestamp_utc": "2026-06-19T00:00:00Z",
        "record_id": record_id,
        "interview_id": "INT01",
        "segment_id": segment_id,
        "input_text": "Participant text.",
        "metadata": {
            "interview_id": "INT01",
            "segment_id": segment_id,
            "speaker": "participant",
        },
        "source": {"segments_path": "segments.jsonl", "segments_line": 1},
        "strategy": "self_consistency",
        "num_samples": len(samples),
        "aggregation_status": "not_implemented_yet",
        "samples": samples,
    }


def _sample(
    sample_index: int,
    status: str,
    validation_errors: list[str],
    *,
    parsed_output: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "final_parse_status": status,
        "validation_errors": validation_errors,
        "reasoning_text": f"Reasoning {sample_index}",
        "parsed_output": parsed_output
        if parsed_output is not None
        else (None if status == "invalid" else {"ok": True}),
    }


def _event(sample_index: int, attempt_index: int, raw_output_text: str) -> dict[str, object]:
    return {
        "timestamp_utc": "2026-06-19T00:00:00Z",
        "event": "teacher_generation",
        "record_id": "INT01_SEG001",
        "strategy": "self_consistency",
        "step": "sample",
        "sample_index": sample_index,
        "attempt_index": attempt_index,
        "parse_status": "invalid",
        "validation_errors": ["bad"],
        "reasoning_parse_status": "missing_close_think_tag",
        "reasoning_text": "raw reasoning",
        "json_text": "",
        "raw_output_text": raw_output_text,
        "generation_options": {"seed": 10},
        "elapsed_seconds": 1.2,
    }


def _valid_parsed_output(*, target_text: str) -> dict[str, object]:
    return {
        "schema_version": "segment_enrichment_sample_v1",
        "record_id": "INT01_SEG001",
        "codebook_version": "v1",
        "analysis_unit": {
            "interview_id": "INT01",
            "segment_id": "SEG001",
            "speaker": "participant",
            "target_text": target_text,
            "previous_context_used": True,
            "next_context_used": False,
            "context_warning": "",
        },
        "research_question_relevance": {
            "relevant_research_questions": ["How do participants discuss energy?"],
            "segment_relevance_summary": "The segment is analytically useful.",
            "is_segment_analytically_useful": True,
            "why_or_why_not": "It speaks to the supplied research question.",
        },
        "candidate_code_matches": [],
        "possible_new_codes": [],
        "code_quality_examples": {
            "wrong_code": {
                "code_label": "Unsupported code",
                "actual_segment_quote": "Participant text",
                "why_plausible_for_wider_dataset": "It could occur elsewhere.",
                "why_unsupported_by_this_segment": "This segment does not support it.",
                "relation_to_research_questions": "It would mislead the analysis.",
                "category_boundary": "Unsupported for this specific segment.",
            },
            "descriptive_not_answering_research_question": {
                "code_label": "Mentions participant text",
                "evidence_quote": "Participant text",
                "surface_description": "It describes the text.",
                "why_true_of_segment": "The text is present.",
                "why_not_useful_for_research_questions": "It is too surface-level.",
                "relation_to_research_questions": "It does not answer much analytically.",
                "category_boundary": "True but analytically weak here.",
            },
            "too_broad_code": {
                "code_label": "Energy issue",
                "evidence_quote": "Participant text",
                "broad_relevance_to_research_questions": "It is broadly relevant.",
                "specific_meaning_lost": "The specific participant meaning is lost.",
                "why_it_is_too_broad": "It is too general.",
                "relation_to_research_questions": "It is relevant but vague.",
                "category_boundary": "Relevant but too broad here.",
            },
            "useful_analytical_code": {
                "code_label": "Specific useful analytical code",
                "evidence_quote": "Participant text",
                "specific_analytical_insight": "It preserves the specific meaning.",
                "why_it_is_useful": "It helps answer the question.",
                "relation_to_research_questions": "It is directly useful.",
                "why_better_than_other_three": "It is grounded and specific.",
                "category_boundary": "Useful for this specific segment.",
            },
        },
        "contrastive_judgement": {
            "wrong_vs_descriptive": "The wrong code is unsupported; the descriptive code is shallow.",
            "descriptive_vs_too_broad": "The descriptive code is surface-level; the broad code is vague.",
            "too_broad_vs_useful": "The useful code keeps the specific meaning.",
            "final_preference_reason": "The useful code is preferable.",
        },
        "reflective_question_candidates": [
            {
                "question_id": "Q1",
                "question": "Does this over-read the evidence?",
                "linked_code_ids": [],
                "linked_provisional_code_ids": [],
                "linked_code_quality_example": "useful_analytical_code",
                "question_type": "devils_advocate",
                "reflexive_dimension": "methodological",
                "trigger_quote": "Participant text",
                "why_this_question_is_useful": "It checks over-interpretation.",
                "what_human_researcher_should_inspect": "The quote.",
                "risk_if_ignored": "The code may overstate the data.",
                "confidence": 8,
            },
            {
                "question_id": "Q2",
                "question": "Is participant voice preserved?",
                "linked_code_ids": [],
                "linked_provisional_code_ids": [],
                "linked_code_quality_example": "useful_analytical_code",
                "question_type": "participant_voice_check",
                "reflexive_dimension": "methodological",
                "trigger_quote": "Participant text",
                "why_this_question_is_useful": "It checks voice preservation.",
                "what_human_researcher_should_inspect": "Alternative readings.",
                "risk_if_ignored": "The analysis may lose nuance.",
                "confidence": 8,
            },
            {
                "question_id": "Q3",
                "question": "What context is missing?",
                "linked_code_ids": [],
                "linked_provisional_code_ids": [],
                "linked_code_quality_example": "useful_analytical_code",
                "question_type": "context_check",
                "reflexive_dimension": "contextual",
                "trigger_quote": "Participant text",
                "why_this_question_is_useful": "It identifies missing context.",
                "what_human_researcher_should_inspect": "Neighboring turns.",
                "risk_if_ignored": "The analysis may miss context.",
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
