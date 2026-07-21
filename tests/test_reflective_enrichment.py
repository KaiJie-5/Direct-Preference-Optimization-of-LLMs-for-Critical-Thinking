from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from enrichment.teachers import GenerationOptions, GenerationResult
from reflective_enrichment.config import (
    GenerationConfig,
    ReflectiveConfig,
    TeacherConfig,
    config_to_jsonable,
    load_reflective_config,
)
from reflective_enrichment.loaders import load_reflective_inputs
from reflective_enrichment.runner import run_reflective_enrichment
from reflective_enrichment.schema import CATEGORY_ORDER, validate_payload


class QueueTeacher:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, GenerationOptions]] = []

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        self.calls.append((prompt, options))
        if not self.responses:
            raise AssertionError("Unexpected generation call")
        return GenerationResult(
            text=self.responses.pop(0),
            raw={"backend": "queue"},
            rendered_prompt=prompt,
            elapsed_seconds=0.01,
        )

    def metadata(self) -> dict[str, Any]:
        return {"backend": "queue"}


@pytest.mark.parametrize("candidate_count", [4, 5])
def test_loader_selects_top_ranked_code_and_renders_full_context(
    tmp_path: Path, candidate_count: int
) -> None:
    config = _fixture(tmp_path, candidate_count=candidate_count)
    loaded = load_reflective_inputs(
        ranking_run_dir=config.ranking_run_dir,
        review_pack_path=config.review_pack_path,
    )
    assert len(loaded.records) == 1
    record = loaded.records[0]
    assert [item["hint"] for item in record.selected_codes] == list(CATEGORY_ORDER)
    assert [item["selected_candidate_label"] for item in record.selected_codes] == [
        "D",
        "C",
        "B",
        "A",
    ]
    assert [item["original_sample_index"] for item in record.selected_codes] == [4, 3, 2, 1]
    assert record.selected_codes[0]["code"]["code_label"].endswith("sample 4")
    assert "Turn 2 | participant [TARGET SEGMENT]: target words" in record.full_interview_context
    assert "Turn 1 | interviewer: opening question" in record.full_interview_context


def test_schema_requires_exact_order_values_fields_and_distinct_questions(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    record = load_reflective_inputs(
        ranking_run_dir=config.ranking_run_dir,
        review_pack_path=config.review_pack_path,
    ).records[0]
    selected = list(record.selected_codes)
    valid = _payload(selected)
    assert validate_payload(valid, selected) == []

    missing = {"reflective_questions": valid["reflective_questions"][:-1]}
    assert validate_payload(missing, selected)
    reordered = {"reflective_questions": list(reversed(valid["reflective_questions"]))}
    assert any("exactly match" in error for error in validate_payload(reordered, selected))
    extra_field = json.loads(json.dumps(valid))
    extra_field["reflective_questions"][0]["explanation"] = "not allowed"
    assert any("exactly code" in error for error in validate_payload(extra_field, selected))
    wrong_hint = json.loads(json.dumps(valid))
    wrong_hint["reflective_questions"][0]["hint"] = "too_broad_code"
    assert any("hint must be" in error for error in validate_payload(wrong_hint, selected))
    duplicate = json.loads(json.dumps(valid))
    duplicate["reflective_questions"][1]["question"] = duplicate["reflective_questions"][0]["question"]
    assert any("distinct" in error for error in validate_payload(duplicate, selected))


def test_runner_repairs_with_temperature_zero_and_writes_auditable_output(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    selected = list(
        load_reflective_inputs(
            ranking_run_dir=config.ranking_run_dir,
            review_pack_path=config.review_pack_path,
        ).records[0].selected_codes
    )
    teacher = QueueTeacher(["not json", _response(selected)])
    run_dir = run_reflective_enrichment(
        config, teacher_factory=lambda _config: teacher
    )
    assert len(teacher.calls) == 2
    assert teacher.calls[0][1].temperature == 0.6
    assert teacher.calls[1][1].temperature == 0.0
    trace = _read_json(run_dir / "segment_traces" / "energy" / "INT01_SEG001.json")
    assert trace["status"] == "success"
    assert trace["attempt_count"] == 2
    assert len(trace["reflective_questions"]) == 4
    assert trace["attempts"][0]["status"] == "invalid"
    assert trace["attempts"][1]["status"] == "valid"
    assert "[TARGET SEGMENT]" in trace["full_interview_context"]
    assert len((run_dir / "reflective_questions.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert (run_dir / "failures.jsonl").read_text(encoding="utf-8") == ""


def test_resume_skips_success_without_loading_teacher(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    selected = list(
        load_reflective_inputs(
            ranking_run_dir=config.ranking_run_dir,
            review_pack_path=config.review_pack_path,
        ).records[0].selected_codes
    )
    run_dir = run_reflective_enrichment(
        config, teacher_factory=lambda _config: QueueTeacher([_response(selected)])
    )

    def forbidden(_config: ReflectiveConfig) -> QueueTeacher:
        raise AssertionError("Completed resume must not load the teacher")

    assert run_reflective_enrichment(
        config, resume_dir=run_dir, teacher_factory=forbidden
    ) == run_dir.resolve()
    manifest = _read_json(run_dir / "run_manifest.json")
    assert manifest["run_state"]["resume_count"] == 1
    assert manifest["run_state"]["status"] == "complete"


def test_resume_retries_failed_trace_and_preserves_attempt_history(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    loaded = load_reflective_inputs(
        ranking_run_dir=config.ranking_run_dir,
        review_pack_path=config.review_pack_path,
    )
    selected = list(loaded.records[0].selected_codes)
    failing = QueueTeacher(["bad", "bad", "bad"])
    run_dir = run_reflective_enrichment(
        config, teacher_factory=lambda _config: failing
    )
    failed = _read_json(run_dir / "segment_traces" / "energy" / "INT01_SEG001.json")
    assert failed["status"] == "failed"
    assert len(failed["attempts"]) == 3

    succeeding = QueueTeacher([_response(selected)])
    run_reflective_enrichment(
        config, resume_dir=run_dir, teacher_factory=lambda _config: succeeding
    )
    resumed = _read_json(run_dir / "segment_traces" / "energy" / "INT01_SEG001.json")
    assert resumed["status"] == "success"
    assert len(resumed["attempts"]) == 4
    assert resumed["attempts"][:3] == failed["attempts"]
    assert (run_dir / "failures.jsonl").read_text(encoding="utf-8") == ""


def test_resume_regenerates_a_missing_interrupted_checkpoint(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    selected = list(
        load_reflective_inputs(
            ranking_run_dir=config.ranking_run_dir,
            review_pack_path=config.review_pack_path,
        ).records[0].selected_codes
    )
    run_dir = run_reflective_enrichment(
        config, teacher_factory=lambda _config: QueueTeacher([_response(selected)])
    )
    trace = run_dir / "segment_traces" / "energy" / "INT01_SEG001.json"
    trace.unlink()
    replacement = QueueTeacher([_response(selected)])
    run_reflective_enrichment(
        config, resume_dir=run_dir, teacher_factory=lambda _config: replacement
    )
    assert len(replacement.calls) == 1
    assert _read_json(trace)["status"] == "success"


def test_resume_rejects_prompt_drift_and_unknown_checkpoint(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    selected = list(
        load_reflective_inputs(
            ranking_run_dir=config.ranking_run_dir,
            review_pack_path=config.review_pack_path,
        ).records[0].selected_codes
    )
    run_dir = run_reflective_enrichment(
        config, teacher_factory=lambda _config: QueueTeacher([_response(selected)])
    )
    original_prompt = config.prompt_path.read_text(encoding="utf-8")
    config.prompt_path.write_text(original_prompt + " changed", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_sha256"):
        run_reflective_enrichment(config, resume_dir=run_dir)
    config.prompt_path.write_text(original_prompt, encoding="utf-8")
    unknown = run_dir / "segment_traces" / "energy" / "UNKNOWN.json"
    unknown.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        run_reflective_enrichment(config, resume_dir=run_dir)


def test_single_pass_loader_accepts_successes_skips_failures_and_renders_window(
    tmp_path: Path,
) -> None:
    config, source_run = _single_pass_fixture(tmp_path)
    loaded = load_reflective_inputs(
        input_mode=config.input_mode,
        single_pass_run_dir=config.single_pass_run_dir,
        input_status_policy=config.input_status_policy,
        context_scope=config.context_scope,
        context_turns_before=config.context_turns_before,
        context_turns_after=config.context_turns_after,
    )

    assert len(loaded.records) == 1
    record = loaded.records[0]
    assert record.record_id == "int001t_SEG001"
    assert record.research_questions == ("How does employment shape family life?",)
    assert record.full_interview_context.count("[TARGET SEGMENT SEG001]") == 2
    assert "Context scope: centered turn window" in record.full_interview_context
    assert [item["hint"] for item in record.selected_codes] == list(CATEGORY_ORDER)
    assert all(item["source_strategy"] == "single_pass" for item in record.selected_codes)
    snapshot = loaded.input_snapshot
    assert snapshot["accepted_count"] == 1
    assert snapshot["skipped_count"] == 1
    assert snapshot["skipped_records"][0]["record_id"] == "int001t_SEG002"
    assert snapshot["skipped_records"][0]["validation_errors"] == ["invalid output"]
    assert snapshot["source_run_dir"] == str(source_run)


def test_single_pass_loader_rejects_manifest_count_drift(tmp_path: Path) -> None:
    config, source_run = _single_pass_fixture(tmp_path)
    manifest_path = source_run / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["run_state"]["success_count"] = 2
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="do not add up"):
        load_reflective_inputs(
            input_mode="single_pass",
            single_pass_run_dir=config.single_pass_run_dir,
            context_scope="turn_window",
        )


def test_single_pass_loader_rejects_malformed_success(tmp_path: Path) -> None:
    config, source_run = _single_pass_fixture(tmp_path)
    path = source_run / "int001t_single_pass" / "segments" / "int001t_SEG001.json"
    payload = _read_json(path)
    payload["selected_json"]["code_quality_examples"].pop("too_broad_code")
    payload["samples"][0]["parsed_output"] = payload["selected_json"]
    _write_json(path, payload)

    with pytest.raises(ValueError, match="Invalid successful single-pass checkpoint"):
        load_reflective_inputs(
            input_mode="single_pass",
            single_pass_run_dir=config.single_pass_run_dir,
            context_scope="turn_window",
        )


def test_single_pass_resume_keeps_frozen_success_snapshot(tmp_path: Path) -> None:
    config, source_run = _single_pass_fixture(tmp_path)
    loaded = load_reflective_inputs(
        input_mode="single_pass",
        single_pass_run_dir=source_run,
        context_scope="turn_window",
    )
    selected = list(loaded.records[0].selected_codes)
    run_dir = run_reflective_enrichment(
        config, teacher_factory=lambda _config: QueueTeacher([_response(selected)])
    )

    failed_path = source_run / "int001t_single_pass" / "segments" / "int001t_SEG002.json"
    repaired = _single_pass_checkpoint("int001t_SEG002", "SEG002")
    _write_json(failed_path, repaired)
    manifest_path = source_run / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["run_state"].update({"status": "complete", "success_count": 2, "failure_count": 0})
    _write_json(manifest_path, manifest)

    def forbidden(_config: ReflectiveConfig) -> QueueTeacher:
        raise AssertionError("Frozen completed resume must not load the teacher")

    assert run_reflective_enrichment(
        config, resume_dir=run_dir, teacher_factory=forbidden
    ) == run_dir.resolve()
    reflective_manifest = _read_json(run_dir / "run_manifest.json")
    assert reflective_manifest["record_count"] == 1
    assert reflective_manifest["input_snapshot"]["accepted_count"] == 1
    assert len((run_dir / "reflective_questions.jsonl").read_text().splitlines()) == 1


def test_single_pass_resume_rejects_accepted_record_drift(tmp_path: Path) -> None:
    config, source_run = _single_pass_fixture(tmp_path)
    loaded = load_reflective_inputs(
        input_mode="single_pass",
        single_pass_run_dir=source_run,
        context_scope="turn_window",
    )
    selected = list(loaded.records[0].selected_codes)
    run_dir = run_reflective_enrichment(
        config, teacher_factory=lambda _config: QueueTeacher([_response(selected)])
    )
    source_path = source_run / "int001t_single_pass" / "segments" / "int001t_SEG001.json"
    payload = _read_json(source_path)
    code = payload["selected_json"]["code_quality_examples"]["wrong_code"]
    code["code_label"] = "Different unsupported code"
    payload["samples"][0]["parsed_output"] = payload["selected_json"]
    _write_json(source_path, payload)

    with pytest.raises(ValueError, match="input_fingerprint"):
        run_reflective_enrichment(config, resume_dir=run_dir)


def test_single_pass_config_and_legacy_serialization(tmp_path: Path) -> None:
    config, _ = _single_pass_fixture(tmp_path)
    config_path = tmp_path / "single_pass_config.json"
    _write_json(config_path, config_to_jsonable(config))
    loaded = load_reflective_config(config_path)
    assert loaded.input_mode == "single_pass"
    assert loaded.context_turns_before == 20
    assert loaded.context_turns_after == 20

    legacy = _fixture(tmp_path / "legacy")
    serialized = config_to_jsonable(legacy)
    assert "input_mode" not in serialized
    assert "single_pass_run_dir" not in serialized
    assert serialized["ranking_run_dir"] == str(legacy.ranking_run_dir)


def test_ukda_reflective_assets_are_consistent() -> None:
    root = Path(__file__).parents[1]
    config = json.loads(
        (root / "configs" / "reflective_questions_enrichment_ukda4688.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["input_mode"] == "single_pass"
    assert config["input_status_policy"] == "successful_only"
    assert config["context_scope"] == "turn_window"
    assert config["context_turns_before"] == config["context_turns_after"] == 20
    assert config["generation"]["max_new_tokens"] == 8192
    assert "UKDA-4688-rtf-reflective-questions-enriched" in config["output_root"]

    prompt = (
        root / "prompts" / "enrichment" / "reflective_questions_enrichment_ukda4688.txt"
    ).read_text(encoding="utf-8")
    assert "Centered interview-context window" in prompt
    assert "Selected stage-one codes and provenance" in prompt
    assert "Full interview context:" not in prompt

    script = (
        root / "submit_job_reflective_questions_enrichment_ukda4688.slurm"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=quad_h200" in script
    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=200G" in script
    assert "#SBATCH --time=2-12:00:00" in script
    assert 'RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"' in script
    assert 'ARGS+=(--resume "${RESUME_RUN_DIR}")' in script


def _fixture(tmp_path: Path, *, candidate_count: int = 5) -> ReflectiveConfig:
    ranking = tmp_path / "ranking"
    review = tmp_path / "review"
    source = tmp_path / "source.json"
    output = tmp_path / "output"
    prompt = tmp_path / "prompt.txt"
    (ranking / "debate_traces" / "energy").mkdir(parents=True)
    review.mkdir()
    output.mkdir()
    prompt.write_text(
        "Record {record_id}\nQuestions {research_questions}\nTarget {target_segment}\n"
        "Context {full_interview_context}\nCodes {selected_codes_json}\n"
        "Output {required_output_json}",
        encoding="utf-8",
    )
    labels = list("ABCDE"[:candidate_count])
    tops = ["D", "C", "B", "A"]
    rankings = {
        category: [top, *[label for label in labels if label != top]]
        for category, top in zip(CATEGORY_ORDER, tops)
    }
    _write_json(
        ranking / "run_manifest.json",
        {
            "run_state": {"status": "complete"},
            "config": {
                "datasets": [
                    {
                        "dataset": "energy",
                        "research_questions": ["How is the system experienced?"],
                    }
                ]
            }
        },
    )
    _write_jsonl(
        ranking / "final_rankings.jsonl",
        [
            {
                "dataset": "energy",
                "record_id": "INT01_SEG001",
                "candidate_labels": labels,
                "rankings": rankings,
                "block_status": {category: "success" for category in CATEGORY_ORDER},
            }
        ],
    )
    _write_json(
        ranking / "debate_traces" / "energy" / "INT01_SEG001.json",
        {
            "dataset": "energy",
            "record_id": "INT01_SEG001",
            "transcript_id": "INT01",
            "segment_id": "SEG001",
            "segment_json_path": str(source),
        },
    )
    samples = []
    for sample_index in range(1, candidate_count + 1):
        samples.append(
            {
                "sample_index": sample_index,
                "parsed_output": {
                    "code_quality_examples": {
                        category: _code(category, sample_index)
                        for category in CATEGORY_ORDER
                    }
                },
            }
        )
    _write_json(
        source,
        {
            "record_id": "INT01_SEG001",
            "input_text": "target words",
            "metadata": {
                "interview_id": "INT01",
                "segment_id": "SEG001",
                "turn_index": 2,
                "interview_turns": [
                    {"turn_index": 1, "speaker": "interviewer", "text": "opening question"},
                    {"turn_index": 2, "speaker": "participant", "text": "target words"},
                    {"turn_index": 3, "speaker": "interviewer", "text": "follow up"},
                ],
            },
            "samples": samples,
        },
    )
    with (review / "internal_candidate_mapping.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "dataset",
            "record_id",
            "review_block",
            "candidate_label",
            "original_sample_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category in CATEGORY_ORDER:
            for index, label in enumerate(labels, 1):
                writer.writerow(
                    {
                        "dataset": "energy",
                        "record_id": "INT01_SEG001",
                        "review_block": category,
                        "candidate_label": label,
                        "original_sample_index": index,
                    }
                )
    return ReflectiveConfig(
        ranking_run_dir=ranking,
        review_pack_path=review,
        output_root=output,
        prompt_path=prompt,
        teacher=TeacherConfig(backend="dry-run"),
        generation=GenerationConfig(json_repair_attempts=2),
        run_name="test_reflective",
    )


def _single_pass_fixture(tmp_path: Path) -> tuple[ReflectiveConfig, Path]:
    source_run = tmp_path / "single_pass_source"
    segments = source_run / "int001t_single_pass" / "segments"
    segments.mkdir(parents=True)
    _write_json(
        segments / "int001t_SEG001.json",
        _single_pass_checkpoint("int001t_SEG001", "SEG001"),
    )
    failed = _single_pass_checkpoint("int001t_SEG002", "SEG002")
    failed.update(
        {
            "status": "failed",
            "selected_sample_index": None,
            "selected_output": None,
            "selected_json": None,
        }
    )
    failed["samples"][0].update(
        {
            "final_parse_status": "invalid",
            "validation_errors": ["invalid output"],
            "parsed_output": None,
        }
    )
    _write_json(segments / "int001t_SEG002.json", failed)
    _write_json(
        source_run / "run_manifest.json",
        {
            "schema_version": "single_pass_enrichment_run_v1",
            "record_count": 2,
            "execution_fingerprint": "single-pass-source-fingerprint",
            "execution_config": {
                "strategy": "single_pass",
                "research_question": ["How does employment shape family life?"],
            },
            "run_state": {
                "status": "incomplete",
                "success_count": 1,
                "failure_count": 1,
                "missing_count": 0,
            },
        },
    )
    prompt = tmp_path / "single_pass_prompt.txt"
    prompt.write_text(
        "Record {record_id}\nQuestions {research_questions}\nTarget {target_segment}\n"
        "Context {full_interview_context}\nCodes {selected_codes_json}\n"
        "Output {required_output_json}",
        encoding="utf-8",
    )
    output = tmp_path / "single_pass_output"
    output.mkdir()
    config = ReflectiveConfig(
        ranking_run_dir=None,
        review_pack_path=None,
        output_root=output,
        prompt_path=prompt,
        teacher=TeacherConfig(backend="dry-run"),
        input_mode="single_pass",
        single_pass_run_dir=source_run,
        input_status_policy="successful_only",
        context_scope="turn_window",
        context_turns_before=20,
        context_turns_after=20,
        generation=GenerationConfig(json_repair_attempts=2),
        run_name="test_single_pass_reflective",
    )
    return config, source_run


def _single_pass_checkpoint(record_id: str, segment_id: str) -> dict[str, Any]:
    target_text = "Female: first target words\nFemale: second target words"
    metadata = {
        "interview_id": "int001t",
        "segment_id": segment_id,
        "speaker": "participant",
        "turn_index": 2,
        "target_turn_indexes": [2, 4],
        "dataset_id": "ukda-4688",
        "interview_turns": [
            {
                "turn_index": 1,
                "speaker": "interviewer",
                "speaker_label": "Interviewer",
                "text": "opening question",
                "paragraph_index": 1,
            },
            {
                "turn_index": 2,
                "speaker": "participant",
                "speaker_label": "Female",
                "text": "first target words",
                "paragraph_index": 2,
            },
            {
                "turn_index": 3,
                "speaker": "interviewer",
                "speaker_label": "Interviewer",
                "text": "brief prompt",
                "paragraph_index": 3,
            },
            {
                "turn_index": 4,
                "speaker": "participant",
                "speaker_label": "Female",
                "text": "second target words",
                "paragraph_index": 4,
            },
            {
                "turn_index": 5,
                "speaker": "interviewer",
                "speaker_label": "Interviewer",
                "text": "follow up",
                "paragraph_index": 5,
            },
        ],
    }
    selected = _single_pass_selected_json(record_id, segment_id, target_text)
    sample = {
        "sample_index": 1,
        "attempt_count": 1,
        "final_parse_status": "valid",
        "validation_errors": [],
        "validation_warnings": [],
        "parsed_output": selected,
        "output_text": "<think>source reasoning</think>\n{}",
        "attempts": [{}],
    }
    return {
        "record_id": record_id,
        "interview_id": "int001t",
        "segment_id": segment_id,
        "input_text": target_text,
        "metadata": metadata,
        "source": {"segments_path": "source.jsonl", "segments_line": 1},
        "strategy": "single_pass",
        "status": "success",
        "context_scope": "turn_window",
        "context_turns_before": 20,
        "context_turns_after": 20,
        "selected_sample_index": 1,
        "selected_output": sample["output_text"],
        "selected_json": selected,
        "samples": [sample],
    }


def _single_pass_selected_json(
    record_id: str, segment_id: str, target_text: str
) -> dict[str, Any]:
    examples = {
        category: _code(category, 1)
        for category in CATEGORY_ORDER
    }
    examples["wrong_code"]["actual_segment_quote"] = "first target words"
    for category in (
        "descriptive_not_answering_research_question",
        "too_broad_code",
        "useful_analytical_code",
    ):
        examples[category]["evidence_quote"] = "second target words"
    return {
        "schema_version": "segment_enrichment_sample_v3",
        "record_id": record_id,
        "codebook_version": "v1",
        "analysis_unit": {
            "interview_id": "int001t",
            "segment_id": segment_id,
            "speaker": "participant",
            "target_text": target_text,
            "analysis_context_used": True,
            "analysis_context_scope": "turn_window",
            "context_warning": "",
        },
        "research_question_relevance": {
            "relevant_research_questions": ["How does employment shape family life?"],
            "segment_relevance_summary": "The segment contains relevant family-life evidence.",
            "is_segment_analytically_useful": True,
            "why_or_why_not": "It provides evidence relevant to the supplied question.",
        },
        "code_quality_examples": examples,
        "quality_control": {
            "hallucination_risk": "low",
            "over_generalisation_risk": "low",
            "participant_voice_loss_risk": "low",
            "needs_human_review": True,
            "review_reason": "Manual review is required.",
            "overall_confidence": 8,
        },
    }


def _code(category: str, sample_index: int) -> dict[str, str]:
    from debate.schema import REVIEW_BLOCK_BY_ID

    return {
        field: (
            f"{category} sample {sample_index}" if field == "code_label" else f"{field} value"
        )
        for field in REVIEW_BLOCK_BY_ID[category].fields
    }


def _payload(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reflective_questions": [
            {
                "code": item["code"]["code_label"],
                "hint": item["hint"],
                "question": f"What evidence would make you reconsider interpretation {index}?",
            }
            for index, item in enumerate(selected, 1)
        ]
    }


def _response(selected: list[dict[str, Any]]) -> str:
    return "<think>grounded category checks</think>\n" + json.dumps(_payload(selected))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
