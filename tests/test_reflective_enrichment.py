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
