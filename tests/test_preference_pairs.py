from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

from debate.schema import REVIEW_BLOCK_BY_ID
from preference_pairs.builder import (
    AUDIT_FILENAME,
    CATEGORY_EVIDENCE_FILENAME,
    MANIFEST_FILENAME,
    QUESTION_ONLY_FILENAME,
    build_preference_pairs,
)
from preference_pairs.config import (
    PreferencePairConfig,
    SourceConfig,
    load_preference_pair_config,
)
from reflective_enrichment.schema import CATEGORY_ORDER


def test_builds_four_pairs_with_shared_intact_negative_and_exact_schema(
    tmp_path: Path,
) -> None:
    trace = _successful_trace(
        dataset="energy",
        record_id="private-record-id",
        transcript_id="private-transcript-id",
        segment_id="private-segment-id",
        context="Full context — participant paid £38,000.",
        target="Target segment — participant’s exact words.",
    )
    source = _write_source(
        tmp_path / "source",
        name="full",
        dataset="energy",
        traces=[trace],
        context_scope="full_interview",
    )
    config = _config(tmp_path, (source,), {"energy": 1})

    run_dir = build_preference_pairs(
        config,
        negative_selector=lambda candidates: candidates[0],
    )

    evidence_rows = _read_jsonl(run_dir / CATEGORY_EVIDENCE_FILENAME)
    question_rows = _read_jsonl(run_dir / QUESTION_ONLY_FILENAME)
    audit_rows = _read_jsonl(run_dir / AUDIT_FILENAME)
    assert len(evidence_rows) == len(question_rows) == len(audit_rows) == 4

    for index, category in enumerate(CATEGORY_ORDER):
        evidence = evidence_rows[index]
        question_only = question_rows[index]
        audit = audit_rows[index]
        expected_rejected_index = 1 if index == 0 else 0
        rejected_category = CATEGORY_ORDER[expected_rejected_index]
        target_question = trace["reflective_questions"][index]["question"]
        rejected_question = trace["reflective_questions"][expected_rejected_index][
            "question"
        ]

        assert set(evidence) == {"prompt", "chosen", "rejected"}
        assert set(question_only) == {"prompt", "chosen", "rejected"}
        _assert_conversation_roles(evidence)
        _assert_conversation_roles(question_only)

        assert question_only["chosen"][0]["content"] == target_question
        assert question_only["rejected"][0]["content"] == rejected_question
        assert audit["target_category"] == category
        assert audit["rejected_source_category"] == rejected_category
        assert audit["rejected_source_category"] != audit["target_category"]
        assert audit["chosen_question"] == target_question
        assert audit["rejected_question"] == rejected_question

        chosen_text = evidence["chosen"][0]["content"]
        rejected_text = evidence["rejected"][0]["content"]
        _assert_complete_rendered_bundle(
            chosen_text,
            category,
            trace["selected_codes"][index]["code"],
            target_question,
        )
        _assert_complete_rendered_bundle(
            rejected_text,
            rejected_category,
            trace["selected_codes"][expected_rejected_index]["code"],
            rejected_question,
        )

        evidence_prompt = evidence["prompt"][0]["content"]
        question_prompt = question_only["prompt"][0]["content"]
        for prompt in (evidence_prompt, question_prompt):
            assert trace["research_questions"][0] in prompt
            assert trace["full_interview_context"] in prompt
            assert trace["target_segment"] in prompt
            assert trace["selected_codes"][index]["code"]["code_label"] in prompt
            assert trace["record_id"] not in prompt
            assert trace["transcript_id"] not in prompt
            assert trace["segment_id"] not in prompt
            assert category not in prompt
        assert "Identify its code-quality category" in evidence_prompt
        assert "Return only the reflective question" in question_prompt

    assert "£38,000" in (run_dir / CATEGORY_EVIDENCE_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "participant’s" in (run_dir / QUESTION_ONLY_FILENAME).read_text(
        encoding="utf-8"
    )


def test_full_interview_and_turn_window_contexts_are_passed_through_unchanged(
    tmp_path: Path,
) -> None:
    full_trace = _successful_trace(
        dataset="sexual-health",
        record_id="full-record",
        context="FULL-CONTEXT-BEGIN\nTurn 1 | interviewer: Hello\nFULL-CONTEXT-END",
        target="Full-context target.",
    )
    window_trace = _successful_trace(
        dataset="ukda-4688",
        record_id="window-record",
        context=(
            "WINDOW-BEGIN\nTurn 10 | Interviewer: Before\n"
            "Turn 11 | Female [TARGET SEGMENT]: Exact\n"
            "Turn 12 | Interviewer: After\nWINDOW-END"
        ),
        target="Window target.",
    )
    full_source = _write_source(
        tmp_path / "full",
        name="full",
        dataset="sexual-health",
        traces=[full_trace],
        context_scope="full_interview",
    )
    window_source = _write_source(
        tmp_path / "window",
        name="window",
        dataset="ukda-4688",
        traces=[window_trace],
        context_scope="turn_window",
        before=20,
        after=20,
    )

    run_dir = build_preference_pairs(
        _config(
            tmp_path,
            (full_source, window_source),
            {"sexual-health": 1, "ukda-4688": 1},
        ),
        negative_selector=lambda candidates: candidates[-1],
    )
    rows = _read_jsonl(run_dir / QUESTION_ONLY_FILENAME)
    audits = _read_jsonl(run_dir / AUDIT_FILENAME)
    full_prompt = rows[0]["prompt"][0]["content"]
    window_prompt = rows[4]["prompt"][0]["content"]

    assert full_trace["full_interview_context"] in full_prompt
    assert "Full interview context:" in full_prompt
    assert window_trace["full_interview_context"] in window_prompt
    assert (
        "Interview context window (up to 20 turns before and 20 turns after "
        "the target):"
    ) in window_prompt
    assert audits[0]["context_scope"] == "full_interview"
    assert audits[0]["context_turns_before"] is None
    assert audits[4]["context_scope"] == "turn_window"
    assert audits[4]["context_turns_before"] == 20
    assert audits[4]["context_turns_after"] == 20


def test_failed_and_missing_traces_are_skipped_and_summarized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    success = _successful_trace(dataset="energy", record_id="success")
    failed = {
        "dataset": "energy",
        "record_id": "failed",
        "status": "failed",
        "validation_errors": ["question code did not match"],
    }
    source = _write_source(
        tmp_path / "source",
        name="partial",
        dataset="energy",
        traces=[success, failed],
        manifest_record_count=3,
        context_scope="full_interview",
        manifest_status="incomplete",
    )

    run_dir = build_preference_pairs(
        _config(tmp_path, (source,), {"energy": 3}),
        negative_selector=lambda candidates: candidates[0],
    )
    manifest = _read_json(run_dir / MANIFEST_FILENAME)
    output = capsys.readouterr().out

    assert manifest["run_state"]["status"] == "complete"
    assert manifest["counts"] == {
        "accepted_record_count": 1,
        "skipped_failed_trace_count": 1,
        "missing_trace_count": 1,
        "pair_count_per_version": 4,
        "category_evidence_row_count": 4,
        "question_only_row_count": 4,
        "audit_row_count": 4,
    }
    source_summary = manifest["source_summaries"][0]
    assert source_summary["manifest_status"] == "incomplete"
    assert source_summary["missing_trace_count"] == 1
    assert source_summary["skipped_failed_trace_count"] == 1
    assert source_summary["skipped_records"] == [
        {
            "dataset": "energy",
            "record_id": "failed",
            "status": "failed",
            "validation_errors": ["question code did not match"],
            "trace_path": str(source.trace_root / "failed.json"),
        }
    ]
    assert "failed_skipped=1" in output
    assert "missing=1" in output
    assert "pairs_per_version=4" in output


@pytest.mark.parametrize(
    ("mutate", "error_text"),
    [
        (
            lambda trace: trace.update(
                {"validation_errors": ["successful-but-invalid"]}
            ),
            "empty validation_errors",
        ),
        (
            lambda trace: (
                trace["reflective_questions"][0].update(
                    {"code": "misaligned label"}
                ),
                trace["parsed_output"]["reflective_questions"][0].update(
                    {"code": "misaligned label"}
                ),
            ),
            "does not match its selected code label",
        ),
        (
            lambda trace: trace["selected_codes"][0]["code"].pop(
                "why_unsupported_by_this_segment"
            ),
            "fields do not match",
        ),
        (
            lambda trace: trace["parsed_output"].update(
                {"reflective_questions": []}
            ),
            "top-level/parsed reflective questions differ",
        ),
    ],
)
def test_malformed_successful_traces_fail_fast_without_finalized_training_files(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    error_text: str,
) -> None:
    trace = _successful_trace(dataset="energy", record_id="malformed")
    mutate(trace)
    source = _write_source(
        tmp_path / "source",
        name="malformed",
        dataset="energy",
        traces=[trace],
        context_scope="full_interview",
    )

    with pytest.raises(ValueError, match=error_text):
        build_preference_pairs(_config(tmp_path, (source,), {"energy": 1}))

    run_dir = _only_run_dir(tmp_path / "output")
    manifest = _read_json(run_dir / MANIFEST_FILENAME)
    assert manifest["run_state"]["status"] == "failed"
    assert not (run_dir / CATEGORY_EVIDENCE_FILENAME).exists()
    assert not (run_dir / QUESTION_ONLY_FILENAME).exists()
    assert not (run_dir / AUDIT_FILENAME).exists()


def test_invalid_json_fails_fast_and_records_failed_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    trace_root = source_root / "traces"
    trace_root.mkdir(parents=True)
    (trace_root / "broken.json").write_text("{not-json", encoding="utf-8")
    manifest_path = source_root / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "reflective_question_run_v2",
            "record_count": 1,
            "run_state": {
                "status": "incomplete",
                "success_count": 0,
                "failure_count": 1,
            },
        },
    )
    source = SourceConfig(
        name="invalid",
        trace_root=trace_root,
        manifest_path=manifest_path,
        datasets=("energy",),
        context_scope="full_interview",
    )

    with pytest.raises(ValueError, match="Could not read reflective trace"):
        build_preference_pairs(_config(tmp_path, (source,), {"energy": 1}))

    manifest = _read_json(_only_run_dir(tmp_path / "output") / MANIFEST_FILENAME)
    assert manifest["run_state"]["status"] == "failed"
    assert manifest["run_state"]["error_type"] == "ValueError"


def test_duplicate_dataset_record_identity_fails_fast(tmp_path: Path) -> None:
    first = _successful_trace(dataset="energy", record_id="duplicate")
    second = _successful_trace(dataset="energy", record_id="duplicate")
    source = _write_source(
        tmp_path / "source",
        name="duplicates",
        dataset="energy",
        traces=[first, second],
        context_scope="full_interview",
        filenames=("first.json", "second.json"),
    )

    with pytest.raises(ValueError, match="Duplicate trace identity"):
        build_preference_pairs(_config(tmp_path, (source,), {"energy": 2}))

    run_dir = _only_run_dir(tmp_path / "output")
    assert _read_json(run_dir / MANIFEST_FILENAME)["run_state"]["status"] == "failed"
    assert not (run_dir / CATEGORY_EVIDENCE_FILENAME).exists()


def test_selector_cannot_return_target_or_unknown_index(tmp_path: Path) -> None:
    trace = _successful_trace(dataset="energy", record_id="bad-selector")
    source = _write_source(
        tmp_path / "source",
        name="selector",
        dataset="energy",
        traces=[trace],
        context_scope="full_interview",
    )

    with pytest.raises(ValueError, match="Negative selector returned"):
        build_preference_pairs(
            _config(tmp_path, (source,), {"energy": 1}),
            negative_selector=lambda _candidates: 99,
        )


def test_manifest_hashes_alignment_and_random_provenance_are_consistent(
    tmp_path: Path,
) -> None:
    traces = [
        _successful_trace(dataset="energy", record_id="one"),
        _successful_trace(dataset="energy", record_id="two"),
    ]
    source = _write_source(
        tmp_path / "source",
        name="hashes",
        dataset="energy",
        traces=traces,
        context_scope="full_interview",
    )
    run_dir = build_preference_pairs(
        _config(tmp_path, (source,), {"energy": 2}),
        negative_selector=lambda candidates: candidates[-1],
    )
    manifest = _read_json(run_dir / MANIFEST_FILENAME)
    evidence_rows = _read_jsonl(run_dir / CATEGORY_EVIDENCE_FILENAME)
    question_rows = _read_jsonl(run_dir / QUESTION_ONLY_FILENAME)
    audit_rows = _read_jsonl(run_dir / AUDIT_FILENAME)

    assert manifest["audit_alignment"] == {
        "is_line_aligned": True,
        "category_evidence_rows": 8,
        "question_only_rows": 8,
        "audit_rows": 8,
    }
    assert manifest["random_negative_sampling"] == {
        "strategy": "secrets.choice",
        "seed": None,
        "reproducible_across_runs": False,
        "same_selection_used_for_both_versions": True,
        "candidate_pool": "the other three aligned same-segment bundles",
    }
    assert sum(
        count
        for rejected_counts in manifest["rejection_category_matrix"].values()
        for count in rejected_counts.values()
    ) == 8

    files = {
        "category_evidence": run_dir / CATEGORY_EVIDENCE_FILENAME,
        "question_only": run_dir / QUESTION_ONLY_FILENAME,
        "audit": run_dir / AUDIT_FILENAME,
    }
    for name, path in files.items():
        metadata = manifest["output_files"][name]
        assert metadata["path"] == str(path)
        assert metadata["row_count"] == 8
        assert metadata["byte_count"] == path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    for index, audit in enumerate(audit_rows):
        assert audit["line_number"] == index + 1
        assert audit["category_evidence_row_sha256"] == _row_sha256(
            evidence_rows[index]
        )
        assert audit["question_only_row_sha256"] == _row_sha256(
            question_rows[index]
        )


def test_checked_in_config_has_hpc_paths_expected_totals_and_cli_registration() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_preference_pair_config(
        root / "configs" / "dpo_preference_pairs.json"
    )

    assert config.output_root == Path(
        "/iridisfs/scratch/kjl1a21/DPO/data/dpo_preference_pairs"
    )
    assert config.expected_counts() == {
        "energy": 104,
        "sexual-health": 117,
        "ukda-4688": 6315,
    }
    assert sum(config.expected_counts().values()) == 6536
    assert sum(config.expected_counts().values()) * 4 == 26144
    assert config.sources[0].context_scope == "full_interview"
    assert config.sources[1].context_scope == "turn_window"
    assert config.sources[1].context_turns_before == 20
    assert config.sources[1].context_turns_after == 20
    assert (
        'dpo-build-preferences = "preference_pairs.cli:main"'
        in (root / "pyproject.toml").read_text(encoding="utf-8")
    )


def test_config_validation_rejects_duplicate_datasets_and_bad_window(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicate.json"
    _write_json(
        duplicate_path,
        {
            "sources": [
                {
                    "name": "one",
                    "trace_root": "one",
                    "manifest_path": "one.json",
                    "datasets": ["energy"],
                    "context_scope": "full_interview",
                },
                {
                    "name": "two",
                    "trace_root": "two",
                    "manifest_path": "two.json",
                    "datasets": ["energy"],
                    "context_scope": "full_interview",
                },
            ],
            "output_root": "output",
        },
    )
    with pytest.raises(ValueError, match="only one source"):
        load_preference_pair_config(duplicate_path)

    bad_window_path = tmp_path / "bad-window.json"
    _write_json(
        bad_window_path,
        {
            "sources": [
                {
                    "name": "window",
                    "trace_root": "traces",
                    "manifest_path": "manifest.json",
                    "datasets": ["ukda-4688"],
                    "context_scope": "turn_window",
                    "context_turns_before": 20,
                }
            ],
            "output_root": "output",
        },
    )
    with pytest.raises(ValueError, match="requires both context turn counts"):
        load_preference_pair_config(bad_window_path)


def _config(
    tmp_path: Path,
    sources: tuple[SourceConfig, ...],
    expected_counts: dict[str, int],
) -> PreferencePairConfig:
    return PreferencePairConfig(
        sources=sources,
        output_root=tmp_path / "output",
        run_name="test_pairs",
        expected_record_counts=tuple(expected_counts.items()),
    )


def _write_source(
    root: Path,
    *,
    name: str,
    dataset: str,
    traces: list[dict[str, Any]],
    context_scope: str,
    before: int | None = None,
    after: int | None = None,
    manifest_record_count: int | None = None,
    manifest_status: str = "complete",
    filenames: tuple[str, ...] | None = None,
) -> SourceConfig:
    trace_root = root / "traces"
    trace_root.mkdir(parents=True)
    resolved_filenames = filenames or tuple(
        f"{trace['record_id']}.json" for trace in traces
    )
    assert len(resolved_filenames) == len(traces)
    for filename, trace in zip(resolved_filenames, traces):
        _write_json(trace_root / filename, trace)
    record_count = (
        manifest_record_count
        if manifest_record_count is not None
        else len(traces)
    )
    success_count = sum(trace.get("status") == "success" for trace in traces)
    failure_count = sum(trace.get("status") == "failed" for trace in traces)
    manifest_path = root / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "reflective_question_run_v2",
            "record_count": record_count,
            "run_state": {
                "status": manifest_status,
                "success_count": success_count,
                "failure_count": failure_count,
            },
        },
    )
    return SourceConfig(
        name=name,
        trace_root=trace_root,
        manifest_path=manifest_path,
        datasets=(dataset,),
        context_scope=context_scope,
        context_turns_before=before,
        context_turns_after=after,
    )


def _successful_trace(
    *,
    dataset: str,
    record_id: str,
    transcript_id: str = "interview-one",
    segment_id: str = "segment-one",
    context: str = "Turn 1 | interviewer: Context.\nTurn 2 | participant: Target.",
    target: str = "Target participant text.",
) -> dict[str, Any]:
    selected_codes = [
        {
            "hint": category,
            "source_strategy": "fixture",
            "code": _code(category, index),
        }
        for index, category in enumerate(CATEGORY_ORDER)
    ]
    questions = [
        {
            "code": selected["code"]["code_label"],
            "hint": selected["hint"],
            "question": (
                f"How should the researcher examine {selected['code']['code_label']} "
                f"without losing the participant’s meaning £{index}?"
            ),
        }
        for index, selected in enumerate(selected_codes)
    ]
    return {
        "schema_version": "reflective_question_enrichment_v2",
        "dataset": dataset,
        "record_id": record_id,
        "transcript_id": transcript_id,
        "segment_id": segment_id,
        "target_segment": target,
        "research_questions": [
            "How do participants explain their experiences?",
            "What barriers shape those experiences?",
        ],
        "full_interview_context": context,
        "source_segment_path": f"/source/{dataset}/{record_id}.json",
        "selected_codes": selected_codes,
        "status": "success",
        "parsed_output": {"reflective_questions": deepcopy(questions)},
        "validation_errors": [],
        "reflective_questions": questions,
    }


def _code(category: str, index: int) -> dict[str, str]:
    return {
        field: (
            f"Distinct analytical label {index}"
            if field == "code_label"
            else f"{field} evidence — £{index}"
        )
        for field in REVIEW_BLOCK_BY_ID[category].fields
    }


def _assert_conversation_roles(row: dict[str, Any]) -> None:
    assert row["prompt"][0]["role"] == "user"
    assert row["chosen"][0]["role"] == "assistant"
    assert row["rejected"][0]["role"] == "assistant"
    assert len(row["prompt"]) == len(row["chosen"]) == len(row["rejected"]) == 1


def _assert_complete_rendered_bundle(
    text: str,
    category: str,
    code: dict[str, str],
    question: str,
) -> None:
    block = REVIEW_BLOCK_BY_ID[category]
    lines = text.splitlines()
    assert lines[0] == f"Code category: {block.title}"
    expected_field_lines = [
        f"{field.replace('_', ' ').capitalize()}: {code[field]}"
        for field in block.fields
    ]
    assert lines[1:-1] == expected_field_lines
    assert lines[-1] == f"Reflective question: {question}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _only_run_dir(output_root: Path) -> Path:
    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]
