from __future__ import annotations

import json
from pathlib import Path

import pytest

from post_processing import collect_failed_outputs, main, write_aggregate


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


def test_main_accepts_repair_process_as_not_implemented_placeholder() -> None:
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
        "metadata": {"interview_id": "INT01", "segment_id": segment_id},
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
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "final_parse_status": status,
        "validation_errors": validation_errors,
        "reasoning_text": f"Reasoning {sample_index}",
        "parsed_output": None if status == "invalid" else {"ok": True},
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
