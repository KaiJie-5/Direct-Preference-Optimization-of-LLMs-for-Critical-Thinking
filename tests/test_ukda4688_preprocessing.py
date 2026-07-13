from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from enrichment.cli import build_parser as build_enrichment_parser
from enrichment.data import DatasetRecord
from enrichment.prompts import align_prompt_context_contract
from preprocessing.cli import build_parser as build_preprocessing_parser
from preprocessing.exclusions import (
    UKDA_4688_REVIEW_POLICY,
    approve_exclusions,
    generate_target_review,
)
from preprocessing.rtf import (
    UKDA_4688_EXPECTED_TRANSCRIPT_COUNT,
    UKDA_4688_TARGET_SELECTION,
    RtfTurn,
    _build_exchange_segments_with_audit,
    _documented_transcript_names,
    _evidence_tokens,
    _has_clear_cutoff,
    _has_complete_claim,
    _is_filler_only,
    _is_interviewer_backchannel,
    _is_unclear_only,
    _normalize_turn_text,
    _parse_ukda_4688_interview,
    _select_analytical_target,
    _validate_inventory,
    preprocess_ukda_4688_dataset,
)


def test_conservative_normalization_marks_uncertainty_and_preserves_questions() -> None:
    normalized, paired, isolated = _normalize_turn_text(
        "Why? ?? uncertain words ?? B?? High ???"
    )

    assert normalized == "Why? [unclear: uncertain words] B[unclear] High [unclear]"
    assert paired == 1
    assert isolated == 2
    assert _is_filler_only("Mmm, yeah.") is True
    assert _is_filler_only("No.") is False
    assert _is_filler_only("Erm, I changed jobs.") is False
    assert _is_unclear_only("[unclear]") is True
    assert _is_unclear_only("[unclear: transcribed words]") is False
    assert _is_unclear_only("Maybe [unclear]") is False


def test_interviewer_backchannels_are_context_not_exchange_boundaries() -> None:
    for text in ("Right.", "Right, yeah.", "Yes, yes.", "Sure.", "(Laughs)."):
        assert _is_interviewer_backchannel(text) is True
    assert _is_interviewer_backchannel("Tell me how that changed.") is False
    assert _is_interviewer_backchannel("You moved after the birth.") is False


@pytest.mark.parametrize(
    "text",
    [
        "I have a cleaning lady.",
        "It was private.",
        "He wasn't born then.",
        "The children were young.",
        "I worked there for many years…",
        "What will you do with this research?",
    ],
)
def test_analytical_filter_keeps_complete_claims_and_participant_questions(
    text: str,
) -> None:
    selection = _selection_for(text)

    assert selection.retained is True
    assert selection.primary_reason is None


@pytest.mark.parametrize(
    ("question", "text", "reason"),
    [
        ("These are state schools?", "State schools.", "question_echo"),
        ("How many vehicles did you have?", "One car.", "short_fragment"),
        ("How long did that last?", "10 months.", "short_fragment"),
        ("Who was responsible?", "Livingstone.", "short_fragment"),
        ("What happened next?", "Took over.", "short_fragment"),
        ("Did that work?", "Yes.", "acknowledgement_or_no_information"),
        ("What happened?", "[unclear: damaged words] (Laughs).", "no_clear_evidence"),
        ("Tell me about the job.", "I had a job and.", "clear_cutoff"),
    ],
)
def test_analytical_filter_rejects_non_evidence_targets(
    question: str,
    text: str,
    reason: str,
) -> None:
    selection = _selection_for(text, question=question)

    assert selection.retained is False
    assert selection.primary_reason == reason


def test_uncertainty_is_ignored_for_selection_without_changing_target_text() -> None:
    text = "I changed jobs after [unclear: the difficult meeting] last year."
    total_tokens, clear_tokens = _evidence_tokens(text)
    nested_total, nested_clear = _evidence_tokens(
        "Before [unclear: nested [unclear] damaged words] after."
    )
    selection = _selection_for(text)

    assert "difficult" in total_tokens
    assert "difficult" not in clear_tokens
    assert nested_total == ["before", "nested", "damaged", "words", "after"]
    assert nested_clear == ["before", "after"]
    assert selection.retained is True
    assert selection.retained_turns[0].text == text


def test_mixed_exchange_prunes_weak_turns_but_keeps_them_available_as_context() -> None:
    question = _test_turn("How did childcare affect work?", role="interviewer", index=1)
    responses = [
        _test_turn("Yes.", index=2),
        _test_turn("I changed my hours to manage childcare better.", index=3),
        _test_turn("(Laughs).", index=4),
    ]

    selection = _select_analytical_target(question, responses)

    assert selection.retained is True
    assert [turn.turn_index for turn in selection.retained_turns] == [3]
    assert [item["turn_index"] for item in selection.pruned_turns] == [2, 4]
    assert [item["reason"] for item in selection.pruned_turns] == [
        "acknowledgement_or_no_information",
        "no_clear_evidence",
    ]


def test_complete_claim_and_cutoff_helpers_are_conservative() -> None:
    assert _has_complete_claim(_evidence_tokens("The children were young.")[1])
    assert not _has_complete_claim(_evidence_tokens("State schools.")[1])
    assert _has_clear_cutoff("I had a job and.") is True
    assert _has_clear_cutoff("I worked there for many years…") is False


def test_target_review_generates_broad_deterministic_review_queue(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "target_filter_audit.jsonl"
    output_path = tmp_path / "enrichment_exclusion_review.jsonl"
    records = [
        _audit_record(
            "int001t_SEG007",
            "Female: Oh I love it, yeah, [unclear] lovely.",
            "Do you feel attached to Edinburgh?",
            clear_words=6,
            total_words=6,
        ),
        _audit_record(
            "int001t_SEG008",
            "Female: We have a car.",
            "Do you run a car?",
            clear_words=4,
            total_words=4,
        ),
        _audit_record(
            "int001t_SEG009",
            "Female: I worked in a research organisation and managed several "
            "long-term public health projects for local hospitals.",
            "What work did you do?",
            clear_words=17,
            total_words=17,
        ),
        _audit_record(
            "int001t_SEG010",
            "Female: I wanted to continue working but I was [unclear]",
            "What happened next?",
            clear_words=9,
            total_words=12,
        ),
        _audit_record(
            "int001t_SEG011",
            "Female: What are you going to do with all this information?",
            "Do you have any questions about the research?",
            clear_words=10,
            total_words=10,
        ),
        _audit_record(
            "int001t_SEG012",
            "Female: The remaining clear account still describes how our family "
            "managed the change despite extensive damaged speech.",
            "How did your family manage the change?",
            clear_words=15,
            total_words=40,
        ),
    ]
    _write_test_jsonl(audit_path, records)

    manifest = generate_target_review(
        profile="ukda-4688",
        audit_path=audit_path,
        output_path=output_path,
    )

    review = _read_test_jsonl(output_path)
    assert [record["record_id"] for record in review] == [
        "int001t_SEG007",
        "int001t_SEG008",
        "int001t_SEG010",
        "int001t_SEG011",
        "int001t_SEG012",
    ]
    assert all(record["decision"] == "review" for record in review)
    assert review[0]["suggested_reasons"] == ["very_short", "bare_evaluation"]
    assert review[1]["suggested_reasons"] == ["very_short"]
    assert "explicitly_unfinished" in review[2]["suggested_reasons"]
    assert "interview_management" in review[3]["suggested_reasons"]
    assert "participant_question" in review[3]["suggested_reasons"]
    assert review[4]["suggested_reasons"] == ["unclear_or_damaged"]
    assert manifest["review_policy"] == UKDA_4688_REVIEW_POLICY
    assert manifest["candidate_count"] == 5
    assert manifest["source_audit_sha256"]
    assert (tmp_path / "enrichment_exclusion_review_manifest.json").is_file()
    with pytest.raises(FileExistsError, match="use --overwrite"):
        generate_target_review(
            profile="ukda-4688",
            audit_path=audit_path,
            output_path=output_path,
        )


def test_approve_exclusions_requires_resolved_exact_review_rows(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "target_filter_audit.jsonl"
    review_path = tmp_path / "enrichment_exclusion_review.jsonl"
    approved_path = tmp_path / "enrichment_exclusions.jsonl"
    audit = [
        _audit_record(
            "int001t_SEG007",
            "Female: Oh I love it, yeah, [unclear] lovely.",
            "Do you feel attached?",
            clear_words=6,
            total_words=6,
        ),
        _audit_record(
            "int001t_SEG008",
            "Female: We have a car.",
            "Do you run a car?",
            clear_words=4,
            total_words=4,
        ),
    ]
    _write_test_jsonl(audit_path, audit)
    generate_target_review(
        profile="ukda-4688",
        audit_path=audit_path,
        output_path=review_path,
    )

    with pytest.raises(ValueError, match="unresolved records"):
        approve_exclusions(
            review_path=review_path,
            audit_path=audit_path,
            output_path=approved_path,
        )

    review = _read_test_jsonl(review_path)
    review[0]["decision"] = "exclude"
    review[1]["decision"] = "keep"
    _write_test_jsonl(review_path, review)
    manifest = approve_exclusions(
        review_path=review_path,
        audit_path=audit_path,
        output_path=approved_path,
    )

    assert _read_test_jsonl(approved_path) == [
        {
            "record_id": "int001t_SEG007",
            "text": "Female: Oh I love it, yeah, [unclear] lovely.",
        }
    ]
    assert manifest["keep_count"] == 1
    assert manifest["exclude_count"] == 1
    assert (tmp_path / "enrichment_exclusions_manifest.json").is_file()

    duplicate_review_path = tmp_path / "duplicate_review.jsonl"
    _write_test_jsonl(duplicate_review_path, [*review, review[0]])
    with pytest.raises(ValueError, match="Duplicate exclusion review"):
        approve_exclusions(
            review_path=duplicate_review_path,
            audit_path=audit_path,
            output_path=tmp_path / "duplicate_approved.jsonl",
        )

    stale_review = _read_test_jsonl(review_path)
    stale_review[0]["text"] = "Female: stale text."
    _write_test_jsonl(review_path, stale_review)
    with pytest.raises(ValueError, match="does not match retained audit"):
        approve_exclusions(
            review_path=review_path,
            audit_path=audit_path,
            output_path=tmp_path / "stale.jsonl",
        )


def test_ukda_parser_groups_question_led_adult_responses_only(tmp_path: Path) -> None:
    source = tmp_path / "int001t.rtf"
    source.write_text(_synthetic_rtf(), encoding="cp1252")

    parsed = _parse_ukda_4688_interview(source, strict_speakers=True)
    segments, audit = _build_exchange_segments_with_audit(
        parsed,
        source_text_path=tmp_path / "source.txt",
        normalized_html_path=tmp_path / "normalized.html",
    )

    assert parsed.participant_characteristics == {
        "pseudonym": "Example",
        "interview_date": "01.02.2001",
        "city": "Edinburgh",
        "suburb_or_urban": "Central",
        "household_structure": "Dual Career",
        "household_composition": "Married couple",
        "childcare": "Nursery",
    }
    assert parsed.speaker_corrections == {"MALEALE": 1, "FEMAIL": 1, "M": 1}
    assert len(segments) == 2

    first = segments[0]
    assert first["interviewer_question"] == "How did work change?"
    assert first["text"] == (
        "Male: I changed jobs, erm, after moving near a caf\u00e9.\n"
        "Female: We discussed [unclear: the hours] together.\n"
        "Female: It made childcare much easier."
    )
    assert first["response_speakers"] == ["Male", "Female"]
    assert len(first["target_turn_indexes"]) == 3
    assert "How did work change?" not in first["text"]
    assert all("Preamble" not in item["text"] for item in segments)
    assert any(turn["text"] == "Right." for turn in first["interview_turns"])

    second = segments[1]
    assert second["segment_id"] == "SEG003"
    assert second["interviewer_question"] == "Where did you stay?"
    assert second["text"] == "Female: We stayed near the school."
    assert [item["decision"] for item in audit] == [
        "retained",
        "rejected",
        "retained",
    ]
    assert audit[1]["primary_reason"] == "acknowledgement_or_no_information"
    assert any(
        turn["speaker_label"] == "Incidental"
        and turn["target_eligible"] is False
        for turn in first["interview_turns"]
    )
    assert all(turn["text"] != "Mmm, yeah." for turn in first["interview_turns"])


def test_profile_writes_auditable_artifacts_without_touching_source(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    transcript_dir = archive / "rtf"
    transcript_dir.mkdir(parents=True)
    source = transcript_dir / "int001t.rtf"
    source_payload = _synthetic_rtf()
    source.write_text(source_payload, encoding="cp1252")
    output = tmp_path / "derived"

    manifest = preprocess_ukda_4688_dataset(
        input_path=archive,
        output_dir=output,
        strict_inventory=False,
    )

    assert source.read_text(encoding="cp1252") == source_payload
    assert manifest["transcript_count"] == 1
    assert manifest["target_selection_policy"] == UKDA_4688_TARGET_SELECTION
    assert manifest["target_filter_audit_path"].endswith(
        "target_filter_audit.jsonl"
    )
    assert (output / "source_text" / "int001t.txt").is_file()
    assert (output / "normalized_html" / "int001t.html").is_file()
    segments_path = output / "segments_jsonl" / "int001t_segments.jsonl"
    assert segments_path.is_file()
    qa = json.loads((output / "preprocessing_qa.json").read_text(encoding="utf-8"))
    assert qa["transcript_count"] == 1
    assert qa["exchange_count"] == 2
    assert qa["candidate_exchange_count"] == 3
    assert qa["retained_exchange_count"] == 2
    assert qa["rejected_exchange_count"] == 1
    assert qa["candidate_exchange_count"] == (
        qa["retained_exchange_count"] + qa["rejected_exchange_count"]
    )
    assert qa["target_rejection_primary_counts"] == {
        "acknowledgement_or_no_information": 1
    }
    assert qa["excluded_filler_turn_count"] >= 1
    records = [json.loads(line) for line in segments_path.read_text().splitlines()]
    assert records[0]["source_rtf_path"] == str(source)
    assert records[0]["source_text_path"].endswith("int001t.txt")
    assert records[0]["target_selection_policy"] == UKDA_4688_TARGET_SELECTION
    audit_records = [
        json.loads(line)
        for line in (output / "target_filter_audit.jsonl").read_text().splitlines()
    ]
    assert len(audit_records) == 3
    assert "interview_turns" not in audit_records[0]
    assert audit_records[1]["decision"] == "rejected"


def test_strict_inventory_requires_exact_documented_archive() -> None:
    names = {f"int{index:03d}t.rtf" for index in range(1, 86)}
    paths = [Path(name) for name in sorted(names)]

    _validate_inventory(
        transcript_paths=paths,
        documented_names=names,
        strict=True,
    )
    with pytest.raises(ValueError, match="strict inventory validation failed"):
        _validate_inventory(
            transcript_paths=paths[:-1],
            documented_names=names,
            strict=True,
        )
    assert len(names) == UKDA_4688_EXPECTED_TRANSCRIPT_COUNT


def test_strict_speaker_validation_rejects_unknown_label(tmp_path: Path) -> None:
    source = tmp_path / "int001t.rtf"
    source.write_text(
        r"{\rtf1\ansi\ansicpg1252 *Q: Known question.\par *OTHER: Unknown.}",
        encoding="cp1252",
    )
    with pytest.raises(ValueError, match="Unknown speaker labels.*OTHER"):
        _parse_ukda_4688_interview(source, strict_speakers=True)


def test_turn_window_renders_metadata_boundaries_and_multi_turn_target() -> None:
    turns = [
        {
            "turn_index": index,
            "speaker": "interviewer" if index % 2 else "participant",
            "text": f"turn {index}",
            "paragraph_index": index,
            "speaker_label": "Interviewer" if index % 2 else "Male",
        }
        for index in range(1, 11)
    ]
    turns[5]["speaker_label"] = "Male"
    record = DatasetRecord(
        record_id="int001t_SEG001",
        text="Male: turn 4\nMale: turn 6",
        metadata={
            "interview_id": "int001t",
            "segment_id": "SEG001",
            "speaker": "participant",
            "turn_index": 4,
            "target_turn_indexes": [4, 6],
            "interviewer_question": "What changed?",
            "participant_characteristics": {"city": "Edinburgh"},
            "interview_turns": turns,
        },
    )

    context = record.analysis_context(
        "turn_window", context_turns_before=2, context_turns_after=2
    )

    assert "- city: Edinburgh" in context
    assert "Question leading to target exchange:\nWhat changed?" in context
    assert "[Earlier normalized turns omitted: 1]" in context
    assert "[Later normalized turns omitted: 2]" in context
    assert context.count("[TARGET SEGMENT SEG001]") == 2
    assert "Turn 1 |" not in context
    assert "Turn 9 |" not in context
    full_context = record.analysis_context("full_interview")
    assert full_context.count("[TARGET SEGMENT SEG001]") == 2


def test_turn_window_cli_defaults_and_prompt_contract_are_additive() -> None:
    args = build_enrichment_parser().parse_args(
        [
            "--segments-path",
            "segments",
            "--output-dir",
            "output",
            "--strategy",
            "self_consistency",
            "--prompt-path",
            "prompt.txt",
        ]
    )
    assert args.context_scope == "immediate"
    assert args.context_turns_before == 20
    assert args.context_turns_after == 20

    original = '{"analysis_context_scope": "full_interview"}'
    assert align_prompt_context_contract(original, "full_interview") == original
    assert align_prompt_context_contract(original, "immediate") == original
    assert align_prompt_context_contract(original, "turn_window") == (
        '{"analysis_context_scope": "turn_window"}'
    )

    preprocessing_args = build_preprocessing_parser().parse_args(
        [
            "rtf",
            "--profile",
            "ukda-4688",
            "--input-path",
            "archive",
            "--output-dir",
            "derived",
            "--strict-inventory",
        ]
    )
    assert preprocessing_args.command == "rtf"
    assert preprocessing_args.profile == "ukda-4688"
    assert preprocessing_args.strict_inventory is True

    review_args = build_preprocessing_parser().parse_args(
        [
            "target-review",
            "--profile",
            "ukda-4688",
            "--audit-path",
            "target_filter_audit.jsonl",
            "--output-path",
            "enrichment_exclusion_review.jsonl",
        ]
    )
    assert review_args.command == "target-review"
    assert review_args.profile == "ukda-4688"

    approve_args = build_preprocessing_parser().parse_args(
        [
            "approve-exclusions",
            "--review-path",
            "enrichment_exclusion_review.jsonl",
            "--audit-path",
            "target_filter_audit.jsonl",
            "--output-path",
            "enrichment_exclusions.jsonl",
        ]
    )
    assert approve_args.command == "approve-exclusions"


def test_ukda_hpc_job_requires_codebook_and_embeds_research_questions() -> None:
    script = (
        Path(__file__).parents[1]
        / "submit_job_enrichment_self_consistency_ukda4688.slurm"
    ).read_text(encoding="utf-8")
    assert 'CONTEXT_SCOPE="turn_window"' in script
    assert "CONTEXT_TURNS_BEFORE=20" in script
    assert "CONTEXT_TURNS_AFTER=20" in script
    assert "SELF_CONSISTENCY_SAMPLES" not in script
    assert "--strategy single_pass" in script
    assert "prompts/enrichment/self_consistency_four_codes.txt" in script
    assert 'CODEBOOK_PATH="${SCRATCH_DPO}/data/codebooks/example_codes_v1.json"' in script
    assert "CODEBOOK_PATH must be supplied" not in script
    assert "RESEARCH_QUESTIONS_FILE" not in script
    assert (
        '"What factors and processes affect household choices of where to live?"'
        in script
    )
    assert '"How does employment shape family life?"' in script
    assert "EXCLUDE_RECORDS_PATH" in script
    assert 'require_file "${EXCLUDE_RECORDS_PATH}"' in script
    assert '--exclude-records-path "${EXCLUDE_RECORDS_PATH}"' in script
    assert "source ~/.bashrc" not in script
    assert 'source "${CONDA_BASE}/etc/profile.d/conda.sh"' in script
    assert "set +u" in script
    assert "set -eo pipefail" in script
    assert "set -euo pipefail" not in script


def test_ukda_preprocessing_script_avoids_user_bashrc_nounset_failure() -> None:
    script = (
        Path(__file__).parents[1] / "run_preprocessing_ukda4688.sh"
    ).read_text(encoding="utf-8")

    assert "set +u" in script
    assert "source ~/.bashrc" not in script
    assert 'source "${CONDA_BASE}/etc/profile.d/conda.sh"' in script
    assert "target_filter_audit.jsonl" in script


@pytest.mark.skipif(
    not os.getenv("UKDA4688_PATH"),
    reason="Set UKDA4688_PATH to run the optional private-archive integration check.",
)
def test_optional_real_archive_inventory_and_speaker_parsing() -> None:
    root = Path(os.environ["UKDA4688_PATH"])
    paths = sorted((root / "rtf").glob("int*t.rtf"))
    documented = _documented_transcript_names(root / "4688_file_information.rtf")
    _validate_inventory(
        transcript_paths=paths,
        documented_names=documented,
        strict=True,
    )
    parsed = [
        _parse_ukda_4688_interview(path, strict_speakers=True) for path in paths
    ]
    assert len(parsed) == UKDA_4688_EXPECTED_TRANSCRIPT_COUNT


def _audit_record(
    record_id: str,
    text: str,
    question: str,
    *,
    clear_words: int,
    total_words: int,
) -> dict[str, object]:
    return {
        "candidate_record_id": record_id,
        "decision": "retained",
        "selected_text": text,
        "interviewer_question": question,
        "clear_word_count": clear_words,
        "total_word_count": total_words,
        "target_turn_indexes": [2],
    }


def _write_test_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_test_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _test_turn(
    text: str,
    *,
    role: str = "participant",
    index: int = 2,
) -> RtfTurn:
    participant = role == "participant"
    return RtfTurn(
        role=role,
        speaker_label="Female" if participant else "Interviewer",
        raw_speaker_label="FEMALE" if participant else "Q",
        text=text,
        raw_text=text,
        turn_index=index,
        paragraph_index=index,
        target_eligible=participant,
    )


def _selection_for(
    text: str,
    *,
    question: str = "Please tell me more about your experience.",
):
    return _select_analytical_target(
        _test_turn(question, role="interviewer", index=1),
        [_test_turn(text, index=2)],
    )


def _synthetic_rtf() -> str:
    return (
        r"{\rtf1\ansi\ansicpg1252\deff0 "
        r"Pseudonym\tab Date of Interview\tab City\tab Suburb / Urban\par "
        r"Example\tab 01.02.2001\tab Edinburgh\tab Central\par "
        r"Household Structure\tab Household Composition\tab Childcare\par "
        r"Dual Career\tab Married couple\tab Nursery\par "
        r"*FEMALE: Preamble material.\par "
        r"*Q: How did work change?\par "
        r"*MALEALE: I changed jobs, erm, after moving near a caf\'e9.\par "
        r"*Q: Mmm, yeah.\par "
        r"*FEMAIL: We discussed ?? the hours ?? together.\par "
        r"*Q: Right.\par "
        r"*FEMALE: It made childcare much easier.\par "
        r"*IE1: Hi.\par "
        r"*Q; Would you move again?\par "
        r"*M: No.\par "
        r"*Q: Where did you stay?\par "
        r"*FEMALE: We stayed near the school.\par "
        r"*Q: Huh huh.\par "
        r"*FEMALE: ???\par }"
    )
