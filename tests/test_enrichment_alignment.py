from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from dpo_critical_thinking.enrichment.data import load_records
from dpo_critical_thinking.enrichment.logging import RunLogger
from dpo_critical_thinking.enrichment.prompts import PromptTemplate
from dpo_critical_thinking.enrichment.strategies import (
    run_self_consistency,
    run_self_refine,
)
from dpo_critical_thinking.enrichment.teachers import (
    GenerationOptions,
    GenerationResult,
)


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


def test_participant_html_split_matches_observed_shape(tmp_path: Path) -> None:
    html_path = tmp_path / "transcripts-energy.html"
    html_path.write_text(_synthetic_transcript_html(), encoding="utf-8")

    records = load_records(
        html_path,
        input_format="html",
        html_split_mode="participant",
    )

    assert len(records) == 10
    assert records[0].record_id == "P1"
    assert records[-1].record_id == "P10"
    assert sum(record.metadata["interviewer_turn_count"] for record in records) == 104
    assert sum(record.metadata["participant_turn_count"] for record in records) == 104
    assert all(record.metadata["demographics"] for record in records)
    assert records[0].text.startswith("Interviewer: Question 1")
    assert records[0].source["section_index"] == 1


def test_self_consistency_scaffold_logs_samples_without_selection(tmp_path: Path) -> None:
    prompt_path = _write_prompt(tmp_path, "Prompt {record_id}: {input_text}")
    logger = RunLogger(tmp_path / "logs")
    teacher = QueueTeacher(["sample one", "sample two"])
    record = _record()

    enriched = run_self_consistency(
        record=record,
        teacher=teacher,
        prompt=PromptTemplate(prompt_path),
        prompt_vars={},
        generation_options=GenerationOptions(seed=10),
        num_samples=2,
        aggregation="scaffold",
        logger=logger,
    )

    assert enriched["selected_output"] is None
    assert enriched["selected_sample_index"] is None
    assert (
        enriched["aggregation_status"]
        == "deferred_open_text_consistency_metric_required"
    )
    assert [sample["generation_options"]["seed"] for sample in enriched["samples"]] == [
        10,
        11,
    ]


@pytest.mark.parametrize("aggregation", ["first", "longest", "none"])
def test_self_consistency_rejects_old_selection_modes(
    tmp_path: Path, aggregation: str
) -> None:
    prompt_path = _write_prompt(tmp_path, "Prompt {record_id}: {input_text}")

    with pytest.raises(ValueError, match="scaffold-only"):
        run_self_consistency(
            record=_record(),
            teacher=QueueTeacher(["sample"]),
            prompt=PromptTemplate(prompt_path),
            prompt_vars={},
            generation_options=GenerationOptions(),
            num_samples=1,
            aggregation=aggregation,
            logger=RunLogger(tmp_path / "logs"),
        )


def test_self_refine_stops_on_feedback_indicator(tmp_path: Path) -> None:
    initial = _write_prompt(tmp_path, "Initial {input_text}")
    feedback = _write_prompt(tmp_path, "Feedback {current_answer}")
    revision = _write_prompt(tmp_path, "Revision {refinement_history} {feedback}")
    teacher = QueueTeacher(
        [
            "initial answer",
            '{"needs_refinement": false, "reason": "already sufficient"}',
        ]
    )

    enriched = run_self_refine(
        record=_record(),
        teacher=teacher,
        initial_prompt=PromptTemplate(initial),
        critique_prompt=PromptTemplate(feedback),
        revision_prompt=PromptTemplate(revision),
        prompt_vars={},
        generation_options=GenerationOptions(),
        refine_rounds=3,
        stop_parser="json",
        history_format="text",
        logger=RunLogger(tmp_path / "logs"),
    )

    assert enriched["selected_output"] == "initial answer"
    assert enriched["completed_refinement_rounds"] == 0
    assert enriched["final_stop_decision"]["should_stop"] is True
    assert len(teacher.prompts) == 2


def test_self_refine_passes_full_history_until_max_rounds(tmp_path: Path) -> None:
    initial = _write_prompt(tmp_path, "Initial {input_text}")
    feedback = _write_prompt(tmp_path, "Feedback {current_answer}")
    revision = _write_prompt(
        tmp_path,
        "Revision\nHistory:\n{refinement_history}\nFeedback:\n{feedback}",
    )
    teacher = QueueTeacher(
        [
            "initial answer",
            '{"needs_refinement": true, "reason": "improve once"}',
            "revision one",
            '{"needs_refinement": true, "reason": "improve twice"}',
            "revision two",
        ]
    )

    enriched = run_self_refine(
        record=_record(),
        teacher=teacher,
        initial_prompt=PromptTemplate(initial),
        critique_prompt=PromptTemplate(feedback),
        revision_prompt=PromptTemplate(revision),
        prompt_vars={},
        generation_options=GenerationOptions(),
        refine_rounds=2,
        stop_parser="json",
        history_format="text",
        logger=RunLogger(tmp_path / "logs"),
    )

    assert enriched["selected_output"] == "revision two"
    assert enriched["completed_refinement_rounds"] == 2
    assert "Round 0 initial" in teacher.prompts[2]
    assert "Round 1 feedback" in teacher.prompts[2]
    assert "Round 1 revision" in teacher.prompts[4]
    assert "Round 2 feedback" in teacher.prompts[4]


def _record() -> Any:
    from dpo_critical_thinking.enrichment.data import DatasetRecord

    return DatasetRecord(record_id="P1", text="Example interview")


def _write_prompt(tmp_path: Path, text: str) -> Path:
    path = tmp_path / f"prompt_{len(list(tmp_path.glob('prompt_*')))}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _synthetic_transcript_html() -> str:
    section_turn_counts = [11, 11, 11, 11, 10, 10, 10, 10, 10, 10]
    sections = []
    for participant_index, turn_count in enumerate(section_turn_counts, start=1):
        turns = []
        for turn_index in range(1, turn_count + 1):
            turns.append(
                '<p class="interviewer"><strong>Interviewer</strong> '
                f"Question {turn_index}</p>"
            )
            turns.append(
                '<p class="participant"><strong>Participant</strong> '
                f"Answer {turn_index}</p>"
            )
        sections.append(
            f"""
            <h2>P{participant_index}</h2>
            <table>
              <tr><td>Gender</td><td>Example</td></tr>
              <tr><td>Age</td><td>{20 + participant_index}</td></tr>
            </table>
            {''.join(turns)}
            """
        )
    return f"<!DOCTYPE html><html><body>{''.join(sections)}</body></html>"
