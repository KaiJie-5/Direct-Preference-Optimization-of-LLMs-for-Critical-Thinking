from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from dpo_critical_thinking.debate.agents import DebateGenerationResult
from dpo_critical_thinking.debate.aggregation import borda_aggregate
from dpo_critical_thinking.debate.compare import compare_rankings
from dpo_critical_thinking.debate.config import (
    DebateConfig,
    GenerationConfig,
    TurnConfig,
    debate_config_from_mapping,
)
from dpo_critical_thinking.debate.loaders import build_block_inputs
from dpo_critical_thinking.debate.ranking import _rank_block, run_debate_ranking
from dpo_critical_thinking.debate.schema import (
    REVIEW_BLOCKS,
    configured_review_blocks,
    validate_ranking_payload,
)


def test_review_pack_loader_reconstructs_candidates_and_reflective_questions(
    tmp_path: Path,
) -> None:
    review_pack, enriched_parent = _write_review_pack_fixture(tmp_path)

    block_inputs = build_block_inputs(
        review_pack_path=review_pack,
        dataset_configs=(
            _dataset_config(
                "energy",
                enriched_parent,
                research_questions=("What interactions happen?",),
            ),
        ),
        review_blocks=configured_review_blocks(None),
    )

    assert len(block_inputs) == 7
    wrong_code = block_inputs[0]
    assert wrong_code.review_block.id == "wrong_code"
    assert wrong_code.participant_segment_text == "Participant text."
    assert wrong_code.previous_context == "Interviewer: Previous question?"
    assert wrong_code.next_context == "Interviewer: Next question?"
    assert wrong_code.research_questions == ("What interactions happen?",)
    assert wrong_code.candidate_table[0]["candidate_label"] == "A"
    assert wrong_code.candidate_table[0]["original_sample_index"] == 2
    assert wrong_code.candidate_table[0]["fields"]["code_label"] == "wrong 2"

    q3 = block_inputs[-1]
    assert q3.review_block.id == "reflective_question_q3"
    assert q3.candidate_table[-1]["candidate_label"] == "E"
    assert q3.candidate_table[-1]["fields"]["question"] == "q3 sample 5"


def test_validate_ranking_payload_rejects_incomplete_duplicate_and_unknown_labels() -> None:
    assert validate_ranking_payload(
        {"ranking": ["A", "B", "C", "D", "E"], "rationale": "ok"},
        record_id="INT01_SEG001",
        review_block="wrong_code",
    ) == []

    duplicate_errors = validate_ranking_payload(
        {"ranking": ["A", "A", "C", "D", "E"], "rationale": "ok"},
        record_id="INT01_SEG001",
        review_block="wrong_code",
    )
    assert any("complete permutation" in error for error in duplicate_errors)

    incomplete_errors = validate_ranking_payload(
        {"ranking": ["A", "B"], "rationale": "ok"},
        record_id="INT01_SEG001",
        review_block="wrong_code",
    )
    assert any("exactly 5" in error for error in incomplete_errors)

    unknown_errors = validate_ranking_payload(
        {"ranking": ["A", "B", "C", "D", "Z"], "rationale": "ok"},
        record_id="INT01_SEG001",
        review_block="wrong_code",
    )
    assert any("complete permutation" in error for error in unknown_errors)


def test_borda_aggregation_uses_qwen_72b_tiebreak_and_records_it() -> None:
    result = borda_aggregate(
        [
            {
                "agent_id": "qwen_32b",
                "ranking": ["A", "B", "C", "D", "E"],
            },
            {
                "agent_id": "qwen_72b",
                "ranking": ["B", "A", "C", "D", "E"],
            },
        ],
        tiebreak_agent_id="qwen_72b",
    )

    assert result["ranking"][:2] == ["B", "A"]
    assert result["scores"]["A"] == 9
    assert result["scores"]["B"] == 9
    assert result["tiebreak_applied"] is True
    assert result["tiebreaks"][0]["applied_order"] == ["B", "A"]


def test_qwen_debate_prompt_templates_start_with_task_body() -> None:
    repo_root = Path(__file__).parents[1]
    qwen32 = (
        repo_root / "prompts" / "debate" / "qwen_32b_debate_placeholder.txt"
    ).read_text(encoding="utf-8")
    qwen72 = (
        repo_root / "prompts" / "debate" / "qwen_72b_debate_placeholder.txt"
    ).read_text(encoding="utf-8")

    assert qwen32.splitlines()[0] == "Rank Candidate A-E for one qualitative review block."
    assert (
        qwen72.splitlines()[0]
        == "Audit and rank Candidate A-E for one qualitative review block."
    )
    for template in (qwen32, qwen72):
        assert not template.startswith("You are {agent_name}. Act as")
        assert "Previous rankings and rationales JSON:" in template
        assert "Previous agent trace JSON:" not in template


def test_dry_run_debate_writes_segment_trace_final_jsonl_and_long_csv(
    tmp_path: Path,
) -> None:
    review_pack, enriched_parent = _write_review_pack_fixture(tmp_path)
    qwen32_prompt = tmp_path / "qwen32.txt"
    qwen72_prompt = tmp_path / "qwen72.txt"
    qwen32_prompt.write_text(
        "QWEN32 {agent_role} {turn_id} {record_id} {review_block} "
        "{research_questions} {previous_context} {participant_segment_text} "
        "{next_context} {candidate_table_json} "
        "Previous rankings and rationales JSON: {previous_agent_trace_json}",
        encoding="utf-8",
    )
    qwen72_prompt.write_text(
        "QWEN72 {agent_role} {turn_id} {record_id} {review_block} "
        "Previous rankings and rationales JSON: {previous_agent_trace_json}",
        encoding="utf-8",
    )
    config = debate_config_from_mapping(
        {
            "review_pack_path": str(review_pack),
            "output_root": str(tmp_path / "debate_outputs"),
            "run_name": "dry_run_debate",
            "datasets": [
                {
                    "dataset": "energy",
                    "enriched_parent_path": str(enriched_parent),
                    "research_questions": ["What interactions happen?"],
                }
            ],
            "agents": [
                {
                    "id": "qwen_32b",
                    "name": "Qwen 32B",
                    "role": "Interpretive QDA Methodologist",
                    "backend": "dry-run",
                    "prompt_path": str(qwen32_prompt),
                },
                {
                    "id": "qwen_72b",
                    "name": "Qwen 72B",
                    "role": "Reflexive Evidence and Interpretation Auditor",
                    "backend": "dry-run",
                    "prompt_path": str(qwen72_prompt),
                },
            ],
            "turns": [
                {
                    "id": "turn1_initial_32b",
                    "agent_id": "qwen_32b",
                    "role": "initial_ranking",
                    "contributes_to_aggregation": False,
                },
                {
                    "id": "turn2_response_72b",
                    "agent_id": "qwen_72b",
                    "role": "response_agreement_disagreement",
                    "contributes_to_aggregation": False,
                },
                {
                    "id": "turn3_revision_32b",
                    "agent_id": "qwen_32b",
                    "role": "revised_ranking",
                    "contributes_to_aggregation": True,
                },
                {
                    "id": "turn4_final_72b",
                    "agent_id": "qwen_72b",
                    "role": "final_ranking",
                    "contributes_to_aggregation": True,
                },
            ],
            "generation": {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "json_retry_attempts": 0,
            },
        },
        base_dir=tmp_path,
    )

    run_dir = run_debate_ranking(config)

    assert not (run_dir / "debate_trace.jsonl").exists()
    segment_trace_path = run_dir / "debate_traces" / "energy" / "INT01_SEG001.json"
    segment_trace = json.loads(segment_trace_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    final_rows = _read_jsonl(run_dir / "final_rankings.jsonl")
    long_rows = _read_csv(run_dir / "final_rankings_long.csv")
    trace_rows = [
        turn
        for block in segment_trace["review_blocks"].values()
        for turn in block["turns"]
    ]

    assert "debate_trace" not in manifest["output_files"]
    assert "debate_traces_dir" in manifest["output_files"]
    assert manifest["trace_storage"]["path_template"] == (
        "debate_traces/{dataset}/{record_id}.json"
    )
    assert segment_trace["dataset"] == "energy"
    assert segment_trace["record_id"] == "INT01_SEG001"
    assert segment_trace["participant_segment_text"] == "Participant text."
    assert segment_trace["previous_context"] == "Interviewer: Previous question?"
    assert segment_trace["next_context"] == "Interviewer: Next question?"
    assert segment_trace["research_questions"] == ["What interactions happen?"]
    assert len(segment_trace["review_blocks"]) == 7
    assert segment_trace["review_blocks"]["wrong_code"]["status"] == "success"
    assert len(segment_trace["review_blocks"]["wrong_code"]["turns"]) == 4
    assert segment_trace["review_blocks"]["wrong_code"]["final_ranking"] == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]
    assert segment_trace["review_blocks"]["wrong_code"]["borda_scores"]["A"] == 10
    assert "tiebreaks" in segment_trace["review_blocks"]["wrong_code"]
    assert len(trace_rows) == 28
    assert len(final_rows) == 1
    assert len(long_rows) == 7
    assert final_rows[0]["rankings"]["wrong_code"] == ["A", "B", "C", "D", "E"]
    assert long_rows[0]["review_block"] == "wrong_code"
    assert long_rows[0]["segment_trace_file"] == str(
        Path("debate_traces") / "energy" / "INT01_SEG001.json"
    )
    assert long_rows[0]["candidate_A_sample_index"] == "2"
    assert trace_rows[0]["rendered_prompt"].startswith(
        "You are Qwen 32B. Act as the Interpretive QDA Methodologist.\n\nQWEN32"
    )
    assert trace_rows[1]["rendered_prompt"].startswith(
        "You are Qwen 72B. Act as the Reflexive Evidence and Interpretation "
        "Auditor.\n\nQWEN72"
    )
    assert "What interactions happen?" in trace_rows[0]["rendered_prompt"]
    assert "Interviewer: Previous question?" in trace_rows[0]["rendered_prompt"]
    assert "Interviewer: Next question?" in trace_rows[0]["rendered_prompt"]
    assert "QWEN32" in trace_rows[0]["rendered_prompt"]
    assert "QWEN72" in trace_rows[1]["rendered_prompt"]
    assert "QWEN32" in trace_rows[2]["rendered_prompt"]
    assert "QWEN72" in trace_rows[3]["rendered_prompt"]
    assert "Interpretive QDA Methodologist" in trace_rows[0]["rendered_prompt"]
    assert (
        "Reflexive Evidence and Interpretation Auditor"
        in trace_rows[1]["rendered_prompt"]
    )
    assert trace_rows[1]["turn_id"] == "turn2_response_72b"
    assert '"turn_id": "turn1_initial_32b"' in trace_rows[1]["rendered_prompt"]
    assert '"turn_id": "turn2_response_72b"' in trace_rows[2]["rendered_prompt"]
    assert '"turn_id": "turn3_revision_32b"' in trace_rows[3]["rendered_prompt"]
    assert "Dry-run ranking for smoke testing." in trace_rows[1]["rendered_prompt"]


def test_four_turn_debate_aggregates_only_turn_3_and_turn_4(tmp_path: Path) -> None:
    review_pack, enriched_parent = _write_review_pack_fixture(tmp_path)
    block_input = build_block_inputs(
        review_pack_path=review_pack,
        dataset_configs=(_dataset_config("energy", enriched_parent),),
        review_blocks=configured_review_blocks(["wrong_code"]),
    )[0]
    prompt_paths = _write_turn_prompts(tmp_path)
    turns = _turns(prompt_paths)
    qwen32 = QueueDebateAgent(
        "qwen_32b",
        [
            _ranking_text(["A", "B", "C", "D", "E"], "early 32b"),
            _ranking_text(["E", "D", "C", "B", "A"], "terminal 32b"),
        ],
    )
    qwen72 = QueueDebateAgent(
        "qwen_72b",
        [
            _ranking_text(["B", "A", "C", "D", "E"], "early 72b"),
            _ranking_text(["D", "E", "C", "B", "A"], "terminal 72b"),
        ],
    )

    result = _rank_block(
        block_input=block_input,
        turns=turns,
        agent_by_id={"qwen_32b": qwen32, "qwen_72b": qwen72},
        agent_config_by_id=_agent_config_by_id(prompt_paths),
        prompt_by_agent_id=_prompt_templates(prompt_paths),
        config=_dummy_config(tmp_path, review_pack, enriched_parent, turns),
    )

    assert result["status"] == "success"
    assert result["final_ranking"][:2] == ["D", "E"]
    assert result["borda_scores"]["A"] == 2
    assert result["borda_scores"]["D"] == 9
    assert result["borda_scores"]["E"] == 9
    assert [item["turn_id"] for item in result["aggregation_rankings"]] == [
        "turn3_revision_32b",
        "turn4_final_72b",
    ]
    assert result["tiebreak_agent_id"] == "qwen_72b"
    for call in [*qwen32.calls, *qwen72.calls]:
        assert [message["role"] for message in call] == ["system", "user"]

    assert qwen32.calls[0][0]["content"] == (
        "You are qwen_32b. Act as the Interpretive QDA Methodologist."
    )
    assert qwen72.calls[0][0]["content"] == (
        "You are qwen_72b. Act as the Reflexive Evidence and Interpretation Auditor."
    )
    assert not qwen32.calls[0][1]["content"].startswith("You are qwen_32b. Act as")
    assert "Previous rankings and rationales JSON:" in qwen72.calls[0][1]["content"]

    turn2_user_prompt = qwen72.calls[0][1]["content"]
    assert '"turn_id": "turn1_initial_32b"' in turn2_user_prompt
    assert "early 32b" in turn2_user_prompt
    assert "early 72b" not in turn2_user_prompt

    turn3_user_prompt = qwen32.calls[1][1]["content"]
    assert '"turn_id": "turn1_initial_32b"' in turn3_user_prompt
    assert '"turn_id": "turn2_response_72b"' in turn3_user_prompt
    assert "early 32b" in turn3_user_prompt
    assert "early 72b" in turn3_user_prompt

    turn4_user_prompt = qwen72.calls[1][1]["content"]
    assert '"turn_id": "turn1_initial_32b"' in turn4_user_prompt
    assert '"turn_id": "turn2_response_72b"' in turn4_user_prompt
    assert '"turn_id": "turn3_revision_32b"' in turn4_user_prompt
    assert "terminal 32b" in turn4_user_prompt
    assert "terminal 72b" not in turn4_user_prompt


def test_failed_terminal_turn_marks_block_failed(tmp_path: Path) -> None:
    review_pack, enriched_parent = _write_review_pack_fixture(tmp_path)
    block_input = build_block_inputs(
        review_pack_path=review_pack,
        dataset_configs=(_dataset_config("energy", enriched_parent),),
        review_blocks=configured_review_blocks(["wrong_code"]),
    )[0]
    prompt_paths = _write_turn_prompts(tmp_path)
    turns = _turns(prompt_paths)
    qwen32 = QueueDebateAgent(
        "qwen_32b",
        [
            _ranking_text(["A", "B", "C", "D", "E"], "early 32b"),
            '{"ranking": ["A", "A"], "rationale": "bad"}',
        ],
    )
    qwen72 = QueueDebateAgent(
        "qwen_72b",
        [_ranking_text(["B", "A", "C", "D", "E"], "early 72b")],
    )

    result = _rank_block(
        block_input=block_input,
        turns=turns,
        agent_by_id={"qwen_32b": qwen32, "qwen_72b": qwen72},
        agent_config_by_id=_agent_config_by_id(prompt_paths),
        prompt_by_agent_id=_prompt_templates(prompt_paths),
        config=_dummy_config(tmp_path, review_pack, enriched_parent, turns),
    )

    assert result["status"] == "failed"
    assert result["failed_turn_id"] == "turn3_revision_32b"
    assert result["failed_agent_id"] == "qwen_32b"
    assert len(result["turns"]) == 3
    assert result["turns"][-1]["parse_status"] == "invalid"


def test_missing_context_fields_are_empty_strings(tmp_path: Path) -> None:
    review_pack, enriched_parent = _write_review_pack_fixture(
        tmp_path,
        include_context=False,
    )

    block_input = build_block_inputs(
        review_pack_path=review_pack,
        dataset_configs=(_dataset_config("energy", enriched_parent),),
        review_blocks=configured_review_blocks(["wrong_code"]),
    )[0]

    assert block_input.previous_context == ""
    assert block_input.next_context == ""


def test_compare_rankings_outputs_descriptive_alignment_only(tmp_path: Path) -> None:
    model_csv = tmp_path / "model.csv"
    reviewer_csv = tmp_path / "reviewer.csv"
    _write_csv(
        model_csv,
        [
            {
                "dataset": "energy",
                "record_id": "INT01_SEG001",
                "review_block": "wrong_code",
                "status": "success",
                "final_rank_1": "A",
                "final_rank_2": "B",
                "final_rank_3": "C",
                "final_rank_4": "D",
                "final_rank_5": "E",
            }
        ],
    )
    _write_csv(
        reviewer_csv,
        [
            {
                "reviewer_name": "Dr Richard",
                "dataset": "energy",
                "record_id": "INT01_SEG001",
                "review_block": "wrong_code",
                "rank_1_most_preferable": "A",
                "rank_2": "C",
                "rank_3": "B",
                "rank_4": "D",
                "rank_5_least_preferable": "E",
            }
        ],
    )

    outputs = compare_rankings(
        model_csv=model_csv,
        reviewer_csvs=[reviewer_csv],
        output_dir=tmp_path / "alignment",
    )

    rows = _read_csv(outputs["alignment_rows"])
    summary = _read_csv(outputs["summary"])
    assert rows[0]["top1_match"] == "True"
    assert rows[0]["full_rank_match"] == "False"
    assert summary[0]["dataset"] == "energy"
    assert summary[0]["review_block"] == "wrong_code"


def _dataset_config(
    dataset: str,
    enriched_parent: Path,
    research_questions: tuple[str, ...] = (),
):
    from dpo_critical_thinking.debate.config import DatasetConfig

    return DatasetConfig(
        dataset=dataset,
        enriched_parent_path=enriched_parent,
        research_questions=research_questions,
    )


class QueueDebateAgent:
    def __init__(self, agent_id: str, outputs: list[str]) -> None:
        self.agent_id = agent_id
        self.name = agent_id
        self.outputs = outputs
        self.calls: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
    ) -> DebateGenerationResult:
        self.calls.append([dict(message) for message in messages])
        return DebateGenerationResult(
            text=self.outputs.pop(0),
            raw={"message_count": len(messages)},
            rendered_prompt=messages[-1]["content"],
            elapsed_seconds=0.0,
        )

    def metadata(self) -> dict[str, Any]:
        return {"backend": "queue", "agent_id": self.agent_id}


def _ranking_text(ranking: list[str], rationale: str) -> str:
    return json.dumps({"ranking": ranking, "rationale": rationale})


def _write_turn_prompts(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "qwen_32b": tmp_path / "qwen32.txt",
        "qwen_72b": tmp_path / "qwen72.txt",
    }
    for agent_id, path in paths.items():
        path.write_text(
            f"{agent_id} {{agent_role}} {{turn_id}} {{record_id}} "
            "{review_block} Previous rankings and rationales JSON: "
            "{previous_agent_trace_json}",
            encoding="utf-8",
        )
    return paths


def _turns(prompt_paths: dict[str, Path]) -> list[TurnConfig]:
    return [
        TurnConfig(
            id="turn1_initial_32b",
            agent_id="qwen_32b",
            role="initial_ranking",
            contributes_to_aggregation=False,
        ),
        TurnConfig(
            id="turn2_response_72b",
            agent_id="qwen_72b",
            role="response_agreement_disagreement",
            contributes_to_aggregation=False,
        ),
        TurnConfig(
            id="turn3_revision_32b",
            agent_id="qwen_32b",
            role="revised_ranking",
            contributes_to_aggregation=True,
        ),
        TurnConfig(
            id="turn4_final_72b",
            agent_id="qwen_72b",
            role="final_ranking",
            contributes_to_aggregation=True,
        ),
    ]


def _prompt_templates(prompt_paths: dict[str, Path]):
    from dpo_critical_thinking.enrichment.prompts import PromptTemplate

    return {
        agent_id: PromptTemplate(path)
        for agent_id, path in prompt_paths.items()
    }


def _agent_config_by_id(prompt_paths: dict[str, Path]):
    from dpo_critical_thinking.debate.config import AgentConfig

    return {
        "qwen_32b": AgentConfig(
            id="qwen_32b",
            name="Qwen 32B",
            role="Interpretive QDA Methodologist",
            backend="dry-run",
            model_path=None,
            prompt_path=prompt_paths["qwen_32b"],
        ),
        "qwen_72b": AgentConfig(
            id="qwen_72b",
            name="Qwen 72B",
            role="Reflexive Evidence and Interpretation Auditor",
            backend="dry-run",
            model_path=None,
            prompt_path=prompt_paths["qwen_72b"],
        ),
    }


def _dummy_config(
    tmp_path: Path,
    review_pack: Path,
    enriched_parent: Path,
    turns: list[TurnConfig],
) -> DebateConfig:
    from dpo_critical_thinking.debate.config import AgentConfig, DatasetConfig

    return DebateConfig(
        review_pack_path=review_pack,
        output_root=tmp_path,
        datasets=(DatasetConfig(dataset="energy", enriched_parent_path=enriched_parent),),
        agents=(
            AgentConfig(
                id="qwen_32b",
                name="Qwen 32B",
                role="Interpretive QDA Methodologist",
                backend="dry-run",
                model_path=None,
                prompt_path=tmp_path / "qwen32.txt",
            ),
            AgentConfig(
                id="qwen_72b",
                name="Qwen 72B",
                role="Reflexive Evidence and Interpretation Auditor",
                backend="dry-run",
                model_path=None,
                prompt_path=tmp_path / "qwen72.txt",
            ),
        ),
        turns=tuple(turns),
        generation=GenerationConfig(json_retry_attempts=0),
    )


def _write_review_pack_fixture(
    tmp_path: Path,
    *,
    include_context: bool = True,
) -> tuple[Path, Path]:
    review_pack = tmp_path / "review_pack"
    review_pack.mkdir()
    enriched_parent = tmp_path / "enriched_parent"
    run_dir = enriched_parent / "run1" / "INT01_self_consistency" / "segments"
    run_dir.mkdir(parents=True)
    segment_path = run_dir / "INT01_SEG001.json"
    segment_path.write_text(
        json.dumps(_segment_payload(include_context=include_context), ensure_ascii=False),
        encoding="utf-8",
    )

    relative = "INT01_self_consistency\\segments\\INT01_SEG001.json"
    _write_csv(
        review_pack / "review_segments.csv",
        [
            {
                "dataset": "energy",
                "transcript_id": "INT01",
                "segment_id": "SEG001",
                "record_id": "INT01_SEG001",
                "whole_interview_file": "INT01.html",
                "run_name": "run1",
                "segment_json_relative_to_run": relative,
            }
        ],
    )
    mapping_rows = []
    sample_indexes = {"A": 2, "B": 1, "C": 3, "D": 4, "E": 5}
    for block in REVIEW_BLOCKS:
        for label, sample_index in sample_indexes.items():
            mapping_rows.append(
                {
                    "dataset": "energy",
                    "transcript_id": "INT01",
                    "segment_id": "SEG001",
                    "record_id": "INT01_SEG001",
                    "review_block": block.id,
                    "candidate_label": label,
                    "original_sample_index": sample_index,
                    "run_name": "run1",
                    "segment_json_relative_to_run": relative,
                }
            )
    _write_csv(review_pack / "internal_candidate_mapping.csv", mapping_rows)
    return review_pack, enriched_parent


def _segment_payload(*, include_context: bool = True) -> dict:
    payload = {
        "record_id": "INT01_SEG001",
        "input_text": "Participant text.",
        "samples": [
            {
                "sample_index": index,
                "final_parse_status": "valid",
                "parsed_output": _parsed_output(index),
            }
            for index in range(1, 6)
        ],
    }
    if include_context:
        payload["metadata"] = {
            "previous_context": "Interviewer: Previous question?",
            "next_context": "Interviewer: Next question?",
        }
    return payload


def _parsed_output(index: int) -> dict:
    return {
        "code_quality_examples": {
            "wrong_code": {
                "code_label": f"wrong {index}",
                "actual_segment_quote": f"quote {index}",
                "why_plausible_for_wider_dataset": "plausible",
                "why_unsupported_by_this_segment": "unsupported",
                "relation_to_research_questions": "relation",
                "category_boundary": "boundary",
            },
            "descriptive_not_answering_research_question": {
                "code_label": f"descriptive {index}",
                "evidence_quote": "evidence",
                "surface_description": "surface",
                "why_true_of_segment": "true",
                "why_not_useful_for_research_questions": "not useful",
                "relation_to_research_questions": "relation",
                "category_boundary": "boundary",
            },
            "too_broad_code": {
                "code_label": f"broad {index}",
                "evidence_quote": "evidence",
                "broad_relevance_to_research_questions": "broad",
                "specific_meaning_lost": "lost",
                "why_it_is_too_broad": "too broad",
                "relation_to_research_questions": "relation",
                "category_boundary": "boundary",
            },
            "useful_analytical_code": {
                "code_label": f"useful {index}",
                "evidence_quote": "evidence",
                "specific_analytical_insight": "insight",
                "why_it_is_useful": "useful",
                "relation_to_research_questions": "relation",
                "why_better_than_other_three": "better",
                "category_boundary": "boundary",
            },
        },
        "reflective_question_candidates": [
            {"question": f"q1 sample {index}", "question_type": "type"},
            {"question": f"q2 sample {index}", "question_type": "type"},
            {"question": f"q3 sample {index}", "question_type": "type"},
        ],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
