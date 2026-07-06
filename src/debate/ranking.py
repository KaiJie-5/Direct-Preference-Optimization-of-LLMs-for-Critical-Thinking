from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from enrichment.prompts import PromptTemplate
from enrichment.schema import parse_json_object

from .agents import DebateAgent, build_agent
from .aggregation import borda_aggregate
from .config import DebateConfig, TurnConfig, config_to_jsonable
from .loaders import DebateBlockInput, build_block_inputs
from .schema import (
    CANDIDATE_LABELS,
    configured_review_blocks,
    ranking_from_payload,
    validate_ranking_payload,
)


def run_debate_ranking(config: DebateConfig) -> Path:
    run_dir = _new_run_dir(config.output_root, config.run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = run_dir / "debate_traces"
    trace_dir.mkdir()
    failures_path = run_dir / "failures.jsonl"
    final_jsonl_path = run_dir / "final_rankings.jsonl"
    final_csv_path = run_dir / "final_rankings_long.csv"

    review_blocks = configured_review_blocks(list(config.review_blocks))
    agents = [build_agent(agent_config) for agent_config in config.agents]
    agent_by_id = {agent.agent_id: agent for agent in agents}
    agent_config_by_id = {agent_config.id: agent_config for agent_config in config.agents}
    prompt_by_agent_id = {
        agent_config.id: PromptTemplate(agent_config.prompt_path)
        for agent_config in config.agents
    }
    block_inputs = build_block_inputs(
        review_pack_path=config.review_pack_path,
        dataset_configs=config.datasets,
        review_blocks=review_blocks,
        limit=config.limit,
    )

    _write_json(
        run_dir / "run_manifest.json",
        {
            "timestamp_utc": _timestamp(),
            "config": config_to_jsonable(config),
            "review_blocks": [asdict(block) for block in review_blocks],
            "agents": [agent.metadata() for agent in agents],
            "agent_configs": [
                {
                    "id": agent_config.id,
                    "name": agent_config.name,
                    "role": agent_config.role,
                    "prompt_path": str(agent_config.prompt_path),
                }
                for agent_config in config.agents
            ],
            "output_files": {
                "debate_traces_dir": str(trace_dir),
                "failures": str(failures_path),
                "final_rankings": str(final_jsonl_path),
                "final_rankings_long": str(final_csv_path),
            },
            "trace_storage": {
                "layout": "one JSON file per dataset + record_id",
                "path_template": "debate_traces/{dataset}/{record_id}.json",
            },
        },
    )
    failures_path.touch()

    record_results: dict[tuple[str, str], dict[str, Any]] = {}
    record_traces: dict[tuple[str, str], dict[str, Any]] = {}
    long_rows: list[dict[str, Any]] = []
    for index, block_input in enumerate(block_inputs, start=1):
        print(
            "Ranking "
            f"{block_input.segment.dataset} {block_input.segment.record_id} "
            f"{block_input.review_block.id} ({index}/{len(block_inputs)})",
            flush=True,
        )
        result = _rank_block(
            block_input=block_input,
            turns=list(config.turns),
            agent_by_id=agent_by_id,
            agent_config_by_id=agent_config_by_id,
            prompt_by_agent_id=prompt_by_agent_id,
            config=config,
        )
        segment_trace_file = _segment_trace_file(trace_dir, block_input)
        result["segment_trace_file"] = str(segment_trace_file.relative_to(run_dir))
        if result["status"] != "success":
            _append_jsonl(failures_path, result)

        record_key = (block_input.segment.dataset, block_input.segment.record_id)
        record_trace = record_traces.setdefault(
            record_key,
            _new_segment_trace_payload(block_input, config.ranking_method),
        )
        record_trace["updated_at_utc"] = _timestamp()
        record_trace["review_blocks"][block_input.review_block.id] = _block_trace_payload(
            result
        )
        _write_json(segment_trace_file, record_trace)

        record_payload = record_results.setdefault(
            record_key,
            {
                "dataset": block_input.segment.dataset,
                "record_id": block_input.segment.record_id,
                "transcript_id": block_input.segment.transcript_id,
                "segment_id": block_input.segment.segment_id,
                "ranking_method": config.ranking_method,
                "num_candidates": block_input.candidate_count,
                "candidate_labels": list(block_input.candidate_labels),
                "rankings": {},
                "block_status": {},
            },
        )
        record_payload["block_status"][block_input.review_block.id] = result["status"]
        if result["status"] == "success":
            record_payload["rankings"][block_input.review_block.id] = result[
                "final_ranking"
            ]
        long_rows.append(_long_row(block_input, result, config.ranking_method))

    for payload in record_results.values():
        _append_jsonl(final_jsonl_path, payload)
    _write_csv(final_csv_path, long_rows)
    print(f"Debate ranking complete: {run_dir}", flush=True)
    return run_dir


def _rank_block(
    *,
    block_input: DebateBlockInput,
    turns: list[TurnConfig],
    agent_by_id: dict[str, DebateAgent],
    agent_config_by_id: dict[str, Any],
    prompt_by_agent_id: dict[str, PromptTemplate],
    config: DebateConfig,
) -> dict[str, Any]:
    previous_agent_trace: list[dict[str, Any]] = []
    agent_rankings: list[dict[str, Any]] = []
    aggregation_rankings: list[dict[str, Any]] = []
    trace_ids: list[str] = []
    turns_trace: list[dict[str, Any]] = []

    for turn_index, turn_config in enumerate(turns, start=1):
        agent = agent_by_id[turn_config.agent_id]
        agent_config = agent_config_by_id[turn_config.agent_id]
        prompt = prompt_by_agent_id[turn_config.agent_id]
        messages = [
            {
                "role": "system",
                "content": f"You are {agent.name}. Act as the {agent_config.role}.",
            },
            {
                "role": "user",
                "content": prompt.render(
                    _prompt_variables(
                        block_input,
                        previous_agent_trace=previous_agent_trace,
                        agent=agent,
                        agent_role=agent_config.role,
                        turn=turn_config,
                        turn_index=turn_index,
                    )
                ),
            }
        ]
        turn = _generate_valid_ranking(
            agent=agent,
            messages=messages,
            generation=config.generation,
            record_id=block_input.segment.record_id,
            review_block=block_input.review_block.id,
            candidate_labels=block_input.candidate_labels,
        )
        trace_id = (
            f"{block_input.segment.dataset}:"
            f"{block_input.segment.record_id}:"
            f"{block_input.review_block.id}:"
            f"{turn_config.id}:"
            f"{agent.agent_id}"
        )
        trace_payload = {
            "timestamp_utc": _timestamp(),
            "trace_id": trace_id,
            "dataset": block_input.segment.dataset,
            "record_id": block_input.segment.record_id,
            "review_block": block_input.review_block.id,
            "turn_id": turn_config.id,
            "turn_role": turn_config.role,
            "contributes_to_aggregation": turn_config.contributes_to_aggregation,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "agent_role": agent_config.role,
            "turn_index": turn_index,
            **turn,
        }
        turns_trace.append(trace_payload)
        trace_ids.append(trace_id)
        if turn["parse_status"] != "valid":
            return {
                "status": "failed",
                "dataset": block_input.segment.dataset,
                "record_id": block_input.segment.record_id,
                "review_block": block_input.review_block.id,
                "candidate_count": block_input.candidate_count,
                "candidate_labels": list(block_input.candidate_labels),
                "failed_turn_id": turn_config.id,
                "failed_turn_role": turn_config.role,
                "failed_agent_id": agent.agent_id,
                "failure_reason": "; ".join(turn["validation_errors"]),
                "trace_ids": trace_ids,
                "turns": turns_trace,
                "agent_rankings": agent_rankings,
                "aggregation_rankings": aggregation_rankings,
            }

        ranking = ranking_from_payload(turn["parsed_output"])
        agent_entry = {
            "turn_id": turn_config.id,
            "turn_index": turn_index,
            "turn_role": turn_config.role,
            "contributes_to_aggregation": turn_config.contributes_to_aggregation,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "candidate_assessments": turn["parsed_output"][
                "candidate_assessments"
            ],
            "debate_response": turn["parsed_output"]["debate_response"],
            "uncertainty": turn["parsed_output"]["uncertainty"],
            "ranking": ranking,
            "rationale": turn["parsed_output"]["rationale"],
        }
        agent_rankings.append(agent_entry)
        previous_agent_trace.append(agent_entry)
        if turn_config.contributes_to_aggregation:
            aggregation_rankings.append(agent_entry)

    aggregation = borda_aggregate(
        aggregation_rankings,
        tiebreak_agent_id=aggregation_rankings[-1]["agent_id"],
        candidate_labels=block_input.candidate_labels,
    )
    return {
        "status": "success",
        "dataset": block_input.segment.dataset,
        "record_id": block_input.segment.record_id,
        "review_block": block_input.review_block.id,
        "candidate_count": block_input.candidate_count,
        "candidate_labels": list(block_input.candidate_labels),
        "trace_ids": trace_ids,
        "turns": turns_trace,
        "agent_rankings": agent_rankings,
        "aggregation_rankings": aggregation_rankings,
        "final_ranking": aggregation["ranking"],
        "borda_scores": aggregation["scores"],
        "tiebreak_agent_id": aggregation["tiebreak_agent_id"],
        "tiebreak_applied": aggregation["tiebreak_applied"],
        "tiebreaks": aggregation["tiebreaks"],
    }


def _generate_valid_ranking(
    *,
    agent: DebateAgent,
    messages: list[dict[str, str]],
    generation: Any,
    record_id: str,
    review_block: str,
    candidate_labels: tuple[str, ...] = CANDIDATE_LABELS,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    current_messages = list(messages)
    final_payload: dict[str, Any] | None = None
    final_errors: list[str] = []
    final_text = ""
    final_rendered_prompt = ""
    final_raw: dict[str, Any] = {}
    final_elapsed = 0.0

    for attempt_index in range(1, generation.json_retry_attempts + 2):
        result = agent.generate(current_messages, generation)
        parsed, parse_error = parse_json_object(result.text)
        errors = validate_ranking_payload(
            parsed,
            record_id=record_id,
            review_block=review_block,
            candidate_labels=candidate_labels,
        )
        if parse_error and parsed is None:
            errors = [parse_error, *errors]
        attempts.append(
            {
                "attempt_index": attempt_index,
                "raw_output_text": result.text,
                "parsed_output": parsed,
                "parse_error": parse_error,
                "validation_errors": errors,
                "rendered_prompt": result.rendered_prompt,
                "raw_response": result.raw,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        final_payload = parsed
        final_errors = errors
        final_text = result.text
        final_rendered_prompt = result.rendered_prompt
        final_raw = result.raw
        final_elapsed = result.elapsed_seconds
        if not errors:
            break
        current_messages = [
            *messages,
            {"role": "assistant", "content": result.text},
            {
                "role": "user",
                "content": (
                    "Your previous answer was invalid for this debate ranking task. "
                    "Return only one strict JSON object with record_id, review_block, "
                    "candidate_assessments containing exactly non-empty strings for "
                    f"{list(candidate_labels)}, "
                    "non-empty debate_response, non-empty uncertainty, a complete "
                    f"ranking of exactly {list(candidate_labels)}, and a non-empty "
                    "rationale. "
                    f"Validation errors: {errors}"
                ),
            },
        ]

    return {
        "attempt_count": len(attempts),
        "parse_status": "valid" if not final_errors else "invalid",
        "validation_errors": final_errors,
        "raw_output_text": final_text,
        "parsed_output": final_payload,
        "rendered_prompt": final_rendered_prompt,
        "raw_response": final_raw,
        "elapsed_seconds": final_elapsed,
        "attempts": attempts,
    }


def _prompt_variables(
    block_input: DebateBlockInput,
    *,
    previous_agent_trace: list[dict[str, Any]],
    agent: DebateAgent,
    agent_role: str,
    turn: TurnConfig,
    turn_index: int,
) -> dict[str, Any]:
    required_output_shape = {
        "record_id": block_input.segment.record_id,
        "review_block": block_input.review_block.id,
        "candidate_assessments": {
            label: f"Concise evidence and block-fit assessment for Candidate {label}."
            for label in block_input.candidate_labels
        },
        "debate_response": (
            "Stage-appropriate acceptance, challenge, or revision of prior reasoning."
        ),
        "uncertainty": "Most important ambiguity or evidential limitation.",
        "ranking": list(block_input.candidate_labels),
        "rationale": "Comparative justification for the complete ordering.",
    }
    return {
        "agent_id": agent.agent_id,
        "agent_name": agent.name,
        "agent_role": agent_role,
        "turn_id": turn.id,
        "turn_index": turn_index,
        "turn_role": turn.role,
        "contributes_to_aggregation": turn.contributes_to_aggregation,
        "dataset": block_input.segment.dataset,
        "record_id": block_input.segment.record_id,
        "transcript_id": block_input.segment.transcript_id,
        "segment_id": block_input.segment.segment_id,
        "review_block": block_input.review_block.id,
        "review_block_title": block_input.review_block.title,
        "candidate_count": block_input.candidate_count,
        "candidate_labels": ", ".join(block_input.candidate_labels),
        "candidate_labels_json": json.dumps(list(block_input.candidate_labels)),
        "participant_segment_text": block_input.participant_segment_text,
        "previous_context": block_input.previous_context,
        "next_context": block_input.next_context,
        "research_questions": _format_research_questions(
            block_input.research_questions
        ),
        "research_questions_json": json.dumps(
            list(block_input.research_questions), ensure_ascii=False, indent=2
        ),
        "candidate_table_json": json.dumps(
            block_input.candidate_table, ensure_ascii=False, indent=2
        ),
        "candidate_mapping_json": json.dumps(
            block_input.candidate_mapping, ensure_ascii=False, indent=2
        ),
        "previous_agent_trace_json": json.dumps(
            previous_agent_trace, ensure_ascii=False, indent=2
        ),
        "required_output_shape_json": json.dumps(
            required_output_shape, ensure_ascii=False, indent=2
        ),
    }


def _long_row(
    block_input: DebateBlockInput,
    result: dict[str, Any],
    ranking_method: str,
) -> dict[str, Any]:
    final = result.get("final_ranking") or []
    scores = result.get("borda_scores") or {}
    agent_rankings = {
        item["turn_id"]: {
            "agent_id": item["agent_id"],
            "candidate_assessments": item["candidate_assessments"],
            "debate_response": item["debate_response"],
            "uncertainty": item["uncertainty"],
            "ranking": item["ranking"],
            "rationale": item["rationale"],
            "contributes_to_aggregation": item["contributes_to_aggregation"],
        }
        for item in result.get("agent_rankings", [])
    }
    aggregation_rankings = {
        item["turn_id"]: {
            "agent_id": item["agent_id"],
            "ranking": item["ranking"],
        }
        for item in result.get("aggregation_rankings", [])
    }
    row = {
        "dataset": block_input.segment.dataset,
        "transcript_id": block_input.segment.transcript_id,
        "segment_id": block_input.segment.segment_id,
        "record_id": block_input.segment.record_id,
        "review_block": block_input.review_block.id,
        "candidate_count": block_input.candidate_count,
        "candidate_labels_json": json.dumps(
            list(block_input.candidate_labels), ensure_ascii=False
        ),
        "ranking_method": ranking_method,
        "status": result["status"],
        "failure_reason": result.get("failure_reason", ""),
        "segment_trace_file": result.get("segment_trace_file", ""),
        "trace_ids": json.dumps(result.get("trace_ids", []), ensure_ascii=False),
        "tiebreak_agent_id": result.get("tiebreak_agent_id", ""),
        "tiebreak_applied": result.get("tiebreak_applied", False),
        "tiebreaks_json": json.dumps(result.get("tiebreaks", []), ensure_ascii=False),
        "borda_scores_json": json.dumps(scores, ensure_ascii=False),
        "agent_rankings_json": json.dumps(agent_rankings, ensure_ascii=False),
        "terminal_agent_rankings_json": json.dumps(
            aggregation_rankings, ensure_ascii=False
        ),
    }
    for index in range(5):
        row[f"final_rank_{index + 1}"] = final[index] if index < len(final) else ""
    for label in CANDIDATE_LABELS:
        row[f"score_{label}"] = scores.get(label, "")
        mapping = next(
            (
                item
                for item in block_input.candidate_mapping
                if item["candidate_label"] == label
            ),
            None,
        )
        row[f"candidate_{label}_sample_index"] = (
            mapping["original_sample_index"] if mapping else ""
        )
    return row


def _new_segment_trace_payload(
    block_input: DebateBlockInput,
    ranking_method: str,
) -> dict[str, Any]:
    return {
        "created_at_utc": _timestamp(),
        "updated_at_utc": _timestamp(),
        "dataset": block_input.segment.dataset,
        "record_id": block_input.segment.record_id,
        "transcript_id": block_input.segment.transcript_id,
        "segment_id": block_input.segment.segment_id,
        "ranking_method": ranking_method,
        "num_candidates": block_input.candidate_count,
        "candidate_labels": list(block_input.candidate_labels),
        "participant_segment_text": block_input.participant_segment_text,
        "previous_context": block_input.previous_context,
        "next_context": block_input.next_context,
        "research_questions": list(block_input.research_questions),
        "segment_json_path": str(block_input.segment_json_path),
        "review_blocks": {},
    }


def _block_trace_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "status": result["status"],
        "review_block": result.get("review_block", ""),
        "candidate_count": result.get("candidate_count", 0),
        "candidate_labels": result.get("candidate_labels", []),
        "trace_ids": result.get("trace_ids", []),
        "turns": result.get("turns", []),
        "agent_rankings": result.get("agent_rankings", []),
        "aggregation_rankings": result.get("aggregation_rankings", []),
        "final_ranking": result.get("final_ranking", []),
        "borda_scores": result.get("borda_scores", {}),
        "tiebreak_agent_id": result.get("tiebreak_agent_id", ""),
        "tiebreak_applied": result.get("tiebreak_applied", False),
        "tiebreaks": result.get("tiebreaks", []),
    }
    if result["status"] != "success":
        payload.update(
            {
                "failed_turn_id": result.get("failed_turn_id", ""),
                "failed_turn_role": result.get("failed_turn_role", ""),
                "failed_agent_id": result.get("failed_agent_id", ""),
                "failure_reason": result.get("failure_reason", ""),
            }
        )
    return payload


def _segment_trace_file(trace_dir: Path, block_input: DebateBlockInput) -> Path:
    dataset_dir = trace_dir / block_input.segment.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir / f"{block_input.segment.record_id}.json"


def _new_run_dir(output_root: Path, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return output_root / f"{run_name}_{timestamp}"


def _format_research_questions(research_questions: tuple[str, ...]) -> str:
    if not research_questions:
        return "No research questions supplied."
    return "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(research_questions, start=1)
    )


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
