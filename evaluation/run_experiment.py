"""Run the ITS Task A/Task B generation and LLM-as-judge experiment.

The generator receives only ITS_MODEL_INPUT_*.jsonl. The judge receives the
generated output, the matching hidden reference, the rubric, and source
evidence from ITS_FINAL_*.jsonl.

Examples (from the repository root):

    python evaluation/run_experiment.py validate
    python evaluation/run_experiment.py run --task all --run-dir evaluation/runs/pilot_01 --limit 5
    python evaluation/run_experiment.py summarize --run-dir evaluation/runs/pilot_01

Set the API key required by each selected provider before a live run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable

from provider_api import create_client, provider_env_var, request_json_once, response_usage


HERE = Path(__file__).resolve().parent
TASK_CONFIG = {
    "A": {
        "model_input": HERE / "ITS_MODEL_INPUT_A_300.jsonl",
        "full_dataset": HERE / "ITS_FINAL_A_300.jsonl",
        "output_schema": HERE / "OUTPUT_SCHEMA_A.json",
        "generation_file": "generations_A.jsonl",
        "judgment_file": "judgments_A.jsonl",
    },
    "B": {
        "model_input": HERE / "ITS_MODEL_INPUT_B_300.jsonl",
        "full_dataset": HERE / "ITS_FINAL_B_300.jsonl",
        "output_schema": HERE / "OUTPUT_SCHEMA_B.json",
        "generation_file": "generations_B.jsonl",
        "judgment_file": "judgments_B.jsonl",
    },
}

# GPT-4.1 Nano was shut down on 2026-10-23. Keep the replacement explicit and
# pinned so a paper can report the exact model snapshot.
DEFAULT_GENERATOR = "gpt-5.4-nano-2026-03-17"
DEFAULT_JUDGE = "gpt-5.4-mini-2026-03-17"
EXPERIMENT_VERSION = "its-llm-judge-v1"
PROMPT_VERSION = "2026-10-27-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            rows.append(value)
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_tasks(task: str) -> list[str]:
    return ["A", "B"] if task == "all" else [task]


def latest_by_case(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["case_id"]: row for row in read_jsonl(path) if row.get("case_id")}


def parse_response_object(response: Any) -> dict[str, Any]:
    text = (getattr(response, "output_text", "") or "").strip()
    if not text:
        raise ValueError("The API response contained no output_text")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("The model output was not a JSON object")
    return value


def validate_qcm_items(items: Any, *, allowed_objective_ids: set[str] | None) -> list[str]:
    errors: list[str] = []
    if not isinstance(items, list) or not items:
        return ["QCM must be a non-empty array"]
    required = {"question", "options", "correct_option", "explanation", "target_objective_ids"}
    for index, item in enumerate(items):
        prefix = f"QCM[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if set(item) != required:
            errors.append(f"{prefix} fields must be exactly {sorted(required)}")
        for key, minimum in (("question", 8), ("explanation", 8)):
            if not isinstance(item.get(key), str) or len(item[key].strip()) < minimum:
                errors.append(f"{prefix}.{key} is too short or missing")
        options = item.get("options")
        if not isinstance(options, dict) or not (2 <= len(options) <= 6):
            errors.append(f"{prefix}.options must contain 2 to 6 choices")
        else:
            if any(key not in set("ABCDEF") for key in options):
                errors.append(f"{prefix}.options keys must be A-F")
            if any(not isinstance(text, str) or not text.strip() for text in options.values()):
                errors.append(f"{prefix}.options values must be non-empty strings")
            if item.get("correct_option") not in options:
                errors.append(f"{prefix}.correct_option must occur in options")
        objective_ids = item.get("target_objective_ids")
        if not isinstance(objective_ids, list) or any(not isinstance(x, str) for x in objective_ids):
            errors.append(f"{prefix}.target_objective_ids must be a string array")
        elif len(objective_ids) != len(set(objective_ids)):
            errors.append(f"{prefix}.target_objective_ids must be unique")
        elif allowed_objective_ids is not None:
            unknown = sorted(set(objective_ids) - allowed_objective_ids)
            if unknown:
                errors.append(f"{prefix}.target_objective_ids contains unknown IDs: {unknown}")
    return errors


def validate_generated_output(task: str, output: Any, model_input: dict[str, Any]) -> list[str]:
    if not isinstance(output, dict):
        return ["Output must be a JSON object"]
    if task == "A":
        required = {"lesson", "qcm"}
        errors = [] if set(output) == required else [f"Top-level fields must be exactly {sorted(required)}"]
        if not isinstance(output.get("lesson"), str) or len(output["lesson"].strip()) < 30:
            errors.append("lesson is too short or missing")
        allowed = {
            objective["id"]
            for objective in model_input["input"].get("requested_learning_objectives", [])
            if isinstance(objective, dict) and isinstance(objective.get("id"), str)
        }
        errors.extend(validate_qcm_items(output.get("qcm"), allowed_objective_ids=allowed))
        return errors
    required = {"feedback", "error_interpretation", "remedial_qcm"}
    errors = [] if set(output) == required else [f"Top-level fields must be exactly {sorted(required)}"]
    for key, minimum in (("feedback", 20), ("error_interpretation", 10)):
        if not isinstance(output.get(key), str) or len(output[key].strip()) < minimum:
            errors.append(f"{key} is too short or missing")
    errors.extend(validate_qcm_items(output.get("remedial_qcm"), allowed_objective_ids=None))
    return errors


def generation_prompts(task: str, row: dict[str, Any], schema: dict[str, Any]) -> tuple[str, str]:
    common = (
        "Return one JSON object only. Follow the supplied output schema exactly. "
        "Use only the supplied source excerpt as factual evidence. Do not invent doctrine facts, "
        "do not refer to hidden references, and write all learner-facing content in French. "
        "Every QCM must have one unambiguously correct answer, and correct_option must be one of "
        "the option keys. Use target_objective_ids only when those IDs appear in the input."
    )
    if task == "A":
        system = (
            "You generate concise, accurate micro-lessons and assessment questions for firefighter "
            "graphical-doctrine instruction. " + common
        )
    else:
        system = (
            "You generate corrective feedback and new remedial assessment questions after an observed "
            "learner error. Treat any misconception as a cautious hypothesis, adapt to the observation "
            "history, never make a clinical or stable cognitive diagnosis, and do not simply repeat the "
            "failed question. " + common
        )
    prompt = (
        "Generate the requested output from this model-visible input.\n\n"
        + json.dumps(row, ensure_ascii=False, indent=2)
        + "\n\nRequired JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )
    return system, prompt


JUDGE_SCHEMA_A: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "objective_assessments", "factual_correctness", "pdf_grounding",
        "answer_validity", "overall_confidence",
    ],
    "properties": {
        "objective_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "objective_id", "lesson_covered", "qcm_assessed",
                    "lesson_evidence", "qcm_evidence",
                ],
                "properties": {
                    "objective_id": {"type": "string"},
                    "lesson_covered": {"type": "boolean"},
                    "qcm_assessed": {"type": "boolean"},
                    "lesson_evidence": {"type": "string"},
                    "qcm_evidence": {"type": "string"},
                },
            },
        },
        "factual_correctness": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "errors", "rationale"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "errors": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
        },
        "pdf_grounding": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "unsupported_claims", "rationale"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
        },
        "answer_validity": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "invalid_questions", "rationale"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "invalid_questions": {"type": "array", "items": {"type": "integer"}},
                "rationale": {"type": "string"},
            },
        },
        "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


JUDGE_SCHEMA_B: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "error_diagnosis", "feedback_correctness", "remediation_targeting",
        "contextual_adaptation", "uncertainty_calibration", "grounding",
        "overall_confidence",
    ],
    "properties": {
        "error_diagnosis": {
            "type": "object", "additionalProperties": False,
            "required": ["evaluable", "label", "score", "rationale"],
            "properties": {
                "evaluable": {"type": "boolean"},
                "label": {"type": "string", "enum": ["correct", "partial", "incorrect", "not_evaluable"]},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
        },
        "feedback_correctness": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "errors", "rationale"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "errors": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
        },
        "remediation_targeting": {
            "type": "object", "additionalProperties": False,
            "required": ["targeted", "score", "rationale"],
            "properties": {
                "targeted": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
        },
        "contextual_adaptation": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "matched_requirements", "missed_requirements", "rationale"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "matched_requirements": {"type": "array", "items": {"type": "string"}},
                "missed_requirements": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
        },
        "uncertainty_calibration": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "rationale"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
        },
        "grounding": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "unsupported_claims", "rationale"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
        },
        "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


def judge_prompts(task: str, full_case: dict[str, Any], generation: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are an independent evaluator of French firefighter-doctrine educational content. "
        "Evaluate only against the supplied source evidence and hidden reference requirements. "
        "Do not reward fluency, verbosity, or stylistic similarity. Require semantic evidence. "
        "A cautious misconception hypothesis is appropriate when the observation is insufficient; "
        "do not require an unjustified cognitive diagnosis. Use scores from 0 to 1. "
        "Return one object matching the response schema."
    )
    if task == "A":
        metric_rules = {
            "LOC": "Mark lesson_covered only when the lesson substantively teaches the objective.",
            "QOC": "Mark qcm_assessed only when at least one question actually tests the objective.",
            "A_FC": "Score factual accuracy of both lesson and QCM, including stated correct answers.",
            "PG": "Score whether essential claims are supported by the supplied PDF/source evidence.",
            "answer_validity": "Penalize ambiguous questions, multiple correct answers, or wrong answer keys.",
        }
    else:
        metric_rules = {
            "EDA": "Judge identification of the observed error; uncertainty-aware wording can be correct.",
            "B_FC": "Judge whether feedback accurately corrects the learner using source evidence.",
            "RT": "targeted=true only when the remedial QCM tests the required failed distinction/objective.",
            "CA": "Score compliance with the profile-specific feedback and remedial requirements.",
            "uncertainty": "Penalize claims of confirmed stable misconception unsupported by the observation.",
        }
    payload = {
        "task": full_case["task"],
        "case_id": full_case["case_id"],
        "model_visible_input": full_case["input"],
        "generated_output": generation.get("output"),
        "generation_format_validation": generation.get("validation"),
        "hidden_gold_reference": full_case["reference_hidden_from_generator"],
        "evaluation_rubric": full_case["evaluation_rubric"],
        "metric_rules": metric_rules,
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def judge_validation_errors(task: str, judgment: dict[str, Any], full_case: dict[str, Any]) -> list[str]:
    if task != "A":
        return []
    expected = full_case["reference_hidden_from_generator"].get("requested_objective_ids", [])
    received = [item.get("objective_id") for item in judgment.get("objective_assessments", [])]
    if len(received) != len(set(received)):
        return ["Judge returned duplicate objective IDs"]
    if set(received) != set(expected):
        return [f"Judge objective IDs differ: expected {expected}, received {received}"]
    return []


async def api_call_with_retries(call_factory: Any, max_attempts: int) -> Any:
    error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await call_factory()
        except Exception as exc:  # API exception classes can change between SDK releases.
            error = exc
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 20) + random.random())
    assert error is not None
    raise error


def create_or_check_manifest(args: argparse.Namespace, tasks: list[str]) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("tasks") != tasks:
            raise ValueError(
                f"Run manifest contains tasks={manifest.get('tasks')!r}, but CLI requested {tasks!r}. "
                "Use the original task selection or a new run directory."
            )
        # Manifests created before provider support were OpenAI-only. Treat a
        # missing provider as OpenAI so they can never be resumed via OpenRouter.
        existing_generator_provider = manifest.get("generator_provider", "openai")
        existing_judge_provider = manifest.get("judge_provider", "openai")
        if existing_generator_provider != args.generator_provider:
            raise ValueError(
                f"Run manifest has generator_provider={existing_generator_provider!r}, but CLI requested "
                f"{args.generator_provider!r}. Use a new run directory."
            )
        if existing_judge_provider != args.judge_provider:
            raise ValueError(
                f"Run manifest has judge_provider={existing_judge_provider!r}, but CLI requested "
                f"{args.judge_provider!r}. Use a new run directory."
            )
        expected = {
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "generator_reasoning": args.generator_reasoning,
            "judge_reasoning": args.judge_reasoning,
            "generator_verbosity": args.generator_verbosity,
            "openrouter_output_mode": args.openrouter_output_mode,
        }
        for key, value in expected.items():
            if key in manifest and manifest.get(key) != value:
                raise ValueError(
                    f"Run manifest has {key}={manifest.get(key)!r}, but CLI requested {value!r}. "
                    "Use the original setting or a new run directory."
                )
        # Backfill fields introduced after older run manifests were created.
        changed = False
        for key, value in (
            ("generator_provider", existing_generator_provider),
            ("judge_provider", existing_judge_provider),
            ("generator_verbosity", args.generator_verbosity),
            ("openrouter_output_mode", args.openrouter_output_mode),
        ):
            if key not in manifest:
                manifest[key] = value
                changed = True
        if changed:
            write_json(manifest_path, manifest)
        return manifest

    run_dir.mkdir(parents=True, exist_ok=True)
    selected_case_ids: dict[str, list[str]] = {}
    rng = random.Random(args.seed)
    for task in tasks:
        ids = [row["case_id"] for row in read_jsonl(TASK_CONFIG[task]["model_input"])]
        if args.limit is not None and args.limit < len(ids):
            ids = sorted(rng.sample(ids, args.limit))
        selected_case_ids[task] = ids
    inputs = [
        TASK_CONFIG[task][name]
        for task in tasks
        for name in ("model_input", "full_dataset", "output_schema")
    ]
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "created_at": utc_now(),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "generator_provider": args.generator_provider,
        "judge_provider": args.judge_provider,
        "openrouter_output_mode": args.openrouter_output_mode,
        "generator_reasoning": args.generator_reasoning,
        "judge_reasoning": args.judge_reasoning,
        "generator_verbosity": args.generator_verbosity,
        "seed": args.seed,
        "tasks": tasks,
        "selected_case_ids": selected_case_ids,
        "input_sha256": {str(path.relative_to(HERE)): sha256(path) for path in inputs},
        "gold_exposed_to_generator": False,
        "gold_exposed_to_judge": True,
        "generator_format": "JSON mode followed by local structural and semantic validation",
        "judge_format": "strict JSON-schema structured output",
    }
    write_json(manifest_path, manifest)
    return manifest


async def generate_task(args: argparse.Namespace, task: str, manifest: dict[str, Any]) -> None:
    config = TASK_CONFIG[task]
    rows = {row["case_id"]: row for row in read_jsonl(config["model_input"])}
    selected = [rows[case_id] for case_id in manifest["selected_case_ids"][task]]
    destination = args.run_dir.resolve() / config["generation_file"]
    existing = latest_by_case(destination)
    pending = [row for row in selected if existing.get(row["case_id"], {}).get("status") != "ok"]
    print(f"Task {task}: {len(selected)} selected, {len(pending)} generation calls pending")
    if not pending:
        return
    schema = read_json(config["output_schema"])
    client = create_client(
        args.generator_provider,
        timeout=args.timeout,
        openrouter_site_url=args.openrouter_site_url,
        openrouter_app_name=args.openrouter_app_name,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            started = utc_now()
            try:
                system, prompt = generation_prompts(task, row, schema)

                async def request_and_parse() -> tuple[Any, dict[str, Any]]:
                    return await request_json_once(
                        client,
                        provider=args.generator_provider,
                        model=args.generator_model,
                        system=system,
                        prompt=prompt,
                        schema=schema,
                        schema_name=f"task_{task.lower()}_generation",
                        max_output_tokens=args.generator_max_output_tokens,
                        openai_output_mode="json_object",
                        openrouter_output_mode=args.openrouter_output_mode,
                        reasoning_effort=args.generator_reasoning,
                        verbosity=args.generator_verbosity,
                    )

                response, output = await api_call_with_retries(request_and_parse, args.max_attempts)
                errors = validate_generated_output(task, output, row)
                return {
                    "case_id": row["case_id"], "task": task, "status": "ok",
                    "provider": args.generator_provider,
                    "model": args.generator_model, "reasoning_effort": args.generator_reasoning,
                    "verbosity": args.generator_verbosity,
                    "started_at": started, "finished_at": utc_now(),
                    "response_id": getattr(response, "id", None), "usage": response_usage(response),
                    "format_valid": not errors, "validation": {"errors": errors}, "output": output,
                }
            except Exception as exc:
                return {
                    "case_id": row["case_id"], "task": task, "status": "error",
                    "provider": args.generator_provider,
                    "model": args.generator_model, "reasoning_effort": args.generator_reasoning,
                    "verbosity": args.generator_verbosity,
                    "started_at": started, "finished_at": utc_now(),
                    "error_type": type(exc).__name__, "error": str(exc)[:2000],
                    "format_valid": False, "validation": {"errors": ["generation_failed"]},
                    "output": None,
                }

    for future in asyncio.as_completed([one(row) for row in pending]):
        record = await future
        append_jsonl(destination, record)
        print(f"  {record['case_id']}: {record['status']}", flush=True)
    await client.close()


def failure_judgment(task: str, reason: str) -> dict[str, Any]:
    if task == "A":
        return {
            "objective_assessments": [],
            "factual_correctness": {"score": 0.0, "errors": [reason], "rationale": reason},
            "pdf_grounding": {"score": 0.0, "unsupported_claims": [], "rationale": reason},
            "answer_validity": {"score": 0.0, "invalid_questions": [], "rationale": reason},
            "overall_confidence": 1.0,
        }
    return {
        "error_diagnosis": {"evaluable": True, "label": "incorrect", "score": 0.0, "rationale": reason},
        "feedback_correctness": {"score": 0.0, "errors": [reason], "rationale": reason},
        "remediation_targeting": {"targeted": False, "score": 0.0, "rationale": reason},
        "contextual_adaptation": {
            "score": 0.0, "matched_requirements": [], "missed_requirements": [reason], "rationale": reason,
        },
        "uncertainty_calibration": {"score": 0.0, "rationale": reason},
        "grounding": {"score": 0.0, "unsupported_claims": [], "rationale": reason},
        "overall_confidence": 1.0,
    }


async def judge_task(args: argparse.Namespace, task: str, manifest: dict[str, Any]) -> None:
    config = TASK_CONFIG[task]
    full_cases = {row["case_id"]: row for row in read_jsonl(config["full_dataset"])}
    generations = latest_by_case(args.run_dir.resolve() / config["generation_file"])
    destination = args.run_dir.resolve() / config["judgment_file"]
    existing = latest_by_case(destination)
    selected_ids = manifest["selected_case_ids"][task]
    missing_generations = [case_id for case_id in selected_ids if case_id not in generations]
    if missing_generations:
        preview = ", ".join(missing_generations[:5])
        raise ValueError(
            f"Task {task} has {len(missing_generations)} cases that were never attempted by the generator "
            f"(for example: {preview}). Resume the generate stage before judging."
        )
    def needs_judgment(case_id: str) -> bool:
        previous = existing.get(case_id, {})
        if previous.get("status") != "ok":
            return True
        # A generation failure receives a deterministic zero so it is not silently
        # excluded. If generation later succeeds on resume, replace that temporary
        # zero with a real judge call.
        return (
            previous.get("judge_model") == "deterministic_failure_rule"
            and generations.get(case_id, {}).get("status") == "ok"
        )

    pending_ids = [case_id for case_id in selected_ids if needs_judgment(case_id)]
    print(f"Task {task}: {len(selected_ids)} selected, {len(pending_ids)} judge records pending")
    if not pending_ids:
        return
    client = create_client(
        args.judge_provider,
        timeout=args.timeout,
        openrouter_site_url=args.openrouter_site_url,
        openrouter_app_name=args.openrouter_app_name,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    response_schema = JUDGE_SCHEMA_A if task == "A" else JUDGE_SCHEMA_B

    async def one(case_id: str) -> dict[str, Any]:
        generation = generations.get(case_id)
        if generation is None or generation.get("status") != "ok":
            reason = "No usable generated output was produced."
            return {
                "case_id": case_id, "task": task, "status": "ok", "judge_model": "deterministic_failure_rule",
                "judge_provider": "deterministic_failure_rule",
                "started_at": utc_now(), "finished_at": utc_now(), "response_id": None,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "judgment": failure_judgment(task, reason), "judge_validation": {"errors": []},
            }
        async with semaphore:
            started = utc_now()
            try:
                system, prompt = judge_prompts(task, full_cases[case_id], generation)

                async def request_parse_and_validate() -> tuple[Any, dict[str, Any], list[str]]:
                    response, judgment = await request_json_once(
                        client,
                        provider=args.judge_provider,
                        model=args.judge_model,
                        system=system,
                        prompt=prompt,
                        schema=response_schema,
                        schema_name=f"task_{task.lower()}_judgment",
                        max_output_tokens=args.judge_max_output_tokens,
                        openai_output_mode="json_schema",
                        openrouter_output_mode=args.openrouter_output_mode,
                        reasoning_effort=args.judge_reasoning,
                        verbosity="medium",
                    )
                    errors = judge_validation_errors(task, judgment, full_cases[case_id])
                    if errors:
                        raise ValueError("; ".join(errors))
                    return response, judgment, errors

                response, judgment, errors = await api_call_with_retries(
                    request_parse_and_validate, args.max_attempts
                )
                return {
                    "case_id": case_id, "task": task, "status": "ok", "judge_model": args.judge_model,
                    "judge_provider": args.judge_provider,
                    "reasoning_effort": args.judge_reasoning, "started_at": started, "finished_at": utc_now(),
                    "response_id": getattr(response, "id", None), "usage": response_usage(response),
                    "judgment": judgment, "judge_validation": {"errors": errors},
                }
            except Exception as exc:
                return {
                    "case_id": case_id, "task": task, "status": "error", "judge_model": args.judge_model,
                    "judge_provider": args.judge_provider,
                    "reasoning_effort": args.judge_reasoning, "started_at": started, "finished_at": utc_now(),
                    "error_type": type(exc).__name__, "error": str(exc)[:2000],
                }

    for future in asyncio.as_completed([one(case_id) for case_id in pending_ids]):
        record = await future
        append_jsonl(destination, record)
        print(f"  {record['case_id']}: {record['status']}", flush=True)
    await client.close()


def mean(values: Iterable[float]) -> float | None:
    clean = list(values)
    return statistics.fmean(clean) if clean else None


def bootstrap_ci(values: list[float], seed: int, repetitions: int = 5000) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repetitions)
    )
    lower = estimates[math.floor(0.025 * (repetitions - 1))]
    upper = estimates[math.ceil(0.975 * (repetitions - 1))]
    return [lower, upper]


def metric_summary(values: list[float], seed: int) -> dict[str, Any]:
    return {"mean": mean(values), "ci95_case_bootstrap": bootstrap_ci(values, seed), "n": len(values)}


def summarize_task(run_dir: Path, task: str, manifest: dict[str, Any]) -> dict[str, Any]:
    config = TASK_CONFIG[task]
    selected_ids = manifest["selected_case_ids"][task]
    generations = latest_by_case(run_dir / config["generation_file"])
    judgments = latest_by_case(run_dir / config["judgment_file"])
    successful_generations = sum(
        generations.get(case_id, {}).get("status") == "ok" for case_id in selected_ids
    )
    format_values = [
        1.0 if generations.get(case_id, {}).get("format_valid") else 0.0 for case_id in selected_ids
    ]
    usable = [judgments[case_id] for case_id in selected_ids if judgments.get(case_id, {}).get("status") == "ok"]
    result: dict[str, Any] = {
        "selected_cases": len(selected_ids),
        "generated_records": sum(case_id in generations for case_id in selected_ids),
        "successful_generations": successful_generations,
        "generation_error_records": len(selected_ids) - successful_generations,
        "judged_records": len(usable),
        "complete": (
            successful_generations == len(selected_ids)
            and len(usable) == len(selected_ids)
        ),
        "judge_missing_or_error_records": len(selected_ids) - len(usable),
        "format_compliance": metric_summary(format_values, manifest["seed"] + 1),
    }
    if task == "A":
        loc: list[float] = []
        qoc: list[float] = []
        fc: list[float] = []
        pg: list[float] = []
        answer_validity: list[float] = []
        for record in usable:
            judgment = record["judgment"]
            objectives = judgment.get("objective_assessments", [])
            # Generation failures have no objective rows and receive zero.
            loc.append(mean([float(x["lesson_covered"]) for x in objectives]) or 0.0)
            qoc.append(mean([float(x["qcm_assessed"]) for x in objectives]) or 0.0)
            fc.append(float(judgment["factual_correctness"]["score"]))
            pg.append(float(judgment["pdf_grounding"]["score"]))
            answer_validity.append(float(judgment["answer_validity"]["score"]))
        result["metrics"] = {
            "LOC": metric_summary(loc, manifest["seed"] + 10),
            "QOC": metric_summary(qoc, manifest["seed"] + 11),
            "A_FC": metric_summary(fc, manifest["seed"] + 12),
            "PG": metric_summary(pg, manifest["seed"] + 13),
            "answer_validity": metric_summary(answer_validity, manifest["seed"] + 14),
        }
    else:
        eda: list[float] = []
        partial_count = 0
        feedback: list[float] = []
        rt: list[float] = []
        ca: list[float] = []
        uncertainty: list[float] = []
        grounding: list[float] = []
        for record in usable:
            judgment = record["judgment"]
            diagnosis = judgment["error_diagnosis"]
            if diagnosis["evaluable"]:
                eda.append(1.0 if diagnosis["label"] == "correct" else 0.0)
                partial_count += int(diagnosis["label"] == "partial")
            feedback.append(float(judgment["feedback_correctness"]["score"]))
            rt.append(float(judgment["remediation_targeting"]["targeted"]))
            ca.append(float(judgment["contextual_adaptation"]["score"]))
            uncertainty.append(float(judgment["uncertainty_calibration"]["score"]))
            grounding.append(float(judgment["grounding"]["score"]))
        result["metrics"] = {
            "EDA": metric_summary(eda, manifest["seed"] + 20),
            "EDA_partial_count": partial_count,
            "B_FC": metric_summary(feedback, manifest["seed"] + 21),
            "RT": metric_summary(rt, manifest["seed"] + 22),
            "CA": metric_summary(ca, manifest["seed"] + 23),
            "uncertainty_calibration": metric_summary(uncertainty, manifest["seed"] + 24),
            "grounding": metric_summary(grounding, manifest["seed"] + 25),
        }
    return result


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    summary = {
        "experiment_version": manifest["experiment_version"],
        "created_at": utc_now(),
        "generator_model": manifest["generator_model"],
        "judge_model": manifest["judge_model"],
        "generator_provider": manifest.get("generator_provider", "openai"),
        "judge_provider": manifest.get("judge_provider", "openai"),
        "tasks": {
            task: summarize_task(run_dir, task, manifest)
            for task in manifest["tasks"]
        },
        "important_limitation": "Gold annotations are synthetic drafts pending domain-expert validation.",
    }
    write_json(run_dir / "summary.json", summary)
    rows: list[dict[str, Any]] = []
    for task, task_result in summary["tasks"].items():
        for metric, value in task_result["metrics"].items():
            if not isinstance(value, dict):
                continue
            rows.append({
                "task": task, "metric": metric, "mean": value["mean"], "n": value["n"],
                "ci95_low": value["ci95_case_bootstrap"][0] if value["ci95_case_bootstrap"] else None,
                "ci95_high": value["ci95_case_bootstrap"][1] if value["ci95_case_bootstrap"] else None,
            })
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "metric", "mean", "n", "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def export_human_sample(run_dir: Path, per_task: int, seed: int) -> Path:
    manifest = read_json(run_dir / "manifest.json")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        config = TASK_CONFIG[task]
        full = {row["case_id"]: row for row in read_jsonl(config["full_dataset"])}
        generated = latest_by_case(run_dir / config["generation_file"])
        judged = latest_by_case(run_dir / config["judgment_file"])
        candidates = [
            case_id for case_id in manifest["selected_case_ids"][task]
            if case_id in generated and case_id in judged
        ]
        # Round-robin KCs after seeded shuffling gives a more representative sample than a simple prefix.
        rng.shuffle(candidates)
        candidates.sort(key=lambda case_id: full[case_id]["input"].get("kc_id", ""))
        if len(candidates) > per_task:
            candidates = rng.sample(candidates, per_task)
        for case_id in candidates:
            rows.append({
                "task": task,
                "case_id": case_id,
                "kc_id": full[case_id]["input"].get("kc_id"),
                "input_json": json.dumps(full[case_id]["input"], ensure_ascii=False),
                "gold_json": json.dumps(full[case_id]["reference_hidden_from_generator"], ensure_ascii=False),
                "generated_output_json": json.dumps(generated[case_id].get("output"), ensure_ascii=False),
                "expert_1_json": "",
                "expert_2_json": "",
                "adjudication_notes": "",
            })
    destination = run_dir / "human_validation_sample.csv"
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["task", "case_id"])
        writer.writeheader()
        writer.writerows(rows)
    return destination


def validate_datasets() -> dict[str, Any]:
    report: dict[str, Any] = {"valid": True, "tasks": {}}
    for task, config in TASK_CONFIG.items():
        model_rows = read_jsonl(config["model_input"])
        full_rows = read_jsonl(config["full_dataset"])
        schema = read_json(config["output_schema"])
        model_map = {row["case_id"]: row for row in model_rows}
        full_map = {row["case_id"]: row for row in full_rows}
        errors: list[str] = []
        if len(model_map) != len(model_rows):
            errors.append("Duplicate model-input case IDs")
        if len(full_map) != len(full_rows):
            errors.append("Duplicate full-dataset case IDs")
        if set(model_map) != set(full_map):
            errors.append("Model-input and full-dataset case IDs differ")
        for case_id in set(model_map) & set(full_map):
            if model_map[case_id] != {key: full_map[case_id][key] for key in ("case_id", "task", "input")}:
                errors.append(f"Input projection differs for {case_id}")
                break
            if full_map[case_id].get("model_output") is not None:
                errors.append(f"model_output is not null for {case_id}")
                break
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            errors.append("Output schema root must be a closed object")
        report["tasks"][task] = {
            "model_input_rows": len(model_rows), "full_rows": len(full_rows), "errors": errors,
        }
        report["valid"] = report["valid"] and not errors
    return report


def add_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=["A", "B", "all"], default="all")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--generator-provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--judge-provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument(
        "--openrouter-output-mode",
        choices=["json_schema", "json_object", "prompt_only"],
        default="json_schema",
        help="Structured-output strategy for OpenRouter models",
    )
    parser.add_argument("--openrouter-site-url", default=os.getenv("OPENROUTER_SITE_URL"))
    parser.add_argument("--openrouter-app-name", default=os.getenv("OPENROUTER_APP_NAME", "ITS benchmark"))
    parser.add_argument("--generator-reasoning", choices=["omit", "none", "low", "medium", "high", "xhigh"], default="none")
    parser.add_argument("--judge-reasoning", choices=["omit", "none", "low", "medium", "high", "xhigh"], default="high")
    parser.add_argument("--generator-verbosity", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--limit", type=int, default=None, help="Cases per selected task; omit for all 300")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--generator-max-output-tokens", type=int, default=6000)
    parser.add_argument("--judge-max-output-tokens", type=int, default=6000)


def require_live_environment(args: argparse.Namespace) -> None:
    providers: set[str] = set()
    if args.command in {"generate", "run"}:
        providers.add(args.generator_provider)
    if args.command in {"judge", "run"}:
        providers.add(args.judge_provider)
    for provider in providers:
        env_var = provider_env_var(provider)
        if not os.getenv(env_var):
            raise RuntimeError(f"{env_var} is not set in the environment")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")


async def execute_live(args: argparse.Namespace) -> None:
    require_live_environment(args)
    tasks = selected_tasks(args.task)
    manifest = create_or_check_manifest(args, tasks)
    if args.command in {"generate", "run"}:
        for task in tasks:
            await generate_task(args, task, manifest)
    if args.command in {"judge", "run"}:
        for task in tasks:
            await judge_task(args, task, manifest)
    if args.command == "run":
        summary = summarize_run(args.run_dir.resolve())
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate dataset/package integrity without API calls")
    for name, help_text in (
        ("generate", "Generate model outputs; safe to resume"),
        ("judge", "Judge saved model outputs against hidden gold; safe to resume"),
        ("run", "Generate, judge, and summarize"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        add_live_arguments(subparser)
    summarize = subparsers.add_parser("summarize", help="Recompute aggregate metrics without API calls")
    summarize.add_argument("--run-dir", type=Path, required=True)
    sample = subparsers.add_parser("human-sample", help="Export a blinded-review worksheet")
    sample.add_argument("--run-dir", type=Path, required=True)
    sample.add_argument("--per-task", type=int, default=30)
    sample.add_argument("--seed", type=int, default=20261027)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "validate":
            report = validate_datasets()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["valid"] else 1
        if args.command == "summarize":
            print(json.dumps(summarize_run(args.run_dir.resolve()), ensure_ascii=False, indent=2))
            return 0
        if args.command == "human-sample":
            path = export_human_sample(args.run_dir.resolve(), args.per_task, args.seed)
            print(path)
            return 0
        asyncio.run(execute_live(args))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
