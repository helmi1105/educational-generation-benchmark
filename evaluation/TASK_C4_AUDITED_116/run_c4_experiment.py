"""Run the Task C4 generator + independent LLM-judge pilot.

The generator sees MODEL_INPUT_C4_116.jsonl only. The judge sees the matching
full record, generated output, draft gold, source texts, and C4 judge protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parent
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from provider_api import create_client, provider_env_var, request_json_once, response_usage


MODEL_INPUT = HERE / "MODEL_INPUT_C4_116.jsonl"
FULL_DATA = HERE / "FULL_WITH_GOLD_C4_116.jsonl"
OUTPUT_SCHEMA = HERE / "OUTPUT_SCHEMA_C4.json"
JUDGE_PROTOCOL = HERE / "JUDGE_PROTOCOL_C4.json"

# Pinned defaults used by the repository's A/B runner.
DEFAULT_GENERATOR = "gpt-5.4-nano-2026-03-17"
DEFAULT_JUDGE = "gpt-5.4-mini-2026-03-17"


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "lesson_relation_quality", "condition_adherence", "lesson_kc_coverage",
        "qcm_kc_coverage", "qcm_multi_kc_reasoning", "kc_attribution",
        "factual_correctness", "answer_validity", "pdf_grounding", "overall_confidence",
    ],
    "properties": {
        "lesson_relation_quality": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "rationale"],
            "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 2}, "rationale": {"type": "string"}},
        },
        "condition_adherence": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "rationale"],
            "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"}},
        },
        "lesson_kc_coverage": {
            "type": "object", "additionalProperties": False,
            "required": ["covered_kc_ids", "rationale"],
            "properties": {"covered_kc_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}, "rationale": {"type": "string"}},
        },
        "qcm_kc_coverage": {
            "type": "object", "additionalProperties": False,
            "required": ["covered_kc_ids", "rationale"],
            "properties": {"covered_kc_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}, "rationale": {"type": "string"}},
        },
        "qcm_multi_kc_reasoning": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "rationale"],
            "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"}},
        },
        "kc_attribution": {
            "type": "object", "additionalProperties": False,
            "required": ["precision", "recall", "f1", "rationale"],
            "properties": {
                "precision": {"type": "number", "minimum": 0, "maximum": 1},
                "recall": {"type": "number", "minimum": 0, "maximum": 1},
                "f1": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
        },
        "factual_correctness": {"$ref": "#/$defs/binary_metric"},
        "answer_validity": {"$ref": "#/$defs/binary_metric"},
        "pdf_grounding": {"$ref": "#/$defs/binary_metric"},
        "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "$defs": {
        "binary_metric": {
            "type": "object", "additionalProperties": False,
            "required": ["score", "rationale"],
            "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"}},
        }
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def latest(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in read_jsonl(path)} if path.exists() else {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def api_compatible_schema(value: Any) -> Any:
    """Remove JSON-Schema assertions unsupported by strict API outputs.

    The complete schema remains the local contract. Uniqueness, lengths, and
    cardinalities are checked locally after generation where applicable.
    """
    unsupported = {"$schema", "minLength", "minItems", "maxItems", "uniqueItems"}
    if isinstance(value, dict):
        return {
            key: api_compatible_schema(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [api_compatible_schema(item) for item in value]
    return value


def generation_errors(output: dict[str, Any], row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(output) != {"lesson", "qcm"} or not isinstance(output.get("lesson"), str):
        errors.append("Invalid top-level output")
    qcm = output.get("qcm")
    if not isinstance(qcm, dict):
        return errors + ["qcm must be an object"]
    required = {"question", "options", "correct_option", "explanation", "kc_ids"}
    if set(qcm) != required:
        errors.append("Invalid qcm fields")
    options = qcm.get("options")
    if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
        errors.append("options must contain exactly A-D")
    elif qcm.get("correct_option") not in options:
        errors.append("correct_option is not an option key")
    allowed = {row["current_kc"]["id"], *(kc["id"] for kc in row["related_kcs"])}
    kc_ids = qcm.get("kc_ids")
    if not isinstance(kc_ids, list) or not kc_ids or len(kc_ids) != len(set(kc_ids)):
        errors.append("kc_ids must be a non-empty unique array")
    elif not set(kc_ids) <= allowed:
        errors.append("kc_ids contains an ID not supplied in the input")
    return errors


async def api_json(
    client: Any,
    *,
    provider: str,
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int,
    openrouter_output_mode: str,
    reasoning_effort: str,
    attempts: int,
    send_openrouter_reasoning: bool = False,
) -> tuple[Any, dict[str, Any]]:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await request_json_once(
                client,
                provider=provider,
                model=model,
                system=system,
                prompt=prompt,
                schema=schema,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
                openai_output_mode="json_schema",
                openrouter_output_mode=openrouter_output_mode,
                reasoning_effort=reasoning_effort,
                verbosity="low",
                send_openrouter_reasoning=send_openrouter_reasoning,
            )
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(2 ** attempt)
    assert last is not None
    raise last


async def run(args: argparse.Namespace) -> None:
    for provider in {args.generator_provider, args.judge_provider}:
        env_var = provider_env_var(provider)
        if not os.getenv(env_var):
            raise RuntimeError(f"{env_var} is not set")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1, or omitted for all 116 cases")
    inputs = read_jsonl(MODEL_INPUT)
    full = {row["case_id"]: row for row in read_jsonl(FULL_DATA)}
    selected = inputs[: args.limit] if args.limit is not None else inputs
    schema = read_json(OUTPUT_SCHEMA)
    generation_api_schema = api_compatible_schema(schema)
    judge_api_schema = api_compatible_schema(JUDGE_SCHEMA)
    protocol = read_json(JUDGE_PROTOCOL)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "manifest.json"
    previous_manifest = read_json(manifest_path) if manifest_path.exists() else None
    if previous_manifest:
        if previous_manifest.get("generator_reasoning", "omit") != args.generator_reasoning:
            raise ValueError("Run directory uses different generator reasoning. Use a new --run-dir.")
        old_generator_provider = previous_manifest.get("generator_provider", "openai")
        old_judge_provider = previous_manifest.get("judge_provider", "openai")
        if old_generator_provider != args.generator_provider:
            raise ValueError(
                f"Run directory already uses generator_provider={old_generator_provider!r}, but this "
                f"command requested {args.generator_provider!r}. Use a new --run-dir."
            )
        if old_judge_provider != args.judge_provider:
            raise ValueError(
                f"Run directory already uses judge_provider={old_judge_provider!r}, but this command "
                f"requested {args.judge_provider!r}. Use a new --run-dir."
            )
        for key, requested in (("generator_model", args.generator_model), ("judge_model", args.judge_model)):
            existing = previous_manifest.get(key)
            if existing != requested:
                raise ValueError(
                    f"Run directory already uses {key}={existing!r}, but this command requested "
                    f"{requested!r}. Use the original model or a new --run-dir."
                )
        if previous_manifest.get("selected_case_ids") != [row["case_id"] for row in selected]:
            raise ValueError(
                "Run directory already has a different case selection. Use the original --limit "
                "or a new --run-dir."
            )
        if (
            "openrouter_output_mode" in previous_manifest
            and previous_manifest["openrouter_output_mode"] != args.openrouter_output_mode
        ):
            raise ValueError(
                "Run directory already uses a different OpenRouter output mode. Use a new --run-dir."
            )
        previous_judge_reasoning = previous_manifest.get("judge_reasoning", "omit")
        if previous_judge_reasoning != args.judge_reasoning:
            existing_judgments = latest(args.run_dir / "judgments_C4.jsonl")
            legacy_unjudged_run = (
                "judge_reasoning" not in previous_manifest and not existing_judgments
            )
            if not legacy_unjudged_run:
                raise ValueError(
                    f"Run directory already uses judge_reasoning={previous_judge_reasoning!r}, but "
                    f"this command requested {args.judge_reasoning!r}. Use the original setting or "
                    "a new --run-dir."
                )
    manifest = {
        "created_at": previous_manifest.get("created_at", now()) if previous_manifest else now(),
        "selection": "all rows" if args.limit is None else f"first {args.limit} rows in dataset order",
        "selected_case_ids": [row["case_id"] for row in selected],
        "generator_model": args.generator_model, "judge_model": args.judge_model,
        "generator_provider": args.generator_provider, "judge_provider": args.judge_provider,
        "judge_reasoning": args.judge_reasoning,
        "generator_reasoning": args.generator_reasoning,
        "openrouter_output_mode": args.openrouter_output_mode,
        "gold_exposed_to_generator": False, "gold_exposed_to_judge": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    generation_path = args.run_dir / "generations_C4.jsonl"
    judgment_path = args.run_dir / "judgments_C4.jsonl"
    generator_client = create_client(
        args.generator_provider,
        timeout=args.timeout,
        openrouter_site_url=args.openrouter_site_url,
        openrouter_app_name=args.openrouter_app_name,
    )
    judge_client = create_client(
        args.judge_provider,
        timeout=args.timeout,
        openrouter_site_url=args.openrouter_site_url,
        openrouter_app_name=args.openrouter_app_name,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    generated = latest(generation_path)

    async def generate(row: dict[str, Any]) -> None:
        async with semaphore:
            try:
                prompt = "Generate from this input only:\n" + json.dumps(row, ensure_ascii=False) + "\nSchema:\n" + json.dumps(schema, ensure_ascii=False)
                response, output = await api_json(
                    generator_client,
                    provider=args.generator_provider,
                    model=args.generator_model,
                    system="Generate grounded French firefighter-doctrine educational content. Return only the requested JSON.",
                    prompt=prompt,
                    schema=generation_api_schema,
                    schema_name="task_c4_output",
                    max_output_tokens=4000,
                    openrouter_output_mode=args.openrouter_output_mode,
                    reasoning_effort=args.generator_reasoning,
                    send_openrouter_reasoning=True,
                    attempts=args.max_attempts,
                )
                errors = generation_errors(output, row)
                record = {"case_id": row["case_id"], "status": "ok", "provider": args.generator_provider, "model": args.generator_model, "format_valid": not errors, "validation_errors": errors, "output": output, "response_id": response.id, "usage": response_usage(response)}
            except Exception as exc:
                record = {"case_id": row["case_id"], "status": "error", "provider": args.generator_provider, "model": args.generator_model, "error": str(exc)[:2000], "output": None}
            record["reasoning_effort"] = args.generator_reasoning
            append_jsonl(generation_path, record)

    await asyncio.gather(*(generate(row) for row in selected if generated.get(row["case_id"], {}).get("status") != "ok"))
    generated = latest(generation_path)
    judged = latest(judgment_path)

    async def judge(row: dict[str, Any]) -> None:
        case_id = row["case_id"]
        async with semaphore:
            generation = generated.get(case_id)
            if not generation or generation.get("status") != "ok":
                append_jsonl(judgment_path, {"case_id": case_id, "status": "error", "error": "No usable generation"})
                return
            payload = {"case": full[case_id], "generated_output": generation["output"], "judge_protocol": protocol}
            try:
                response, judgment = await api_json(
                    judge_client,
                    provider=args.judge_provider,
                    model=args.judge_model,
                    system="You are an independent judge. Evaluate only from the supplied doctrine evidence and draft requirements; do not reward style. Return the schema exactly.",
                    prompt=json.dumps(payload, ensure_ascii=False),
                    schema=judge_api_schema,
                    schema_name="task_c4_judgment",
                    max_output_tokens=5000,
                    openrouter_output_mode=args.openrouter_output_mode,
                    reasoning_effort=args.judge_reasoning,
                    attempts=args.max_attempts,
                )
                record = {"case_id": case_id, "status": "ok", "provider": args.judge_provider, "model": args.judge_model, "reasoning_effort": args.judge_reasoning, "judgment": judgment, "response_id": response.id, "usage": response_usage(response)}
            except Exception as exc:
                record = {"case_id": case_id, "status": "error", "provider": args.judge_provider, "model": args.judge_model, "error": str(exc)[:2000]}
            append_jsonl(judgment_path, record)

    await asyncio.gather(*(judge(row) for row in selected if judged.get(row["case_id"], {}).get("status") != "ok"))
    usable = [row["judgment"] for row in latest(judgment_path).values() if row.get("status") == "ok" and row["case_id"] in manifest["selected_case_ids"]]
    metric_paths = {
        "lesson_relation_quality": ("lesson_relation_quality", "score"),
        "condition_adherence": ("condition_adherence", "score"),
        "multi_kc_reasoning": ("qcm_multi_kc_reasoning", "score"),
        "kc_attribution_f1": ("kc_attribution", "f1"),
        "factual_correctness": ("factual_correctness", "score"),
        "answer_validity": ("answer_validity", "score"),
        "pdf_grounding": ("pdf_grounding", "score"),
    }
    metrics = {name: (fmean(float(j[a][b]) for j in usable) if usable else None) for name, (a, b) in metric_paths.items()}
    summary = {"selected_cases": len(selected), "judged_cases": len(usable), "generator_provider": args.generator_provider, "generator_model": args.generator_model, "judge_provider": args.judge_provider, "judge_model": args.judge_model, "judge_reasoning": args.judge_reasoning, "metrics": metrics, "limitation": "Draft gold; human expert validation pending."}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    await generator_client.close()
    await judge_client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="First N cases; omit for all 116")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR)
    parser.add_argument(
        "--generator-reasoning", default="omit",
        choices=["omit", "none", "low", "medium", "high", "xhigh"],
        help="Explicit generator reasoning effort; omit preserves provider defaults",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--judge-reasoning",
        choices=["omit", "none", "low", "medium", "high", "xhigh"],
        default="omit",
        help="Reasoning effort sent to the judge; omit preserves the legacy behavior",
    )
    parser.add_argument("--generator-provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--judge-provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--openrouter-output-mode", choices=["json_schema", "json_object", "prompt_only"], default="json_schema")
    parser.add_argument("--openrouter-site-url", default=os.getenv("OPENROUTER_SITE_URL"))
    parser.add_argument("--openrouter-app-name", default=os.getenv("OPENROUTER_APP_NAME", "ITS benchmark"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
