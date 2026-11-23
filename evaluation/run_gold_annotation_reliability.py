"""Independent two-model audit of the benchmark's draft gold annotations.

The source JSONL files are opened read-only. Results are written to a separate
output directory and must be reported as inter-model, not human-expert,
agreement.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from provider_api import create_client, request_json_once, response_usage


ROOT = Path(__file__).resolve().parent
V1_RESULTS_DIR = ROOT / "gold_reliability_gpt54_vs_gpt56sol"
TARGETED_REPAIR_DIR = ROOT / "gold_reliability_gpt54_vs_gpt56sol_targeted_v2_repair"
DATASETS = {
    "A": ROOT / "ITS_FINAL_A_300.jsonl",
    "B": ROOT / "ITS_FINAL_B_300.jsonl",
    "C4": ROOT / "TASK_C4_AUDITED_116" / "FULL_WITH_GOLD_C4_116.jsonl",
}
EXPECTED_COUNTS = {"A": 300, "B": 300, "C4": 116}
PROMPT_VERSION = "gold_reliability_gold_only_v2"
ORDINAL_LEVELS = {
    "source_support": ["unsupported", "partially_supported", "fully_supported"],
    "internal_consistency": ["major_issue", "minor_issue", "consistent"],
    "task_gold_alignment": ["misaligned", "partially_aligned", "aligned"],
    "overall_adequacy": ["reject", "minor_revision", "accept"],
}
DIMENSIONS = [*ORDINAL_LEVELS, "acceptable_without_major_revision"]

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "case_id", "task", "source_support", "internal_consistency",
        "task_gold_alignment", "overall_adequacy",
        "acceptable_without_major_revision", "confidence", "issues", "rationale",
    ],
    "properties": {
        "case_id": {"type": "string"},
        "task": {"type": "string", "enum": ["A", "B", "C4"]},
        "source_support": {"type": "string", "enum": ORDINAL_LEVELS["source_support"]},
        "internal_consistency": {"type": "string", "enum": ORDINAL_LEVELS["internal_consistency"]},
        "task_gold_alignment": {"type": "string", "enum": ORDINAL_LEVELS["task_gold_alignment"]},
        "overall_adequacy": {"type": "string", "enum": ORDINAL_LEVELS["overall_adequacy"]},
        "acceptable_without_major_revision": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are an independent research-data auditor. Your sole target is the
supplied draft_gold_annotation. Assess it only against evidence_and_inputs in the same payload;
do not use outside factual knowledge. This is NOT an audit of a generator response: no generated
lesson, feedback, or QCM is expected, and absence of such output must never be treated as a
defect. You are not a human domain expert and must not claim expert certification. Determine
whether the draft gold is source-supported, internally consistent, aligned with the task, and
usable without a major correction. Treat stylistic imperfections as minor. Treat an unsupported,
contradictory, or task-invalid gold target as major. For Task C4, do not infer that every supplied
KC is strictly necessary when necessity is marked unverified. Audit annotation content, not
status labels such as draft, pending, unvalidated, or synthetic; those labels are expected and
are not defects. Return only the required JSON. Do not mention or infer another auditor's
decision."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                records.append(value)
    return records


def case_id(record: dict[str, Any], task: str) -> str:
    value = record.get("case_id")
    if not value and task == "C4":
        value = record.get("input", {}).get("case_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing case_id in Task {task}")
    return value


def cluster_id(record: dict[str, Any], task: str) -> str:
    if task == "A":
        return f"A:{record.get('input', {}).get('kc_id', case_id(record, task))}"
    if task == "B":
        template = record.get("metadata", {}).get("question_template_id")
        return f"B:{template or case_id(record, task)}"
    base = re.sub(r"_(conceptual|operational)$", "", case_id(record, task), flags=re.I)
    return f"C4:{base}"


def audit_payload(record: dict[str, Any], task: str) -> dict[str, Any]:
    """Return only evidence/input and the draft gold being audited.

    In particular, generator-output placeholders and generator-scoring rubrics
    are structurally excluded so they cannot be mistaken for the audit target.
    """
    gold_key = "gold" if task == "C4" else "reference_hidden_from_generator"
    if not isinstance(record.get("input"), dict) or not isinstance(record.get(gold_key), dict):
        raise ValueError(f"Task {task}: missing input or {gold_key}")
    return {
        "case_id": case_id(record, task),
        "task": task,
        "evidence_and_inputs": record["input"],
        "draft_gold_annotation": record[gold_key],
    }


def load_cases(selected: Iterable[str], limit_per_task: int | None) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in selected:
        path = DATASETS[task]
        records = read_jsonl(path)
        if limit_per_task is None and len(records) != EXPECTED_COUNTS[task]:
            raise ValueError(f"Task {task}: expected {EXPECTED_COUNTS[task]} cases, found {len(records)}")
        if limit_per_task is not None:
            records = records[:limit_per_task]
        for record in records:
            cid = case_id(record, task)
            if cid in seen:
                raise ValueError(f"Duplicate case_id across selected data: {cid}")
            seen.add(cid)
            loaded.append({
                "task": task,
                "case_id": cid,
                "cluster_id": cluster_id(record, task),
                "payload": audit_payload(record, task),
            })
    return loaded


def validate_annotation(value: dict[str, Any], expected_case: str, expected_task: str) -> None:
    if value.get("case_id") != expected_case or value.get("task") != expected_task:
        raise ValueError("Model returned the wrong case_id or task")
    for name, levels in ORDINAL_LEVELS.items():
        if value.get(name) not in levels:
            raise ValueError(f"Invalid {name}: {value.get(name)!r}")
    if not isinstance(value.get("acceptable_without_major_revision"), bool):
        raise ValueError("acceptable_without_major_revision must be boolean")
    expected_acceptable = value.get("overall_adequacy") != "reject"
    if value["acceptable_without_major_revision"] != expected_acceptable:
        raise ValueError("Acceptability must be false exactly when overall_adequacy is reject")
    if value.get("confidence") not in {"low", "medium", "high"}:
        raise ValueError("Invalid confidence")
    if not isinstance(value.get("issues"), list) or not isinstance(value.get("rationale"), str):
        raise ValueError("Invalid issues or rationale")


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["case_id"]: row for row in read_jsonl(path) if row.get("status") == "ok"}


def validate_annotation_file(path: Path, model: str, reasoning: str) -> None:
    for row in read_completed(path).values():
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"{path} contains results from a different prompt protocol")
        if row.get("model") != model or row.get("reasoning_effort") != reasoning:
            raise ValueError(f"{path} contains results from a different model configuration")


def normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def is_invalid_v1_task_a_judgment(row: dict[str, Any]) -> bool:
    annotation = row.get("annotation", {})
    text = normalized_text(" ".join([
        *(str(item) for item in annotation.get("issues", [])),
        str(annotation.get("rationale", "")),
    ]))
    missing_output_signal = (
        "model_output" in text
        or ("sortie" in text and "modele" in text and ("nul" in text or "null" in text))
    )
    return (
        row.get("task") == "A"
        and row.get("model") == "gpt-5.6-sol"
        and annotation.get("overall_adequacy") == "reject"
        and not annotation.get("acceptable_without_major_revision", True)
        and missing_output_signal
    )


def targeted_v1_case_ids(v1_model_2_path: Path) -> list[str]:
    rows = read_completed(v1_model_2_path)
    selected = sorted(cid for cid, row in rows.items() if is_invalid_v1_task_a_judgment(row))
    if len(selected) != 15:
        raise ValueError(
            f"Safety stop: expected exactly 15 invalid v1 Task A judgments, found {len(selected)}"
        )
    return selected


async def annotate_model(
    cases: list[dict[str, Any]], model: str, reasoning: str, output_path: Path,
    concurrency: int, timeout: float, retries: int,
) -> None:
    validate_annotation_file(output_path, model, reasoning)
    completed = read_completed(output_path)
    pending = [item for item in cases if item["case_id"] not in completed]
    print(f"{model}: {len(completed)} completed, {len(pending)} pending", flush=True)
    if not pending:
        return
    client = create_client("openai", timeout=timeout)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def one(item: dict[str, Any]) -> None:
        prompt = (
            f"Audit Task {item['task']} case {item['case_id']}.\n"
            "Interpret task_gold_alignment as: A=objective/evidence alignment; "
            "B=observed-error/target/remediation alignment; C4=KC/integration/condition alignment.\n"
            "Evaluate only draft_gold_annotation against evidence_and_inputs. "
            "There is intentionally no generator output to evaluate.\n"
            "Audit payload JSON:\n" + json.dumps(item["payload"], ensure_ascii=False, separators=(",", ":"))
        )
        last_error: Exception | None = None
        async with semaphore:
            for attempt in range(retries + 1):
                try:
                    response, annotation = await request_json_once(
                        client, provider="openai", model=model, system=SYSTEM_PROMPT,
                        prompt=prompt, schema=SCHEMA, schema_name="gold_annotation_audit",
                        max_output_tokens=3000, openai_output_mode="json_schema",
                        openrouter_output_mode="json_schema", reasoning_effort=reasoning,
                        verbosity="low",
                    )
                    validate_annotation(annotation, item["case_id"], item["task"])
                    row = {
                        "case_id": item["case_id"], "task": item["task"],
                        "cluster_id": item["cluster_id"], "model": model,
                        "response_model": getattr(response, "model", None),
                        "reasoning_effort": reasoning, "prompt_version": PROMPT_VERSION,
                        "status": "ok", "annotation": annotation,
                        "usage": response_usage(response), "created_at": utc_now(),
                    }
                    break
                except Exception as exc:  # retry transient API and validation failures
                    last_error = exc
                    if attempt == retries:
                        raise RuntimeError(f"{model} failed on {item['case_id']}: {exc}") from exc
                    await asyncio.sleep(min(2 ** attempt, 20))
        async with write_lock:
            with output_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            done = len(read_completed(output_path))
            print(f"{model}: {done}/{len(cases)} ({item['case_id']})", flush=True)

    try:
        await asyncio.gather(*(one(item) for item in pending))
    finally:
        await client.close()


def cohen_kappa(a: list[Any], b: list[Any]) -> tuple[float | None, float]:
    if len(a) != len(b) or not a:
        return None, float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    labels = set(a) | set(b)
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[x] / len(a)) * (cb[x] / len(b)) for x in labels)
    return (None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)), observed


def weighted_kappa(a: list[str], b: list[str], levels: list[str]) -> float | None:
    if not a:
        return None
    index = {value: i for i, value in enumerate(levels)}
    denominator = max(len(levels) - 1, 1)
    observed_disagreement = statistics.fmean(abs(index[x] - index[y]) / denominator for x, y in zip(a, b))
    ca, cb = Counter(a), Counter(b)
    expected_disagreement = sum(
        (ca[x] / len(a)) * (cb[y] / len(b)) * abs(index[x] - index[y]) / denominator
        for x in levels for y in levels
    )
    return None if math.isclose(expected_disagreement, 0.0) else 1 - observed_disagreement / expected_disagreement


def gwet_coefficient(
    a: list[Any], b: list[Any], levels: list[Any], *, ordinal: bool,
) -> float | None:
    """Calculate Gwet's AC1 (nominal) or linear-weighted AC2 (ordinal)."""
    if len(a) != len(b) or not a or len(levels) < 2:
        return None
    index = {value: i for i, value in enumerate(levels)}
    if any(value not in index for value in [*a, *b]):
        raise ValueError("Observed value is absent from the declared Gwet levels")
    q = len(levels)

    def weight(left: Any, right: Any) -> float:
        if not ordinal:
            return 1.0 if left == right else 0.0
        return 1.0 - abs(index[left] - index[right]) / (q - 1)

    observed = statistics.fmean(weight(left, right) for left, right in zip(a, b))
    pooled = Counter([*a, *b])
    proportions = {value: pooled[value] / (2 * len(a)) for value in levels}
    total_weight = sum(weight(left, right) for left in levels for right in levels)
    expected = (
        total_weight
        / (q * (q - 1))
        * sum(proportions[value] * (1.0 - proportions[value]) for value in levels)
    )
    return None if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def agreement_for_rows(rows: list[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows), "dimensions": {}}
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row["cluster_id"]].append(row)
    cluster_names = list(clusters)
    rng = random.Random(seed)
    for dimension in DIMENSIONS:
        a = [row["a"][dimension] for row in rows]
        b = [row["b"][dimension] for row in rows]
        kappa, observed = cohen_kappa(a, b)
        ordinal = dimension in ORDINAL_LEVELS
        levels: list[Any] = ORDINAL_LEVELS[dimension] if ordinal else [False, True]
        gwet = gwet_coefficient(a, b, levels, ordinal=ordinal)
        boot_kappa: list[float] = []
        boot_gwet: list[float] = []
        if repetitions and cluster_names:
            for _ in range(repetitions):
                sampled = [rng.choice(cluster_names) for _ in cluster_names]
                sampled_rows = [row for name in sampled for row in clusters[name]]
                sampled_a = [row["a"][dimension] for row in sampled_rows]
                sampled_b = [row["b"][dimension] for row in sampled_rows]
                kappa_value, _ = cohen_kappa(sampled_a, sampled_b)
                gwet_value = gwet_coefficient(sampled_a, sampled_b, levels, ordinal=ordinal)
                if kappa_value is not None:
                    boot_kappa.append(kappa_value)
                if gwet_value is not None:
                    boot_gwet.append(gwet_value)
        entry: dict[str, Any] = {
            "cohen_kappa": kappa,
            "agreement": observed,
            "cohen_kappa_cluster_bootstrap_95_ci": [
                percentile(boot_kappa, 0.025), percentile(boot_kappa, 0.975)
            ],
            "annotator_1_distribution": dict(Counter(a)),
            "annotator_2_distribution": dict(Counter(b)),
        }
        if ordinal:
            entry["linear_weighted_kappa"] = weighted_kappa(a, b, ORDINAL_LEVELS[dimension])
            entry["gwet_ac2_linear"] = gwet
            entry["gwet_ac2_cluster_bootstrap_95_ci"] = [
                percentile(boot_gwet, 0.025), percentile(boot_gwet, 0.975)
            ]
        else:
            entry["gwet_ac1"] = gwet
            entry["gwet_ac1_cluster_bootstrap_95_ci"] = [
                percentile(boot_gwet, 0.025), percentile(boot_gwet, 0.975)
            ]
        result["dimensions"][dimension] = entry
    return result


def paired_rows(path_a: Path, path_b: Path) -> list[dict[str, Any]]:
    a = read_completed(path_a)
    b = read_completed(path_b)
    ids = sorted(set(a) & set(b))
    return [{
        "case_id": cid, "task": a[cid]["task"], "cluster_id": a[cid]["cluster_id"],
        "a": a[cid]["annotation"], "b": b[cid]["annotation"],
    } for cid in ids]


def cross_tab(rows: list[dict[str, Any]], dimension: str) -> dict[str, int]:
    return dict(Counter(
        f"model_1={row['a'][dimension]} | model_2={row['b'][dimension]}"
        for row in rows
    ))


def write_reports(
    output_dir: Path, model_a: str, model_b: str, rows: list[dict[str, Any]],
    repetitions: int, seed: int, reasoning_effort: str, *,
    audit_protocol: str = "v2_gold_only_payload",
    report_prompt_version: str = PROMPT_VERSION,
    additional_limitations: list[str] | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    limitations = [
        "Both annotators are language models, not independent human domain experts.",
        "Agreement does not establish doctrinal correctness or expert-certified gold validity.",
        "Models from one provider may share training or systematic biases.",
    ]
    limitations.extend(additional_limitations or [])
    summary: dict[str, Any] = {
        "report_type": "inter_model_gold_annotation_reliability_not_human_expert_agreement",
        "audit_protocol": audit_protocol,
        "audit_target": "draft_gold_annotation_against_case_evidence_and_inputs",
        "excluded_from_model_payload": (
            ["model_output", "evaluation_rubric", "unrelated_metadata"]
            if audit_protocol == "v2_gold_only_payload" else None
        ),
        "models": [model_a, model_b], "reasoning_effort": reasoning_effort,
        "prompt_version": report_prompt_version, "paired_cases": len(rows),
        "bootstrap": {"method": "paired cluster bootstrap", "repetitions": repetitions, "seed": seed},
        "reliability_statistics": {
            "nominal": ["exact agreement", "Cohen's kappa", "Gwet's AC1"],
            "ordinal": ["exact agreement", "Cohen's kappa", "linear weighted kappa", "linear weighted Gwet's AC2"],
            "gwet_expected_agreement": "standard pooled-marginal AC1/AC2 formulation",
        },
        "overall": agreement_for_rows(rows, repetitions, seed),
        "by_task": {
            task: agreement_for_rows([row for row in rows if row["task"] == task], repetitions, seed + i + 1)
            for i, task in enumerate(("A", "B", "C4"))
        },
        "limitations": limitations,
        "created_at": utc_now(),
    }
    summary["case_outcomes"] = {
        "both_acceptable_without_major_revision": sum(
            row["a"]["acceptable_without_major_revision"] and row["b"]["acceptable_without_major_revision"]
            for row in rows
        ),
        "either_model_requires_major_revision": sum(
            not row["a"]["acceptable_without_major_revision"] or not row["b"]["acceptable_without_major_revision"]
            for row in rows
        ),
        "binary_disagreements": sum(
            row["a"]["acceptable_without_major_revision"] != row["b"]["acceptable_without_major_revision"]
            for row in rows
        ),
    }
    c4_rows = [row for row in rows if row["task"] == "C4"]
    summary["c4_disagreement_analysis"] = {
        "n": len(c4_rows),
        "cross_tabs": {dimension: cross_tab(c4_rows, dimension) for dimension in DIMENSIONS},
        "both_require_major_revision": sum(
            not row["a"]["acceptable_without_major_revision"]
            and not row["b"]["acceptable_without_major_revision"]
            for row in c4_rows
        ),
        "either_requires_major_revision": sum(
            not row["a"]["acceptable_without_major_revision"]
            or not row["b"]["acceptable_without_major_revision"]
            for row in c4_rows
        ),
    }
    if extra_summary:
        summary.update(extra_summary)
    (output_dir / "agreement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "agreement_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "scope", "dimension", "n", "agreement", "cohen_kappa",
            "cohen_kappa_ci_low", "cohen_kappa_ci_high", "linear_weighted_kappa",
            "gwet_statistic", "gwet_value", "gwet_ci_low", "gwet_ci_high",
        ])
        for scope, value in [("overall", summary["overall"]), *summary["by_task"].items()]:
            for dimension, metric in value["dimensions"].items():
                gwet_name = "AC2_linear" if dimension in ORDINAL_LEVELS else "AC1"
                gwet_key = "gwet_ac2_linear" if dimension in ORDINAL_LEVELS else "gwet_ac1"
                gwet_ci_key = (
                    "gwet_ac2_cluster_bootstrap_95_ci"
                    if dimension in ORDINAL_LEVELS else "gwet_ac1_cluster_bootstrap_95_ci"
                )
                writer.writerow([
                    scope, dimension, value["n"], metric["agreement"], metric["cohen_kappa"],
                    *metric["cohen_kappa_cluster_bootstrap_95_ci"],
                    metric.get("linear_weighted_kappa"), gwet_name, metric[gwet_key],
                    *metric[gwet_ci_key],
                ])
    with (output_dir / "case_level_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "case_id", "task", "cluster_id", "model_1_adequacy", "model_2_adequacy",
            "model_1_acceptable", "model_2_acceptable", "binary_disagreement",
            "model_1_issues", "model_2_issues",
        ])
        for row in rows:
            writer.writerow([
                row["case_id"], row["task"], row["cluster_id"],
                row["a"]["overall_adequacy"], row["b"]["overall_adequacy"],
                row["a"]["acceptable_without_major_revision"], row["b"]["acceptable_without_major_revision"],
                row["a"]["acceptable_without_major_revision"] != row["b"]["acceptable_without_major_revision"],
                " | ".join(row["a"]["issues"]), " | ".join(row["b"]["issues"]),
            ])
    with (output_dir / "c4_disagreement_analysis.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "case_id", "cluster_id",
            *[f"model_1_{dimension}" for dimension in DIMENSIONS],
            *[f"model_2_{dimension}" for dimension in DIMENSIONS],
            "any_dimension_disagreement", "model_1_issues", "model_2_issues",
        ])
        for row in c4_rows:
            writer.writerow([
                row["case_id"], row["cluster_id"],
                *[row["a"][dimension] for dimension in DIMENSIONS],
                *[row["b"][dimension] for dimension in DIMENSIONS],
                any(row["a"][dimension] != row["b"][dimension] for dimension in DIMENSIONS),
                " | ".join(row["a"]["issues"]), " | ".join(row["b"]["issues"]),
            ])
    lines = [
        "# Inter-model gold-annotation reliability", "",
        f"Models: `{model_a}` and `{model_b}`; reasoning effort: `{reasoning_effort}`.", "",
        f"Audit protocol: `{audit_protocol}`; report prompt provenance: `{report_prompt_version}`.", "",
        "> This is model-model agreement, not human expert agreement or gold certification.", "",
        "| Scope | Dimension | n | Agreement | Cohen's kappa | 95% CI | Weighted kappa | Gwet | 95% CI |",
        "|:--|:--|--:|--:|--:|:--|--:|--:|:--|",
    ]
    for scope, value in [("Overall", summary["overall"]), *[(k, v) for k, v in summary["by_task"].items()]]:
        for dimension, metric in value["dimensions"].items():
            fmt = lambda x: "NA" if x is None else f"{x:.3f}"
            kappa_ci = metric["cohen_kappa_cluster_bootstrap_95_ci"]
            if dimension in ORDINAL_LEVELS:
                gwet_value = metric["gwet_ac2_linear"]
                gwet_ci = metric["gwet_ac2_cluster_bootstrap_95_ci"]
                gwet_label = f"AC2={fmt(gwet_value)}"
            else:
                gwet_value = metric["gwet_ac1"]
                gwet_ci = metric["gwet_ac1_cluster_bootstrap_95_ci"]
                gwet_label = f"AC1={fmt(gwet_value)}"
            lines.append(
                f"| {scope} | {dimension} | {value['n']} | {metric['agreement']:.3f} | "
                f"{fmt(metric['cohen_kappa'])} | [{fmt(kappa_ci[0])}, {fmt(kappa_ci[1])}] | "
                f"{fmt(metric.get('linear_weighted_kappa'))} | {gwet_label} | "
                f"[{fmt(gwet_ci[0])}, {fmt(gwet_ci[1])}] |"
            )
    lines += [
        "", "## Task C4 disagreement analysis", "",
        f"Cases: {len(c4_rows)}; either model requires major revision: "
        f"{summary['c4_disagreement_analysis']['either_requires_major_revision']}; "
        f"both require major revision: {summary['c4_disagreement_analysis']['both_require_major_revision']}.",
        "", "See `c4_disagreement_analysis.csv` for paired case-level labels and reasons.",
    ]
    lines += ["", "## Limitations", "", *[f"- {item}" for item in summary["limitations"]]]
    (output_dir / "agreement_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_targeted_v1_repair(args: argparse.Namespace) -> None:
    """Re-audit only the 15 objectively invalid GPT-5.6 Task A responses."""
    v1_model_1 = V1_RESULTS_DIR / "annotations_gpt_5_4.jsonl"
    v1_model_2 = V1_RESULTS_DIR / "annotations_gpt_5_6_sol.jsonl"
    if not v1_model_1.exists() or not v1_model_2.exists():
        raise FileNotFoundError("The complete v1 annotation files are required for targeted repair")
    original_1 = read_completed(v1_model_1)
    original_2 = read_completed(v1_model_2)
    if len(original_1) != 716 or len(original_2) != 716 or set(original_1) != set(original_2):
        raise ValueError("Safety stop: v1 must contain exactly 716 paired successful annotations")

    target_ids = targeted_v1_case_ids(v1_model_2)
    all_task_a = {item["case_id"]: item for item in load_cases(["A"], None)}
    target_cases = [all_task_a[cid] for cid in target_ids]
    output_dir = args.output_dir if args.output_dir_was_explicit else TARGETED_REPAIR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    corrections_path = output_dir / "corrected_gpt_5_6_sol_15.jsonl"
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "audit_protocol": "targeted_v2_technical_repair_over_v1",
        "reason": "15 GPT-5.6 Task A responses audited missing model_output instead of draft gold",
        "model": args.model_2,
        "reasoning_effort": args.reasoning_effort,
        "prompt_version_for_retries": PROMPT_VERSION,
        "api_call_count_expected": 15,
        "target_case_ids": target_ids,
        "source_v1_files": {
            str(v1_model_1): sha256(v1_model_1),
            str(v1_model_2): sha256(v1_model_2),
        },
        "datasets": {task: {"path": str(path), "sha256": sha256(path)} for task, path in DATASETS.items()},
        "datasets_modified": False,
        "started_or_resumed_at": utc_now(),
    }
    if manifest_path.exists() and corrections_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("audit_protocol", "model", "reasoning_effort", "prompt_version_for_retries", "target_case_ids", "source_v1_files", "datasets"):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume targeted repair: manifest field {key!r} changed")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    await annotate_model(
        target_cases, args.model_2, args.reasoning_effort, corrections_path,
        args.concurrency, args.timeout, args.retries,
    )
    corrections = read_completed(corrections_path)
    if set(corrections) != set(target_ids):
        raise ValueError("Targeted repair did not produce all and only the 15 expected corrections")

    repaired_2 = dict(original_2)
    repaired_2.update(corrections)
    rows = [{
        "case_id": cid,
        "task": original_1[cid]["task"],
        "cluster_id": original_1[cid]["cluster_id"],
        "a": original_1[cid]["annotation"],
        "b": repaired_2[cid]["annotation"],
    } for cid in sorted(original_1)]
    write_reports(
        output_dir, args.model_1, args.model_2, rows,
        args.bootstrap_repetitions, args.seed, args.reasoning_effort,
        audit_protocol="targeted_v2_technical_repair_over_v1",
        report_prompt_version="v1_with_15_predefined_gpt_5_6_sol_v2_retries",
        additional_limitations=[
            "This report combines 701 original v1 GPT-5.6 judgments with 15 predefined v2 technical retries.",
            "The targeted retry corrects an objective protocol failure but is not a full uniform-protocol rerun.",
        ],
        extra_summary={
            "targeted_repair": {
                "corrected_model": args.model_2,
                "corrected_case_count": 15,
                "target_case_ids": target_ids,
                "selection_rule": "Task A reject whose rationale explicitly treats absent model_output as the defect",
                "retry_payload_excluded_fields": ["model_output", "evaluation_rubric", "unrelated_metadata"],
                "original_results_preserved": True,
            }
        },
    )
    print(f"Targeted repair complete: 15 API calls; repaired 716-case reports in {output_dir}", flush=True)


async def async_main(args: argparse.Namespace) -> None:
    if not os.getenv("OPENAI_API_KEY") and not args.report_only:
        raise RuntimeError("OPENAI_API_KEY is not set in this terminal")
    if args.repair_v1_invalid_task_a:
        if args.report_only:
            raise ValueError("--report-only cannot be combined with --repair-v1-invalid-task-a")
        await run_targeted_v1_repair(args)
        return
    selected = ["A", "B", "C4"] if args.tasks == "all" else args.tasks.split(",")
    cases = load_cases(selected, args.limit_per_task)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = [args.output_dir / "annotations_gpt_5_4.jsonl", args.output_dir / "annotations_gpt_5_6_sol.jsonl"]
    manifest = {
        "models": [args.model_1, args.model_2], "reasoning_effort": args.reasoning_effort,
        "tasks": selected, "case_count": len(cases), "prompt_version": PROMPT_VERSION,
        "audit_protocol": "v2_gold_only_payload",
        "audit_target": "draft_gold_annotation_against_case_evidence_and_inputs",
        "included_in_model_payload": ["case_id", "task", "evidence_and_inputs", "draft_gold_annotation"],
        "excluded_from_model_payload": ["model_output", "evaluation_rubric", "unrelated_metadata"],
        "datasets": {task: {"path": str(DATASETS[task]), "sha256": sha256(DATASETS[task])} for task in selected},
        "datasets_modified": False, "started_or_resumed_at": utc_now(),
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and any(path.exists() for path in model_paths):
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("models", "reasoning_effort", "prompt_version", "audit_protocol", "tasks", "datasets"):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume: manifest field {key!r} changed")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.report_only:
        await annotate_model(cases, args.model_1, args.reasoning_effort, model_paths[0], args.concurrency, args.timeout, args.retries)
        await annotate_model(cases, args.model_2, args.reasoning_effort, model_paths[1], args.concurrency, args.timeout, args.retries)
    validate_annotation_file(model_paths[0], args.model_1, args.reasoning_effort)
    validate_annotation_file(model_paths[1], args.model_2, args.reasoning_effort)
    rows = paired_rows(*model_paths)
    if not rows:
        raise RuntimeError("No paired successful annotations are available")
    write_reports(
        args.output_dir, args.model_1, args.model_2, rows,
        args.bootstrap_repetitions, args.seed, args.reasoning_effort,
    )
    print(f"Reports written to {args.output_dir} for {len(rows)} paired cases", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-1", default="gpt-5.4")
    parser.add_argument("--model-2", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--tasks", default="all", help="all or comma-separated A,B,C4")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--repair-v1-invalid-task-a",
        action="store_true",
        help="Make exactly 15 GPT-5.6 calls to repair the predefined invalid v1 Task A judgments",
    )
    args = parser.parse_args()
    args.output_dir_was_explicit = args.output_dir is not None
    if args.output_dir is None:
        args.output_dir = ROOT / "gold_reliability_gpt54_vs_gpt56sol_v2"
    return args


if __name__ == "__main__":
    try:
        asyncio.run(async_main(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted; completed cases are saved and the run can be resumed.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
