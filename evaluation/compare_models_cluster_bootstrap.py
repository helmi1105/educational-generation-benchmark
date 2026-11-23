"""Paired, cluster-bootstrap comparison of two saved ITS evaluation runs.

This is an offline statistical analysis. It does not call a generator or
judge, and it never modifies either source run.

Examples from the evaluation directory:

  python compare_models_cluster_bootstrap.py \
    --task A --run-a runs/full_01 \
    --run-b runs/openrouter_ab_full_qwen35_9b \
    --output-dir comparisons/gpt54nano_vs_qwen35/task_A

  python compare_models_cluster_bootstrap.py \
    --task C4 --run-a runs/c4_full_gpt5mini_20250807 \
    --run-b runs/openrouter_c4_full_qwen35_9b \
    --output-dir comparisons/gpt54nano_vs_qwen35/task_C4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
DATASET_A = HERE / "ITS_FINAL_A_300.jsonl"
DATASET_B = HERE / "ITS_FINAL_B_300.jsonl"
DATASET_C4 = HERE / "TASK_C4_AUDITED_116" / "FULL_WITH_GOLD_C4_116.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in read_jsonl(path)}


def selected_ids(manifest: dict[str, Any], task: str) -> list[str]:
    value = manifest.get("selected_case_ids")
    if task in {"A", "B"}:
        if not isinstance(value, dict) or not isinstance(value.get(task), list):
            raise ValueError(f"Manifest has no selected_case_ids for Task {task}")
        return list(value[task])
    if not isinstance(value, list):
        raise ValueError("C4 manifest selected_case_ids must be an array")
    return list(value)


def judgment_path(run_dir: Path, task: str) -> Path:
    name = f"judgments_{task}.jsonl" if task in {"A", "B"} else "judgments_C4.jsonl"
    return run_dir / name


def run_model_label(manifest: dict[str, Any], fallback: str) -> str:
    return str(manifest.get("generator_model") or fallback)


def cluster_map(task: str) -> dict[str, str]:
    if task == "A":
        return {
            row["case_id"]: row["input"]["kc_id"]
            for row in read_jsonl(DATASET_A)
        }
    if task == "B":
        return {
            row["case_id"]: row["metadata"]["question_template_id"]
            for row in read_jsonl(DATASET_B)
        }
    return {
        row["case_id"]: re.sub(r"_(conceptual|operational)$", "", row["case_id"])
        for row in read_jsonl(DATASET_C4)
    }


def mean_bool(items: list[dict[str, Any]], key: str) -> float:
    return fmean(float(bool(item[key])) for item in items) if items else 0.0


def metric_extractors(task: str) -> dict[str, Callable[[dict[str, Any]], float | None]]:
    if task == "A":
        return {
            "LOC": lambda j: mean_bool(j.get("objective_assessments", []), "lesson_covered"),
            "QOC": lambda j: mean_bool(j.get("objective_assessments", []), "qcm_assessed"),
            "A_FC": lambda j: float(j["factual_correctness"]["score"]),
            "PG": lambda j: float(j["pdf_grounding"]["score"]),
            "answer_validity": lambda j: float(j["answer_validity"]["score"]),
        }
    if task == "B":
        def eda(judgment: dict[str, Any]) -> float | None:
            diagnosis = judgment["error_diagnosis"]
            if not diagnosis["evaluable"]:
                return None
            return 1.0 if diagnosis["label"] == "correct" else 0.0

        return {
            "EDA": eda,
            "B_FC": lambda j: float(j["feedback_correctness"]["score"]),
            "RT": lambda j: float(bool(j["remediation_targeting"]["targeted"])),
            "CA": lambda j: float(j["contextual_adaptation"]["score"]),
            "uncertainty_calibration": lambda j: float(j["uncertainty_calibration"]["score"]),
            "grounding": lambda j: float(j["grounding"]["score"]),
        }
    return {
        "lesson_relation_quality": lambda j: float(j["lesson_relation_quality"]["score"]),
        "condition_adherence": lambda j: float(j["condition_adherence"]["score"]),
        "multi_kc_reasoning": lambda j: float(j["qcm_multi_kc_reasoning"]["score"]),
        # Exploratory only: the original C4 judge self-reported this F1.
        "kc_attribution_f1_exploratory": lambda j: float(j["kc_attribution"]["f1"]),
        "factual_correctness": lambda j: float(j["factual_correctness"]["score"]),
        "answer_validity": lambda j: float(j["answer_validity"]["score"]),
        "pdf_grounding": lambda j: float(j["pdf_grounding"]["score"]),
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def cluster_bootstrap_ci(
    pairs: list[tuple[str, str, float, float]],
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for _, cluster_id, value_a, value_b in pairs:
        grouped.setdefault(cluster_id, []).append((value_a, value_b))
    clusters = sorted(grouped)
    if not clusters:
        raise ValueError("No paired clusters available")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choices(clusters, k=len(clusters))
        differences = [
            value_b - value_a
            for cluster_id in sampled
            for value_a, value_b in grouped[cluster_id]
        ]
        estimates.append(fmean(differences))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def metric_pairs(
    case_ids: list[str],
    clusters: dict[str, str],
    records_a: dict[str, dict[str, Any]],
    records_b: dict[str, dict[str, Any]],
    extractor: Callable[[dict[str, Any]], float | None],
    missing_policy: str,
) -> tuple[list[tuple[str, str, float, float]], dict[str, int]]:
    pairs: list[tuple[str, str, float, float]] = []
    counts = {
        "candidate_cases": len(case_ids),
        "missing_or_error_a": 0,
        "missing_or_error_b": 0,
        "metric_not_evaluable": 0,
    }
    for case_id in case_ids:
        row_a = records_a.get(case_id)
        row_b = records_b.get(case_id)
        ok_a = row_a is not None and row_a.get("status") == "ok"
        ok_b = row_b is not None and row_b.get("status") == "ok"
        counts["missing_or_error_a"] += int(not ok_a)
        counts["missing_or_error_b"] += int(not ok_b)
        if missing_policy == "complete_case" and (not ok_a or not ok_b):
            continue
        try:
            value_a = extractor(row_a["judgment"]) if ok_a else 0.0
            value_b = extractor(row_b["judgment"]) if ok_b else 0.0
        except (KeyError, TypeError, ValueError):
            counts["metric_not_evaluable"] += 1
            continue
        if value_a is None or value_b is None:
            counts["metric_not_evaluable"] += 1
            continue
        cluster_id = clusters.get(case_id)
        if cluster_id is None:
            raise ValueError(f"No cluster ID for {case_id}")
        pairs.append((case_id, cluster_id, float(value_a), float(value_b)))
    return pairs, counts


def compare(args: argparse.Namespace) -> dict[str, Any]:
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    if run_a == run_b:
        raise ValueError("--run-a and --run-b must differ")
    manifest_a = read_json(run_a / "manifest.json")
    manifest_b = read_json(run_b / "manifest.json")
    selected_a = set(selected_ids(manifest_a, args.task))
    selected_b = set(selected_ids(manifest_b, args.task))
    shared_ids = sorted(selected_a & selected_b)
    if not shared_ids:
        raise ValueError("The two runs have no shared selected cases")
    records_a = latest(judgment_path(run_a, args.task))
    records_b = latest(judgment_path(run_b, args.task))
    clusters = cluster_map(args.task)
    extractors = metric_extractors(args.task)
    label_a = args.label_a or run_model_label(manifest_a, run_a.name)
    label_b = args.label_b or run_model_label(manifest_b, run_b.name)

    metrics: dict[str, Any] = {}
    for index, (metric, extractor) in enumerate(extractors.items()):
        pairs, missing = metric_pairs(
            shared_ids, clusters, records_a, records_b, extractor, args.missing_policy
        )
        if not pairs:
            metrics[metric] = {"status": "not_evaluable", **missing}
            continue
        values_a = [row[2] for row in pairs]
        values_b = [row[3] for row in pairs]
        difference = fmean(b - a for a, b in zip(values_a, values_b))
        low, high = cluster_bootstrap_ci(
            pairs, args.bootstrap_repetitions, args.seed + index
        )
        metrics[metric] = {
            "status": "ok",
            "mean_a": fmean(values_a),
            "mean_b": fmean(values_b),
            "paired_difference_b_minus_a": difference,
            "cluster_bootstrap_ci95": [low, high],
            "ci_excludes_zero": low > 0 or high < 0,
            "paired_cases": len(pairs),
            "clusters": len({row[1] for row in pairs}),
            **missing,
        }

    return {
        "analysis": "paired_cluster_bootstrap",
        "task": args.task,
        "model_a": label_a,
        "model_b": label_b,
        "difference_direction": "model_b_minus_model_a",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "selected_cases_a": len(selected_a),
        "selected_cases_b": len(selected_b),
        "shared_selected_cases": len(shared_ids),
        "clusters_for_shared_cases": len({clusters[cid] for cid in shared_ids}),
        "missing_policy": args.missing_policy,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "metrics": metrics,
        "notes": [
            "Cases are paired by case_id before comparison.",
            "Clusters, not individual rows, are resampled with replacement.",
            "C4 kc_attribution_f1_exploratory is judge-reported and is not validated gold-based F1.",
        ],
    }


def write_results(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_cluster_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for metric, value in result["metrics"].items():
        if value.get("status") != "ok":
            continue
        rows.append({
            "task": result["task"],
            "metric": metric,
            "model_a": result["model_a"],
            "mean_a": value["mean_a"],
            "model_b": result["model_b"],
            "mean_b": value["mean_b"],
            "difference_b_minus_a": value["paired_difference_b_minus_a"],
            "ci95_low": value["cluster_bootstrap_ci95"][0],
            "ci95_high": value["cluster_bootstrap_ci95"][1],
            "ci_excludes_zero": value["ci_excludes_zero"],
            "paired_cases": value["paired_cases"],
            "clusters": value["clusters"],
            "missing_or_error_a": value["missing_or_error_a"],
            "missing_or_error_b": value["missing_or_error_b"],
        })
    fields = list(rows[0]) if rows else ["task", "metric"]
    with (output_dir / "paired_cluster_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["A", "B", "C4"], required=True)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--missing-policy",
        choices=["complete_case", "zero"],
        default="complete_case",
        help="complete_case uses only cases successfully judged for both models; zero assigns failures zero.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100")
    result = compare(args)
    write_results(result, args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
