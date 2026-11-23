# ITS evaluation benchmark

This directory contains three synthetic evaluation tasks grounded in the French ENSOSP visual-doctrine source and its page-level instructional transcriptions.

> The gold annotations are draft research references. They require firefighter-trainer validation before publication or claims of benchmark validity.

## Tasks

| Task | Cases | Model-visible input | Generated output | Hidden gold |
|---|---:|---|---|---|
| **A — Initial teaching** | 300 | KC, doctrine excerpt and teaching requirements | Micro-lesson and QCM | Expected objectives, concepts and supported facts |
| **B — Error-aware remediation** | 300 | KC, doctrine excerpt, failed QCM, wrong answer and observation context | Corrective feedback and remedial QCM | Error label, cautious misconception hypothesis and remediation requirements |
| **C4 — Cross-KC generation** | 116 | Current KC, 1–3 related KCs, doctrine texts and conceptual/operational condition | Cross-KC lesson, QCM, explanation and used KC IDs | Draft integration target, condition requirement, source pages and redundancy warnings |

The generator receives only model-visible input. The independent judge receives the generated output, hidden gold and doctrine evidence.

## Main metrics

- **Task A:** Lesson Objective Coverage, QCM Objective Coverage, factual correctness, answer validity and PDF grounding.
- **Task B:** Error Diagnosis Accuracy, feedback correctness, remediation targeting, contextual adaptation, uncertainty calibration and grounding.
- **Task C4:** lesson relation quality, condition adherence, KC coverage, genuine multi-KC reasoning, KC attribution, factual correctness, answer validity and grounding.

For C4, the 116 rows represent 58 paired KC combinations. Report conceptual and operational results separately and cluster uncertainty estimates by KC combination.

## Requirements

From Git Bash:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENROUTER_API_KEY="YOUR_NEW_OPENROUTER_API_KEY"
```

Set only the keys required by the selected providers. Never commit either key. Revoke and replace any key pasted into a chat, terminal log, document or repository.

All commands below assume the terminal is inside the `evaluation` directory.

## Validate Tasks A and B

```bash
python run_experiment.py validate
```

## Task A command

```bash
python run_experiment.py run \
  --task A \
  --run-dir runs/task_A_pilot_5 \
  --limit 5 \
  --concurrency 2
```

## Task B command

```bash
python run_experiment.py run \
  --task B \
  --run-dir runs/task_B_pilot_5 \
  --limit 5 \
  --concurrency 2
```

For the full 300 cases of A or B, remove `--limit 5` and use a new run directory.

## Tasks A and B together

```bash
python run_experiment.py run \
  --task all \
  --run-dir runs/tasks_AB_pilot_5 \
  --limit 5 \
  --concurrency 2
```

Here, `--limit 5` means five cases from A and five cases from B.

## Task C4 commands

Five-case pilot:

```bash
python TASK_C4_AUDITED_116/run_c4_experiment.py run \
  --run-dir runs/c4_pilot_5 \
  --limit 5 \
  --concurrency 2 \
  --generator-model gpt-5.4-nano-2026-03-17 \
  --judge-model gpt-5.4-mini-2026-03-17
```

### Full Task C4 — all 116 cases

```bash
python TASK_C4_AUDITED_116/run_c4_experiment.py run \
  --run-dir runs/c4_full_gpt54nano \
  --concurrency 2 \
  --generator-model gpt-5.4-nano-2026-03-17 \
  --judge-model gpt-5.4-mini-2026-03-17
```

Omitting `--limit` selects all 116 C4 rows. The run is resumable: execute the same command again after a recoverable interruption. Use a different `--run-dir` when changing models.

The C4 runner rejects attempts to mix different generator or judge models in one run directory. Name the directory after the generator, for example `c4_full_gpt4omini` when using `gpt-4o-mini-2024-07-18`.

The full C4 experiment normally makes 116 generator calls and 116 judge calls, excluding retries.

## OpenRouter and open-source models

OpenRouter support is independent of the existing OpenAI path. Always use a new run directory for an OpenRouter experiment. Existing manifests without provider fields are treated as OpenAI runs and cannot be resumed with OpenRouter.

Replace `ORGANIZATION/MODEL_ID` with the exact OpenRouter model identifier.

### Tasks A and B: OpenRouter generator, OpenAI judge

Five cases per task:

```bash
python run_experiment.py run \
  --task all \
  --run-dir runs/openrouter_ab_pilot_5 \
  --limit 5 \
  --concurrency 2 \
  --generator-provider openrouter \
  --generator-model "ORGANIZATION/MODEL_ID" \
  --judge-provider openai \
  --judge-model gpt-5.4-mini-2026-03-17
```

Remove `--limit 5` and use another new directory for all 300 A cases and all 300 B cases.

### Task C4: OpenRouter generator, OpenAI judge

Five-case pilot:

```bash
python TASK_C4_AUDITED_116/run_c4_experiment.py run \
  --run-dir runs/openrouter_c4_pilot_5 \
  --limit 5 \
  --concurrency 2 \
  --generator-provider openrouter \
  --generator-model "ORGANIZATION/MODEL_ID" \
  --judge-provider openai \
  --judge-model gpt-5.4-mini-2026-03-17
```

Omit `--limit 5` and use another new directory for all 116 C4 cases.

To use OpenRouter for the judge too, set `--judge-provider openrouter` and provide its OpenRouter model ID. The default OpenRouter output mode is `json_schema`. If a selected model does not support it, retry in a **new run directory** with `--openrouter-output-mode json_object`; use `prompt_only` only as a final compatibility fallback because it provides weaker format enforcement.

## Command parameters

| Parameter | Meaning |
|---|---|
| `run` | Generate outputs, judge them and write the summary |
| `--task` | `A`, `B` or `all` for the shared A/B runner |
| `--run-dir` | Directory containing the manifest, generations, judgments and summary |
| `--limit` | Number of selected cases; omit for the complete task |
| `--concurrency` | Maximum simultaneous API calls |
| `--generator-model` | Model that generates the educational output |
| `--judge-model` | Independent model that evaluates the output |
| `--generator-provider` | `openai` (default) or `openrouter` |
| `--judge-provider` | `openai` (default) or `openrouter` |
| `--openrouter-output-mode` | `json_schema` (default), `json_object` or `prompt_only` |
| `--openrouter-site-url` | Optional OpenRouter attribution URL; can use `OPENROUTER_SITE_URL` |
| `--openrouter-app-name` | Optional OpenRouter application title; can use `OPENROUTER_APP_NAME` |
| `--max-attempts` | Maximum API attempts for a failed call |
| `--timeout` | Timeout per API request |
| `--seed` | Reproducible sampling seed for the A/B runner |

## Output files

Each run directory contains:

- `manifest.json`: providers, models, selected cases and configuration.
- `generations_*.jsonl`: generator outputs and validation results.
- `judgments_*.jsonl`: independent judge results.
- `summary.json`: aggregate scores.
- `summary.csv`: A/B tabular summary when using `run_experiment.py`.

## Implementation status and limitations

- Tasks A and B are implemented in `run_experiment.py`; validation currently reports 300/300 valid rows for each task.
- Task C4 is implemented separately in `TASK_C4_AUDITED_116/run_c4_experiment.py` and has been pilot-tested.
- Dataset A contains 25 KCs; Dataset B contains 50 distinct failed-question templates with six simulated observation profiles each. Rows are therefore not fully independent.
- Task B observation contexts are simulated summaries, not authentic learner-event logs.
- The C4 gold remains draft, strict all-KC necessity is not expert-certified, and KC-attribution precision/recall/F1 should ultimately be calculated deterministically from judged KC sets rather than estimated directly by the judge.
- The PDF is strongly visual; text-only evidence may not fully support image-dependent assessments. Perform blinded domain-expert validation on a stratified sample.
