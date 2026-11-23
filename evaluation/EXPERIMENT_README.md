# Reproducible Task A / Task B experiment

This runner performs two isolated stages:

1. The **generator** receives only `ITS_MODEL_INPUT_A_300.jsonl` or
   `ITS_MODEL_INPUT_B_300.jsonl`.
2. The **judge** receives the saved generation, PDF/source evidence, hidden
   reference requirements, and the evaluation rubric.

The default pinned models are:

- Generator: `gpt-5.4-nano-2026-03-17`, reasoning effort `none`.
- Judge: `gpt-5.4-mini-2026-03-17`, reasoning effort `high`.

`gpt-4.1-nano-2025-04-14` is not used because OpenAI shut it down on
2026-10-23. If you already possess outputs produced by that snapshot, they can
be converted to the generation JSONL format and judged separately.

## Setup

Use Python 3.10 or later and an OpenAI Python SDK that supports the Responses
API. The repository's `server/requirements.txt` already includes `openai`.

In PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python evaluation/run_experiment.py validate
```

The script never writes the API key to the run directory.

## Small pilot first

```powershell
python evaluation/run_experiment.py run `
  --task all `
  --run-dir evaluation/runs/pilot_01 `
  --limit 5 `
  --concurrency 2
```

Inspect `generations_A.jsonl`, `generations_B.jsonl`, `judgments_A.jsonl`,
`judgments_B.jsonl`, and `summary.json` before launching the complete run.

## Complete experiment

```powershell
python evaluation/run_experiment.py run `
  --task all `
  --run-dir evaluation/runs/full_01 `
  --concurrency 5
```

Running the same command again resumes incomplete/error cases. The manifest
prevents accidental model-setting changes inside an existing run.

Generation uses JSON mode because the dataset schemas contain dynamic option
keys (`A` through `F`), which are outside the safest subset for strict
Structured Outputs. Every response is validated locally. Judge responses use a
fixed strict JSON schema.

An API failure is not silently excluded: if no usable generation exists, the
case receives deterministic zero scores during judging. A parsed but
schema-invalid generation remains visible to the judge and also reduces the
reported format-compliance score.

## Metrics

Task A:

- `LOC`: lesson objective coverage.
- `QOC`: QCM objective coverage.
- `A_FC`: lesson and QCM factual correctness.
- `PG`: PDF/source grounding.
- Answer validity is reported as a safeguard.

Task B:

- `EDA`: exact error-diagnosis accuracy over evaluable cases.
- `B_FC`: feedback correctness.
- `RT`: binary remediation-targeting rate.
- `CA`: contextual adaptation.
- Uncertainty calibration and grounding are reported as safeguards.

`summary.json` and `summary.csv` include case-level bootstrap 95% confidence
intervals. The synthetic draft gold annotations must still be validated by
domain experts before publication. Treat scores as final only when each task's
`complete` field is `true`; otherwise resume the failed judge calls.

## Human-validation worksheet

After the run:

```powershell
python evaluation/run_experiment.py human-sample `
  --run-dir evaluation/runs/full_01 `
  --per-task 30
```

This creates `human_validation_sample.csv` with empty columns for two expert
ratings and adjudication notes. LLM-judge decisions are intentionally excluded
from this worksheet to prevent anchoring. Reviewers should also be blinded to
the generator identity when the worksheet is distributed. Join the completed
worksheet back to `judgments_A.jsonl` and `judgments_B.jsonl` by `case_id` only
after expert ratings have been locked.

## Separate stages

The stages can also be resumed independently:

```powershell
python evaluation/run_experiment.py generate --task A --run-dir evaluation/runs/full_01
python evaluation/run_experiment.py judge --task A --run-dir evaluation/runs/full_01
python evaluation/run_experiment.py summarize --run-dir evaluation/runs/full_01
```

When resuming, pass the same model/reasoning flags recorded in `manifest.json`.
