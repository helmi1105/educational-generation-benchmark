# Task C4 — Corrected cross-KC benchmark (116 cases)

## Audit result

- Total: 116 cases, original 72 two-KC, 28 three-KC, 16 four-KC cases preserved.
- **70 cases** contain at least one identifiable overlapping KC convention (conservative heuristic/semantic manual flag); 46 cases are **unverified**, not proven strict.
- **0 cases** are labelled as expert-certified strict all-KC necessity. This is intentional.
- Complete KC-level instructional **reconstructed transcriptions** are unchanged, not verbatim PDF OCR.
- The prompt no longer orders the model to list all KCs regardless of actual usage or fabricate impossible necessity.
- The 58 KC combinations now each have two explicit, meaningfully different conditions: `conceptual` and `operational`.
- Every input points to `OUTPUT_SCHEMA_C4.json`; reported `qcm.kc_ids` must be selected from the KCs supplied in that input.
- **All gold necessity claims have been replaced with review statuses.** This corrects the previous overclaim, not the underlying difficulty of constructing true all-KC necessity cases.

### Field semantics: `required_kcs_for_input`

`required_kcs_for_input` records the KCs that must be **supplied as input for the
predefined benchmark case**. It does **not** assert that every listed KC is
strictly necessary to answer a generated QCM. Strict necessity is represented
only by `kc_necessity_status`, `strict_all_kc_necessity_certified`, the
redundancy warnings, and the documented ablation protocol. Consequently,
`required_kcs_for_input` is compatible with `kc_necessity_status=unverified` or
`likely_redundant` and is not, by itself, an internal contradiction.

## What the benchmark can validly test now

Lesson relation quality, KC coverage, genuine multi-KC reasoning when demonstrated, correct answers and PDF grounding. Score necessary integration separately from topical overlap.

### Conditions

- `conceptual`: explain and assess meanings, relationships or distinctions without requiring an invented operational situation.
- `operational`: apply the concepts in a concrete map-reading or map-use scenario supported entirely by the supplied texts.

## What is not yet certified

All-KC strict necessity is **not established** for any case and should not be the headline statistic. Pair/triple/quadruple counts describe input load, not validated reasoning depth. Review visually derived conventions against the original 24-page PDF and obtain firefighter-expert validation of the gold targets and candidate ablation judgments.

## Ablation test for a generated question

For each KC, remove its text from the evidence; ask a blinded evaluator if the same question remains answerable **from the remaining supplied evidence**, with no outside knowledge. For stronger proof, enumerate every proper subset of source texts (up to four KCs: at most 14 nonempty proper subsets). If any subset suffices, do not claim all-KC necessity. Preserve QCM validity and avoid allowing option wording to expose the answer. When ambiguous, mark `uncertain`, not passing.

## Files

- `MODEL_INPUT_C4_116.jsonl` model-only inputs.
- `FULL_WITH_GOLD_C4_116.jsonl` full cases + draft annotations and flags.
- `AUDIT_OVERVIEW_C4_116.csv` per-case status/redundancy examples.
- `KC_TEXTS_27.json` original transcriptions (not changed).
- `OUTPUT_SCHEMA_C4.json` requires accurate KC attribution, not all KCs.
- `JUDGE_PROTOCOL_C4.json` judge rubric and ablation instructions.
- `repair_c4_dataset.ps1` reproducibly applies the condition/schema repair and checks all 58 pairs.
- `run_c4_experiment.py` runs a resumable generator + independent-judge pilot with gold hidden from the generator.

## Five-case pilot

From the repository root:

```bash
python evaluation/TASK_C4_AUDITED_116/run_c4_experiment.py run \
  --run-dir evaluation/runs/c4_pilot_5 \
  --limit 5 \
  --concurrency 2 \
  --generator-model gpt-5.4-nano-2026-03-17 \
  --judge-model gpt-5.4-mini-2026-03-17
```

The selection is the first five dataset rows in file order. The generated artifacts are `manifest.json`, `generations_C4.jsonl`, `judgments_C4.jsonl`, and `summary.json` inside the run directory.

## Study-design caveats

There are 58 underlying KC combinations, each evaluated under two controlled conditions. The paired cases are correlated, so report condition-level results and confidence intervals clustered by KC set; never treat the 116 rows as 116 independent KC combinations. Gold targets remain synthetic drafts pending domain expert review.
