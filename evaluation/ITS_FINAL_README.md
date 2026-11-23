# ITS evaluation datasets — final testing package (2026-10-27)

## Source provenance
The uploaded 24-page ENSOSP PDF, ECG JSON and user-supplied assistant-created page audit supplied the facts. All 25 ECG KCs used in the earlier scoped datasets are retained. Source excerpts are KC-specific and consistent across tasks. Visual pages require human subject-matter review before being treated as normative gold. The page audit is not itself an expert-certified source.

## Contents
- ITS_FINAL_A_300.jsonl and .csv — 300 controlled teaching cases, 25 KCs × 12 instructional configurations.
- ITS_FINAL_B_300.jsonl and .csv — 300 simulated remediation cases, 25 KCs × 2 question templates × 6 learner-observation profiles.
- ITS_MODEL_INPUT_A_300.jsonl and ITS_MODEL_INPUT_B_300.jsonl — *ONLY* generator-visible input; hidden references omitted to avoid leakage.

## Task A
Input: `kc_id`, `kc_title`, `pdf_pages`, `source_excerpt`, `teaching_requirement`, `requested_learning_objectives`. Output expected: lesson + QCM, no fixed count. Hidden reference: draft objective-to-source evidence, full fact bank. Evaluate lesson objective coverage, QCM objective coverage, answer validity, grounding and coherence. Requested learning objectives are *part of the task input*; hidden reference lists source evidence rather than gold lessons/questions.

## Task B
Input: KC, PDF evidence, failed MCQ with correct answer, chosen wrong answer, limited *simulated* observation context. Output expected: feedback + new remedial QCM. Hidden reference: observed error, plausible confusion and remediation requirements. A wrong answer does NOT establish a unique misconception diagnosis.

## Critical limitations
- These are **synthetic pilot/test cases**, not expert-validated research gold.
- A uses 25 KCs, repeated in 12 controlled configurations; B uses 50 underlying distinct failure questions, each repeated in six simulated history contexts. Treat repetitions as correlated when reporting uncertainty.
- B's synthetic observations do not assert genuine learning behavior. The profile descriptions are scenario constraints, not empirical learner histories.
- Some tasks reference visual-symbol conventions. A text-only LLM should receive the text transcription and should not be evaluated for image perception.
- Generated lessons, QCMs and feedback are intentionally blank. Never provide `reference_hidden_from_generator` to the generator.
- For a defensible publication: independent trainer validation, expanded distinct failure questions and/or real learner answers, separate annotation from evaluation, leakage-aware splitting at KC level, baseline comparisons, and human cross-checks of judges.

## Example run
Use each line from `ITS_MODEL_INPUT_A_300.jsonl` or `ITS_MODEL_INPUT_B_300.jsonl` as model input. Save predictions keyed by `case_id`, then evaluate against the matching hidden reference in the full dataset.
