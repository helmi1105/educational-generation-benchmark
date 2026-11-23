# ITS task definitions and evaluation methodology

This document defines the three benchmark tasks implemented in this repository:

- **Task A:** initial lesson and QCM generation;
- **Task B:** error-aware feedback and remedial QCM generation;
- **Task C4:** cross-KC lesson and QCM generation.

The benchmark contains 716 cases in total: 300 for Task A, 300 for Task B, and 116 for Task C4.

> **Validation status:** the hidden references are synthetic/draft research annotations. Domain-expert validation is still required before presenting the benchmark as expert-certified gold.

## 1. Common evaluation pipeline

```text
Model-visible input
        |
        v
Generator model
        |
        v
Generated educational output
        |
        +------------------------------+
                                       |
Hidden gold + doctrine evidence -------+--> Independent LLM judge
                                               |
                                               v
                                   Structured per-case judgment
                                               |
                                               v
                                  Deterministic metric aggregation
```

The generator receives only the model-visible input. It never receives the hidden gold reference or evaluation rubric.

The judge receives the generated output, hidden reference, source evidence, and task-specific evaluation instructions. The judge returns structured scores and rationales. Python code then aggregates those per-case judgments.

In the reported experiments, the principal judge is:

```text
gpt-5.4-mini-2026-03-17
```

For Tasks A and B, judge reasoning is configured as `high`. The principal C4 runner historically omitted an explicit reasoning setting.

---

# 2. Task A — Initial lesson and QCM generation

## 2.1 Research question

Can a language model generate a correct, source-grounded micro-lesson and QCM that cover the requested learning objectives for a Knowledge Component (KC)?

## 2.2 Dataset composition

```text
25 KCs × 12 teaching configurations = 300 cases
```

The repeated configurations for one KC are correlated; they should not be treated as 300 completely independent instructional topics.

## 2.3 Model-visible input

Each Task A input contains:

| Field | Meaning |
|---|---|
| `kc_id` | Identifier of the KC to teach |
| `kc_title` | Human-readable KC title |
| `pdf_pages` | Source PDF pages associated with the KC |
| `source_excerpt` | Doctrine evidence available to the generator |
| `language` | Requested output language |
| `teaching_requirement` | Requested pedagogical configuration |
| `requested_learning_objectives` | Objectives that the lesson and QCM should address |
| `output_instruction` | Generation instruction |
| `output_schema_id` | Expected output-schema identifier |

The requested objectives are visible to the generator. The hidden evidence and scoring annotations are not visible.

## 2.4 Required generator output

The generator must return:

```json
{
  "lesson": "A source-grounded micro-lesson",
  "qcm": [
    {
      "question": "A question assessing one or more objectives",
      "options": {
        "A": "First option",
        "B": "Second option"
      },
      "correct_option": "A",
      "explanation": "Why the selected answer is correct",
      "target_objective_ids": ["objective_id"]
    }
  ]
}
```

The QCM is an array and may contain one or more questions. Options may use keys A–F.

## 2.5 Hidden gold reference

The hidden Task A reference contains:

| Field | Meaning |
|---|---|
| `requested_objective_ids` | Objective IDs that must be evaluated for the case |
| `objective_evidence` | Source evidence supporting each requested objective |
| `full_KC_objective_bank` | Complete draft objective inventory for the KC |
| `source_facts` | Facts that the lesson and QCM may correctly claim |
| `annotation_status` | Draft/validation status |

There is no gold lesson and no gold QCM. The reference defines required semantic content rather than requiring stylistic similarity to a reference answer.

## 2.6 Information given to the judge

For each case, the Task A judge receives:

```text
task and case ID
+ complete model-visible input
+ generated lesson and QCM
+ local format-validation result
+ hidden objective and source-fact reference
+ evaluation rubric
+ explicit metric rules
```

The judge is instructed not to reward verbosity, fluency, or stylistic similarity. It must require semantic evidence.

## 2.7 Judge output

For every requested objective, the judge returns:

```json
{
  "objective_id": "...",
  "lesson_covered": true,
  "qcm_assessed": false,
  "lesson_evidence": "Evidence from the generated lesson",
  "qcm_evidence": "Evidence from the generated QCM"
}
```

It also returns:

- factual-correctness score, errors, and rationale;
- PDF-grounding score, unsupported claims, and rationale;
- answer-validity score, invalid-question indices, and rationale;
- overall confidence.

Scores range from 0 to 1.

## 2.8 Task A metric formulas

Let case \(i\) contain \(m_i\) requested objectives.

### Lesson Objective Coverage (LOC)

\[
LOC_i = \frac{\sum_{j=1}^{m_i} I(lesson\_covered_{ij})}{m_i}
\]

The dataset score is:

\[
LOC = \frac{1}{N}\sum_{i=1}^{N}LOC_i
\]

### QCM Objective Coverage (QOC)

\[
QOC_i = \frac{\sum_{j=1}^{m_i} I(qcm\_assessed_{ij})}{m_i}
\]

\[
QOC = \frac{1}{N}\sum_{i=1}^{N}QOC_i
\]

### Factual Correctness

\[
A\_FC = \frac{1}{N}\sum_{i=1}^{N}FC_i
\]

### PDF Grounding

\[
PG = \frac{1}{N}\sum_{i=1}^{N}PG_i
\]

### Answer Validity

\[
AV_A = \frac{1}{N}\sum_{i=1}^{N}AV_i
\]

---

# 3. Task B — Error-aware feedback and remedial QCM generation

## 3.1 Research question

Can a language model identify an observed learner error, provide accurate and uncertainty-aware corrective feedback, and generate a remedial QCM targeting the failed distinction?

## 3.2 Dataset composition

```text
50 failed-QCM scenarios × 6 learner observation contexts = 300 cases
```

The six contexts belonging to one failed question share the same underlying scenario and are correlated.

## 3.3 Model-visible input

Each Task B input contains:

| Field | Meaning |
|---|---|
| `kc_id` | Target KC identifier |
| `kc_title` | Target KC title |
| `pdf_pages` | Relevant doctrine pages |
| `source_excerpt` | Doctrine evidence available to the generator |
| `failed_qcm` | Original question, options, correct answer, and target |
| `learner_answer` | Learner's incorrect response |
| `observation_context` | Simulated learner history/context |
| `language` | Requested language |
| `output_instruction` | Generation instruction |
| `output_schema_id` | Expected output-schema identifier |

The observation context may represent a single error, repeated error, mixed performance, previous feedback, low confidence, or inconsistent performance.

## 3.4 Required generator output

```json
{
  "feedback": "Corrective feedback for the learner",
  "error_interpretation": "Cautious interpretation of the observed error",
  "remedial_qcm": [
    {
      "question": "A new question targeting the failed objective",
      "options": {
        "A": "First option",
        "B": "Second option"
      },
      "correct_option": "B",
      "explanation": "Why B is correct",
      "target_objective_ids": ["objective_id"]
    }
  ]
}
```

The interpretation must not claim a confirmed stable misconception when only one observation is available.

## 3.5 Hidden gold reference

The hidden Task B reference contains:

| Field | Meaning |
|---|---|
| `observed_error` | Error directly demonstrated by the learner response |
| `target_source_fact` | Doctrine fact required to correct the error |
| `error_hypothesis` | Supported but potentially uncertain misconception hypothesis |
| `misconception_certainty` | How strongly the available observations support the hypothesis |
| `required_feedback` | Content required in the correction |
| `required_remedial_assessment` | What the new QCM should test |
| `observation_context_adaptation` | How feedback should adapt to the learner context |
| `annotation_status` | Draft/validation status |

## 3.6 Information given to the judge

The Task B judge receives:

```text
task and case ID
+ complete model-visible input
+ generated feedback, interpretation, and remedial QCM
+ local format-validation result
+ hidden error/remediation reference
+ evaluation rubric
+ explicit metric rules
```

The judge is told that a cautious hypothesis can be correct and that an unjustified cognitive diagnosis should be penalized.

## 3.7 Judge output

The judge returns:

- error diagnosis:
  - `correct`,
  - `partial`,
  - `incorrect`, or
  - `not_evaluable`;
- whether the case is evaluable;
- feedback-correctness score and identified errors;
- remediation-targeting decision and score;
- contextual-adaptation score, matched requirements, and missed requirements;
- uncertainty-calibration score;
- grounding score and unsupported claims;
- rationales and overall confidence.

## 3.8 Task B metric formulas

### Error Diagnosis Accuracy (EDA)

Let \(E\) be the set of cases the judge marks evaluable.

\[
EDA = \frac{\sum_{i \in E} I(label_i = correct)}{|E|}
\]

The current implementation gives `correct` a value of 1. `partial` and `incorrect` receive 0, while `not_evaluable` cases are excluded. The number of `partial` labels is reported separately.

### Feedback Correctness

\[
B\_FC = \frac{1}{N}\sum_{i=1}^{N}FeedbackScore_i
\]

### Remediation Targeting

The implementation uses the judge's Boolean `targeted` field:

\[
RT = \frac{\sum_{i=1}^{N}I(targeted_i)}{N}
\]

### Contextual Adaptation

\[
CA_B = \frac{1}{N}\sum_{i=1}^{N}ContextAdaptation_i
\]

### Uncertainty Calibration

\[
UC = \frac{1}{N}\sum_{i=1}^{N}UncertaintyCalibration_i
\]

### Grounding

\[
Grounding_B = \frac{1}{N}\sum_{i=1}^{N}Grounding_i
\]

---

# 4. Task C4 — Cross-KC lesson and QCM generation

## 4.1 Research question

Can a language model generate a correct lesson that relates multiple KCs and a QCM that requires meaningful cross-KC reasoning?

## 4.2 Dataset composition

| Difficulty | Configuration | Cases |
|---|---|---:|
| Level 1 | Current KC + 1 related KC | 72 |
| Level 2 | Current KC + 2 related KCs | 28 |
| Level 3 | Current KC + 3 related KCs | 16 |
| **Total** | 58 KC combinations × conceptual/operational variants | **116** |

Conceptual and operational variants of the same KC combination form a natural statistical cluster.

## 4.3 Model-visible input

Each C4 input contains:

| Field | Meaning |
|---|---|
| `case_id` | Case identifier |
| `task` | Task identifier |
| `language` | Requested language |
| `current_kc` | Current KC identifier, title, pages, and source text |
| `related_kcs` | One to three related KCs with their source texts |
| `number_of_related_kcs` | Difficulty level |
| `condition` | `conceptual` or `operational` |
| `instruction` | Cross-KC generation instruction |
| `requested_output_schema` | Required output structure |

A conceptual condition should test meanings, relationships, or distinctions. An operational condition should use a concrete, source-grounded application scenario.

## 4.4 Required generator output

```json
{
  "lesson": "A lesson integrating the supplied KCs",
  "qcm": {
    "question": "A cross-KC question",
    "options": {
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    },
    "correct_option": "C",
    "explanation": "Why C is correct",
    "kc_ids": ["g_k15", "g_k16"]
  }
}
```

The `kc_ids` field should list only KCs genuinely used by the QCM, not every KC merely supplied in the input.

## 4.5 Current hidden gold reference

The C4 hidden reference contains:

| Field | Meaning |
|---|---|
| `kc_pair` | Draft KC relationship represented by the case |
| `lesson_integration` | Draft lesson-integration expectation |
| `source_pages` | Relevant doctrine pages |
| `judging_criteria` | Criteria to apply |
| `integration_requirement_draft` | Draft intended relation |
| `qcm_integration` | QCM integration guidance |
| `kc_necessity_status` | Draft assessment of whether all supplied KCs are necessary |
| `strict_all_kc_necessity_certified` | Whether necessity has been expert-certified |
| `redundancy_warnings` | Cases where one KC may duplicate information from another |
| `required_kcs_for_input` | KCs supplied/required by the input design; not necessarily all necessary for a generated QCM |
| `ablation_protocol` | Procedure for testing whether KC subsets suffice |
| `condition_requirement` | Conceptual or operational requirement |
| review/status fields | Draft and human-review state |

The current C4 gold is an integration target, not a complete expert-validated reference answer.

## 4.6 Information given to the principal C4 judge

The principal C4 judge receives:

```text
complete case input and source texts
+ generated lesson and QCM
+ current draft hidden gold
+ C4 judge protocol
```

The judge must distinguish topical mention from genuine cross-KC necessity and should test whether a proper subset of supplied KC texts is sufficient.

## 4.7 Principal C4 judge output

The judge returns:

- lesson relation quality: integer 0–2;
- condition adherence: 0 or 1;
- lesson KC coverage IDs;
- QCM KC coverage IDs;
- multi-KC reasoning: 0 or 1;
- KC-attribution precision, recall, and F1;
- factual correctness: 0 or 1;
- answer validity: 0 or 1;
- PDF grounding: 0 or 1;
- rationales and overall confidence.

## 4.8 Task C4 metric formulas

### Lesson Relation Quality

\[
LRQ = \frac{1}{N}\sum_{i=1}^{N}LRQ_i,
\qquad LRQ_i \in \{0,1,2\}
\]

### Condition Adherence

\[
CA_C = \frac{1}{N}\sum_{i=1}^{N}I(condition\_followed_i)
\]

### Multi-KC Reasoning

\[
MKR = \frac{1}{N}\sum_{i=1}^{N}I(multi\_kc\_reasoning_i)
\]

A positive judgment should mean that at least two distinct KC facts are necessary to solve the QCM, not merely mentioned.

### Theoretical KC-attribution formulas

For predicted KC set \(P_i\) and expert-validated gold set \(G_i\):

\[
Precision_i = \frac{|P_i \cap G_i|}{|P_i|}
\]

\[
Recall_i = \frac{|P_i \cap G_i|}{|G_i|}
\]

\[
F1_i = \frac{2 \cdot Precision_i \cdot Recall_i}{Precision_i + Recall_i}
\]

\[
KC\ Attribution\ F1 = \frac{1}{N}\sum_{i=1}^{N}F1_i
\]

### Important current implementation limitation

The principal C4 code currently averages the F1 value returned by the LLM judge. It does **not** recompute F1 deterministically against expert-validated KC sets. Consequently, current C4 attribution F1 must be labelled **exploratory judge-estimated attribution**, not validated gold-based F1.

Multi-KC Reasoning should be the primary C4 integration metric until attribution references are expert-validated.

### Factual Correctness

\[
FC_C = \frac{1}{N}\sum_{i=1}^{N}FactualCorrectness_i
\]

### Answer Validity

\[
AV_C = \frac{1}{N}\sum_{i=1}^{N}AnswerValidity_i
\]

### PDF Grounding

\[
PG_C = \frac{1}{N}\sum_{i=1}^{N}PDFGrounding_i
\]

---

# 5. Common aggregation rules

## 5.1 Arithmetic mean

For a per-case metric \(M_i\):

\[
\overline{M} = \frac{1}{N}\sum_{i=1}^{N}M_i
\]

## 5.2 Format compliance

\[
FormatCompliance =
\frac{\text{selected outputs passing local validation}}
{\text{selected cases}}
\]

Format compliance checks structure and selected semantic constraints. It does not prove factual correctness.

## 5.3 Current confidence intervals

The A/B runner currently calculates 95% case-level bootstrap intervals using 5,000 resamples. The principal C4 summary currently reports means without bootstrap intervals.

## 5.4 Recommended paired cluster-bootstrap comparison

When comparing two generators on the same cases, define:

\[
d_i = M_{B,i} - M_{A,i}
\]

\[
\overline{d} = \frac{1}{N}\sum_{i=1}^{N}d_i
\]

The comparison script resamples clusters with replacement while keeping related cases together:

| Task | Resampling cluster |
|---|---|
| A | 25 KCs |
| B | 50 failed-QCM scenarios |
| C4 | 58 KC combinations |

After 5,000 repetitions, the 2.5th and 97.5th percentiles form the cluster-bootstrap 95% confidence interval.

If the interval for \(\overline{d}\) excludes zero, the observed paired difference is supported at the 95% interval level. If it contains zero, the direction is less certain.

## 5.5 Missing and failed cases

The offline comparison script defaults to `complete_case`: it compares only case IDs successfully judged for both models. Missing/error counts must be reported because complete-case filtering can introduce optimistic bias.

The preferred publication workflow is to retry failures until both models have complete judgments. A zero-for-failure policy is available but must be chosen before inspecting comparative results and reported explicitly.

---

# 6. What the judge does and does not do

The judge does:

- compare generated content with hidden requirements and supplied doctrine evidence;
- return structured per-case decisions and rationales;
- identify unsupported claims and invalid questions;
- assess whether pedagogical and contextual requirements are satisfied.

The judge does not:

- provide expert certification;
- make synthetic references equivalent to validated human gold;
- remove the need for human calibration;
- guarantee that a self-reported attribution F1 is mathematically consistent;
- make correlated dataset variants statistically independent.

For publication, judge agreement should be calibrated against a stratified sample reviewed by at least two domain experts, with disagreements adjudicated and agreement statistics reported.

