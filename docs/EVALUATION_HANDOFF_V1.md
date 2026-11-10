# ALLYN evaluation handoff v1 — source-only, $0, not qualified

**Status:** implementation handoff for the local Python 3.11+ prototype, not an
installed independent evaluator or a successful teaching claim. This contract
was checked against `src/allyn/protocol.py`, `teaching.py`, `experiment.py`,
`four_arm_custodian.py`, `custodian.py`, `student_exchange.py`, and
`experiment_report.py` on the `feat/learning-inquiry-prototype-v0` branch.

The scope is **one toy n-gram four-arm comparison**: unchanged baseline,
preselected genuine lesson, irrelevant-placebo lesson, and misleading lesson.
No model weights, network, paid services, GPU leases, or hosted backend are used.
The caller supplies all lessons, declared source references, and ground truth;
none is independently authenticated by these modules. H1/H2 remain **UNKNOWN**.

## 1. DEV-only selection, with no HELDOUT in the teacher process

1. An evaluator/researcher independently prepares TRAIN, DEV and HELDOUT
   source-family partitions and separately verifies actual license/data-steward
   permission. `allyn.provenance.build_manifest` checks only *declarations* and
   source-family split consistency. `allyn.split_controls.audit_partition_overlap`
   can flag normalized exact/near-duplicate leakage, not prove semantic
   independence. Neither grants legal rights or authenticates a source ID.
2. Give the **teacher** only already-admitted TRAIN examples, DEV selection
   probes, lesson candidates and their declared source references. Never give
   it HELDOUT rows, targets, source IDs, scores, specimen digest, or a path to
   evaluator-only data.
3. For DEV-only selection, the existing callable is
   `allyn.teaching.compare_teaching(student_examples, lesson_examples,
   heldout_examples, order=3)` with **DEV probes passed in its third argument**.
   That argument is named `heldout_examples` for the general comparison
   function but, in this step, holds **DEV** only. Run it for each candidate
   using the same TRAIN/DEV and a no-lesson control (`lesson_examples=()`).
   `TransferResult.gain` is a DEV-pattern difference, not an independent test.
4. Choose once, before any evaluator handoff. `run_teaching_cycle` provides
   deterministic tie-breaking: greatest DEV `gain`, then greatest DEV
   `taught_accuracy`, then lexicographically smallest `candidate_id`; **abstain**
   unless the best `gain` exceeds the no-lesson gain (zero). Store the chosen
   candidate ID, exact lesson tokens, controls, order, source refs, and a local
   digest/commit of those *public* inputs for reproducibility. Those digests
   are not digital signatures or rights evidence.

**Important API boundary:** `allyn.protocol.run_teaching_cycle(...)` itself
accepts `heldout_examples` and scores that final set after candidate selection.
It is a useful **source-only demonstration**, but not a DEV-only custody
handoff API: do **not** pass private HELDOUT into a teacher-controlled call to
it. There is currently no integrated `select_on_dev_only()` public API. The
DEV-only use of `compare_teaching()` above is the available building block.

## 2. Freeze the four arms *before* evaluator custody receives HELDOUT

- Freeze all four declared **training configurations**: the same original
  student observations for each arm; no lesson for baseline; the selected
  genuine lesson; the irrelevant control; and the misleading control. Fix
  their exact tokens, run IDs, order (1–5), source refs, and selection receipt
  before opening HELDOUT. No tuning or replacing lessons after seeing a score.
- Use distinct source families across every partition. `freeze_four_arm_specimen`
  rejects exact cross-partition overlaps and source-family reuse, but not
  semantic paraphrases or dishonest labels. It requires the **irrelevant**
  lesson to have the same total token count as the genuine lesson. It does
  **not** currently require a size-matched misleading lesson. Require that
  match in the experiment plan yourself; `audit_four_arm()` now reports
  `misleading_budget_matched`, and an unequal control cannot support a fair H2
  contrast. Equal token counts still cannot match difficulty or information.
- Do **not** call `freeze_four_arm_specimen()` in the teacher process before
  handoff: its signature includes the HELDOUT examples. Freeze the four
  training/lesson inputs without HELDOUT, transfer them to a separately
  operated evaluator, and have that evaluator supply its HELDOUT data and
  invoke the four-arm freeze/score *once*.
- Agree on a single evaluator-owned state directory **before** scoring and
  preserve it across runs. Claims are advisory local files, not a trusted
  cross-account service. Pre-register sample size, source-group policy,
  controls, analysis, and stop rules *outside* this prototype. The current
  100-heldout/3-declared-group size gate is necessary only for its scale label,
  never sufficient for qualification.

## 3. Exact one-shot four-arm evaluator API and CLI

**Python API** (evaluator-side only):

```python
from allyn.evaluation import SourcedExample
from allyn.experiment import freeze_four_arm_specimen
from allyn.four_arm_custodian import evaluate_four_arm_once

# All five arguments are iterables of SourcedExample(source_id, record_id,
# tokens_tuple). The evaluator, not the teacher, must fill in private heldout.
specimen = freeze_four_arm_specimen(
    student, genuine, irrelevant, misleading, heldout, order=3
)
receipt = evaluate_four_arm_once(
    specimen, state_dir="EVALUATOR_ONLY_PERSISTENT_STATE", request_id="trial-001"
)
```

**CLI**: `python -m allyn.four_arm_custodian --state-dir DIR --request-id ID`
reads **one** UTF-8 JSON request on stdin (maximum 2,000,000 bytes), returning
a JSON receipt on stdout. Run from a source checkout with `PYTHONPATH=src`
(or install the package). Example Linux shell command, run by the *evaluator*
after a private request file has been prepared:

```bash
PYTHONPATH=src python3 -m allyn.four_arm_custodian \
  --state-dir /path/owned/by/evaluator/state \
  --request-id trial-001 < /path/owned/by/evaluator/request.json
```

Exact accepted top-level JSON keys (no extra keys): `schema`, `student`,
`genuine`, `irrelevant`, `misleading`, `heldout`, `order`. Every partition is a
list of objects with exactly `source_id` (string), `record_id` (string), and
`tokens` (list of at least two nonempty strings). This is a **public toy
shape example**, **not** a request containing confidential answer keys:

```json
{
  "schema": "allyn.four-arm-custody-request.v0",
  "student": [
    {"source_id": "train-01", "record_id": "s1", "tokens": ["red", "bird", "sings"]}
  ],
  "genuine": [
    {"source_id": "lesson-01", "record_id": "g1", "tokens": ["yellow", "bird", "chirps"]}
  ],
  "irrelevant": [
    {"source_id": "placebo-01", "record_id": "i1", "tokens": ["river", "flows", "down"]}
  ],
  "misleading": [
    {"source_id": "misleading-01", "record_id": "m1", "tokens": ["blue", "bird", "barks"]}
  ],
  "heldout": [
    {"source_id": "evaluator-group-01", "record_id": "h1", "tokens": ["green", "bird", "chirps"]}
  ],
  "order": 3
}
```

`request_id` is a safe 1–64-character ASCII identifier (`[A-Za-z0-9_-]`
initial; remaining letters/numbers/underscore/dot/hyphen). A second attempt
with the same request ID or exact HELDOUT tokens in the **same cooperative
state directory** returns exit code 2 and a generic stderr rejection. Failed
or partially reserved evaluations can consume their trial, so fix validity
*before* supplying the real key. A new directory or even a modified subset
can defeat the replay guard; do not call it tamper-proof.

The receipt schema is `allyn.four-arm-custody-receipt.v0`, with
`request_id`, `status`, `schema`, `claim_ceiling`, `evaluated_count`,
`heldout_group_count`, `size_gate`, four `arms` of `ArmScore`
(`arm`, `correct`, `evaluated_count`, `accuracy`, `gain_vs_baseline`) and
anonymized `groups` with ordinal `group_index` and their four arm scores.
It omits HELDOUT tokens/source IDs, record IDs, and full specimen SHA-256,
but aggregate scores—especially groups of one—can still reveal information.
Only provide the minimum agreed aggregate receipt to the teacher, if any,
**after** the experiment is over; do not use it to choose or retune this trial.

## 4. Researcher-only paired audit is separate from teacher feedback

`allyn.experiment_report.audit_four_arm(specimen)` generates per-example
booleans, signed baseline-relative changes, group scores, control diagnostics,
lesson token budgets, and unmet H1/H2 qualification gates. It **re-evaluates
the public `FourArmSpecimen`** and exposes per-example source/record IDs and a
specimen digest. Keep both the specimen and this report inside the
evaluator/researcher scope. **Never send this report or the private specimen
to the teacher**, and never run it as an adaptive lesson-selection oracle.

For a deliberately public example only:

```bash
PYTHONPATH=src python3 examples/synthetic_four_arm_audit_demo.py
```

The example's answers are public, so its apparent gains cannot support
independent generalization. Real H1/H2 results are **UNKNOWN**.

## 5. External student mismatch: NOT IMPLEMENTED

`allyn.student_exchange.run_student_exchange(...)` is a separate orchestration
contract. Its `evaluate_heldout(state, phase)` callback is called twice:
once for `"baseline"` before `apply_lesson` and again for `"post"` after
the update. `HeldoutAssessment` links each declared model-state hash,
evaluator ID, and count. The returned `UpdateReceipt.kind` can be
`"NO_CHANGE"`, `"CONTEXT_UPDATE"`, or `"WEIGHT_UPDATE"`, but the code does
not inspect actual neural weights or authenticate those digests.

That two-phase callback **cannot** be connected to
`four_arm_custodian.evaluate_four_arm_once` by making two calls on the same
HELDOUT. The first call consumes the exact-token claim; the second should be
rejected. Nor can an external student's four different saved weight states be
passed as four lesson lists: the custodian currently constructs its own
`NGramLearner` instances and does not accept external model snapshots.

**Required future integration:** freeze and authenticate both baseline/post
external state handles, four arm assignments, evaluator identity, and one
HELDOUT key **before** evaluation; submit all of them to one custodian request
that evaluates the same key across all frozen states in one permitted pass,
then returns paired aggregates without teacher-visible answers. Build a new
explicit protocol/API, process/data separation and fail-closed replay
semantics, and test it adversarially. **None of that is implemented here.**

> **Hostile challenge:** A teacher and evaluator running under the same local
> account can inspect or modify each other's files, feed themselves held-out
> answers, alias source groups, or start a new state directory. Separate
> processes and SHA-256 receipts do not establish independent custody.

> **Hostile challenge:** Even the four-arm aggregate report reveals outcome
> information. Repeated adaptive trials, tiny groups, unbalanced misleading
> controls, or data selection after scoring can fabricate a positive-looking
> result without meaningful learning.

The operating budget is **$0**. No hosted Supabase/SQL service, private
third-party evaluator, token, account, daemon, or Windows host action has been
created or authorized by this source-only handoff. H1/H2 qualification stays
**UNKNOWN** until actual independent source-rights custody, preregistered
evaluation, larger held-out groups, adversarial leakage controls and real
student-state readback are independently established.
