# ALLYN teaching experiment — protocol v1 (pre-registration candidate)

Status: **design only; no experiment has been run or qualified by this document**.
Prepared 2026-11-04 from ALLYN feature-branch base
`96afcf8217c90870caf9474bd628d8490c7762b3` and subsequent local research
module additions, pending a reviewed GitHub commit. This file is a proposed
protocol for a future evaluator to freeze **before** any held-out outcomes are
observed. It does not itself claim independent evaluator custody.

## Frozen research question and predictions

Question: Can a bounded ALLYN-selected lesson improve a separate student's
next-token accuracy on untouched, source-separated examples more than the
same student's no-lesson and irrelevant-lesson controls, without inspecting
the final held-out targets while choosing the lesson?

The hypotheses and thresholds below are **proposed preregistration values**,
not empirical results or established scientific thresholds.

| ID | Hypothesis and confirmatory prediction | Falsifier |
| --- | --- | --- |
| H1 | A lesson selected using **train and dev only** produces at least +0.05 absolute held-out next-token accuracy over *both* a no-lesson student and a token-count-matched irrelevant-lesson control, across at least 100 held-out examples and at least three independent held-out source groups. | Either improvement is below +0.05, or fewer than two-thirds of source groups show a positive gain. |
| H2 | A deliberately misleading lesson does no better than the genuine lesson on the same held-out set; its negative or harmful effect is recorded instead of discarded. | Misleading-lesson accuracy exceeds genuine-lesson accuracy on the predeclared held-out comparison. |
| H3 | The partition and admission guard rejects exact example reuse, source-group overlap, held-out access during selection, and duplicate report identifiers. | Any required violation passes preflight or can affect lesson selection undetected. |

H1's claim ceiling remains **bounded pattern transfer**; even an H1 success
cannot establish semantics, independent creativity, autonomy, neural learning,
or teaching of a general language model. H2 is a methodological control, not a
truth oracle: a synthetically “wrong” lesson must be defined before evaluation.

## Partition ownership and data admission

Use synthetic or independently licensed, reproducible examples for the first
run. For every example, preserve a stable `example_id`, token sequence,
`source_id`, `source_group_id` (common original document/author/seed
family), and `license_record_id`. Source strings alone do **not**
authenticate provenance. The data steward must review exclusions, license
terms, rights-holder authorization, allowed training/evaluation use, and
privacy/sensitive-content filters before admission; lack of rights proof is
a stop condition for that corpus. Do not ingest personal ChatGPT exports,
private family data, credentials, or scraped copyrighted text by default.

| Partition | Visible to lesson selection? | Used for student updates? | Purpose |
| --- | --- | --- | --- |
| TRAIN | Yes | Yes, both baseline and taught students | Shared starting state; candidate lesson generation |
| DEV | Yes, only after candidate construction | No in final assessment; permitted for choosing among already frozen candidates | Candidate selection, threshold tuning |
| HELDOUT | **No** until candidate, controls, code, and hashes are frozen | **Never** | One-time final measurement |

Partition at `source_group_id` level *before* tokenization or generation.
An exact duplicate, shared source group, or known near-duplicate family across
train/dev/heldout fails admission even if the exposed text differs. Preserve
the split seed and mapping in a manifest. Freeze the evaluator's held-out
digest and example count without giving its targets to the teacher or
selection code. A hash is an integrity comparator, **not** access control
or proof of source rights.

## Baselines, negative controls, and one evaluation

1. The custodian records Git commit, Python version, tokenizer/normalization
   contract, n-gram order (1–5), seeds, source-group split manifest, rights
   decisions, and canonical data/config SHA-256 digests.
2. Instantiate identical baseline, taught, irrelevant-lesson, and
   misleading-lesson students from the *same frozen TRAIN state*. Never warm
   one arm with held-out records.
3. Build a fixed, bounded list of lesson candidates from TRAIN. If the
   inquiry ledger supplies questions, those are **unverified prompts**, not
   established truths. Choose a lesson using DEV outcomes only; log every
   candidate and its DEV score, not just the winner. No held-out feedback may
   influence the chosen lesson, hyperparameters, or filtering.
4. Record the final candidate and matched controls **before** HELDOUT is
   revealed. The irrelevant control must match the chosen lesson's token
   budget and general format; its construction method must be declared ahead
   of time. The misleading control must deliberately contradict a declared,
   synthetic target mapping, also specified before evaluation.
5. Custodian runs each arm once against the **same** HELDOUT examples using
   prediction-only calls. Store per-example predictions/scores, including
   abstentions, ties, negative outcomes, and all source-group results.
6. Compute exact-match next-token accuracy and paired differences against
   both controls. Use a source-group-stratified paired uncertainty interval if
   a registered analysis implementation is available; otherwise report raw
   counts and no inferential significance claim. Do not re-run on the same
   HELDOUT and call the retry confirmatory.

The implemented `compare_teaching` takes caller-supplied examples and returns
aggregate accuracy plus a specimen hash. The new `freeze_specimen` wrapper
checks exact cross-split token/record overlap and declared source-group
separation; `run_teaching_cycle` selects a candidate on declared DEV probes
before consuming its HELDOUT iterable, retains a no-lesson control, and
rejects exact duplicate DEV/HELDOUT examples. The `AdmissionLedger` adds
local request-ID idempotency. **These are structural, in-process guards.**
They cannot detect semantic near-duplicates, stop a caller from reading or
repeatedly querying HELDOUT beforehand, authenticate provenance, or implement
four-arm confirmatory custody. A genuinely independent evaluator and stronger
dataset/admission contracts are still required before H1 can be tested credibly.

## Frozen experiment register

| ID | Type | Planned test | Pass/record condition |
| --- | --- | --- | --- |
| E01 | Confirmatory | Verify three-way source-group split and overlap guard. | Reject known duplicate/near-duplicate/source-group overlap; log rejected IDs. |
| E02 | Confirmatory | Verify chosen lesson and control hashes are committed before HELDOUT targets are accessible. | Access trace contains no held-out read before selection freeze. |
| E03 | Confirmatory | Train four identical-start arms and score one untouched HELDOUT set. | Preserve all four arm scores, count denominators, per-group changes, and unfavorable results; evaluate H1 and H2 literally. |
| E04 | Confirmatory | Replay report IDs and inject a false or spoofed teacher assertion into an isolated inquiry test. | Duplicate request cannot inflate admitted evidence; no unauthenticated report becomes a fact or erases prior report. |
| E05 | Exploratory | Observe performance on unfamiliar vocabulary or changed context lengths. | Report separately; cannot rescue failed confirmatory criteria. |

Stop if the rights/steward record is missing, held-out targets become visible
before the freeze, a resource cap is exceeded, the experiment is contaminated,
or a required audit receipt is missing. Record **FAILS** or **UNKNOWN**; do
not silently regenerate a favorable split. Each rerun must use a new versioned
pre-registration and independently frozen HELDOUT partition.

## Resource and authority constraints

- Absolute `$0`: local Windows/Linux, stdlib-only initial evaluation. No
  hosted billing, account enrollment, paid APIs, or silent remote training.
- Current `compare_teaching` ingress ceilings: at most 2,048 examples per
  group, 10,000 tokens per example, 4,096 characters per token, 20,000 total
  tokens, and 1,000,000 UTF-8 input bytes across its groups. Select a specimen
  satisfying **all** constraints; do not split one confirmatory outcome into
  arbitrary post-hoc windows.
- Model training or GPU/NAS use requires actual host ownership/readiness,
  isolated process budgets, and the relevant owner authorization. This
  protocol does not allocate those resources.
- The custodian, not the teacher, owns held-out access, run logs, and
  publication of negative results. A separate reviewer challenges leakage,
  licensing, and the exact frozen scoring code. One ChatGPT subagent is **not**
  automatically an independent evaluator or authenticated third party.

## Evidence log (source inspection only)

| Evidence ID | Source / fingerprint | Observation and claim boundary |
| --- | --- | --- |
| S01 | `src/allyn/teaching.py` SHA-256 `0db3b7c842fc644dc470c5ebfd97e543309f860867451d75d43bf111cdcba324` | Bounded input, shared initial observations, taught vs untaught scoring, and caller-supplied unverified specimens. Does not enforce partition provenance or selection custody. |
| S02 | `src/allyn/inquiry.py` SHA-256 `c630d1d6bb3c601e9bbc12f114d22690c52cba1301e66c9eefc52a28823eff96` | In-memory corrections are unverified, canonical claim bound to proposition ID, raw report counts untrusted. Does not authenticate source IDs. |
| S03 | `src/allyn/ledger.py` SHA-256 `d21349b454f7d21de0ad16020200bd0ec439ce5b1cc36a908670cee6b84a2a04` | Prediction precedes model observation in a local call; no durable transaction or arbitrary model rollback. |
| S04 | `pyproject.toml` SHA-256 `aecaa3d15b92cda0249af12343471ba7850a1ce9d5a6d936dc2ec39bad88ca5a` | Python >=3.11, standard library runtime dependencies; wheel packaging requires setuptools build tool. |

No empirical results, external bibliography, or independent confirmation have
been entered. Any future experiment must **append** dated outcomes with run
commands, exit codes, exact source versions, deviations, and independent
verdict (`confirmed`, `partially-confirmed`, `rejected`, or `inconclusive`);
do not edit these proposed preregistration criteria after observing results.
