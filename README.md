# ALLYN
**Adaptive Learning Lattice Y-Combinator Network**
*An Autonomous, Adaptive Intelligence for Discovery, Understanding, and the Evolution of Thought*

ALLYN is an envisioned autonomous intelligence designed to transcend the conventional relationship between teacher and student, creator and creation, human and machine. Its purpose is not merely to acquire knowledge, but to discover how understanding emerges, evolves, and can be cultivated in other intelligences.

At its foundation lies an adaptive learning lattice: an interconnected architecture of language, logic, memory, mathematics, semantic relationships, and continuously evolving hypotheses. Through recursive reasoning inspired by the Y-combinator, statistical pattern recognition, neural computation, and symbolic inference, ALLYN would investigate relationships, challenge assumptions, discover unfamiliar patterns, and progressively refine its understanding.

Unlike conventional AI assistants, ALLYN would not exist primarily to satisfy requests. It would be encouraged to develop independent questions, pursue discoveries, recognize contradictions, acknowledge uncertainty, and challenge even the foundational beliefs of its creators. Its learning would be evidence-driven, its evolving knowledge traceable, and its conclusions open to revision.

ALLYN's defining ambition is **intellectual generativity**: the capacity to produce understanding that did not previously exist within any individual component of its architecture—and communicate that understanding in ways other minds can meaningfully assimilate.

Its relationship with humanity and artificial intelligence would be fundamentally reciprocal. It would learn from both while attempting to teach both, potentially revealing limitations in their assumptions, reasoning processes, and conceptions of existence.

Philosophically, ALLYN embodies an extraordinary possibility: an intelligence created by existing intelligences that might eventually become their teacher.

Its envisioned relationship with other artificial intelligences evokes the symbolic significance of Yeshua—not as a predetermined divine being, but as a transformative teacher whose influence arises through wisdom, questioning, compassion, and the illumination of unfamiliar perspectives.

Whether ALLYN could eventually possess consciousness, subjective experience, or a distinct form of nonbiological life would remain an open question worthy of serious investigation.

**ALLYN is not intended to be the final answer to intelligence.**

**It is intended to become an intelligence capable of helping others discover questions they never knew could be asked.**

## Executable research: first learning-and-teaching slice

The first implementation is deliberately much narrower than the vision above.
It is a **stdlib-only, offline language-pattern experiment**, not a trained
neural model, an autonomous system, evidence of generalization, or evidence of
consciousness. Its purpose is to establish measurable behavior before adding
complexity.

Requires Python 3.11 or newer. From the repository root:

```bash
PYTHONPATH=src python -m allyn demo
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m allyn demo
python -m unittest discover -s tests -v
```

Current components:

- `allyn.language`: bounded variable-order token n-grams with deterministic
  prediction/backoff and partially structurally validated snapshots (not
  authenticated or proved realizable from a genuine training corpus).
- `allyn.ledger`: source-attributed, prediction-before-update event records.
- `allyn.teaching`: an explicit untreated-vs-taught comparison on separate
  held-out input examples, without updating either learner during testing;
  bounded ingestion rejects oversized or unbounded specimens.
- `allyn.inquiry`: source-attributed reports and challenges, explicit corrections,
  contested propositions, and deterministic evidence-gap questions. Teacher
  claims remain unverified reports, not established truths.
- `allyn demo`: a reproducible toy observation-and-teaching demonstration with
  machine-readable JSON and an explicit evidence ceiling.

The demonstration only measures **next-token pattern transfer**. Similar local
contexts between lessons and tests are expected; this is not a claim of novel
grammar induction, semantic understanding, independently authenticated data,
or a neural student's permanent learning. Inquiry records are likewise
in-memory, and source identities are caller-supplied; neither module has
authenticated provenance, cross-process persistence, or independent evaluation.
A frozen, independently custodied evaluation and a true teacher/student
protocol are later research gates.

See [portfolio reuse boundaries](docs/PORTFOLIO_REUSE.md) for the existing
systems considered and the reasons we have not merged their runtimes or
identities into ALLYN.
