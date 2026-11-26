# ALLYN qualification falsifiers — reproducible source-level counterexamples

These cases are executable tests, **not** adversarial penetration tests of a
separately controlled runtime. Run with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_qualification_falsifiers.py' -v
```

## F1: Perfect score, fabricated independence

`test_fabricated_source_groups_clear_scale_gate_without_custody` creates 100
distinct held-out token sequences assigned by their **caller** to three
different `source_id` strings. All 100 holdouts share the same target token,
which is repeated in the genuine lesson. A two-gram learner gets 100/100 in
the genuine arm and 0/100 in the no-lesson and irrelevant controls.

The specimen passes the declared size and group-count thresholds, but the
result reports `SCALE_ONLY_NOT_QUALIFIED`, appropriately. Neither the 100
records nor the three strings prove independent source families, unseen
answers, legal provenance, effective teaching of a neural model, or any
statistical generalization. The adversarial examples were constructed *for*
the experiment by the same caller.

## F2: Deferred consumption is not held-out custody

`test_caller_owned_heldout_generator_can_adapt_at_iteration_time` confirms
that `run_teaching_cycle` consumes DEV before it enumerates the HELDOUT
generator. The generator, owned by the same caller, then derives its target
from the caller's preselected winning lesson and yields a favorable answer.
This preserves the function's important internal ordering, while illustrating
that a caller who controls the held-out iterable can tailor its contents at
the moment of enumeration. The test **does not** demonstrate retroactive
selection leakage or an implementation bug in the selection order.

## Interpretation and next gate

The correct claim from these tests is that in-process structural checks and
digest receipts are helpful safeguards but cannot establish independent
evaluation or a genuine teaching outcome. For a confirmatory run:

1. Freeze candidate lessons, controls, model state, scoring code, protocol,
   split mapping, and rights approvals before held-out targets are exposed.
2. Assign held-out targets and score custody to an independently authorized
   evaluator with access boundaries that the teacher/caller cannot bypass.
3. Permit one pre-registered confirmatory score, preserving failed/negative
   outcomes and per-example four-arm predictions. Later adaptive queries are
   exploratory unless a newly protected held-out set is frozen.
4. Verify source-family independence and contamination beyond string IDs;
   exact/near-duplicate screening alone does not catch paraphrase families.

Operational separation on the same unrestricted OS account is not a
tamper-proof or independently authenticated evaluation boundary.

### Relevant primary research

- Dwork et al., [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629)
- Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- Yang et al., [Rethinking Benchmark and Contamination for Language Models with Rephrased Samples](https://arxiv.org/abs/2311.04850)
