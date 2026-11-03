"""Behavioral contracts for ALLYN's bounded statistical language learner."""

from __future__ import annotations

import copy
import json
import math
import unittest

from allyn.language import NGramLearner


class NGramLearnerTests(unittest.TestCase):
    def test_predicts_observed_context_and_breaks_ties_deterministically(self) -> None:
        learner = NGramLearner(order=3)
        learner.observe(["the", "cat", "sat"])
        learner.observe(["the", "cat", "slept"])

        probabilities = learner.next_distribution(["the", "cat"])
        self.assertEqual(set(probabilities), {"the", "cat", "sat", "slept"})
        self.assertTrue(math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-12))
        self.assertAlmostEqual(probabilities["sat"], probabilities["slept"])
        self.assertGreater(probabilities["sat"], probabilities["the"])
        self.assertEqual(learner.predict_next(["the", "cat"]), "sat")

    def test_smoothing_and_suffix_backoff_on_unseen_context(self) -> None:
        learner = NGramLearner(order=3)
        learner.observe(["blue", "sky"])
        learner.observe(["blue", "sea"])

        observed = learner.next_distribution(["blue"])
        unseen = learner.next_distribution(["unknown", "blue"])
        self.assertEqual(observed, unseen)
        self.assertEqual(list(unseen), ["blue", "sea", "sky"])
        self.assertTrue(all(0 < value < 1 for value in unseen.values()))
        self.assertAlmostEqual(sum(unseen.values()), 1.0)
        self.assertGreater(unseen["sea"], unseen["blue"])

        # Completely unknown contexts still receive a normalized unigram prior.
        self.assertEqual(learner.next_distribution(["absent", "unknown"]),
                         learner.next_distribution([]))

    def test_utterances_do_not_create_cross_boundary_transitions(self) -> None:
        learner = NGramLearner(order=2)
        learner.observe(["alpha", "beta"])
        learner.observe(["gamma", "delta"])
        self.assertEqual(learner.next_distribution(["beta"]), learner.next_distribution([]))
        self.assertNotEqual(learner.next_distribution(["alpha"]), learner.next_distribution([]))

    def test_empty_and_unigram_model(self) -> None:
        learner = NGramLearner(order=1)
        self.assertEqual(learner.next_distribution(["unknown"]), {})
        self.assertIsNone(learner.predict_next([]))
        learner.observe([])
        self.assertEqual(learner.snapshot()["rows"], [])
        learner.observe(["z", "a", "a"])
        self.assertEqual(learner.predict_next(["any", "context"]), "a")
        self.assertEqual(learner.next_distribution(["x"]), learner.next_distribution([]))

    def test_snapshot_json_roundtrip_and_restart_equivalence(self) -> None:
        learner = NGramLearner(order=4)
        for words in (["one", "two", "three", "four"], ["one", "two", "three", "five"]):
            learner.observe(words)

        snapshot = learner.snapshot()
        restored = NGramLearner.from_snapshot(json.loads(json.dumps(snapshot)))
        self.assertEqual(restored.snapshot(), snapshot)
        for context in ([], ["one"], ["one", "two"], ["one", "two", "three"], ["missing", "two"]):
            self.assertEqual(restored.next_distribution(context), learner.next_distribution(context))
            self.assertEqual(restored.predict_next(context), learner.predict_next(context))

        # Neither caller mutation nor a later model update may alias a snapshot.
        snapshot["rows"][0]["next"]["one"] = 999
        self.assertNotEqual(restored.snapshot(), snapshot)
        restored.observe(["extra", "example"])
        self.assertNotIn("extra", learner.next_distribution([]))

    def test_invalid_orders_and_observations_are_rejected_atomically(self) -> None:
        for order in (0, 6, True, 2.5, "3", None):
            with self.subTest(order=order), self.assertRaises((TypeError, ValueError)):
                NGramLearner(order=order)

        learner = NGramLearner()
        learner.observe(["valid", "sentence"])
        before = learner.snapshot()
        for tokens in ("a b", b"a b", ["good", ""], ["good", 3], ["good", None]):
            with self.subTest(tokens=tokens), self.assertRaises((TypeError, ValueError)):
                learner.observe(tokens)
            self.assertEqual(learner.snapshot(), before)

        for context in ("string", b"bytes", ["", "word"], [42]):
            with self.subTest(context=context), self.assertRaises((TypeError, ValueError)):
                learner.next_distribution(context)

    def test_rejects_malformed_or_untrusted_snapshots(self) -> None:
        learner = NGramLearner(order=2)
        learner.observe(["a", "b"])
        good = learner.snapshot()

        bad = []
        for field, value in (("schema", "wrong"), ("order", True), ("order", 7), ("rows", {})):
            item = copy.deepcopy(good)
            item[field] = value
            bad.append(item)

        duplicate = copy.deepcopy(good)
        duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0]))
        bad.append(duplicate)

        negative = copy.deepcopy(good)
        negative["rows"][0]["next"]["a"] = -1
        bad.append(negative)

        boolean_count = copy.deepcopy(good)
        boolean_count["rows"][0]["next"]["a"] = True
        bad.append(boolean_count)

        unknown_token = copy.deepcopy(good)
        unknown_token["rows"].append({"context": ["intruder"], "next": {"a": 1}})
        bad.append(unknown_token)

        impossible_count = copy.deepcopy(good)
        for row in impossible_count["rows"]:
            if row["context"] == ["a"]:
                row["next"]["b"] = 3
        bad.append(impossible_count)

        too_long = copy.deepcopy(good)
        too_long["rows"].append({"context": ["a", "b"], "next": {"a": 1}})
        bad.append(too_long)

        for value in bad + [None, [], {}, {**good, "surprise": "extra"}]:
            with self.subTest(snapshot=str(value)[:160]), self.assertRaises((TypeError, ValueError)):
                NGramLearner.from_snapshot(value)

    def test_snapshot_structural_checks_do_not_prove_training_history(self) -> None:
        # This count structure satisfies the implemented suffix inequality,
        # but cannot come from observed utterances: one "a" cannot produce
        # the two positions required for an a -> a bigram transition.
        forged = {
            "schema": "allyn.ngram.v1",
            "order": 2,
            "rows": [
                {"context": [], "next": {"a": 1}},
                {"context": ["a"], "next": {"a": 1}},
            ],
        }
        restored = NGramLearner.from_snapshot(forged)
        self.assertEqual(restored.snapshot(), forged)
        self.assertEqual(restored.predict_next(["a"]), "a")
        # Acceptance is partial structural validation, not corpus provenance,
        # authenticity, or a claim of complete count realizability.


if __name__ == "__main__":
    unittest.main()
