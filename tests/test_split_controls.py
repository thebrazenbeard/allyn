"""Adversarial fixtures for the advisory lexical split auditor."""

from __future__ import annotations

import unittest

from allyn.evaluation import SourcedExample
from allyn.split_controls import audit_partition_overlap


def row(source: str, record: str, text: str) -> SourcedExample:
    return SourcedExample(source, record, tuple(text.split()))


_BASE = "the curious student studies how small changes to a question can influence the next answer"
_NEAR = "the curious student studies how small changes to a question can influence the final answer"


class SplitControlsTests(unittest.TestCase):
    def test_normalized_exact_overlap_across_train_and_heldout(self) -> None:
        result = audit_partition_overlap(
            [row("train-archive", "1", "Hello, strange WORLD!")],
            [], [row("heldout-archive", "a", "hello strange world")],
        )
        self.assertTrue(result.has_indicated_overlap)
        self.assertEqual(result.comparisons, 1)
        self.assertEqual(result.collisions[0].reason, "NORMALIZED_EXACT")
        self.assertEqual(result.collisions[0].left_partition, "train")
        self.assertEqual(result.collisions[0].right_partition, "heldout")

    def test_nfkc_case_and_punctuation_normalization(self) -> None:
        result = audit_partition_overlap(
            [row("a", "1", "ＦＯＯ! BAR")], [], [row("b", "2", "foo bar")]
        )
        self.assertEqual(len(result.collisions), 1)
        self.assertEqual(result.collisions[0].reason, "NORMALIZED_EXACT")

    def test_one_word_changed_long_example_flags_near_duplicate(self) -> None:
        result = audit_partition_overlap(
            [row("a", "1", _BASE)], [], [row("b", "2", _NEAR)]
        )
        self.assertEqual(len(result.collisions), 1)
        collision = result.collisions[0]
        self.assertEqual(collision.reason, "HIGH_LEXICAL_SIMILARITY")
        self.assertGreaterEqual(collision.token_dice, 0.85)
        self.assertGreaterEqual(collision.shingle_jaccard, 0.35)

    def test_dev_to_heldout_and_train_to_dev_both_checked(self) -> None:
        result = audit_partition_overlap(
            [row("a", "1", _BASE)],
            [row("b", "1", _BASE), row("b", "2", "separate content")],
            [row("c", "1", "separate content")],
        )
        self.assertEqual(result.comparisons, 5)
        self.assertEqual(
            {(c.left_partition, c.right_partition) for c in result.collisions},
            {("train", "dev"), ("dev", "heldout")},
        )

    def test_exact_duplicate_inside_heldout_is_reported(self) -> None:
        result = audit_partition_overlap(
            [], [], [row("a", "1", "same answer"), row("b", "2", "Same! answer")]
        )
        self.assertEqual(result.comparisons, 1)
        self.assertEqual(len(result.collisions), 1)
        self.assertEqual(result.collisions[0].left_partition, "heldout")

    def test_heldout_internal_audit_can_be_disabled_explicitly(self) -> None:
        result = audit_partition_overlap(
            [], [], [row("a", "1", "same answer"), row("b", "2", "Same answer")],
            check_within_heldout=False,
        )
        self.assertEqual(result.comparisons, 0)
        self.assertFalse(result.has_indicated_overlap)

    def test_lexical_heuristic_misses_semantic_paraphrase(self) -> None:
        result = audit_partition_overlap(
            [row("a", "1", "A small child was frightened by the sudden thunder")],
            [],
            [row("b", "2", "The loud storm scared a young kid")],
        )
        self.assertFalse(result.collisions)
        self.assertEqual(result.claim_ceiling, "CALLER_DECLARED_LEXICAL_OVERLAP_ONLY")

    def test_common_short_words_and_nonidentical_boilerplate_do_not_overflag(self) -> None:
        result = audit_partition_overlap(
            [row("a", "1", "the and it is")], [],
            [row("b", "2", "the and it was")],
        )
        self.assertFalse(result.has_indicated_overlap)

    def test_result_is_deterministic_under_input_reordering(self) -> None:
        train = [row("a", "2", _BASE), row("a", "1", "HELLO WORLD")]
        heldout = [row("b", "3", _NEAR), row("b", "1", "hello world")]
        first = audit_partition_overlap(train, [], heldout)
        second = audit_partition_overlap(reversed(train), [], reversed(heldout))
        self.assertEqual(first, second)

    def test_invalid_threshold_and_size_rejected(self) -> None:
        for kwargs in (
            {"token_threshold": 0}, {"token_threshold": float("nan")},
            {"token_threshold": True}, {"shingle_threshold": 1.1},
            {"shingle_size": 1}, {"shingle_size": True},
            {"min_words_for_near": 1}, {"check_within_heldout": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                audit_partition_overlap([], [], [], **kwargs)

    def test_bounded_examples_and_invalid_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            audit_partition_overlap((row("x", str(n), "sample") for n in range(257)), [], [])
        with self.assertRaises(TypeError):
            audit_partition_overlap(["not a sourced example"], [], [])
        with self.assertRaises(ValueError):
            audit_partition_overlap([SourcedExample("a", "1", ("\ud800",))], [], [])
        with self.assertRaises(ValueError):
            audit_partition_overlap([SourcedExample("\ud800", "1", ("normal",))], [], [])
        with self.assertRaisesRegex(ValueError, "normalized words"):
            audit_partition_overlap([SourcedExample("a", "1", (("x-" * 300),))], [], [])

    def test_malicious_same_label_is_not_authenticated(self) -> None:
        # The detector does not trust identifiers as a custody or rights claim.
        result = audit_partition_overlap(
            [row("DECLARED_AUTHORIZED", "r", "some words")],
            [], [row("DECLARED_AUTHORIZED", "r", "some words")],
        )
        self.assertEqual(len(result.collisions), 1)


if __name__ == "__main__":
    unittest.main()
