"""Behavioral checks for ALLYN's bounded learning-to-teaching experiment."""

from __future__ import annotations

import unittest
from itertools import repeat

from allyn.teaching import compare_teaching


class TeachingComparisonTests(unittest.TestCase):
    def test_lesson_transfers_a_specific_observed_pattern(self) -> None:
        result = compare_teaching(
            student_examples=[("red", "bird", "sings")],
            lesson_examples=[("yellow", "bird", "chirps")] * 2,
            # Different complete sequence: test local suffix/backoff transfer,
            # not replay of an identical training example.
            heldout_examples=[("green", "bird", "chirps")],
            order=3,
        )

        self.assertEqual(result.evaluated_count, 1)
        self.assertEqual(result.baseline_accuracy, 0.0)
        self.assertEqual(result.taught_accuracy, 1.0)
        self.assertEqual(result.gain, 1.0)
        self.assertEqual(result.claim_ceiling, "OBSERVED_PATTERN_TRANSFER_ONLY")
        self.assertEqual(len(result.specimen_sha256), 64)
        self.assertIn("caller_supplied_unverified", result.source_labels)

    def test_no_lesson_is_no_gain_and_digest_is_stable(self) -> None:
        kwargs = dict(
            student_examples=[("sun", "will", "rise")],
            lesson_examples=[],
            heldout_examples=[("sun", "will", "rise")],
        )

        first = compare_teaching(**kwargs)
        second = compare_teaching(**kwargs)

        self.assertEqual(first.baseline_accuracy, first.taught_accuracy)
        self.assertEqual(first.gain, 0.0)
        self.assertEqual(first.specimen_sha256, second.specimen_sha256)
        self.assertEqual(first, second)

    def test_heldout_predictions_do_not_update_a_student(self) -> None:
        result = compare_teaching(
            student_examples=[("red", "bird", "sings")],
            lesson_examples=[],
            heldout_examples=[("green", "bird", "chirps")] * 2,
        )

        self.assertEqual(result.evaluated_count, 2)
        self.assertEqual(result.baseline_accuracy, 0.0)
        self.assertEqual(result.taught_accuracy, 0.0)

    def test_misleading_lesson_can_hurt_accuracy(self) -> None:
        result = compare_teaching(
            student_examples=[("sun", "will", "rise")],
            lesson_examples=[("sun", "will", "fall")] * 3,
            heldout_examples=[("sun", "will", "rise")],
        )

        self.assertEqual(result.baseline_accuracy, 1.0)
        self.assertEqual(result.taught_accuracy, 0.0)
        self.assertEqual(result.gain, -1.0)

    def test_rejects_empty_heldout_or_invalid_tokens(self) -> None:
        with self.assertRaises(ValueError):
            compare_teaching([("a", "b")], [], [])
        with self.assertRaises(ValueError):
            compare_teaching([("a", "b")], [], [("only",)])
        with self.assertRaises(ValueError):
            compare_teaching([("a", "b")], [], [("", "x")])

    def test_infinite_outer_iterable_is_bounded_before_materialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds 2048 examples"):
            compare_teaching(repeat(("a", "b")), [], [("a", "b")])

    def test_infinite_inner_iterable_is_bounded_before_materialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds 10000 tokens"):
            compare_teaching([repeat("a")], [], [("a", "b")])

    def test_rejects_oversized_token_before_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "oversized token"):
            compare_teaching([("a", "b")], [], [("a" * 4097, "b")])

    def test_rejects_oversized_heldout_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds 10000 tokens"):
            compare_teaching([("a", "b")], [], [repeat("a", 10001)])

    def test_rejects_aggregate_utf8_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "UTF-8 byte capacity exceeded"):
            compare_teaching([("a", "b")], repeat(("x" * 2000, "y" * 2000), 251), [("a", "b")])

    def test_rejects_aggregate_token_budget_across_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "total token capacity exceeded"):
            compare_teaching([repeat("a", 10000)], [repeat("a", 10000)], [("a", "b")])

    def test_rejects_unencodable_token_before_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid UTF-8 token"):
            compare_teaching([("a", "b")], [], [("a", "\ud800")])

    def test_rejects_order_before_consuming_stream(self) -> None:
        def explosive():
            raise AssertionError("input stream must not be touched")
            yield ("a", "b")

        with self.assertRaisesRegex(ValueError, "integer from 1 to 5"):
            compare_teaching(explosive(), [], [("a", "b")], order=6)


if __name__ == "__main__":
    unittest.main()
