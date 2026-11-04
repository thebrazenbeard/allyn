"""Structural source/record split protection for offline ALLYN tests."""

from __future__ import annotations

from dataclasses import replace
from itertools import count, repeat
import unittest

from allyn.evaluation import SourcedExample, evaluate_frozen, freeze_specimen


def row(source: str, record: str, *tokens: str) -> SourcedExample:
    return SourcedExample(source, record, tokens)


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.student = [row("student-source", "r1", "red", "bird", "sings")]
        self.lesson = [row("teacher-source", "r1", "yellow", "bird", "chirps")]
        self.heldout = [row("independent-source", "r1", "green", "bird", "chirps")]

    def test_frozen_evaluation_reports_pattern_gain_with_evidence_ceiling(self) -> None:
        frozen = freeze_specimen(self.student, self.lesson, self.heldout)
        receipt = evaluate_frozen(frozen)
        self.assertEqual(receipt.transfer.baseline_accuracy, 0.0)
        self.assertEqual(receipt.transfer.taught_accuracy, 1.0)
        self.assertEqual(receipt.transfer.gain, 1.0)
        self.assertEqual(receipt.transfer.evaluated_count, 1)
        self.assertEqual(receipt.specimen_sha256, frozen.specimen_sha256)
        self.assertEqual(len(frozen.specimen_sha256), 64)
        self.assertIn("NOT_INDEPENDENT_CUSTODY", frozen.claim_ceiling)

    def test_duplicate_exact_heldout_is_rejected(self) -> None:
        repeated = self.heldout + [row("other-source", "r2", "green", "bird", "chirps")]
        with self.assertRaisesRegex(ValueError, "duplicate exact"):
            freeze_specimen(self.student, self.lesson, repeated)

    def test_exact_example_cross_partition_rejected_even_with_different_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact example overlap"):
            freeze_specimen(self.student, [row("different", "new", "red", "bird", "sings")], self.heldout)

    def test_record_id_overlap_with_distinct_tokens_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "record ID overlap"):
            freeze_specimen(
                self.student,
                [row("student-source", "r1", "a", "totally", "different")],
                self.heldout,
                allow_shared_source_groups=True,
            )

    def test_source_group_overlap_fails_closed_by_default_and_can_be_explicitly_relaxed(self) -> None:
        reuse = [row("student-source", "r2", "another", "bird", "chirps")]
        with self.assertRaisesRegex(ValueError, "source-group overlap"):
            freeze_specimen(self.student, reuse, self.heldout)
        frozen = freeze_specimen(
            self.student, reuse, self.heldout, allow_shared_source_groups=True
        )
        self.assertTrue(frozen.allow_shared_source_groups)

    def test_digest_is_deterministic_when_order_reversed_and_input_detached(self) -> None:
        student = self.student + [row("student-source", "r2", "purple", "dog", "barks")]
        original = freeze_specimen(student, self.lesson, self.heldout)
        reversed_input = freeze_specimen(list(reversed(student)), self.lesson, self.heldout)
        self.assertEqual(original, reversed_input)
        student.clear()
        self.assertEqual(len(original.student), 2)
        self.assertIsInstance(original.student[0].tokens, tuple)

    def test_forged_public_constructor_is_revalidated_on_use(self) -> None:
        good = freeze_specimen(self.student, self.lesson, self.heldout)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            evaluate_frozen(replace(good, specimen_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "exact example overlap"):
            evaluate_frozen(replace(good, heldout=self.student[0:1]))

    def test_infinite_outer_input_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds 2048"):
            freeze_specimen(
                (row("student-source", str(i), "a", "b") for i in count()),
                self.lesson,
                self.heldout,
            )

    def test_infinite_inner_input_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds 10000"):
            freeze_specimen(
                [SourcedExample("student-source", "r1", repeat("token"))],
                self.lesson,
                self.heldout,
            )

    def test_invalid_input_types_and_labels_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "order"):
            freeze_specimen(self.student, self.lesson, self.heldout, order=True)
        with self.assertRaisesRegex(ValueError, "allow_shared_source_groups"):
            freeze_specimen(self.student, self.lesson, self.heldout, allow_shared_source_groups=1)
        with self.assertRaisesRegex(ValueError, "source_id"):
            freeze_specimen([row("", "r1", "a", "b")], self.lesson, self.heldout)
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            freeze_specimen([row("source", "r1", "a", "\ud800")], self.lesson, self.heldout)
        with self.assertRaises(TypeError):
            freeze_specimen(["a b"], self.lesson, self.heldout)

    def test_wrong_teacher_can_reduce_score_without_contaminating_heldout(self) -> None:
        baseline = [row("student", "r1", "sun", "will", "rise")]
        wrong = [row("teacher", str(i), "sun", "will", "fall") for i in range(3)]
        heldout = [row("heldout", "r1", "star", "will", "rise")]
        receipt = evaluate_frozen(freeze_specimen(baseline, wrong, heldout))
        self.assertEqual(receipt.transfer.baseline_accuracy, 1.0)
        self.assertEqual(receipt.transfer.taught_accuracy, 0.0)
        self.assertEqual(receipt.transfer.gain, -1.0)


if __name__ == "__main__":
    unittest.main()
