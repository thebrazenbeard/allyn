"""Four-arm pattern evaluation with explicit untrusted-data ceilings."""

from __future__ import annotations

from dataclasses import replace
from itertools import count
import unittest

from allyn.evaluation import SourcedExample
from allyn.experiment import (
    CLAIM_CEILING,
    evaluate_four_arm,
    freeze_four_arm_specimen,
)


def sample(*, correct_start: bool = False):
    initial_target = "chirps" if correct_start else "sings"
    student = [SourcedExample("student", "s1", ("red", "bird", initial_target))]
    genuine = [
        SourcedExample("genuine", f"g{i}", ("yellow", "bird", "chirps"))
        for i in range(2)
    ]
    irrelevant = [
        SourcedExample("irrelevant", f"i{i}", ("river", "flows", "down"))
        for i in range(2)
    ]
    misleading = [
        SourcedExample("misleading", f"m{i}", ("blue", "bird", "barks"))
        for i in range(4)
    ]
    heldout = [SourcedExample("heldout-1", "h1", ("green", "bird", "chirps"))]
    return student, genuine, irrelevant, misleading, heldout


class FourArmExperiments(unittest.TestCase):
    def freeze(self, *parts):
        return freeze_four_arm_specimen(*(parts or sample()), order=2)

    def test_four_arms_score_identical_heldout_with_placebo_control(self):
        result = evaluate_four_arm(self.freeze())
        by_name = {arm.arm: arm for arm in result.arms}
        self.assertEqual(set(by_name), {"baseline", "genuine", "irrelevant", "misleading"})
        self.assertEqual(by_name["baseline"].accuracy, 0)
        self.assertEqual(by_name["genuine"].accuracy, 1)
        self.assertEqual(by_name["irrelevant"].accuracy, 0)
        self.assertEqual(by_name["misleading"].accuracy, 0)
        self.assertEqual(by_name["genuine"].gain_vs_baseline, 1)
        self.assertEqual({a.evaluated_count for a in result.arms}, {1})
        self.assertEqual(result.claim_ceiling, CLAIM_CEILING)
        self.assertEqual(result.size_gate, "BELOW_PREREGISTERED_SCALE")

    def test_misleading_lesson_harms_a_correct_initial_student(self):
        result = evaluate_four_arm(self.freeze(*sample(correct_start=True)))
        by_name = {arm.arm: arm for arm in result.arms}
        self.assertEqual(by_name["baseline"].accuracy, 1)
        self.assertEqual(by_name["misleading"].accuracy, 0)
        self.assertEqual(by_name["misleading"].gain_vs_baseline, -1)

    def test_irrelevant_lesson_requires_equal_total_token_budget(self):
        student, genuine, irrelevant, misleading, heldout = sample()
        with self.assertRaisesRegex(ValueError, "match genuine lesson total token count"):
            self.freeze(student, genuine, irrelevant[:1], misleading, heldout)

    def test_exact_cross_split_overlap_and_source_group_reuse_rejected(self):
        student, genuine, irrelevant, misleading, heldout = sample()
        with self.assertRaisesRegex(ValueError, "exact example overlap"):
            self.freeze(student, genuine, irrelevant, misleading, [
                SourcedExample("heldout-1", "h1", genuine[0].tokens)
            ])
        with self.assertRaisesRegex(ValueError, "source-group overlap"):
            self.freeze(student, genuine, irrelevant, misleading, [
                SourcedExample("student", "h1", ("green", "bird", "chirps"))
            ])
        with self.assertRaisesRegex(ValueError, "source-group overlap"):
            self.freeze(student, genuine, irrelevant, [
                SourcedExample("genuine", "m1", ("blue", "bird", "barks"))
            ], heldout)

    def test_duplicate_heldout_rejected_even_with_new_record_id(self):
        student, genuine, irrelevant, misleading, heldout = sample()
        repeated = heldout + [SourcedExample("heldout-1", "h2", heldout[0].tokens)]
        with self.assertRaisesRegex(ValueError, "duplicate exact examples"):
            self.freeze(student, genuine, irrelevant, misleading, repeated)

    def test_specimen_hash_deterministic_and_tamper_rejected(self):
        parts = sample()
        first = self.freeze(*parts)
        shuffled = self.freeze(parts[0], parts[1][::-1], parts[2][::-1], parts[3], parts[4])
        self.assertEqual(first.specimen_sha256, shuffled.specimen_sha256)
        self.assertEqual(evaluate_four_arm(first), evaluate_four_arm(first))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            evaluate_four_arm(replace(first, specimen_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "invalid four-arm specimen"):
            evaluate_four_arm(replace(first, claim_ceiling="VERIFIED_TRUTH"))

    def test_rejects_malformed_input_and_order_before_stream_access(self):
        parts = sample()
        for order in (0, 6, True, "3"):
            with self.subTest(order=order), self.assertRaises(ValueError):
                freeze_four_arm_specimen(*parts, order=order)
        bad = [SourcedExample("student", "s1", ("", "bird"))]
        with self.assertRaises(ValueError):
            self.freeze(bad, *parts[1:])
        with self.assertRaises(TypeError):
            self.freeze("student", *parts[1:])

    def test_bounds_unbounded_generator_without_eager_materialization(self):
        student, genuine, irrelevant, misleading, heldout = sample()
        endless = (
            SourcedExample("student", f"s{i}", ("red", "bird", "sings"))
            for i in count()
        )
        with self.assertRaisesRegex(ValueError, "exceeds 2048 examples"):
            self.freeze(endless, genuine, irrelevant, misleading, heldout)

    def test_per_source_group_scores_and_scale_gate_not_qualification(self):
        student, genuine, irrelevant, misleading, _ = sample()
        heldout = [
            SourcedExample(f"holdout-{i % 3}", f"h{i:03d}", (f"new-{i}", "bird", "chirps"))
            for i in range(100)
        ]
        result = evaluate_four_arm(self.freeze(student, genuine, irrelevant, misleading, heldout))
        self.assertEqual(result.heldout_group_count, 3)
        self.assertEqual(sum(group.arms[0].evaluated_count for group in result.group_scores), 100)
        self.assertEqual(result.size_gate, "SCALE_ONLY_NOT_QUALIFIED")
        self.assertEqual(tuple(group.source_id for group in result.group_scores),
                         ("holdout-0", "holdout-1", "holdout-2"))

    def test_caller_can_requery_public_data_so_no_custody_claim(self):
        specimen = self.freeze()
        self.assertEqual(evaluate_four_arm(specimen), evaluate_four_arm(specimen))
        self.assertIn("NOT_INDEPENDENT_EVALUATION", evaluate_four_arm(specimen).claim_ceiling)


if __name__ == "__main__":
    unittest.main()
