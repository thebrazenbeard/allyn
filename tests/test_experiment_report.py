"""Paired audit report is diagnostic, not external evaluation evidence."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import unittest

from allyn.evaluation import SourcedExample
from allyn.experiment import freeze_four_arm_specimen
from allyn.experiment_report import (
    ARM_NAMES,
    EVIDENCE_CLASS,
    QUALIFICATION,
    audit_four_arm,
)


def specimen(*, baseline_correct: bool = False, count: int = 1):
    initial = "chirps" if baseline_correct else "sings"
    student = [SourcedExample("student", "s1", ("red", "bird", initial))]
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
    heldout = [
        SourcedExample(f"heldout-{i % 3}", f"h{i}", (f"new-{i}", "bird", "chirps"))
        for i in range(count)
    ]
    return freeze_four_arm_specimen(student, genuine, irrelevant, misleading, heldout, order=2)


class FourArmAuditTests(unittest.TestCase):
    def test_example_level_paired_gain_and_placebo_control(self):
        report = audit_four_arm(specimen())
        by_arm = {row.arm: row for row in report.arms}
        self.assertEqual(tuple(by_arm), ARM_NAMES)
        self.assertEqual({a.evaluated_count for a in report.arms}, {1})
        self.assertEqual(by_arm["baseline"].correct, 0)
        self.assertEqual(by_arm["genuine"].paired_improvements, 1)
        self.assertEqual(by_arm["genuine"].paired_regressions, 0)
        self.assertEqual(by_arm["genuine"].net_correct_vs_baseline, 1)
        self.assertEqual(by_arm["irrelevant"].paired_unchanged, 1)
        self.assertEqual(report.per_example[0].genuine_delta, 1)
        self.assertEqual(report.per_example[0].irrelevant_delta, 0)
        self.assertEqual(report.genuine_minus_irrelevant_accuracy, 1.0)
        self.assertEqual(report.placebo_delta_vs_baseline, 0.0)
        self.assertEqual(report.group_scores[0].source_id, "heldout-0")
        self.assertEqual((report.genuine_lesson_tokens, report.irrelevant_lesson_tokens), (6, 6))
        self.assertEqual(report.misleading_lesson_tokens, 12)
        self.assertFalse(report.misleading_budget_matched)
        self.assertIn("MISLEADING_CONTROL_TOKEN_BUDGET_UNMATCHED", report.unmet_gates)

    def test_misleading_regresses_correct_initial_student(self):
        report = audit_four_arm(specimen(baseline_correct=True))
        by_arm = {row.arm: row for row in report.arms}
        self.assertEqual(by_arm["baseline"].correct, 1)
        self.assertEqual(by_arm["genuine"].paired_unchanged, 1)
        self.assertEqual(by_arm["misleading"].paired_regressions, 1)
        self.assertEqual(by_arm["misleading"].net_correct_vs_baseline, -1)
        self.assertEqual(report.per_example[0].misleading_delta, -1)
        self.assertEqual(report.misleading_delta_vs_baseline, -1.0)

    def test_no_genuine_gain_still_not_qualified(self):
        report = audit_four_arm(specimen(baseline_correct=True))
        genuine = next(arm for arm in report.arms if arm.arm == "genuine")
        self.assertEqual(genuine.paired_improvements, 0)
        self.assertEqual(genuine.paired_regressions, 0)
        self.assertEqual(genuine.net_correct_vs_baseline, 0)
        self.assertEqual(report.qualification, QUALIFICATION)
        self.assertEqual(report.evidence_class, EVIDENCE_CLASS)
        self.assertIn("NOT_PREREGISTERED_OR_INDEPENDENTLY_EVALUATED", report.unmet_gates)

    def test_synthetic_scale_does_not_inflate_claim(self):
        report = audit_four_arm(specimen(count=100))
        self.assertEqual(report.size_gate, "SCALE_ONLY_NOT_QUALIFIED")
        self.assertEqual(report.heldout_group_count, 3)
        self.assertEqual(sum(g.arms[0].evaluated_count for g in report.group_scores), 100)
        self.assertNotIn("BELOW_100_HELDOUT_OR_3_DECLARED_GROUPS", report.unmet_gates)
        self.assertEqual(report.qualification, QUALIFICATION)
        self.assertIn("HELDOUT_EXPOSED_AND_REPEATABLE", report.unmet_gates)

    def test_deterministic_json_compatible_and_tamper_rejected(self):
        declared = specimen()
        report = audit_four_arm(declared)
        self.assertEqual(report, audit_four_arm(declared))
        self.assertIn("genuine_delta", json.dumps(asdict(report), sort_keys=True))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            audit_four_arm(replace(declared, specimen_sha256="0" * 64))


if __name__ == "__main__":
    unittest.main()
