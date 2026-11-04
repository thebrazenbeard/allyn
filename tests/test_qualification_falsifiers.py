"""Reproducible counterexamples to mistaking source-level gates for custody.

These tests intentionally pass: they demonstrate that structural validation
and orderly in-process consumption do not independently protect a holdout.
"""

from __future__ import annotations

import unittest

from allyn.evaluation import SourcedExample
from allyn.experiment import evaluate_four_arm, freeze_four_arm_specimen
from allyn.protocol import LessonCandidate, run_teaching_cycle


class QualificationFalsifiers(unittest.TestCase):
    def test_fabricated_source_groups_clear_scale_gate_without_custody(self):
        """A perfect synthetic score remains only a scale-qualified toy result."""
        student = [SourcedExample("student", "s0", ("red", "bird", "sings"))]
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
        # All 100 targets are the same token. The source IDs differ, but only
        # the caller asserts they are three independent source families.
        heldout = [
            SourcedExample(f"claimed-family-{i % 3}", f"h{i:03d}",
                           (f"unseen-{i}", "bird", "chirps"))
            for i in range(100)
        ]
        specimen = freeze_four_arm_specimen(
            student, genuine, irrelevant, misleading, heldout, order=2
        )
        result = evaluate_four_arm(specimen)
        scores = {arm.arm: arm for arm in result.arms}

        self.assertEqual((scores["genuine"].correct, scores["genuine"].evaluated_count), (100, 100))
        self.assertEqual(scores["baseline"].correct, 0)
        self.assertEqual(scores["irrelevant"].correct, 0)
        self.assertEqual(scores["genuine"].gain_vs_baseline, 1.0)
        self.assertEqual(result.heldout_group_count, 3)
        self.assertEqual(result.size_gate, "SCALE_ONLY_NOT_QUALIFIED")
        self.assertIn("NOT_INDEPENDENT_EVALUATION", result.claim_ceiling)
        self.assertEqual(
            {row.tokens[-1] for row in specimen.heldout}, {"chirps"}
        )

    def test_caller_owned_heldout_generator_can_adapt_at_iteration_time(self):
        """Heldout wasn't used for selection, but its caller still sets answers."""
        lesson = (("yellow", "bird", "chirps"),) * 2
        event_order: list[str] = []

        def selection_probes():
            event_order.append("dev-iterated")
            yield ("green", "bird", "chirps")

        def adaptive_heldout():
            # This runs only when run_teaching_cycle consumes the iterator,
            # after it has computed the winner from the dev probe results.
            self.assertEqual(event_order, ["dev-iterated"])
            event_order.append("heldout-iterated")
            # The caller can derive the expected outcome from its own chosen
            # lesson and tailor what purports to be an independent answer key.
            caller_selected_target = lesson[0][-1]
            yield ("cobalt", "bird", caller_selected_target)

        result = run_teaching_cycle(
            student_examples=(("red", "bird", "sings"),),
            student_source_id="student",
            candidates=(LessonCandidate("chosen-lesson", "lesson-src", lesson),),
            selection_probes=selection_probes(),
            selection_probe_source_id="dev",
            heldout_examples=adaptive_heldout(),
            heldout_source_id="heldout",
            order=2,
        )
        self.assertEqual(event_order, ["dev-iterated", "heldout-iterated"])
        self.assertEqual(result.selected_candidate_id, "chosen-lesson")
        self.assertEqual(result.selection_status, "SELECTED_ON_PROBES")
        self.assertEqual(result.final_transfer.gain, 1.0)
        self.assertEqual(result.provenance_status, "CALLER_DECLARED_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
