"""Model-free tests for source-declared teacher/student callback orchestration."""

from __future__ import annotations

from dataclasses import replace
import unittest

from allyn.student_exchange import (
    DevAssessment,
    HeldoutAssessment,
    LessonProposal,
    StudentState,
    UpdateReceipt,
    run_student_exchange,
)


def digest(char: str) -> str:
    return char * 64


class ExchangeHarness:
    def __init__(self, *, kind: str = "CONTEXT_UPDATE"):
        self.kind = kind
        self.before = StudentState("model-local", digest("a"), digest("1"))
        self.after = StudentState("model-local", digest("b"), digest("2") if kind == "WEIGHT_UPDATE" else digest("1"))
        if kind == "NO_CHANGE":
            self.after = self.before
        self.current = self.before
        self.events = []
        self.teacher_brief = None
        self.baseline_score = 2
        self.after_score = 5

    def read_state(self):
        self.events.append("read")
        return self.current

    def assess_dev(self, state):
        self.events.append("dev")
        return DevAssessment("dev-source", digest("3"), 3, 5)

    def teacher(self, brief):
        self.events.append("teacher")
        self.teacher_brief = brief
        return LessonProposal("lesson-1", "lesson-source", ("Explain the pattern",))

    def apply_lesson(self, lesson):
        self.events.append("apply")
        self.current = self.after
        return UpdateReceipt("op-1", lesson.lesson_id, self.before.state_sha256,
                             self.after.state_sha256, self.kind)

    def evaluate_heldout(self, state, phase):
        self.events.append(f"heldout-{phase}")
        return HeldoutAssessment(phase, state.model_id, state.state_sha256,
                                 "heldout-source", digest("4"), "grader-1",
                                 self.baseline_score if phase == "baseline" else self.after_score, 10)

    def run(self):
        return run_student_exchange(
            run_id="run-1", read_state=self.read_state, assess_dev=self.assess_dev,
            teacher=self.teacher, apply_lesson=self.apply_lesson,
            evaluate_heldout=self.evaluate_heldout,
        )


class StudentExchangeTests(unittest.TestCase):
    def test_context_exchange_retains_gain_and_hides_heldout_from_teacher(self):
        h = ExchangeHarness()
        receipt = h.run()
        self.assertEqual(h.events, ["read", "dev", "teacher", "heldout-baseline",
                                    "apply", "read", "heldout-post"])
        self.assertAlmostEqual(receipt.heldout_accuracy_gain, .3)
        self.assertEqual(receipt.update_evidence, "CALLER_DECLARED_CONTEXT_STATE_CHANGE_ONLY")
        self.assertEqual(set(h.teacher_brief.__dataclass_fields__),
                         {"run_id", "student_state", "dev_assessment"})
        self.assertFalse(hasattr(h.teacher_brief, "heldout"))
        self.assertFalse(hasattr(h.teacher_brief.dev_assessment, "heldout"))

    def test_declared_weight_change_does_not_claim_verified_training(self):
        h = ExchangeHarness(kind="WEIGHT_UPDATE")
        receipt = h.run()
        self.assertEqual(receipt.update_evidence, "CALLER_DECLARED_WEIGHT_DIGEST_CHANGE_UNVERIFIED")
        self.assertEqual(receipt.provenance_status, "CALLER_DECLARED_UNVERIFIED")

    def test_declared_weight_update_needs_changed_weight_digests(self):
        h = ExchangeHarness(kind="WEIGHT_UPDATE")
        h.after = replace(h.after, weights_sha256=h.before.weights_sha256)
        with self.assertRaisesRegex(ValueError, "distinct declared pre/post weight"):
            h.run()

    def test_unexpected_unreported_weight_change_rejected(self):
        h = ExchangeHarness(kind="CONTEXT_UPDATE")
        h.after = replace(h.after, weights_sha256=digest("2"))
        with self.assertRaisesRegex(ValueError, "CONTEXT_UPDATE contradicts"):
            h.run()

    def test_context_change_cannot_drop_weight_digest_after_baseline(self):
        h = ExchangeHarness(kind="CONTEXT_UPDATE")
        h.after = replace(h.after, weights_sha256=None)
        with self.assertRaisesRegex(ValueError, "CONTEXT_UPDATE contradicts"):
            h.run()

    def test_no_change_does_not_require_improvement(self):
        h = ExchangeHarness(kind="NO_CHANGE")
        h.after_score = 1
        receipt = h.run()
        self.assertAlmostEqual(receipt.heldout_accuracy_gain, -.1)
        self.assertEqual(receipt.update_evidence, "DECLARED_NO_STATE_CHANGE")

    def test_update_receipt_must_match_before_and_after(self):
        h = ExchangeHarness()
        def wrong_update(lesson):
            h.current = h.after
            return UpdateReceipt("op", "wrong", h.before.state_sha256, h.after.state_sha256, "CONTEXT_UPDATE")
        h.apply_lesson = wrong_update
        with self.assertRaisesRegex(ValueError, "does not match lesson"):
            h.run()

    def test_after_state_must_match_update(self):
        h = ExchangeHarness()
        def lying_update(lesson):
            return UpdateReceipt("op", lesson.lesson_id, h.before.state_sha256,
                                 h.after.state_sha256, "CONTEXT_UPDATE")
        h.apply_lesson = lying_update
        with self.assertRaisesRegex(ValueError, "post state does not match"):
            h.run()

    def test_same_heldout_custody_declared_both_phases(self):
        h = ExchangeHarness()
        base = h.evaluate_heldout
        def changed_grader(state, phase):
            assessment = base(state, phase)
            return replace(assessment, evaluator_id="other-grader") if phase == "post" else assessment
        h.evaluate_heldout = changed_grader
        with self.assertRaisesRegex(ValueError, "heldout assessment changed"):
            h.run()

    def test_dev_and_heldout_separation(self):
        h = ExchangeHarness()
        h.assess_dev = lambda state: DevAssessment("heldout-source", digest("3"), 3, 5)
        with self.assertRaisesRegex(ValueError, "DEV and HELDOUT"):
            h.run()
        self.assertIn("teacher", h.events)
        self.assertNotIn("apply", h.events)

    def test_invalid_lessons_rejected_before_update(self):
        h = ExchangeHarness()
        h.teacher = lambda brief: LessonProposal("lesson", "lesson-source", ())
        with self.assertRaisesRegex(ValueError, "lesson content"):
            h.run()
        self.assertNotIn("heldout-baseline", h.events)
        self.assertNotIn("apply", h.events)

    def test_teacher_proposes_before_any_heldout_side_effect(self):
        h = ExchangeHarness()
        observed = []
        base_heldout = h.evaluate_heldout
        base_teacher = h.teacher
        def grader(state, phase):
            observed.append(phase)
            return base_heldout(state, phase)
        def teacher(brief):
            self.assertEqual(observed, [])
            return base_teacher(brief)
        h.evaluate_heldout = grader
        h.teacher = teacher
        h.run()
        self.assertEqual(observed, ["baseline", "post"])

    def test_negative_outcome_is_preserved(self):
        h = ExchangeHarness()
        h.after_score = 0
        receipt = h.run()
        self.assertAlmostEqual(receipt.heldout_accuracy_gain, -.2)

    def test_no_callbacks_run_on_invalid_arguments(self):
        h = ExchangeHarness()
        with self.assertRaisesRegex(ValueError, "run_id"):
            run_student_exchange(run_id="", read_state=h.read_state,
                assess_dev=h.assess_dev, teacher=h.teacher,
                apply_lesson=h.apply_lesson, evaluate_heldout=h.evaluate_heldout)
        self.assertEqual(h.events, [])


if __name__ == "__main__":
    unittest.main()
