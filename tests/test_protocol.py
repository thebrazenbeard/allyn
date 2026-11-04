"""Contracts for source-declared, probe-selected offline teaching cycles."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import repeat
import unittest

from allyn.inquiry import InquiryLedger
from allyn.protocol import LessonCandidate, run_teaching_cycle


def _candidate(name: str, source: str, *examples: tuple[str, ...]) -> LessonCandidate:
    return LessonCandidate(name, source, examples)


def _run(**overrides):
    arguments = {
        "student_examples": (("red", "bird", "sings"),),
        "student_source_id": "student:1",
        "candidates": (
            _candidate("chirp", "teacher:chirp", ("yellow", "bird", "chirps")),
            _candidate("honk", "teacher:honk", ("yellow", "bird", "honks")),
        ),
        "selection_probes": (("green", "bird", "chirps"),),
        "selection_probe_source_id": "probe:1",
        "heldout_examples": (("purple", "bird", "chirps"),),
        "heldout_source_id": "heldout:1",
        "order": 3,
    }
    arguments.update(overrides)
    return run_teaching_cycle(**arguments)


class TeachingCycleTests(unittest.TestCase):
    def test_selects_on_probes_and_scores_distinct_heldout_with_control(self) -> None:
        receipt = _run()
        self.assertEqual(receipt.selected_candidate_id, "chirp")
        self.assertEqual(receipt.selected_lesson_source_id, "teacher:chirp")
        self.assertEqual(receipt.selection_status, "SELECTED_ON_PROBES")
        self.assertEqual(receipt.no_lesson_probe.gain, 0.0)
        self.assertEqual(len(receipt.candidate_probes), 2)
        self.assertEqual(receipt.candidate_probes[0].transfer.gain, 1.0)
        self.assertEqual(receipt.final_transfer.gain, 1.0)
        self.assertEqual(receipt.provenance_status, "CALLER_DECLARED_UNVERIFIED")
        self.assertIn("TOY_NEXT_TOKEN", receipt.claim_ceiling)

    def test_change_in_final_heldout_cannot_change_selected_lesson(self) -> None:
        first = _run(heldout_examples=(("purple", "bird", "chirps"),))
        second = _run(heldout_examples=(("purple", "bird", "sings"),))
        self.assertEqual(first.selected_candidate_id, second.selected_candidate_id)
        self.assertEqual(first.candidate_probes, second.candidate_probes)
        self.assertNotEqual(first.final_transfer, second.final_transfer)

    def test_false_teacher_can_win_probes_but_harm_final_result(self) -> None:
        receipt = _run(
            student_examples=(("sun", "will", "rise"),),
            candidates=(
                _candidate("false-pattern", "teacher:false", ("sun", "will", "fall"),
                           ("sun", "will", "fall"), ("sun", "will", "fall")),
            ),
            selection_probes=(("stone", "will", "fall"),),
            heldout_examples=(("moon", "will", "rise"),),
        )
        self.assertEqual(receipt.selected_candidate_id, "false-pattern")
        self.assertEqual(receipt.candidate_probes[0].transfer.gain, 1.0)
        self.assertEqual(receipt.final_transfer.gain, -1.0)
        self.assertEqual(receipt.final_transfer.baseline_accuracy, 1.0)
        self.assertEqual(receipt.final_transfer.taught_accuracy, 0.0)

    def test_abstains_when_probes_have_no_gain(self) -> None:
        receipt = _run(
            candidates=(_candidate("wrong", "teacher:wrong", ("yellow", "bird", "honks")),),
        )
        self.assertIsNone(receipt.selected_candidate_id)
        self.assertIsNone(receipt.selected_lesson_source_id)
        self.assertEqual(receipt.selection_status, "ABSTAINED_NO_PROBE_GAIN")
        self.assertEqual(receipt.final_transfer.lesson_examples_count, 0)
        self.assertEqual(receipt.final_transfer.gain, 0.0)

    def test_inquiry_status_is_reported_not_admitted_as_truth(self) -> None:
        inquiry = InquiryLedger()
        inquiry.record(proposition_id="p", claim="birds chirp", source_id="a",
                       stance="supports", evidence_kind="hypothesis")
        inquiry.record(proposition_id="p", claim="birds chirp", source_id="b",
                       stance="challenges", evidence_kind="observation")
        before = inquiry.events
        receipt = _run(inquiry=inquiry, inquiry_proposition_id="p")
        self.assertEqual(receipt.inquiry_report_status, "CONTESTED")
        self.assertEqual(receipt.inquiry_proposition_id, "p")
        self.assertEqual(receipt.provenance_status, "CALLER_DECLARED_UNVERIFIED")
        self.assertEqual(inquiry.events, before)

    def test_same_specimen_is_deterministic_and_receipt_immutable(self) -> None:
        first, second = _run(), _run()
        self.assertEqual(first, second)
        with self.assertRaises(FrozenInstanceError):
            first.selected_candidate_id = "evil"  # type: ignore[misc]

    def test_exact_heldout_overlap_is_rejected_even_with_unselected_candidate(self) -> None:
        for example in (("red", "bird", "sings"),
                        ("green", "bird", "chirps"),
                        ("yellow", "bird", "honks")):
            with self.subTest(example=example), self.assertRaisesRegex(ValueError, "heldout specimen overlaps"):
                _run(heldout_examples=(example,))

    def test_probe_exact_replay_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection probe overlaps"):
            _run(selection_probes=(("red", "bird", "sings"),))
        with self.assertRaisesRegex(ValueError, "selection probe overlaps"):
            _run(selection_probes=(("yellow", "bird", "chirps"),))

    def test_duplicate_evaluation_specimens_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection probes must contain unique"):
            _run(selection_probes=(("green", "bird", "chirps"),) * 2)
        with self.assertRaisesRegex(ValueError, "heldout specimens must contain unique"):
            _run(heldout_examples=(("purple", "bird", "chirps"),) * 50)

    def test_source_reuse_and_duplicate_candidates_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "source references must differ"):
            _run(heldout_source_id="probe:1")
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            _run(candidates=(_candidate("c", "probe:1", ("a", "b")),))
        with self.assertRaisesRegex(ValueError, "candidate IDs must be unique"):
            _run(candidates=(_candidate("same", "teacher:a", ("a", "b")),
                             _candidate("same", "teacher:b", ("a", "c"))))
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            _run(candidates=(_candidate("one", "teacher:a", ("a", "b")),
                             _candidate("two", "teacher:a", ("a", "c"))))

    def test_unbounded_inputs_rejected_without_hanging(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds 2048 examples"):
            _run(selection_probes=repeat(("a", "b")))
        with self.assertRaisesRegex(ValueError, "exceeds 10000 tokens"):
            _run(candidates=(_candidate("infinite", "teacher:a", repeat("a")),))
        with self.assertRaisesRegex(ValueError, "1 to 16 proposals"):
            _run(candidates=tuple(_candidate("c"+str(i), "teacher:"+str(i), ("a", "b"))
                                  for i in range(17)))

    def test_invalid_order_rejected_before_consuming_examples(self) -> None:
        def explosive():
            raise AssertionError("should never consume")
            yield ("a", "b")

        with self.assertRaisesRegex(ValueError, "order must be an integer"):
            _run(student_examples=explosive(), order=6)


if __name__ == "__main__":
    unittest.main()
