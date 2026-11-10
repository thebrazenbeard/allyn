"""Cross-boundary negative tests for ALLYN's offline research-only interfaces.

Passing these checks does not authenticate a teacher or constitute an
independently custodied evaluation. They protect narrower local invariants.
"""

from __future__ import annotations

from dataclasses import replace
import unittest

from allyn.admission import AdmissionLedger
from allyn.evaluation import SourcedExample, evaluate_frozen, freeze_specimen
from allyn.protocol import LessonCandidate, run_teaching_cycle


def row(source: str, record: str, *tokens: str) -> SourcedExample:
    return SourcedExample(source, record, tokens)


def cycle(**changes):
    inputs = dict(
        student_examples=(("red", "bird", "sings"),),
        student_source_id="student",
        candidates=(LessonCandidate("chirp", "teacher", (("yellow", "bird", "chirps"),)),),
        selection_probes=(("green", "bird", "chirps"),),
        selection_probe_source_id="probe",
        heldout_examples=(("purple", "bird", "chirps"),),
        heldout_source_id="heldout",
        order=3,
    )
    inputs.update(changes)
    return run_teaching_cycle(**inputs)


class AdversarialIntegrationTests(unittest.TestCase):
    def test_teacher_cannot_paste_probe_answer_as_candidate_lesson(self):
        with self.assertRaisesRegex(ValueError, "selection probe overlaps"):
            cycle(candidates=(LessonCandidate(
                "replay", "teacher", (("green", "bird", "chirps"),)),))

    def test_selection_probe_must_not_be_replay_of_student_training(self):
        with self.assertRaisesRegex(ValueError, "selection probe overlaps"):
            cycle(selection_probes=(("red", "bird", "sings"),))

    def test_invalid_lesson_does_not_read_heldout_generator(self):
        def heldout_must_remain_unread():
            raise AssertionError("heldout was read before failed probe admission")
            yield ("purple", "bird", "chirps")

        with self.assertRaisesRegex(ValueError, "selection probe overlaps"):
            cycle(
                candidates=(LessonCandidate("replay", "teacher", (("green", "bird", "chirps"),)),),
                heldout_examples=heldout_must_remain_unread(),
            )

    def test_unselected_candidate_cannot_hide_an_exact_heldout_overlap(self):
        with self.assertRaisesRegex(ValueError, "heldout specimen overlaps"):
            cycle(candidates=(
                LessonCandidate("good", "teacher:1", (("yellow", "bird", "chirps"),)),
                LessonCandidate("bad", "teacher:2", (("purple", "bird", "chirps"),)),
            ))

    def test_false_teacher_can_win_probes_and_harm_unseen_heldout(self):
        result = cycle(
            student_examples=(("sun", "will", "rise"),),
            candidates=(LessonCandidate("false", "teacher", (("sun", "will", "fall"),) * 3),),
            selection_probes=(("stone", "will", "fall"),),
            heldout_examples=(("moon", "will", "rise"),),
        )
        self.assertEqual(result.selected_candidate_id, "false")
        self.assertEqual(result.candidate_probes[0].transfer.gain, 1.0)
        self.assertEqual(result.final_transfer.gain, -1.0)
        self.assertIn("CALLER_DECLARED", result.claim_ceiling)

    def test_public_heldout_oracle_allows_cherry_picking_despite_disjoint_splits(self):
        student = [row("student", "1", "red", "bird", "sings")]
        heldout = [row("heldout", "1", "green", "bird", "chirps")]
        helpful = evaluate_frozen(freeze_specimen(
            student, [row("teacher:a", "1", "yellow", "bird", "chirps")], heldout))
        irrelevant = evaluate_frozen(freeze_specimen(
            student, [row("teacher:b", "1", "yellow", "bird", "honks")], heldout))
        self.assertGreater(helpful.transfer.gain, irrelevant.transfer.gain)
        # This deliberately documents a remaining gap: the caller can query
        # the public evaluator repeatedly. A digest is not sealed custody.
        self.assertEqual(helpful.claim_ceiling, "CALLER_DECLARED_SPLIT_NOT_INDEPENDENT_CUSTODY")

    def test_admission_replay_is_idempotent_and_payload_collision_fails(self):
        ledger = AdmissionLedger(max_requests=2)
        report = dict(request_id="one", proposition_id="p", claim="x",
                      source_id="teacher", stance="supports", evidence_kind="hypothesis")
        first = ledger.submit(**report)
        self.assertTrue(ledger.submit(**report).replayed)
        self.assertEqual(ledger.admitted_count, 1)
        with self.assertRaisesRegex(ValueError, "different payload"):
            ledger.submit(**{**report, "stance": "challenges"})
        self.assertEqual(ledger.events, (first.event,))

    def test_spoofed_teacher_correction_cannot_erase_support(self):
        ledger = AdmissionLedger()
        args = dict(proposition_id="p", claim="the claim", source_id="teacher",
                    stance="supports", evidence_kind="hypothesis")
        original = ledger.submit(request_id="original", **args)
        forged = ledger.submit(request_id="forged", **{
            **args, "stance": "challenges", "corrects_event_id": original.event.event_id})
        later = ledger.submit(request_id="later", **{
            **args, "corrects_event_id": original.event.event_id})
        self.assertEqual(forged.event.correction_status, "UNVERIFIED")
        self.assertEqual(later.event.correction_status, "UNVERIFIED")
        status = ledger.assess("p")
        self.assertEqual(status.status, "CONTESTED")
        self.assertEqual(status.active_support_count, 2)
        self.assertEqual(status.active_challenge_count, 1)

    def test_forged_public_frozen_descriptor_does_not_bypass_revalidation(self):
        student = [row("student", "1", "red", "bird", "sings")]
        lesson = [row("teacher", "1", "yellow", "bird", "chirps")]
        heldout = [row("heldout", "1", "green", "bird", "chirps")]
        descriptor = freeze_specimen(student, lesson, heldout)
        forged = replace(descriptor, heldout=(row("heldout", "1", "yellow", "bird", "chirps"),))
        with self.assertRaises(ValueError):
            evaluate_frozen(forged)

    def test_source_group_reuse_rejected_without_explicit_exception(self):
        student = [row("same", "student", "red", "bird", "sings")]
        lesson = [row("same", "lesson", "yellow", "bird", "chirps")]
        heldout = [row("heldout", "1", "green", "bird", "chirps")]
        with self.assertRaisesRegex(ValueError, "source-group overlap"):
            freeze_specimen(student, lesson, heldout)


if __name__ == "__main__":
    unittest.main()
