"""Tests for bounded, source-attributed inquiry without truth authority."""

from __future__ import annotations

import unittest

from allyn.inquiry import CLAIM_CEILING, InquiryLedger


class InquiryLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = InquiryLedger()

    def report(self, source="teacher", stance="supports", kind="hypothesis", **options):
        return self.ledger.record(
            proposition_id="p1",
            claim="the moon is made of cheese",
            source_id=source,
            stance=stance,
            evidence_kind=kind,
            **options,
        )

    def test_unobserved_proposition_is_unassessed(self):
        result = self.ledger.assess("unknown")
        self.assertEqual(result.status, "UNASSESSED")
        self.assertEqual(result.active_support_count, 0)

    def test_false_teacher_claim_is_not_promoted_to_fact(self):
        self.report(source="teacher", kind="observation")
        result = self.ledger.assess("p1")
        self.assertEqual(result.status, "REPORTED_SUPPORT")
        self.assertEqual(result.claim_ceiling, CLAIM_CEILING)
        self.assertEqual(result.source_ids, ("teacher",))

    def test_different_source_challenge_creates_contested_status(self):
        self.report()
        self.report(source="student", stance="challenges", kind="inference")
        result = self.ledger.assess("p1")
        self.assertEqual(result.status, "CONTESTED")
        self.assertEqual(result.active_support_count, 1)
        self.assertEqual(result.active_challenge_count, 1)
        self.assertEqual(result.source_ids, ("student", "teacher"))

    def test_same_source_correction_is_unverified_and_cannot_hide_original(self):
        first = self.report()
        second = self.ledger.record(
            proposition_id="p1",
            claim="the moon is made of cheese",
            source_id="teacher",
            stance="challenges",
            evidence_kind="inference",
            corrects_event_id=first.event_id,
        )
        result = self.ledger.assess("p1")
        self.assertEqual(result.status, "CONTESTED")
        self.assertEqual(result.active_support_count, 1)
        self.assertEqual(result.active_challenge_count, 1)
        self.assertEqual(self.ledger.events, (first, second))
        self.assertEqual(second.corrects_event_id, first.event_id)
        self.assertEqual(second.correction_status, "UNVERIFIED")
        self.assertEqual(self.ledger.snapshot()["events"][1]["correction_status"], "UNVERIFIED")

    def test_same_source_id_spoof_cannot_supersede_original_support(self):
        original = self.report(source="teacher")
        spoof = self.report(
            source="teacher", stance="challenges", corrects_event_id=original.event_id
        )
        self.assertEqual(spoof.correction_status, "UNVERIFIED")
        self.assertEqual(self.ledger.assess("p1").status, "CONTESTED")
        self.assertEqual(self.ledger.events[0], original)

    def test_proposition_id_collision_with_different_claim_is_rejected(self):
        first = self.report()
        with self.assertRaisesRegex(ValueError, "different canonical claim"):
            self.ledger.record(
                proposition_id="p1", claim="the moon is a rock", source_id="student",
                stance="challenges", evidence_kind="observation",
            )
        self.assertEqual(self.ledger.events, (first,))

    def test_source_cannot_supersede_another_source_report(self):
        first = self.report()
        with self.assertRaisesRegex(ValueError, "same claimed source"):
            self.report(source="student", corrects_event_id=first.event_id)
        self.assertEqual(self.ledger.events, (first,))

    def test_multiple_unverified_corrections_can_link_to_same_report(self):
        first = self.report()
        first_correction = self.report(stance="challenges", corrects_event_id=first.event_id)
        second_correction = self.report(stance="challenges", corrects_event_id=first.event_id)
        self.assertEqual(first_correction.correction_status, "UNVERIFIED")
        self.assertEqual(second_correction.correction_status, "UNVERIFIED")
        self.assertEqual(len(self.ledger.events), 3)
        self.assertEqual(self.ledger.assess("p1").status, "CONTESTED")

    def test_spoofed_correction_cannot_block_later_claimed_real_correction(self):
        first = self.report(source="teacher")
        spoof = self.report(
            source="teacher", stance="challenges", corrects_event_id=first.event_id
        )
        later = self.report(
            source="teacher", stance="supports", corrects_event_id=first.event_id
        )
        self.assertEqual(spoof.correction_status, "UNVERIFIED")
        self.assertEqual(later.correction_status, "UNVERIFIED")
        self.assertEqual([event.event_id for event in self.ledger.events],
                         [first.event_id, spoof.event_id, later.event_id])
        self.assertEqual(self.ledger.assess("p1").active_support_count, 2)
        self.assertEqual(self.ledger.assess("p1").active_challenge_count, 1)

    def test_questions_are_deterministic_and_expose_conflicts(self):
        self.report()
        self.report(source="student", stance="challenges")
        self.ledger.record(
            proposition_id="a0", claim="maybe", source_id="student",
            stance="supports", evidence_kind="hypothesis",
        )
        questions = self.ledger.questions()
        self.assertEqual([item.proposition_id for item in questions], ["a0", "p1"])
        self.assertEqual(questions[1].status, "CONTESTED")
        self.assertIn("conflicting", questions[1].question)

    def test_invalid_inputs_are_rejected_atomically(self):
        invalid = [
            {"source": ""},
            {"source": 12},
            {"stance": "truth"},
            {"kind": "proof"},
            {"corrects_event_id": "missing"},
        ]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.report(**values)
                self.assertEqual(len(self.ledger.events), 0)

    def test_snapshot_is_detached_and_json_compatible(self):
        self.report()
        snapshot = self.ledger.snapshot()
        snapshot["events"][0]["claim"] = "tampered"
        snapshot["events"].append({"event_id": "fake"})
        self.assertEqual(self.ledger.events[0].claim, "the moon is made of cheese")
        self.assertEqual(len(self.ledger.events), 1)


if __name__ == "__main__":
    unittest.main()
