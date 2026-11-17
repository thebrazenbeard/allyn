"""Bounded tests for idempotent local inquiry admission; no identity assertions."""

from __future__ import annotations

import json
import unittest

from allyn.admission import AdmissionLedger, CLAIM_CEILING


class AdmissionLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.admission = AdmissionLedger()

    def send(self, request_id="r1", **changes):
        values = {
            "request_id": request_id,
            "proposition_id": "p1",
            "claim": "the moon is made of cheese",
            "source_id": "teacher",
            "stance": "supports",
            "evidence_kind": "hypothesis",
        }
        values.update(changes)
        return self.admission.submit(**values)

    def test_first_request_returns_unverified_receipt(self):
        receipt = self.send()
        self.assertFalse(receipt.replayed)
        self.assertEqual(receipt.event.event_id, "allyn-inq-000000000001")
        self.assertEqual(receipt.claim_ceiling, CLAIM_CEILING)
        self.assertEqual(self.admission.admitted_count, 1)

    def test_identical_replay_is_idempotent(self):
        first = self.send()
        replay = self.send()
        self.assertTrue(replay.replayed)
        self.assertEqual(first.event, replay.event)
        self.assertEqual(first.payload_sha256, replay.payload_sha256)
        self.assertEqual(len(self.admission.events), 1)
        self.assertEqual(self.admission.assess("p1").active_support_count, 1)

    def test_same_request_id_changed_payload_is_rejected_atomically(self):
        first = self.send()
        for changed in ({"stance": "challenges"}, {"source_id": "attacker"}, {"claim": "false"}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "different payload"):
                self.send(**changed)
        self.assertEqual(self.admission.events, (first.event,))
        self.assertEqual(self.admission.admitted_count, 1)

    def test_different_request_id_is_not_independent_corroboration(self):
        self.send("r1")
        self.send("r2")
        self.assertEqual(self.admission.assess("p1").active_support_count, 2)
        self.assertEqual(self.admission.assess("p1").source_ids, ("teacher",))
        # These are raw reports, not separate authenticated witnesses.

    def test_conflicting_report_remains_contested(self):
        self.send()
        self.send("r2", source_id="student", stance="challenges")
        self.assertEqual(self.admission.assess("p1").status, "CONTESTED")

    def test_claim_collision_does_not_reserve_request_id(self):
        self.send()
        with self.assertRaisesRegex(ValueError, "different canonical claim"):
            self.send("r2", claim="moon is rock", source_id="attacker")
        self.assertEqual(self.admission.admitted_count, 1)
        recovered = self.send("r2", source_id="student", stance="challenges")
        self.assertEqual(recovered.event.sequence_number, 2)

    def test_spoofed_correction_cannot_erase_original_or_block_later_claim(self):
        first = self.send("real")
        spoof = self.send("spoof", stance="challenges", corrects_event_id=first.event.event_id)
        later = self.send("later", stance="supports", corrects_event_id=first.event.event_id)
        self.assertEqual(spoof.event.correction_status, "UNVERIFIED")
        self.assertEqual(later.event.correction_status, "UNVERIFIED")
        self.assertEqual(self.admission.assess("p1").status, "CONTESTED")
        self.assertEqual(len(self.admission.events), 3)

    def test_bounded_capacity_allows_replay_at_limit(self):
        admission = AdmissionLedger(max_requests=1)
        args = dict(request_id="r1", proposition_id="p", claim="x", source_id="s",
                    stance="supports", evidence_kind="hypothesis")
        first = admission.submit(**args)
        self.assertTrue(admission.submit(**args).replayed)
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            admission.submit(**{**args, "request_id": "r2"})
        self.assertEqual(admission.events, (first.event,))

    def test_invalid_input_rejected_without_state_change(self):
        invalid = ({"request_id": ""}, {"request_id": 1}, {"request_id": "x" * 257},
                   {"claim": "x" * 4097}, {"evidence_kind": "verified"},
                   {"corrects_event_id": ""}, {"stance": "agrees"})
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.send(**changes)
            self.assertEqual(self.admission.admitted_count, 0)
        with self.assertRaises(ValueError):
            AdmissionLedger(max_requests=True)

    def test_snapshot_detached_and_json_compatible(self):
        self.send()
        snap = self.admission.snapshot()
        json.dumps(snap)
        snap["requests"][0]["request_id"] = "tampered"
        snap["inquiry"]["events"][0]["source_id"] = "tampered"
        self.assertEqual(self.admission.snapshot()["requests"][0]["request_id"], "r1")
        self.assertEqual(self.admission.events[0].source_id, "teacher")


if __name__ == "__main__":
    unittest.main()
