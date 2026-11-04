"""Behavioral and adversarial contract for train-only inquiry scheduling."""

from __future__ import annotations

import json
import unittest
from itertools import count

from allyn.curriculum import CurriculumScheduler, CLAIM_CEILING
from allyn.inquiry import InquiryLedger, InquiryQuestion


def question(name: str, status: str = "CONTESTED") -> InquiryQuestion:
    return InquiryQuestion(
        proposition_id=name,
        status=status,
        question=f"What evidence would test {name}?",
    )


def record(scheduler: CurriculumScheduler, **overrides):
    args = dict(
        observation_id="r1",
        proposition_id="a",
        source_id="self-reported-grader",
        score=1.0,
        split="train",
    )
    args.update(overrides)
    return scheduler.record_train_observation(**args)


class CurriculumSchedulerTests(unittest.TestCase):
    def test_empty_questions_allow_no_selection(self):
        scheduler = CurriculumScheduler([])
        self.assertEqual(scheduler.question_count, 0)
        self.assertIsNone(scheduler.next_question())
        self.assertEqual(scheduler.observations, ())

    def test_priority_order_and_stable_id_tie_break(self):
        qs = [
            question("reported", "REPORTED_SUPPORT"),
            question("unknown", "UNASSESSED"),
            question("challenge", "REPORTED_CHALLENGE"),
            question("z"),
            question("a"),
        ]
        first = CurriculumScheduler(qs)
        second = CurriculumScheduler(reversed(qs))
        self.assertEqual(first.next_question(), second.next_question())
        self.assertEqual(first.next_question().question.proposition_id, "a")
        self.assertEqual(first.next_question().priority_rank, 0)
        self.assertEqual(first.next_question().claim_ceiling, CLAIM_CEILING)

    def test_train_attempts_rotate_equal_priority_without_making_mastery_claim(self):
        scheduler = CurriculumScheduler([question("z"), question("a")])
        self.assertEqual(scheduler.next_question().question.proposition_id, "a")
        result = record(scheduler)
        self.assertFalse(result.replayed)
        self.assertEqual(result.observation.score, 1.0)
        self.assertEqual(scheduler.next_question().question.proposition_id, "z")
        record(scheduler, observation_id="r2", proposition_id="z", score=0.0)
        self.assertEqual(scheduler.next_question().question.proposition_id, "a")
        self.assertEqual(scheduler.next_question().local_attempt_count, 1)
        self.assertFalse(hasattr(scheduler.next_question(), "mastered"))
        self.assertFalse(hasattr(scheduler.next_question(), "verified"))

    def test_unverified_high_scores_never_change_status_priority(self):
        scheduler = CurriculumScheduler([question("a", "CONTESTED"), question("b", "UNASSESSED")])
        for idx in range(4):
            record(scheduler, observation_id=f"train-{idx}", score=1.0)
        pick = scheduler.next_question()
        self.assertEqual(pick.question.proposition_id, "a")
        self.assertEqual(pick.question.status, "CONTESTED")
        self.assertEqual(scheduler.observations[0].source_id, "self-reported-grader")

    def test_split_leakage_rejected_without_mutation(self):
        scheduler = CurriculumScheduler([question("a")])
        for split in ("heldout", "dev", "test", "TRAIN", None, True):
            with self.subTest(split=split), self.assertRaises(ValueError):
                record(scheduler, split=split)
            self.assertEqual(scheduler.observations, ())
            self.assertEqual(scheduler.next_question().local_attempt_count, 0)

    def test_replay_idempotent_collision_rejected_atomically(self):
        scheduler = CurriculumScheduler([question("a"), question("b")])
        first = record(scheduler)
        second = record(scheduler)
        self.assertFalse(first.replayed)
        self.assertTrue(second.replayed)
        self.assertEqual(first.observation, second.observation)
        self.assertEqual(len(scheduler.observations), 1)
        for changed in (dict(score=0.0), dict(source_id="other"), dict(proposition_id="b")):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "changed payload"):
                record(scheduler, **changed)
        self.assertEqual(scheduler.snapshot()["questions"][0]["local_attempt_count"], 1)

    def test_invalid_scores_and_unknown_propositions_leave_state_unchanged(self):
        scheduler = CurriculumScheduler([question("a")])
        for score in (True, float("nan"), float("inf"), -0.1, 1.1, "1", None, 10**1000):
            with self.subTest(score=str(score)[:30]), self.assertRaises(ValueError):
                record(scheduler, score=score)
            self.assertEqual(scheduler.observations, ())
        with self.assertRaises(ValueError):
            record(scheduler, proposition_id="missing")
        self.assertEqual(scheduler.next_question().local_attempt_count, 0)

    def test_invalid_questions_and_duplicate_ids_are_rejected(self):
        bad_sets = (
            [question("a"), question("a")],
            [question("a", "CLAIMED_TRUE")],
            [question(" ")],
            [InquiryQuestion("a", "CONTESTED", "")],
            ["bad"],
        )
        for questions in bad_sets:
            with self.subTest(questions=questions), self.assertRaises((TypeError, ValueError)):
                CurriculumScheduler(questions)
        with self.assertRaises(ValueError):
            CurriculumScheduler([], max_observations=True)

    def test_unbounded_question_stream_rejected_at_capacity(self):
        with self.assertRaisesRegex(ValueError, "2048 questions"):
            CurriculumScheduler(question(str(index)) for index in count())

    def test_capacity_allows_duplicate_replay_and_rejects_new_attempt(self):
        scheduler = CurriculumScheduler([question("a")], max_observations=1)
        record(scheduler)
        self.assertTrue(record(scheduler).replayed)
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            record(scheduler, observation_id="r2")
        self.assertEqual(len(scheduler.observations), 1)

    def test_snapshot_is_detached_and_json_compatible(self):
        inquiry = InquiryLedger()
        inquiry.record(
            proposition_id="a", claim="apple is blue", source_id="teacher",
            stance="supports", evidence_kind="hypothesis",
        )
        scheduler = CurriculumScheduler(inquiry.questions())
        record(scheduler)
        snapshot = scheduler.snapshot()
        json.dumps(snapshot)
        self.assertEqual(snapshot["claim_ceiling"], CLAIM_CEILING)
        snapshot["questions"][0]["question"] = "tampered"
        snapshot["observations"][0]["score"] = 0.0
        self.assertNotEqual(snapshot, scheduler.snapshot())
        self.assertEqual(scheduler.observations[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
