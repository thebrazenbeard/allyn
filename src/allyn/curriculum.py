"""Bounded inquiry-gap scheduling with explicitly unverified training feedback.

This module helps select *questions*, not truths or mastered skills. Questions
come from caller-declared inquiry reports; scores and source IDs are likewise
unverified. A high score never removes a question or promotes its proposition.
The scheduler cannot access a HELDOUT partition or an evaluator answer key.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from threading import Lock

from .inquiry import InquiryQuestion


SCHEMA = "allyn.curriculum.v0"
CLAIM_CEILING = "UNVERIFIED_TRAINING_FEEDBACK_SCHEDULING_ONLY"
_MAX_QUESTIONS = 2_048
_MAX_OBSERVATIONS = 10_000
_MAX_ID_CHARS = 256
_MAX_QUESTION_CHARS = 4_096
_PRIORITY = {
    "CONTESTED": 0,
    "UNASSESSED": 1,
    "REPORTED_CHALLENGE": 2,
    "REPORTED_SUPPORT": 3,
}


def _text(value: object, label: str, limit: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be nonblank text up to {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be UTF-8 encodable") from exc
    return value


@dataclass(frozen=True, slots=True)
class TrainingObservation:
    """One caller-declared TRAIN report, not an authenticated grade."""

    observation_id: str
    proposition_id: str
    source_id: str
    score: float
    split: str = "train"
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True, slots=True)
class TrainingObservationReceipt:
    observation: TrainingObservation
    replayed: bool
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True, slots=True)
class CurriculumSelection:
    """A deterministic scheduling hint, with no mastery or truth field."""

    question: InquiryQuestion
    local_attempt_count: int
    priority_rank: int
    claim_ceiling: str = CLAIM_CEILING


class CurriculumScheduler:
    """Rotate unresolved questions using TRAIN attempt counts, not grades.

    Priority is CONTESTED, UNASSESSED, REPORTED_CHALLENGE, then
    REPORTED_SUPPORT. Within a priority group, questions with fewer recorded
    attempts precede others; proposition ID breaks ties. Every question stays
    eligible regardless of an unverified score. This is *not* progress
    verification, a grader, or an independently supervised curriculum.
    """

    def __init__(
        self,
        questions: Iterable[InquiryQuestion],
        *,
        max_observations: int = _MAX_OBSERVATIONS,
    ) -> None:
        if type(max_observations) is not int or not 1 <= max_observations <= _MAX_OBSERVATIONS:
            raise ValueError(f"max_observations must be 1 through {_MAX_OBSERVATIONS}")
        if isinstance(questions, (str, bytes, bytearray)):
            raise TypeError("questions must be an iterable of InquiryQuestion")
        try:
            iterator = iter(questions)
        except TypeError as exc:
            raise TypeError("questions must be an iterable of InquiryQuestion") from exc

        frozen: dict[str, InquiryQuestion] = {}
        for index, question in enumerate(iterator):
            if index >= _MAX_QUESTIONS:
                raise ValueError(f"curriculum exceeds {_MAX_QUESTIONS} questions")
            if type(question) is not InquiryQuestion:
                raise TypeError("questions must contain InquiryQuestion objects")
            proposition_id = _text(question.proposition_id, "proposition_id", _MAX_ID_CHARS)
            _text(question.question, "question", _MAX_QUESTION_CHARS)
            if type(question.status) is not str or question.status not in _PRIORITY:
                raise ValueError("unsupported inquiry question status")
            if proposition_id in frozen:
                raise ValueError("duplicate proposition_id in curriculum")
            frozen[proposition_id] = question

        self._questions = frozen
        self._max_observations = max_observations
        self._observations: dict[str, TrainingObservation] = {}
        self._attempts: dict[str, int] = {identifier: 0 for identifier in frozen}
        self._lock = Lock()

    @property
    def question_count(self) -> int:
        return len(self._questions)

    @property
    def observations(self) -> tuple[TrainingObservation, ...]:
        with self._lock:
            return tuple(self._observations.values())

    def next_question(self) -> CurriculumSelection | None:
        """Choose without reading, trusting, or revealing a heldout score."""
        with self._lock:
            if not self._questions:
                return None
            chosen = min(
                self._questions.values(),
                key=lambda q: (
                    _PRIORITY[q.status],
                    self._attempts[q.proposition_id],
                    q.proposition_id,
                ),
            )
            return CurriculumSelection(
                question=chosen,
                local_attempt_count=self._attempts[chosen.proposition_id],
                priority_rank=_PRIORITY[chosen.status],
            )

    def record_train_observation(
        self,
        *,
        observation_id: str,
        proposition_id: str,
        source_id: str,
        score: float,
        split: str,
    ) -> TrainingObservationReceipt:
        """Record a bounded unverified TRAIN report, idempotent by ID/payload.

        The score is retained only as a raw caller report; the scheduling
        heuristic uses attempt count, *never* score. Passing
        ``split='heldout'`` or ``'dev'`` is rejected before state changes.
        """
        if type(split) is not str or split != "train":
            raise ValueError("only split='train' observations may enter curriculum")
        observation_id = _text(observation_id, "observation_id", _MAX_ID_CHARS)
        proposition_id = _text(proposition_id, "proposition_id", _MAX_ID_CHARS)
        source_id = _text(source_id, "source_id", _MAX_ID_CHARS)
        if type(score) not in (int, float) or not 0 <= score <= 1 or not isfinite(float(score)):
            raise ValueError("score must be a finite number from 0 to 1")
        observation = TrainingObservation(
            observation_id=observation_id,
            proposition_id=proposition_id,
            source_id=source_id,
            score=float(score),
        )

        with self._lock:
            if proposition_id not in self._questions:
                raise ValueError("unknown proposition_id")
            old = self._observations.get(observation_id)
            if old is not None:
                if old != observation:
                    raise ValueError("observation_id reused with changed payload")
                return TrainingObservationReceipt(observation=old, replayed=True)
            if len(self._observations) >= self._max_observations:
                raise ValueError("training observation capacity exceeded")
            self._observations[observation_id] = observation
            self._attempts[proposition_id] += 1
            return TrainingObservationReceipt(observation=observation, replayed=False)

    def snapshot(self) -> dict[str, object]:
        """Detached JSON-compatible report; no verified provenance or restore."""
        with self._lock:
            return {
                "schema": SCHEMA,
                "claim_ceiling": CLAIM_CEILING,
                "questions": [
                    {
                        "proposition_id": q.proposition_id,
                        "status": q.status,
                        "question": q.question,
                        "local_attempt_count": self._attempts[q.proposition_id],
                    }
                    for q in sorted(self._questions.values(), key=lambda q: q.proposition_id)
                ],
                "observations": [
                    {
                        "observation_id": o.observation_id,
                        "proposition_id": o.proposition_id,
                        "source_id": o.source_id,
                        "score": o.score,
                        "split": o.split,
                    }
                    for o in self._observations.values()
                ],
            }
