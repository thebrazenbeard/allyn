"""Bounded Socratic question plans from *unverified* inquiry gaps.

Plans are fixed, non-executing prompts for possible future teachers. They do
not repeat caller-supplied instructions as model instructions, adjudicate the
underlying proposition, authenticate a witness, train a model, or contact a
student. An inquiry status is a description of reports, not verified truth.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .curriculum import CurriculumSelection
from .inquiry import InquiryLedger, InquiryQuestion


SCHEMA = "allyn.socratic-plan.v0"
CLAIM_CEILING = "UNVERIFIED_INQUIRY_PROMPT_TEMPLATE_NOT_TEACHING_OUTCOME"
_MAX_PLANS = 64
_MAX_INPUT_ROWS = 128
_MAX_INQUIRY_EVENTS = 512
_MAX_TEXT_LENGTH = 4_096
_MAX_ID_LENGTH = 256
_STATUSES = frozenset(("UNASSESSED", "REPORTED_SUPPORT", "REPORTED_CHALLENGE", "CONTESTED"))
_PRIORITY = {
    "CONTESTED": 0,
    "UNASSESSED": 1,
    "REPORTED_CHALLENGE": 2,
    "REPORTED_SUPPORT": 3,
}


def _text(value: object, label: str, length: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > length:
        raise ValueError(f"{label} must be nonblank text up to {length} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be UTF-8 encodable") from exc
    return value


def _question(value: object) -> InquiryQuestion:
    if type(value) is not InquiryQuestion:
        raise TypeError("question must be an InquiryQuestion")
    _text(value.proposition_id, "proposition_id", _MAX_ID_LENGTH)
    _text(value.question, "unverified inquiry hint", _MAX_TEXT_LENGTH)
    if type(value.status) is not str or value.status not in _STATUSES:
        raise ValueError("unsupported reported inquiry status")
    return value


_COMMON_QUESTIONS = (
    "Which assumptions are required for this proposition to be true?",
    "What observation would falsify the proposition?",
    "What counterexample or alternative explanation should be tested?",
    "Which independently sourced evidence would change the current assessment?",
)
_LESSON_STEPS = (
    "Ask the learner to restate the proposition as a testable hypothesis.",
    "Ask the learner for an assumption, a counterexample, and a falsifier.",
    "Compare proposed evidence against alternatives; record remaining uncertainty.",
)
_STATUS_QUESTION = {
    "UNASSESSED": "What evidence is needed before drawing any conclusion?",
    "REPORTED_SUPPORT": "How could the reported support be independently challenged?",
    "REPORTED_CHALLENGE": "What evidence would distinguish a valid challenge from a mistaken one?",
    "CONTESTED": "What independent observation could distinguish the conflicting reports?",
}


@dataclass(frozen=True, slots=True)
class SocraticPlan:
    """Immutable generic inquiry plan; source references are caller-declared."""

    proposition_id: str
    reported_status: str
    planned_questions: tuple[str, ...]
    lesson_steps: tuple[str, ...]
    claimed_source_ids: tuple[str, ...]
    origin: Literal["INQUIRY", "QUESTION_LIST", "TRAIN_CURRICULUM"]
    local_attempt_count: int | None = None
    provenance_status: str = "CALLER_DECLARED_UNVERIFIED"
    claim_ceiling: str = CLAIM_CEILING
    schema: str = SCHEMA


def _plan(
    question: InquiryQuestion,
    *,
    sources: Iterable[str] = (),
    origin: Literal["INQUIRY", "QUESTION_LIST", "TRAIN_CURRICULUM"],
    attempts: int | None = None,
) -> SocraticPlan:
    checked = _question(question)
    claimed = tuple(sorted(set(_text(source, "source_id", _MAX_ID_LENGTH) for source in sources)))
    if len(claimed) > _MAX_INQUIRY_EVENTS:
        raise ValueError("source reference capacity exceeded")
    if attempts is not None and (type(attempts) is not int or not 0 <= attempts <= 10_000):
        raise ValueError("local_attempt_count must be a bounded nonnegative integer")
    return SocraticPlan(
        proposition_id=checked.proposition_id,
        reported_status=checked.status,
        planned_questions=(_STATUS_QUESTION[checked.status],) + _COMMON_QUESTIONS,
        lesson_steps=_LESSON_STEPS,
        claimed_source_ids=claimed,
        origin=origin,
        local_attempt_count=attempts,
    )


def plans_from_questions(
    questions: Iterable[InquiryQuestion], *, max_plans: int = 32
) -> tuple[SocraticPlan, ...]:
    """Create stable plans without elevating the untrusted question text.

    Exact duplicate questions are idempotent; differing payloads under one
    proposition ID are rejected. The limit also applies to unique plans.
    """
    if type(max_plans) is not int or not 1 <= max_plans <= _MAX_PLANS:
        raise ValueError(f"max_plans must be 1 through {_MAX_PLANS}")
    if isinstance(questions, (str, bytes, bytearray)):
        raise TypeError("questions must be an iterable of InquiryQuestion")
    try:
        iterator = iter(questions)
    except TypeError as exc:
        raise TypeError("questions must be an iterable of InquiryQuestion") from exc
    admitted: dict[str, InquiryQuestion] = {}
    for index, value in enumerate(iterator):
        if index >= _MAX_INPUT_ROWS:
            raise ValueError("inquiry input row capacity exceeded")
        question = _question(value)
        previous = admitted.get(question.proposition_id)
        if previous is not None and previous != question:
            raise ValueError("proposition_id reused with changed inquiry report")
        admitted[question.proposition_id] = question
        if len(admitted) > max_plans:
            raise ValueError("unique inquiry plan capacity exceeded")
    return tuple(_plan(admitted[key], origin="QUESTION_LIST") for key in sorted(admitted))


def plans_from_inquiry(
    inquiry: InquiryLedger | None, *, max_plans: int = 32
) -> tuple[SocraticPlan, ...]:
    """Plan using recorded report status and unverified claimed source IDs.

    InquiryLedger.questions performs a per-proposition scan of its events, so
    this adapter refuses large histories before invoking it. No state changes.
    """
    if type(max_plans) is not int or not 1 <= max_plans <= _MAX_PLANS:
        raise ValueError(f"max_plans must be 1 through {_MAX_PLANS}")
    if inquiry is None:
        return ()
    if not isinstance(inquiry, InquiryLedger):
        raise TypeError("inquiry must be an InquiryLedger or None")
    events = inquiry.events
    if len(events) > _MAX_INQUIRY_EVENTS:
        raise ValueError("inquiry history exceeds planning event capacity")
    questions = inquiry.questions()
    if len(questions) > max_plans:
        raise ValueError("inquiry has more distinct propositions than max_plans")
    source_map: dict[str, set[str]] = {}
    for event in events:
        source_map.setdefault(event.proposition_id, set()).add(event.source_id)
    return tuple(
        _plan(question, sources=source_map.get(question.proposition_id, ()), origin="INQUIRY")
        for question in questions
    )


def plan_from_selection(selection: CurriculumSelection | None) -> SocraticPlan | None:
    """Turn a train-only scheduling hint into a non-executing lesson outline."""
    if selection is None:
        return None
    if type(selection) is not CurriculumSelection:
        raise TypeError("selection must be CurriculumSelection or None")
    question = _question(selection.question)
    if type(selection.priority_rank) is not int or selection.priority_rank != _PRIORITY[question.status]:
        raise ValueError("curriculum priority is inconsistent with reported status")
    return _plan(question, origin="TRAIN_CURRICULUM", attempts=selection.local_attempt_count)
