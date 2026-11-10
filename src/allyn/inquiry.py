"""Bounded, source-attributed inquiry without authority or truth promotion.

Each record is a *caller-supplied report*. Even an ``observation`` label does
not authenticate a witness or establish that the proposition is true. This
module never invokes external tools or changes state outside its own instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Stance = Literal["supports", "challenges"]
EvidenceKind = Literal["observation", "inference", "hypothesis"]
AssessmentStatus = Literal[
    "UNASSESSED", "REPORTED_SUPPORT", "REPORTED_CHALLENGE", "CONTESTED"
]

SCHEMA = "allyn.inquiry.v0"
CLAIM_CEILING = "UNAUTHENTICATED_REPORTS_NOT_TRUTH"
_MAX_EVENTS = 10_000
_MAX_ID_LENGTH = 256
_MAX_CLAIM_LENGTH = 4096
_STANCES = frozenset(("supports", "challenges"))
_EVIDENCE_KINDS = frozenset(("observation", "inference", "hypothesis"))


def _nonempty_string(value: object, name: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be nonblank text of at most {maximum} characters")
    return value


@dataclass(frozen=True, slots=True)
class InquiryEvent:
    event_id: str
    sequence_number: int
    proposition_id: str
    claim: str
    source_id: str
    stance: Stance
    evidence_kind: EvidenceKind
    corrects_event_id: str | None
    correction_status: Literal["NOT_APPLICABLE", "UNVERIFIED"] = "NOT_APPLICABLE"
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True, slots=True)
class InquiryAssessment:
    """Counts of raw, unverified reports, not independent corroboration."""

    proposition_id: str
    status: AssessmentStatus
    active_support_count: int
    active_challenge_count: int
    source_ids: tuple[str, ...]
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True, slots=True)
class InquiryQuestion:
    proposition_id: str
    status: AssessmentStatus
    question: str


class InquiryLedger:
    """In-memory append-only reports, explicit correction links, and questions.

    Each proposition ID is bound to its first exact claim string; changed claims
    must receive a different ID. Corrections are UNVERIFIED challenge/support
    reports and never suppress the original: a caller can spoof ``source_id``.
    Multiple reports may refer to the same correction target; untrusted actors
    cannot reserve an exclusive correction slot. Counts are raw unverified
    reports and must not be mistaken for independent corroboration.
    This object has no cross-process durability or concurrent-call lock.
    """

    def __init__(self) -> None:
        self._events: list[InquiryEvent] = []

    @property
    def events(self) -> tuple[InquiryEvent, ...]:
        """Return a detached, immutable event-list view."""
        return tuple(self._events)

    def record(
        self,
        *,
        proposition_id: str,
        claim: str,
        source_id: str,
        stance: Stance,
        evidence_kind: EvidenceKind,
        corrects_event_id: str | None = None,
    ) -> InquiryEvent:
        """Append an untrusted report after validating all input and capacity."""
        _nonempty_string(proposition_id, "proposition_id", _MAX_ID_LENGTH)
        _nonempty_string(claim, "claim", _MAX_CLAIM_LENGTH)
        _nonempty_string(source_id, "source_id", _MAX_ID_LENGTH)
        if type(stance) is not str or stance not in _STANCES:
            raise ValueError("stance must be supports or challenges")
        if type(evidence_kind) is not str or evidence_kind not in _EVIDENCE_KINDS:
            raise ValueError("invalid evidence_kind")
        if len(self._events) >= _MAX_EVENTS:
            raise ValueError("inquiry event capacity exceeded")

        first = next(
            (event for event in self._events if event.proposition_id == proposition_id),
            None,
        )
        if first is not None and first.claim != claim:
            raise ValueError("proposition_id is bound to a different canonical claim")

        if corrects_event_id is not None:
            if type(corrects_event_id) is not str:
                raise ValueError("corrects_event_id must be text or None")
            previous = next(
                (event for event in self._events if event.event_id == corrects_event_id),
                None,
            )
            if previous is None:
                raise ValueError("correction target not found")
            if previous.proposition_id != proposition_id:
                raise ValueError("correction must address the same proposition")
            if previous.source_id != source_id:
                raise ValueError("correction requires the same claimed source")

        number = len(self._events) + 1
        event = InquiryEvent(
            event_id=f"allyn-inq-{number:012d}",
            sequence_number=number,
            proposition_id=proposition_id,
            claim=claim,
            source_id=source_id,
            stance=stance,
            evidence_kind=evidence_kind,
            corrects_event_id=corrects_event_id,
            correction_status=(
                "UNVERIFIED" if corrects_event_id is not None else "NOT_APPLICABLE"
            ),
        )
        self._events.append(event)
        return event

    def assess(self, proposition_id: str) -> InquiryAssessment:
        """Classify raw unverified reports, not independent supporting witnesses."""
        _nonempty_string(proposition_id, "proposition_id", _MAX_ID_LENGTH)
        # No unauthenticated correction has the authority to hide prior reports.
        active = tuple(
            event for event in self._events if event.proposition_id == proposition_id
        )
        supports = sum(event.stance == "supports" for event in active)
        challenges = len(active) - supports
        if supports and challenges:
            status: AssessmentStatus = "CONTESTED"
        elif supports:
            status = "REPORTED_SUPPORT"
        elif challenges:
            status = "REPORTED_CHALLENGE"
        else:
            status = "UNASSESSED"
        return InquiryAssessment(
            proposition_id=proposition_id,
            status=status,
            active_support_count=supports,
            active_challenge_count=challenges,
            source_ids=tuple(sorted({event.source_id for event in active})),
        )

    def questions(self) -> tuple[InquiryQuestion, ...]:
        """Expose contested/uncorroborated propositions for further inquiry."""
        propositions = sorted({event.proposition_id for event in self._events})
        questions: list[InquiryQuestion] = []
        for proposition_id in propositions:
            assessment = self.assess(proposition_id)
            if assessment.status == "CONTESTED":
                prompt = "Which independent observations could resolve conflicting reports?"
            elif assessment.status == "UNASSESSED":
                prompt = "Is there any active evidence for this proposition?"
            else:
                prompt = "What independent evidence could confirm or falsify this report?"
            questions.append(
                InquiryQuestion(
                    proposition_id=proposition_id,
                    status=assessment.status,
                    question=prompt,
                )
            )
        return tuple(questions)

    def snapshot(self) -> dict[str, object]:
        """Return a detached JSON-compatible history, not authenticated state."""
        return {
            "schema": SCHEMA,
            "claim_ceiling": CLAIM_CEILING,
            "events": [
                {
                    "event_id": event.event_id,
                    "sequence_number": event.sequence_number,
                    "proposition_id": event.proposition_id,
                    "claim": event.claim,
                    "source_id": event.source_id,
                    "stance": event.stance,
                    "evidence_kind": event.evidence_kind,
                    "corrects_event_id": event.corrects_event_id,
                    "correction_status": event.correction_status,
                }
                for event in self._events
            ],
        }
