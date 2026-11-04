"""Bounded in-memory request deduplication for unauthenticated inquiry reports.

Request IDs and source IDs are supplied by callers and have *no* identity or
authorization meaning. This adapter prevents accidental repeated delivery of
one request from inflating a local raw-report count; it cannot stop a caller
from issuing new IDs, impersonating a source, or replaying across restarts.
The SHA-256 receipt checks payload consistency, not authorship or truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from threading import Lock
from typing import Any

from .inquiry import InquiryAssessment, InquiryEvent, InquiryLedger, InquiryQuestion


SCHEMA = "allyn.admission.v0"
CLAIM_CEILING = "LOCAL_IDEMPOTENCY_UNAUTHENTICATED_REPORTS_ONLY"
_MAX_REQUESTS = 10_000
_MAX_ID_CHARS = 256
_MAX_CLAIM_CHARS = 4_096
_STANCES = frozenset(("supports", "challenges"))
_EVIDENCE_KINDS = frozenset(("observation", "inference", "hypothesis"))


def _text(value: object, field: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be nonblank text up to {maximum} characters")
    return value


@dataclass(frozen=True, slots=True)
class AdmissionReceipt:
    """One local request attempt; replayed does not signify an extra report."""

    request_id: str
    event: InquiryEvent
    payload_sha256: str
    replayed: bool
    claim_ceiling: str = CLAIM_CEILING


class AdmissionLedger:
    """Atomically admit bounded unique requests to a private InquiryLedger.

    It does not authenticate the declared source, establish independence of
    witnesses, verify truth, durably persist history, or grant correction
    authority. All assessed counts remain raw, unverified report counts.
    """

    def __init__(self, *, max_requests: int = _MAX_REQUESTS) -> None:
        if type(max_requests) is not int or not 1 <= max_requests <= _MAX_REQUESTS:
            raise ValueError(f"max_requests must be an integer from 1 to {_MAX_REQUESTS}")
        self._max_requests = max_requests
        self._inquiry = InquiryLedger()
        self._requests: dict[str, tuple[str, InquiryEvent]] = {}
        self._lock = Lock()

    @property
    def events(self) -> tuple[InquiryEvent, ...]:
        with self._lock:
            return self._inquiry.events

    @property
    def admitted_count(self) -> int:
        with self._lock:
            return len(self._requests)

    def submit(
        self,
        *,
        request_id: str,
        proposition_id: str,
        claim: str,
        source_id: str,
        stance: str,
        evidence_kind: str,
        corrects_event_id: str | None = None,
    ) -> AdmissionReceipt:
        """Admit once per request ID; reject changed-payload reuse atomically.

        A duplicate exact request returns the original event and replayed=True,
        including when at configured capacity. New request IDs are counted as
        new *reports*, even when their asserted evidence is identical.
        """
        _text(request_id, "request_id", _MAX_ID_CHARS)
        _text(proposition_id, "proposition_id", _MAX_ID_CHARS)
        _text(claim, "claim", _MAX_CLAIM_CHARS)
        _text(source_id, "source_id", _MAX_ID_CHARS)
        if type(stance) is not str or stance not in _STANCES:
            raise ValueError("stance must be supports or challenges")
        if type(evidence_kind) is not str or evidence_kind not in _EVIDENCE_KINDS:
            raise ValueError("invalid evidence_kind")
        if corrects_event_id is not None:
            _text(corrects_event_id, "corrects_event_id", _MAX_ID_CHARS)

        payload = {
            "claim": claim,
            "corrects_event_id": corrects_event_id,
            "evidence_kind": evidence_kind,
            "proposition_id": proposition_id,
            "source_id": source_id,
            "stance": stance,
        }
        encoded = dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        digest = sha256(encoded).hexdigest()

        with self._lock:
            previous = self._requests.get(request_id)
            if previous is not None:
                old_digest, event = previous
                if old_digest != digest:
                    raise ValueError("request_id reused with different payload")
                return AdmissionReceipt(request_id, event, digest, replayed=True)

            if len(self._requests) >= self._max_requests:
                raise ValueError("admission request capacity exceeded")
            event = self._inquiry.record(
                proposition_id=proposition_id,
                claim=claim,
                source_id=source_id,
                stance=stance,
                evidence_kind=evidence_kind,
                corrects_event_id=corrects_event_id,
            )
            self._requests[request_id] = (digest, event)
            return AdmissionReceipt(request_id, event, digest, replayed=False)

    def assess(self, proposition_id: str) -> InquiryAssessment:
        with self._lock:
            return self._inquiry.assess(proposition_id)

    def questions(self) -> tuple[InquiryQuestion, ...]:
        with self._lock:
            return self._inquiry.questions()

    def snapshot(self) -> dict[str, Any]:
        """Detached JSON-compatible export, not verified/restorable durable state."""
        with self._lock:
            return {
                "schema": SCHEMA,
                "claim_ceiling": CLAIM_CEILING,
                "max_requests": self._max_requests,
                "inquiry": self._inquiry.snapshot(),
                "requests": [
                    {"request_id": request_id, "payload_sha256": digest, "event_id": event.event_id}
                    for request_id, (digest, event) in sorted(self._requests.items())
                ],
            }
