"""Source-attributed, prequential next-token evaluation for ALLYN.

Each successful event measures one prediction made *before* the model observes
the corresponding example. The ledger does not train models, determine truth,
establish generalization, or authorize an update to durable cognitive state.
The caller owns source admission, privacy, and any persistent record storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Protocol


CLAIM_CEILING = "OBSERVED_NEXT_TOKEN_MATCH_ONLY"


class NextTokenModel(Protocol):
    """Minimal interface for an independently owned incremental learner."""

    def predict_next(self, context: Sequence[str]) -> str | None: ...

    def observe(self, tokens: Sequence[str]) -> None: ...


@dataclass(frozen=True, slots=True, repr=False)
class PrequentialEvent:
    """Immutable evidence for one completed prediction-then-observation step.

    Source IDs are caller-supplied references, not proof of permission or
    authenticity. Event IDs are monotonically increasing within one ledger,
    and are not globally unique across independent processes or restarts.
    """

    event_id: str
    sequence_number: int
    context: tuple[str, ...]
    target: str
    prediction: str | None
    correct: bool
    source_id: str
    claim_ceiling: str = CLAIM_CEILING


class PrequentialLedger:
    """In-memory evaluator with immutable event readback and bounded claims.

    The lock serializes predictions, observations, and event numbering within
    this instance. No evidence event is committed if either model operation
    raises. This is *ledger* atomicity only: an arbitrary model may partially
    mutate itself before ``observe`` fails, and this ledger cannot undo that
    change. This in-memory API also provides no durable write-ahead record or
    crash-recovery guarantee; a persistent implementation must separately
    address the prediction/update transaction boundary.
    """

    def __init__(self) -> None:
        self._events: list[PrequentialEvent] = []
        self._lock = Lock()

    @property
    def events(self) -> tuple[PrequentialEvent, ...]:
        """Return an immutable snapshot in event order."""

        with self._lock:
            return tuple(self._events)

    def predict_then_observe(
        self,
        model: NextTokenModel,
        tokens: Sequence[str],
        source_id: str,
    ) -> PrequentialEvent:
        """Predict the last token using its prefix, then present the example.

        One-token examples are valid and use the empty prefix. Empty or
        malformed token sequences are rejected without calling the model.
        ``None`` prediction means abstention/unseen context and is scored as
        an incorrect token prediction, not evidence of a false proposition.
        """

        if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
            raise TypeError("tokens must be a sequence of token strings")
        if not tokens:
            raise ValueError("at least one token is required")
        if any(type(token) is not str or not token for token in tokens):
            raise ValueError("tokens must contain only nonempty strings")
        if type(source_id) is not str or not source_id.strip():
            raise ValueError("source_id must be a nonempty source reference")

        example = tuple(tokens)
        context, target = example[:-1], example[-1]

        with self._lock:
            # No model updates, observations, or evidence writes before this.
            prediction = model.predict_next(context)
            if prediction is not None and type(prediction) is not str:
                raise TypeError("predict_next must return a token string or None")

            model.observe(example)

            number = len(self._events) + 1
            event = PrequentialEvent(
                event_id=f"allyn-pq-{number:012d}",
                sequence_number=number,
                context=context,
                target=target,
                prediction=prediction,
                correct=prediction == target,
                source_id=source_id,
            )
            self._events.append(event)
            return event

    def summary(self) -> dict[str, int | float | None]:
        """Score observed next-token exact matches only.

        A zero-event ledger reports unknown accuracy rather than 0% accuracy.
        This metric does not prove language mastery, novelty, or teaching.
        """

        with self._lock:
            total = len(self._events)
            correct = sum(event.correct for event in self._events)
            return {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else None,
            }
