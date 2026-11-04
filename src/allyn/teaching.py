"""Bounded, deterministic comparison of an n-gram lesson with an un-taught control.

This measures transfer of *observed token patterns*, not semantic understanding,
novel generalization, model-weight updates, consciousness, or autonomy.
Inputs are caller-supplied and unverified; held-out specimens are evaluated
without admitting them to either student's training history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Iterable, Sequence

from .language import NGramLearner


_SCHEMA = "allyn.teaching-comparison.v0"
_CLAIM_CEILING = "OBSERVED_PATTERN_TRANSFER_ONLY"
_SOURCE_LABELS = ("caller_supplied_unverified",)
_MAX_EXAMPLES_PER_GROUP = 2_048
_MAX_TOKENS_PER_EXAMPLE = 10_000  # NGramLearner's maximum utterance size
_MAX_TOKEN_CHARS = 4_096  # NGramLearner's maximum token size
_MAX_TOTAL_TOKENS = 20_000
_MAX_TOTAL_UTF8_BYTES = 1_000_000


@dataclass(slots=True)
class _IngressBudget:
    """Shared across all groups before a specimen is hashed or learned."""

    tokens: int = 0
    utf8_bytes: int = 0


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Immutable comparison receipt for a single, fully specified specimen."""

    baseline_accuracy: float
    taught_accuracy: float
    gain: float
    evaluated_count: int
    student_examples_count: int
    lesson_examples_count: int
    specimen_sha256: str
    source_labels: tuple[str, ...] = _SOURCE_LABELS
    claim_ceiling: str = _CLAIM_CEILING
    schema: str = _SCHEMA


def _examples(
    values: Iterable[Sequence[str]],
    *,
    label: str,
    allow_empty: bool,
    budget: _IngressBudget,
) -> tuple[tuple[str, ...], ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be an iterable of token sequences")
    try:
        examples_iter = iter(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of token sequences") from exc

    normalized: list[tuple[str, ...]] = []
    for index, example in enumerate(examples_iter):
        if index >= _MAX_EXAMPLES_PER_GROUP:
            raise ValueError(f"{label} exceeds {_MAX_EXAMPLES_PER_GROUP} examples")
        if isinstance(example, (str, bytes, bytearray)):
            raise TypeError(f"{label}[{index}] must be a token sequence, not text")
        try:
            token_iter = iter(example)
        except TypeError as exc:
            raise TypeError(f"{label}[{index}] must be a token sequence") from exc

        bounded_tokens: list[str] = []
        for token_index, token in enumerate(token_iter):
            if token_index >= _MAX_TOKENS_PER_EXAMPLE:
                raise ValueError(f"{label}[{index}] exceeds {_MAX_TOKENS_PER_EXAMPLE} tokens")
            if type(token) is not str or not token or len(token) > _MAX_TOKEN_CHARS:
                raise ValueError(f"{label}[{index}] contains an invalid or oversized token")
            if budget.tokens >= _MAX_TOTAL_TOKENS:
                raise ValueError("teaching specimen total token capacity exceeded")
            try:
                token_bytes = len(token.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ValueError(f"{label}[{index}] contains an invalid UTF-8 token") from exc
            if budget.utf8_bytes + token_bytes > _MAX_TOTAL_UTF8_BYTES:
                raise ValueError("teaching specimen total UTF-8 byte capacity exceeded")
            budget.tokens += 1
            budget.utf8_bytes += token_bytes
            bounded_tokens.append(token)

        if len(bounded_tokens) < 2:
            raise ValueError(f"{label}[{index}] needs context and a target token")
        normalized.append(tuple(bounded_tokens))
    if not normalized and not allow_empty:
        raise ValueError(f"{label} must contain at least one example")
    return tuple(normalized)


def _accuracy(learner: NGramLearner, cases: tuple[tuple[str, ...], ...]) -> float:
    correct = sum(
        learner.predict_next(example[:-1]) == example[-1]
        for example in cases
    )
    return correct / len(cases)


def compare_teaching(
    student_examples: Iterable[Sequence[str]],
    lesson_examples: Iterable[Sequence[str]],
    heldout_examples: Iterable[Sequence[str]],
    order: int = 3,
) -> TransferResult:
    """Measure whether caller-supplied lessons improve next-token accuracy.

    Two independent learners receive identical initial observations. Only the
    taught learner sees lessons; both predict the *same* held-out specimens,
    which are never used as training input. A negative gain is preserved.

    A held-out example's final token is its target and all previous tokens are
    the prediction context. No independent source or semantic verification is
    performed; exact patterns may overlap between lessons and held-out cases.
    """
    if type(order) is not int or not 1 <= order <= 5:
        raise ValueError("order must be an integer from 1 to 5")
    budget = _IngressBudget()
    student = _examples(student_examples, label="student_examples", allow_empty=False, budget=budget)
    lessons = _examples(lesson_examples, label="lesson_examples", allow_empty=True, budget=budget)
    heldout = _examples(heldout_examples, label="heldout_examples", allow_empty=False, budget=budget)

    specimen = {
        "schema": _SCHEMA,
        "order": order,
        "student_examples": student,
        "lesson_examples": lessons,
        "heldout_examples": heldout,
    }
    canonical = json.dumps(
        specimen, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()

    baseline = NGramLearner(order=order)
    taught = NGramLearner(order=order)
    for example in student:
        baseline.observe(example)
        taught.observe(example)
    for example in lessons:
        taught.observe(example)

    baseline_accuracy = _accuracy(baseline, heldout)
    taught_accuracy = _accuracy(taught, heldout)
    return TransferResult(
        baseline_accuracy=baseline_accuracy,
        taught_accuracy=taught_accuracy,
        gain=taught_accuracy - baseline_accuracy,
        evaluated_count=len(heldout),
        student_examples_count=len(student),
        lesson_examples_count=len(lessons),
        specimen_sha256=digest,
    )
