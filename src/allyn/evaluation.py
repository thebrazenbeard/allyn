"""Frozen, caller-declared teaching specimens with explicit leakage checks.

This module provides *structural* separation, not independent custody. Its
source and record labels are unverified, and a caller can choose lessons after
looking at the held-out data before passing anything to this module. Nothing
here proves semantic transfer or authenticates a test corpus.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json

from .teaching import TransferResult, compare_teaching


SCHEMA = "allyn.frozen-evaluation.v0"
CLAIM_CEILING = "CALLER_DECLARED_SPLIT_NOT_INDEPENDENT_CUSTODY"
_MAX_EXAMPLES_PER_PARTITION = 2_048
_MAX_TOKENS_PER_EXAMPLE = 10_000
_MAX_TOTAL_TOKENS = 20_000
_MAX_TOKEN_CHARS = 4_096
_MAX_TOTAL_UTF8_BYTES = 1_000_000
_MAX_ID_CHARS = 256


@dataclass(frozen=True, slots=True)
class SourcedExample:
    """A declared source group and record reference, not verified provenance."""

    source_id: str
    record_id: str
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenSpecimen:
    """Detached immutable split with a reproducible (unauthenticated) digest."""

    student: tuple[SourcedExample, ...]
    lessons: tuple[SourcedExample, ...]
    heldout: tuple[SourcedExample, ...]
    order: int
    allow_shared_source_groups: bool
    specimen_sha256: str
    schema: str = SCHEMA
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True, slots=True)
class FrozenEvaluationResult:
    """Comparison against a structurally frozen split, not model qualification."""

    specimen_sha256: str
    transfer: TransferResult
    claim_ceiling: str = CLAIM_CEILING


@dataclass(slots=True)
class _Budget:
    tokens: int = 0
    utf8_bytes: int = 0


def _id(value: object, name: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > _MAX_ID_CHARS:
        raise ValueError(f"{name} must be nonblank text up to {_MAX_ID_CHARS} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be UTF-8 encodable") from exc
    return value


def _partition(
    values: Iterable[SourcedExample], name: str, budget: _Budget, *, required: bool
) -> tuple[SourcedExample, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of SourcedExample")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of SourcedExample") from exc

    normalized: list[SourcedExample] = []
    keys: set[tuple[str, str]] = set()
    for index, item in enumerate(iterator):
        if index >= _MAX_EXAMPLES_PER_PARTITION:
            raise ValueError(f"{name} exceeds {_MAX_EXAMPLES_PER_PARTITION} examples")
        if type(item) is not SourcedExample:
            raise TypeError(f"{name}[{index}] must be SourcedExample")
        source = _id(item.source_id, f"{name}[{index}].source_id")
        record = _id(item.record_id, f"{name}[{index}].record_id")
        key = (source, record)
        if key in keys:
            raise ValueError(f"{name} contains duplicate source/record reference")
        keys.add(key)
        if isinstance(item.tokens, (str, bytes, bytearray)):
            raise TypeError(f"{name}[{index}].tokens must be token iterable")
        try:
            token_iterator = iter(item.tokens)
        except TypeError as exc:
            raise TypeError(f"{name}[{index}].tokens must be token iterable") from exc

        words: list[str] = []
        for token_index, token in enumerate(token_iterator):
            if token_index >= _MAX_TOKENS_PER_EXAMPLE:
                raise ValueError(f"{name}[{index}] exceeds {_MAX_TOKENS_PER_EXAMPLE} tokens")
            if type(token) is not str or not token or len(token) > _MAX_TOKEN_CHARS:
                raise ValueError(f"{name}[{index}] contains invalid or oversized token")
            try:
                size = len(token.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ValueError(f"{name}[{index}] has invalid UTF-8 token") from exc
            if budget.tokens >= _MAX_TOTAL_TOKENS:
                raise ValueError("frozen specimen total token capacity exceeded")
            if budget.utf8_bytes + size > _MAX_TOTAL_UTF8_BYTES:
                raise ValueError("frozen specimen total UTF-8 byte capacity exceeded")
            words.append(token)
            budget.tokens += 1
            budget.utf8_bytes += size
        if len(words) < 2:
            raise ValueError(f"{name}[{index}] needs context and target")
        normalized.append(SourcedExample(source, record, tuple(words)))

    if required and not normalized:
        raise ValueError(f"{name} must contain an example")
    return tuple(sorted(normalized, key=lambda x: (x.source_id, x.record_id)))


def _serializable(records: tuple[SourcedExample, ...]) -> list[dict[str, object]]:
    return [
        {"source_id": item.source_id, "record_id": item.record_id, "tokens": item.tokens}
        for item in records
    ]


def freeze_specimen(
    student: Iterable[SourcedExample],
    lessons: Iterable[SourcedExample],
    heldout: Iterable[SourcedExample],
    *,
    order: int = 3,
    allow_shared_source_groups: bool = False,
) -> FrozenSpecimen:
    """Bound and freeze declared splits, rejecting exact examples crossing splits.

    By default a source group may appear in only one partition. Set
    ``allow_shared_source_groups=True`` only when a separate provenance review
    justifies shared groups; exact example and record overlap still fails.
    This check cannot detect paraphrase leakage or dishonest source labels.
    """
    if type(order) is not int or not 1 <= order <= 5:
        raise ValueError("order must be integer 1 through 5")
    if type(allow_shared_source_groups) is not bool:
        raise ValueError("allow_shared_source_groups must be bool")
    budget = _Budget()
    partitions = (
        _partition(student, "student", budget, required=True),
        _partition(lessons, "lessons", budget, required=False),
        _partition(heldout, "heldout", budget, required=True),
    )
    names = ("student", "lessons", "heldout")
    # Repeated held-out specimens can artificially inflate an accuracy score.
    targets = [entry.tokens for entry in partitions[2]]
    if len(set(targets)) != len(targets):
        raise ValueError("heldout contains duplicate exact examples")
    for left in range(3):
        for right in range(left + 1, 3):
            l, r = partitions[left], partitions[right]
            if {x.tokens for x in l} & {x.tokens for x in r}:
                raise ValueError(f"exact example overlap between {names[left]} and {names[right]}")
            if {(x.source_id, x.record_id) for x in l} & {(x.source_id, x.record_id) for x in r}:
                raise ValueError(f"record ID overlap between {names[left]} and {names[right]}")
            if not allow_shared_source_groups and {x.source_id for x in l} & {x.source_id for x in r}:
                raise ValueError(f"source-group overlap between {names[left]} and {names[right]}")

    document = {
        "schema": SCHEMA,
        "order": order,
        "allow_shared_source_groups": allow_shared_source_groups,
        "student": _serializable(partitions[0]),
        "lessons": _serializable(partitions[1]),
        "heldout": _serializable(partitions[2]),
    }
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return FrozenSpecimen(*partitions, order, allow_shared_source_groups, hashlib.sha256(encoded).hexdigest())


def evaluate_frozen(specimen: FrozenSpecimen) -> FrozenEvaluationResult:
    """Compare two students after revalidating the public specimen constructor.

    Neither the frozen descriptor nor its digest prevents prior peeking at the
    held-out split. The evaluator does not update either model with heldout.
    """
    if type(specimen) is not FrozenSpecimen or specimen.schema != SCHEMA or specimen.claim_ceiling != CLAIM_CEILING:
        raise ValueError("invalid frozen evaluation specimen")
    checked = freeze_specimen(
        specimen.student, specimen.lessons, specimen.heldout,
        order=specimen.order,
        allow_shared_source_groups=specimen.allow_shared_source_groups,
    )
    if checked.specimen_sha256 != specimen.specimen_sha256:
        raise ValueError("frozen specimen digest mismatch")
    transfer = compare_teaching(
        (row.tokens for row in checked.student),
        (row.tokens for row in checked.lessons),
        (row.tokens for row in checked.heldout),
        order=checked.order,
    )
    return FrozenEvaluationResult(specimen_sha256=checked.specimen_sha256, transfer=transfer)
