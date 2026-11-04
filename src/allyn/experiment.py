"""Four-arm, offline teaching comparison with explicit structural boundaries.

All inputs, source groups, control lessons, and held-out targets are supplied
by the same caller. This module can reproduce a controlled *pattern* comparison;
it cannot independently protect its answer key, authenticate sources, prevent
adaptive re-querying, or establish semantic teaching effectiveness.

The genuine lesson is *preselected* by the caller. No lesson selection takes
place here, and held-out examples are never used to update a learner.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json

from .evaluation import SourcedExample, _Budget, _partition, _serializable
from .language import NGramLearner


SCHEMA = "allyn.four-arm-experiment.v0"
CLAIM_CEILING = "CALLER_DECLARED_FOUR_ARM_PATTERN_COMPARISON_NOT_INDEPENDENT_EVALUATION"
_ARM_NAMES = ("baseline", "genuine", "irrelevant", "misleading")
_MIN_DESIGN_HELDOUT = 100
_MIN_DESIGN_GROUPS = 3


@dataclass(frozen=True, slots=True)
class FourArmSpecimen:
    """Immutable, digest-bound declared inputs, not sealed evaluator custody."""

    student: tuple[SourcedExample, ...]
    genuine: tuple[SourcedExample, ...]
    irrelevant: tuple[SourcedExample, ...]
    misleading: tuple[SourcedExample, ...]
    heldout: tuple[SourcedExample, ...]
    order: int
    specimen_sha256: str
    schema: str = SCHEMA
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True, slots=True)
class ArmScore:
    arm: str
    correct: int
    evaluated_count: int
    accuracy: float
    gain_vs_baseline: float


@dataclass(frozen=True, slots=True)
class SourceGroupScore:
    """Declared source group; its label is not proof of independence."""

    source_id: str
    arms: tuple[ArmScore, ...]


@dataclass(frozen=True, slots=True)
class FourArmResult:
    specimen_sha256: str
    arms: tuple[ArmScore, ...]
    group_scores: tuple[SourceGroupScore, ...]
    size_gate: str
    heldout_group_count: int
    claim_ceiling: str = CLAIM_CEILING
    schema: str = SCHEMA


def freeze_four_arm_specimen(
    student: Iterable[SourcedExample],
    genuine: Iterable[SourcedExample],
    irrelevant: Iterable[SourcedExample],
    misleading: Iterable[SourcedExample],
    heldout: Iterable[SourcedExample],
    *,
    order: int = 3,
) -> FourArmSpecimen:
    """Bound and digest the five declared partitions before any scoring.

    Reject exact cross-partition overlaps in tokens, claimed source groups,
    and record references. The irrelevant arm must have the same total number
    of tokens as the genuine lesson. This is a structural preflight only;
    source-group aliases and semantic near-duplicates remain undetected.
    """
    if type(order) is not int or not 1 <= order <= 5:
        raise ValueError("order must be an integer from 1 to 5")

    budget = _Budget()
    partitions = (
        _partition(student, "student", budget, required=True),
        _partition(genuine, "genuine", budget, required=True),
        _partition(irrelevant, "irrelevant", budget, required=True),
        _partition(misleading, "misleading", budget, required=True),
        _partition(heldout, "heldout", budget, required=True),
    )
    names = ("student", "genuine", "irrelevant", "misleading", "heldout")
    if len({row.tokens for row in partitions[-1]}) != len(partitions[-1]):
        raise ValueError("heldout contains duplicate exact examples")

    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            a, b = partitions[left], partitions[right]
            if {row.tokens for row in a} & {row.tokens for row in b}:
                raise ValueError(f"exact example overlap between {names[left]} and {names[right]}")
            if {(row.source_id, row.record_id) for row in a} & {
                (row.source_id, row.record_id) for row in b
            }:
                raise ValueError(f"record ID overlap between {names[left]} and {names[right]}")
            if {row.source_id for row in a} & {row.source_id for row in b}:
                raise ValueError(f"source-group overlap between {names[left]} and {names[right]}")

    genuine_tokens = sum(len(row.tokens) for row in partitions[1])
    irrelevant_tokens = sum(len(row.tokens) for row in partitions[2])
    if genuine_tokens != irrelevant_tokens:
        raise ValueError("irrelevant lesson must match genuine lesson total token count")

    document = {
        "schema": SCHEMA,
        "order": order,
        **{name: _serializable(rows) for name, rows in zip(names, partitions)},
    }
    encoded = json.dumps(
        document, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return FourArmSpecimen(*partitions, order, hashlib.sha256(encoded).hexdigest())


def _arm_scores(
    group_records: tuple[SourcedExample, ...],
    predictions: dict[tuple[str, str], dict[str, bool]],
) -> tuple[ArmScore, ...]:
    """Aggregate predictions for one declared group or the whole specimen."""
    count = len(group_records)
    correct = {
        arm: sum(predictions[(row.source_id, row.record_id)][arm] for row in group_records)
        for arm in _ARM_NAMES
    }
    baseline = correct["baseline"] / count
    return tuple(
        ArmScore(
            arm=arm,
            correct=correct[arm],
            evaluated_count=count,
            accuracy=correct[arm] / count,
            gain_vs_baseline=(correct[arm] / count) - baseline,
        )
        for arm in _ARM_NAMES
    )


def evaluate_four_arm(specimen: FourArmSpecimen) -> FourArmResult:
    """Evaluate four independent models against identical declared heldout.

    Re-freezes and checks a public specimen to detect accidental mutation or
    digest mismatch. This does not authenticate the digest or prevent repeated
    evaluation against an exposed answer key.
    """
    if (
        type(specimen) is not FourArmSpecimen
        or specimen.schema != SCHEMA
        or specimen.claim_ceiling != CLAIM_CEILING
    ):
        raise ValueError("invalid four-arm specimen")
    checked = freeze_four_arm_specimen(
        specimen.student,
        specimen.genuine,
        specimen.irrelevant,
        specimen.misleading,
        specimen.heldout,
        order=specimen.order,
    )
    if checked.specimen_sha256 != specimen.specimen_sha256:
        raise ValueError("four-arm specimen digest mismatch")

    models = {name: NGramLearner(order=checked.order) for name in _ARM_NAMES}
    for row in checked.student:
        for learner in models.values():
            learner.observe(row.tokens)
    for name, rows in (
        ("genuine", checked.genuine),
        ("irrelevant", checked.irrelevant),
        ("misleading", checked.misleading),
    ):
        for row in rows:
            models[name].observe(row.tokens)

    predictions: dict[tuple[str, str], dict[str, bool]] = {}
    for row in checked.heldout:
        context, target = row.tokens[:-1], row.tokens[-1]
        predictions[(row.source_id, row.record_id)] = {
            name: learner.predict_next(context) == target
            for name, learner in models.items()
        }

    group_ids = sorted({row.source_id for row in checked.heldout})
    groups = tuple(
        SourceGroupScore(
            source_id=source_id,
            arms=_arm_scores(
                tuple(row for row in checked.heldout if row.source_id == source_id),
                predictions,
            ),
        )
        for source_id in group_ids
    )
    size_gate = (
        "SCALE_ONLY_NOT_QUALIFIED"
        if len(checked.heldout) >= _MIN_DESIGN_HELDOUT
        and len(groups) >= _MIN_DESIGN_GROUPS
        else "BELOW_PREREGISTERED_SCALE"
    )
    return FourArmResult(
        specimen_sha256=checked.specimen_sha256,
        arms=_arm_scores(checked.heldout, predictions),
        group_scores=groups,
        size_gate=size_gate,
        heldout_group_count=len(groups),
    )
