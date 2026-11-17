"""Offline, selection-then-evaluation teaching cycle with an explicit control.

This is a toy next-token *pattern* experiment. Caller-declared source IDs are
not authenticated; neither selection probes nor final heldout labels are under
independent custody. A gain can reflect overlap, curation, or leakage outside
this function, not semantic learning or a teacher's epistemic authority.

The final heldout specimens are first ingested *after* candidate selection.
Their contents cannot influence the choice inside this function. A no-lesson
control is kept, a negative outcome is retained, and no learner is persisted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .inquiry import AssessmentStatus, InquiryLedger
from .teaching import (
    TransferResult,
    _IngressBudget,
    _examples,
    compare_teaching,
)


SCHEMA = "allyn.teaching-cycle.v0"
CLAIM_CEILING = "CALLER_DECLARED_SOURCES_TOY_NEXT_TOKEN_PATTERN_TRANSFER_ONLY"
_MAX_CANDIDATES = 16
_MAX_ID_LENGTH = 256


def _source_id(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > _MAX_ID_LENGTH:
        raise ValueError(f"{label} must be nonblank text of at most {_MAX_ID_LENGTH} characters")
    return value


@dataclass(frozen=True, slots=True)
class LessonCandidate:
    """A source-declared teaching proposal, not an authenticated assertion."""

    candidate_id: str
    source_id: str
    examples: Iterable[Sequence[str]]


@dataclass(frozen=True, slots=True)
class ProbeReceipt:
    candidate_id: str
    source_id: str
    transfer: TransferResult


@dataclass(frozen=True, slots=True)
class TeachingCycleReceipt:
    """Immutable scored receipt, not a certificate of semantic improvement."""

    selected_candidate_id: str | None
    selected_lesson_source_id: str | None
    selection_status: Literal["SELECTED_ON_PROBES", "ABSTAINED_NO_PROBE_GAIN"]
    no_lesson_probe: TransferResult
    candidate_probes: tuple[ProbeReceipt, ...]
    final_transfer: TransferResult
    student_source_id: str
    selection_probe_source_id: str
    heldout_source_id: str
    inquiry_proposition_id: str | None
    inquiry_report_status: AssessmentStatus | None
    provenance_status: str = "CALLER_DECLARED_UNVERIFIED"
    claim_ceiling: str = CLAIM_CEILING
    schema: str = SCHEMA


def run_teaching_cycle(
    *,
    student_examples: Iterable[Sequence[str]],
    student_source_id: str,
    candidates: Sequence[LessonCandidate],
    selection_probes: Iterable[Sequence[str]],
    selection_probe_source_id: str,
    heldout_examples: Iterable[Sequence[str]],
    heldout_source_id: str,
    order: int = 3,
    inquiry: InquiryLedger | None = None,
    inquiry_proposition_id: str | None = None,
) -> TeachingCycleReceipt:
    """Select with training-only probes, then report a separate heldout score.

    Probe scores choose a candidate only if its gain exceeds the no-lesson
    control (zero). The final heldout examples are *not read* until that choice
    is fixed. Exact utterance overlaps with any student, proposed lesson, or
    selection-probe specimen are rejected, but this necessary separation does
    not detect semantic duplicates, source collusion or independently verify
    heldout custody.

    The inquiry assessment is attached as a reported status, never used as an
    authorization or independent truth oracle. This function never mutates it.
    """
    # Fail before consuming any caller-provided iterators.
    if type(order) is not int or not 1 <= order <= 5:
        raise ValueError("order must be an integer from 1 to 5")
    student_source_id = _source_id(student_source_id, "student_source_id")
    selection_probe_source_id = _source_id(selection_probe_source_id, "selection_probe_source_id")
    heldout_source_id = _source_id(heldout_source_id, "heldout_source_id")
    if len({student_source_id, selection_probe_source_id, heldout_source_id}) != 3:
        raise ValueError("student, probe, and heldout source references must differ")
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a bounded sequence of LessonCandidate objects")
    if not 1 <= len(candidates) <= _MAX_CANDIDATES:
        raise ValueError(f"candidates must contain 1 to {_MAX_CANDIDATES} proposals")
    if (inquiry is None) != (inquiry_proposition_id is None):
        raise ValueError("inquiry and inquiry_proposition_id must be supplied together")
    if inquiry is not None and not isinstance(inquiry, InquiryLedger):
        raise TypeError("inquiry must be an InquiryLedger")
    if inquiry_proposition_id is not None:
        _source_id(inquiry_proposition_id, "inquiry_proposition_id")

    # Private, shared admission budget ensures bounded *aggregate* input even
    # across multiple candidate lessons, before any scoring and hashing.
    budget = _IngressBudget()
    student = _examples(student_examples, label="student_examples", allow_empty=False, budget=budget)
    probes = _examples(selection_probes, label="selection_probes", allow_empty=False, budget=budget)
    if len(set(probes)) != len(probes):
        raise ValueError("selection probes must contain unique exact utterances")

    admitted: list[tuple[str, str, tuple[tuple[str, ...], ...]]] = []
    candidate_ids: set[str] = set()
    lesson_sources: set[str] = set()
    for index, candidate in enumerate(candidates):
        if index >= _MAX_CANDIDATES:
            raise ValueError(f"candidates exceed {_MAX_CANDIDATES} proposals")
        if not isinstance(candidate, LessonCandidate):
            raise TypeError("each candidate must be a LessonCandidate")
        candidate_id = _source_id(candidate.candidate_id, "candidate_id")
        source_id = _source_id(candidate.source_id, "candidate.source_id")
        if candidate_id in candidate_ids:
            raise ValueError("candidate IDs must be unique")
        if source_id in lesson_sources or source_id in (
            student_source_id, selection_probe_source_id, heldout_source_id
        ):
            raise ValueError("lesson source references must be distinct from each other and all controls")
        candidate_ids.add(candidate_id)
        lesson_sources.add(source_id)
        lesson = _examples(candidate.examples, label=f"candidate[{candidate_id}]", allow_empty=False, budget=budget)
        admitted.append((candidate_id, source_id, lesson))

    # Block the easiest self-test leakage: an evaluation probe replayed
    # verbatim as a student observation or a teacher's candidate example.
    probe_set = set(probes)
    if probe_set.intersection(student) or any(
        probe_set.intersection(lesson) for _, _, lesson in admitted
    ):
        raise ValueError("selection probe overlaps a student or candidate lesson utterance")

    # No candidate selection receives the final heldout dataset.
    no_lesson_probe = compare_teaching(student, (), probes, order=order)
    candidate_probe_results = tuple(
        ProbeReceipt(candidate_id, source_id, compare_teaching(student, lesson, probes, order=order))
        for candidate_id, source_id, lesson in admitted
    )
    ranked = sorted(candidate_probe_results, key=lambda receipt: (
        -receipt.transfer.gain, -receipt.transfer.taught_accuracy, receipt.candidate_id
    ))
    winner = ranked[0] if ranked[0].transfer.gain > no_lesson_probe.gain else None

    # Heldout admission happens after the winner/abstention is frozen.
    heldout = _examples(heldout_examples, label="heldout_examples", allow_empty=False, budget=budget)
    if len(set(heldout)) != len(heldout):
        raise ValueError("heldout specimens must contain unique exact utterances")
    non_heldout = {example for example in student + probes}
    for _, _, lesson in admitted:
        non_heldout.update(lesson)
    if any(example in non_heldout for example in heldout):
        raise ValueError("heldout specimen overlaps a student, probe, or candidate lesson utterance")

    selected_lesson = ()
    if winner is not None:
        selected_lesson = next(lesson for ident, _, lesson in admitted if ident == winner.candidate_id)
    final_transfer = compare_teaching(student, selected_lesson, heldout, order=order)
    inquiry_status = inquiry.assess(inquiry_proposition_id).status if inquiry is not None else None

    return TeachingCycleReceipt(
        selected_candidate_id=winner.candidate_id if winner is not None else None,
        selected_lesson_source_id=winner.source_id if winner is not None else None,
        selection_status="SELECTED_ON_PROBES" if winner is not None else "ABSTAINED_NO_PROBE_GAIN",
        no_lesson_probe=no_lesson_probe,
        candidate_probes=candidate_probe_results,
        final_transfer=final_transfer,
        student_source_id=student_source_id,
        selection_probe_source_id=selection_probe_source_id,
        heldout_source_id=heldout_source_id,
        inquiry_proposition_id=inquiry_proposition_id,
        inquiry_report_status=inquiry_status,
    )
