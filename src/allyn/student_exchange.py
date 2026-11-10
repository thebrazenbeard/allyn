"""Offline teacher-to-student *exchange* contracts, not a training backend.

An orchestrator owns callback ordering, not the external student's internals.
The teacher receives only a caller-declared DEV assessment; HELDOUT results
remain in this function's receipt and are never passed to the teacher callback.
External callbacks, state digests, source labels and grader identity are not
authenticated. A malicious or coupled callback can still leak heldout data.

No network, model runtime, weight mutation, independent grader custody or
durable replay protection is supplied by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Literal


SCHEMA = "allyn.student-exchange.v0"
CLAIM_CEILING = "CALLER_DECLARED_EXTERNAL_STUDENT_EXCHANGE_ONLY"
UpdateKind = Literal["NO_CHANGE", "CONTEXT_UPDATE", "WEIGHT_UPDATE"]
Phase = Literal["baseline", "post"]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class StudentState:
    """External student snapshot identifiers, not verified weights."""

    model_id: str
    state_sha256: str
    weights_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class DevAssessment:
    """Only the DEV score is revealed to the teacher."""

    source_id: str
    specimen_sha256: str
    correct: int
    total: int


@dataclass(frozen=True, slots=True)
class TeacherBrief:
    run_id: str
    student_state: StudentState
    dev_assessment: DevAssessment


@dataclass(frozen=True, slots=True)
class LessonProposal:
    lesson_id: str
    source_id: str
    content: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UpdateReceipt:
    """Student callback's report of its update, not independent attestation."""

    operation_id: str
    lesson_id: str
    before_sha256: str
    after_sha256: str
    kind: UpdateKind


@dataclass(frozen=True, slots=True)
class HeldoutAssessment:
    phase: Phase
    model_id: str
    state_sha256: str
    source_id: str
    specimen_sha256: str
    evaluator_id: str
    correct: int
    total: int


@dataclass(frozen=True, slots=True)
class ExchangeReceipt:
    run_id: str
    baseline_state: StudentState
    dev_assessment: DevAssessment
    lesson: LessonProposal
    update: UpdateReceipt
    post_state: StudentState
    baseline_heldout: HeldoutAssessment
    post_heldout: HeldoutAssessment
    heldout_accuracy_gain: float
    update_evidence: str
    provenance_status: str = "CALLER_DECLARED_UNVERIFIED"
    claim_ceiling: str = CLAIM_CEILING
    schema: str = SCHEMA


def _id(value: str, label: str) -> None:
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ValueError(f"{label} must be nonblank text at most 256 characters long")


def _digest(value: str, label: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _counts(correct: int, total: int) -> None:
    if type(total) is not int or not 1 <= total <= 10_000:
        raise ValueError("assessment total must be an integer from 1 to 10000")
    if type(correct) is not int or not 0 <= correct <= total:
        raise ValueError("assessment correct must be an integer in [0, total]")


def _state(state: StudentState) -> None:
    if not isinstance(state, StudentState):
        raise TypeError("read_state must return StudentState")
    _id(state.model_id, "model_id")
    _digest(state.state_sha256, "state_sha256")
    if state.weights_sha256 is not None:
        _digest(state.weights_sha256, "weights_sha256")


def _dev(assessment: DevAssessment) -> None:
    if not isinstance(assessment, DevAssessment):
        raise TypeError("assess_dev must return DevAssessment")
    _id(assessment.source_id, "dev source_id")
    _digest(assessment.specimen_sha256, "dev specimen_sha256")
    _counts(assessment.correct, assessment.total)


def _lesson(lesson: LessonProposal) -> None:
    if not isinstance(lesson, LessonProposal):
        raise TypeError("teacher must return LessonProposal")
    _id(lesson.lesson_id, "lesson_id")
    _id(lesson.source_id, "lesson source_id")
    if type(lesson.content) is not tuple or not 1 <= len(lesson.content) <= 64:
        raise ValueError("lesson content must be a tuple with 1 to 64 pieces")
    for piece in lesson.content:
        if type(piece) is not str or not piece.strip() or len(piece) > 4096:
            raise ValueError("lesson pieces must be nonblank text <= 4096 characters")
    if sum(len(piece.encode("utf-8")) for piece in lesson.content) > 65_536:
        raise ValueError("lesson content exceeds 64 KiB")


def _heldout(assessment: HeldoutAssessment, state: StudentState, phase: Phase) -> None:
    if not isinstance(assessment, HeldoutAssessment):
        raise TypeError("evaluate_heldout must return HeldoutAssessment")
    if assessment.phase != phase:
        raise ValueError("heldout phase does not match request")
    if assessment.model_id != state.model_id or assessment.state_sha256 != state.state_sha256:
        raise ValueError("heldout assessment refers to the wrong student state")
    for label, value in (("heldout source_id", assessment.source_id),
                         ("evaluator_id", assessment.evaluator_id)):
        _id(value, label)
    _digest(assessment.specimen_sha256, "heldout specimen_sha256")
    _counts(assessment.correct, assessment.total)


def run_student_exchange(
    *,
    run_id: str,
    read_state: Callable[[], StudentState],
    assess_dev: Callable[[StudentState], DevAssessment],
    teacher: Callable[[TeacherBrief], LessonProposal],
    apply_lesson: Callable[[LessonProposal], UpdateReceipt],
    evaluate_heldout: Callable[[StudentState, Phase], HeldoutAssessment],
) -> ExchangeReceipt:
    """Order a single exchange; retain a matched private baseline/post score.

    External callback authors control all backing data and can lie or share
    memory. This checks *consistency of declarations*, not their authenticity.
    In particular, an update kind of WEIGHT_UPDATE requires supplied changed
    weight digests, but those hashes are not read from a real model here.

    If a callback raises or a receipt is invalid after ``apply_lesson``, its
    outside effects are **not rolled back**. Callers must isolate their student.
    """
    _id(run_id, "run_id")
    for label, callback in (("read_state", read_state), ("assess_dev", assess_dev),
                            ("teacher", teacher), ("apply_lesson", apply_lesson),
                            ("evaluate_heldout", evaluate_heldout)):
        if not callable(callback):
            raise TypeError(f"{label} must be callable")

    baseline = read_state()
    _state(baseline)
    dev = assess_dev(baseline)
    _dev(dev)

    # Freeze the lesson before *any* HELDOUT evaluation callback can execute.
    # This is deliberately the only object given to the teacher. It contains
    # neither heldout answers nor heldout metrics, labels, IDs or digests.
    proposal = teacher(TeacherBrief(run_id, baseline, dev))
    _lesson(proposal)
    if proposal.source_id == dev.source_id:
        raise ValueError("lesson source must differ from DEV source")

    heldout_before = evaluate_heldout(baseline, "baseline")
    _heldout(heldout_before, baseline, "baseline")
    if dev.source_id == heldout_before.source_id or dev.specimen_sha256 == heldout_before.specimen_sha256:
        raise ValueError("DEV and HELDOUT must use distinct declared source and specimen")
    if proposal.source_id in (dev.source_id, heldout_before.source_id):
        raise ValueError("lesson source must differ from DEV and HELDOUT sources")

    update = apply_lesson(proposal)
    if not isinstance(update, UpdateReceipt):
        raise TypeError("apply_lesson must return UpdateReceipt")
    _id(update.operation_id, "operation_id")
    if update.lesson_id != proposal.lesson_id or update.before_sha256 != baseline.state_sha256:
        raise ValueError("update receipt does not match lesson or baseline state")
    _digest(update.after_sha256, "update after_sha256")
    if update.kind not in ("NO_CHANGE", "CONTEXT_UPDATE", "WEIGHT_UPDATE"):
        raise ValueError("unknown update kind")

    after = read_state()
    _state(after)
    if after.model_id != baseline.model_id or after.state_sha256 != update.after_sha256:
        raise ValueError("post state does not match declared model or update")
    if update.kind == "NO_CHANGE":
        if after.state_sha256 != baseline.state_sha256 or after.weights_sha256 != baseline.weights_sha256:
            raise ValueError("NO_CHANGE contradicts state or weight receipts")
    else:
        if after.state_sha256 == baseline.state_sha256:
            raise ValueError("update kind contradicts identical state receipts")
        if update.kind == "WEIGHT_UPDATE":
            if (baseline.weights_sha256 is None or after.weights_sha256 is None
                    or baseline.weights_sha256 == after.weights_sha256):
                raise ValueError("WEIGHT_UPDATE requires distinct declared pre/post weight digests")
        elif baseline.weights_sha256 != after.weights_sha256:
            raise ValueError("CONTEXT_UPDATE contradicts declared weight digest change")

    heldout_after = evaluate_heldout(after, "post")
    _heldout(heldout_after, after, "post")
    if (heldout_before.source_id != heldout_after.source_id
            or heldout_before.specimen_sha256 != heldout_after.specimen_sha256
            or heldout_before.evaluator_id != heldout_after.evaluator_id
            or heldout_before.total != heldout_after.total):
        raise ValueError("heldout assessment changed source, specimen, evaluator or total")

    evidence = {
        "NO_CHANGE": "DECLARED_NO_STATE_CHANGE",
        "CONTEXT_UPDATE": "CALLER_DECLARED_CONTEXT_STATE_CHANGE_ONLY",
        "WEIGHT_UPDATE": "CALLER_DECLARED_WEIGHT_DIGEST_CHANGE_UNVERIFIED",
    }[update.kind]
    return ExchangeReceipt(
        run_id=run_id, baseline_state=baseline, dev_assessment=dev,
        lesson=proposal, update=update, post_state=after,
        baseline_heldout=heldout_before, post_heldout=heldout_after,
        heldout_accuracy_gain=(heldout_after.correct / heldout_after.total
                               - heldout_before.correct / heldout_before.total),
        update_evidence=evidence,
    )
