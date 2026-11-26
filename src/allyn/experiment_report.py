"""Transparent, deterministic audit of caller-declared four-arm toy results.

The *same caller* supplies every source, lesson, and exposed held-out answer.
This report is a diagnostic reconstruction, not a preregistered experiment,
independent grading, trusted custody, model-weight update, or teaching proof.
"""

from __future__ import annotations

from dataclasses import dataclass

from .experiment import FourArmSpecimen, SourceGroupScore, evaluate_four_arm
from .language import NGramLearner


SCHEMA = "allyn.four-arm-audit.v0"
EVIDENCE_CLASS = "REPRODUCIBLE_CALLER_DECLARED_TOY_DIAGNOSTIC_ONLY"
QUALIFICATION = "H1_H2_NOT_QUALIFIED"
ARM_NAMES = ("baseline", "genuine", "irrelevant", "misleading")


@dataclass(frozen=True, slots=True)
class PairedExample:
    """One held-out answer's four boolean outcomes and signed changes."""

    source_id: str
    record_id: str
    baseline_correct: bool
    genuine_correct: bool
    irrelevant_correct: bool
    misleading_correct: bool
    genuine_delta: int
    irrelevant_delta: int
    misleading_delta: int


@dataclass(frozen=True, slots=True)
class ArmAudit:
    arm: str
    correct: int
    evaluated_count: int
    accuracy: float
    paired_improvements: int
    paired_regressions: int
    paired_unchanged: int
    net_correct_vs_baseline: int
    accuracy_delta_vs_baseline: float


@dataclass(frozen=True, slots=True)
class FourArmAudit:
    schema: str
    specimen_sha256: str
    evidence_class: str
    qualification: str
    size_gate: str
    heldout_group_count: int
    arms: tuple[ArmAudit, ...]
    group_scores: tuple[SourceGroupScore, ...]
    per_example: tuple[PairedExample, ...]
    genuine_lesson_tokens: int
    irrelevant_lesson_tokens: int
    misleading_lesson_tokens: int
    misleading_budget_matched: bool
    control_budget_note: str
    genuine_minus_irrelevant_accuracy: float
    genuine_minus_misleading_accuracy: float
    placebo_delta_vs_baseline: float
    misleading_delta_vs_baseline: float
    unmet_gates: tuple[str, ...]


def audit_four_arm(specimen: FourArmSpecimen) -> FourArmAudit:
    """Reconstruct paired predictions and cross-check each published aggregate.

    The public specimen can be rerun or cherry-picked. All gate labels are
    informational; none can be elevated by increasing toy sample size alone.
    """
    result = evaluate_four_arm(specimen)  # Revalidates structure and digest.
    models = {name: NGramLearner(order=specimen.order) for name in ARM_NAMES}
    for row in specimen.student:
        for model in models.values():
            model.observe(row.tokens)
    for name, rows in (
        ("genuine", specimen.genuine),
        ("irrelevant", specimen.irrelevant),
        ("misleading", specimen.misleading),
    ):
        for row in rows:
            models[name].observe(row.tokens)

    paired: list[PairedExample] = []
    for row in specimen.heldout:
        outcomes = {
            name: models[name].predict_next(row.tokens[:-1]) == row.tokens[-1]
            for name in ARM_NAMES
        }
        baseline = int(outcomes["baseline"])
        paired.append(
            PairedExample(
                source_id=row.source_id,
                record_id=row.record_id,
                baseline_correct=outcomes["baseline"],
                genuine_correct=outcomes["genuine"],
                irrelevant_correct=outcomes["irrelevant"],
                misleading_correct=outcomes["misleading"],
                genuine_delta=int(outcomes["genuine"]) - baseline,
                irrelevant_delta=int(outcomes["irrelevant"]) - baseline,
                misleading_delta=int(outcomes["misleading"]) - baseline,
            )
        )

    audits: list[ArmAudit] = []
    for score in result.arms:
        arm = score.arm
        changes = (
            (0 for _ in paired) if arm == "baseline" else
            (getattr(row, f"{arm}_delta") for row in paired)
        )
        deltas = tuple(changes)
        correct = sum(getattr(row, f"{arm}_correct") for row in paired)
        if correct != score.correct or len(paired) != score.evaluated_count:
            raise ValueError("per-example reconstruction differs from aggregate")
        improvements = sum(delta == 1 for delta in deltas)
        regressions = sum(delta == -1 for delta in deltas)
        unchanged = sum(delta == 0 for delta in deltas)
        audits.append(
            ArmAudit(
                arm=arm,
                correct=correct,
                evaluated_count=len(paired),
                accuracy=score.accuracy,
                paired_improvements=improvements,
                paired_regressions=regressions,
                paired_unchanged=unchanged,
                net_correct_vs_baseline=improvements - regressions,
                accuracy_delta_vs_baseline=score.gain_vs_baseline,
            )
        )

    by_name = {arm.arm: arm for arm in audits}
    gates = [
        "NOT_PREREGISTERED_OR_INDEPENDENTLY_EVALUATED",
        "HELDOUT_EXPOSED_AND_REPEATABLE",
        "UNAUTHENTICATED_SOURCE_LICENSE_AND_GRADER",
        "NO_EXTERNAL_STUDENT_OR_NEURAL_WEIGHT_UPDATE",
        "NO_INDEPENDENT_SEMANTIC_TRANSFER_MEASURE",
    ]
    if result.size_gate == "BELOW_PREREGISTERED_SCALE":
        gates.append("BELOW_100_HELDOUT_OR_3_DECLARED_GROUPS")
    token_counts = {
        arm: sum(len(row.tokens) for row in getattr(specimen, arm))
        for arm in ("genuine", "irrelevant", "misleading")
    }
    misleading_matched = token_counts["misleading"] == token_counts["genuine"]
    if not misleading_matched:
        gates.append("MISLEADING_CONTROL_TOKEN_BUDGET_UNMATCHED")

    return FourArmAudit(
        schema=SCHEMA,
        specimen_sha256=result.specimen_sha256,
        evidence_class=EVIDENCE_CLASS,
        qualification=QUALIFICATION,
        size_gate=result.size_gate,
        heldout_group_count=result.heldout_group_count,
        arms=tuple(audits),
        group_scores=result.group_scores,
        per_example=tuple(paired),
        genuine_lesson_tokens=token_counts["genuine"],
        irrelevant_lesson_tokens=token_counts["irrelevant"],
        misleading_lesson_tokens=token_counts["misleading"],
        misleading_budget_matched=misleading_matched,
        control_budget_note=(
            "Equal token count alone does not match topic, information, difficulty, or influence"
        ),
        genuine_minus_irrelevant_accuracy=(
            by_name["genuine"].accuracy - by_name["irrelevant"].accuracy
        ),
        genuine_minus_misleading_accuracy=(
            by_name["genuine"].accuracy - by_name["misleading"].accuracy
        ),
        placebo_delta_vs_baseline=by_name["irrelevant"].accuracy_delta_vs_baseline,
        misleading_delta_vs_baseline=by_name["misleading"].accuracy_delta_vs_baseline,
        unmet_gates=tuple(gates),
    )
