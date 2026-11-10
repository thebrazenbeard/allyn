"""Deterministic, reviewer-facing *toy* four-arm paired audit.

Run from the repository root:

    PYTHONPATH=src python examples/synthetic_four_arm_audit_demo.py

The held-out answers are public in this demonstration and explicitly NOT
teacher-hidden or independently custodied. It is not an H1/H2 result or a
measurement of a real model's semantic learning or neural-weight updates.
"""

from __future__ import annotations

from dataclasses import asdict
import json

from allyn.evaluation import SourcedExample
from allyn.experiment import CLAIM_CEILING, freeze_four_arm_specimen
from allyn.experiment_report import audit_four_arm


SCHEMA = "allyn.synthetic-four-arm-audit-demo.v0"


def run_audit_demo() -> dict[str, object]:
    """Return a JSON-compatible report of the same declared synthetic case.

    The fixture intentionally mirrors synthetic_four_arm_demo.py. Both
    scripts should retain matching specimen digests and aggregate outcomes.
    No external observations or student model are involved.
    """
    student = (
        SourcedExample("synthetic:student", "student-1", ("red", "bird", "sings")),
        SourcedExample("synthetic:student", "student-2", ("blue", "bird", "sings")),
    )
    genuine = tuple(
        SourcedExample("synthetic:genuine", f"genuine-{index}", (color, "bird", "chirps"))
        for index, color in enumerate(("yellow", "orange", "gold"), 1)
    )
    irrelevant = tuple(
        SourcedExample("synthetic:irrelevant", f"irrelevant-{index}", (color, "tree", "sways"))
        for index, color in enumerate(("yellow", "orange", "gold"), 1)
    )
    misleading = tuple(
        SourcedExample("synthetic:misleading", f"misleading-{index}", (color, "bird", "barks"))
        for index, color in enumerate(("yellow", "orange", "gold"), 1)
    )
    heldout = (
        SourcedExample("synthetic:test-group-a", "test-1", ("green", "bird", "chirps")),
        SourcedExample("synthetic:test-group-b", "test-2", ("purple", "bird", "chirps")),
        SourcedExample("synthetic:test-group-c", "test-3", ("black", "bird", "sings")),
    )
    specimen = freeze_four_arm_specimen(
        student=student,
        genuine=genuine,
        irrelevant=irrelevant,
        misleading=misleading,
        heldout=heldout,
        order=3,
    )
    report = audit_four_arm(specimen)
    return {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "dataset": "public, caller-declared synthetic next-token examples",
        "heldout_access": "PUBLIC_TO_CALLER_AND_TEACHER_NOT_CUSTODIED",
        "teacher_facing": "NOT_A_TEACHER_FACING_EVALUATION_API",
        "source_and_grader_authentication": "NONE",
        "H1_H2_status": "NOT_CONFIRMATORY_OR_PREREGISTERED",
        "real_model_training": False,
        "audit": asdict(report),
    }


def main() -> int:
    print(json.dumps(run_audit_demo(), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
