"""Four-arm *toy* teaching comparison using entirely synthetic token strings.

Run from the repository root with:

    PYTHONPATH=src python examples/synthetic_four_arm_demo.py

This data is invented within the script; all source IDs are declarations. A
positive score cannot establish independent evaluation, general language
understanding, or the preregistered ALLYN teaching hypotheses.
"""

from __future__ import annotations

from dataclasses import asdict
import json

from allyn.evaluation import SourcedExample
from allyn.experiment import evaluate_four_arm, freeze_four_arm_specimen


SCHEMA = "allyn.synthetic-four-arm-demo.v0"
EVIDENCE_CLASS = "TOY_NOT_CONFIRMATORY_CALLER_DECLARED_SOURCES"


def run_demo() -> dict[str, object]:
    """Return a reproducible four-arm comparison without external data."""
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
    result = evaluate_four_arm(specimen)
    return {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "dataset": "local synthetic next-token examples; no real-world claims",
        "source_authentication": "NONE_CALLER_DECLARED",
        "result": asdict(result),
    }


def main() -> int:
    print(json.dumps(run_demo(), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
