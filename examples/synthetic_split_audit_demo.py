"""Demonstrate lexical TRAIN/DEV/HELDOUT leakage with invented data.

From the repository root:

    PYTHONPATH=src python examples/synthetic_split_audit_demo.py
    PYTHONPATH=src python examples/synthetic_split_audit_demo.py --clean

The default fixture is intentionally contaminated and exits 2 to demonstrate
a strict audit gate. ``--clean`` is a toy negative control and exits 0. Neither
outcome establishes rights, true semantic decontamination or independent
evaluation custody.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from allyn.evaluation import SourcedExample
from allyn.split_controls import audit_partition_overlap


SCHEMA = "allyn.synthetic-split-audit-demo.v0"
EVIDENCE_CLASS = "SYNTHETIC_LEXICAL_LEAKAGE_TEST_NOT_INDEPENDENT_CUSTODY"


def _row(partition: str, key: str, text: str) -> SourcedExample:
    return SourcedExample(f"synthetic:{partition}", key, tuple(text.split()))


def _fixtures(clean: bool) -> tuple[tuple[SourcedExample, ...], ...]:
    if clean:
        return (
            (_row("train", "train-1", "violet circuits carry gentle impulses"),),
            (_row("dev", "dev-1", "amber gardens receive seasonal rainfall"),),
            (_row("heldout", "test-1", "silver lanterns illuminate quiet corridors"),),
        )

    return (
        (
            _row("train", "train-1", "Hello, strange WORLD!"),
            _row("train", "train-2", "the curious student studies how small changes to a question can influence the next answer"),
        ),
        (_row("dev", "dev-1", "the careful observer learns to compare multiple plausible explanations before choosing one"),),
        (
            _row("heldout", "test-1", "hello strange world"),
            _row("heldout", "test-2", "the curious student studies how small changes to a question can influence the final answer"),
            _row("heldout", "test-3", "the careful observer learns to compare multiple plausible explanations before choosing another"),
        ),
    )


def run_demo(*, clean: bool = False) -> dict[str, object]:
    """Return stable JSON-safe result, with record references but not text."""
    audit = audit_partition_overlap(*_fixtures(clean))
    contaminated = audit.has_indicated_overlap
    return {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "demonstration": True,
        "data_origin": "INVENTED_IN_THIS_SCRIPT",
        "mode": "SYNTHETIC_CLEAN_CONTROL" if clean else "INTENTIONALLY_CONTAMINATED",
        "verdict": "FAIL" if contaminated else "PASS",
        "exit_code": 2 if contaminated else 0,
        "claim_ceiling": audit.claim_ceiling,
        "audit": {
            "train_count": audit.train_count,
            "dev_count": audit.dev_count,
            "heldout_count": audit.heldout_count,
            "comparisons": audit.comparisons,
            "collision_count": len(audit.collisions),
            "collisions": [asdict(row) for row in audit.collisions],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic lexical split-contamination gate")
    parser.add_argument("--clean", action="store_true", help="evaluate separate synthetic clean-control fixture")
    args = parser.parse_args()
    report = run_demo(clean=args.clean)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
