"""Offline demonstration entry point for ALLYN's bounded first research slice."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from typing import Sequence


def run_demo() -> dict[str, object]:
    """Run deterministic toy controls; not an intelligence qualification."""
    from .language import NGramLearner
    from .ledger import PrequentialLedger
    from .teaching import compare_teaching

    student_examples = (("red", "dog", "barks"), ("blue", "dog", "barks"))
    lesson_examples = (("yellow", "bird", "chirps"),)
    heldout_examples = (("green", "bird", "chirps"),)

    transfer = compare_teaching(
        student_examples,
        lesson_examples,
        heldout_examples,
        order=2,
    )

    learner = NGramLearner(order=2)
    learner.observe(("red", "dog", "barks"))
    ledger = PrequentialLedger()
    event = ledger.predict_then_observe(
        learner, ("blue", "dog", "barks"), source_id="allyn-toy-demo-v1"
    )

    return {
        "schema": "allyn.toy-demonstration.v1",
        "evidence_class": "DETERMINISTIC_TOY_TEST_NOT_LANGUAGE_GENERALIZATION",
        "prequential": {
            "events": len(ledger.events),
            "last_prediction_correct": event.correct,
        },
        "transfer": asdict(transfer),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="allyn")
    parser.add_argument("command", choices=("demo",))
    arguments = parser.parse_args(argv)
    if arguments.command == "demo":
        print(json.dumps(run_demo(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
