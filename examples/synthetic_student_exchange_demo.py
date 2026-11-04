"""Local mock student/teacher exchange over genuine, irrelevant and bad lessons.

Run from repository root:

    PYTHONPATH=src python3 examples/synthetic_student_exchange_demo.py

All material is synthetic and caller-declared. Student changes are local
NGramLearner count-state updates (CONTEXT_UPDATE), NOT model-weight training.
The grader is a function in this process, not independent custody.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json

from allyn.language import NGramLearner
from allyn.student_exchange import (
    DevAssessment,
    HeldoutAssessment,
    LessonProposal,
    StudentState,
    UpdateReceipt,
    run_student_exchange,
)


SCHEMA = "allyn.synthetic-student-exchange-demo.v0"
EVIDENCE_CLASS = "TOY_LOCAL_NGRAM_CONTEXT_UPDATE_NOT_MODEL_WEIGHT_TRAINING"

_INITIAL = (("red", "bird", "sings"), ("blue", "bird", "sings"))
_DEV = (("silver", "bird", "chirps"), ("gray", "bird", "sings"))
_HELDOUT = (("green", "bird", "chirps"), ("purple", "bird", "chirps"),
            ("black", "bird", "sings"))
_LESSONS = {
    "genuine": ("yellow bird chirps", "orange bird chirps", "gold bird chirps"),
    "irrelevant": ("yellow tree sways", "orange tree sways", "gold tree sways"),
    "misleading": ("yellow bird barks", "orange bird barks", "gold bird barks"),
}


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")).hexdigest()


def _score(model: NGramLearner, examples: tuple[tuple[str, ...], ...]) -> int:
    return sum(model.predict_next(example[:-1]) == example[-1] for example in examples)


class _ToyStudent:
    def __init__(self) -> None:
        self.model = NGramLearner(order=3)
        for example in _INITIAL:
            self.model.observe(example)

    def read_state(self) -> StudentState:
        # Snapshot hashes are from locally read count state; they are not
        # authenticated source/weight digests for an external model.
        return StudentState("synthetic-ngram", _digest(self.model.snapshot()))

    def assess_dev(self, state: StudentState) -> DevAssessment:
        assert state == self.read_state()
        return DevAssessment("synthetic:dev", _digest(_DEV), _score(self.model, _DEV), len(_DEV))

    def apply_lesson(self, lesson: LessonProposal) -> UpdateReceipt:
        before = self.read_state().state_sha256
        for text in lesson.content:
            self.model.observe(tuple(text.split()))
        return UpdateReceipt(f"synthetic:operation:{lesson.lesson_id}", lesson.lesson_id,
                             before, self.read_state().state_sha256, "CONTEXT_UPDATE")

    def evaluate_heldout(self, state: StudentState, phase: str) -> HeldoutAssessment:
        assert state == self.read_state()
        return HeldoutAssessment(phase, state.model_id, state.state_sha256,
                                 "synthetic:heldout", _digest(_HELDOUT),
                                 "synthetic:local-grader", _score(self.model, _HELDOUT), len(_HELDOUT))


def run_demo() -> dict[str, object]:
    """Run three independent NGramLearner students with the same hidden labels."""
    outcomes: dict[str, object] = {}
    for arm in ("genuine", "irrelevant", "misleading"):
        student = _ToyStudent()

        def teacher(brief, *, _arm=arm):
            # Only the DEV result is exposed to this callback; the fixed lesson
            # is chosen before either HELDOUT evaluation callback executes.
            assert brief.dev_assessment.source_id == "synthetic:dev"
            return LessonProposal(f"synthetic:lesson:{_arm}", f"synthetic:lesson-source:{_arm}",
                                  _LESSONS[_arm])

        receipt = run_student_exchange(
            run_id=f"synthetic:run:{arm}",
            read_state=student.read_state,
            assess_dev=student.assess_dev,
            teacher=teacher,
            apply_lesson=student.apply_lesson,
            evaluate_heldout=student.evaluate_heldout,
        )
        outcomes[arm] = asdict(receipt)
    return {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "data": "fixed synthetic token strings, caller-declared partition/source IDs",
        "custody": "LOCAL_SAME_PROCESS_NO_INDEPENDENT_CUSTODY",
        "source_authentication": "NONE",
        "outcomes": outcomes,
    }


def main() -> int:
    print(json.dumps(run_demo(), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
