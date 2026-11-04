"""One-shot aggregate four-arm custody; toy patterns only."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from allyn.custodian import AlreadyEvaluatedError, evaluate_once
from allyn.evaluation import SourcedExample, freeze_specimen
from allyn.experiment import freeze_four_arm_specimen
from allyn.four_arm_custodian import (
    CLAIM_CEILING, REQUEST_SCHEMA, evaluate_four_arm_once,
)


SECRET = "private-answer-chirps-F52b6"


def _parts():
    student = [
        SourcedExample("source-student", "s1", ("red", "bird", "sings")),
        SourcedExample("source-student", "s2", ("blue", "bird", "sings")),
    ]
    genuine = [
        SourcedExample("source-genuine", f"g{i}", (color, "bird", SECRET))
        for i, color in enumerate(("yellow", "orange", "gold"), 1)
    ]
    irrelevant = [
        SourcedExample("source-irrelevant", f"i{i}", (color, "tree", "sways"))
        for i, color in enumerate(("yellow", "orange", "gold"), 1)
    ]
    misleading = [
        SourcedExample("source-misleading", f"m{i}", (color, "bird", "barks"))
        for i, color in enumerate(("yellow", "orange", "gold"), 1)
    ]
    heldout = [
        SourcedExample("private-heldout-alpha", "h1", ("green", "bird", SECRET)),
        SourcedExample("private-heldout-beta", "h2", ("purple", "bird", SECRET)),
        SourcedExample("private-heldout-gamma", "h3", ("black", "bird", "sings")),
    ]
    return student, genuine, irrelevant, misleading, heldout


def _frozen():
    return freeze_four_arm_specimen(*_parts())


def _wire():
    return {
        "schema": REQUEST_SCHEMA,
        **{
            label: [
                {"source_id": row.source_id, "record_id": row.record_id, "tokens": list(row.tokens)}
                for row in records
            ]
            for label, records in zip(
                ("student", "genuine", "irrelevant", "misleading", "heldout"), _parts()
            )
        },
        "order": 3,
    }


def _child(state: Path, request_id: str, payload: object):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "allyn.four_arm_custodian", "--state-dir", str(state), "--request-id", request_id],
        input=json.dumps(payload).encode("utf-8"), capture_output=True,
        timeout=15, env=environment, check=False,
    )


class FourArmCustodyTests(unittest.TestCase):
    def test_disposable_process_returns_four_arm_aggregates_without_answers(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            run = _child(state, "four-arm-one", _wire())
            self.assertEqual(run.returncode, 0, run.stderr.decode("utf-8"))
            report = json.loads(run.stdout)
            self.assertEqual(report["status"], "evaluated")
            self.assertNotIn("specimen_sha256", report)
            self.assertNotIn(_frozen().specimen_sha256, run.stdout.decode("utf-8"))
            self.assertEqual(report["claim_ceiling"], CLAIM_CEILING)
            self.assertEqual(report["evaluated_count"], 3)
            self.assertEqual(report["heldout_group_count"], 3)
            self.assertEqual(report["size_gate"], "BELOW_PREREGISTERED_SCALE")
            self.assertEqual(
                {item["arm"]: item["correct"] for item in report["arms"]},
                {"baseline": 1, "genuine": 2, "irrelevant": 1, "misleading": 0},
            )
            self.assertEqual(len(report["groups"]), 3)
            self.assertEqual([group["group_index"] for group in report["groups"]], [1, 2, 3])
            for group in report["groups"]:
                self.assertEqual(len(group["arms"]), 4)
                self.assertTrue(all(arm["evaluated_count"] == 1 for arm in group["arms"]))
            public = run.stdout + (state / "request-four-arm-one.json").read_bytes()
            for hidden in (SECRET, "private-heldout-alpha", "private-heldout-beta", "h1", "h2"):
                self.assertNotIn(hidden.encode("utf-8"), public)
            self.assertEqual(json.loads((state / "request-four-arm-one.json").read_text()), report)
            self.assertEqual(len(list(state.glob("heldout-*.claim"))), 1)

    def test_replay_changing_only_genuine_lesson_still_blocks_same_heldout(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            first = evaluate_four_arm_once(_frozen(), state_dir=state, request_id="first")
            student, _, irrelevant, misleading, heldout = _parts()
            changed = [
                SourcedExample("changed-genuine", f"new{i}", (color, "bird", "tweets"))
                for i, color in enumerate(("yellow", "orange", "gold"), 1)
            ]
            second = freeze_four_arm_specimen(student, changed, irrelevant, misleading, heldout)
            self.assertNotEqual(_frozen().specimen_sha256, second.specimen_sha256)
            self.assertNotIn("specimen_sha256", asdict(first))
            with self.assertRaises(AlreadyEvaluatedError):
                evaluate_four_arm_once(second, state_dir=state, request_id="new-lesson")

    def test_shared_registry_blocks_cross_interface_requery(self):
        student, genuine, _, _, heldout = _parts()
        single_arm = freeze_specimen(student, genuine, heldout)
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            evaluate_once(single_arm, state_dir=state, request_id="one-arm")
            with self.assertRaises(AlreadyEvaluatedError):
                evaluate_four_arm_once(_frozen(), state_dir=state, request_id="four-arm")

    def test_same_request_or_digest_replay_errors_have_no_answer_text(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            self.assertEqual(_child(state, "once", _wire()).returncode, 0)
            for name in ("once", "new-request"):
                duplicate = _child(state, name, _wire())
                self.assertEqual(duplicate.returncode, 2)
                self.assertIn(b"already consumed", duplicate.stderr)
                self.assertEqual(duplicate.stdout, b"")
                self.assertNotIn(SECRET.encode(), duplicate.stderr)

    def test_wire_rejects_invalid_extra_fields_without_leak(self):
        with tempfile.TemporaryDirectory() as root:
            payload = _wire()
            payload["hidden"] = SECRET
            run = _child(Path(root), "bad", payload)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(run.stdout, b"")
            self.assertNotIn(SECRET.encode(), run.stderr)

    def test_internal_failure_leaves_generic_receipt_and_consumed_heldout(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            with patch("allyn.four_arm_custodian.evaluate_four_arm", side_effect=RuntimeError(SECRET)):
                with self.assertRaises(RuntimeError):
                    evaluate_four_arm_once(_frozen(), state_dir=state, request_id="broken")
            stored = (state / "request-broken.json").read_text()
            self.assertEqual(json.loads(stored)["status"], "failed")
            self.assertNotIn(SECRET, stored)
            with self.assertRaises(AlreadyEvaluatedError):
                evaluate_four_arm_once(_frozen(), state_dir=state, request_id="retry")


if __name__ == "__main__":
    unittest.main()
