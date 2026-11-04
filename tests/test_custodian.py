"""Local one-shot evaluator, not independent custody or an external review."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from allyn.custodian import (
    AlreadyEvaluatedError, CLAIM_CEILING, CustodyReceipt,
    REQUEST_SCHEMA, evaluate_once,
)
from allyn.evaluation import SourcedExample, freeze_specimen


HIDDEN = "HIDDEN_EVALUATION_ANSWER_8bc74f"


def _specimen():
    return freeze_specimen(
        [SourcedExample("train-source", "train-1", ("bridge", "old"))],
        [SourcedExample("lesson-source", "lesson-1", ("bridge", HIDDEN))],
        [SourcedExample("heldout-source", "held-1", ("bridge", "fresh", HIDDEN))],
    )


def _wire():
    return {
        "schema": REQUEST_SCHEMA,
        "student": [{"source_id": "train-source", "record_id": "train-1", "tokens": ["bridge", "old"]}],
        "lessons": [{"source_id": "lesson-source", "record_id": "lesson-1", "tokens": ["bridge", HIDDEN]}],
        "heldout": [{"source_id": "heldout-source", "record_id": "held-1", "tokens": ["bridge", "fresh", HIDDEN]}],
        "order": 3,
        "allow_shared_source_groups": False,
    }


def _child(state: Path, request_id: str, payload: object):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "allyn.custodian", "--state-dir", str(state), "--request-id", request_id],
        input=json.dumps(payload).encode("utf-8"), capture_output=True,
        timeout=15, env=environment, check=False,
    )


class CustodyTests(unittest.TestCase):
    def test_one_shot_child_process_receipt_excludes_private_examples(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / "custody"
            result = _child(state, "repro-1", _wire())
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["request_id"], "repro-1")
            self.assertEqual(receipt["schema"], "allyn.custody-receipt.v0")
            self.assertEqual(receipt["evaluated_count"], 1)
            self.assertEqual(receipt["claim_ceiling"], CLAIM_CEILING)
            self.assertNotIn("specimen_sha256", receipt)
            self.assertNotIn(_specimen().specimen_sha256, result.stdout.decode("utf-8"))
            serialized = result.stdout + (state / "request-repro-1.json").read_bytes()
            self.assertNotIn(HIDDEN.encode(), serialized)
            self.assertNotIn(b"heldout-source", serialized)
            self.assertNotIn(b"held-1", serialized)
            self.assertEqual(json.loads((state / "request-repro-1.json").read_text()), receipt)
            self.assertEqual(len(list(state.glob("specimen-*.claim"))), 1)
            self.assertEqual(len(list(state.glob("heldout-*.claim"))), 1)

    def test_same_request_and_same_digest_new_request_both_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            first = _child(state, "first", _wire())
            self.assertEqual(first.returncode, 0, first.stderr.decode("utf-8"))
            for identifier in ("first", "second"):
                repeated = _child(state, identifier, _wire())
                self.assertEqual(repeated.returncode, 2)
                self.assertEqual(repeated.stdout, b"")
                self.assertIn(b"already consumed", repeated.stderr)
                self.assertNotIn(HIDDEN.encode(), repeated.stderr)

    def test_different_state_dir_cannot_prevent_cherry_picking(self):
        # Demonstrates the limit rather than claiming any global seal.
        with tempfile.TemporaryDirectory() as root:
            a = _child(Path(root) / "a", "first", _wire())
            b = _child(Path(root) / "b", "second", _wire())
            self.assertEqual(a.returncode, 0)
            self.assertEqual(b.returncode, 0)

    def test_direct_api_returns_only_metrics_and_blocks_new_request_same_digest(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            receipt = evaluate_once(_specimen(), state_dir=state, request_id="direct")
            self.assertIsInstance(receipt, CustodyReceipt)
            self.assertEqual(receipt.evaluated_count, 1)
            self.assertNotIn(HIDDEN, repr(receipt))
            self.assertNotIn(_specimen().specimen_sha256, repr(receipt))
            with self.assertRaises(AlreadyEvaluatedError):
                evaluate_once(_specimen(), state_dir=state, request_id="new")

    def test_replay_with_changed_lessons_and_relabelled_heldout_is_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            evaluate_once(_specimen(), state_dir=state, request_id="first")
            changed_lesson = freeze_specimen(
                [SourcedExample("train-source", "train-1", ("bridge", "old"))],
                [SourcedExample("changed-lesson-source", "lesson-2", ("bridge", "another"))],
                [SourcedExample("heldout-source", "held-1", ("bridge", "fresh", HIDDEN))],
            )
            self.assertNotEqual(changed_lesson.specimen_sha256, _specimen().specimen_sha256)
            with self.assertRaises(AlreadyEvaluatedError):
                evaluate_once(changed_lesson, state_dir=state, request_id="changed-lesson")
            changed_source_label = freeze_specimen(
                [SourcedExample("train-source", "train-1", ("bridge", "old"))],
                [SourcedExample("changed-lesson-source", "lesson-2", ("bridge", "another"))],
                [SourcedExample("renamed-heldout-source", "new-record", ("bridge", "fresh", HIDDEN))],
            )
            with self.assertRaises(AlreadyEvaluatedError):
                evaluate_once(changed_source_label, state_dir=state, request_id="relabelled-heldout")

    def test_bad_wire_and_bad_id_fail_without_leaking_contents(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            invalid = _wire()
            invalid["intrusive"] = HIDDEN
            for request_id, payload in (("malformed", invalid), ("../bad", _wire())):
                result = _child(state, request_id, payload)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertNotIn(HIDDEN.encode(), result.stderr)
            self.assertEqual(list(state.glob("*.claim")), [])

    def test_failure_during_evaluation_consumes_digest_and_writes_generic_failed_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            with patch("allyn.custodian.evaluate_frozen", side_effect=RuntimeError(HIDDEN)):
                with self.assertRaises(RuntimeError):
                    evaluate_once(_specimen(), state_dir=state, request_id="failed1")
            public = (state / "request-failed1.json").read_text(encoding="utf-8")
            self.assertIn('"status":"failed"', public)
            self.assertNotIn(HIDDEN, public)
            with self.assertRaises(AlreadyEvaluatedError):
                evaluate_once(_specimen(), state_dir=state, request_id="failed2")

    def test_symlinked_state_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / "state"
            state.mkdir()
            link = Path(root) / "link"
            try:
                link.symlink_to(state, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("OS cannot create test symlink")
            with self.assertRaises(ValueError):
                evaluate_once(_specimen(), state_dir=link, request_id="one")


if __name__ == "__main__":
    unittest.main()
