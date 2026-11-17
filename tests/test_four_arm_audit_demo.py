"""The public four-arm demo is a parseable, explicitly nonconfirmatory toy."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "synthetic_four_arm_audit_demo.py"


def load_example(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticAuditDemoTests(unittest.TestCase):
    def test_machine_readable_and_matched_to_original_fixture(self):
        from os import environ

        env = {**environ, "PYTHONPATH": str(ROOT / "src")}
        process = subprocess.run(
            [sys.executable, str(SCRIPT)], cwd=ROOT, env=env,
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(process.stdout)
        self.assertEqual(
            payload,
            json.loads(json.dumps(load_example(SCRIPT).run_audit_demo())),
        )
        self.assertEqual(payload["schema"], "allyn.synthetic-four-arm-audit-demo.v0")
        self.assertEqual(payload["audit"]["schema"], "allyn.four-arm-audit.v0")

        original = load_example(ROOT / "examples" / "synthetic_four_arm_demo.py").run_demo()
        self.assertEqual(
            payload["audit"]["specimen_sha256"], original["result"]["specimen_sha256"]
        )
        self.assertEqual(
            {a["arm"]: a["correct"] for a in payload["audit"]["arms"]},
            {a["arm"]: a["correct"] for a in original["result"]["arms"]},
        )
        self.assertEqual(
            {a["arm"]: a["evaluated_count"] for a in payload["audit"]["arms"]},
            {"baseline": 3, "genuine": 3, "irrelevant": 3, "misleading": 3},
        )
        self.assertEqual(
            {a["arm"]: a["correct"] for a in payload["audit"]["arms"]},
            {"baseline": 1, "genuine": 2, "irrelevant": 1, "misleading": 0},
        )
        self.assertEqual(len(payload["audit"]["per_example"]), 3)
        self.assertEqual(len(payload["audit"]["group_scores"]), 3)
        self.assertTrue(payload["audit"]["misleading_budget_matched"])
        self.assertEqual(payload["audit"]["genuine_lesson_tokens"], 9)
        self.assertEqual(payload["audit"]["misleading_lesson_tokens"], 9)

    def test_toy_ceiling_is_explicit_even_with_positive_gain(self):
        payload = load_example(SCRIPT).run_audit_demo()
        self.assertEqual(payload["real_model_training"], False)
        self.assertEqual(payload["H1_H2_status"], "NOT_CONFIRMATORY_OR_PREREGISTERED")
        self.assertEqual(payload["heldout_access"], "PUBLIC_TO_CALLER_AND_TEACHER_NOT_CUSTODIED")
        self.assertEqual(payload["teacher_facing"], "NOT_A_TEACHER_FACING_EVALUATION_API")
        self.assertIn("NOT_INDEPENDENT_EVALUATION", payload["claim_ceiling"])
        self.assertEqual(payload["audit"]["qualification"], "H1_H2_NOT_QUALIFIED")
        self.assertIn("HELDOUT_EXPOSED_AND_REPEATABLE", payload["audit"]["unmet_gates"])
        self.assertEqual(payload["audit"]["genuine_minus_irrelevant_accuracy"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
