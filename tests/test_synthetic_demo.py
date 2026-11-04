"""Reproducibility smoke tests for the explicitly toy four-arm demonstration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "synthetic_four_arm_demo.py"


class SyntheticFourArmDemoTests(unittest.TestCase):
    @staticmethod
    def invoke() -> dict[str, object]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(f"synthetic demo failed: {completed.stderr}")
        return json.loads(completed.stdout)

    def test_demo_uses_local_synthetic_data_and_discloses_claim_ceiling(self) -> None:
        result = self.invoke()
        self.assertEqual(result["schema"], "allyn.synthetic-four-arm-demo.v0")
        self.assertEqual(result["evidence_class"], "TOY_NOT_CONFIRMATORY_CALLER_DECLARED_SOURCES")
        self.assertEqual(result["source_authentication"], "NONE_CALLER_DECLARED")
        self.assertIn("synthetic", result["dataset"])

    def test_all_four_arms_are_present_and_negative_result_preserved(self) -> None:
        result = self.invoke()["result"]
        scores = {arm["arm"]: arm for arm in result["arms"]}
        self.assertEqual(set(scores), {"baseline", "genuine", "irrelevant", "misleading"})
        self.assertGreater(scores["genuine"]["accuracy"], scores["baseline"]["accuracy"])
        self.assertLess(scores["misleading"]["accuracy"], scores["baseline"]["accuracy"])
        self.assertEqual(scores["irrelevant"]["accuracy"], scores["baseline"]["accuracy"])

    def test_runs_are_byte_for_byte_deterministic(self) -> None:
        self.assertEqual(self.invoke(), self.invoke())

    def test_small_specimen_is_not_misrepresented_as_confirmatory(self) -> None:
        result = self.invoke()["result"]
        self.assertEqual(result["size_gate"], "BELOW_PREREGISTERED_SCALE")
        self.assertEqual(len(result["specimen_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
