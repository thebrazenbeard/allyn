"""Executable smoke tests for the synthetic split-audit gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "synthetic_split_audit_demo.py"


def _invoke(*args: str) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(DEMO), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return completed, json.loads(completed.stdout)


class SyntheticSplitAuditDemoTests(unittest.TestCase):
    def test_intentionally_contaminated_fixture_has_fail_exit_status(self) -> None:
        process, report = _invoke()
        self.assertEqual(process.returncode, 2, process.stderr)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["exit_code"], 2)
        self.assertTrue(report["demonstration"])
        self.assertEqual(report["mode"], "INTENTIONALLY_CONTAMINATED")
        self.assertEqual(report["data_origin"], "INVENTED_IN_THIS_SCRIPT")
        self.assertEqual(report["audit"]["collision_count"], 3)
        reasons = {row["reason"] for row in report["audit"]["collisions"]}
        self.assertEqual(reasons, {"NORMALIZED_EXACT", "HIGH_LEXICAL_SIMILARITY"})
        self.assertEqual({(c["left_partition"], c["right_partition"])
                          for c in report["audit"]["collisions"]},
                         {("train", "heldout"), ("dev", "heldout")})
        self.assertNotIn("curious student studies", process.stdout)

    def test_clean_fixture_has_pass_exit_status(self) -> None:
        process, report = _invoke("--clean")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["mode"], "SYNTHETIC_CLEAN_CONTROL")
        self.assertEqual(report["audit"]["collision_count"], 0)

    def test_deterministic_byte_identical_json(self) -> None:
        first, report = _invoke()
        second, second_report = _invoke()
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(report, second_report)


if __name__ == "__main__":
    unittest.main()
