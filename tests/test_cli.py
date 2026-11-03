"""Integration smoke test through the public module entry point."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestOfflineDemo(unittest.TestCase):
    def test_demo_is_machine_readable_and_reports_only_a_toy_result(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-m", "allyn", "demo"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["schema"], "allyn.toy-demonstration.v1")
        self.assertEqual(
            record["evidence_class"],
            "DETERMINISTIC_TOY_TEST_NOT_LANGUAGE_GENERALIZATION",
        )
        self.assertEqual(record["prequential"]["events"], 1)
        self.assertIn("transfer", record)
        self.assertIn("baseline_accuracy", record["transfer"])
        self.assertIn("taught_accuracy", record["transfer"])


if __name__ == "__main__":
    unittest.main()
