"""Reproducibility tests for the explicitly synthetic, local exchange."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "synthetic_student_exchange_demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("synthetic_student_exchange_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticStudentExchangeDemoTests(unittest.TestCase):
    def test_three_arms_report_positive_zero_negative_without_weight_claim(self):
        result = _load_demo().run_demo()
        self.assertEqual(result["custody"], "LOCAL_SAME_PROCESS_NO_INDEPENDENT_CUSTODY")
        arms = result["outcomes"]
        self.assertEqual(set(arms), {"genuine", "irrelevant", "misleading"})
        for arm in arms.values():
            self.assertEqual(arm["update"]["kind"], "CONTEXT_UPDATE")
            self.assertIsNone(arm["post_state"]["weights_sha256"])
            self.assertEqual(arm["baseline_heldout"]["correct"], 1)
            self.assertEqual(arm["baseline_heldout"]["total"], 3)
            self.assertEqual(arm["post_heldout"]["specimen_sha256"],
                             arm["baseline_heldout"]["specimen_sha256"])
        self.assertEqual(arms["genuine"]["post_heldout"]["correct"], 2)
        self.assertEqual(arms["irrelevant"]["post_heldout"]["correct"], 1)
        self.assertEqual(arms["misleading"]["post_heldout"]["correct"], 0)

    def test_fixed_synthetic_data_yields_deterministic_receipts(self):
        demo = _load_demo()
        self.assertEqual(demo.run_demo(), demo.run_demo())


if __name__ == "__main__":
    unittest.main()
