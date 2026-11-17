"""Portable, offline verification entrypoints for Bash and PowerShell."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH_SCRIPT = ROOT / "scripts" / "check.sh"
POWERSHELL_SCRIPT = ROOT / "scripts" / "check.ps1"


class CheckScriptsTests(unittest.TestCase):
    def _skip_if_nested(self) -> None:
        if os.environ.get("ALLYN_CHECK_RUNNING") == "1":
            self.skipTest("Avoid recursively invoking the whole-suite checker")

    def test_bash_syntax(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("Bash is unavailable on this host")
        run = subprocess.run((bash, "-n", str(BASH_SCRIPT)), capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_bash_works_outside_repo_and_ignores_inherited_pythonpath(self) -> None:
        self._skip_if_nested()
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("Bash is unavailable on this host")
        with tempfile.TemporaryDirectory() as other_cwd:
            environment = {**os.environ, "PYTHONPATH": str(Path(other_cwd) / "nonexistent")}
            run = subprocess.run(
                (bash, str(BASH_SCRIPT)), cwd=other_cwd, env=environment,
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Ran ", run.stderr)
        self.assertIn('"DETERMINISTIC_TOY_TEST_NOT_LANGUAGE_GENERALIZATION"', run.stdout)
        self.assertIn('"schema": "allyn.teaching-cycle', run.stdout)

    def test_bash_propagates_test_failure(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("Bash is unavailable on this host")
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "python3"
            executable.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = '-c' ]; then exit 0; fi\n"
                "exit 17\n", encoding="utf-8",
            )
            executable.chmod(0o755)
            environment = {**os.environ, "PATH": temp_dir + os.pathsep + os.environ.get("PATH", "")}
            run = subprocess.run(
                (bash, str(BASH_SCRIPT)), cwd=temp_dir, env=environment,
                capture_output=True, text=True, timeout=15,
            )
        self.assertEqual(run.returncode, 17)

    def test_powershell_uses_script_directory_and_fails_closed(self) -> None:
        source = POWERSHELL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("$PSScriptRoot", source)
        self.assertIn("$LASTEXITCODE", source)
        self.assertIn("finally", source)
        self.assertIn("$env:PYTHONPATH", source)
        self.assertNotIn("Invoke-WebRequest", source)
        self.assertNotIn("Invoke-RestMethod", source)

    def test_powershell_works_outside_repo_when_available(self) -> None:
        self._skip_if_nested()
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            self.skipTest("PowerShell is unavailable on this host")
        with tempfile.TemporaryDirectory() as other_cwd:
            run = subprocess.run(
                (pwsh, "-NoProfile", "-File", str(POWERSHELL_SCRIPT)),
                cwd=other_cwd, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Ran ", run.stderr)


if __name__ == "__main__":
    unittest.main()
