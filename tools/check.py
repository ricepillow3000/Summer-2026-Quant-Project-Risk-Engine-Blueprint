"""
One command before every push - the easy path is the checked path.

    python -m tools.check

Runs, in order, and stops nothing early so you see every problem at once:

1. lint      ruff: undefined names, unused imports/variables, syntax errors
2. security  bandit: medium/high findings in src/, main.py, tools/
3. tests     tests.test_engine - math invariants, security controls, the
             layering rule (src/ never imports the UI), design rules
4. narrative tests.test_narrative

Live Yahoo checks skip themselves offline; for the live answer run the
"Live audit" GitHub workflow (RUNBOOK.md, "After every deploy").
Exit code 1 if any step fails.
"""

import subprocess
import sys

PY = sys.executable
STEPS = [
    ("lint", [PY, "-m", "ruff", "check", "--select", "F,E9,PLE", "--ignore", "F541",
              "main.py", "src", "tests", "tools"]),
    ("security", [PY, "-m", "bandit", "-q", "-r", "src", "main.py", "tools", "-ll"]),
    ("tests", [PY, "-m", "tests.test_engine"]),
    ("narrative", [PY, "-m", "tests.test_narrative"]),
]


def main() -> int:
    results = []
    for name, cmd in STEPS:
        print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
        code = subprocess.run(cmd).returncode  # nosec B603 - fixed argv, no shell
        if code and name in ("lint", "security"):
            print(f"{name} failed - if the tool is missing: python -m pip install ruff bandit")
        results.append((name, code))
    print("\n" + "\n".join(f"{'PASS' if c == 0 else 'FAIL'}  {n}" for n, c in results))
    return 1 if any(c for _, c in results) else 0


if __name__ == "__main__":
    sys.exit(main())
