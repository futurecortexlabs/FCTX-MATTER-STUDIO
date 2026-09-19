"""Run every test file and summarise.

    .venv/Scripts/python.exe tools/run_tests.py            # everything
    .venv/Scripts/python.exe tools/run_tests.py --cpu      # skip GPU tests
    .venv/Scripts/python.exe tools/run_tests.py material   # matching names

Each test file is a standalone program, so they run in separate processes: a
segfault in a Warp kernel or a lost OpenGL context takes down one file instead
of the whole suite.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS = REPO_ROOT / "tests"
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

#: Test files that need a GPU or a display.
NEEDS_DEVICE = {"test_solver.py", "test_render.py", "test_smoke.py", "test_resilience.py"}


def main(argv: list[str]) -> int:
    cpu_only = "--cpu" in argv
    patterns = [a for a in argv if not a.startswith("-")]

    files = sorted(p for p in TESTS.glob("test_*.py"))
    if patterns:
        files = [p for p in files
                 if any(pat in p.name for pat in patterns)]
    if cpu_only:
        files = [p for p in files if p.name not in NEEDS_DEVICE]
    if not files:
        print("no test files matched")
        return 1

    width = max(len(p.name) for p in files)
    results: list[tuple[str, int, float, str]] = []
    for path in files:
        start = time.perf_counter()
        proc = subprocess.run(
            [str(PYTHON), path.name],
            cwd=TESTS, capture_output=True, text=True, timeout=1800)
        elapsed = time.perf_counter() - start
        out = proc.stdout + proc.stderr
        results.append((path.name, proc.returncode, elapsed, out))
        status = "ok  " if proc.returncode == 0 else "FAIL"
        summary = ""
        for line in reversed(out.splitlines()):
            if line.startswith("--- ") and "passed" in line:
                summary = line[4:]
                break
        print(f"{status} {path.name:<{width}}  {elapsed:6.1f} s   {summary}")

    failed = [r for r in results if r[1] != 0]
    if failed:
        print("\n" + "=" * 70)
        for name, code, _, out in failed:
            print(f"\n----- {name} (exit {code}) " + "-" * 30)
            print(out.rstrip()[-8000:])
    total = len(results)
    print(f"\n{total - len(failed)}/{total} test files passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
