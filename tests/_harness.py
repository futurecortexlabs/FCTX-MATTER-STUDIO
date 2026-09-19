"""A 60-line test harness, because pytest is not a dependency of this project.

Every test file imports this, registers checks with ``@case``, and ends with
``run(__file__)``.  Output is one line per check so a failure says exactly
which claim stopped being true.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

_CASES: list[tuple[str, Callable[[], None]]] = []


def case(fn: Callable[[], None]) -> Callable[[], None]:
    _CASES.append((fn.__name__, fn))
    return fn


def approx(a: float, b: float, rel: float = 1e-6, abs_: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def require(condition: bool, message: str = "requirement not met") -> None:
    if not condition:
        raise AssertionError(message)


def note(message: str) -> None:
    print(f"       {message}")


def run(source: str | None = None) -> int:
    name = Path(source).stem if source else "tests"
    print(f"\n=== {name} " + "=" * max(4, 60 - len(name)))
    failures = 0
    for label, fn in _CASES:
        start = time.perf_counter()
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures += 1
            ms = (time.perf_counter() - start) * 1000.0
            print(f"FAIL   {label}  ({ms:.0f} ms)")
            traceback.print_exc()
        else:
            ms = (time.perf_counter() - start) * 1000.0
            print(f"ok     {label}  ({ms:.0f} ms)")
    total = len(_CASES)
    print(f"--- {total - failures}/{total} passed")
    return 1 if failures else 0
