"""Analyse a study log -- kept for existing scripts; the tool is ``fctx-analyze``.

    uv run fctx-analyze studies/results/hardness_jnd.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fctx.analysis import *  # noqa: E402,F403
from fctx.analysis import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
