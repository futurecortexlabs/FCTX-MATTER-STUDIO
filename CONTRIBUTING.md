# Contributing

Thanks for looking. Issues and pull requests are welcome.

## Setup

```bash
uv sync --extra figures
uv run python tools/download_models.py
uv run fctx --check
```

A CUDA-capable NVIDIA GPU is needed for the physics and rendering tests; the
geometry, tracking, study and analysis tests run on any machine:

```bash
uv run python tools/run_tests.py            # everything (GPU, ~4 min)
uv run python tools/run_tests.py --cpu      # what CI runs
uvx ruff check src tests tools
```

## Ground rules

* **Measure, then claim.** Numbers in the README and `docs/` come from a
  script in this repository (`tools/make_figures.py`, `tools/soak.py`,
  `--benchmark`). A change that moves one updates it.
* **Every behaviour change gets a test** in `tests/`, named as a sentence
  about what must hold (`a_hard_surface_holds_the_drawn_hand_back...`).
* **Warp kernels**: declare loop-mutated locals as `x = int(0)` / `float(0.0)`;
  ruff's UP018 is disabled for `kernels.py` for this reason.
* **Determinism**: headless runs are lockstep and bit-reproducible
  (`test_smoke.py`). Do not introduce wall-clock or unseeded randomness on
  that path.

The design, the maths and the known limits are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); read §7 (stability rules)
before touching the solver.
