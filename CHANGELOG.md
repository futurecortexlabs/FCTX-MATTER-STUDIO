# Changelog

## v0.2.0 — 2026-09-30

First public release: a touchless sensory-evaluation lab built on
bare-hand GPU soft-matter physics.

### Sensory evaluation
- **Blind A/B studies** (`fctx --study file.toml`): two identical samples
  side by side, colour and HUD blinded; visitors press both and answer by
  raising a hand over their choice (dwell-to-confirm, no touch, no mouse).
- **Discrimination**: 2AFC with an n-down-1-up staircase per condition,
  pooled across visitors and restored from the CSV after a restart.
- **Preference**: every ordered pair of catalogue materials.
- **`fctx-analyze`**: psychometric fit with bootstrap intervals, staircase
  threshold, modulus Weber fraction, pseudo-haptics effect (off/on ratio),
  Bradley–Terry ranking, side-bias and response-time checks.

### Pseudo-haptics
- The drawn and simulated hand is held back by a control/display ratio that
  falls with the pressed body's hardness; per-(hand, body) contact kernel
  with asynchronous double-buffered pinned readback (no measured frame cost).

### Physics, interaction, rendering
- XPBD solver in NVIDIA Warp: cloth, stable Neo-Hookean soft bodies,
  24,000-grain granular matter; graph-coloured Gauss–Seidel; CUDA graphs.
- Embedded render skin; wrist-anchored grabs with palm-frame rotation.
- ModernGL HDR pipeline with shadows, SSAO, bloom and a CJK label cache.

### Operations
- Kiosk mode: attract mode, camera hot-plug recovery, resilient frame loop,
  scheduled restart when nobody is present, visitor analytics and daily
  report, material catalogue, visitor prompts in any language.
- `setup.bat`, `--check` (validates config and study files), soak test.

### Evidence and packaging
- `tools/make_figures.py` regenerates every figure and number in the README.
- Corrected an overstated claim: the converged material is substep-independent,
  but at one iteration per substep cloth strain varies 2.3% / 0.78% / 0.22%
  at 6 / 12 / 24 substeps.
- MIT license, CITATION.cff, `fctx` / `fctx-analyze` entry points,
  CI on Linux and Windows.

### Known limits
- Not yet validated with human participants; see README "Limits".

## v0.1.0

Internal: hand-driven cloth, soft body and granular matter with a live
hardness dial, demo choreography and showcase renderer.
