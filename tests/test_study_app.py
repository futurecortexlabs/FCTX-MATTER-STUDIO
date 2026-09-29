"""Study mode and pseudo-haptics through the real application (GPU + OpenGL).

The pure logic is covered in test_study.py and test_haptics.py; these check
the wiring: two blinded samples on stage, the study driving the materials
and the pseudo-haptics condition, a staff answer landing in the CSV, and the
solver's contact report holding a real hand back harder on the harder sample.
"""

from __future__ import annotations

import csv
import dataclasses
import tempfile
from pathlib import Path

import numpy as np
from _harness import case, note, require, run

from fctx.config import preset
from fctx.core.types import HandPose
from fctx.haptics import palm_centre
from fctx.study import Phase


def _study_file(tmp: Path, **extra: object) -> Path:
    fields = {"name": "t", "protocol": "discrimination", "trials": 2,
              "intro_seconds": 0.1, "explore_min": 0.1, "touch_min": 0.0,
              "recorded_seconds": 0.1, "pseudo_haptics": "alternate",
              "output": "out.csv", "resolution": 12}
    fields.update(extra)
    lines = ["[study]"]
    for k, v in fields.items():
        lines.append(f"{k} = " + (f'"{v}"' if isinstance(v, str) else repr(v)))
    path = tmp / "study.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _app(study: Path | None = None, kind: str = "soft"):
    from fctx.app import MatterStudio

    cfg = preset(kind)
    cfg = dataclasses.replace(
        cfg, headless=True, lockstep=True, max_frames=0, study=study,
        render=dataclasses.replace(cfg.render, width=480, height=270, bloom=False,
                                   ssao_samples=0),
        tracking=dataclasses.replace(cfg.tracking, source="synthetic"))
    return MatterStudio(cfg)


def _frame(app, key: int | None = None, poses: list[HandPose] | None = None) -> None:
    dt = app.clock.dt
    ctrl = app.controls.update(dt)
    ctrl.study_key = key
    app._apply_commands(ctrl)
    if poses is None:
        app._update_tracking(ctrl, dt)
    else:
        app._poses = poses
    app.materials = app._with_effective_young(app._evaluate_materials(ctrl.hardness))
    app.state.upload_materials(app.materials)
    app._update_physics(ctrl, dt)


def flat_hand(cx: float, y: float, track_id: int = 0) -> HandPose:
    j = np.zeros((21, 3), np.float32)
    for i in range(21):
        j[i] = [cx + (i % 5 - 2) * 0.014, y, (i // 5) * 0.016 - 0.03]
    return HandPose(joints=j, velocities=np.zeros((21, 3), np.float32),
                    pinch_point=j[8].copy(), confidence=1.0, track_id=track_id)


@case
def study_mode_puts_two_blinded_samples_on_stage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = _app(_study_file(Path(tmp)))
        try:
            require(len(app.bodies) == 2, f"{len(app.bodies)} bodies")
            xs = [float(b.positions[:, 0].mean()) for b in app.bodies]
            require(xs[0] < -0.1 < 0.1 < xs[1], f"samples not left and right: {xs}")
            require(not app.cfg.render.show_hud and app.cfg.idle_demo == 0.0,
                    "the HUD or attract mode is on in a blind study")
            app.study.hardness = [0.1, 0.9]
            mats = app._evaluate_materials(0.5)
            require(abs(mats[0].hardness - 0.1) < 1e-9 and abs(mats[1].hardness - 0.9) < 1e-9)
            look = [(m.color, m.roughness, m.metallic, m.translucency) for m in mats]
            require(look[0] == look[1], f"the samples look different: {look}")
            require(mats[0].young < mats[1].young / 100, "the physics did not differ")
            # The keyboard cannot change the scene or start the demo.
            ctrl = app.controls.update(app.clock.dt)
            ctrl.preset_requested = "cloth"
            ctrl.demo_toggle_requested = True
            app._apply_commands(ctrl)
            require(len(app.bodies) == 2 and app.demo is None,
                    "a preset key or the demo key took over the study")
            note(f"samples at x = {xs[0]:.2f}, {xs[1]:.2f}; one look for both")
        finally:
            app.close()


@case
def a_staff_answer_is_recorded_and_the_next_trial_starts_clean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _study_file(Path(tmp))
        app = _app(path)
        try:
            s = app.study
            seen_conditions = set()
            for _ in range(400):
                key = 1 if s.phase is Phase.RESPOND else None
                if s.phase is Phase.WAITING and s.participant == 0:
                    key = 0          # a key press brings a participant in
                _frame(app, key)
                if s.trial is not None:
                    seen_conditions.add(s.condition)
                    require(app.haptics.enabled == (s.condition == "on"),
                            "pseudo-haptics does not follow the trial's condition")
                if s.phase is Phase.DONE:
                    break
            require(s.phase is Phase.DONE, f"the study did not finish: {s.phase}")
            require(seen_conditions == {"on", "off"}, f"conditions seen: {seen_conditions}")
            out = Path(tmp) / "out.csv"
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            require(len(rows) == 2 and all(r["response_mode"] == "key" for r in rows),
                    f"rows: {rows}")
            require({r["condition"] for r in rows} == {"on", "off"})
            note(f"2 trials recorded to {out.name}: "
                 + ", ".join(f"{r['condition']} {r['response_side']} correct={r['correct']}"
                             for r in rows))
        finally:
            app.close()


@case
def the_drawn_hand_is_held_back_harder_on_the_harder_sample() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = _app(_study_file(Path(tmp), pseudo_haptics="on"))
        try:
            app.study.hardness = [0.0, 1.0]
            app.haptics.enabled = True
            for _ in range(90):                     # let both samples settle
                _frame(app, poses=[])
            depths = {}
            for side, body in ((0, app.bodies[0]), (1, app.bodies[1])):
                app.solver.reset()
                app.haptics.reset()
                for _ in range(60):
                    _frame(app, poses=[])
                x = app.state.x.numpy()
                lo = sum(b.positions.shape[0] for b in app.bodies[:side])
                top = float(x[lo:lo + body.positions.shape[0], 1].max())
                cx = float(body.positions[:, 0].mean())
                # Come down from above and press 5 cm past the top.
                for k in range(80):
                    y = top + 0.06 - 0.11 * min(1.0, k / 50)
                    app.study.hardness = [0.0, 1.0]
                    app.study.phase = Phase.EXPLORE
                    _frame(app, poses=[flat_hand(cx, y)])
                real, shown = app.haptics.depths(0)
                drawn = float(palm_centre(app._display_poses[0])[1])
                depths[side] = (real, shown, top, drawn, app.haptics.touching(0))
            (r0, s0, _, _, t0), (r1, s1, _, _, t1) = depths[0], depths[1]
            note(f"soft: real {r0 * 1000:.0f} mm drawn {s0 * 1000:.0f} mm (body {t0}); "
                 f"hard: real {r1 * 1000:.0f} mm drawn {s1 * 1000:.0f} mm (body {t1})")
            require(t0 == 0 and t1 == 1, f"contact attributed to the wrong sample: {t0}, {t1}")
            # The contact report is a frame behind (async readback), so the anchor
            # lands a few millimetres into the surface: the press reads a little short.
            require(r0 > 0.01 and r1 > 0.01, "the press never registered as contact")
            require(s0 / r0 > 2.5 * (s1 / r1),
                    "the hard sample did not hold the drawn hand back much harder")
        finally:
            app.close()


@case
def free_play_still_runs_with_pseudo_haptics_on_every_preset() -> None:
    for kind in ("cloth", "soft", "grain"):
        app = _app(None, kind)
        try:
            require(app.haptics.enabled and app.study is None)
            for _ in range(120):
                _frame(app)
            x = app.state.x.numpy()
            require(np.isfinite(x).all(), f"{kind}: non-finite positions")
        finally:
            app.close()
    note("cloth, soft, grain: 120 frames each on the synthetic hand")


if __name__ == "__main__":
    raise SystemExit(run(__file__))
