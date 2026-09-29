"""Regenerate every figure and number the README claims, from measurement.

    uv run --extra figures python tools/make_figures.py            # all (GPU)
    uv run --extra figures python tools/make_figures.py validation # CPU only

Each figure is produced by running the code it describes:

``validation``   simulated observers with known thresholds answer the real
                 study state machine; the analysis must recover them (CPU).
``haptics``      a hand pressed into the soft body through the full app and
                 GPU solver at five hardnesses; drawn vs real press depth
                 against the configured gain curve.
``substeps``     the same hanging cloth and resting soft body at 6, 12 and
                 24 substeps: how much the material's look depends on the
                 numerics (the XPBD claim is "not much").
``performance``  --benchmark for every preset and for a two-sample study.

Figures go to ``docs/figures/`` and the numbers to
``docs/figures/measurements.json``; both are committed so the README can be
checked against the script that made it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs" / "figures"

BLUE, ORANGE, GREY, INK = "#2a7de1", "#e0782a", "#8a8f98", "#1b1f24"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })
    return plt


# ---------------------------------------------------------------------------
# validation (CPU)
# ---------------------------------------------------------------------------


def validation(results: dict) -> None:
    from fctx import analysis
    from fctx.config import TrackingConfig
    from fctx.core.material import DEFAULT_MATERIALS
    from fctx.core.types import HandPose, MatterKind
    from fctx.study import Phase, Study, StudyConfig

    soft = DEFAULT_MATERIALS[MatterKind.SOFT]
    track = TrackingConfig()
    truth = {"on": 0.05, "off": 0.15}
    slope = 0.3
    rng = random.Random(1)
    low = HandPose(joints=np.full((21, 3), 0.15, np.float32),
                   velocities=np.zeros((21, 3), np.float32))

    def answer(trial) -> int:
        t = truth[trial.condition]
        p = 0.5 + 0.5 / (1 + math.exp(-(math.log(trial.delta) - math.log(t)) / slope))
        return trial.correct_side if rng.random() < p else 1 - trial.correct_side

    study = Study(StudyConfig(trials=12, pseudo_haptics="alternate", seed=5), soft, track)
    dt = 1 / 30
    for _ in range(40):
        for _ in range(4000):
            if study.phase is Phase.DONE:
                break
            touching, key = set(), None
            if study.phase is Phase.EXPLORE:
                touching = {0, 1}
            elif study.phase is Phase.RESPOND:
                key = answer(study.trial)
            study.update([low], touching, dt, key)
        for _ in range(80):
            study.update([], set(), dt)
    rows = study.log.rows
    res = analysis.discrimination(rows, np.random.default_rng(0))

    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    xs = np.geomspace(0.005, 0.6, 200)
    for cond, colour in (("on", BLUE), ("off", ORANGE)):
        d = res[cond]
        true_p = 0.5 + 0.5 / (1 + np.exp(-(np.log(xs) - math.log(truth[cond])) / slope))
        m, s = analysis.fit_psychometric(*d["points"])
        fit_p = 0.5 + 0.5 / (1 + np.exp(-(np.log(xs) - m) / s))
        a.plot(xs, true_p, color=colour, lw=1, ls="--", alpha=0.7)
        a.plot(xs, fit_p, color=colour, lw=2,
               label=f"{cond}: true {truth[cond]:.3f}, fit {d['threshold']:.3f} "
                     f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]")
        delta = [float(r["delta"]) for r in rows if r["condition"] == cond]
        b.plot(range(1, len(delta) + 1), delta, color=colour, lw=1.2, label=cond)
        b.axhline(truth[cond] * 0.84, color=colour, lw=0.8, ls=":")
    a.axhline(0.75, color=GREY, lw=0.8, ls=":")
    a.set_xscale("log")
    a.set_xlabel("hardness difference on the dial")
    a.set_ylabel("P(correct)")
    a.set_title("Psychometric fit recovers the true observer")
    a.legend(frameon=False, fontsize=7.5, loc="upper left")
    a.text(0.03, 0.60, "dashed: true observer\nsolid: fitted from the answers",
           transform=a.transAxes, fontsize=7, color=GREY)
    b.set_yscale("log")
    b.set_xlabel("trial (pooled across 40 simulated visitors)")
    b.set_ylabel("delta presented")
    b.set_title("Interleaved 2-down-1-up staircases")
    b.legend(frameon=False, fontsize=8, loc="lower left")
    b.text(0.99, 0.02, "dotted: true 70.7% point", transform=b.transAxes,
           fontsize=7, color=GREY, ha="right")
    fig.savefig(OUT / "validation.png")
    plt.close(fig)
    e = res["effect"]
    results["validation"] = {
        "true": truth, "trials": len(rows),
        "fit": {c: {"threshold": res[c]["threshold"], "ci": res[c]["ci"],
                    "staircase": res[c]["staircase"]} for c in ("on", "off")},
        "effect_ratio": e["ratio"], "effect_ci": e["ci"],
        "true_effect_ratio": truth["off"] / truth["on"],
    }
    print(f"validation: on {res['on']['threshold']:.3f} (true 0.050), "
          f"off {res['off']['threshold']:.3f} (true 0.150), off/on {e['ratio']:.2f} "
          f"[{e['ci'][0]:.2f}, {e['ci'][1]:.2f}] (true 3.00)")


# ---------------------------------------------------------------------------
# haptics (GPU, full app)
# ---------------------------------------------------------------------------


def _headless_app(kind: str, **kw):
    from fctx.app import MatterStudio
    from fctx.config import preset

    cfg = preset(kind)
    cfg = dataclasses.replace(
        cfg, headless=True, lockstep=True, max_frames=0,
        render=dataclasses.replace(cfg.render, width=480, height=270, bloom=False,
                                   ssao_samples=0),
        tracking=dataclasses.replace(cfg.tracking, source="synthetic"), **kw)
    return MatterStudio(cfg)


def _frame(app, poses=None) -> None:
    dt = app.clock.dt
    ctrl = app.controls.update(dt)
    app._apply_commands(ctrl)
    if poses is None:
        app._update_tracking(ctrl, dt)
    else:
        app._poses = poses
    app.materials = app._with_effective_young(app._evaluate_materials(ctrl.hardness))
    app.state.upload_materials(app.materials)
    app._update_physics(ctrl, dt)


def _flat_hand(cx: float, y: float, cz: float = 0.0):
    from fctx.core.types import HandPose

    j = np.zeros((21, 3), np.float32)
    for i in range(21):
        j[i] = [cx + (i % 5 - 2) * 0.014, y, cz + (i // 5) * 0.016 - 0.03]
    return HandPose(joints=j, velocities=np.zeros((21, 3), np.float32),
                    pinch_point=j[8].copy(), confidence=1.0, track_id=0)


def haptics(results: dict) -> None:
    app = _headless_app("soft")
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    try:
        for h in levels:
            app.controls.state.hardness = app.controls.state.hardness_target = h
            app.solver.reset()
            app.haptics.reset()
            # A hard body is bouncy (restitution rises with the dial): give it
            # time to come to rest, then press on its actual centre.
            for _ in range(400):
                _frame(app, [])
            x = app.state.x.numpy()
            top = float(x[:, 1].max())
            cx, cz = float(x[:, 0].mean()), float(x[:, 2].mean())
            track = []
            for k in range(90):
                y = top + 0.06 - 0.10 * min(1.0, k / 55)
                _frame(app, [_flat_hand(cx, y, cz)])
                track.append(app.haptics.depths(0))
            # The deepest moment of the press: a stiff ball can roll out from
            # under a flat hand afterwards, which ends the press honestly.
            real, shown = max(track, key=lambda d: d[0])
            rows.append((h, real, shown, app.haptics.gain(h)))
            print(f"haptics: hardness {h:.2f}: real {real * 1000:5.1f} mm, drawn "
                  f"{shown * 1000:5.1f} mm, ratio {shown / max(real, 1e-9):.2f} "
                  f"(gain {app.haptics.gain(h):.2f})")
    finally:
        app.close()

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    hs = np.linspace(0, 1, 100)
    from fctx.config import HapticsConfig
    from fctx.haptics import PseudoHaptics
    ph = PseudoHaptics(HapticsConfig())
    ax.plot(hs, [ph.gain(h) for h in hs], color=GREY, lw=1.5,
            label="configured gain  g(h)")
    ax.scatter([r[0] for r in rows], [r[2] / max(r[1], 1e-9) for r in rows], color=BLUE,
               zorder=3, s=36, label="measured: drawn / real press (GPU)")
    for h, real, shown, _ in rows:
        ax.annotate(f"{real * 1000:.0f}->{shown * 1000:.0f} mm", (h, shown / max(real, 1e-9)),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, color=INK)
    ax.set_xlabel("hardness of the pressed body")
    ax.set_ylabel("displayed depth / real depth")
    ax.set_title("Pseudo-haptics: the drawn hand is held back")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.02, 0.04, "measured at the deepest point of a moving press; the offset\n"
            "follows its target with a 30 ms time constant, so hard presses\n"
            "read slightly above the static gain", transform=ax.transAxes,
            fontsize=6.5, color=GREY)
    fig.savefig(OUT / "haptics.png")
    plt.close(fig)
    results["haptics"] = [{"hardness": h, "real_mm": r * 1000, "drawn_mm": s * 1000,
                           "gain": g} for h, r, s, g in rows]


# ---------------------------------------------------------------------------
# substeps (GPU, solver only)
# ---------------------------------------------------------------------------


def substeps(results: dict) -> None:
    import warp as wp

    from fctx.bodies import build_scene
    from fctx.config import preset
    from fctx.core.material import DEFAULT_MATERIALS, evaluate
    from fctx.solver.solver import XPBDSolver
    from fctx.solver.state import SolverState

    wp.init()
    dev = wp.get_device("cuda:0")
    out: dict[str, dict] = {}
    counts = [6, 12, 24]
    for kind, hardness, seconds in (("cloth", 0.3, 4.0), ("soft", 0.3, 3.0)):
        out[kind] = {}
        for n in counts:
            cfg = preset(kind)
            cfg = dataclasses.replace(cfg, solver=dataclasses.replace(
                cfg.solver, substeps=n, wind=(0.0, 0.0, 0.0)))
            bodies = build_scene(cfg.scene)
            st = SolverState(bodies, cfg, dev)
            sv = XPBDSolver(st, cfg)
            st.upload_materials([evaluate(DEFAULT_MATERIALS[b.kind], hardness) for b in bodies])
            dt = 1.0 / cfg.solver.rate_hz
            for _ in range(int(seconds * cfg.solver.rate_hz)):
                sv.step(dt)
            wp.synchronize()
            x = st.x.numpy()
            if kind == "cloth":
                # Mean strain of the structural edges: the material's own
                # response, not confounded by where the hem meets the floor.
                b = bodies[0]
                i, j = b.dist_idx[:, 0], b.dist_idx[:, 1]
                length = np.linalg.norm(x[i] - x[j], axis=1)
                metric = float(np.mean(length / b.dist_rest - 1.0) * 100.0)
                label = "mean edge strain (%)"
            else:
                rest = bodies[0].positions
                metric = float((x[:, 1].max() - x[:, 1].min())
                               / (rest[:, 1].max() - rest[:, 1].min()))
                label = "height / rest height"
            out[kind][n] = metric
            print(f"substeps: {kind} at {n:2d} substeps: {label} {metric:.4f}")
        ref = out[kind][12]
        spread = max(abs(v - ref) for v in out[kind].values())
        out[kind]["label"] = label
        out[kind]["max_deviation_from_12"] = spread
    results["substeps"] = out

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for ax, kind, colour in ((axes[0], "cloth", BLUE), (axes[1], "soft", ORANGE)):
        vals = [out[kind][n] for n in counts]
        ax.plot(counts, vals, marker="o", color=colour)
        ax.set_xticks(counts)
        ax.set_xlabel("substeps per 90 Hz step")
        ax.set_ylabel(out[kind]["label"])
        ax.set_title(f"{kind}, hardness 0.3")
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 2, abs(hi) * 0.02, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Same material, different numerics", fontweight="bold", fontsize=11, y=1.04)
    fig.savefig(OUT / "substeps.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# performance (GPU, subprocess per run)
# ---------------------------------------------------------------------------


def _bench(args: list[str]) -> dict:
    cmd = [sys.executable, "-m", "fctx", "--benchmark", "--frames", "900", *args]
    text = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=900).stdout
    out = {}
    for line in text.splitlines():
        line = line.strip()
        for key, name in (("mean fps", "fps"), ("frame  mean", "frame_ms"),
                          ("physics mean", "physics_ms"), ("render  mean", "render_ms")):
            if line.startswith(key):
                out[name] = float(line.split()[2 if key != "mean fps" else 2])
        if line.startswith("particles"):
            out["particles"] = int(line.split()[1].replace(",", ""))
    return out


REPLOT = False


def performance(results: dict) -> None:
    if REPLOT and results.get("performance", {}).get("runs"):
        _plot_performance(results["performance"]["runs"], results["performance"]["gpu"])
        return
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        study = Path(tmp) / "s.toml"
        study.write_text('[study]\nname = "bench"\noutput = "b.csv"\n', encoding="utf-8")
        for name, args in (("cloth", ["--preset", "cloth"]), ("soft", ["--preset", "soft"]),
                           ("grain", ["--preset", "grain"]),
                           ("study (2 samples)", ["--preset", "soft", "--study", str(study)])):
            # Best of three: on a shared workstation another application's GPU
            # work lands on some runs and not others (a desktop app took one run
            # of this from 219 to 54 fps), and the question is what the code
            # costs, not what the neighbours were doing.
            trials = [_bench(args) for _ in range(3)]
            runs[name] = min(trials, key=lambda r: r.get("frame_ms", 1e9))
            runs[name]["frame_ms_all"] = [t.get("frame_ms") for t in trials]
            r = runs[name]
            print(f"performance: {name:18} {r.get('fps', 0):6.1f} fps  frame "
                  f"{r.get('frame_ms', 0):5.2f} ms  particles {r.get('particles', 0):,}")
    import platform

    gpu = "unknown GPU"
    try:
        import warp as wp
        wp.init()
        gpu = wp.get_device("cuda:0").name
    except Exception:  # noqa: BLE001
        pass
    results["performance"] = {"gpu": gpu, "cpu": platform.processor(), "runs": runs,
                              "note": "headless 1600x900, 900 frames, synthetic hand, "
                                      "best of 3 runs on a shared workstation"}
    _plot_performance(runs, gpu)


def _plot_performance(runs: dict, gpu: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    names = list(runs)
    ms = [runs[n].get("frame_ms", 0) for n in names]
    bars = ax.barh(names, ms, color=[BLUE, BLUE, BLUE, ORANGE])
    for b, n in zip(bars, names):
        r = runs[n]
        ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2,
                f"{r.get('frame_ms', 0):.1f} ms  ({r.get('fps', 0):.0f} fps, "
                f"{r.get('particles', 0):,} particles)", va="center", fontsize=8)
    ax.axvline(1000 / 60, color=GREY, lw=0.8, ls=":")
    ax.text(1000 / 60, -0.62, "60 Hz budget", color=GREY, fontsize=8, ha="center")
    ax.invert_yaxis()
    ax.set_xlabel(f"mean frame time, ms  (physics + render, {gpu})")
    ax.set_title("Whole frame, headless 1600x900 (best of 3)")
    ax.set_xlim(0, max(max(ms) * 1.9, 19.0))
    fig.savefig(OUT / "performance.png")
    plt.close(fig)


PARTS = {"validation": validation, "haptics": haptics, "substeps": substeps,
         "performance": performance}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parts", nargs="*", choices=[*PARTS, []], default=[])
    ap.add_argument("--replot", action="store_true",
                    help="redraw performance from measurements.json instead of rerunning")
    args = ap.parse_args(argv)
    global REPLOT
    REPLOT = args.replot
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "measurements.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in args.parts or list(PARTS):
        t0 = time.perf_counter()
        PARTS[name](results)
        print(f"  [{name}: {time.perf_counter() - t0:.0f} s]")
    results["generated"] = time.strftime("%Y-%m-%d")
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
