"""The venue layer: dial gestures, prompts, the visitor record, the catalogue.

None of this touches the GPU.  The director, coach and log are driven with
hand-built poses and grip facts, the way the application drives them.
"""

from __future__ import annotations

import csv
import dataclasses
import tempfile
from pathlib import Path

import numpy as np
from _harness import approx, case, note, require, run

from fctx.config import ExhibitConfig, TrackingConfig
from fctx.core import catalog
from fctx.core.material import DEFAULT_MATERIALS, evaluate
from fctx.core.types import HandPose, MatterKind
from fctx.exhibit import Coach, HardnessDirector, SessionLog
from fctx.ui.controls import ControlState


def pose(track_id: int, y: float, *, pinching: bool = False,
         confidence: float = 1.0) -> HandPose:
    joints = np.zeros((21, 3), np.float32)
    joints[:, 1] = y
    return HandPose(joints=joints, velocities=np.zeros((21, 3), np.float32),
                    pinching=pinching, pinch=1.0 if pinching else 0.0,
                    confidence=confidence, track_id=track_id)


TRACKING = TrackingConfig()
TOP = TRACKING.stage_center_y + TRACKING.stage_half_height
BOTTOM = TRACKING.stage_center_y - TRACKING.stage_half_height


# -- director ---------------------------------------------------------------


@case
def the_free_hand_sets_the_dial_only_while_the_other_hand_holds() -> None:
    d = HardnessDirector(ExhibitConfig(hardness_by_free_hand=True), TRACKING)
    s = ControlState(hardness_target=0.35)
    high = pose(2, TOP)
    d.update([pose(1, 0.3), high], set(), s, 1 / 60)
    require(d.driving is None and approx(s.hardness_target, 0.35),
            "a free hand moved the dial with nothing held")
    d.update([pose(1, 0.3, pinching=True), high], {1}, s, 1 / 60)
    require(d.driving == "free_hand" and s.hardness_target == 1.0,
            f"a hand at the top of the stage did not set hard: {s.hardness_target}")
    d.update([pose(1, 0.3, pinching=True), pose(2, BOTTOM)], {1}, s, 1 / 60)
    require(s.hardness_target == 0.0, "a hand at the bottom did not set soft")
    mid = TRACKING.stage_center_y
    d.update([pose(1, 0.3, pinching=True), pose(2, mid)], {1}, s, 1 / 60)
    require(abs(s.hardness_target - 0.5) < 1e-6,
            f"the stage centre is not the middle of the dial: {s.hardness_target}")
    # A second pinching hand is not a free hand; neither is a doubtful one.
    d.update([pose(1, 0.3, pinching=True), pose(2, TOP, pinching=True)], {1, 2}, s, 1 / 60)
    require(d.driving is None, "a holding hand was taken for the free hand")
    d.update([pose(1, 0.3, pinching=True), pose(2, TOP, confidence=0.2)], {1}, s, 1 / 60)
    require(d.driving is None, "a barely tracked hand was allowed to drive")
    note("free hand: bottom = 0, centre = 0.5, top = 1; only while something is held")


@case
def the_sweep_runs_only_while_holding_and_starts_where_the_dial_is() -> None:
    cfg = ExhibitConfig(hardness_by_free_hand=False, sweep_while_holding=True,
                        sweep_period=4.0)
    d = HardnessDirector(cfg, TRACKING)
    s = ControlState(hardness_target=0.8)
    d.update([pose(1, 0.3)], set(), s, 0.1)
    require(d.driving is None and s.hardness_target == 0.8, "swept with nothing held")
    d.update([pose(1, 0.3, pinching=True)], {1}, s, 1e-6)
    require(d.driving == "sweep" and abs(s.hardness_target - 0.8) < 1e-3,
            f"the sweep jumped on its first frame: {s.hardness_target}")
    lo, hi = 1.0, 0.0
    for _ in range(400):
        d.update([pose(1, 0.3, pinching=True)], {1}, s, 0.01)
        lo, hi = min(lo, s.hardness_target), max(hi, s.hardness_target)
    require(lo < 0.02 and hi > 0.98, f"one period did not cover the dial: {lo}..{hi}")
    held = s.hardness_target
    d.update([pose(1, 0.3)], set(), s, 0.5)
    require(s.hardness_target == held, "the dial moved after the hand let go")
    note("sweep: continuous from the current value, full range in one period, stops on release")


@case
def a_free_hand_beats_the_sweep() -> None:
    cfg = ExhibitConfig(hardness_by_free_hand=True, sweep_while_holding=True)
    d = HardnessDirector(cfg, TRACKING)
    s = ControlState(hardness_target=0.5)
    d.update([pose(1, 0.3, pinching=True), pose(2, TOP)], {1}, s, 0.1)
    require(d.driving == "free_hand" and s.hardness_target == 1.0)
    d.update([pose(1, 0.3, pinching=True)], {1}, s, 0.1)
    require(d.driving == "sweep", "the sweep did not take over when the free hand left")


# -- coach ------------------------------------------------------------------


def _coach(**kw) -> Coach:
    return Coach(ExhibitConfig(coach=True, prompt_seconds=2.0, session_gap=1.0, **kw))


@case
def the_coach_says_one_thing_at_a_time_and_then_stops() -> None:
    c = _coach()
    dt = 0.1
    require(c.update(0, False, 0.3, None, dt) is None, "prompted an empty room")
    # A hand passing through does not trigger anything.
    for _ in range(5):
        p = c.update(1, False, 0.3, None, dt)
    require(p is None, "prompted a hand that had only just appeared")
    for _ in range(10):
        p = c.update(1, False, 0.3, None, dt)
    require(p is not None and p.text == c.cfg.prompt_grab, f"expected the grab prompt, got {p}")
    # Grab: the hardness prompt replaces it.
    p = c.update(1, True, 0.3, None, dt)
    require(p is not None and p.text == c.cfg.prompt_hardness, f"expected hardness prompt, got {p}")
    # The dial moves while holding: the release prompt, once.
    for _ in range(5):
        p = c.update(1, True, 0.3, None, dt)
    p = c.update(1, True, 0.6, "free_hand", dt)
    require(p is not None and p.text == c.cfg.prompt_release, f"expected release prompt, got {p}")
    # Let go, wait out the prompt: silence.
    for _ in range(40):
        p = c.update(1, False, 0.6, None, dt)
    require(p is None and c.stage == "done", f"the coach did not fall silent: {c.stage} {p}")
    # Everybody leaves; the next visitor gets the sequence again.
    for _ in range(15):
        c.update(0, False, 0.6, None, dt)
    require(c.stage == "nobody", "the coach did not reset for the next visitor")
    note("grab -> hardness -> release, then silent; resets after the session gap")


@case
def prompts_fade_time_out_and_repeat_while_stuck() -> None:
    c = _coach()
    dt = 0.05
    for _ in range(int(Coach.SETTLE / dt) + 2):
        c.update(1, False, 0.3, None, dt)
    p = c.update(1, False, 0.3, None, dt)
    require(p is not None and 0.0 < p.alpha < 1.0, f"no fade-in: {p}")
    alphas = [c.update(1, False, 0.3, None, dt).alpha for _ in range(10)]
    require(alphas == sorted(alphas), "alpha did not rise monotonically")
    # Times out after prompt_seconds...
    for _ in range(int(2.0 / dt) + 12):
        p = c.update(1, False, 0.3, None, dt)
    require(p is None, "the grab prompt never timed out")
    # ...and comes back if the visitor is still stuck.
    for _ in range(int(Coach.REPEAT_AFTER / dt) + 2):
        p = c.update(1, False, 0.3, None, dt)
    require(p is not None and p.text == c.cfg.prompt_grab, "the prompt did not repeat")


@case
def the_sweep_gets_its_own_wording_and_moves_on_by_itself() -> None:
    c = _coach()
    dt = 0.1
    for _ in range(12):
        c.update(1, False, 0.3, None, dt)
    p = c.update(1, True, 0.3, "sweep", dt)
    require(p is not None and p.text == c.cfg.prompt_feel, f"expected the feel prompt, got {p}")
    for _ in range(25):
        p = c.update(1, True, 0.3, "sweep", dt)
    require(c.stage == "changed", "a sweep never counted as the hardness changing")


@case
def a_coach_that_is_off_says_nothing() -> None:
    c = Coach(ExhibitConfig(coach=False))
    for _ in range(50):
        require(c.update(1, True, 0.5, "free_hand", 0.1) is None)


# -- session log ------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.wall = 1_800_000_000.0

    def mono(self) -> float:
        return self.t

    def now(self) -> float:
        return self.wall

    def tick(self, dt: float) -> None:
        self.t += dt
        self.wall += dt


@case
def visitors_are_runs_of_hands_separated_by_the_gap() -> None:
    clk = Clock()
    log = SessionLog(None, gap=5.0, wall_clock=clk.now, monotonic=clk.mono)
    for _ in range(30):            # a visitor, 3 s
        log.hands(1)
        clk.tick(0.1)
    for _ in range(30):            # steps away for 3 s: same visitor
        log.hands(0)
        clk.tick(0.1)
    for _ in range(20):
        log.hands(2)
        clk.tick(0.1)
    require(log.summary.visitors == 0 and log.summary.open_visitor,
            "a 3 s absence ended the visit")
    for _ in range(60):            # gone for 6 s: over
        log.hands(0)
        clk.tick(0.1)
    require(log.summary.visitors == 1 and not log.summary.open_visitor,
            f"one visitor expected: {log.summary}")
    require(abs(log.summary.interaction_seconds - 8.0) < 0.3,
            f"visit length wrong: {log.summary.interaction_seconds}")
    log.hands(1)
    s = log.close()
    require(s.visitors == 2, "an open visit was not closed at exit")
    note("gap 5 s: a 3 s absence is the same visitor, 6 s is the next one")


@case
def grabs_holds_and_dial_moves_are_counted_and_written() -> None:
    clk = Clock()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "logs" / "events.csv"
        log = SessionLog(path, gap=5.0, wall_clock=clk.now, monotonic=clk.mono)
        log.hands(1)
        log.grab(7)
        for i in range(20):
            clk.tick(0.1)
            log.hands(1)
            log.dial(0.3 + 0.03 * i, True)   # 0.3 -> 0.87: moves past 0.5 and 0.7
        log.release(7)
        log.dial(0.9, False)
        log.attract(True)
        log.attract(False)
        log.camera(False, "unplugged")
        log.camera(True, "back")
        log.error("RuntimeError('x')")
        s = log.close()
        require(s.grabs == 1 and abs(s.hold_seconds - 2.0) < 1e-6, f"hold wrong: {s}")
        require(s.dial_moves == 2, f"expected 2 dial moves, got {s.dial_moves}")
        require(s.attract_starts == 1 and s.camera_losses == 1 and s.errors == 1)
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        events = [r["event"] for r in rows]
        for expected in ("start", "visitor_start", "grab", "dial", "release", "attract_on",
                         "attract_off", "camera_lost", "camera_back", "error",
                         "visitor_end", "stop"):
            require(expected in events, f"{expected} missing from {events}")
        require(rows[0]["time"].startswith("20"), "no wall-clock stamp")
        # A second run appends without a second header.
        log2 = SessionLog(path, gap=5.0, wall_clock=clk.now, monotonic=clk.mono)
        log2.close()
        text = path.read_text(encoding="utf-8")
        require(text.count("time,uptime_s,event,detail") == 1, "header repeated")
        note(f"{len(rows)} rows; describe: {log.describe()}")


@case
def the_report_tool_reads_what_the_log_writes() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import report

    clk = Clock()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.csv"
        log = SessionLog(path, gap=2.0, wall_clock=clk.now, monotonic=clk.mono)
        for _visitor in range(3):
            for _ in range(10):
                log.hands(1)
                clk.tick(0.1)
            log.grab(1)
            clk.tick(1.0)
            log.release(1)
            for _ in range(40):
                log.hands(0)
                clk.tick(0.1)
        log.close()
        rows = report._parse(path)
        hours, total = report.summarise(rows)
        require(total["visitors"] == 3 and total["grabs"] == 3, f"totals: {dict(total)}")
        require(total["uptime"] > 0, "uptime not derived from start/stop")
        require(report.main([str(path)]) == 0, "report.main failed")


# -- catalogue --------------------------------------------------------------


@case
def the_catalogue_names_the_dial_and_the_dial_can_be_stepped_through_it() -> None:
    for kind in MatterKind:
        params = DEFAULT_MATERIALS[kind]
        entries = catalog.for_kind(catalog.DEFAULT_CATALOG, kind)
        require(len(entries) >= 5, f"{kind}: a thin catalogue")
        require(not catalog.check_range(catalog.DEFAULT_CATALOG, params),
                f"{kind}: built-in entries outside the dial")
        # Round trip: the hardness of an entry evaluates to the entry.
        for e in entries:
            h = catalog.hardness_for(params, e.value)
            m = evaluate(params, h)
            require(catalog.nearest(catalog.DEFAULT_CATALOG, m) is e,
                    f"{kind}: {e.name} at h={h:.3f} is nearest to something else")
            require(abs(catalog.headline(m) / e.value - 1.0) < 1e-3,
                    f"{kind}: {e.name} round trip off: {catalog.headline(m)} vs {e.value}")
        # Stepping walks end to end and does not stick.
        m = evaluate(params, 0.0)
        seen = []
        for _ in range(len(entries) + 2):
            e = catalog.step(catalog.DEFAULT_CATALOG, m, +1)
            seen.append(e.name)
            m = evaluate(params, catalog.hardness_for(params, e.value))
        require(seen[:len(entries)] == [e.name for e in entries]
                or seen[:len(entries) - 1] == [e.name for e in entries[1:]],
                f"{kind}: stepping up did not walk the catalogue: {seen}")
        require(seen[-1] == entries[-1].name, "stepping past the end did not stay there")
        e = catalog.step(catalog.DEFAULT_CATALOG, m, -1)
        require(e is entries[-2], "stepping down from the top did not give the next one down")
    note("every built-in entry round-trips; M/N walk the list end to end")


@case
def a_venue_catalogue_loads_and_a_bad_one_says_why() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "materials.toml"
        good.write_text(
            '[[material]]\nname = "Our foam"\nkind = "soft"\nvalue = 4.5e4\nnote = "seat"\n'
            '[[material]]\nname = "Our fabric"\nkind = "cloth"\nstretch = 5000\n',
            encoding="utf-8")
        entries = catalog.load_catalog(good)
        require(len(entries) == 2 and entries[0].name == "Our foam"
                and entries[1].kind is MatterKind.CLOTH and entries[1].value == 5000.0)
        for text, fragment in (
                ('[[material]]\nname = "x"\nkind = "steel"\nvalue = 1\n', "steel"),
                ('[[material]]\nname = "x"\nkind = "soft"\nvalue = -1\n', "value"),
                ('[[material]]\nname = "x"\nkind = "soft"\nvalue = 1\nyoung_mod = 2\n', "young_mod"),
                ('[[material]]\nkind = "soft"\nvalue = 1\n', "name"),
                ('nothing = 1\n', "[[material]]"),
                ('[[material\n', ""),
        ):
            bad = Path(tmp) / "bad.toml"
            bad.write_text(text, encoding="utf-8")
            try:
                catalog.load_catalog(bad)
            except catalog.CatalogError as exc:
                require(fragment in str(exc), f"error does not mention {fragment!r}: {exc}")
                continue
            raise AssertionError(f"accepted: {text!r}")
        try:
            catalog.load_catalog(Path(tmp) / "missing.toml")
        except catalog.CatalogError:
            pass
        else:
            raise AssertionError("a missing file did not raise")


@case
def the_exhibit_section_round_trips_through_the_config_file() -> None:
    from fctx.config import AppConfig
    from fctx.settings import apply_toml, dump_config, load_config

    cfg = apply_toml({"exhibit": {"coach": True, "prompt_grab": "つまんでください",
                                  "analytics": "logs/e.csv", "sweep_while_holding": True}})
    require(cfg.exhibit.coach and cfg.exhibit.prompt_grab == "つまんでください")
    require(isinstance(cfg.exhibit.analytics, Path) and cfg.exhibit.sweep_while_holding)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v.toml"
        path.write_text(dump_config(cfg), encoding="utf-8")
        back = load_config(path)
    require(back.exhibit == cfg.exhibit, "the exhibit section did not round-trip")
    require(dataclasses.replace(back, exhibit=AppConfig().exhibit) ==
            dataclasses.replace(cfg, exhibit=AppConfig().exhibit))


if __name__ == "__main__":
    raise SystemExit(run(__file__))
