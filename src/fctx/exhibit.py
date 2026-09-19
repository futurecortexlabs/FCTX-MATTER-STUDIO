"""What a venue needs on top of the simulation.

Three things, none of them physics:

* :class:`HardnessDirector` -- ways to move the dial without a keyboard.
  A visitor at an exhibit has two hands and no mouse, so while one hand
  holds the matter the other hand's height sets the hardness; a venue that
  expects one-handed use can instead have the dial sweep on its own for as
  long as something is held.  Either way the showpiece -- hardness changing
  in the hand -- happens without instructions.
* :class:`Coach` -- the instructions anyway, as short prompts that appear
  when they are needed and go away once the visitor has done the thing.
* :class:`SessionLog` -- a record of visitors, grabs and dial use, because
  the person who paid for the installation will ask how many people used it.

All three are plain Python over the per-frame facts the application already
has (poses, grips, the control state), so they are tested without a GPU.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ExhibitConfig, TrackingConfig
from .core.types import HandPose

__all__ = ["HardnessDirector", "Coach", "Prompt", "SessionLog"]


# ---------------------------------------------------------------------------
# the dial, hands-only
# ---------------------------------------------------------------------------


class HardnessDirector:
    """Set the dial from the hands, when the configuration asks for it.

    Both modes write ``hardness_target`` on the control state and leave the
    smoothing to :class:`fctx.ui.controls.Controls`, so a hand that jumps
    still produces a continuous change in the material.
    """

    #: A free hand has to be this sure before it is allowed to drive.
    MIN_CONFIDENCE = 0.5
    #: Fraction of the stage height at either end that maps to the dial's
    #: end stops, so a visitor does not have to reach the very edge.
    MARGIN = 0.12

    def __init__(self, cfg: ExhibitConfig, tracking: TrackingConfig) -> None:
        self.cfg = cfg
        self.tracking = tracking
        self._phase = 0.0
        self._sweeping = False
        #: What moved the dial this frame: "free_hand", "sweep" or None.
        self.driving: str | None = None

    def _height_to_hardness(self, y: float) -> float:
        lo = self.tracking.stage_center_y - self.tracking.stage_half_height
        span = 2.0 * self.tracking.stage_half_height
        lo += span * self.MARGIN
        span *= 1.0 - 2.0 * self.MARGIN
        return min(max((y - lo) / max(span, 1e-6), 0.0), 1.0)

    def update(self, poses: list[HandPose], holding_ids: set[int],
               state, dt: float) -> None:
        """``holding_ids`` are the track ids currently holding matter."""
        self.driving = None
        cfg = self.cfg
        if not holding_ids:
            self._sweeping = False
            return

        if cfg.hardness_by_free_hand:
            free = [p for p in poses
                    if p.track_id not in holding_ids and not p.pinching
                    and p.confidence >= self.MIN_CONFIDENCE]
            if free:
                # The highest free hand wins, which is also the deliberate one.
                hand = max(free, key=lambda p: float(p.joints[0][1]))
                state.hardness_target = self._height_to_hardness(float(hand.joints[0][1]))
                state.auto_sweep = False
                self.driving = "free_hand"
                self._sweeping = False
                return

        if cfg.sweep_while_holding:
            if not self._sweeping:
                # Start from where the dial is, so the first frame of the
                # sweep does not jump.
                t = min(max(float(state.hardness_target), 0.0), 1.0)
                self._phase = math.asin(2.0 * t - 1.0)
                self._sweeping = True
            self._phase += 2.0 * math.pi * dt / max(cfg.sweep_period, 0.5)
            state.hardness_target = 0.5 + 0.5 * math.sin(self._phase)
            state.auto_sweep = False
            self.driving = "sweep"


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Prompt:
    """A prompt as the renderer sees it: text plus a fade."""

    text: str
    alpha: float = 1.0


class Coach:
    """Tell a visitor the one thing they need next, then get out of the way.

    Stages: nobody -> hand seen (say how to grab) -> holding (say how to
    change hardness) -> hardness changed (say how to let go, once) -> done.
    A prompt stays for ``prompt_seconds``, fades, and comes back after a
    pause if the visitor is still stuck at the same stage.  Everything resets
    when the hands have been gone for ``session_gap`` seconds, so the next
    visitor gets the full sequence.
    """

    FADE = 0.35
    #: Seconds a hand must be present before the first prompt, so a hand
    #: passing through the frame does not flash text at nobody.
    SETTLE = 0.8
    #: Seconds between repeats of the same prompt.
    REPEAT_AFTER = 8.0
    #: Dial travel while holding that counts as "changed the hardness".
    MOVED = 0.15

    def __init__(self, cfg: ExhibitConfig) -> None:
        self.cfg = cfg
        self.stage = "nobody"
        self._since_hand = 0.0
        self._absent_for = 0.0
        self._shown_for = 0.0
        self._hidden_for = 0.0
        self._visible = False
        self._alpha = 0.0
        self._hold_start_hardness: float | None = None
        self._release_told = False

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _text(self, driving: str | None) -> str:
        cfg = self.cfg
        if self.stage == "hand":
            return cfg.prompt_grab
        if self.stage == "holding":
            if driving == "sweep":
                return cfg.prompt_feel
            return cfg.prompt_hardness
        if self.stage == "changed":
            return cfg.prompt_release
        return ""

    def update(self, hands: int, any_held: bool, hardness: float,
               driving: str | None, dt: float) -> Prompt | None:
        cfg = self.cfg
        if not cfg.coach:
            return None

        # -- stage transitions ---------------------------------------------
        if hands > 0:
            self._since_hand += dt
            self._absent_for = 0.0
        else:
            self._since_hand = 0.0
            self._absent_for += dt
            if self._absent_for >= cfg.session_gap and self.stage != "nobody":
                self.reset()
                return None

        previous = self.stage
        if self.stage == "nobody":
            if self._since_hand >= self.SETTLE:
                self.stage = "hand"
        elif self.stage == "hand":
            if any_held:
                self.stage = "holding"
                self._hold_start_hardness = hardness
        elif self.stage == "holding":
            if not any_held:
                self._hold_start_hardness = None
            else:
                if self._hold_start_hardness is None:
                    self._hold_start_hardness = hardness
                moved = abs(hardness - self._hold_start_hardness) >= self.MOVED
                if moved or (driving == "sweep" and self._shown_for >= cfg.prompt_seconds):
                    self.stage = "changed"
        elif self.stage == "changed":
            if not any_held and self._shown_for >= cfg.prompt_seconds:
                self.stage = "done"

        if self.stage != previous:
            self._shown_for = 0.0
            self._hidden_for = 0.0
            self._visible = self.stage in ("hand", "holding", "changed")

        # -- visibility: show, time out, repeat ------------------------------
        text = self._text(driving)
        if not text:
            self._visible = False
        if self._visible:
            self._shown_for += dt
            if self._shown_for >= cfg.prompt_seconds:
                self._visible = False
                self._hidden_for = 0.0
        else:
            self._hidden_for += dt
            if text and self._hidden_for >= self.REPEAT_AFTER and self.stage != "changed":
                self._visible = True
                self._shown_for = 0.0

        target = 1.0 if (self._visible and text) else 0.0
        step = dt / self.FADE
        self._alpha += max(-step, min(step, target - self._alpha))
        if self._alpha <= 0.0 or not text:
            return None
        return Prompt(text, self._alpha)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Visitor:
    started: float
    last_seen: float
    grabs: int = 0
    hold_seconds: float = 0.0
    dial_moves: int = 0
    max_hands: int = 0


@dataclass(slots=True)
class Summary:
    visitors: int = 0
    grabs: int = 0
    hold_seconds: float = 0.0
    dial_moves: int = 0
    interaction_seconds: float = 0.0
    attract_starts: int = 0
    camera_losses: int = 0
    errors: int = 0
    uptime_seconds: float = 0.0
    #: The current visitor, if someone is there.
    open_visitor: bool = False
    counters: dict = field(default_factory=dict)


class SessionLog:
    """Append one CSV row per event; count visitors by gaps between hands.

    A *visitor* is a run of frames with at least one hand, ended by
    ``session_gap`` seconds without any.  That undercounts a group taking
    turns and overcounts one person who steps away and comes back, which is
    what any camera-only count does; the number is honest as a lower bound
    on people and an upper bound on sessions, and the grabs and dial moves
    say how much they actually did.

    ``path`` None keeps the counts in memory only, which is what the tests
    and a developer's run want.
    """

    COLUMNS = ("time", "uptime_s", "event", "detail")
    #: Dial travel that counts as one deliberate move.
    DIAL_MOVE = 0.2

    def __init__(self, path: Path | None, gap: float,
                 wall_clock=time.time, monotonic=time.perf_counter) -> None:
        self.path = Path(path) if path is not None else None
        self.gap = float(gap)
        self._wall = wall_clock
        self._mono = monotonic
        self._started = self._mono()
        self._visitor: _Visitor | None = None
        self._holding_since: dict[int, float] = {}
        self._dial_anchor: float | None = None
        self.summary = Summary()
        self._closed = False
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(self.COLUMNS)
                w.writerow([self._stamp(), "0.0", "start", ""])

    # -- plumbing ---------------------------------------------------------

    def _stamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._wall()))

    def event(self, name: str, detail: str = "", now: float | None = None) -> None:
        now = self._mono() if now is None else now
        c = self.summary.counters
        c[name] = c.get(name, 0) + 1
        if self.path is None:
            return
        try:
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(
                    [self._stamp(), f"{now - self._started:.1f}", name, detail])
        except OSError:
            pass

    # -- per-frame facts ---------------------------------------------------

    def hands(self, count: int, now: float | None = None) -> None:
        now = self._mono() if now is None else now
        v = self._visitor
        if count > 0:
            if v is None:
                self._visitor = v = _Visitor(started=now, last_seen=now)
                self.summary.open_visitor = True
                self.event("visitor_start", "", now)
            v.last_seen = now
            v.max_hands = max(v.max_hands, count)
        elif v is not None and now - v.last_seen >= self.gap:
            self._end_visitor(v.last_seen)

    def _end_visitor(self, now: float) -> None:
        v = self._visitor
        if v is None:
            return
        duration = max(0.0, now - v.started)
        s = self.summary
        s.visitors += 1
        s.interaction_seconds += duration
        s.open_visitor = False
        self.event("visitor_end",
                   f"{duration:.1f}s grabs={v.grabs} hold={v.hold_seconds:.1f}s "
                   f"dial_moves={v.dial_moves} hands={v.max_hands}", now)
        self._visitor = None
        self._holding_since.clear()
        self._dial_anchor = None

    def grab(self, track_id: int, now: float | None = None) -> None:
        now = self._mono() if now is None else now
        self._holding_since[track_id] = now
        self.summary.grabs += 1
        if self._visitor is not None:
            self._visitor.grabs += 1
        self.event("grab", f"hand {track_id}", now)

    def release(self, track_id: int, now: float | None = None) -> None:
        now = self._mono() if now is None else now
        since = self._holding_since.pop(track_id, None)
        held = (now - since) if since is not None else 0.0
        self.summary.hold_seconds += held
        if self._visitor is not None:
            self._visitor.hold_seconds += held
        self.event("release", f"hand {track_id} after {held:.1f}s", now)

    def dial(self, hardness: float, any_held: bool, now: float | None = None) -> None:
        """Count a deliberate dial move while something is held."""
        if not any_held:
            self._dial_anchor = None
            return
        if self._dial_anchor is None:
            self._dial_anchor = hardness
            return
        if abs(hardness - self._dial_anchor) >= self.DIAL_MOVE:
            self._dial_anchor = hardness
            self.summary.dial_moves += 1
            if self._visitor is not None:
                self._visitor.dial_moves += 1
            self.event("dial", f"{hardness:.2f}", now)

    # -- notable moments ---------------------------------------------------

    def attract(self, on: bool) -> None:
        if on:
            self.summary.attract_starts += 1
        self.event("attract_on" if on else "attract_off")

    def camera(self, back: bool, detail: str = "") -> None:
        if not back:
            self.summary.camera_losses += 1
        self.event("camera_back" if back else "camera_lost", detail)

    def error(self, detail: str) -> None:
        self.summary.errors += 1
        self.event("error", detail)

    # -- the end -----------------------------------------------------------

    def close(self, reason: str = "exit") -> Summary:
        if self._closed:
            return self.summary
        self._closed = True
        now = self._mono()
        if self._visitor is not None:
            self._end_visitor(now)
        self.summary.uptime_seconds = now - self._started
        s = self.summary
        self.event("stop", f"{reason} uptime={s.uptime_seconds:.0f}s visitors={s.visitors} "
                           f"grabs={s.grabs} dial_moves={s.dial_moves}", now)
        return s

    def describe(self) -> str:
        s = self.summary
        hours = max(s.uptime_seconds, 1e-9) / 3600.0
        return (f"{s.visitors} visitors in {s.uptime_seconds / 60:.0f} min "
                f"({s.visitors / hours:.1f}/h), {s.grabs} grabs, "
                f"{s.hold_seconds:.0f} s held, {s.dial_moves} dial moves, "
                f"{s.attract_starts} attract cycles, {s.camera_losses} camera losses, "
                f"{s.errors} errors")
