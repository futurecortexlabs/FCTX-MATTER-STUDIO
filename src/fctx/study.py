"""Sensory evaluation without samples: blind A/B tests on simulated materials.

A materials company that wants to know whether customers can tell two foams
apart, or which of three they prefer, normally makes the foams and runs a
panel.  This runs the panel on the simulation instead: two samples side by
side that look identical -- same shape, same colour, the dial and the
material names hidden -- and differ only in how they behave under the hand
(and, with pseudo-haptics on, in how the drawn hand is resisted).  The
visitor explores both, then holds a hand up over the one they choose.  No
keyboard, no mouse, no touching anything.

Two protocols:

``discrimination``
    Two-alternative forced choice, "which is harder?", reference against
    reference + delta, the harder one on a random side.  ``delta`` follows
    an n-down-1-up staircase (2-down converges on 70.7% correct), so the
    trials concentrate where the answer is informative and the reversals
    estimate the just-noticeable difference.  One staircase per pseudo-
    haptics condition, interleaved, so the same run says whether the
    illusion changes the threshold.  ``staircase_scope = "study"`` pools one
    staircase across everybody -- right for an exhibition, where each
    visitor does a handful of trials -- and it is restored from the CSV on
    start, so a restart continues where it left off.

``preference``
    Paired comparison, "which do you prefer?", between named materials from
    the catalogue, every pair in both side orders.  ``tools/analyze_study.py``
    fits Bradley-Terry strengths.

Everything here is plain Python over per-frame facts (poses, which bodies
are being touched) and is tested without a GPU; the application owns the
scene, the materials and the drawing.
"""

from __future__ import annotations

import csv
import dataclasses
import enum
import itertools
import math
import random
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import TrackingConfig
from .core import catalog as catalog_mod
from .core.material import MaterialParams
from .core.types import HandPose, MatterKind
from .haptics import palm_centre

__all__ = ["StudyConfig", "StudyError", "load_study", "Staircase", "Trial", "Panel",
           "Phase", "Study", "StudyLog", "COLUMNS"]


class StudyError(ValueError):
    """A study file said something unusable."""


@dataclass(frozen=True, slots=True)
class StudyConfig:
    name: str = "study"
    protocol: str = "discrimination"          # discrimination | preference
    kind: str = "soft"                        # soft | cloth
    #: Reference hardness on the dial, or a catalogue material to take it from.
    reference: float = 0.5
    reference_material: str = ""
    #: Preference: catalogue material names to compare.
    materials: tuple[str, ...] = ()
    #: Trials per participant.
    trials: int = 10
    # -- staircase (discrimination) -----------------------------------------
    start_delta: float = 0.25
    min_delta: float = 0.01
    max_delta: float = 0.5
    down: int = 2
    step_factor: float = 1.6
    fine_step_factor: float = 1.25
    fine_after_reversals: int = 4
    staircase_scope: str = "study"            # study | participant
    # -- conditions ----------------------------------------------------------
    pseudo_haptics: str = "alternate"         # on | off | alternate
    # -- pacing --------------------------------------------------------------
    intro_seconds: float = 3.0
    explore_min: float = 2.0
    touch_min: float = 0.3
    dwell: float = 1.2
    recorded_seconds: float = 0.9
    session_gap: float = 12.0
    #: A response zone is above this fraction of the stage's half height
    #: over its centre, and beyond ``respond_x`` metres either side.
    respond_height: float = 0.45
    respond_x: float = 0.06
    # -- scene ---------------------------------------------------------------
    spacing: float = 0.34
    sample_size: float = 0.20
    resolution: int = 15
    seed: int = 1
    output: Path = Path("studies/study.csv")
    # -- words ---------------------------------------------------------------
    prompt_welcome: str = "HOLD UP A HAND TO TAKE PART"
    prompt_intro: str = "TWO SAMPLES. PRESS BOTH, THEN CHOOSE."
    prompt_explore: str = "PRESS BOTH SAMPLES"
    prompt_choose_harder: str = "RAISE A HAND OVER THE HARDER ONE"
    prompt_choose_preferred: str = "RAISE A HAND OVER THE ONE YOU PREFER"
    prompt_recorded: str = "THANK YOU"
    prompt_thanks: str = "ALL DONE -- THANK YOU"
    label_choice: str = "THIS ONE"


_CHOICES = {
    "protocol": ("discrimination", "preference"),
    "kind": ("soft", "cloth"),
    "staircase_scope": ("study", "participant"),
    "pseudo_haptics": ("on", "off", "alternate"),
}


def load_study(path: str | Path) -> StudyConfig:
    """Read a study file: a ``[study]`` table of :class:`StudyConfig` fields."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StudyError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise StudyError(f"{path}: {exc}") from exc
    table = data.get("study")
    if not isinstance(table, dict):
        raise StudyError(f"{path}: expected a [study] table")
    extra = set(data) - {"study"}
    if extra:
        raise StudyError(f"{path}: unknown top-level key(s) {', '.join(sorted(extra))}")
    return study_from_dict(table, str(path), base_dir=path.parent)


def study_from_dict(table: dict, where: str = "study",
                    base_dir: Path | None = None) -> StudyConfig:
    fields = {f.name: f for f in dataclasses.fields(StudyConfig)}
    defaults = StudyConfig()
    values: dict[str, object] = {}
    for key, value in table.items():
        if key not in fields:
            near = [n for n in fields if n.startswith(key[:4])]
            hint = f" (did you mean {', '.join(near)}?)" if near else ""
            raise StudyError(f"{where}: unknown key {key!r}{hint}")
        default = getattr(defaults, key)
        here = f"{where}: {key}"
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise StudyError(f"{here} must be true or false")
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise StudyError(f"{here} must be a whole number")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise StudyError(f"{here} must be a number")
            value = float(value)
        elif isinstance(default, tuple):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise StudyError(f"{here} must be a list of names")
            value = tuple(value)
        elif isinstance(default, Path):
            if not isinstance(value, str) or not value:
                raise StudyError(f"{here} must be a path")
            value = Path(value)
            if base_dir is not None and not value.is_absolute():
                value = base_dir / value
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise StudyError(f"{here} must be a string")
            if key in _CHOICES and value not in _CHOICES[key]:
                raise StudyError(f"{here} {value!r} is not one of {', '.join(_CHOICES[key])}")
        values[key] = value
    cfg = dataclasses.replace(defaults, **values)
    _validate(cfg, where)
    return cfg


def _validate(cfg: StudyConfig, where: str) -> None:
    if not cfg.name.strip() or any(c in cfg.name for c in ",\n\r"):
        raise StudyError(f"{where}: name must be non-empty and contain no commas")
    if cfg.trials < 1:
        raise StudyError(f"{where}: trials must be at least 1")
    if not 0.0 <= cfg.reference <= 1.0:
        raise StudyError(f"{where}: reference must be on the dial, 0..1")
    if not 0.0 < cfg.min_delta <= cfg.start_delta <= cfg.max_delta <= 1.0:
        raise StudyError(f"{where}: need 0 < min_delta <= start_delta <= max_delta <= 1")
    if cfg.down < 1 or cfg.step_factor <= 1.0 or cfg.fine_step_factor <= 1.0:
        raise StudyError(f"{where}: down >= 1 and step factors > 1")
    if cfg.protocol == "preference" and len(cfg.materials) < 2:
        raise StudyError(f"{where}: a preference study needs at least two materials")
    if cfg.dwell <= 0.0 or cfg.session_gap <= 0.0:
        raise StudyError(f"{where}: dwell and session_gap must be positive")


# ---------------------------------------------------------------------------
# staircase
# ---------------------------------------------------------------------------


class Staircase:
    """n-down-1-up on a multiplicative step, with reversals recorded.

    ``delta`` shrinks by the step factor after ``down`` correct answers in a
    row and grows by it after any wrong one; after ``fine_after`` reversals
    the factor drops to the fine one.  2-down-1-up converges on the delta
    answered correctly 70.7% of the time (Levitt, 1971).
    """

    def __init__(self, start: float, lo: float, hi: float, down: int = 2,
                 factor: float = 1.6, fine_factor: float = 1.25,
                 fine_after: int = 4) -> None:
        self.delta = float(start)
        self.lo, self.hi = float(lo), float(hi)
        self.down = int(down)
        self.factor, self.fine_factor = float(factor), float(fine_factor)
        self.fine_after = int(fine_after)
        self.run = 0
        self.direction = 0
        self.reversals: list[float] = []
        self.trials = 0
        self.correct = 0

    @classmethod
    def from_config(cls, cfg: StudyConfig) -> Staircase:
        return cls(cfg.start_delta, cfg.min_delta, cfg.max_delta, cfg.down,
                   cfg.step_factor, cfg.fine_step_factor, cfg.fine_after_reversals)

    def record(self, correct: bool) -> None:
        self.trials += 1
        if correct:
            self.correct += 1
            self.run += 1
            if self.run >= self.down:
                self.run = 0
                self._move(-1)
        else:
            self.run = 0
            self._move(+1)

    def _move(self, direction: int) -> None:
        if self.direction and direction != self.direction:
            self.reversals.append(self.delta)
        self.direction = direction
        f = self.fine_factor if len(self.reversals) >= self.fine_after else self.factor
        d = self.delta / f if direction < 0 else self.delta * f
        self.delta = min(max(d, self.lo), self.hi)

    def threshold(self, last: int = 6) -> float | None:
        """Geometric mean of the last ``last`` reversals (None before two)."""
        rev = self.reversals[-last:]
        if len(rev) < 2:
            return None
        return float(math.exp(sum(math.log(r) for r in rev) / len(rev)))


# ---------------------------------------------------------------------------
# trials and the record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trial:
    index: int
    condition: str                      # "on" | "off": pseudo-haptics
    hardness: tuple[float, float]       # left, right
    labels: tuple[str, str]
    #: Discrimination: the side of the harder sample.
    correct_side: int | None = None
    delta: float | None = None


@dataclass(slots=True)
class Panel:
    """A screen-space response target, for the renderer."""

    text: str
    x: float                    # centre, fraction of the width
    y: float                    # top, fraction of the height
    w: float                    # width, fraction of the width
    progress: float = 0.0
    active: bool = False


class Phase(enum.Enum):
    WAITING = "waiting"
    INTRO = "intro"
    EXPLORE = "explore"
    RESPOND = "respond"
    RECORDED = "recorded"
    DONE = "done"


COLUMNS = ("time", "study", "protocol", "participant", "trial", "condition",
           "left_hardness", "right_hardness", "left_label", "right_label",
           "reference", "delta", "correct_side", "response_side", "correct",
           "chosen", "rt_s", "explore_s", "touch_left_s", "touch_right_s",
           "response_mode")


class StudyLog:
    """One CSV row per answered trial; ``path`` None keeps rows in memory."""

    def __init__(self, path: Path | None, wall_clock=time.time) -> None:
        self.path = Path(path) if path is not None else None
        self._wall = wall_clock
        self.rows: list[dict[str, str]] = []
        if self.path is not None and self.path.exists() and self.path.stat().st_size:
            with self.path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                missing = set(COLUMNS) - set(reader.fieldnames or ())
                if missing:
                    raise StudyError(f"{self.path}: not a study log (missing "
                                     f"{', '.join(sorted(missing))})")
                self.rows = list(reader)

    def previous(self, study: str) -> list[dict[str, str]]:
        return [r for r in self.rows if r.get("study") == study]

    def write(self, row: dict[str, object]) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._wall()))
        out = {k: "" for k in COLUMNS}
        out["time"] = stamp
        for k, v in row.items():
            if v is None:
                continue
            out[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
        self.rows.append(out)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow(out)


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Timers:
    phase: float = 0.0
    explore: float = 0.0
    respond: float = 0.0
    touch: list = field(default_factory=lambda: [0.0, 0.0])
    absent: float = 0.0
    present: float = 0.0
    dwell: float = 0.0
    dwell_side: int = -1


class Study:
    """Run a study, one participant after another, frame by frame.

    Per frame the application calls :meth:`update` with the raw hand poses
    and the set of sample indices being touched, and reads back
    :attr:`hardness` (per sample), :attr:`condition`, :attr:`prompt` and
    :attr:`panels`.  When :attr:`reset_scene` is set it puts the samples
    back to rest and clears it: every trial starts from undisturbed matter.
    """

    SIDES = ("left", "right")
    #: Seconds a hand must be present before it counts as a participant.
    ARRIVE = 0.5
    #: Seconds without hands after the last trial before the next person.
    LEAVE = 2.0

    def __init__(self, cfg: StudyConfig, params: MaterialParams,
                 tracking: TrackingConfig,
                 entries: tuple[catalog_mod.CatalogEntry, ...] = catalog_mod.DEFAULT_CATALOG,
                 log: StudyLog | None = None) -> None:
        self.cfg = cfg
        self.params = params
        self.tracking = tracking
        self.log = log if log is not None else StudyLog(None)
        kind = MatterKind[cfg.kind.upper()]
        if params.kind is not kind:
            raise StudyError(f"study kind {cfg.kind} but material {params.kind.name.lower()}")

        self.reference = cfg.reference
        if cfg.reference_material:
            entry = self._entry(entries, cfg.reference_material)
            self.reference = catalog_mod.hardness_for(params, entry.value)
        self.options: list[tuple[str, float]] = []
        if cfg.protocol == "preference":
            for name in cfg.materials:
                entry = self._entry(entries, name)
                self.options.append((entry.name, catalog_mod.hardness_for(params, entry.value)))

        self.staircases: dict[str, Staircase] = {}
        self._participant_stairs: dict[str, Staircase] = {}
        previous = self.log.previous(cfg.name)
        self.participant = max((int(r["participant"]) for r in previous
                                if r.get("participant", "").isdigit()), default=0)
        if cfg.protocol == "discrimination" and cfg.staircase_scope == "study":
            for r in previous:
                if r.get("correct") in ("0", "1"):
                    self._stair(r.get("condition", "on")).record(r["correct"] == "1")

        self.phase = Phase.WAITING
        self.trial: Trial | None = None
        self.trial_count = 0
        self.hardness = [self.reference, self.reference]
        self.condition = "on" if cfg.pseudo_haptics != "off" else "off"
        self.prompt: str | None = cfg.prompt_welcome
        self.panels: list[Panel] = []
        self.reset_scene = False
        self.last_choice: int | None = None
        self._t = _Timers()
        self._rng = random.Random(cfg.seed)
        self._first_condition = "on"
        self._pairs: list[tuple[int, int]] = []

    @staticmethod
    def _entry(entries, name: str) -> catalog_mod.CatalogEntry:
        for e in entries:
            if e.name.lower() == name.lower():
                return e
        raise StudyError(f"no catalogue material named {name!r}")

    def _stair(self, condition: str) -> Staircase:
        table = (self.staircases if self.cfg.staircase_scope == "study"
                 else self._participant_stairs)
        if condition not in table:
            table[condition] = Staircase.from_config(self.cfg)
        return table[condition]

    # -- trial generation -------------------------------------------------

    def _condition_for(self, index: int) -> str:
        mode = self.cfg.pseudo_haptics
        if mode != "alternate":
            return mode
        other = "off" if self._first_condition == "on" else "on"
        return self._first_condition if index % 2 == 0 else other

    def _next_trial(self) -> Trial:
        cfg = self.cfg
        index = self.trial_count
        condition = self._condition_for(index)
        if cfg.protocol == "discrimination":
            ref = self.reference
            delta = self._stair(condition).delta
            harder = min(ref + delta, 1.0)
            delta = harder - ref
            if delta <= 0.0:      # reference at the top of the dial: go down instead
                ref, harder = max(self.reference - self._stair(condition).delta, 0.0), ref
                delta = harder - ref
            side = self._rng.randrange(2)
            h = [ref, ref]
            h[side] = harder
            return Trial(index, condition, (h[0], h[1]), ("A", "B"), side, delta)
        a, b = self._pairs[index % len(self._pairs)]
        (na, ha), (nb, hb) = self.options[a], self.options[b]
        return Trial(index, condition, (ha, hb), (na, nb))

    def _begin_participant(self) -> None:
        self.participant += 1
        self.trial_count = 0
        self._participant_stairs = {}
        self._rng = random.Random(f"{self.cfg.seed}:{self.participant}")
        self._first_condition = self._rng.choice(("on", "off"))
        if self.cfg.protocol == "preference":
            pairs = []
            for a, b in itertools.combinations(range(len(self.options)), 2):
                pairs += [(a, b), (b, a)]
            self._rng.shuffle(pairs)
            self._pairs = pairs

    def _start_trial(self) -> None:
        self.trial = self._next_trial()
        self.hardness = list(self.trial.hardness)
        self.condition = self.trial.condition
        self.reset_scene = True
        self._t.explore = 0.0
        self._t.respond = 0.0
        self._t.touch = [0.0, 0.0]
        self._t.dwell = 0.0
        self._t.dwell_side = -1
        self._enter(Phase.EXPLORE)

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._t.phase = 0.0

    # -- per frame ----------------------------------------------------------

    def _zone(self, pose: HandPose) -> int:
        if pose.pinching:
            return -1
        p = palm_centre(pose)
        top = self.tracking.stage_center_y + self.cfg.respond_height * self.tracking.stage_half_height
        if p[1] < top:
            return -1
        if p[0] < -self.cfg.respond_x:
            return 0
        if p[0] > self.cfg.respond_x:
            return 1
        return -1

    def update(self, poses: list[HandPose], touching: set[int], dt: float,
               key: int | None = None) -> None:
        cfg = self.cfg
        t = self._t
        t.phase += dt
        hands = len(poses)
        if hands:
            t.present += dt
            t.absent = 0.0
        else:
            t.present = 0.0
            t.absent += dt

        if self.phase is Phase.WAITING:
            self.panels = []
            self.prompt = cfg.prompt_welcome
            if t.present >= self.ARRIVE or key is not None:
                self._begin_participant()
                self._enter(Phase.INTRO)
            return

        if self.phase is Phase.DONE:
            self.panels = []
            self.prompt = cfg.prompt_thanks
            if t.absent >= self.LEAVE:
                self._to_waiting()
            return

        if t.absent >= cfg.session_gap:
            self._to_waiting()      # walked away mid-study: the rows so far stand
            return

        if self.phase is Phase.INTRO:
            self.prompt = cfg.prompt_intro
            self.panels = []
            if t.phase >= cfg.intro_seconds:
                self._start_trial()
            return

        if self.phase is Phase.RECORDED:
            self.prompt = cfg.prompt_recorded
            self.panels = []
            if t.phase >= cfg.recorded_seconds:
                if self.trial_count >= cfg.trials:
                    self._enter(Phase.DONE)
                    self.trial = None
                else:
                    self._start_trial()
            return

        # EXPLORE and RESPOND: exploring continues until an answer lands.
        t.explore += dt
        for i in touching:
            if 0 <= i < 2:
                t.touch[i] += dt
        explored = (t.explore >= cfg.explore_min
                    and min(t.touch) >= cfg.touch_min)
        if self.phase is Phase.EXPLORE:
            self.prompt = cfg.prompt_explore
            self.panels = []
            if explored:
                self._enter(Phase.RESPOND)
            elif key is None:
                return
        # RESPOND (or a staff key during EXPLORE, which answers at once).
        t.respond += dt if self.phase is Phase.RESPOND else 0.0
        self.prompt = (cfg.prompt_choose_harder if cfg.protocol == "discrimination"
                       else cfg.prompt_choose_preferred)
        side = -1
        for pose in poses:
            z = self._zone(pose)
            if z >= 0:
                side = z
                break
        if side < 0 or side != t.dwell_side:
            t.dwell = 0.0
        else:
            t.dwell += dt
        t.dwell_side = side
        progress = [0.0, 0.0]
        if side >= 0:
            progress[side] = min(t.dwell / cfg.dwell, 1.0)
        self.panels = [Panel(cfg.label_choice, 0.25, 0.14, 0.26, progress[0], side == 0),
                       Panel(cfg.label_choice, 0.75, 0.14, 0.26, progress[1], side == 1)]
        if key in (0, 1):
            self._answer(int(key), "key")
        elif side >= 0 and t.dwell >= cfg.dwell:
            self._answer(side, "hand")

    def _answer(self, side: int, mode: str) -> None:
        cfg = self.cfg
        trial = self.trial
        assert trial is not None
        t = self._t
        correct = None
        if trial.correct_side is not None:
            correct = side == trial.correct_side
            self._stair(trial.condition).record(correct)
        self.last_choice = side
        self.log.write({
            "study": cfg.name, "protocol": cfg.protocol, "participant": self.participant,
            "trial": trial.index + 1, "condition": trial.condition,
            "left_hardness": trial.hardness[0], "right_hardness": trial.hardness[1],
            "left_label": trial.labels[0], "right_label": trial.labels[1],
            "reference": self.reference if cfg.protocol == "discrimination" else None,
            "delta": trial.delta,
            "correct_side": self.SIDES[trial.correct_side] if trial.correct_side is not None
            else None,
            "response_side": self.SIDES[side],
            "correct": None if correct is None else int(correct),
            "chosen": trial.labels[side],
            "rt_s": t.respond, "explore_s": t.explore,
            "touch_left_s": t.touch[0], "touch_right_s": t.touch[1],
            "response_mode": mode,
        })
        self.trial_count += 1
        self.panels = []
        self._enter(Phase.RECORDED)

    def _to_waiting(self) -> None:
        self.trial = None
        self.hardness = [self.reference, self.reference]
        self.condition = "on" if self.cfg.pseudo_haptics != "off" else "off"
        self.panels = []
        self.prompt = self.cfg.prompt_welcome
        self.reset_scene = True
        self._t = _Timers()
        self._enter(Phase.WAITING)

    # -- reporting ------------------------------------------------------------

    def status(self) -> str:
        if self.phase in (Phase.WAITING, Phase.DONE, Phase.INTRO):
            return f"{self.cfg.name}: participant {self.participant}, {self.phase.value}"
        parts = [f"{self.cfg.name}: P{self.participant} trial {self.trial_count + 1}/"
                 f"{self.cfg.trials} [{self.condition}]"]
        for cond, st in sorted(self.staircases.items()):
            thr = st.threshold()
            parts.append(f"{cond}: delta {st.delta:.3f}, {len(st.reversals)} rev"
                         + (f", JND~{thr:.3f}" if thr else ""))
        return "  ".join(parts)
