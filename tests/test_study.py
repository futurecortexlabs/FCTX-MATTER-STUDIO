"""Sensory evaluation: the study state machine, the staircase, the log, the analysis.

A simulated observer with a known threshold answers the study through the
same per-frame interface the application uses; the staircase and the
analysis tool must recover that threshold.  No GPU.
"""

from __future__ import annotations

import csv
import math
import random
import tempfile
from pathlib import Path

import numpy as np
from _harness import case, note, require, run

from fctx import analysis as analyze_study  # noqa: E402
from fctx.config import TrackingConfig
from fctx.core.material import DEFAULT_MATERIALS
from fctx.core.types import HandPose, MatterKind
from fctx.study import (
    COLUMNS,
    Phase,
    Staircase,
    Study,
    StudyConfig,
    StudyError,
    StudyLog,
    load_study,
    study_from_dict,
)

SOFT = DEFAULT_MATERIALS[MatterKind.SOFT]
TRACK = TrackingConfig()
DT = 1 / 30


def pose(x: float, y: float, track_id: int = 0, pinching: bool = False) -> HandPose:
    j = np.zeros((21, 3), np.float32)
    j[:, 0] = x
    j[:, 1] = y
    return HandPose(joints=j, velocities=np.zeros((21, 3), np.float32),
                    track_id=track_id, pinching=pinching)


LOW = pose(0.0, 0.15)                                   # down among the samples
HIGH_Y = TRACK.stage_center_y + 0.8 * TRACK.stage_half_height


def observer(threshold: float, slope: float = 0.35, seed: int = 0):
    """P(correct) = 0.5 + 0.5 * logistic((ln d - ln thr) / slope)."""
    rng = random.Random(seed)

    def answer(trial) -> int:
        p = 0.5 + 0.5 / (1 + math.exp(-(math.log(max(trial.delta, 1e-6))
                                         - math.log(threshold)) / slope))
        right = rng.random() < p
        return trial.correct_side if right else 1 - trial.correct_side
    return answer


def drive(study: Study, answer, participants: int, by_hand: bool = False) -> None:
    """Run ``participants`` people through the study frame by frame."""
    for _ in range(participants):
        for _ in range(200 * (study.cfg.trials + 2)):   # arrive, intro, trials, done
            if study.phase is Phase.DONE:
                break
            touching: set[int] = set()
            key = None
            poses = [LOW]
            if study.phase is Phase.EXPLORE:
                touching = {0, 1}
            elif study.phase is Phase.RESPOND:
                side = answer(study.trial)
                if by_hand:
                    poses = [pose(-0.2 if side == 0 else 0.2, HIGH_Y)]
                else:
                    key = side
            study.update(poses, touching, DT, key)
        require(study.phase is Phase.DONE, f"participant stuck in {study.phase}")
        for _ in range(int(Study.LEAVE / DT) + 3):      # walks away
            study.update([], set(), DT)
        require(study.phase is Phase.WAITING, "the next participant was not awaited")


# -- configuration ------------------------------------------------------------


@case
def the_shipped_example_studies_load() -> None:
    root = Path(__file__).resolve().parents[1] / "studies"
    files = sorted(root.glob("*.toml"))
    require(len(files) >= 2, f"expected example studies in {root}")
    for f in files:
        cfg = load_study(f)
        Study(cfg, SOFT, TRACK)                     # names resolve against the catalogue
        require(cfg.output.is_absolute() or cfg.output.parts[0] != "results",
                "output was not made relative to the study file")
    note(", ".join(f.name for f in files))


@case
def a_bad_study_file_says_what_is_wrong() -> None:
    for table, fragment in (
            ({"protocl": "discrimination"}, "protocl"),
            ({"protocol": "triangle"}, "triangle"),
            ({"trials": 0}, "trials"),
            ({"reference": 1.5}, "reference"),
            ({"min_delta": 0.3, "start_delta": 0.2}, "min_delta"),
            ({"protocol": "preference", "materials": ["DENIM"]}, "two materials"),
            ({"materials": [1, 2]}, "materials"),
            ({"name": "a,b"}, "name"),
            ({"pseudo_haptics": "sometimes"}, "sometimes"),
    ):
        try:
            study_from_dict(table)
        except StudyError as exc:
            require(fragment in str(exc), f"{table}: error lacks {fragment!r}: {exc}")
            continue
        raise AssertionError(f"accepted {table}")
    try:
        Study(StudyConfig(protocol="preference", materials=("DENIM", "NOPE")), SOFT, TRACK)
    except StudyError as exc:
        require("NOPE" in str(exc))
    else:
        raise AssertionError("an unknown material was accepted")


# -- staircase ----------------------------------------------------------------


@case
def the_staircase_moves_two_down_one_up_and_records_reversals() -> None:
    s = Staircase(0.2, 0.01, 0.5, down=2, factor=2.0, fine_factor=1.5, fine_after=2)
    s.record(True)
    require(s.delta == 0.2, "moved after one correct")
    s.record(True)
    require(abs(s.delta - 0.1) < 1e-12, "did not halve after two correct")
    s.record(False)
    require(abs(s.delta - 0.2) < 1e-12 and s.reversals == [0.1], f"{s.delta} {s.reversals}")
    s.record(True)
    s.record(True)
    require(s.reversals == [0.1, 0.2], f"{s.reversals}")
    require(abs(s.delta - 0.2 / 1.5) < 1e-12, "the fine factor did not take over")
    for _ in range(40):
        s.record(True)
    require(s.delta == 0.01, "the floor did not hold")


@case
def the_study_recovers_a_known_threshold() -> None:
    true_thr = 0.08
    cfg = StudyConfig(trials=10, pseudo_haptics="on", seed=3)
    study = Study(cfg, SOFT, TRACK)
    drive(study, observer(true_thr, seed=11), participants=24)
    rows = study.log.rows
    require(len(rows) == 240, f"expected 240 rows, got {len(rows)}")
    stair = study.staircases["on"].threshold(8)
    rng = np.random.default_rng(0)
    res = analyze_study.discrimination(rows, rng)["on"]
    note(f"true 0.080; staircase {stair:.3f}; fit {res['threshold']:.3f} "
         f"[{res['ci'][0]:.3f}, {res['ci'][1]:.3f}]; "
         f"modulus Weber {res['weber_E'] * 100:.0f}%")
    # 2-down-1-up targets 70.7%, the fit reports 75%: both near the truth.
    require(0.5 * true_thr < stair < 1.8 * true_thr, f"staircase {stair}")
    require(0.6 * true_thr < res["threshold"] < 1.6 * true_thr, f"fit {res['threshold']}")
    require(res["ci"][0] < true_thr * 1.3 and res["ci"][1] > true_thr * 0.7,
            "the interval excludes the truth by a wide margin")


@case
def interleaved_conditions_get_separate_staircases_and_the_effect_is_measured() -> None:
    cfg = StudyConfig(trials=12, pseudo_haptics="alternate", seed=5)
    study = Study(cfg, SOFT, TRACK)
    thr = {"on": 0.05, "off": 0.15}
    rng = random.Random(1)

    def answer(trial) -> int:
        t = thr[trial.condition]
        p = 0.5 + 0.5 / (1 + math.exp(-(math.log(trial.delta) - math.log(t)) / 0.3))
        return trial.correct_side if rng.random() < p else 1 - trial.correct_side
    drive(study, answer, participants=30)
    conds = [r["condition"] for r in study.log.rows]
    require(abs(conds.count("on") - conds.count("off")) <= 0,
            f"conditions unbalanced: {conds.count('on')} on, {conds.count('off')} off")
    by_p: dict[str, list[str]] = {}
    for r in study.log.rows:
        by_p.setdefault(r["participant"], []).append(r["condition"])
    require(all(all(a != b for a, b in zip(c, c[1:])) for c in by_p.values()),
            "conditions did not alternate within a participant")
    firsts = {c[0] for c in by_p.values()}
    require(firsts == {"on", "off"}, "the first condition was not counterbalanced")
    res = analyze_study.discrimination(study.log.rows, np.random.default_rng(1))
    e = res["effect"]
    note(f"on {res['on']['threshold']:.3f}, off {res['off']['threshold']:.3f}, "
         f"off/on {e['ratio']:.2f} [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]")
    require(e["ratio"] > 1.8 and e["ci"][0] > 1.0,
            "a threefold difference between conditions was not detected")


@case
def a_restart_continues_the_staircase_and_the_participant_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.csv"
        cfg = StudyConfig(trials=8, pseudo_haptics="on", output=path, seed=2)
        a = Study(cfg, SOFT, TRACK, log=StudyLog(path))
        drive(a, observer(0.1, seed=4), participants=5)
        b = Study(cfg, SOFT, TRACK, log=StudyLog(path))
        require(b.participant == 5, f"participant counter restarted at {b.participant}")
        sa, sb = a.staircases["on"], b.staircases["on"]
        require(abs(sa.delta - sb.delta) < 1e-12 and sa.reversals == sb.reversals,
                "the replayed staircase differs from the live one")
        with path.open(encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        require(tuple(header) == COLUMNS, "the CSV header is not the documented one")
        other = Study(StudyConfig(name="other", output=path), SOFT, TRACK, log=StudyLog(path))
        require(other.participant == 0 and not other.staircases.get("on", Staircase(1, 0, 1)).reversals,
                "another study picked up this one's rows")


# -- the session ----------------------------------------------------------------


@case
def answers_by_hand_need_a_dwell_above_the_chosen_side() -> None:
    cfg = StudyConfig(trials=1, pseudo_haptics="on", dwell=1.0, explore_min=0.5,
                      touch_min=0.2, intro_seconds=0.2)
    s = Study(cfg, SOFT, TRACK)
    for _ in range(40):
        s.update([LOW], {0, 1} if s.phase is Phase.EXPLORE else set(), DT)
        if s.phase is Phase.RESPOND:
            break
    require(s.phase is Phase.RESPOND, f"never reached the response: {s.phase}")
    require(len(s.panels) == 2 and not any(p.active for p in s.panels))
    # Low hands, pinching hands and the middle do not count.
    for p in (LOW, pose(-0.2, HIGH_Y, pinching=True), pose(0.0, HIGH_Y)):
        for _ in range(60):
            s.update([p], set(), DT)
        require(s.phase is Phase.RESPOND, f"{p.joints[0]} answered")
    # Half a dwell on the left, then switching right starts over.
    for _ in range(15):
        s.update([pose(-0.2, HIGH_Y)], set(), DT)
    require(s.panels[0].active and 0.4 < s.panels[0].progress < 0.6)
    for _ in range(20):
        s.update([pose(0.2, HIGH_Y)], set(), DT)
    require(s.phase is Phase.RESPOND and s.panels[1].progress < 0.75, "switching kept the dwell")
    for _ in range(15):
        s.update([pose(0.2, HIGH_Y)], set(), DT)
    require(s.phase is Phase.RECORDED, f"a full dwell did not answer: {s.phase}")
    row = s.log.rows[-1]
    require(row["response_side"] == "right" and row["response_mode"] == "hand")
    note(f"answered right after {float(row['rt_s']):.2f} s in the response phase")


@case
def no_answer_before_both_samples_were_explored() -> None:
    cfg = StudyConfig(trials=1, explore_min=1.0, touch_min=0.5, intro_seconds=0.1)
    s = Study(cfg, SOFT, TRACK)
    for _ in range(90):
        s.update([pose(-0.2, HIGH_Y)], {0}, DT)      # only ever touches the left one
    require(s.phase is Phase.EXPLORE, f"answered without touching both: {s.phase}")
    for _ in range(20):
        s.update([LOW], {1}, DT)
    require(s.phase is Phase.RESPOND, "exploring both did not open the response")


@case
def a_participant_who_walks_away_ends_their_session() -> None:
    cfg = StudyConfig(trials=5, session_gap=1.0, intro_seconds=0.1, explore_min=0.1,
                      touch_min=0.05)
    s = Study(cfg, SOFT, TRACK)
    for _ in range(30):
        s.update([LOW], {0, 1}, DT, 0 if s.phase is Phase.RESPOND else None)
    answered = len(s.log.rows)
    require(answered >= 1, "no trial was answered")
    for _ in range(40):
        s.update([], set(), DT)
    require(s.phase is Phase.WAITING and s.reset_scene, "an abandoned session did not end")
    for _ in range(30):
        s.update([LOW], set(), DT)
    require(s.participant == 2, "the next person was not a new participant")
    require(len(s.log.rows) == answered, "walking away wrote rows")


@case
def the_samples_differ_only_in_hardness_and_the_harder_side_is_random() -> None:
    cfg = StudyConfig(trials=40, pseudo_haptics="on", reference=0.4, seed=9)
    s = Study(cfg, SOFT, TRACK)
    drive(s, observer(0.05, seed=2), participants=1)
    sides = [r["correct_side"] for r in s.log.rows]
    require(12 <= sides.count("left") <= 28, f"side not random: {sides.count('left')}/40 left")
    for r in s.log.rows:
        hl, hr = float(r["left_hardness"]), float(r["right_hardness"])
        ref = min(hl, hr)
        require(abs(ref - 0.4) < 1e-3, f"the reference moved: {r}")
        require(abs(abs(hr - hl) - float(r["delta"])) < 1e-3, f"delta mismatch: {r}")


@case
def a_reference_at_the_top_of_the_dial_compares_downward() -> None:
    s = Study(StudyConfig(reference=1.0, trials=2), SOFT, TRACK)
    drive(s, observer(0.05), participants=1)
    for r in s.log.rows:
        require(max(float(r["left_hardness"]), float(r["right_hardness"])) <= 1.0 + 1e-9)
        require(float(r["delta"]) > 0.0)


@case
def preference_runs_every_pair_both_ways_and_bradley_terry_orders_them() -> None:
    names = ("SOFT SILICONE GEL", "SOFT TISSUE", "MEMORY FOAM", "SILICONE RUBBER")
    cfg = StudyConfig(protocol="preference", materials=names, trials=12, pseudo_haptics="on")
    s = Study(cfg, SOFT, TRACK)
    liking = {n: w for n, w in zip(names, (0.3, 1.0, 3.0, 0.6))}
    rng = random.Random(7)

    def answer(trial) -> int:
        a, b = (liking[n] for n in trial.labels)
        return 0 if rng.random() < a / (a + b) else 1
    drive(s, answer, participants=40)
    pairs = {}
    for r in s.log.rows[:12]:
        pairs[(r["left_label"], r["right_label"])] = pairs.get((r["left_label"], r["right_label"]), 0) + 1
    require(len(pairs) == 12 and set(pairs.values()) == {1},
            "one participant did not see all 12 ordered pairs once")
    res = analyze_study.preference(s.log.rows)
    order = [res["names"][i] for i in np.argsort(-res["strength"])]
    note("recovered order: " + " > ".join(order))
    require(order[0] == "MEMORY FOAM" and order[-1] == "SOFT SILICONE GEL",
            f"Bradley-Terry order wrong: {order}")


@case
def the_analysis_tool_runs_on_a_log_and_writes_a_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.csv"
        cfg = StudyConfig(trials=10, output=path)
        s = Study(cfg, SOFT, TRACK, log=StudyLog(path))
        drive(s, observer(0.1, seed=3), participants=6)
        out = Path(tmp) / "summary.csv"
        require(analyze_study.main([str(path), "--csv", str(out)]) == 0)
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        require({r["condition_or_material"] for r in rows} >= {"on", "off"},
                f"summary rows: {rows}")
        require(analyze_study.main([str(Path(tmp) / "missing.csv")]) == 2)


if __name__ == "__main__":
    raise SystemExit(run(__file__))
