"""The wiring the application layer owns: slots, the clock, and configuration.

None of this needs a GPU.  All of it is the kind of thing that only goes
wrong once two subsystems have to agree -- which hand is in which slot, which
half of a preset survives a runtime switch -- so it is exactly what a unit
test of either subsystem alone cannot see.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from _harness import case, note, require, run  # noqa: E402

from fctx.__main__ import build_parser, config_from_args  # noqa: E402
from fctx.config import (  # noqa: E402
    PRESETS,
    RECORDING_DIR,
    GrabConfig,
    preset,
    switch_preset,
)
from fctx.core.clock import FixedTimestep  # noqa: E402
from fctx.core.types import HandPose  # noqa: E402
from fctx.interaction import GripManager  # noqa: E402
from fctx.ui.controls import Controls, ControlState, EventKind, InputEvent  # noqa: E402


def _pose(track_id: int, x: float, pinch: float) -> HandPose:
    return HandPose(
        joints=np.zeros((21, 3), np.float32),
        velocities=np.zeros((21, 3), np.float32),
        track_id=track_id, pinch=pinch, confidence=0.9,
        pinch_point=np.array([x, 0.3, 0.0], np.float32),
        pinch_velocity=np.zeros(3, np.float32))


class _SolverStub:
    """XPBDSolver's slot bookkeeping, as far as GripManager can see it.

    The defect this file guards against is the two modules disagreeing about
    what a slot is, so this stub has to mirror the real solver rather than
    some simpler idea of it: ``slot_map`` is the authority and is keyed on
    ``track_id``, ``set_hands`` releases every slot that map does not name,
    and ``end_grab`` tolerates ``pose=None`` because a vanished hand has no
    pose left to hand over.
    """

    def __init__(self, slots: int = 2) -> None:
        self.counts = [0] * slots

    def slot_map(self, poses: list[HandPose]) -> dict[int, HandPose]:
        out: dict[int, HandPose] = {}
        spare: list[HandPose] = []
        for pose in poses[: len(self.counts)]:
            slot = int(pose.track_id)
            if 0 <= slot < len(self.counts) and slot not in out:
                out[slot] = pose
            else:
                spare.append(pose)
        free = (s for s in range(len(self.counts)) if s not in out)
        for pose, slot in zip(spare, free):
            out[slot] = pose
        return out

    def set_hands(self, poses: list[HandPose], dt: float) -> None:
        present = self.slot_map(poses)
        for slot in range(len(self.counts)):
            if slot not in present:
                self.counts[slot] = 0

    def begin_grab(self, slot: int, pose: HandPose) -> int:
        if self.counts[slot]:
            return self.counts[slot]
        self.counts[slot] = 40 + slot
        return self.counts[slot]

    def end_grab(self, slot: int, pose: HandPose | None) -> None:
        if not self.counts[slot]:
            return
        if pose is not None:
            _ = np.asarray(pose.pinch_velocity, np.float32)
        self.counts[slot] = 0


def _drive(grips: GripManager, solver: _SolverStub,
           poses: list[HandPose], frames: int, dt: float = 1.0 / 90.0) -> None:
    """One frame, in the order fctx.app runs it."""
    for _ in range(frames):
        solver.set_hands(poses, dt)
        grips.update(poses, dt, solver)


# -- hand slots ----------------------------------------------------------


@case
def a_grip_slot_means_the_same_thing_to_the_grip_manager_and_the_solver() -> None:
    # HandTracker compacts its output, so the surviving hand's track id and
    # its position in the list stop agreeing the moment a lower hand leaves.
    # Either rule works on its own; what does not work is the two modules
    # picking different ones, and each of them was independently "fixed" to
    # the opposite rule once.  The solver owns the mapping now and this side
    # asks for it.
    grips, solver = GripManager(GrabConfig(), 2), _SolverStub(2)
    _drive(grips, solver, [_pose(0, -0.1, 0.0), _pose(1, 0.1, 0.9)], 10)
    require(solver.counts[1] > 0, "the second hand never took hold")
    require(grips.total_held == solver.counts[1],
            f"HUD says {grips.total_held}, solver holds {solver.counts}")

    _drive(grips, solver, [_pose(1, 0.1, 0.9)], 10)
    require(grips.total_held == sum(solver.counts),
            f"after the other hand left, HUD says {grips.total_held} but the "
            f"solver holds {solver.counts}")
    require(grips.any_held, "the surviving hand lost its grip for good")
    note(f"surviving hand holds {grips.total_held} particles in solver slot "
         f"{[i for i, c in enumerate(solver.counts) if c][0]}")


@case
def a_hand_leaving_mid_grab_does_not_release_the_wrong_slot() -> None:
    # The crash: hand 0 holds, hand 0 leaves, hand 1 stays.  The tracker's
    # list then holds only track id 1, and whichever module is wrong about
    # the slot releases one the other has not, with a None pose, straight
    # into end_grab's pose.pinch_velocity.
    grips, solver = GripManager(GrabConfig(), 2), _SolverStub(2)
    _drive(grips, solver, [_pose(0, -0.1, 0.9), _pose(1, 0.1, 0.0)], 10)
    require(solver.counts[0] > 0, "the first hand never took hold")
    _drive(grips, solver, [_pose(1, 0.1, 0.0)], 3)
    require(sum(solver.counts) == 0,
            f"the vanished hand's grab survived it: {solver.counts}")
    require(not grips.any_held, "a grip outlived the hand that took it")


@case
def reset_clears_every_field_a_grip_carries() -> None:
    grips = GripManager(GrabConfig(), 2)
    g = grips.grips[0]
    g.held, g.count, g.age, g.closing_for = True, 12, 0.5, 0.3
    g.just_grabbed = g.just_released = True
    g.release_speed = 3.5
    grips.reset()
    stale = [f.name for f in dataclasses.fields(g)
             if getattr(g, f.name) not in (0, 0.0, False, -1)]
    require(not stale, f"reset left {stale} set")


# -- the clock -----------------------------------------------------------


@case
def the_clock_never_hands_out_a_negative_step_count() -> None:
    clock = FixedTimestep(rate_hz=90.0)
    clock.tick(100.00)
    require(clock.tick(100.05) == 4, "forward time stopped working")
    steps = clock.tick(100.00)          # 50 ms backwards
    require(steps >= 0, f"a backward clock jump asked for {steps} steps")
    require(not steps, f"a backward jump produced {steps} steps out of nothing")
    # Time resumes from where it went back to, rather than staying stuck.
    require(clock.tick(100.05) == 4, "the clock did not recover")


@case
def a_clock_with_no_rate_is_rejected_at_construction() -> None:
    for rate in (0.0, -30.0):
        try:
            FixedTimestep(rate_hz=rate)
        except ValueError:
            continue
        raise AssertionError(f"FixedTimestep accepted rate_hz={rate}")


# -- configuration -------------------------------------------------------


@case
def a_runtime_preset_switch_carries_the_whole_preset() -> None:
    # A preset is a scene, a solver and an interaction volume.  Carrying only
    # `scene` gave the granular preset no basin and left the hand reaching at
    # chest height above a pile on the floor.
    for name in PRESETS:
        got = switch_preset(preset("cloth"), name)
        want = preset(name)
        for group in ("scene", "solver", "tracking"):
            for f in dataclasses.fields(getattr(want, group)):
                if f.name == "hardness":
                    continue
                a = getattr(getattr(got, group), f.name)
                b = getattr(getattr(want, group), f.name)
                require(a == b, f"switching to {name}: {group}.{f.name} "
                                f"is {a!r}, --preset {name} gives {b!r}")
    note("all seven presets reached by key match their command-line twin")


@case
def a_preset_switch_puts_back_what_the_previous_preset_changed() -> None:
    # The leak in the other direction: grain's basin standing as an invisible
    # cylinder in the cloth scene that replaced it.
    grain = switch_preset(preset("cloth"), "grain")
    require(grain.solver.basin_radius > 0.0, "grain arrived with no basin")
    back = switch_preset(grain, "cloth")
    require(back.solver.basin_radius == 0.0,
            f"the basin stayed up in the cloth scene ({back.solver.basin_radius})")
    require(back.tracking.stage_center_y == preset("cloth").tracking.stage_center_y,
            "the lowered interaction volume stayed behind")


@case
def a_preset_switch_leaves_the_session_alone() -> None:
    # Everything the CLI and the hardware set has to survive: a number key is
    # not a request to reopen the camera or drop --no-cuda-graph.
    args = build_parser().parse_args(
        ["--source", "synthetic", "--hands", "1", "--substeps", "6",
         "--rate", "120", "--no-cuda-graph", "--camera", "3"])
    cfg = config_from_args(args)
    after = switch_preset(cfg, "grain")
    require(after.tracking.source == "synthetic", "the hand source was reopened")
    require(after.tracking.max_hands == 1, "--hands was discarded")
    require(after.tracking.camera_index == 3, "--camera was discarded")
    require(after.solver.substeps == 6, "--substeps was discarded")
    require(after.solver.rate_hz == 120.0, "--rate was discarded")
    require(not after.solver.use_cuda_graph, "--no-cuda-graph was discarded")
    require(after.solver.basin_radius == preset("grain").solver.basin_radius,
            "and the basin still has to arrive")


@case
def the_dial_survives_a_preset_switch() -> None:
    cfg = dataclasses.replace(
        preset("cloth"), scene=dataclasses.replace(preset("cloth").scene,
                                                   hardness=0.83))
    require(switch_preset(cfg, "grain").scene.hardness == 0.83,
            "the hardness the user dialled in was thrown away")


@case
def a_bare_recording_name_means_the_same_file_to_record_and_replay() -> None:
    # The README prints `--record take01.fhr` then `--replay take01.fhr`.
    rec = config_from_args(build_parser().parse_args(["--record", "t.fhr"]))
    rep = config_from_args(
        build_parser().parse_args(["--source", "replay", "--replay", "t.fhr"]))
    require(rec.tracking.record_path == rep.tracking.replay_path,
            f"--record wrote {rec.tracking.record_path} but --replay looks in "
            f"{rep.tracking.replay_path}")
    require(rec.tracking.record_path.parent == RECORDING_DIR,
            "a bare name no longer lands in the recordings directory")
    absolute = Path(__file__).resolve()
    got = config_from_args(build_parser().parse_args(
        ["--source", "replay", "--replay", str(absolute)]))
    require(got.tracking.replay_path == absolute,
            "an absolute --replay path was rewritten")


# -- controls ------------------------------------------------------------


@case
def the_pointer_is_normalised_against_the_window_it_is_in() -> None:
    # The synthetic hand is the no-camera path, and its pointer is the cursor
    # divided by this.  glfw only reports a size when it changes, so a window
    # that is never dragged keeps whatever the state was seeded with.
    controls = Controls(ControlState(_window_size=(1280, 720)))
    controls.handle([InputEvent(kind=EventKind.CURSOR, x=640.0, y=360.0)])
    px, py = controls.state.pointer
    require(abs(px - 0.5) < 1e-6 and abs(py - 0.5) < 1e-6,
            f"the centre of a 1280x720 window mapped to {(px, py)}")
    controls.handle([InputEvent(kind=EventKind.RESIZE, x=1600.0, y=900.0)])
    controls.handle([InputEvent(kind=EventKind.CURSOR, x=800.0, y=450.0)])
    px, py = controls.state.pointer
    require(abs(px - 0.5) < 1e-6 and abs(py - 0.5) < 1e-6,
            f"after a resize the centre mapped to {(px, py)}")


@case
def a_control_state_can_be_seeded_from_a_config() -> None:
    # What fctx.app does at startup, and what it has to redo on a preset
    # switch: every toggle starts where the config already is, or the first
    # frame reads the dataclass default as a key press.
    cfg = preset("banner")
    state = ControlState(wind=cfg.solver.wind != (0.0, 0.0, 0.0),
                         show_hud=cfg.render.show_hud)
    require(state.wind, "a preset that ships with wind seeded the toggle off")
    require(ControlState().wind is False,
            "the bare default changed; app.py's seeding assumes it is off")


if __name__ == "__main__":
    raise SystemExit(run(__file__))
