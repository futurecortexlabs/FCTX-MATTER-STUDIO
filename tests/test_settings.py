"""The configuration file must round-trip, and must refuse to be misread."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from _harness import case, note, require, run

from fctx.config import AppConfig, preset
from fctx.core.types import MatterKind
from fctx.settings import ConfigError, apply_toml, dump_config, load_config


@case
def dumped_defaults_load_back_identically() -> None:
    for name in ("cloth", "soft", "grain"):
        cfg = preset(name)
        text = dump_config(cfg, name)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fctx.toml"
            path.write_text(text, encoding="utf-8")
            back = load_config(path)
        # Paths dump as strings; everything else must compare exactly.
        require(back == cfg, f"{name}: the dumped config did not load back equal")
    note(f"round trip is exact; {len(text.splitlines())} lines for the grain preset")


@case
def a_section_overrides_a_field_and_leaves_the_rest_alone() -> None:
    cfg = apply_toml({"tracking": {"camera_index": 3, "mirror": False},
                      "render": {"fullscreen": True}})
    require(cfg.tracking.camera_index == 3)
    require(cfg.tracking.mirror is False)
    require(cfg.render.fullscreen is True)
    require(cfg.solver == AppConfig().solver, "an untouched section changed")


@case
def a_preset_in_the_file_is_the_starting_point() -> None:
    cfg = apply_toml({"preset": "grain", "scene": {"hardness": 0.9}})
    require(cfg.scene.kind is MatterKind.GRAIN)
    require(abs(cfg.scene.hardness - 0.9) < 1e-12)
    require(cfg.solver.basin_radius > 0.0, "the grain preset's basin was lost")


@case
def tuples_enums_and_paths_are_coerced() -> None:
    cfg = apply_toml({
        "solver": {"gravity": [0, -1.5, 0]},
        "scene": {"kind": "soft"},
        "tracking": {"replay_path": "takes/one.fhr"},
        "log_file": "run.log",
    })
    require(cfg.solver.gravity == (0.0, -1.5, 0.0))
    require(cfg.scene.kind is MatterKind.SOFT)
    require(isinstance(cfg.tracking.replay_path, Path))
    require(isinstance(cfg.log_file, Path) and cfg.log_file.name == "run.log")


@case
def a_typo_fails_loudly_and_names_the_key() -> None:
    for data, fragment in (
            ({"trackng": {"camera_index": 1}}, "trackng"),
            ({"tracking": {"camera_idx": 1}}, "camera_idx"),
            ({"tracking": {"mirror": "yes"}}, "mirror"),
            ({"solver": {"substeps": 12.5}}, "substeps"),
            ({"solver": {"gravity": [0, -9.8]}}, "gravity"),
            ({"preset": "granite"}, "granite"),
            ({"scene": {"kind": "steel"}}, "steel"),
    ):
        try:
            apply_toml(data)
        except ConfigError as exc:
            require(fragment in str(exc),
                    f"the error for {data} does not mention {fragment!r}: {exc}")
            continue
        raise AssertionError(f"{data} was accepted")


@case
def an_unreadable_or_malformed_file_is_a_config_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.toml"
        bad.write_text("[tracking\ncamera_index = 1\n", encoding="utf-8")
        for path in (bad, Path(tmp) / "missing.toml"):
            try:
                load_config(path)
            except ConfigError:
                continue
            raise AssertionError(f"{path.name} did not raise ConfigError")


@case
def the_cli_layers_file_then_flags() -> None:
    from fctx.__main__ import build_parser, config_from_args

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "site.toml"
        path.write_text('preset = "soft"\n[tracking]\ncamera_index = 2\n'
                        '[render]\nshow_hud = false\n', encoding="utf-8")
        parser = build_parser()
        cfg = config_from_args(parser.parse_args(["--config", str(path)]))
        require(cfg.scene.kind is MatterKind.SOFT, "the file's preset was ignored")
        require(cfg.tracking.camera_index == 2)
        require(cfg.render.show_hud is False)

        cfg = config_from_args(parser.parse_args(
            ["--config", str(path), "--preset", "cloth", "--camera", "5"]))
        require(cfg.scene.kind is MatterKind.CLOTH, "--preset did not win over the file")
        require(cfg.tracking.camera_index == 5, "--camera did not win over the file")
        require(cfg.render.show_hud is False, "the file's other sections were dropped")

        kiosk = config_from_args(parser.parse_args(["--kiosk"]))
        require(kiosk.render.fullscreen and kiosk.resilient and kiosk.idle_demo > 0.0,
                "--kiosk did not switch on fullscreen, resilience and attract mode")


if __name__ == "__main__":
    os.environ.setdefault("FCTX_DEVICE", "cpu")
    raise SystemExit(run(__file__))
