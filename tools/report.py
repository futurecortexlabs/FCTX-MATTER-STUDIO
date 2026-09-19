"""Turn the analytics CSV into the numbers an installation gets asked for.

    uv run python tools/report.py logs/events.csv
    uv run python tools/report.py logs/events.csv --day 2026-09-19 --csv out.csv

The input is what ``--analytics`` (or ``--kiosk``) appends: one row per
event with a wall-clock stamp.  The report counts, per hour and in total:
visitors (runs of frames with a hand, ended by ``session_gap`` seconds
without one), grabs, seconds spent holding matter, deliberate dial moves,
attract-mode cycles, camera losses, errors and restarts -- and uptime, from
``start``/``stop`` pairs.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _parse(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                when = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                continue
            rows.append((when, row.get("event", ""), row.get("detail", "")))
    rows.sort(key=lambda r: r[0])
    return rows


_END = re.compile(r"([0-9.]+)s grabs=(\d+) hold=([0-9.]+)s dial_moves=(\d+)")


def summarise(rows, day: str | None = None):
    if day:
        rows = [r for r in rows if r[0].strftime("%Y-%m-%d") == day]
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total: dict[str, float] = defaultdict(float)
    uptime = 0.0
    started: datetime | None = None
    for when, event, detail in rows:
        key = when.strftime("%Y-%m-%d %H:00")
        h = hours[key]
        if event == "visitor_end":
            m = _END.search(detail)
            h["visitors"] += 1
            total["visitors"] += 1
            if m:
                secs, hold = (float(m.group(1)),
                              float(m.group(3)))
                h["seconds"] += secs
                total["seconds"] += secs
                h["hold"] += hold
                total["hold"] += hold
        elif event == "grab":
            h["grabs"] += 1
            total["grabs"] += 1
        elif event == "dial":
            h["dial"] += 1
            total["dial"] += 1
        elif event == "attract_on":
            h["attract"] += 1
            total["attract"] += 1
        elif event == "camera_lost":
            h["camera_lost"] += 1
            total["camera_lost"] += 1
        elif event == "error":
            h["errors"] += 1
            total["errors"] += 1
        elif event == "restart":
            h["restarts"] += 1
            total["restarts"] += 1
        elif event == "start":
            started = when
        elif event == "stop":
            if started is not None:
                uptime += (when - started).total_seconds()
                started = None
    if started is not None and rows:
        uptime += (rows[-1][0] - started).total_seconds()
    total["uptime"] = uptime
    return hours, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", type=Path, nargs="+", help="events CSV file(s)")
    ap.add_argument("--day", help="only this day, YYYY-MM-DD")
    ap.add_argument("--csv", dest="out", type=Path, help="also write the hourly table here")
    args = ap.parse_args(argv)

    rows = []
    for path in args.csv:
        if not path.exists():
            print(f"report: {path} does not exist", file=sys.stderr)
            return 2
        rows.extend(_parse(path))
    if not rows:
        print("report: no events")
        return 1
    hours, total = summarise(rows, args.day)

    print(f"{'hour':<17}{'visitors':>9}{'grabs':>7}{'held s':>8}{'dial':>6}"
          f"{'use min':>9}{'attract':>8}{'cam':>5}{'err':>5}")
    for key in sorted(hours):
        h = hours[key]
        print(f"{key:<17}{int(h['visitors']):>9}{int(h['grabs']):>7}"
              f"{h['hold']:>8.0f}{int(h['dial']):>6}{h['seconds'] / 60:>9.1f}"
              f"{int(h['attract']):>8}{int(h['camera_lost']):>5}{int(h['errors']):>5}")
    up_h = total["uptime"] / 3600.0
    print()
    print(f"total  visitors {int(total['visitors'])}"
          f"  ({total['visitors'] / up_h:.1f}/h over {up_h:.1f} h uptime)"
          if up_h > 0 else f"total  visitors {int(total['visitors'])}")
    if total["visitors"]:
        print(f"       {total['seconds'] / total['visitors']:.0f} s per visitor, "
              f"{total['grabs'] / total['visitors']:.1f} grabs each, "
              f"{total['hold'] / max(total['seconds'], 1e-9) * 100:.0f}% of their time holding, "
              f"{total['dial'] / total['visitors']:.1f} dial moves each")
    print(f"       attract cycles {int(total['attract'])}, camera losses "
          f"{int(total['camera_lost'])}, errors {int(total['errors'])}, "
          f"restarts {int(total['restarts'])}")

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["hour", "visitors", "grabs", "held_s", "dial_moves", "use_s",
                        "attract", "camera_lost", "errors"])
            for key in sorted(hours):
                h = hours[key]
                w.writerow([key, int(h["visitors"]), int(h["grabs"]), round(h["hold"], 1),
                            int(h["dial"]), round(h["seconds"], 1), int(h["attract"]),
                            int(h["camera_lost"]), int(h["errors"])])
        print(f"       hourly table -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
