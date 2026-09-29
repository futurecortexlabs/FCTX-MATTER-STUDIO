"""Analyse a study log: thresholds, the pseudo-haptics effect, preferences.

    uv run python tools/analyze_study.py studies/results/hardness_jnd.csv
    uv run python tools/analyze_study.py studies/results/foam_preference.csv --csv summary.csv
    uv run python tools/analyze_study.py results.csv --plot psychometric.png

Discrimination (two-alternative forced choice, "which is harder?"), per
pseudo-haptics condition:

* the staircase estimate -- geometric mean of the last reversals of the
  delta series, the classic 2-down-1-up reading (70.7% correct);
* a psychometric fit, P(correct) = 0.5 + 0.5 / (1 + exp(-(ln d - m) / s)),
  by maximum likelihood over a grid, with its 75% point exp(m) as the
  threshold and a bootstrap 95% interval;
* the same threshold as a Weber fraction of Young's modulus: the dial maps
  hardness to modulus geometrically, so a difference of d on the dial is a
  modulus ratio of (E_hard / E_soft) ** d whatever the reference;
* the effect of pseudo-haptics: threshold(off) / threshold(on) with a
  bootstrap interval.  Above 1 means the illusion made materials easier to
  tell apart.

Preference (paired comparison): win counts, Bradley-Terry strengths by the
MM algorithm (Hunter, 2004), and the implied probability that each material
beats the median one.

Also for both: side bias (a real panel should choose left about half the
time) and response times.  The numbers are only as good as the panel:
this says how many answers each estimate rests on, and says so loudly
when that is too few.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fctx.core.material import DEFAULT_MATERIALS  # noqa: E402
from fctx.core.types import MatterKind  # noqa: E402

MIN_TRIALS = 30          # below this a threshold is reported as unreliable
#: Above this on the soft dial the shipped samples' tetrahedra are at their
#: resolvable stiffness ceiling (ARCHITECTURE 6.4): ordering holds, the
#: modulus ratio becomes nominal.
NOMINAL_ABOVE = 0.4
BOOT = 400


def load(paths: list[Path], study: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if study:
        rows = [r for r in rows if r.get("study") == study]
    return rows


# ---------------------------------------------------------------------------
# discrimination
# ---------------------------------------------------------------------------


def reversals(deltas: list[float]) -> list[float]:
    """Values of ``deltas`` where the series turns around (plateaus skipped)."""
    out: list[float] = []
    direction = 0
    for prev, cur in zip(deltas, deltas[1:]):
        if abs(cur - prev) < 1e-9:
            continue
        d = 1 if cur > prev else -1
        if direction and d != direction:
            out.append(prev)
        direction = d
    return out


def staircase_threshold(deltas: list[float], last: int = 6) -> float | None:
    rev = reversals(deltas)[-last:]
    if len(rev) < 2:
        return None
    return float(math.exp(np.mean(np.log(rev))))


_M = np.linspace(math.log(0.003), math.log(1.0), 140)
_S = np.geomspace(0.04, 2.0, 60)


def _fit_counts(levels: np.ndarray, n: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    """(m, s) maximising the 2AFC logistic likelihood, from counts per level.

    A staircase visits a few dozen distinct deltas however many trials it
    runs, so the likelihood is summed per level (n trials, k correct) rather
    than per trial; the bootstrap below leans on that for its speed.
    """
    x = np.log(np.maximum(levels, 1e-6))[None, None, :]
    m = _M[:, None, None]
    s = _S[None, :, None]
    p = 0.5 + 0.5 / (1.0 + np.exp(-(x - m) / s))
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    ll = (k[None, None, :] * np.log(p) + (n - k)[None, None, :] * np.log(1.0 - p)).sum(axis=2)
    i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)
    return float(_M[i]), float(_S[j])


def fit_psychometric(delta: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """(m, s) maximising the 2AFC logistic likelihood in log delta."""
    levels, inv = np.unique(np.round(delta, 6), return_inverse=True)
    n = np.bincount(inv, minlength=levels.size).astype(float)
    k = np.bincount(inv, weights=correct, minlength=levels.size)
    return _fit_counts(levels, n, k)


def bootstrap(delta: np.ndarray, correct: np.ndarray, rng: np.random.Generator,
              n: int = BOOT) -> np.ndarray:
    """Thresholds refitted on trials resampled with replacement."""
    levels, inv = np.unique(np.round(delta, 6), return_inverse=True)
    out = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, delta.size, delta.size)
        cnt = np.bincount(inv[idx], minlength=levels.size).astype(float)
        hit = np.bincount(inv[idx], weights=correct[idx], minlength=levels.size)
        out[b] = math.exp(_fit_counts(levels, cnt, hit)[0])
    return out


def modulus_ratio(delta: float, kind: MatterKind = MatterKind.SOFT) -> float:
    p = DEFAULT_MATERIALS[kind]
    lo, hi = ((p.young_soft, p.young_hard) if kind is MatterKind.SOFT
              else (p.stretch_soft, p.stretch_hard))
    return float((hi / lo) ** delta)


def discrimination(rows: list[dict[str, str]], rng: np.random.Generator) -> dict:
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r.get("correct") in ("0", "1") and r.get("delta"):
            by[r.get("condition", "on")].append(r)
    result: dict[str, dict] = {}
    boots: dict[str, np.ndarray] = {}
    for cond, rs in sorted(by.items()):
        delta = np.array([float(r["delta"]) for r in rs])
        correct = np.array([float(r["correct"]) for r in rs])
        m, s = fit_psychometric(delta, correct)
        b = bootstrap(delta, correct, rng)
        boots[cond] = b
        thr = math.exp(m)
        stair = staircase_threshold([float(r["delta"]) for r in rs])
        result[cond] = {
            "trials": int(delta.size),
            "participants": len({r.get("participant") for r in rs}),
            "percent_correct": float(correct.mean() * 100.0),
            "staircase": stair,
            "threshold": thr,
            "ci": (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))),
            "slope": s,
            "weber_E": modulus_ratio(thr) - 1.0,
            "reliable": delta.size >= MIN_TRIALS,
            "reference": float(np.median([float(r["reference"]) for r in rs
                                          if r.get("reference")] or [0.0])),
            "points": (delta, correct),
        }
    if "on" in boots and "off" in boots:
        ratio = boots["off"] / boots["on"]
        result["effect"] = {
            "ratio": result["off"]["threshold"] / result["on"]["threshold"],
            "ci": (float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))),
        }
    return result


# ---------------------------------------------------------------------------
# preference
# ---------------------------------------------------------------------------


def bradley_terry(names: list[str], wins: np.ndarray, iters: int = 500) -> np.ndarray:
    """Strengths p (geometric mean 1) with wins[i, j] = times i beat j."""
    n = len(names)
    games = wins + wins.T
    p = np.ones(n)
    w = wins.sum(axis=1)
    for _ in range(iters):
        denom = np.array([sum(games[i, j] / (p[i] + p[j]) for j in range(n) if j != i)
                          for i in range(n)])
        new = np.where(denom > 0, (w + 0.5) / np.maximum(denom, 1e-12), p)
        new /= math.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < 1e-10:
            p = new
            break
        p = new
    return p


def preference(rows: list[dict[str, str]]) -> dict:
    rs = [r for r in rows if r.get("chosen") and r.get("protocol") == "preference"]
    names = sorted({r["left_label"] for r in rs} | {r["right_label"] for r in rs})
    idx = {n: i for i, n in enumerate(names)}
    wins = np.zeros((len(names), len(names)))
    for r in rs:
        a, b = r["left_label"], r["right_label"]
        winner = r["chosen"]
        loser = b if winner == a else a
        wins[idx[winner], idx[loser]] += 1
    strength = bradley_terry(names, wins) if names else np.zeros(0)
    return {"names": names, "wins": wins, "strength": strength, "trials": len(rs),
            "participants": len({r.get("participant") for r in rs})}


# ---------------------------------------------------------------------------
# common checks
# ---------------------------------------------------------------------------


def checks(rows: list[dict[str, str]]) -> dict:
    answered = [r for r in rows if r.get("response_side")]
    left = sum(1 for r in answered if r["response_side"] == "left")
    rt = [float(r["rt_s"]) for r in answered if r.get("rt_s")]
    ex = [float(r["explore_s"]) for r in answered if r.get("explore_s")]
    return {
        "answers": len(answered),
        "left_rate": left / len(answered) if answered else float("nan"),
        "rt_median": float(np.median(rt)) if rt else float("nan"),
        "explore_median": float(np.median(ex)) if ex else float("nan"),
        "by_hand": sum(1 for r in answered if r.get("response_mode") == "hand"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--study", help="only rows of this study name")
    ap.add_argument("--csv", dest="out", type=Path, help="write a summary table here")
    ap.add_argument("--plot", type=Path, help="psychometric plot (needs matplotlib)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    for p in args.csv:
        if not p.exists():
            print(f"analyze: {p} does not exist", file=sys.stderr)
            return 2
    rows = load(args.csv, args.study)
    if not rows:
        print("analyze: no rows")
        return 1
    rng = np.random.default_rng(args.seed)
    studies = sorted({r.get("study", "") for r in rows})
    print(f"{len(rows)} answers, studies: {', '.join(studies)}, "
          f"participants: {len({(r.get('study'), r.get('participant')) for r in rows})}")
    c = checks(rows)
    print(f"side bias: chose left {c['left_rate'] * 100:.0f}% "
          f"(50% expected), median explore {c['explore_median']:.1f} s, "
          f"median response {c['rt_median']:.1f} s, {c['by_hand']}/{c['answers']} by hand")

    summary: list[list[object]] = []
    disc = discrimination(rows, rng)
    if disc:
        print("\nDISCRIMINATION  (threshold = hardness-dial difference at 75% correct)")
        for cond in ("on", "off"):
            if cond not in disc:
                continue
            d = disc[cond]
            stair = f"{d['staircase']:.3f}" if d["staircase"] else "n/a"
            warn = "" if d["reliable"] else f"   << only {d['trials']} trials; need {MIN_TRIALS}+"
            print(f"  pseudo-haptics {cond:>3}: {d['trials']:4d} trials, "
                  f"{d['participants']:3d} people, {d['percent_correct']:.0f}% correct")
            print(f"      threshold {d['threshold']:.3f} "
                  f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]  staircase {stair}  "
                  f"-> modulus Weber fraction {d['weber_E'] * 100:.0f}%{warn}")
            if d["threshold"] + d.get("reference", 0.0) > NOMINAL_ABOVE:
                print(f"      note: comparisons reach above {NOMINAL_ABOVE} on the dial, where "
                      "the tetrahedra are at their stiffness ceiling; the modulus "
                      "ratio is nominal")
            summary.append(["discrimination", cond, d["trials"], d["participants"],
                            round(d["threshold"], 4), round(d["ci"][0], 4),
                            round(d["ci"][1], 4), round(d["weber_E"], 4)])
        if "effect" in disc:
            e = disc["effect"]
            verdict = ("helps" if e["ci"][0] > 1.0 else
                       "hurts" if e["ci"][1] < 1.0 else "no clear effect")
            print(f"  pseudo-haptics effect: threshold off/on = {e['ratio']:.2f} "
                  f"[{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]  -> {verdict}")
            summary.append(["effect", "off/on", "", "", round(e["ratio"], 4),
                            round(e["ci"][0], 4), round(e["ci"][1], 4), ""])

    pref = preference(rows)
    if pref["trials"]:
        print(f"\nPREFERENCE  ({pref['trials']} choices, {pref['participants']} people)")
        order = np.argsort(-pref["strength"])
        med = float(np.median(pref["strength"]))
        for rank, i in enumerate(order, 1):
            name = pref["names"][i]
            won = int(pref["wins"][i].sum())
            played = int(pref["wins"][i].sum() + pref["wins"][:, i].sum())
            s = float(pref["strength"][i])
            print(f"  {rank}. {name:<28} strength {s:6.2f}  "
                  f"beats the median {s / (s + med) * 100:3.0f}%  "
                  f"({won}/{played} won)")
            summary.append(["preference", name, played, pref["participants"],
                            round(s, 4), "", "", ""])

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "condition_or_material", "trials", "participants",
                        "value", "ci_low", "ci_high", "weber_E"])
            w.writerows(summary)
        print(f"\nsummary -> {args.out}")

    if args.plot and disc:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("analyze: matplotlib is not installed; no plot")
            return 0
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = np.geomspace(0.005, 0.8, 200)
        for cond, colour in (("on", "#2a7de1"), ("off", "#e0782a")):
            if cond not in disc:
                continue
            d = disc[cond]
            delta, correct = d["points"]
            bins = np.geomspace(max(delta.min(), 1e-3), delta.max() * 1.001, 8)
            which = np.digitize(delta, bins)
            for b in np.unique(which):
                sel = which == b
                ax.scatter(np.exp(np.log(delta[sel]).mean()), correct[sel].mean(),
                           s=12 + 6 * sel.sum(), color=colour, alpha=0.6)
            m, s = fit_psychometric(delta, correct)
            ax.plot(xs, 0.5 + 0.5 / (1 + np.exp(-(np.log(xs) - m) / s)), color=colour,
                    label=f"pseudo-haptics {cond}: {math.exp(m):.3f}")
        ax.axhline(0.75, color="#999", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("hardness difference on the dial")
        ax.set_ylabel("proportion correct")
        ax.set_ylim(0.4, 1.02)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"plot -> {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
