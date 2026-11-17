"""Label-free self-check of a possession file against Metrica's automatic Smart Tagging.

RULES.md forbids using the public 5-minute ground truth to choose between versions of a
pipeline, and there are no human labels on this game we are allowed to use. This script is how
Team Let'sC compared versions instead. It reads only the Smart Tagging, which DATA.md describes as
Metrica's *automatic* annotation of the tactical video (no manual review), so it is algorithmic
evidence about the game, not human-produced data.

The tags are noisy and are phases, not possessions, so none of these numbers is a score. They are
directional signals; a change is kept only when they agree it helps and nothing gets worse.

    python evaluation/self_check.py evaluation/results/Let_sC.csv [more.csv ...]
    python evaluation/self_check.py base.csv variant.csv --baseline base.csv   # paired bootstrap

Signals
-------
team_agree      share of predicted-alive frames inside attacking phases (BUILD UP, PROGRESSION,
                ATTACKING TRANSITION, CREATING CHANCES) whose team matches the phase's Team tag.
                The DEFENDING IN ... codes are exact mirrors of these (same intervals, other team)
                and are not counted twice.
trans_hit       share of ATTACKING TRANSITION tags (the moment a team wins the ball) with a
                predicted hand-over to that team within +-5 s of the tag start.
trans_med       median offset, in frames, of those hand-overs from the tag start (+ = late). It is
                over whichever transitions a file matches, so it is not like-for-like.
restart_agree   share of team-tagged set pieces (corners, goal kicks, free kicks) where the first
                predicted-alive frame in the tagged window belongs to the tagged team.
dead_cover      share of all set-piece windows that contain at least one second of predicted D.
switches_min    predicted H<->A hand-overs per minute of predicted-alive play inside attacking
                phases. A tagged phase is mostly one team's ball, so lower means less flicker.
pingpong        H->A->H or A->H->A sequences where the middle spell lasts under 2 s, across the
                whole match. Needs no tags: a direct count of flicker.
possessions     maximal H/A runs after the scorer's X forward-fill. Reported, not optimised.
dur_p10/50/90   possession duration percentiles in seconds (X treated as a break).
short_2s        share of possessions shorter than 2 s.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags  # noqa: E402

TEAM = {"barcelona": "H", "atletic club": "A"}
ATTACKING = {"build up", "progression", "attacking transition", "creating chances"}
RESTART_TYPES = {"corner kick", "goal kick", "free kick"}
FPS = 25
TRANS_WINDOW = 5 * FPS
PINGPONG_MAX = 2 * FPS
BOOTSTRAP_DRAWS = 2000


def team_of(tag):
    return TEAM.get((tag["tags"].get("Team") or "").strip().lower())


def code_of(tag):
    return tag["code"].strip().lower()


def forward_fill_x(labels):
    out = labels.copy()
    last = None
    for i, lbl in enumerate(out):
        if lbl == "X":
            if last is not None:
                out[i] = last
        else:
            last = lbl
    return out


def runs(labels):
    """(start, end, label) for every maximal run, end exclusive."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return [(int(s), int(e), labels[s]) for s, e in zip(starts, ends)]


def count_possessions(filled):
    return sum(1 for _, _, lbl in runs(filled) if lbl in ("H", "A"))


def spell_stats(labels):
    spells = [(s, e, lbl) for s, e, lbl in runs(labels) if lbl in ("H", "A")]
    dur = np.array([(e - s) / FPS for s, e, _ in spells]) if spells else np.zeros(1)
    # Ping-pong: three consecutive possessions (ignoring dead ball between them) going X-Y-X with
    # a short middle one.
    pingpong = 0
    for (_, _, a), (s, e, b), (_, _, c) in zip(spells, spells[1:], spells[2:]):
        if a == c != b and e - s < PINGPONG_MAX:
            pingpong += 1
    return {
        "dur_p10": float(np.percentile(dur, 10)),
        "dur_p50": float(np.percentile(dur, 50)),
        "dur_p90": float(np.percentile(dur, 90)),
        "short_2s": float((dur < 2).mean()),
        "pingpong": pingpong,
    }


def per_tag(labels, tags):
    """Per-tag evidence, kept separate so it can be bootstrapped."""
    filled = forward_fill_x(labels)
    team_rows, restart_rows, trans_rows = [], [], []
    dead_hit = dead_tot = 0

    idx = np.flatnonzero((filled == "H") | (filled == "A"))
    seq = filled[idx]
    change = idx[1:][seq[1:] != seq[:-1]]
    change_team = filled[change]

    for t in tags:
        code, team = code_of(t), team_of(t)
        seg = labels[t["start_frame"]:t["end_frame"]]
        if code in ATTACKING and team is not None:
            alive = (seg == "H") | (seg == "A")
            a = seg[alive]
            team_rows.append((int((seg == team).sum()), int(alive.sum()),
                              int((a[1:] != a[:-1]).sum())))
        if code == "attacking transition" and team is not None:
            cand = change[(change_team == team)
                          & (np.abs(change - t["start_frame"]) <= TRANS_WINDOW)]
            trans_rows.append(int(cand[np.argmin(np.abs(cand - t["start_frame"]))]
                                  - t["start_frame"]) if len(cand) else None)
        if code == "set pieces":
            dead_tot += 1
            dead_hit += int((seg == "D").sum() >= FPS)
            typ = (t["tags"].get("Type") or "").strip().lower()
            if typ in RESTART_TYPES and team is not None:
                window = labels[t["start_frame"]:t["end_frame"] + TRANS_WINDOW]
                alive = np.flatnonzero((window == "H") | (window == "A"))
                restart_rows.append(int(len(alive) > 0 and window[alive[0]] == team))

    return {
        "team": np.array(team_rows, dtype=float).reshape(-1, 3),
        "restart": np.array(restart_rows, dtype=float),
        "trans": trans_rows,
        "dead": (dead_hit, dead_tot),
        "filled": filled,
    }


def summarise(ev, labels):
    team = ev["team"]
    hits = [o for o in ev["trans"] if o is not None]
    alive_min = team[:, 1].sum() / (60 * FPS)
    out = {
        "team_agree": team[:, 0].sum() / max(team[:, 1].sum(), 1),
        "trans_hit": len(hits) / max(len(ev["trans"]), 1),
        "trans_med": float(np.median(hits)) if hits else float("nan"),
        "restart_agree": ev["restart"].mean() if len(ev["restart"]) else float("nan"),
        "dead_cover": ev["dead"][0] / max(ev["dead"][1], 1),
        "switches_min": team[:, 2].sum() / max(alive_min, 1e-9),
        "possessions": count_possessions(ev["filled"]),
    }
    out.update(spell_stats(labels))
    return out


def check(labels, tags):
    return summarise(per_tag(labels, tags), labels)


def paired_bootstrap(base, other, seed=0):
    """95% intervals for (other - base) on the tag-level signals, resampling tags in pairs."""
    rng = np.random.default_rng(seed)
    out = {}
    tb, to = base["team"], other["team"]
    n = len(tb)
    i = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    d = (to[i, 0].sum(1) / to[i, 1].sum(1)) - (tb[i, 0].sum(1) / tb[i, 1].sum(1))
    out["team_agree"] = d
    rb, ro = base["restart"], other["restart"]
    j = rng.integers(0, len(rb), size=(BOOTSTRAP_DRAWS, len(rb)))
    out["restart_agree"] = ro[j].mean(1) - rb[j].mean(1)
    hb = np.array([o is not None for o in base["trans"]], dtype=float)
    ho = np.array([o is not None for o in other["trans"]], dtype=float)
    k = rng.integers(0, len(hb), size=(BOOTSTRAP_DRAWS, len(hb)))
    out["trans_hit"] = ho[k].mean(1) - hb[k].mean(1)
    per_min = 60 * FPS
    out["switches_min"] = per_min * ((to[i, 2].sum(1) / to[i, 1].sum(1))
                                     - (tb[i, 2].sum(1) / tb[i, 1].sum(1)))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def read_labels(path):
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline", help="file to compare the others against (paired bootstrap)")
    args = ap.parse_args()
    tags = load_tags()
    rows, evidence = [], {}
    for path in args.files:
        labels = read_labels(path)
        ev = per_tag(labels, tags)
        evidence[path] = ev
        rows.append(dict(file=os.path.basename(path), **summarise(ev, labels)))
    df = pd.DataFrame(rows).set_index("file")
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(df)

    if args.baseline:
        base = per_tag(read_labels(args.baseline), tags)
        print("\npaired bootstrap vs {} (95% interval of variant - baseline; "
              "switches_min: lower is steadier)".format(os.path.basename(args.baseline)))
        for path in args.files:
            if os.path.abspath(path) == os.path.abspath(args.baseline):
                continue
            ci = paired_bootstrap(base, evidence[path])
            print("  {:28s} ".format(os.path.basename(path)) + "  ".join(
                "{} [{:+.3f}, {:+.3f}]".format(k, lo, hi) for k, (lo, hi) in ci.items()))


if __name__ == "__main__":
    main()
