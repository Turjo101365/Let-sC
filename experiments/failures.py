"""Failure cases of a candidate variant against the baseline, from Smart Tags and signals only.

    python experiments/failures.py speed8
"""

import os
import sys

import numpy as np

import common
sys.path.insert(0, os.path.join(common.ROOT, "evaluation"))
import self_check  # noqa: E402
import variants  # noqa: E402


def clock(frame):
    """Match clock-ish position: video minute:second, for finding the moment in the video."""
    s = frame / common.FPS
    return "{:d}:{:02d}".format(int(s // 60), int(s % 60))


def main(name):
    tags = self_check.load_tags()
    base = self_check.read_labels(os.path.join(common.OUT_DIR, "base.csv"))
    cand = self_check.read_labels(os.path.join(common.OUT_DIR, name + ".csv"))
    eb, ec = self_check.per_tag(base, tags), self_check.per_tag(cand, tags)

    raw = common.load_raw()
    bx, by, vis, _ = common.baseline_track(raw)
    speed = variants.ball_speed(bx, by)

    at = [t for t in tags if self_check.code_of(t) == "attacking transition"
          and self_check.team_of(t)]
    print("== ATTACKING TRANSITION tags found by base but missed by", name)
    for t, ob, oc in zip(at, eb["trans"], ec["trans"]):
        if ob is not None and oc is None:
            f0 = t["start_frame"] + ob
            win = slice(max(f0 - 25, 0), f0 + 25)
            fast = np.nanmean(speed[win] > 8)
            print("  tag at video {} team {}: base switched {:+d} frames from tag; "
                  "ball visible {:.0%}, faster than 8 m/s on {:.0%} of the +-1 s window".format(
                      clock(t["start_frame"]), self_check.team_of(t), ob, vis[win].mean(), fast))

    rs = [t for t in tags if self_check.code_of(t) == "set pieces"
          and (t["tags"].get("Type") or "").strip().lower() in self_check.RESTART_TYPES
          and self_check.team_of(t)]
    print("\n== Team-tagged restarts", name, "still gets wrong")
    for t, hit in zip(rs, ec["restart"]):
        if not hit:
            print("  {} at video {} tagged {}".format(t["tags"].get("Type"), clock(t["start_frame"]),
                                                     self_check.team_of(t)))

    print("\n== Remaining ping-pongs in", name, "(middle possession under 2 s)")
    spells = [(s, e, lbl) for s, e, lbl in self_check.runs(cand) if lbl in ("H", "A")]
    n = 0
    for (_, _, a), (s, e, b), (_, _, c) in zip(spells, spells[1:], spells[2:]):
        if a == c != b and e - s < self_check.PINGPONG_MAX:
            n += 1
            if n <= 8:
                print("  {}-{} at video {} lasting {:.2f} s; ball visible {:.0%}".format(
                    a, b, clock(s), (e - s) / common.FPS, vis[s:e].mean()))
    print("  ... {} in total".format(n))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "speed8")
