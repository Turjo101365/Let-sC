"""Audit of the submitted pipeline's weak points, from its own signals only (no labels).

    python experiments/audit.py
"""

import numpy as np

import common


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main():
    raw = common.load_raw()
    bx, by, vis, out = common.baseline_track(raw)
    ev = common.evidence(raw, bx, by, vis)
    dead = common.dead_mask(raw, out)
    labels = common.baseline_labels(raw, ev, dead)

    h1, h2, _ = common.halves()
    graded = np.zeros(raw["n_frames"], bool)
    for s, e in (h1, h2):
        graded[s:e + 1] = True
    alive = graded & ~dead

    print("== Ball visibility in live play (after the 4-frame bridge)")
    raw_vis = ~np.isnan(raw["raw_bx"])
    print("  raw ball seen   {:.1%} of live frames".format(raw_vis[alive].mean()))
    print("  after bridging  {:.1%}".format(vis[alive].mean()))
    gaps = [(s, e) for s, e in runs(alive & ~vis)]
    glen = np.array([e - s for s, e in gaps])
    for lo, hi in ((1, 25), (25, 75), (75, 250), (250, 10 ** 9)):
        sel = (glen >= lo) & (glen < hi)
        print("  gaps {:>4}-{:<5} frames: {:5d} gaps, {:6.1%} of live frames".format(
            lo, hi if hi < 10 ** 9 else "inf", sel.sum(), glen[sel].sum() / alive.sum()))

    print("\n== Live stretches between dead balls")
    stretches = runs(alive)
    slen = np.array([e - s for s, e in stretches]) / common.FPS
    print("  {} stretches, median {:.1f} s; {} shorter than 2 s, {} shorter than 5 s".format(
        len(slen), np.median(slen), (slen < 2).sum(), (slen < 5).sum()))

    print("\n== Where short possessions (< 2 s) come from")
    lab = np.array(labels)
    spells = []
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(lab)]))
    for s, e in zip(bounds[:-1], bounds[1:]):
        if lab[s] in ("H", "A"):
            spells.append((s, e))
    cats = {"dead-dead (short live stretch)": 0, "dead-handover": 0, "handover-dead": 0,
            "handover-handover": 0}
    for s, e in spells:
        if e - s >= 2 * common.FPS:
            continue
        before = "dead" if s == 0 or lab[s - 1] in ("D", "X") else "handover"
        after = "dead" if e >= len(lab) or lab[e] in ("D", "X") else "handover"
        key = {("dead", "dead"): "dead-dead (short live stretch)",
               ("dead", "handover"): "dead-handover", ("handover", "dead"): "handover-dead",
               ("handover", "handover"): "handover-handover"}[(before, after)]
        cats[key] += 1
    n_short = sum(cats.values())
    print("  {} of {} possessions are under 2 s".format(n_short, len(spells)))
    for k, v in cats.items():
        print("    {:32s} {:4d}".format(k, v))

    print("\n== Hysteresis delay: frames from a winning streak's first frame to confirmation")
    delays, visible_share = [], []
    for s, e in (h1, h2):
        curr, st_team, st_len, st_start = None, None, 0, -1
        f_lab = lab
        for f in range(s, e + 1):
            if dead[f]:
                st_team, st_len = None, 0
                continue
            if curr is None or f_lab[f - 1] in ("D", "X"):
                curr = f_lab[f]
            if vis[f]:
                dh, da = ev["min_deff_h"][f], ev["min_deff_a"][f]
                ctrl = "H" if (dh < da and dh < 3.0) else ("A" if (da < dh and da < 3.0) else None)
                if ctrl is not None:
                    if ctrl == st_team:
                        st_len += 1
                    else:
                        st_team, st_len, st_start = ctrl, 1, f
            if f_lab[f] != curr and f_lab[f] in ("H", "A"):
                if st_team == f_lab[f] and st_start >= 0:
                    delays.append(f - st_start)
                    visible_share.append(vis[st_start:f].mean())
                curr = f_lab[f]
    delays = np.array(delays)
    print("  {} hand-overs; delay median {:.0f} frames, p90 {:.0f}, max {}; ball visible on "
          "{:.0%} of delay frames".format(len(delays), np.median(delays),
                                          np.percentile(delays, 90), delays.max(),
                                          np.mean(visible_share)))

    print("\n== Restarts")
    found = flipped = total = 0
    fusion = common.apex.ReliabilityAwareFusionAndRefinement()
    for s, e in stretches:
        total += 1
        half_end = h1[1] if s <= h1[1] else h2[1]
        team = fusion._restart_team(s, half_end, dead, vis, ev["min_dh"], ev["min_da"])
        if team is not None:
            found += 1
            seg = lab[s:min(e, s + 3 * common.FPS)]
            if (seg != team).any():
                flipped += 1
    print("  {} restarts; team found for {}; of those, {} change team within 3 s "
          "(a sign the restart guess was wrong)".format(total, found, flipped))


if __name__ == "__main__":
    main()
