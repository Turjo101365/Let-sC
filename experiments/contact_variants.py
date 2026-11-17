"""Contact-aware and tag-restart variants, against the submitted speed-gate model.

    python experiments/contact_variants.py

Writes label files to data/cache/contact/ (gitignored) and prints every comparison. No ground
truth is read. Inputs vs validation:

- ATTACKING-phase tags (BUILD UP, PROGRESSION, ATTACKING TRANSITION, CREATING CHANCES) are never
  model inputs, so team agreement and transition hits are held out for every variant.
- SET PIECES tags are inputs to the ``tagrestart`` variants. Their restart accuracy is measured by
  2-fold cross-validation by half: the model run with first-half tags is scored on second-half
  tags and vice versa. The full-input number is printed too, marked as not independent.
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

import common
import contact
sys.path.insert(0, os.path.join(common.ROOT, "evaluation"))
import self_check  # noqa: E402

OUT = os.path.join(common.ROOT, "data", "cache", "contact")


def handovers(labels):
    alive = labels[(labels == "H") | (labels == "A")]
    return int((alive[1:] != alive[:-1]).sum())


def main():
    raw = common.load_raw()
    bx, by, vis, out = common.baseline_track(raw)
    ev = common.evidence(raw, bx, by, vis)
    dead = common.dead_mask(raw, out)
    cvis = common.apex.controllable(bx, by, vis, 8.0)
    ctrl = contact.control_array(dict(ev, ball_vis=cvis))
    h1, h2, h2t = common.halves()
    segs = [(h1[0], h1[1], contact.A), (h2[0], h2[1], contact.H)]
    live = np.zeros(raw["n_frames"], bool)
    for s, e, _ in segs:
        live[s:e + 1] = True
    live &= ~dead

    # Tracking-noise floor for touches, measured on this match without labels.
    floors = {}
    for pct in (95, 99):
        floors[pct], n_ref = contact.noise_floor(bx, by, ev["min_dh"], ev["min_da"], live, pct=pct)
    print("noise floor of relative velocity change (fast ball, nobody within 5 m, n={}): "
          "p95 {:.3f}, p99 {:.3f}".format(n_ref, floors[95], floors[99]))

    touch_sets = {}
    for key, kw in {"t99_r2": dict(tau=floors[99]), "t95_r2": dict(tau=floors[95]),
                    "t99_r15": dict(tau=floors[99], r_contact=1.5),
                    "t99_r25": dict(tau=floors[99], r_contact=2.5)}.items():
        tt = contact.detect_touches(bx, by, vis, ev["min_dh"], ev["min_da"], **kw)
        tt = [x for x in tt if live[x[0]]]
        touch_sets[key] = tt
        kinds = pd.Series([k for _, _, k, _ in tt]).value_counts().to_dict()
        print("  touches {:8s} {:5d} in live play ({:.1f}/min): {}  contested {:.0%}".format(
            key, len(tt), len(tt) / (live.sum() / 1500), kinds,
            np.mean([a for *_, a in tt]) if tt else 0))

    # Live-stretch starts, for mapping set-piece tags to restarts.
    d = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    tags = self_check.load_tags()
    sp = contact.set_piece_tags(tags)
    fold = {1: [t for t in sp if t["start_frame"] < h2[0]], 2: [t for t in sp if t["start_frame"] >= h2[0]]}
    print("  team-tagged set pieces: {} (1st half {}, 2nd half {}); mapped to a restart: {}".format(
        len(sp), len(fold[1]), len(fold[2]), len(contact.tag_restarts(sp, starts))))

    def run(touches=(), override=None, **opts):
        return contact.decode(segs, dead, ctrl, cvis, ev["min_dh"], ev["min_da"], raw["n_frames"],
                              touches=touches, restart_override=override,
                              prefill=[(h2t, h2[0], "D")], **opts)

    full = dict(intercept=True, consecutive=True, teammate=True)
    variants = {
        "base": lambda o=None: run(override=o),
        "c_intercept": lambda o=None: run(touch_sets["t99_r2"], o, intercept=True),
        "c_consecutive": lambda o=None: run(touch_sets["t99_r2"], o, consecutive=True),
        "c_teammate": lambda o=None: run(touch_sets["t99_r2"], o, teammate=True),
        "c_all": lambda o=None: run(touch_sets["t99_r2"], o, **full),
        "c_all_tau95": lambda o=None: run(touch_sets["t95_r2"], o, **full),
        "c_all_r15": lambda o=None: run(touch_sets["t99_r15"], o, **full),
        "c_all_r25": lambda o=None: run(touch_sets["t99_r25"], o, **full),
    }

    os.makedirs(OUT, exist_ok=True)
    results, labels_of, cost = {}, {}, {}
    for name, fn in list(variants.items()):
        for tagged in (False, True):
            vname = name + ("+tagrestart" if tagged else "")
            if tagged and name not in ("base", "c_all"):
                continue
            with common.Timer() as t:
                lab = fn(contact.tag_restarts(sp, starts) if tagged else None)
            cost[vname] = t.seconds
            labels_of[vname] = np.array(lab, dtype=object).astype(str)
            path = os.path.join(OUT, vname + ".csv")
            with open(path, "w", newline="\n") as f:
                f.write("frame,label\n")
                f.writelines("{},{}\n".format(i, x) for i, x in enumerate(lab))
            ev_tags = self_check.per_tag(labels_of[vname], tags)
            res = self_check.summarise(ev_tags, labels_of[vname])
            res["handovers"] = handovers(labels_of[vname])
            res["seconds"] = round(cost[vname], 2)
            res["sha12"] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
            if tagged:
                # Cross-validated restart accuracy: tags of one half drive the model, the other
                # half's tags score it.
                hits = tot = 0
                for k_in, k_out in ((1, 2), (2, 1)):
                    lab_cv = np.array(fn(contact.tag_restarts(fold[k_in], starts)),
                                      dtype=object).astype(str)
                    r = self_check.per_tag(lab_cv, fold[k_out])["restart"]
                    hits += r.sum()
                    tot += len(r)
                res["restart_cv"] = hits / tot
                res["restart_agree(not indep.)"] = res.pop("restart_agree")
            results[vname] = (res, ev_tags)

    df = pd.DataFrame({k: v[0] for k, v in results.items()}).T
    cols = ["team_agree", "trans_hit", "trans_med", "restart_agree", "restart_cv",
            "restart_agree(not indep.)", "handovers", "pingpong", "short_2s", "possessions",
            "dur_p50", "switches_min", "seconds", "sha12"]
    df = df[[c for c in cols if c in df.columns]]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print("\n", df)

    base_ev = results["base"][1]
    print("\npaired bootstrap vs base (95% interval of variant - base)")
    for name, (_, ev_tags) in results.items():
        if name == "base":
            continue
        ci = self_check.paired_bootstrap(base_ev, ev_tags)
        print("  {:22s} ".format(name) + "  ".join(
            "{} [{:+.3f}, {:+.3f}]".format(k, lo, hi) for k, (lo, hi) in ci.items()))

    print("\ntagged transitions (24 ATTACKING TRANSITION tags): detected / changed vs base")
    at = [t for t in tags if self_check.code_of(t) == "attacking transition" and self_check.team_of(t)]
    bt = base_ev["trans"]
    for name, (_, ev_tags) in results.items():
        vt = ev_tags["trans"]
        gained = [i for i, (b, v) in enumerate(zip(bt, vt)) if b is None and v is not None]
        lost = [i for i, (b, v) in enumerate(zip(bt, vt)) if b is not None and v is None]
        moved = [(i, b, v) for i, (b, v) in enumerate(zip(bt, vt))
                 if b is not None and v is not None and b != v]
        print("  {:22s} detected {:2d}/24  gained {}  lost {}  timing changed {}".format(
            name, sum(v is not None for v in vt),
            [self_clock(at[i]) for i in gained], [self_clock(at[i]) for i in lost],
            ["{} {:+d}->{:+d}".format(self_clock(at[i]), b, v) for i, b, v in moved]))


def self_clock(tag):
    s = tag["start_frame"] / common.FPS
    return "{:d}:{:02d}".format(int(s // 60), int(s % 60))


if __name__ == "__main__":
    main()
