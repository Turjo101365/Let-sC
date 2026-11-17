"""Isolated possession experiments, each replacing exactly one stage of the submitted pipeline.

    python experiments/variants.py            # writes data/cache/variants/*.csv and a timing table
    python evaluation/self_check.py data/cache/variants/*.csv --baseline data/cache/variants/base.csv

No variant reads any ground truth. Selection uses evaluation/self_check.py only.

Experiments
-----------
gapN       Trajectory reconstruction: bridge whole ball gaps of up to N frames linearly, but only
           when the implied ball speed across the gap is physically plausible (<= 30 m/s).
speedV     Event-aware control: a team can only be "controlling" on frames where the ball moves
           at most V m/s over the ground. A ball travelling faster than a sprinting player is a
           pass, shot or clearance in flight, not a ball under someone's control.
restartM   Restart detection by majority: the team with more control frames in the first
           M seconds of live play takes the restart, instead of the single first sighting.
hmmK       Viterbi segmentation: inside each live stretch, choose the H/A sequence maximising
           (control evidence for the chosen team) - K per hand-over. Evidence is +1/-1 on frames
           where a team controls the ball (same definition as the baseline), 0 otherwise. The
           stretch's first team is free, so restarts are decided by the same evidence. Hand-overs
           are placed at the middle of the evidence-free gap between the two teams' control.
"""

import hashlib
import sys

import numpy as np
import pandas as pd

import common

MAX_GAP_SPEED = 30.0  # m/s; faster than any shot in open play over a whole gap


# ----------------------------------------------------------------------------------------------
# Experiment 1: trajectory reconstruction
# ----------------------------------------------------------------------------------------------
def bridged_track(raw, max_gap):
    bx = raw["raw_bx"].copy()
    by = raw["raw_by"].copy()
    missing = np.isnan(bx)
    d = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        n = e - s
        if n > max_gap or s == 0 or e >= len(bx):
            continue
        x0, y0, x1, y1 = bx[s - 1], by[s - 1], bx[e], by[e]
        if np.hypot(x1 - x0, y1 - y0) / ((n + 1) / common.FPS) > MAX_GAP_SPEED:
            continue
        w = (np.arange(1, n + 1, dtype=np.float32) / np.float32(n + 1))
        bx[s:e] = x0 + (x1 - x0) * w
        by[s:e] = y0 + (y1 - y0) * w
    # Everything else exactly as the baseline (including its 4-frame bridge on longer gaps).
    raw2 = dict(raw, raw_bx=bx, raw_by=by)
    return common.baseline_track(raw2)


# ----------------------------------------------------------------------------------------------
# Experiment 2: event-aware control
# ----------------------------------------------------------------------------------------------
def ball_speed(bx, by):
    """Ground speed (m/s); the pipeline's own function, so experiment and pipeline agree."""
    return common.apex.ball_ground_speed(bx, by)


def speed_gated(ev, bx, by, vmax):
    """Evidence with fast-ball frames treated as 'nobody in control' (ball_vis False)."""
    v = ball_speed(bx, by)
    fast = v > vmax  # NaN compares False: unknown speed keeps the frame
    return dict(ev, ball_vis=ev["ball_vis"] & ~fast)


# ----------------------------------------------------------------------------------------------
# Experiment 3: restart majority
# ----------------------------------------------------------------------------------------------
class MajorityRestart(common.apex.ReliabilityAwareFusionAndRefinement):
    """Restart team = the team with more control frames in the first ``window`` frames."""

    def __init__(self, window, **kw):
        super().__init__(**kw)
        self.window = window
        self._ev = None

    def bind(self, ev):
        self._ev = ev
        return self

    def _restart_team(self, f, e_f, dead_mask, ball_vis, min_dh, min_da):
        ev = self._ev
        h = a = 0
        for g in range(f, min(f + self.window, e_f + 1)):
            if dead_mask[g]:
                break
            if ball_vis[g]:
                c = self._controller(g, ev["spat_h"], ev["spat_a"], ev["min_deff_h"],
                                     ev["min_deff_a"], ev["gat_h"], ev["gat_a"])
                h += c == "H"
                a += c == "A"
        if h == a:
            return None
        return "H" if h > a else "A"


# ----------------------------------------------------------------------------------------------
# Experiment 4: Viterbi segmentation
# ----------------------------------------------------------------------------------------------
def control_evidence(ev, radius=3.0):
    """+1 where home controls, -1 where away controls, 0 otherwise (baseline's controller)."""
    dh, da = ev["min_deff_h"], ev["min_deff_a"]
    hs = 0.5 * ev["spat_h"] + 0.5 * ev["gat_h"]
    as_ = 0.5 * ev["spat_a"] + 0.5 * ev["gat_a"]
    home = ((dh < da) | (hs > as_)) & (dh < radius)
    away = ~home & ((da < dh) | (as_ > hs)) & (da < radius)
    e = np.zeros(len(dh), dtype=np.int8)
    e[home] = 1
    e[away] = -1
    e[~ev["ball_vis"]] = 0
    return e


def viterbi_stretch(e, K, start=None):
    """Best H(+1)/A(-1) sequence for evidence e with switch cost K.

    Ties resolve to staying in the forward pass, which puts a hand-over at the start of the
    evidence-free gap before the new team's first control frame; midpoint_boundaries() then
    moves it to the middle of that gap.
    """
    n = len(e)
    score = np.array([0.0, 0.0])  # [H, A]
    if start is not None:  # restart rule as a prior: starting on the other team costs a switch
        score[1 - start] = -K
    back = np.zeros((n, 2), dtype=bool)  # True = came from the other state
    gain = np.stack([e, -e], axis=1).astype(float)
    for t in range(n):
        if t:
            stay_h, cross_h = score[0], score[1] - K
            stay_a, cross_a = score[1], score[0] - K
            back[t, 0] = cross_h > stay_h
            back[t, 1] = cross_a > stay_a
            score = np.array([max(stay_h, cross_h), max(stay_a, cross_a)])
        score = score + gain[t]
    state = np.empty(n, dtype=np.int8)
    s = 0 if score[0] >= score[1] else 1
    for t in range(n - 1, -1, -1):
        state[t] = s
        if back[t, s]:
            s = 1 - s
    return state  # 0 = H, 1 = A


def midpoint_boundaries(state, e):
    """Move each hand-over to the middle of the evidence-free gap around it."""
    out = state.copy()
    change = np.flatnonzero(state[1:] != state[:-1]) + 1
    for b in change:
        old = 1 if state[b - 1] == 0 else -1
        prior = np.flatnonzero(e[:b] == old)
        left = prior[-1] + 1 if len(prior) else b
        nxt = np.flatnonzero(e[b:] == -old)
        right = b + nxt[0] if len(nxt) else b
        mid = (left + right) // 2
        if mid > b:
            out[b:mid] = state[b - 1]
        elif mid < b:
            out[mid:b] = state[b]
    return out


def hmm_labels(raw, ev, dead, K, midpoint=True, restart=False):
    h1, h2, h2_track_start = common.halves()
    labels = np.full(raw["n_frames"], "X", dtype=object)
    labels[h2_track_start:h2[0]] = "D"
    e_all = control_evidence(ev)
    for s_f, e_f in (h1, h2):
        seg_dead = dead[s_f:e_f + 1]
        labels[s_f:e_f + 1][seg_dead] = "D"
        d = np.diff(np.concatenate(([1], seg_dead.astype(np.int8), [1])))
        for a, b in zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)):
            e = e_all[s_f + a:s_f + b]
            start = None
            if restart:
                team = _RESTART._restart_team(s_f + a, e_f, dead, ev["ball_vis"], ev["min_dh"],
                                              ev["min_da"])
                start = None if team is None else (0 if team == "H" else 1)
            st = viterbi_stretch(e, K, start)
            if midpoint:
                st = midpoint_boundaries(st, e)
            labels[s_f + a:s_f + b] = np.where(st == 0, "H", "A")
    return labels


_RESTART = common.apex.ReliabilityAwareFusionAndRefinement()


# ----------------------------------------------------------------------------------------------
def main():
    raw = common.load_raw()
    bx, by, vis, out = common.baseline_track(raw)
    ev = common.evidence(raw, bx, by, vis)
    dead = common.dead_mask(raw, out)

    rows = []

    def emit(name, fn):
        with common.Timer() as t:
            labels = fn()
        path = common.write_labels(name, labels)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
        rows.append((name, round(t.seconds, 2), digest))

    emit("base", lambda: common.baseline_labels(raw, ev, dead))

    for n in (12, 25):
        def gap_variant(n=n):
            gx, gy, gvis, gout = bridged_track(raw, n)
            return common.baseline_labels(raw, common.evidence(raw, gx, gy, gvis),
                                          common.dead_mask(raw, gout))
        emit("gap%d" % n, gap_variant)

    for v in (6, 8, 10):
        emit("speed%d" % v, lambda v=v: common.baseline_labels(
            raw, speed_gated(ev, bx, by, v), dead))

    for m in (2, 3):
        def restart_variant(m=m):
            fusion = MajorityRestart(m * common.FPS).bind(ev)
            return fusion.refine_and_predict(dead, ev["spat_h"], ev["spat_a"], ev["min_dh"],
                                             ev["min_da"], ev["min_deff_h"], ev["min_deff_a"],
                                             ev["gat_h"], ev["gat_a"], ev["ball_vis"],
                                             raw["n_frames"])
        emit("restart%d" % m, restart_variant)

    for K in (9, 18, 25, 37, 50, 75):
        emit("hmm%d" % K, lambda K=K: hmm_labels(raw, ev, dead, K))
    emit("hmm25_nomid", lambda: hmm_labels(raw, ev, dead, 25, midpoint=False))
    emit("hmm25_r", lambda: hmm_labels(raw, ev, dead, 25, restart=True))

    # Combinations with the event-aware speed gate
    for v in (6, 8, 10):
        g = speed_gated(ev, bx, by, v)
        emit("speed%d_backdate" % v, lambda g=g: common.baseline_labels(raw, g, dead,
                                                                        backdate=True))
        for K in (18, 25, 37):
            emit("speed%d_hmm%d_r" % (v, K), lambda g=g, K=K: hmm_labels(raw, g, dead, K,
                                                                         restart=True))

    print(pd.DataFrame(rows, columns=["variant", "seconds", "sha256"]).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
