"""Research decoder: the frozen speed-gate pipeline plus experiment options A and B.

Works on any match prepared by prepare(): BAR-ATH (the competition match) or an IDSSE match.
With every option at its default it reproduces the submitted file exactly (tested).

B. Boundary placement (``boundary``) -- only *moves* a hand-over the hysteresis already confirmed:
   "confirm"      label changes at confirmation (the submitted behaviour)
   "start"        from the first frame of the winning streak
   "bounded:M"    from the streak start, but at most M frames before confirmation
   "mid"          midway between the old team's last control frame and the streak start
   Confirmation length itself is ``switch_frames`` (18 in the submission).

A. Evidence confidence:
   ``margin``     a frame counts as control only if the two teams' approach-adjusted distances
                  differ by at least this many metres (contested frames carry no evidence)
   ``jump_guard`` frames within this many frames of an implausible raw-ball jump (> 30 m/s
                  between consecutive detections: likely a detector identity swap) carry none
"""

import numpy as np

import apex_architecture_model as apex

H, A = 1, -1
LETTER = {H: "H", A: "A"}
JUMP_SPEED = 30.0


def prepare(m, control_speed=8.0):
    """Pipeline signals for a match dict with p_pos, raw_bx, raw_by, mean_speed, track_teams, pitch."""
    length, width = float(m["pitch"][0]), float(m["pitch"][1])
    n = int(m["n_frames"])
    analyzer = apex.TrackQualityAndMotionAnalyzer(length, width)
    bx, by, vis, out = analyzer.ball_track(m["raw_bx"], m["raw_by"])
    tt, p = m["track_teams"], m["p_pos"]
    spat_h, spat_a, min_dh, min_da = apex.SpatialExpert(tt).compute(p, bx, by, vis, n)
    deff_h, deff_a = apex.MotionExpert(tt).compute(p, bx, by, vis, n)
    gat_h, gat_a = apex.InteractionExpert(tt).compute(deff_h, deff_a, vis, n)
    dead = apex.PhysicsExpert().detect_stoppage(m["mean_speed"], out, n)
    cvis = apex.controllable(bx, by, vis, control_speed)
    return dict(n=n, bx=bx, by=by, vis=vis, cvis=cvis, dead=dead, min_dh=min_dh, min_da=min_da,
                deff_h=deff_h, deff_a=deff_a, spat_h=spat_h, spat_a=spat_a, gat_h=gat_h,
                gat_a=gat_a, raw_bx=m["raw_bx"], raw_by=m["raw_by"])


def control(prep, margin=0.0, jump_guard=0, radius=3.0):
    """+1 / -1 / 0 per frame: the submitted controller, optionally confidence-gated."""
    dh, da = prep["deff_h"], prep["deff_a"]
    hs = 0.5 * prep["spat_h"] + 0.5 * prep["gat_h"]
    as_ = 0.5 * prep["spat_a"] + 0.5 * prep["gat_a"]
    home = ((dh < da) | (hs > as_)) & (dh < radius)
    away = ~home & ((da < dh) | (as_ > hs)) & (da < radius)
    c = np.zeros(prep["n"], dtype=np.int8)
    c[home] = H
    c[away] = A
    c[~prep["cvis"]] = 0
    if margin > 0:
        c[np.abs(da - dh) < margin] = 0
    if jump_guard > 0:
        c[jump_frames(prep["raw_bx"], prep["raw_by"], jump_guard)] = 0
    return c


def jump_frames(raw_bx, raw_by, guard):
    """Mask of frames within ``guard`` of a raw-ball jump faster than JUMP_SPEED."""
    seen = np.flatnonzero(~np.isnan(raw_bx))
    step = np.hypot(np.diff(raw_bx[seen]), np.diff(raw_by[seen])) / (np.diff(seen) / 25.0)
    jumps = seen[1:][step > JUMP_SPEED]
    mask = np.zeros(len(raw_bx), bool)
    for j in jumps:
        mask[max(j - guard, 0):j + guard + 1] = True
    return mask


def restart_team(f, e_f, dead, vis, min_dh, min_da, radius=3.0, lookahead=50):
    for g in range(f, min(f + lookahead, e_f + 1)):
        if dead[g]:
            return None
        if vis[g] and min(min_dh[g], min_da[g]) < radius:
            return H if min_dh[g] <= min_da[g] else A
    return None


def decode(prep, segments, ctrl, switch_frames=18, boundary="confirm", prefill=()):
    """Possession labels. segments: [(first, last, fallback_team)] inclusive."""
    n, dead, vis = prep["n"], prep["dead"], prep["cvis"]
    labels = np.full(n, "X", dtype=object)
    for s, e, lbl in prefill:
        labels[s:e] = lbl
    bound = None
    if boundary.startswith("bounded:"):
        bound = int(boundary.split(":")[1])

    for s_f, e_f, fallback in segments:
        curr = fallback
        streak_team, streak_len, streak_start = 0, 0, -1
        last_own = -1
        after_stoppage = True
        for f in range(s_f, e_f + 1):
            if dead[f]:
                labels[f] = "D"
                streak_team, streak_len = 0, 0
                after_stoppage = True
                continue
            if after_stoppage:
                after_stoppage = False
                team = restart_team(f, e_f, dead, vis, prep["min_dh"], prep["min_da"])
                if team is not None:
                    curr = team
                last_own = f
            if vis[f]:
                c = int(ctrl[f])
                if c == curr:
                    last_own = f
                if c != 0:
                    if c == streak_team:
                        streak_len += 1
                    else:
                        streak_team, streak_len, streak_start = c, 1, f
                if streak_team != 0 and streak_team != curr and streak_len >= switch_frames:
                    curr = streak_team
                    if boundary == "start":
                        labels[streak_start:f] = LETTER[curr]
                    elif bound is not None:
                        labels[max(streak_start, f - bound):f] = LETTER[curr]
                    elif boundary == "mid":
                        labels[(last_own + 1 + streak_start) // 2:f] = LETTER[curr]
                    streak_len = 0
                    last_own = f
            labels[f] = LETTER[curr]
    return labels.astype(str)
