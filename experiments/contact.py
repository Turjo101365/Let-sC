"""Contact-aware possession: touches read from the ball's trajectory, on top of the speed gate.

The submitted model (speed gate + hysteresis + restart rule) cannot see a fast ball being
intercepted: control evidence only exists once the ball is under 8 m/s. This module detects
*touches* from the ball's own path and uses them to separate four situations:

  fly-by        a fast ball passes near an opponent with no change in its path   -> no evidence
  interception  an opponent's touch kills the ball (it leaves the touch slow) and
                that team out-controls the old one over the next second           -> hand-over at the touch
  deflection    an opponent's touch redirects a fast ball, and the next touch is
                by the original team                                              -> no change
  one-touch     an opponent's touch redirects it, and the next touch (within 3 s)
                is by the same opponent team (two consecutive touches)            -> hand-over at the first
  teammate      a touch by the team in possession resets any opponent streak     -> possession confirmed
  loose ball    touches where both teams are within ``margin`` of each other are
                ambiguous and ignored; possession stays where it was

Touch detection
---------------
On frames where the ball arrives fast (speed k frames earlier above ``v_fast``) and a player is
within ``r_contact``, a touch is a change in the ball's velocity vector, |v(t+k) - v(t-k)|, larger
than ``tau`` times its incoming speed. ``tau`` is not chosen by hand: it is the 99th percentile of
the same relative change for fast balls with no player within 5 m, i.e. the tracking-noise floor.
One touch is kept per burst of candidate frames (the strongest).

Nothing here reads a label of any kind.
"""

import numpy as np
import pandas as pd

H, A = 1, -1


# ----------------------------------------------------------------------------------------------
# Touches
# ----------------------------------------------------------------------------------------------
def ball_velocity(bx, by, fps=25):
    """Velocity (m/s) from a 5-frame centred average of position, as the speed gate uses."""
    sx = pd.Series(bx).rolling(5, center=True, min_periods=5).mean().values
    sy = pd.Series(by).rolling(5, center=True, min_periods=5).mean().values
    vx = np.full(len(bx), np.nan)
    vy = np.full(len(bx), np.nan)
    vx[1:] = np.diff(sx) * fps
    vy[1:] = np.diff(sy) * fps
    return vx, vy


def trajectory_change(bx, by, k=3):
    """(incoming speed, outgoing speed, |velocity change|) around every frame."""
    vx, vy = ball_velocity(bx, by)
    n = len(bx)
    before = np.full(n, np.nan)
    after = np.full(n, np.nan)
    dv = np.full(n, np.nan)
    before[k:] = np.hypot(vx[:-k], vy[:-k])
    after[:-k] = np.hypot(vx[k:], vy[k:])
    dv[k:-k] = np.hypot(vx[2 * k:] - vx[:-2 * k], vy[2 * k:] - vy[:-2 * k])
    return before, after, dv


def noise_floor(bx, by, min_dh, min_da, mask, v_fast=8.0, far=5.0, pct=99.0, k=3):
    """Relative velocity change of fast balls with nobody within ``far`` m: the noise level."""
    before, _, dv = trajectory_change(bx, by, k)
    nearest = np.minimum(min_dh, min_da)
    sel = mask & (before > v_fast) & (nearest > far) & (nearest < 900) & ~np.isnan(dv)
    return float(np.percentile(dv[sel] / before[sel], pct)), int(sel.sum())


def detect_touches(bx, by, vis, min_dh, min_da, tau, v_fast=8.0, r_contact=2.0, margin=0.5,
                   k=3, merge_gap=2):
    """[(frame, team, kind, ambiguous)] with team H/A, kind 'control' or 'redirect'."""
    before, after, dv = trajectory_change(bx, by, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = dv / before
    nearest = np.minimum(min_dh, min_da)
    cand = vis & (before > v_fast) & (nearest < r_contact) & (rel > tau)
    cand &= ~np.isnan(rel)
    idx = np.flatnonzero(cand)
    touches = []
    if not len(idx):
        return touches
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > merge_gap + 1) + 1)
    for g in groups:
        t = int(g[np.argmax(rel[g])])
        team = H if min_dh[t] <= min_da[t] else A
        ambiguous = bool(abs(min_dh[t] - min_da[t]) < margin)
        kind = "control" if (not np.isnan(after[t]) and after[t] <= v_fast) else "redirect"
        touches.append((t, team, kind, ambiguous))
    return touches


# ----------------------------------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------------------------------
def control_array(ev, radius=3.0):
    """+1/-1/0 per frame with exactly the submitted pipeline's controller definition."""
    dh, da = ev["min_deff_h"], ev["min_deff_a"]
    hs = 0.5 * ev["spat_h"] + 0.5 * ev["gat_h"]
    as_ = 0.5 * ev["spat_a"] + 0.5 * ev["gat_a"]
    home = ((dh < da) | (hs > as_)) & (dh < radius)
    away = ~home & ((da < dh) | (as_ > hs)) & (da < radius)
    c = np.zeros(len(dh), dtype=np.int8)
    c[home] = H
    c[away] = A
    c[~ev["ball_vis"]] = 0
    return c


def restart_team(f, e_f, dead, vis, min_dh, min_da, radius=3.0, lookahead=50):
    """The submitted restart rule: team nearest the ball at its first sighting within 2 s."""
    for g in range(f, min(f + lookahead, e_f + 1)):
        if dead[g]:
            return None
        if vis[g] and min(min_dh[g], min_da[g]) < radius:
            return H if min_dh[g] <= min_da[g] else A
    return None


def decode(segments, dead, ctrl, vis, min_dh, min_da, n_frames, touches=(), switch_frames=18,
           intercept=False, consecutive=False, teammate=False, confirm_window=25,
           next_touch_window=75, restart_override=None, prefill=None):
    """Possession labels. With every contact option off this is the submitted hysteresis.

    segments          [(first, last, fallback_team)] of live play, inclusive
    ctrl              control_array() output (already speed-gated through ``vis``)
    touches           detect_touches() output
    restart_override  {stretch_start_frame: team} from another source (set-piece tags)
    prefill           [(start, end, label)] written before decoding (e.g. pre-kick-off D)
    """
    labels = np.full(n_frames, "X", dtype=object)
    for s, e, lbl in prefill or ():
        labels[s:e] = lbl
    touch_at = {t: (team, kind, amb) for t, team, kind, amb in touches}
    clear = [(t, team) for t, team, _, amb in touches if not amb]
    clear_frames = np.array([t for t, _ in clear], dtype=np.int64)
    letter = {H: "H", A: "A"}

    for s_f, e_f, fallback in segments:
        curr = fallback
        streak_team, streak_len = 0, 0
        after_stoppage = True
        f = s_f
        while f <= e_f:
            if dead[f]:
                labels[f] = "D"
                streak_team, streak_len = 0, 0
                after_stoppage = True
                f += 1
                continue
            if after_stoppage:
                after_stoppage = False
                team = None
                if restart_override and f in restart_override:
                    team = restart_override[f]
                if team is None:
                    team = restart_team(f, e_f, dead, vis, min_dh, min_da)
                if team is not None:
                    curr = team

            if f in touch_at and (intercept or consecutive or teammate):
                team, kind, amb = touch_at[f]
                if not amb:
                    if team == curr and teammate:
                        streak_team, streak_len = 0, 0
                    elif team != curr:
                        switch = False
                        end = min(f + confirm_window, e_f + 1)
                        if intercept and kind == "control":
                            window = ctrl[f:end]
                            window = window[~dead[f:end]]
                            switch = (window == team).sum() > (window == curr).sum()
                        if not switch and consecutive and kind == "redirect":
                            later = clear_frames[(clear_frames > f)
                                                 & (clear_frames <= f + next_touch_window)]
                            if len(later):
                                nxt = later[0]
                                if not dead[f:nxt + 1].any():
                                    switch = touch_at[int(nxt)][0] == team
                        if switch:
                            curr = team
                            streak_team, streak_len = 0, 0

            if vis[f]:
                c = ctrl[f]
                if c != 0:
                    if c == streak_team:
                        streak_len += 1
                    else:
                        streak_team, streak_len = c, 1
                if streak_team != 0 and streak_team != curr and streak_len >= switch_frames:
                    curr = streak_team
                    streak_len = 0
            labels[f] = letter[curr]
            f += 1
    return labels


# ----------------------------------------------------------------------------------------------
# Set-piece restarts from Smart Tagging
# ----------------------------------------------------------------------------------------------
RESTART_TYPES = {"corner kick", "goal kick", "free kick"}
TAG_TEAM = {"barcelona": H, "atletic club": A}


def set_piece_tags(tags):
    out = []
    for t in tags:
        typ = (t["tags"].get("Type") or "").strip().lower()
        team = TAG_TEAM.get((t["tags"].get("Team") or "").strip().lower())
        if t["code"].strip().lower() == "set pieces" and typ in RESTART_TYPES and team:
            out.append(t)
    return out


def tag_restarts(tags, stretch_starts, after=125, before=50):
    """{stretch_start: team} for live stretches that begin inside a team-tagged set piece."""
    starts = np.asarray(sorted(stretch_starts))
    override = {}
    for t in tags:
        team = TAG_TEAM[(t["tags"].get("Team") or "").strip().lower()]
        lo, hi = t["start_frame"] - before, t["end_frame"] + after
        inside = starts[(starts >= lo) & (starts <= hi)]
        if len(inside):
            override[int(inside[0])] = team
    return override
