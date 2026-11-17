"""Independent labelled matches: the IDSSE Bundesliga dataset, read into our pipeline's inputs.

Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data. figshare DOI 10.6084/m9.figshare.28196177, CC-BY 4.0, data owner
Deutsche Fussball Liga. Seven 2022/23 Bundesliga matches, TRACAB Gen5 tracking at 25 Hz.

Every ball frame carries BallStatus (0 dead, 1 alive) and BallPossession (1 home, 2 away),
produced by the official data provider for *other* games. RULES.md allows labelled data about
other games in any amount. Ground truth here is built exactly as the competition describes its
own: D where the ball is dead, otherwise the team the tracking names as in possession.

The files go in data/external/idsse/ (gitignored, not redistributed). Parsed matches are cached
as .npz in data/cache/idsse/.
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "external", "idsse")
CACHE_DIR = os.path.join(ROOT, "data", "cache", "idsse")
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
CV = MATCHES[:5]    # leave-one-match-out cross-validation: all development and selection
TEST = MATCHES[5:]  # untouched until the model is frozen; evaluated exactly once
DEV = CV            # backwards-compatible name
HALF_GAP = 500      # ungraded X frames inserted between the halves
BALL_TEAMS = {"ball", "BALL", "Ball"}


def _file(kind, match):
    hits = glob.glob(os.path.join(RAW_DIR, "DFL_*_{}_*DFL-MAT-{}.xml".format(kind, match)))
    if not hits:
        raise IOError("IDSSE {} file for {} not found in {}".format(kind, match, RAW_DIR))
    return hits[0]


def match_info(match):
    root = ET.parse(_file("matchinformation", match)).getroot()
    g = root.find(".//General")
    env = root.find(".//Environment")
    return {"home": g.get("HomeTeamId"), "away": g.get("GuestTeamId"),
            "pitch": (float(env.get("PitchX")), float(env.get("PitchY")))}


def parse_positions(match):
    """{half: {"players": [(team, n, x, y)], "ball": (n, x, y, z, poss, status)}} raw arrays."""
    halves = {}
    current = None
    buf = None
    for event, el in ET.iterparse(_file("positions_raw_observed", match), events=("start", "end")):
        tag = el.tag
        if event == "start" and tag == "FrameSet":
            current = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
            buf = {"n": [], "x": [], "y": [], "z": [], "p": [], "s": []}
        elif event == "end" and tag == "Frame":
            buf["n"].append(int(el.get("N")))
            buf["x"].append(float(el.get("X")))
            buf["y"].append(float(el.get("Y")))
            if current[1] in BALL_TEAMS:
                buf["z"].append(float(el.get("Z", "nan")))
                buf["p"].append(int(el.get("BallPossession", "0")))
                buf["s"].append(int(el.get("BallStatus", "0")))
            el.clear()
        elif event == "end" and tag == "FrameSet":
            half, team, _ = current
            h = halves.setdefault(half, {"players": [], "ball": None})
            arr = {k: np.asarray(v) for k, v in buf.items()}
            if team in BALL_TEAMS:
                h["ball"] = (arr["n"], arr["x"], arr["y"], arr["z"], arr["p"], arr["s"])
            elif team != "referee":
                h["players"].append((team, arr["n"], arr["x"], arr["y"]))
            el.clear()
    return halves


def build(match):
    """Our pipeline's inputs on one timeline: half 1, an ungraded gap, half 2.

    Returns p_pos (frames x slots x 2, metres from the corner like the ATD), raw ball x/y, mean
    player speed (m/s, from positions), slot teams (0 home / 1 away), ground-truth labels
    (H/A/D on graded frames, X elsewhere), the two half segments and the pitch size.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, match + ".npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}

    info = match_info(match)
    length, width = info["pitch"]
    halves = parse_positions(match)
    order = [h for h in ("firstHalf", "secondHalf") if h in halves]

    # Slots are (half, FrameSet) pairs: a player who plays both halves takes two slots, which the
    # pipeline does not mind (it only uses team membership and positions, like the ATD's slots).
    blocks, offset = [], 0
    n_slots = sum(len(halves[h]["players"]) for h in order)
    for h in order:
        bn = halves[h]["ball"][0]
        start, stop = int(bn.min()), int(bn.max()) + 1
        blocks.append((h, start, stop, offset))
        offset += (stop - start) + HALF_GAP  # each half is followed by an ungraded gap
    n_frames = offset

    p_pos = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    bz = np.full(n_frames, np.nan, dtype=np.float32)
    gt = np.full(n_frames, "X", dtype="<U1")
    teams = np.zeros(n_slots, dtype=np.int8)
    segments = []
    slot = 0
    for h, start, stop, off in blocks:
        for team, n, x, y in halves[h]["players"]:
            idx = n - start + off
            ok = (n >= start) & (n < stop)
            p_pos[idx[ok], slot, 0] = x[ok] + length / 2
            p_pos[idx[ok], slot, 1] = y[ok] + width / 2
            teams[slot] = 0 if team == info["home"] else 1
            slot += 1
        n, x, y, zz, poss, status = halves[h]["ball"]
        idx = n - start + off
        bx[idx] = x + length / 2
        by[idx] = y + width / 2
        bz[idx] = zz
        lab = np.where(status == 0, "D", np.where(poss == 1, "H", np.where(poss == 2, "A", "D")))
        gt[idx] = lab
        segments.append((off, off + (stop - start) - 1))

    # Mean player speed from positions (5-frame centred difference), as a stand-in for the ATD's
    # speed channel.
    vel = np.full(p_pos.shape[:2], np.nan, dtype=np.float32)
    vel[2:-2] = np.hypot(p_pos[4:, :, 0] - p_pos[:-4, :, 0], p_pos[4:, :, 1] - p_pos[:-4, :, 1]) * 25 / 4
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(np.isnan(vel).all(1), 0, np.nanmean(np.where(np.isnan(vel), np.nan, vel), 1))
    mean_speed = np.nan_to_num(mean_speed).astype(np.float32)

    out = dict(p_pos=p_pos, raw_bx=bx, raw_by=by, raw_bz=bz, mean_speed=mean_speed,
               track_teams=teams, gt=gt, segments=np.asarray(segments, dtype=np.int64),
               pitch=np.asarray([length, width]), n_frames=np.asarray(n_frames))
    np.savez_compressed(cache, **out)
    return out
