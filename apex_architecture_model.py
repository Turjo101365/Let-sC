"""APEX: rule-based possession inference from the tactical tracking.
Team: Let'sC
Competition: More than a Hack 2027 (Metrica Sports & FC Barcelona)

Nothing here is trained. Every stage is a fixed, hand-written rule over the automatic tracking:

  1. Load the tactical ATD (players, ball, speeds) on the submission's frame timeline.
  2. Ball track: bridge gaps of up to 4 frames by linear interpolation; flag out-of-pitch balls.
  3. Proximity evidence ("experts"), per ball-visible frame:
     - spatial:     nearest home and away player distance, mapped through a logistic (3 m scale)
     - motion:      the same distance shortened by each player's approach speed over 0.25 s
     - interaction: the two motion distances turned into a soft share (2.5 m scale)
  4. Dead ball: 50-frame mean player speed below 1.32 m/s for 60+ frames, or ball off the pitch.
  5. Possession: a team takes over after 18 consecutive frames as the controlling team (nearest,
     within 3 m, ball moving at most 8 m/s); every restart starts with the team nearest the ball.
  6. Write frame,label for all 199,319 frames, deterministically.

    python apex_architecture_model.py
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, GAMES, require, source_frames, half_video_blocks, anchor_video_frame


# ==============================================================================
# 1. INPUT DATA
# ==============================================================================
class ApexDataLoader:
    """Locates the tactical tracking files (the submission's own timeline)."""

    def __init__(self):
        self.tac_meta, self.tac_raw = require(["atd_meta", "atd_raw"], source="tactical")
        # The broadcast recording is on its own, unsynchronised timeline and is not used by any
        # stage below, so it is not required to run the pipeline.

    def get_team_slots(self):
        root = ET.parse(self.tac_meta).getroot()
        team_ids = [team.get("id") for team in root.iter("Team")]
        home_id = team_ids[0]
        slots = [player.get("teamId") for player in root.iter("Player")]
        return np.array([0 if s == home_id else 1 for s in slots], dtype=np.int8)


# ==============================================================================
# 2. BALL AND PLAYER TRACKS
# ==============================================================================
class TrackQualityAndMotionAnalyzer:
    """Parses the raw ATD and builds the ball track the rest of the pipeline uses."""

    def __init__(self, length, width, gap_limit=4):
        self.length = length
        self.width = width
        self.gap_limit = gap_limit

    def parse(self, raw_path, n_slots, n_frames):
        """Player positions (m), raw ball positions (m, NaN when missing), mean player speed."""
        ball_x = np.full(n_frames, np.nan, dtype=np.float32)
        ball_y = np.full(n_frames, np.nan, dtype=np.float32)
        mean_speed = np.zeros(n_frames, dtype=np.float32)
        p_pos = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

        with open(raw_path) as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) != 3:
                    continue
                fid = int(parts[0]) - 1
                if fid >= n_frames:
                    break

                speeds = []
                for slot, p_str in enumerate(parts[1].split(";")):
                    if p_str and p_str[0] != "N":
                        c1 = p_str.find(",")
                        c2 = p_str.find(",", c1 + 1)
                        px = float(p_str[:c1]) * self.length
                        py = float(p_str[c1 + 1 : c2]) * self.width
                        spd = p_str[c2 + 1 :]
                        if spd != "NaN":
                            speeds.append(float(spd))
                        p_pos[fid, slot, 0] = px
                        p_pos[fid, slot, 1] = py

                if speeds:
                    mean_speed[fid] = sum(speeds) / len(speeds)

                ball_str = parts[2]
                if ball_str and ball_str[0] != "N":
                    c1 = ball_str.find(",")
                    c2 = ball_str.find(",", c1 + 1)
                    ball_x[fid] = float(ball_str[:c1]) * self.length
                    ball_y[fid] = float(ball_str[c1 + 1 : c2]) * self.width

        return p_pos, ball_x, ball_y, mean_speed

    def ball_track(self, ball_x, ball_y):
        """Bridge ball gaps of up to ``gap_limit`` frames linearly; flag off-pitch frames."""
        s_x = pd.Series(ball_x).interpolate(method="linear", limit=self.gap_limit).values
        s_y = pd.Series(ball_y).interpolate(method="linear", limit=self.gap_limit).values
        ball_vis = ~np.isnan(s_x)
        ball_out = (s_x < 0) | (s_x > self.length) | (s_y < 0) | (s_y > self.width)
        return s_x, s_y, ball_vis, ball_out


def nearest_by_team(dist, track_teams, ball_vis):
    """Per-frame minimum of ``dist`` (frames x slots) over home and away slots; 999 if none."""
    out = []
    for team in (0, 1):
        d = np.where(np.isnan(dist[:, track_teams == team]), np.float32(np.inf),
                     dist[:, track_teams == team]).min(axis=1)
        d = np.where(np.isinf(d) | ~ball_vis, np.float32(999.0), d).astype(np.float32)
        out.append(d)
    return out


# ==============================================================================
# 3-4. PROXIMITY EVIDENCE AND DEAD-BALL RULE
# ==============================================================================
class SpatialExpert:
    """Nearest home/away player distance, mapped to a soft score with a logistic."""

    def __init__(self, track_teams, hidden_dim=64, sigma=3.0):
        self.track_teams = track_teams
        self.hidden_dim = hidden_dim
        self.sigma = sigma

    def compute(self, p_pos, bx, by, ball_vis, n_frames):
        dist = np.sqrt((p_pos[:, :, 0] - bx[:, None]) ** 2 + (p_pos[:, :, 1] - by[:, None]) ** 2)
        min_dh, min_da = nearest_by_team(dist, self.track_teams, ball_vis)
        delta = np.clip((min_da - min_dh) / self.sigma, -20.0, 20.0)
        p_h = 1.0 / (1.0 + np.exp(-delta))
        scores_h = np.where(ball_vis, p_h, 0.0).astype(np.float32)
        scores_a = np.where(ball_vis, 1.0 - p_h, 0.0).astype(np.float32)
        return scores_h, scores_a, min_dh, min_da


class MotionExpert:
    """Nearest distance shortened by each player's approach speed over ``tau`` seconds."""

    def __init__(self, track_teams, tau=0.25):
        self.track_teams = track_teams
        self.tau = tau

    def compute(self, p_pos, bx, by, ball_vis, n_frames):
        px, py = p_pos[:, :, 0], p_pos[:, :, 1]
        dx_b, dy_b = bx[:, None] - px, by[:, None] - py
        dist = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
        prev = np.empty_like(p_pos)
        prev[0] = np.nan  # frame 0 has no previous frame
        prev[1:] = p_pos[:-1]
        vx = (px - prev[:, :, 0]) * 25.0
        vy = (py - prev[:, :, 1]) * 25.0
        with np.errstate(invalid="ignore", divide="ignore"):
            v_app = (vx * dx_b + vy * dy_b) / dist
        v_app = np.where(np.isnan(prev[:, :, 0]) | ~(dist > 1e-3), np.float32(0.0), v_app)
        deff = np.maximum(np.float32(0.0), dist - self.tau * v_app)
        vis = ball_vis.copy()
        vis[0] = False  # the original loop starts at frame 1
        return nearest_by_team(deff, self.track_teams, vis)


class InteractionExpert:
    """Soft home/away share from the two approach-adjusted distances (exp(-d / sigma))."""

    def __init__(self, track_teams, sigma=2.5):
        self.track_teams = track_teams
        self.sigma = sigma

    def compute(self, min_deff_h, min_deff_a, ball_vis, n_frames):
        a_h = np.exp(-min_deff_h / self.sigma)
        a_a = np.exp(-min_deff_a / self.sigma)
        tot = a_h + a_a + 1e-6
        gat_h = np.where(ball_vis, a_h / tot, 0.0).astype(np.float32)
        gat_a = np.where(ball_vis, a_a / tot, 0.0).astype(np.float32)
        return gat_h, gat_a


class PhysicsExpert:
    """Dead-ball rule: sustained low collective player speed, or the ball off the pitch."""

    def __init__(self, speed_thresh=1.32, min_dead_frames=60):
        self.speed_thresh = speed_thresh
        self.min_dead_frames = min_dead_frames

    def detect_stoppage(self, mean_speed, ball_out, n_frames):
        # 50-frame temporal moving average of collective kinetic velocity
        kernel = np.ones(50) / 50.0
        smooth_speed = np.convolve(mean_speed, kernel, mode="same")

        is_dead = (smooth_speed < self.speed_thresh) | ball_out
        diffs = np.diff(np.concatenate(([0], is_dead.astype(np.int8), [0])))
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)

        dead_mask = np.zeros(n_frames, dtype=bool)
        for s, e in zip(starts, ends):
            if e - s >= self.min_dead_frames:
                dead_mask[s:e] = True
        return dead_mask


# ==============================================================================
# 5. POSSESSION SEQUENCE
# ==============================================================================
class ReliabilityAwareFusionAndRefinement:
    """Turns per-frame proximity evidence into a possession sequence.

    A team takes over only after it has been the controlling team (nearest, within
    ``control_radius``) on ``switch_frames`` consecutive ball-visible frames. Two refinements sit on
    top of that hysteresis. ``restart`` is on by default; ``backdate`` is off, because the
    label-free self-check (evaluation/self_check.py) was mixed on it -- see TECHNICAL_DOCUMENT.md:

    backdate  The file is produced offline, so once a hand-over is confirmed the frames since the
              opponent first took control are relabelled to the new team. Without this every
              hand-over is reported at least ``switch_frames`` frames late.
    restart   Every stretch of live play after a stoppage (and each half's kick-off) starts with the
              team of the player standing at the ball when it is first seen within
              ``restart_lookahead`` frames, instead of inheriting the team from before the stoppage.
    """

    def __init__(self, switch_frames=18, control_radius=3.0, backdate=False, restart=True,
                 restart_lookahead=50):
        self.switch_frames = switch_frames
        self.control_radius = control_radius
        self.backdate = backdate
        self.restart = restart
        self.restart_lookahead = restart_lookahead

    def _controller(self, f, spat_h, spat_a, min_deff_h, min_deff_a, gat_h, gat_a):
        dh, da = min_deff_h[f], min_deff_a[f]
        home_score = 0.50 * spat_h[f] + 0.50 * gat_h[f]
        away_score = 0.50 * spat_a[f] + 0.50 * gat_a[f]
        if (dh < da or home_score > away_score) and dh < self.control_radius:
            return "H"
        if (da < dh or away_score > home_score) and da < self.control_radius:
            return "A"
        return None

    def _restart_team(self, f, e_f, dead_mask, ball_vis, min_dh, min_da):
        """Team nearest the ball at the first sighting after a restart, or None."""
        for g in range(f, min(f + self.restart_lookahead, e_f + 1)):
            if dead_mask[g]:
                return None
            if ball_vis[g] and min(min_dh[g], min_da[g]) < self.control_radius:
                return "H" if min_dh[g] <= min_da[g] else "A"
        return None

    def refine_and_predict(
        self,
        dead_mask,
        spat_h,
        spat_a,
        min_dh,
        min_da,
        min_deff_h,
        min_deff_a,
        gat_h,
        gat_a,
        ball_vis,
        n_frames,
    ):
        labels = np.full(n_frames, "X", dtype=object)

        # Half boundaries and the second-half kick-off come from games.py
        blocks = half_video_blocks(GAME)
        h1_start, h1_end = blocks[0][1], blocks[0][2]
        h2_start, h2_end = blocks[1][1], blocks[1][2]
        kickoff_2nd = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])

        # Between 2nd half tracking start and kickoff whistle, the game is stopped (dead ball)
        labels[h2_start:kickoff_2nd] = "D"

        # Fallback kick-off teams, used only if the restart rule cannot see who kicks off.
        halves = [
            (h1_start, h1_end, "A"),
            (kickoff_2nd, h2_end, "H"),
        ]

        for s_f, e_f, init_team in halves:
            curr_team = init_team
            streak_team, streak_len, streak_start = None, 0, -1
            after_stoppage = True  # the kick-off is a restart too

            for f in range(s_f, e_f + 1):
                if dead_mask[f]:
                    labels[f] = "D"
                    streak_team, streak_len = None, 0
                    after_stoppage = True
                    continue

                if after_stoppage:
                    after_stoppage = False
                    if self.restart:
                        team = self._restart_team(f, e_f, dead_mask, ball_vis, min_dh, min_da)
                        if team is not None:
                            curr_team = team

                if ball_vis[f]:
                    ctrl = self._controller(f, spat_h, spat_a, min_deff_h, min_deff_a, gat_h, gat_a)
                    if ctrl is not None:
                        if ctrl == streak_team:
                            streak_len += 1
                        else:
                            streak_team, streak_len, streak_start = ctrl, 1, f

                    if streak_team is not None and streak_team != curr_team                             and streak_len >= self.switch_frames:
                        curr_team = streak_team
                        if self.backdate:
                            # Streaks reset on dead ball, so these frames are all live play.
                            labels[streak_start:f] = curr_team
                        streak_len = 0

                # Ball missing or contested: possession stays with the current team.
                labels[f] = curr_team

        return labels


def ball_ground_speed(ball_x, ball_y, fps=25):
    """Ball speed (m/s) from a 5-frame centred position average; NaN where unknown."""
    sx = pd.Series(ball_x).rolling(5, center=True, min_periods=5).mean().values
    sy = pd.Series(ball_y).rolling(5, center=True, min_periods=5).mean().values
    v = np.full(len(ball_x), np.nan, dtype=np.float32)
    v[1:] = np.hypot(np.diff(sx), np.diff(sy)) * fps
    return v


def controllable(ball_x, ball_y, ball_vis, max_speed):
    """Frames on which a player can be in control of the ball.

    A ball moving faster than ``max_speed`` over the ground is a pass, shot or clearance in flight,
    so nobody controls it on that frame: it counts as no evidence, exactly like a missing ball.
    Frames whose speed is unknown keep their visibility. ``max_speed`` of None or 0 disables this.
    """
    if not max_speed:
        return ball_vis
    return ball_vis & ~(ball_ground_speed(ball_x, ball_y) > max_speed)


# ==============================================================================
# 6. SUBMISSION FILE
# ==============================================================================
class ReproducibleSubmissionEngine:
    """Exports and validates the final submission file."""

    def export(self, labels, output_paths):
        for out_path in output_paths:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # Unix line endings on every operating system, so the bytes never depend on the OS
            # the organisers re-run it on (Rule 3).
            with open(out_path, "w", newline="\n") as f:
                f.write("frame,label\n")
                f.writelines(f"{frame},{lbl}\n" for frame, lbl in enumerate(labels))
            print(f"[APEX Engine] Exported: {out_path}")


def run_apex_pipeline(backdate=False, restart=True, control_speed=8.0, out_paths=None):
    print("=" * 70)
    print("APEX Architecture Execution for Team Let'sC")
    print("=" * 70)
    t0 = time.time()

    # 1. Input data
    print("[1/6] Loading the tactical tracking...")
    data_loader = ApexDataLoader()
    track_teams = data_loader.get_team_slots()
    n_frames = source_frames("tactical")
    pitch = GAMES[GAME]["pitch"]

    # 2. Tracks
    print("[2/6] Parsing players and ball, bridging short ball gaps...")
    analyzer = TrackQualityAndMotionAnalyzer(pitch[0], pitch[1])
    p_pos, raw_bx, raw_by, mean_speed = analyzer.parse(data_loader.tac_raw, len(track_teams), n_frames)
    ball_x, ball_y, ball_vis, ball_out = analyzer.ball_track(raw_bx, raw_by)

    # 3-4. Evidence and dead ball
    print("[3/6] Proximity evidence (spatial, motion, interaction)...")
    spatial_expert = SpatialExpert(track_teams)
    spat_h, spat_a, min_dh, min_da = spatial_expert.compute(p_pos, ball_x, ball_y, ball_vis, n_frames)

    motion_expert = MotionExpert(track_teams)
    min_deff_h, min_deff_a = motion_expert.compute(p_pos, ball_x, ball_y, ball_vis, n_frames)

    gat_expert = InteractionExpert(track_teams)
    gat_h, gat_a = gat_expert.compute(min_deff_h, min_deff_a, ball_vis, n_frames)

    print("[4/6] Dead-ball rule...")
    physics_expert = PhysicsExpert()
    dead_mask = physics_expert.detect_stoppage(mean_speed, ball_out, n_frames)

    # 5. Possession sequence
    print("[5/6] Possession sequence (hysteresis + restart rule)...")
    control_vis = controllable(ball_x, ball_y, ball_vis, control_speed)
    fusion = ReliabilityAwareFusionAndRefinement(backdate=backdate, restart=restart)
    labels = fusion.refine_and_predict(
        dead_mask,
        spat_h,
        spat_a,
        min_dh,
        min_da,
        min_deff_h,
        min_deff_a,
        gat_h,
        gat_a,
        control_vis,
        n_frames,
    )

    # 6. Submission file
    print("[6/6] Writing the submission...")
    engine = ReproducibleSubmissionEngine()
    out_paths = out_paths or [
        os.path.join(ROOT, "evaluation/results/Let'sC.csv"),
        os.path.join(ROOT, "evaluation/results/Let_sC.csv"),
    ]
    engine.export(labels, out_paths)

    elapsed = time.time() - t0
    print(f"\n[APEX Pipeline Complete] Finished deterministically in {elapsed:.2f}s")
    counts = pd.Series(labels).value_counts().to_dict()
    print(f"Final Prediction Counts: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate Team Let'sC's possession file.")
    ap.add_argument("--backdate", action="store_true",
                    help="experimental: relabel a confirmed hand-over from when it began")
    ap.add_argument("--no-restart", action="store_true",
                    help="ablation: carry the pre-stoppage team across restarts")
    ap.add_argument("--control-speed", type=float, default=8.0,
                    help="max ball speed (m/s) at which a player can control it; 0 disables")
    ap.add_argument("--out", nargs="+", help="output CSV path(s); default evaluation/results/")
    args = ap.parse_args()
    run_apex_pipeline(backdate=args.backdate, restart=not args.no_restart,
                      control_speed=args.control_speed, out_paths=args.out)
