"""Shared plumbing for Team Let'sC's possession experiments.

Every experiment starts from the same intermediate signals the submitted pipeline computes, so a
variant differs from the baseline only in the stage it replaces. Nothing here is written to disk
except label files under data/cache/ (gitignored: data/* never leaves this machine).
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import apex_architecture_model as apex  # noqa: E402
from games import GAME, GAMES, source_frames, half_video_blocks, anchor_video_frame  # noqa: E402

FPS = 25
OUT_DIR = os.path.join(ROOT, "data", "cache", "variants")


def halves():
    """[(first_frame, last_frame)] of live play per half, as the pipeline uses them."""
    blocks = half_video_blocks(GAME)
    kickoff_2nd = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    return [(blocks[0][1], blocks[0][2]), (kickoff_2nd, blocks[1][2]), blocks[1][1]]


def load_raw():
    """Parse the tactical ATD once: player positions, raw ball, mean speed, slot teams."""
    loader = apex.ApexDataLoader()
    track_teams = loader.get_team_slots()
    n_frames = source_frames("tactical")
    pitch = GAMES[GAME]["pitch"]
    analyzer = apex.TrackQualityAndMotionAnalyzer(pitch[0], pitch[1])
    p_pos, raw_bx, raw_by, mean_speed = analyzer.parse(loader.tac_raw, len(track_teams), n_frames)
    return dict(p_pos=p_pos, raw_bx=raw_bx, raw_by=raw_by, mean_speed=mean_speed,
                track_teams=track_teams, n_frames=n_frames, pitch=pitch)


def evidence(raw, bx, by, ball_vis):
    """The three proximity signals for a given ball track (same code as the pipeline)."""
    n, tt, p = raw["n_frames"], raw["track_teams"], raw["p_pos"]
    spat_h, spat_a, min_dh, min_da = apex.SpatialExpert(tt).compute(p, bx, by, ball_vis, n)
    deff_h, deff_a = apex.MotionExpert(tt).compute(p, bx, by, ball_vis, n)
    gat_h, gat_a = apex.InteractionExpert(tt).compute(deff_h, deff_a, ball_vis, n)
    return dict(spat_h=spat_h, spat_a=spat_a, min_dh=min_dh, min_da=min_da,
                min_deff_h=deff_h, min_deff_a=deff_a, gat_h=gat_h, gat_a=gat_a, ball_vis=ball_vis)


def baseline_track(raw):
    pitch = raw["pitch"]
    analyzer = apex.TrackQualityAndMotionAnalyzer(pitch[0], pitch[1])
    return analyzer.ball_track(raw["raw_bx"], raw["raw_by"])


def dead_mask(raw, ball_out):
    return apex.PhysicsExpert().detect_stoppage(raw["mean_speed"], ball_out, raw["n_frames"])


def baseline_labels(raw, ev, dead, **kwargs):
    fusion = apex.ReliabilityAwareFusionAndRefinement(**kwargs)
    return fusion.refine_and_predict(dead, ev["spat_h"], ev["spat_a"], ev["min_dh"], ev["min_da"],
                                     ev["min_deff_h"], ev["min_deff_a"], ev["gat_h"], ev["gat_a"],
                                     ev["ball_vis"], raw["n_frames"])


def write_labels(name, labels):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".csv")
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lbl) for i, lbl in enumerate(labels))
    return path


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t
