"""Synthetic checks of the research decoder (experiments/research.py), experiments A and B.

Evidence is built directly as per-frame control (+1 home, -1 away, 0 none), so each test isolates
one behaviour. Nothing here touches match data.

    python -m unittest tests/test_research.py -v
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import research  # noqa: E402

H, A = research.H, research.A


def prep(n, dead=None, seen=None):
    """Minimal prepared signals: ball near a home player at the start of every live stretch."""
    vis = np.ones(n, bool) if seen is None else seen
    return dict(n=n, dead=np.zeros(n, bool) if dead is None else dead, cvis=vis,
                min_dh=np.full(n, 1.0, np.float32), min_da=np.full(n, 5.0, np.float32))


def ctrl(spec, n):
    """Control array from [(start, stop, value)]."""
    c = np.zeros(n, np.int8)
    for s, e, v in spec:
        c[s:e] = v
    return c


def first(labels, team, after=0):
    idx = np.flatnonzero(labels[after:] == team)
    return int(idx[0]) + after if len(idx) else None


class UncertainTracking(unittest.TestCase):
    def test_missing_ball_holds_possession(self):
        n = 200
        seen = np.ones(n, bool)
        seen[60:140] = False  # 3.2 s without the ball
        c = ctrl([(0, 40, H), (40, 50, A), (140, 146, A), (146, 200, H)], n)  # 10 + 6 < 18
        lab = research.decode(prep(n, seen=seen), [(0, n - 1, H)], c)
        self.assertTrue((lab == "H").all())

    def test_a_streak_survives_an_occlusion(self):
        n = 200
        seen = np.ones(n, bool)
        seen[60:140] = False
        c = ctrl([(0, 40, H), (40, 50, A), (140, 200, A)], n)  # 10 before, 8 more after = 18
        lab = research.decode(prep(n, seen=seen), [(0, n - 1, H)], c)
        self.assertEqual(first(lab, "A"), 147)

    def test_contested_frames_carry_no_evidence_with_a_margin(self):
        n = 120
        p = prep(n)
        p.update(deff_h=np.full(n, 1.0, np.float32), deff_a=np.full(n, 1.2, np.float32),
                 spat_h=np.full(n, .5, np.float32), spat_a=np.full(n, .5, np.float32),
                 gat_h=np.full(n, .5, np.float32), gat_a=np.full(n, .5, np.float32),
                 raw_bx=np.full(n, 50, np.float32), raw_by=np.full(n, 30, np.float32))
        self.assertTrue((research.control(p) == H).all())  # 0.2 m closer counts as control
        self.assertTrue((research.control(p, margin=0.5) == 0).all())  # but not with a margin


class Interception(unittest.TestCase):
    def setUp(self):
        self.n = 200
        # Home holds, away wins the ball at frame 80 and keeps it.
        self.c = ctrl([(0, 70, H), (80, 200, A)], self.n)

    def run_mode(self, **kw):
        return research.decode(prep(self.n), [(0, self.n - 1, H)], self.c, **kw)

    def test_confirmation_is_18_frames_after_the_physical_turnover(self):
        self.assertEqual(first(self.run_mode(), "A"), 80 + 17)

    def test_start_boundary_moves_it_to_the_first_away_control_frame(self):
        self.assertEqual(first(self.run_mode(boundary="start"), "A"), 80)

    def test_bounded_boundary_moves_it_at_most_m_frames(self):
        self.assertEqual(first(self.run_mode(boundary="bounded:12"), "A"), 97 - 12)

    def test_mid_boundary_splits_the_evidence_free_gap(self):
        # last home control 69, first away control 80 -> boundary (70 + 80) // 2 = 75
        self.assertEqual(first(self.run_mode(boundary="mid"), "A"), 75)

    def test_boundary_modes_never_add_or_remove_a_turnover(self):
        def handovers(lab):
            return int(((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])).sum())
        base = handovers(self.run_mode())
        for mode in ("start", "bounded:12", "mid"):
            self.assertEqual(handovers(self.run_mode(boundary=mode)), base, mode)


class DelayedTurnover(unittest.TestCase):
    def test_streak_interrupted_by_occlusion_is_backdated_only_to_its_visible_start(self):
        n = 260
        seen = np.ones(n, bool)
        seen[90:150] = False
        c = ctrl([(0, 70, H), (80, 90, A), (150, 260, A)], n)  # 10 frames, gap, then more
        lab = research.decode(prep(n, seen=seen), [(0, n - 1, H)], c, boundary="start")
        self.assertEqual(first(lab, "A"), 80)
        bounded = research.decode(prep(n, seen=seen), [(0, n - 1, H)], c, boundary="bounded:25")
        self.assertEqual(first(bounded, "A"), (150 + 7) - 25)  # confirmed at 157


class IdentitySwap(unittest.TestCase):
    def test_jump_guard_removes_evidence_around_an_impossible_ball_jump(self):
        n = 100
        bx = np.full(n, 50, np.float32)
        by = np.full(n, 30, np.float32)
        bx[40:43] = 75  # ball detection jumps 25 m for three frames: a false detection
        mask = research.jump_frames(bx, by, guard=5)
        self.assertTrue(mask[35:48].all())
        self.assertFalse(mask[:30].any() or mask[55:].any())

    def test_a_swap_too_short_to_confirm_does_not_flip_possession(self):
        n = 120
        c = ctrl([(0, 40, H), (40, 52, A), (52, 120, H)], n)  # 12 frames of false away control
        lab = research.decode(prep(n), [(0, n - 1, H)], c, boundary="start")
        self.assertTrue((lab == "H").all())


class DeadBallBoundary(unittest.TestCase):
    def test_backdating_never_crosses_a_dead_ball(self):
        n = 220
        dead = np.zeros(n, bool)
        dead[100:130] = True
        # Away is nearest just before the stoppage and right after the restart, but the streak is
        # reset by the dead ball, so nothing may be backdated into or across it.
        c = ctrl([(0, 90, H), (90, 100, A), (130, 220, A)], n)
        p = prep(n, dead=dead)
        p["min_dh"][130:] = 5.0
        p["min_da"][130:] = 1.0  # restart taken by away
        for mode in ("confirm", "start", "bounded:25", "mid"):
            lab = research.decode(p, [(0, n - 1, H)], c, boundary=mode)
            self.assertTrue((lab[100:130] == "D").all(), mode)
            self.assertTrue((lab[:100] == "H").all(), mode)
            self.assertEqual(lab[130], "A", mode)  # restart rule, not a backdated hand-over


if __name__ == "__main__":
    unittest.main()
