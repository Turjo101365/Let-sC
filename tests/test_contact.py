"""Synthetic checks of the contact-aware possession logic (experiments/contact.py).

Each scene is a few seconds of made-up tracking at 25 fps: two home players (H1, H2, H3) and two
away players (A1, A2), with a ball path built from straight segments. Nothing here touches the
match data.

    python -m unittest tests/test_contact.py -v
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import apex_architecture_model as apex  # noqa: E402
import contact  # noqa: E402

FPS = 25
TEAMS = np.array([0, 0, 0, 1, 1], dtype=np.int8)  # H1 H2 H3 A1 A2
TAU = 0.3  # fixed for synthetic scenes; the real pipeline measures it from the noise floor


def path(points, frames):
    """Ball path through ``points`` reaching each at the given frame (linear in between)."""
    out = np.zeros((frames[-1] + 1, 2), dtype=np.float32)
    for (p0, f0), (p1, f1) in zip(zip(points, frames), zip(points[1:], frames[1:])):
        w = np.linspace(0, 1, f1 - f0 + 1)[:, None]
        out[f0:f1 + 1] = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * w
    return out


def scene(ball, players, dead=None):
    n = len(ball)
    p_pos = np.zeros((n, len(TEAMS), 2), dtype=np.float32)
    for i, pos in enumerate(players):
        p_pos[:, i] = pos(np.arange(n)) if callable(pos) else np.asarray(pos, dtype=np.float32)
    return dict(bx=ball[:, 0].copy(), by=ball[:, 1].copy(), p_pos=p_pos,
                dead=np.zeros(n, bool) if dead is None else dead)


def run(sc, start_team=contact.H, **opts):
    n = len(sc["bx"])
    vis = np.ones(n, bool)
    bx, by, p = sc["bx"], sc["by"], sc["p_pos"]
    spat_h, spat_a, min_dh, min_da = apex.SpatialExpert(TEAMS).compute(p, bx, by, vis, n)
    deff_h, deff_a = apex.MotionExpert(TEAMS).compute(p, bx, by, vis, n)
    gat_h, gat_a = apex.InteractionExpert(TEAMS).compute(deff_h, deff_a, vis, n)
    cvis = apex.controllable(bx, by, vis, 8.0)
    ev = dict(spat_h=spat_h, spat_a=spat_a, min_deff_h=deff_h, min_deff_a=deff_a,
              gat_h=gat_h, gat_a=gat_a, ball_vis=cvis)
    touches = contact.detect_touches(bx, by, vis, min_dh, min_da, TAU)
    labels = contact.decode([(0, n - 1, start_team)], sc["dead"], contact.control_array(ev), cvis,
                            min_dh, min_da, n, touches=touches, **opts)
    return labels, touches


CONTACT = dict(intercept=True, consecutive=True, teammate=True)
STILL = lambda xy: xy  # noqa: E731


def first_change(labels, team):
    idx = np.flatnonzero(labels == team)
    return int(idx[0]) if len(idx) else None


class Pass(unittest.TestCase):
    """H1 passes 30 m to H2 at 20 m/s; the ball then stays at H2's feet."""

    def build(self, a1_offset):
        ball = path([(10, 30), (40, 30), (40, 30)], [0, 38, 100])
        return scene(ball, [(10, 30.6), (40.6, 30), (60, 10), (25, 30 + a1_offset), (70, 50)])

    def test_received_by_teammate_keeps_possession(self):
        labels, touches = run(self.build(a1_offset=8.0), **CONTACT)
        self.assertTrue((labels == "H").all())
        self.assertTrue(any(team == contact.H and kind == "control" for _, team, kind, _ in touches))

    def test_fast_fly_by_near_opponent_is_not_a_touch(self):
        labels, touches = run(self.build(a1_offset=1.0), **CONTACT)
        self.assertTrue((labels == "H").all(), "a ball passing 1 m from A1 must not change hands")
        self.assertFalse(any(team == contact.A for _, team, _, _ in touches))


class Interception(unittest.TestCase):
    """A1 stands on the passing line, kills the ball and dribbles away at 3 m/s."""

    def build(self):
        ball = path([(10, 30), (25, 30), (25, 30), (31, 38)], [0, 19, 24, 74])
        a1 = lambda f: np.stack([np.where(f < 24, 25.4, 25.4 + (f - 24) * 0.12),  # noqa: E731
                                 np.where(f < 24, 30.0, 30.0 + (f - 24) * 0.16)], 1)
        return scene(ball, [(10, 30.6), (40, 30), (60, 10), a1, (70, 50)])

    def test_contact_model_hands_over_at_the_touch(self):
        labels, touches = run(self.build(), **CONTACT)
        touch = [t for t, team, kind, _ in touches if team == contact.A and kind == "control"]
        self.assertTrue(touch, "the interception must be detected as an away control touch")
        self.assertEqual(labels[-1], "A")
        self.assertLessEqual(abs(first_change(labels, "A") - touch[0]), 3)

    def test_baseline_hands_over_later(self):
        base, _ = run(self.build())
        contact_labels, _ = run(self.build(), **CONTACT)
        self.assertEqual(base[-1], "A")
        self.assertGreater(first_change(base, "A"), first_change(contact_labels, "A"))


class Deflection(unittest.TestCase):
    """A1 touches a fast pass, which changes direction but keeps going to H3, who keeps it."""

    def test_deflection_back_to_the_passing_team_is_not_a_hand_over(self):
        ball = path([(10, 30), (25, 30), (40, 40), (40, 40)], [0, 19, 40, 110])
        sc = scene(ball, [(10, 30.6), (60, 60), (40.6, 40), (25, 31.0), (80, 50)])
        labels, touches = run(sc, **CONTACT)
        self.assertTrue(any(team == contact.A and kind == "redirect" for _, team, kind, _ in touches))
        self.assertTrue((labels == "H").all())

    def test_one_touch_interception_to_a_teammate_is_a_hand_over(self):
        ball = path([(10, 30), (25, 30), (40, 40), (40, 40)], [0, 19, 40, 110])
        sc = scene(ball, [(10, 30.6), (60, 60), (80, 10), (25, 31.0), (40.6, 40)])
        labels, touches = run(sc, **CONTACT)
        first = [t for t, team, kind, _ in touches if team == contact.A and kind == "redirect"]
        self.assertTrue(first)
        self.assertEqual(labels[-1], "A")
        self.assertLessEqual(abs(first_change(labels, "A") - first[0]), 3)


class LooseBall(unittest.TestCase):
    """H1 and A1 both within half a metre of a ball that stops between them."""

    def test_contested_touch_does_not_immediately_change_possession(self):
        ball = path([(10, 30), (25, 30), (25, 30)], [0, 19, 40])
        sc = scene(ball, [(10, 30.6), (25.0, 31.0), (60, 10), (25.0, 29.1), (80, 50)])
        labels, touches = run(sc, **CONTACT)
        contested = [t for t, _, _, amb in touches if amb]
        self.assertTrue(contested, "the touch between H2 and A1 must be flagged as contested")
        t = contested[0]
        # The contact rules leave it alone; only the ordinary 18-frame rule can move it later.
        self.assertTrue((labels[:t + 17] == "H").all())

    def test_contested_ball_that_settles_with_nobody_stays(self):
        ball = path([(10, 30), (25, 30), (25, 30)], [0, 19, 60])
        sc = scene(ball, [(10, 30.6), (25.0, 31.0), (60, 10), (25.0, 29.0), (80, 50)])
        labels, _ = run(sc, **CONTACT)
        base, _ = run(sc)
        self.assertEqual(first_change(labels, "A"), first_change(base, "A"),
                         "a contested ball must be decided by the ordinary rule, not by contact")


class Restart(unittest.TestCase):
    """Two seconds of dead ball, then the ball reappears at A1's feet with H1 five metres away."""

    def build(self):
        ball = path([(50, 34), (50, 34)], [0, 120])
        dead = np.zeros(121, bool)
        dead[:50] = True
        return scene(ball, [(55, 34), (70, 20), (80, 10), (50.5, 34), (30, 30)], dead=dead)

    def test_restart_goes_to_the_player_at_the_ball(self):
        labels, _ = run(self.build(), start_team=contact.H)
        self.assertTrue((labels[:50] == "D").all())
        self.assertTrue((labels[50:] == "A").all())

    def test_set_piece_tag_overrides_the_restart(self):
        labels, _ = run(self.build(), start_team=contact.A, restart_override={50: contact.H})
        self.assertEqual(labels[50], "H")


class TagRestartMapping(unittest.TestCase):
    def test_tag_maps_to_the_live_stretch_starting_inside_it(self):
        tag = {"code": "SET PIECES", "start_frame": 1000, "end_frame": 1125,
               "tags": {"Type": "Corner Kick", "Team": "Barcelona"}}
        far = {"code": "SET PIECES", "start_frame": 5000, "end_frame": 5125,
               "tags": {"Type": "Goal Kick", "Team": "Atletic Club"}}
        override = contact.tag_restarts([tag, far], [900, 1080, 3000, 9000])
        self.assertEqual(override, {1080: contact.H})


if __name__ == "__main__":
    unittest.main()
