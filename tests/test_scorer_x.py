"""What an abstention (X) can and cannot do, checked against the official scorer itself.

The scorer forward-fills every predicted X with the last non-X label in the same graded run
(back-filling at the run head) before any temporal metric. So for spellF1, the ranking metric, a
file with X is *identical* to the same file with those frames forward-filled by hand. X can only
change frame-level metrics such as macroF1.

These tests check that on many random ground-truth / prediction pairs, so the conclusion rests on
the scorer's code rather than on reading its documentation.

    python -m unittest tests/test_scorer_x.py -v
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def random_spells(rng, n, mean_len, labels):
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        length = max(1, int(rng.exponential(mean_len)))
        out[i:i + length] = rng.choice(labels)
        i += length
    return out


def make_gt(rng, n=6000):
    """Two graded runs separated by an ungraded X gap, like the two halves."""
    gt = np.full(n, pm.X, dtype=np.int64)
    gt[200:2800] = random_spells(rng, 2600, 120, [pm.H, pm.A, pm.D])
    gt[3300:5900] = random_spells(rng, 2600, 120, [pm.H, pm.A, pm.D])
    return gt


def make_pred(rng, gt):
    """The truth with shifted boundaries and some flipped spells, outside the halves left X."""
    pred = gt.copy()
    pred[pred == pm.X] = pm.D
    flips = random_spells(rng, len(gt), 60, [0, 0, 0, 1])
    pred = np.where(flips == 1, rng.choice([pm.H, pm.A, pm.D], len(gt)), pred)
    return np.roll(pred, int(rng.integers(-15, 16)))


def forward_fill_by_hand(pred, runs):
    out = pred.copy()
    for start, stop in runs:
        last = None
        for i in range(start, stop):
            if out[i] == pm.X:
                if last is not None:
                    out[i] = last
            else:
                last = out[i]
        first = next((out[i] for i in range(start, stop) if out[i] != pm.X), None)
        for i in range(start, stop):
            if out[i] != pm.X:
                break
            out[i] = first
    return out


class AbstentionIsHoldForSpellF1(unittest.TestCase):
    def test_x_equals_forward_fill_on_every_temporal_metric(self):
        rng = np.random.default_rng(0)
        for trial in range(40):
            gt = make_gt(rng)
            pred = make_pred(rng, gt)
            # Abstain on a random 25% of frames, in runs, anywhere (inside possessions and at
            # hand-overs alike).
            mask = random_spells(rng, len(gt), 30, [0, 0, 0, 1]) == 1
            with_x = np.where(mask, pm.X, pred)
            filled = forward_fill_by_hand(with_x, pm.valid_runs(gt))
            a, b = pm.evaluate(gt, with_x), pm.evaluate(gt, filled)
            for key in ("spell_f1_mean", "n_pred_spells", "fragmentation"):
                self.assertEqual(a[key], b[key], "trial {} {}".format(trial, key))
            for tol in pm.TAUS:
                self.assertEqual(a["transitions"][tol], b["transitions"][tol])


class AbstentionOnlyMovesMacroF1(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([pm.X] * 10 + [pm.H] * 50 + [pm.A] * 50 + [pm.D] * 50 + [pm.X] * 10)

    def test_x_on_a_correct_frame_costs_macro_f1(self):
        pred = self.gt.copy()
        pred[pred == pm.X] = pm.D
        with_x = pred.copy()
        with_x[20:30] = pm.X  # these were correct H frames
        self.assertLess(pm.evaluate(self.gt, with_x)["macro_f1"],
                        pm.evaluate(self.gt, pred)["macro_f1"])
        self.assertEqual(pm.evaluate(self.gt, with_x)["spell_f1_mean"],
                         pm.evaluate(self.gt, pred)["spell_f1_mean"])

    def test_x_on_a_wrong_frame_can_raise_macro_f1(self):
        pred = self.gt.copy()
        pred[pred == pm.X] = pm.D
        pred[100:110] = pm.H  # wrong: truth is D
        with_x = pred.copy()
        with_x[100:110] = pm.X  # abstain there instead of being wrong
        self.assertGreater(pm.evaluate(self.gt, with_x)["macro_f1"],
                           pm.evaluate(self.gt, pred)["macro_f1"])


if __name__ == "__main__":
    unittest.main()
