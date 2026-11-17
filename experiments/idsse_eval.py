"""Real-ground-truth evaluation of the frozen pipeline and experiments A/B on IDSSE matches.

    python experiments/idsse_eval.py --split dev      # develop and choose here
    python experiments/idsse_eval.py --split test     # run once, to confirm a DEV choice

Two conditions per match:
  clean     IDSSE as delivered (TRACAB: ball always tracked, smoothed)
  degraded  IDSSE degraded to BAR-ATH's measured tracking noise, so results transfer to the
            competition's automatic tracking. Calibrated without labels from BAR-ATH:
              ball visibility    visible / missing run lengths resampled from BAR-ATH's own runs
              detector jumps     false detections 5-20 m away, 1-3 frames long, at BAR-ATH's rate
                                 of implausible jumps per seen frame (0.47%)
              ball jitter        Gaussian noise on ball x/y, sigma chosen so the fast-ball noise
                                 floor (p95 relative velocity change, nobody within 5 m) matches
                                 BAR-ATH's 0.391
              player dropout     player slots missing in runs so ~20.5 of 22 are tracked
Scored with the official evaluation/possession_metrics.evaluate().
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)

import contact  # noqa: E402
import idsse  # noqa: E402
import possession_metrics as pm  # noqa: E402
import research  # noqa: E402

NOISE = np.load(os.path.join(ROOT, "data", "cache", "barath_noise_stats.npz"))
JUMP_RATE = float(NOISE["jumps_per_seen_frame"]) / 2  # each excursion makes two jumps (in, out)
TRACKED_SHARE = float(NOISE["player_slots_tracked_mean_graded"]) / 22.0
FPS = 25


def visibility_mask(n, rng):
    vis, miss = NOISE["vis_runs"], NOISE["miss_runs"]
    mask = np.zeros(n, bool)
    i, seen = 0, rng.random() < float(NOISE["ball_seen_graded"])
    while i < n:
        length = int(rng.choice(vis if seen else miss))
        mask[i:i + length] = seen
        i += length
        seen = not seen
    return mask


def degrade(m, seed, sigma):
    rng = np.random.default_rng(seed)
    n = int(m["n_frames"])
    bx, by = m["raw_bx"].copy(), m["raw_by"].copy()
    have = ~np.isnan(bx)
    bx[have] += rng.normal(0, sigma, have.sum()).astype(np.float32)
    by[have] += rng.normal(0, sigma, have.sum()).astype(np.float32)
    keep = visibility_mask(n, rng)
    bx[~keep] = np.nan
    by[~keep] = np.nan
    seen = np.flatnonzero(~np.isnan(bx))
    starts = seen[rng.random(len(seen)) < JUMP_RATE]
    for s in starts:
        length = int(rng.integers(1, 4))
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(5, 20)
        seg = slice(s, min(s + length, n))
        ok = ~np.isnan(bx[seg])
        bx[seg][ok] += np.float32(dist * np.cos(ang))
        by[seg][ok] += np.float32(dist * np.sin(ang))
    p = m["p_pos"].copy()
    drop_rate = max(0.0, 1 - TRACKED_SHARE)
    for slot in range(p.shape[1]):
        i = 0
        while i < n:
            if rng.random() < drop_rate:
                length = int(rng.exponential(75)) + 1
                p[i:i + length, slot] = np.nan
                i += length
            else:
                i += int(rng.exponential(75 * (1 - drop_rate) / max(drop_rate, 1e-6))) + 1
    return dict(m, raw_bx=bx, raw_by=by, p_pos=p)


def noise_floor(m, prep):
    live = ~prep["dead"] & (m["gt"] != "X")
    return contact.noise_floor(prep["bx"], prep["by"], prep["min_dh"], prep["min_da"], live,
                               pct=95)[0]


def handover_scores(gt, pred, window=50):
    """Team-to-team hand-overs only: recall, precision and timing (pred - truth, frames)."""
    def handovers(lab):
        idx = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])
                             & np.isin(lab[:-1], ["H", "A"])) + 1
        return idx, lab[idx]
    gi, gteam = handovers(gt)
    pi, pteam = handovers(pred)
    cand = []
    for a, (g, gt_team) in enumerate(zip(gi, gteam)):
        near = np.flatnonzero((np.abs(pi - g) <= window) & (pteam == gt_team))
        cand += [(abs(int(pi[b]) - int(g)), a, b) for b in near]
    used_g, used_p, lat = set(), set(), []
    for d, a, b in sorted(cand):
        if a in used_g or b in used_p:
            continue
        used_g.add(a)
        used_p.add(b)
        lat.append(int(pi[b]) - int(gi[a]))
    lat = np.array(lat)
    return {"ho_recall": len(lat) / max(len(gi), 1), "ho_precision": len(lat) / max(len(pi), 1),
            "ho_med_lat": float(np.median(lat)) if len(lat) else np.nan,
            "ho_mae": float(np.abs(lat).mean()) if len(lat) else np.nan, "ho_gt": len(gi)}


def short_share(lab, segments):
    out = []
    for s, e in segments:
        w = lab[s:e + 1]
        change = np.flatnonzero(w[1:] != w[:-1]) + 1
        b = np.concatenate(([0], change, [len(w)]))
        out += [(b[i + 1] - b[i]) / FPS for i in range(len(b) - 1) if w[b[i]] in ("H", "A")]
    out = np.array(out)
    return float((out < 2).mean()) if len(out) else np.nan


VARIANTS = {
    # name: (prepare kwargs, control kwargs, decode kwargs)
    "base": ({}, {}, {}),
    "gate_off": ({"control_speed": 0}, {}, {}),
    "gate6": ({"control_speed": 6.0}, {}, {}),
    "gate10": ({"control_speed": 10.0}, {}, {}),
    "B_start": ({}, {}, {"boundary": "start"}),
    "B_bounded12": ({}, {}, {"boundary": "bounded:12"}),
    "B_bounded25": ({}, {}, {"boundary": "bounded:25"}),
    "B_mid": ({}, {}, {"boundary": "mid"}),
    "B_confirm9": ({}, {}, {"switch_frames": 9}),
    "B_confirm12": ({}, {}, {"switch_frames": 12}),
    "B_confirm25": ({}, {}, {"switch_frames": 25}),
    "A_margin0.5": ({}, {"margin": 0.5}, {}),
    "A_margin1.0": ({}, {"margin": 1.0}, {}),
    "A_jumpguard5": ({}, {"jump_guard": 5}, {}),
}


def score(gt, pred, segments):
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    r = pm.evaluate(g, p)
    t = r["transitions"][25]
    out = {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
           "trans_P": t["precision"], "trans_R": t["recall"],
           "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"],
           "short_2s": short_share(pred, segments), "alive_dead_F1": r["alive_dead_f1"]}
    out.update(handover_scores(gt, pred))
    return out


def calibrate_sigma(match):
    """Ball jitter whose fast-ball noise floor matches BAR-ATH's (label-free)."""
    m = idsse.build(match)
    target = 0.391
    best = None
    for sigma in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
        dm = degrade(m, seed=0, sigma=sigma)
        nf = noise_floor(dm, research.prepare(dm))
        print("    sigma {:.2f} -> noise floor p95 {:.3f}".format(sigma, nf))
        if best is None or abs(nf - target) < abs(best[1] - target):
            best = (sigma, nf)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--sigma", type=float, default=None)
    args = ap.parse_args()
    matches = idsse.DEV if args.split == "dev" else idsse.TEST

    sigma = args.sigma
    if sigma is None:
        print("calibrating ball jitter on", idsse.DEV[0])
        sigma, nf = calibrate_sigma(idsse.DEV[0])
        print("  chosen sigma {} (noise floor {:.3f} vs BAR-ATH 0.391)".format(sigma, nf))

    rows = []
    for mi, match in enumerate(matches):
        t0 = time.perf_counter()
        m = idsse.build(match)
        segs = [(int(s), int(e), research.H) for s, e in m["segments"]]
        gt = m["gt"].astype(str)
        for cond, mm in (("clean", m), ("degraded", degrade(m, seed=1000 + mi, sigma=sigma))):
            preps = {}
            for name, (pk, ck, dk) in VARIANTS.items():
                key = tuple(sorted(pk.items()))
                if key not in preps:
                    preps[key] = research.prepare(mm, **pk)
                prep = preps[key]
                t = time.perf_counter()
                lab = research.decode(prep, segs, research.control(prep, **ck), **dk)
                row = dict(match=match, cond=cond, variant=name,
                           decode_s=round(time.perf_counter() - t, 2))
                row.update(score(gt, lab, [(s, e) for s, e, _ in segs]))
                rows.append(row)
        print("  {} done in {:.0f} s".format(match, time.perf_counter() - t0))

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "data", "cache", "idsse_{}.csv".format(args.split))
    df.to_csv(out, index=False)
    cols = ["spellF1", "macroF1", "trans_P", "trans_R", "ho_recall", "ho_precision",
            "ho_med_lat", "ho_mae", "short_2s", "n_pred_spells", "n_gt_spells"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        for cond in ("clean", "degraded"):
            print("\n==", args.split, cond, "(mean over", len(matches), "matches)")
            print(df[df.cond == cond].groupby("variant", sort=False)[cols].mean())
            base = df[(df.cond == cond) & (df.variant == "base")].set_index("match")["spellF1"]
            wins = {}
            for v in VARIANTS:
                d = df[(df.cond == cond) & (df.variant == v)].set_index("match")["spellF1"] - base
                wins[v] = "{:+.3f} (better on {}/{})".format(d.mean(), int((d > 0).sum()), len(d))
            print("spellF1 vs base per match:", wins)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
