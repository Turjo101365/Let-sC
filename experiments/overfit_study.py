"""Generalisation study on IDSSE: strict match-level separation, overfitting and domain shift.

    python experiments/overfit_study.py cv            # LOMO CV over the 5 CV matches (+ nested K)
    python experiments/overfit_study.py curves        # learning curves, simple vs complex
    python experiments/overfit_study.py seeds         # variability over degradation/model seeds
    python experiments/overfit_study.py ablate        # feature-group ablations and a leakage test
    python experiments/overfit_study.py domain        # IDSSE vs BAR-ATH distributions + domain classifier
    python experiments/overfit_study.py sensitivity   # noise, missing ball, 12.5 Hz, identity swaps
    python experiments/overfit_study.py final         # the 2 TEST matches, once, frozen config only

Split (experiments/idsse.py): CV = 5 matches for every choice; TEST = 2 matches, untouched until
data/cache/overfit/frozen.json exists. Frames are never split: every fold holds out whole matches.
Nothing reads the competition's public ground truth.
"""

import itertools
import json
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

import idsse  # noqa: E402
import idsse_eval  # noqa: E402
import learn  # noqa: E402
import possession_metrics as pm  # noqa: E402
import research  # noqa: E402

OUT = os.path.join(ROOT, "data", "cache", "overfit")
FROZEN = os.path.join(OUT, "frozen.json")
SIGMA = 0.0          # ball jitter: label-free calibration to BAR-ATH's noise floor chose 0 (jumps
                     # and dropout alone give 0.439 vs 0.391; any jitter overshoots)
K_GRID = [0, 5, 10, 20, 40, 80]
RULE_GRID = {"switch_frames": [12, 18, 25], "boundary": ["confirm", "bounded:12", "mid"],
             "control_speed": [6.0, 8.0, 10.0], "margin": [0.0, 0.5]}
FROZEN_RULE = {"switch_frames": 18, "boundary": "confirm", "control_speed": 8.0, "margin": 0.0}
FPS = 25


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def degrade(m, seed, sigma=SIGMA, jump_mult=1.0, extra_missing=0.0, drop_mult=1.0, hold=1,
            live_vis=None):
    """idsse_eval.degrade with knobs for the sensitivity study."""
    rate = idsse_eval.JUMP_RATE
    idsse_eval.JUMP_RATE = rate * jump_mult
    try:
        share = idsse_eval.TRACKED_SHARE
        idsse_eval.TRACKED_SHARE = 1 - (1 - share) * drop_mult
        d = idsse_eval.degrade(m, seed, sigma)
    finally:
        idsse_eval.JUMP_RATE = rate
        idsse_eval.TRACKED_SHARE = share
    rng = np.random.default_rng(seed + 77)
    if extra_missing > 0:
        bx, by = d["raw_bx"].copy(), d["raw_by"].copy()
        n = len(bx)
        i = 0
        while i < n:
            length = int(rng.exponential(250)) + 1
            if rng.random() < extra_missing:
                bx[i:i + length] = np.nan
                by[i:i + length] = np.nan
            i += length
        d = dict(d, raw_bx=bx, raw_by=by)
    if live_vis is not None:
        # Restore whole missing runs inside play the frozen dead-ball rule calls live (label-free)
        # until live-play ball visibility reaches ``live_vis``, as on BAR-ATH where the missing
        # ball clusters in stoppages.
        live = ~research.prepare(m)["dead"]
        bx, by = d["raw_bx"].copy(), d["raw_by"].copy()
        miss = np.isnan(bx) & live
        share_missing = miss.sum() / max(live.sum(), 1)
        need = max(0.0, (1 - share_missing) and (live_vis - (1 - share_missing)) / max(share_missing, 1e-9))
        dd = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
            if rng.random() < need:
                bx[a:b] = m["raw_bx"][a:b]
                by[a:b] = m["raw_by"][a:b]
        d = dict(d, raw_bx=bx, raw_by=by)
    if hold > 1:  # temporal resolution: sample-and-hold every ``hold`` frames
        idx = (np.arange(len(d["raw_bx"])) // hold) * hold
        d = dict(d, raw_bx=d["raw_bx"][idx], raw_by=d["raw_by"][idx], p_pos=d["p_pos"][idx],
                 mean_speed=d["mean_speed"][idx])
    return d


_CACHE = {}


def match_data(match, seed=0, **knobs):
    """Prepared signals, frozen-rule labels and features for one degraded match (memoised)."""
    key = (match, seed, tuple(sorted(knobs.items())))
    if key in _CACHE:
        return _CACHE[key]
    m = idsse.build(match)
    idx = idsse.MATCHES.index(match)
    d = degrade(m, seed=1000 * (seed + 1) + idx, **knobs)
    preps = {v: research.prepare(d, control_speed=v) for v in RULE_GRID["control_speed"]}
    segs = [(int(s), int(e)) for s, e in m["segments"]]
    rule = rule_labels(preps[8.0], segs, FROZEN_RULE)
    X, names = learn.features(preps[8.0], rule, m["pitch"])
    out = dict(gt=m["gt"].astype(str), segs=segs, preps=preps, rule=rule, X=X, names=names,
               n=int(m["n_frames"]), pitch=m["pitch"])
    _CACHE[key] = out
    return out


def rule_labels(prep, segs, cfg):
    ctrl = research.control(prep, margin=cfg["margin"])
    return research.decode(prep, [(s, e, research.H) for s, e in segs], ctrl,
                           switch_frames=cfg["switch_frames"], boundary=cfg["boundary"])


def rule_for(md, cfg):
    return rule_labels(md["preps"][cfg["control_speed"]], md["segs"], cfg)


# ----------------------------------------------------------------------------------------------
# Scoring (official scorer + hand-over timing + fragmentation)
# ----------------------------------------------------------------------------------------------
def score(gt, pred, segs):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    t = r["transitions"][25]
    out = {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "trans_P": t["precision"],
           "trans_R": t["recall"], "n_pred_spells": r["n_pred_spells"],
           "n_gt_spells": r["n_gt_spells"], "short_2s": idsse_eval.short_share(pred, segs)}
    out.update(idsse_eval.handover_scores(gt, pred))
    return out


def model_labels(model, md, groups, K):
    X, _ = learn.select(md["X"], md["names"], groups)
    p = learn.predict_home(model, X)
    return learn.decode_probs(p, md["preps"][8.0]["dead"], md["segs"], K, md["n"]), p


def fit_on(kind, matches, seed, groups):
    Xs, ys = [], []
    for mt in matches:
        md = match_data(mt, seed)
        X, _ = learn.select(md["X"], md["names"], groups)
        live = ~md["preps"][8.0]["dead"]
        Xs.append(X[live])
        ys.append(md["gt"][live])
    return learn.fit(kind, Xs, ys, seed=seed)


def choose_K(kind, train, seed, groups):
    """Nested LOMO inside the training matches: K maximising out-of-fold spellF1."""
    totals = {K: [] for K in K_GRID}
    for inner in train:
        rest = [t for t in train if t != inner]
        model = fit_on(kind, rest, seed, groups)
        md = match_data(inner, seed)
        for K in K_GRID:
            lab, _ = model_labels(model, md, groups, K)
            totals[K].append(score(md["gt"], lab, md["segs"])["spellF1"])
    return max(K_GRID, key=lambda K: (np.mean(totals[K]), -K))


def choose_rule(train, seed):
    best = None
    keys = list(RULE_GRID)
    for combo in itertools.product(*[RULE_GRID[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        s = np.mean([score(match_data(t, seed)["gt"], rule_for(match_data(t, seed), cfg),
                           match_data(t, seed)["segs"])["spellF1"] for t in train])
        if best is None or s > best[0]:
            best = (s, cfg)
    return best[1]


def save(df, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".csv")
    df.to_csv(path, index=False)
    print("wrote", path)


def show(df, by, cols):
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(df.groupby(by, sort=False)[cols].agg(["mean", "std"]).round(3))


# ----------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------
def cmd_cv(seed=0, kinds=("logreg", "lgbm"), groups=None, tag="cv"):
    groups = groups or learn.DEFAULT_GROUPS
    rows = []
    for val in idsse.CV:
        train = [t for t in idsse.CV if t != val]
        t0 = time.perf_counter()
        # Frozen rule: no fitting, so "train" is just its score on the training matches.
        for name, cfg in (("rule_frozen", FROZEN_RULE), ("rule_tuned", choose_rule(train, seed))):
            tr = np.mean([score(match_data(t, seed)["gt"], rule_for(match_data(t, seed), cfg),
                                match_data(t, seed)["segs"])["spellF1"] for t in train])
            md = match_data(val, seed)
            row = dict(fold=val, candidate=name, seed=seed, train_spellF1=tr, config=json.dumps(cfg))
            row.update(score(md["gt"], rule_for(md, cfg), md["segs"]))
            rows.append(row)
        for kind in kinds:
            K = choose_K(kind, train, seed, groups)
            model = fit_on(kind, train, seed, groups)
            tr = np.mean([score(match_data(t, seed)["gt"], model_labels(model, match_data(t, seed), groups, K)[0],
                                match_data(t, seed)["segs"])["spellF1"] for t in train])
            md = match_data(val, seed)
            lab, p = model_labels(model, md, groups, K)
            row = dict(fold=val, candidate=kind, seed=seed, train_spellF1=tr, config=json.dumps({"K": K}))
            row.update(score(md["gt"], lab, md["segs"]))
            live = np.isin(md["gt"], ["H", "A"]) & ~md["preps"][8.0]["dead"]
            row.update(reliability(p[live], md["gt"][live] == "H"))
            rows.append(row)
        print("  fold {} done in {:.0f} s".format(val, time.perf_counter() - t0))
    df = pd.DataFrame(rows)
    df["gap"] = df["train_spellF1"] - df["spellF1"]
    save(df, tag)
    show(df, "candidate", ["train_spellF1", "spellF1", "gap", "macroF1", "trans_P", "trans_R",
                           "ho_recall", "ho_precision", "ho_med_lat", "short_2s"])
    return df


def reliability(p, y, bins=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Calibration of per-frame P(home): expected calibration error and Brier score."""
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == y
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return {"frame_acc": float(correct.mean()), "ece": float(ece),
            "brier": float(np.mean((p - y) ** 2))}


def cmd_curves(seed=0):
    rows = []
    for val in idsse.CV:
        train_pool = [t for t in idsse.CV if t != val]
        md = match_data(val, seed)
        for kind in ("logreg", "lgbm"):
            for n in range(1, len(train_pool) + 1):
                for subset in itertools.combinations(train_pool, n):
                    model = fit_on(kind, list(subset), seed, learn.DEFAULT_GROUPS)
                    # K chosen in-sample on the subset (cheap; the CV run uses nested selection)
                    best = max(K_GRID, key=lambda K: np.mean([
                        score(match_data(t, seed)["gt"], model_labels(model, match_data(t, seed), learn.DEFAULT_GROUPS, K)[0],
                              match_data(t, seed)["segs"])["spellF1"] for t in subset]))
                    tr = np.mean([score(match_data(t, seed)["gt"], model_labels(model, match_data(t, seed), learn.DEFAULT_GROUPS, best)[0],
                                        match_data(t, seed)["segs"])["spellF1"] for t in subset])
                    va = score(md["gt"], model_labels(model, md, learn.DEFAULT_GROUPS, best)[0], md["segs"])["spellF1"]
                    rows.append(dict(fold=val, kind=kind, n_train=n, subset="+".join(subset),
                                     K=best, train_spellF1=tr, val_spellF1=va))
        print("  curves fold", val, "done")
    df = pd.DataFrame(rows)
    save(df, "curves")
    print(df.groupby(["kind", "n_train"])[["train_spellF1", "val_spellF1"]].agg(["mean", "std"]).round(3))
    plot_curves(df)
    return df


def plot_curves(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, color in (("logreg", "#0f766e"), ("lgbm", "#b45309")):
        g = df[df.kind == kind].groupby("n_train")
        for col, style in (("train_spellF1", "--"), ("val_spellF1", "-")):
            m, s = g[col].mean(), g[col].std()
            ax.plot(m.index, m.values, style, color=color,
                    label="{} {}".format(kind, "train" if "train" in col else "held-out match"))
            ax.fill_between(m.index, (m - s).values, (m + s).values, color=color, alpha=0.12)
    ax.set_xlabel("training matches")
    ax.set_ylabel("spellF1 (degraded IDSSE)")
    ax.set_xticks(sorted(df.n_train.unique()))
    ax.legend(fontsize=8)
    ax.set_title("Learning curves, leave-one-match-out")
    fig.tight_layout()
    path = os.path.join(ROOT, "reports", "learning_curves.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


def cmd_seeds():
    frames = [cmd_cv(seed=s, tag="cv_seed{}".format(s)) for s in (1, 2)]
    base = pd.read_csv(os.path.join(OUT, "cv.csv"))
    df = pd.concat([base] + frames)
    save(df, "seeds")
    per_seed = df.groupby(["candidate", "seed"])["spellF1"].mean().unstack()
    print(per_seed.round(4))
    return df


def cmd_ablate(seed=0):
    cv = pd.read_csv(os.path.join(OUT, "cv.csv"))
    rows = []
    variants = {"all (default)": learn.DEFAULT_GROUPS}
    for g in learn.DEFAULT_GROUPS:
        variants["- " + g] = [x for x in learn.DEFAULT_GROUPS if x != g]
    variants["+ position (shortcut check)"] = learn.DEFAULT_GROUPS + ["position"]
    variants["proximity only"] = ["proximity"]
    for name, groups in variants.items():
        for kind in ("logreg", "lgbm"):
            for val in idsse.CV:
                train = [t for t in idsse.CV if t != val]
                K = int(json.loads(cv[(cv.fold == val) & (cv.candidate == kind)].config.iloc[0])["K"])
                model = fit_on(kind, train, seed, groups)
                md = match_data(val, seed)
                r = score(md["gt"], model_labels(model, md, groups, K)[0], md["segs"])
                rows.append(dict(ablation=name, kind=kind, fold=val, spellF1=r["spellF1"],
                                 macroF1=r["macroF1"], ho_recall=r["ho_recall"]))
        print("  ablation", name, "done")
    # Leakage test: train on labels shuffled in 10 s blocks; held-out spellF1 must collapse.
    for kind in ("logreg", "lgbm"):
        for val in idsse.CV:
            train = [t for t in idsse.CV if t != val]
            Xs, ys = [], []
            rng = np.random.default_rng(seed)
            for t in train:
                md = match_data(t, seed)
                live = ~md["preps"][8.0]["dead"]
                y = md["gt"].copy()
                blocks = np.arange(len(y)) // 250
                flip = rng.random(blocks.max() + 1) < 0.5
                y = np.where(flip[blocks] & (y == "H"), "A", np.where(flip[blocks] & (y == "A"), "H", y))
                X, _ = learn.select(md["X"], md["names"], learn.DEFAULT_GROUPS)
                Xs.append(X[live])
                ys.append(y[live])
            model = learn.fit(kind, Xs, ys, seed=seed)
            md = match_data(val, seed)
            r = score(md["gt"], model_labels(model, md, learn.DEFAULT_GROUPS, 20)[0], md["segs"])
            rows.append(dict(ablation="shuffled labels (leakage test)", kind=kind, fold=val,
                             spellF1=r["spellF1"], macroF1=r["macroF1"], ho_recall=r["ho_recall"]))
    df = pd.DataFrame(rows)
    save(df, "ablations")
    print(df.groupby(["ablation", "kind"], sort=False)[["spellF1", "macroF1", "ho_recall"]].mean().round(3))
    return df


def cmd_domain(seed=0):
    import common
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    raw = common.load_raw()
    bar = dict(raw, pitch=np.asarray(raw["pitch"]))
    prep_b = research.prepare(bar)
    h1, h2, h2t = common.halves()
    segs_b = [(h1[0], h1[1], research.A), (h2[0], h2[1], research.H)]
    rule_b = research.decode(prep_b, segs_b, research.control(prep_b), prefill=[(h2t, h2[0], "D")])
    Xb, names = learn.features(prep_b, rule_b, bar["pitch"])
    graded_b = np.zeros(len(rule_b), bool)
    graded_b[11185:90652] = True
    graded_b[110808:190198] = True
    live_b = graded_b & ~prep_b["dead"]

    sets = {"BAR-ATH (Metrica ATD)": (Xb[live_b], prep_b, live_b, bar)}
    for cond in ("clean", "degraded"):
        Xs, live_all = [], []
        for mt in idsse.CV:
            m = idsse.build(mt)
            d = m if cond == "clean" else degrade(m, seed=1000 * (seed + 1) + idsse.MATCHES.index(mt))
            pr = research.prepare(d)
            segs = [(int(s), int(e), research.H) for s, e in m["segments"]]
            rl = research.decode(pr, segs, research.control(pr))
            X, _ = learn.features(pr, rl, m["pitch"])
            live = (m["gt"] != "X") & ~pr["dead"]
            Xs.append(X[live])
        sets["IDSSE {} (5 CV matches)".format(cond)] = (np.concatenate(Xs), None, None, None)

    # Raw characteristics
    rows = []
    for name, (X, *_rest) in sets.items():
        col = {c: X[:, i] for i, c in enumerate(names)}
        near = np.minimum(col["dh"], col["da"])
        vis = col["ball_vis"] > 0
        sp = col["ball_speed"][vis & (col["ball_speed"] >= 0)]
        rows.append({"dataset": name, "frames (live)": len(X), "frame rate (Hz)": 25,
                     "ball visible": vis.mean(),
                     "ball speed p50 / p90 (m/s)": "{:.1f} / {:.1f}".format(*np.percentile(sp, [50, 90])),
                     "nearest player p50 / p90 (m)": "{:.1f} / {:.1f}".format(*np.percentile(near[vis], [50, 90])),
                     "frames since ball seen p90": float(np.percentile(col["since_seen"], 90)),
                     "control evidence share": float((col["last_ctrl"] != 0).mean()),
                     "rule hand-overs per min": float(np.sum(np.diff(col["rule_team"]) != 0) / (len(X) / 1500))})
    raw_tab = pd.DataFrame(rows)
    save(raw_tab, "domain_characteristics")
    print(raw_tab.to_string(index=False))

    # Per-feature shift and an unlabelled domain classifier (BAR-ATH vs IDSSE degraded), with
    # contiguous 100 s blocks as the CV unit so temporal autocorrelation cannot leak.
    Xa = sets["BAR-ATH (Metrica ATD)"][0]
    Xi = sets["IDSSE degraded (5 CV matches)"][0]
    from scipy.stats import ks_2samp
    feat_rows = []
    for i, c in enumerate(names):
        if c in learn.GROUPS["position"]:
            continue
        y = np.r_[np.zeros(len(Xa)), np.ones(len(Xi))]
        x = np.r_[Xa[:, i], Xi[:, i]]
        auc = roc_auc_score(y, x)
        feat_rows.append({"feature": c, "KS": ks_2samp(Xa[:, i], Xi[:, i]).statistic,
                          "domain AUC": max(auc, 1 - auc)})
    feat = pd.DataFrame(feat_rows).sort_values("domain AUC", ascending=False)
    save(feat, "domain_features")
    print(feat.round(3).to_string(index=False))

    keep = [i for i, c in enumerate(names) if c not in learn.GROUPS["position"]]
    X = np.r_[Xa[:, keep], Xi[:, keep]]
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xi))]
    blocks = np.r_[np.arange(len(Xa)) // 2500, 10 ** 6 + np.arange(len(Xi)) // 2500]
    rng = np.random.default_rng(seed)
    fold_of = {b: f for b, f in zip(np.unique(blocks), rng.integers(0, 5, len(np.unique(blocks))))}
    folds = np.array([fold_of[b] for b in blocks])
    aucs = []
    for f in range(5):
        tr, te = folds != f, folds == f
        clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                                 random_state=seed, verbose=-1, n_jobs=4, deterministic=True,
                                 force_row_wise=True, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    dom = pd.DataFrame({"fold": range(5), "domain_classifier_AUC": aucs})
    save(dom, "domain_classifier")
    print("domain classifier AUC (blocked 5-fold): mean {:.3f}, sd {:.3f}".format(np.mean(aucs), np.std(aucs)))
    plot_domain(sets, names)


def plot_domain(sets, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show_cols = [("ball_speed", (0, 30), "ball speed (m/s), visible frames"),
                 ("dh", (0, 30), "nearest home player (m)"),
                 ("since_seen", (0, 250), "frames since ball last seen"),
                 ("margin_eff", (-15, 15), "approach-adjusted margin (m)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    colors = ["#111827", "#0f766e", "#b45309"]
    for ax, (c, rng, title) in zip(axes, show_cols):
        i = names.index(c)
        for (name, (X, *_)), col in zip(sets.items(), colors):
            v = X[:, i]
            if c == "ball_speed":
                v = v[v >= 0]
            ax.hist(np.clip(v, *rng), bins=50, range=rng, density=True, histtype="step",
                    color=col, label=name)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(ROOT, "reports", "domain_shift.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print("wrote", path)


def cmd_sensitivity(seed=0):
    cv = pd.read_csv(os.path.join(OUT, "cv.csv"))
    settings = {"calibrated": {}, "ball jitter x2": {"sigma": 2 * SIGMA},
                "more missing ball (+30% runs)": {"extra_missing": 0.3},
                "identity swaps x4": {"jump_mult": 4.0}, "player dropout x3": {"drop_mult": 3.0},
                "12.5 Hz (sample-and-hold)": {"hold": 2}}
    rows = []
    for val in idsse.CV:
        train = [t for t in idsse.CV if t != val]
        models = {k: fit_on(k, train, seed, learn.DEFAULT_GROUPS) for k in ("logreg", "lgbm")}
        for name, knobs in settings.items():
            md = match_data(val, seed, **knobs)
            rows.append(dict(setting=name, fold=val, candidate="rule_frozen",
                             **score(md["gt"], md["rule"], md["segs"])))
            for k, model in models.items():
                K = int(json.loads(cv[(cv.fold == val) & (cv.candidate == k)].config.iloc[0])["K"])
                rows.append(dict(setting=name, fold=val, candidate=k,
                                 **score(md["gt"], model_labels(model, md, learn.DEFAULT_GROUPS, K)[0], md["segs"])))
            _CACHE.pop((val, seed, tuple(sorted(knobs.items()))), None)
        print("  sensitivity fold", val, "done")
    df = pd.DataFrame(rows)
    save(df, "sensitivity")
    print(df.pivot_table(index="setting", columns="candidate", values="spellF1", aggfunc="mean").round(3))
    return df


def cmd_robust(seed=0):
    """Extra robustness settings, added after the first run: a real jitter level (the first run's
    'x2' was 2 x 0) and live-play ball visibility matched to BAR-ATH. Reports, never selects."""
    cv = pd.read_csv(os.path.join(OUT, "cv.csv"))
    tuned = json.loads(cv[cv.candidate == "rule_tuned"].config.iloc[0])
    settings = {"calibrated": {}, "ball jitter 0.2 m": {"sigma": 0.2},
                "ball jitter 0.4 m": {"sigma": 0.4}, "live visibility 80% (BAR-ATH-like)": {"live_vis": 0.80}}
    rows = []
    for val in idsse.CV:
        train = [t for t in idsse.CV if t != val]
        models = {k: fit_on(k, train, seed, learn.DEFAULT_GROUPS) for k in ("logreg", "lgbm")}
        for name, knobs in settings.items():
            md = match_data(val, seed, **knobs)
            rows.append(dict(setting=name, fold=val, candidate="rule_frozen", **score(md["gt"], md["rule"], md["segs"])))
            rows.append(dict(setting=name, fold=val, candidate="rule_tuned", **score(md["gt"], rule_for(md, tuned), md["segs"])))
            for k, model in models.items():
                K = int(json.loads(cv[(cv.fold == val) & (cv.candidate == k)].config.iloc[0])["K"])
                rows.append(dict(setting=name, fold=val, candidate=k,
                                 **score(md["gt"], model_labels(model, md, learn.DEFAULT_GROUPS, K)[0], md["segs"])))
            if knobs:
                _CACHE.pop((val, seed, tuple(sorted(knobs.items()))), None)
        print("  robust fold", val, "done", flush=True)
    df = pd.DataFrame(rows)
    save(df, "robust")
    print(df.pivot_table(index="setting", columns="candidate", values="spellF1", aggfunc="mean").round(3))
    return df


def cmd_final(seed=0):
    if not os.path.exists(FROZEN):
        raise SystemExit("refusing: write {} (the frozen configuration) first".format(FROZEN))
    frozen = json.load(open(FROZEN))
    log = os.path.join(OUT, "final_run.log")
    if os.path.exists(log):
        raise SystemExit("refusing: the TEST matches were already evaluated once ({})".format(log))
    rows = []
    for mt in idsse.TEST:
        md = match_data(mt, seed)
        rows.append(dict(match=mt, candidate="rule_frozen", **score(md["gt"], md["rule"], md["segs"])))
        if frozen.get("rule_tuned"):
            rows.append(dict(match=mt, candidate="rule_tuned",
                             **score(md["gt"], rule_for(md, frozen["rule_tuned"]), md["segs"])))
        for kind, K in frozen.get("models", {}).items():
            model = fit_on(kind, idsse.CV, seed, learn.DEFAULT_GROUPS)
            rows.append(dict(match=mt, candidate=kind,
                             **score(md["gt"], model_labels(model, md, learn.DEFAULT_GROUPS, K)[0], md["segs"])))
    df = pd.DataFrame(rows)
    save(df, "final_test")
    with open(log, "w") as f:
        f.write("TEST evaluated once at {} with frozen config {}\n".format(time.ctime(), json.dumps(frozen)))
    print(df.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cv"
    {"cv": cmd_cv, "curves": cmd_curves, "seeds": cmd_seeds, "ablate": cmd_ablate,
     "domain": cmd_domain, "sensitivity": cmd_sensitivity, "robust": cmd_robust,
     "final": cmd_final}[cmd]()
