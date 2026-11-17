"""Supervised possession models trained only on other matches (IDSSE), plus their decoder.

Leakage safety by construction: features() takes the *prepared tracking signals* (research.prepare)
and nothing else, so ground truth cannot reach a feature. Labels enter only in fit().

Task: for every frame the frozen dead-ball rule calls live, predict home vs away. Dead ball stays
the rule's, so models and the frozen pipeline differ only in who has the ball. Per-frame
probabilities are decoded per live stretch by a two-state Viterbi with a per-hand-over cost K.

Feature groups (ablations drop one at a time):
  proximity        nearest home/away distances, approach-adjusted distances, their margins
  ball             ball speed, visibility, frames since / until the ball is seen
  context_past     signed control evidence summed over the last 1 s / 3 s; last control and its age
  context_future   the same looking forward (the whole match is available offline)
  rule_state       the frozen pipeline's own label and the time since its last hand-over
  position         ball x/y on the pitch -- EXCLUDED by default: attacking direction flips by half
                   and by match, so it is a shortcut that cannot generalise; used only as a check
"""

import numpy as np

import apex_architecture_model as apex
import research

CAP = 30.0
AGE_CAP = 250

GROUPS = {
    "proximity": ["dh", "da", "deff_h", "deff_a", "margin", "margin_eff"],
    "ball": ["ball_speed", "ball_vis", "since_seen", "until_seen"],
    "context_past": ["ctrl_p25", "ctrl_p75", "last_ctrl", "since_ctrl"],
    "context_future": ["ctrl_f25", "ctrl_f75", "next_ctrl", "until_ctrl"],
    "rule_state": ["rule_team", "since_rule_switch"],
    "position": ["ball_x", "ball_y"],
}
DEFAULT_GROUPS = ["proximity", "ball", "context_past", "context_future", "rule_state"]


def _age_since(event):
    """Frames since the last True in ``event`` (capped), per frame."""
    idx = np.where(event, np.arange(len(event)), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(event)) - last, AGE_CAP).astype(np.float32)


def _window_sum(x, before, after):
    c = np.concatenate(([0], np.cumsum(x, dtype=np.int64)))
    n = len(x)
    lo = np.clip(np.arange(n) - before, 0, n)
    hi = np.clip(np.arange(n) + after + 1, 0, n)
    return (c[hi] - c[lo]).astype(np.float32)


def features(prep, rule_labels, pitch):
    """(n_frames x n_features) matrix and its column names. Tracking-derived only."""
    vis = prep["vis"]
    ctrl = research.control(prep).astype(np.int64)
    dh = np.where(vis, np.minimum(prep["min_dh"], CAP), CAP)
    da = np.where(vis, np.minimum(prep["min_da"], CAP), CAP)
    eh = np.where(vis, np.minimum(prep["deff_h"], CAP), CAP)
    ea = np.where(vis, np.minimum(prep["deff_a"], CAP), CAP)
    speed = apex.ball_ground_speed(prep["bx"], prep["by"])
    nz = ctrl != 0
    last_idx = np.maximum.accumulate(np.where(nz, np.arange(len(ctrl)), 0))
    next_idx = np.minimum.accumulate(np.where(nz, np.arange(len(ctrl)), len(ctrl) - 1)[::-1])[::-1]
    rule = np.where(rule_labels == "H", 1, np.where(rule_labels == "A", -1, 0))
    switch = np.concatenate(([False], (rule[1:] != rule[:-1]) & (rule[1:] != 0) & (rule[:-1] != 0)))
    cols = {
        "dh": dh, "da": da, "deff_h": eh, "deff_a": ea,
        "margin": np.clip(da - dh, -15, 15), "margin_eff": np.clip(ea - eh, -15, 15),
        "ball_speed": np.nan_to_num(np.minimum(speed, 40), nan=-1.0),
        "ball_vis": vis.astype(np.float32),
        "since_seen": _age_since(vis), "until_seen": _age_since(vis[::-1])[::-1],
        "ctrl_p25": _window_sum(ctrl, 25, 0), "ctrl_p75": _window_sum(ctrl, 75, 0),
        "last_ctrl": ctrl[last_idx].astype(np.float32), "since_ctrl": _age_since(nz),
        "ctrl_f25": _window_sum(ctrl, 0, 25), "ctrl_f75": _window_sum(ctrl, 0, 75),
        "next_ctrl": ctrl[next_idx].astype(np.float32), "until_ctrl": _age_since(nz[::-1])[::-1],
        "rule_team": rule.astype(np.float32), "since_rule_switch": _age_since(switch),
        "ball_x": np.nan_to_num(prep["bx"] / pitch[0], nan=-1.0),
        "ball_y": np.nan_to_num(prep["by"] / pitch[1], nan=-1.0),
    }
    names = [c for g in GROUPS.values() for c in g]
    return np.stack([cols[c] for c in names], 1).astype(np.float32), names


def select(X, names, groups):
    keep = [i for i, c in enumerate(names) if any(c in GROUPS[g] for g in groups)]
    return X[:, keep], [names[i] for i in keep]


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
def make_model(kind, seed):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=seed, n_jobs=4,
                                  deterministic=True, force_row_wise=True, verbose=-1)
    raise ValueError(kind)


def fit(kind, Xs, ys, seed=0, max_rows=400_000):
    """Fit on frames whose label is H or A. Rows subsampled deterministically if too many."""
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    keep = np.isin(y, ["H", "A"])
    X, y = X[keep], (y[keep] == "H").astype(np.int8)
    if len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
        idx.sort()
        X, y = X[idx], y[idx]
    model = make_model(kind, seed)
    model.fit(X, y)
    return model


def predict_home(model, X):
    return model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------------------------------
def viterbi(logodds, K):
    """Two-state segmentation of one live stretch maximising sum(evidence) - K * hand-overs."""
    n = len(logodds)
    if K <= 0:
        return np.where(logodds >= 0, 1, -1)
    sh, sa = 0.0, 0.0
    back = np.zeros((n, 2), dtype=bool)
    for t in range(n):
        if t:
            ch, ca = sa - K, sh - K
            back[t, 0] = ch > sh
            back[t, 1] = ca > sa
            sh, sa = max(sh, ch), max(sa, ca)
        sh += logodds[t]
        sa -= logodds[t]
    state = np.empty(n, dtype=np.int8)
    s = 0 if sh >= sa else 1
    for t in range(n - 1, -1, -1):
        state[t] = 1 if s == 0 else -1
        if back[t, s]:
            s = 1 - s
    return state


def decode_probs(p_home, dead, segments, K, n):
    labels = np.full(n, "X", dtype="<U1")
    lo = np.log(np.clip(p_home, 1e-3, 1 - 1e-3) / np.clip(1 - p_home, 1e-3, 1))
    for s, e in segments:
        seg_dead = dead[s:e + 1]
        labels[s:e + 1][seg_dead] = "D"
        d = np.diff(np.concatenate(([1], seg_dead.astype(np.int8), [1])))
        for a, b in zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)):
            st = viterbi(lo[s + a:s + b], K)
            labels[s + a:s + b] = np.where(st == 1, "H", "A")
    return labels
