"""Apply IDSSE-trained candidates to the competition match and compare label-free.

Models are trained on the 5 CV matches only (never the 2 TEST matches, never any BAR-ATH label).
Output goes to data/cache/transfer/ -- the submitted CSV is never touched. Comparison uses
evaluation/self_check.py (automatic Smart Tags), which is a proxy, not accuracy.

    python experiments/transfer_barath.py
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import common  # noqa: E402
import idsse  # noqa: E402
import learn  # noqa: E402
import overfit_study as S  # noqa: E402
import research  # noqa: E402

OUT = os.path.join(ROOT, "data", "cache", "transfer")


def main(seed=0):
    cv = pd.read_csv(os.path.join(S.OUT, "cv.csv"))
    tuned = json.loads(cv[cv.candidate == "rule_tuned"].config.iloc[0])
    Ks = {k: int(pd.Series([json.loads(c)["K"] for c in cv[cv.candidate == k].config]).mode()[0])
          for k in ("logreg", "lgbm")}

    raw = common.load_raw()
    bar = dict(raw, pitch=np.asarray(raw["pitch"]))
    h1, h2, h2t = common.halves()
    segs = [(h1[0], h1[1], research.A), (h2[0], h2[1], research.H)]
    prefill = [(h2t, h2[0], "D")]
    prep8 = research.prepare(bar)
    rule = research.decode(prep8, segs, research.control(prep8), prefill=prefill)
    X, names = learn.features(prep8, rule, bar["pitch"])

    os.makedirs(OUT, exist_ok=True)
    files = {}

    def write(name, labels):
        path = os.path.join(OUT, name + ".csv")
        with open(path, "w", newline="\n") as f:
            f.write("frame,label\n")
            f.writelines("{},{}\n".format(i, x) for i, x in enumerate(labels))
        files[name] = path

    write("rule_frozen", rule)
    prep_t = research.prepare(bar, control_speed=tuned["control_speed"])
    write("rule_tuned", research.decode(prep_t, segs, research.control(prep_t, margin=tuned["margin"]),
                                        switch_frames=tuned["switch_frames"], boundary=tuned["boundary"],
                                        prefill=prefill))
    for kind, K in Ks.items():
        model = S.fit_on(kind, idsse.CV, seed, learn.DEFAULT_GROUPS)
        Xs, _ = learn.select(X, names, learn.DEFAULT_GROUPS)
        p = learn.predict_home(model, Xs)
        lab = learn.decode_probs(p, prep8["dead"], [(s, e) for s, e, _ in segs], K, prep8["n"])
        lab[h2t:h2[0]] = "D"
        write(kind, lab)

    submitted = os.path.join(ROOT, "evaluation", "results", "Let'sC.csv")
    same = pd.read_csv(files["rule_frozen"])["label"].equals(pd.read_csv(submitted)["label"])
    print("rule_frozen reproduces the submission:", same, "| K used:", Ks, "| tuned rule:", tuned)
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "self_check.py")]
                   + list(files.values()) + ["--baseline", files["rule_frozen"]], check=True)


if __name__ == "__main__":
    main()
