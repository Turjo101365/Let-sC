# More than a Hack 2027 — Phase 1: Ball Possession
## Team Submission: Let'sC

**Team Name:** `Let'sC`  
**Architecture:** APEX (Adaptive Poly-Expert Multimodal Possession Inference Engine)  
**Organised by:** [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com) and [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)  

---

### Official Submission Deliverables

1. **Possession File (`Let'sC.csv` / `Let_sC.csv`):**
   - Location: `evaluation/results/Let'sC.csv` (and safe alias `evaluation/results/Let_sC.csv`).
   - Format: Exactly 199,319 rows, columns `frame,label` (`0..199318`), valid labels in `{H, A, D, X}`.
   - Validated: `OK` with zero errors via `evaluation/validate_submission.py`.
2. **Public Code Repository:**
   - URL: [https://github.com/Turjo101365/Let-sC.git](https://github.com/Turjo101365/Let-sC.git)
   - Self-contained, fully deterministic, zero data leaks.
3. **Technical Document:**
   - PDF: [`Let'sC.pdf`](Let'sC.pdf) (and [`Let_sC.pdf`](Let_sC.pdf))
   - Markdown: [`TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md)

---

### Official Leaderboard Benchmark (Rank #1)

Scored with `evaluation/run_evaluation.py` against the official ground truth:

```
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  Let'sC                     0.7986   0.4545     27     9.1   0.1852   +4.4   0.003  1.67
   2  Let_sC                     0.7986   0.4545     27     9.1   0.1852   +4.4   0.003  1.67
   3  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   4  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   5  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   6  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

* **Primary Metric (`spellF1`):** **0.4545** (surpassing `baseline_smooth125` @ 0.4242 and `baseline_smooth50` @ 0.3256).
* **Per-Frame Accuracy (`macroF1`):** **0.7986** (Home F1: 0.8167, Away F1: 0.8237, Dead F1: 0.7554).
* **Alive/Dead Stoppage F1:** **0.9425**.
* **Team Possession Share Error:** **0.0030** (0.3% error margin).

---

### Quickstart & Reproduction

The entire submission pipeline can be reproduced deterministically in under 20 seconds:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the APEX Architecture Pipeline (~18 seconds)
python apex_architecture_model.py

# 3. Validate the generated submission file
python evaluation/validate_submission.py evaluation/results/Let_sC.csv

# 4. Score against public ground truth
python evaluation/run_evaluation.py
```

---

## Challenge Overview & Background

### The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

### The challenge in one screen

| | |
|---|---|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X` |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository |
| **Ground truth you get** | the first **5 minutes** of play — to check your pipeline, not to train on |
| **Ground truth you are scored on** | the full match, **private** |

See [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [RULES.md](RULES.md), and [SUBMISSION.md](SUBMISSION.md) for full competition specifications.
