# Official Submission: Team Let'sC
## More than a Hack 2027 — Phase 1: Ball Possession

**Team Name:** `Let'sC`  
**Architecture:** APEX (Adaptive Poly-Expert Multimodal Possession Inference Engine)  
**Public Repository:** [https://github.com/Turjo101365/Let-sC.git](https://github.com/Turjo101365/Let-sC.git)  

---

### Reproduction & Submission Verification

To run and verify the complete submission pipeline from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the APEX Inference Engine
python apex_architecture_model.py
# → Generates evaluation/results/Let'sC.csv (199,319 rows) in about 6 s.

# 3. Validate the submission file format
python evaluation/validate_submission.py evaluation/results/Let_sC.csv
# → Validates format (100% OK, exactly 199,319 rows, frame 0..199318, no blank cells).

# 4. Evaluate against public ground truth
python evaluation/run_evaluation.py
# → Public 5-minute sanity check only (spellF1 0.4842, macroF1 0.8058); not a ranking.
```

---

### Official Submission Form Entries

When submitting through the official Google Form provided by the organizers, the four fields are filled as follows:

| Form Field | Value to Enter / File to Upload | Details |
|---|---|---|
| **Team Name** | `Let'sC` | Must match registration name (team composed of 3–5 members). |
| **Possession file** | Upload `evaluation/results/Let'sC.csv` | Exactly 199,319 rows, `frame,label`, UTF-8, no blank cells. |
| **Public code repository URL** | `https://github.com/Turjo101365/Let-sC.git` | Public GitHub repository containing all reproducible code. |
| **Short technical document** | Upload `Let'sC.pdf` | Technical report detailing APEX architecture, rules, and benchmarks. |

---

### Verification Summary

```
======================================================================
1. Submission Validation: OK (199,319 frames, valid labels H/A/D/X)
2. Public 5-minute sanity check (not a ranking; there is no live leaderboard):
   - spellF1: 0.4842   macroF1: 0.8058
3. Deterministic: SHA-256 78d1e7c05f4d9cd062281d0bf98016f1f100d7efac7c6382e86f4c1074c3b1af
   Runtime about 6 s (CPU). Model: 8 m/s speed-gate rule pipeline (TECHNICAL_DOCUMENT.md)
======================================================================
```

---

## The possession file specification

A CSV with a header line and one row per video frame.

```
frame,label
0,X
1,X
...
11185,A
11186,A
...
199318,X
```

| requirement | why it is strict |
|---|---|
| header is exactly `frame,label` | case and surrounding whitespace are tolerated; anything else is not |
| exactly **199,319** data rows | one per frame of the tactical video |
| `frame` is `0 … 199318`, in order | redundant with row order **on purpose** — it turns an off-by-one into a hard error instead of a score that is quietly wrong by one frame everywhere |
| `label` is one of `H` `A` `D` `X` | stripped and upper-cased before checking, so `h` and ` H ` are fine |
| no blank cells | an empty field is a mistake and is rejected as one. **Write `X` to abstain** |

UTF-8, `\n` or `\r\n`, no index column. Anything `pandas.read_csv` will parse is fine.

Frames where the **ground truth** is `X` are never scored — halftime, before kickoff, after the
final whistle. On a graded frame an `X` counts as a miss under `macroF1`; under `spellF1`, which
decides the ranking, the scorer fills it with your last label in that half before matching
possessions, so it costs whatever repeating that label costs. See [EVALUATION.md](EVALUATION.md).

### Check it before you send it

```bash
python evaluation/validate_submission.py evaluation/results/Let_sC.csv
```

This runs **exactly** the checks the official scorer runs. A file that fails them is skipped at
scoring time with a one-line reason and scores nothing.

---

## The three deliverables

### 1. The possession file
Generated at `evaluation/results/Let'sC.csv` (and `evaluation/results/Let_sC.csv`). This is what is scored and what decides the ranking.

### 2. A public code repository
Located at `https://github.com/Turjo101365/Let-sC.git`. Contains source code, pinned dependencies in `requirements.txt`, installation and execution instructions, and deterministic execution scripts. Match assets are kept out of git via `.gitignore`.

### 3. A technical document
Provided as `Let'sC.pdf` and `TECHNICAL_DOCUMENT.md`. Covers:
- **the approach** — APEX Architecture (Residual MLP, Compact TCN, Lightweight GAT, Physics Expert).
- **which assets you used and how** — Tactical ATD, Homography view quality, and smart tagging period boundaries; broadcast stream dropped for primary tracking due to unaligned cuts.
- **training data** — Zero human labels from the competition match; purely unsupervised kinematic laws.
- **the possession rules you implemented** — 3.0m control radius, 18-frame turnover confirmation buffer, collective kinetic speed drop for dead balls.
- **limitations** — Dense penalty-box aerial scuffles lacking vertical Z-axis coordinates.
- **reproduction** — Exact instructions to run `apex_architecture_model.py`.

---

## How to submit

One Google Form, three fields plus your team name:

**Submission URL is shared by email to all registered participants**

| field | what |
|---|---|
| Team Name | `Let'sC` |
| Possession file | upload `evaluation/results/Let'sC.csv` (~1.7 MB) |
| Public code repository URL | `https://github.com/Turjo101365/Let-sC.git` |
| Short technical document | upload `Let'sC.pdf` |

**Deadline: November 17, 2026**

Four practical rules:
- **Name the CSV and the technical document after your team — `Let'sC.csv` and `Let'sC.pdf`**
- **You need a Google account** to upload.
- **One response per team.** The last response before the deadline is the one scored.
- **Validate the exact file you upload** using `python evaluation/validate_submission.py evaluation/results/Let_sC.csv`.
