# More than a Hack 2027 — Phase 1: Ball Possession
## Submission by Team **Let'sC**

[![Competition](https://img.shields.io/badge/Competition-More%20than%20a%20Hack%202027-blue)](https://github.com/metrica-sports/more-than-a-hack-2027)
[![Team](https://img.shields.io/badge/Team-Let'sC-teal)](https://github.com/Turjo101365/Let-sC)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Frame-by-frame ball possession for FC Barcelona (H) vs Athletic Club (A): one label per frame of the
tactical video — `H`, `A`, `D` (dead ball) or `X` (ungraded) — for all **199,319** frames.

The competition's own documentation is unchanged: [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md),
[EVALUATION.md](EVALUATION.md), [RULES.md](RULES.md), [SUBMISSION.md](SUBMISSION.md).

---

## What APEX does

APEX is a **rule-based** pipeline over the automatic tactical tracking. Nothing is trained and no
labels are used. Full write-up: **[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md)**.

```
tactical ATD ──► ball track (bridge gaps ≤ 4 frames, flag off-pitch)
             ──► proximity evidence per visible-ball frame
                   spatial:     nearest home / away distance (logistic, 3 m)
                   motion:      distance shortened by approach speed over 0.25 s
                   interaction: soft home/away share, exp(-d / 2.5 m)
             ──► dead ball: mean player speed < 1.32 m/s for 60+ frames, or ball off pitch
             ──► controllable ball: faster than 8 m/s over the ground = in flight, no control
             ──► possession: switch after 18 consecutive frames of opponent control (≤ 3 m);
                 hold through missing ball; each restart starts with the team at the ball
             ──► frame,label CSV (deterministic)
```

## How versions were chosen

[RULES.md](RULES.md) forbids using the public 5-minute ground truth to choose between versions, so we
compare versions with [`evaluation/self_check.py`](evaluation/self_check.py). It scores a file
against Metrica's **automatic** Smart Tagging: team agreement inside attacking phases, transition
timing, which team takes each tagged restart, possession durations and flicker, with a paired
bootstrap against a baseline. Every experiment lives in [`experiments/`](experiments/) and replaces
exactly one stage:

| Experiment | Outcome |
| :--- | :--- |
| Restart rule (team at the ball when play resumes) | **Kept** |
| Speed gate (no control of a ball faster than 8 m/s) | **Kept**: team agreement up, flicker down; 2 of 24 tagged transitions lost |
| Backdating confirmed hand-overs | Rejected (lowers team agreement) |
| HMM / Viterbi segmentation | Rejected (no better than hysteresis at equal stability; worse restarts) |
| Majority-vote restarts | Rejected (worse than first sighting) |
| Ball-gap rebuilding | No effect (the provided ball track has no plausible short gaps) |
| Broadcast fusion | Not possible (broadcast tracking file not in our download) |
| Contact-aware touches from the ball's path | Rejected (ball-track noise floor above most touches; removed no false hand-overs) |
| Set-piece tags as restart input | Rejected (small held-out gain, split case by case; not independently validated) |
| LightGBM trained on external IDSSE Bundesliga matches (CC BY 4.0) | Not adopted: better on held-out Bundesliga matches, but transfer to Metrica tracking unproven ([OVERFITTING_REPORT.md](OVERFITTING_REPORT.md), [external data](experiments/EXTERNAL_DATA.md)) |

The submitted pipeline is the 8 m/s speed-gate model. Numbers, failure cases and the decision rule
are in section 5 of the technical document; contact and tag details are in
[experiments/CONTACT_RESULTS.md](experiments/CONTACT_RESULTS.md).

**Public-window sanity check**, scored only after each decision (17 possessions; differences
this size are within noise, and the speed gate is a tie here):

| File | spellF1 | macroF1 |
| :--- | :---: | :---: |
| **Let'sC (submitted)** | 0.4842 | 0.8058 |
| Previous version (restart rule only) | 0.4878 | 0.8052 |
| baseline_smooth125 | 0.4242 | 0.5812 |
| baseline_smooth50 (published floor) | 0.3256 | 0.6207 |

There is no live leaderboard; the official score is computed by the organisers on the full match.

---

## Reproduce the submission

```bash
git clone https://github.com/Turjo101365/Let-sC.git
cd Let-sC
pip install -r requirements.txt
```

Unpack the match data into `data/` (see [data/README.md](data/README.md)). Only the tactical files
are needed:

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml   (self-check only)
```

```bash
python apex_architecture_model.py
python evaluation/validate_submission.py "evaluation/results/Let'sC.csv"
python evaluation/self_check.py "evaluation/results/Let'sC.csv"
python evaluation/run_evaluation.py
```

`apex_architecture_model.py` writes `evaluation/results/Let'sC.csv` and an identical
`Let_sC.csv` (no apostrophe) in about 6 s. The output is deterministic, with Unix line endings on
every operating system — SHA-256
`78d1e7c05f4d9cd062281d0bf98016f1f100d7efac7c6382e86f4c1074c3b1af`.

To rerun every experiment: `python experiments/variants.py`, then
`python evaluation/self_check.py data/cache/variants/*.csv --baseline data/cache/variants/base.csv`.

To rebuild the technical document PDF from the Markdown: `python export_pdf.py`.

---

## The three deliverables

| Deliverable | Where |
| :--- | :--- |
| Possession file | `evaluation/results/Let'sC.csv`, regenerated by the command above (not committed: results are gitignored) |
| Public code repository | https://github.com/Turjo101365/Let-sC |
| Technical document | [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md); `Let'sC.pdf` is built from it by `export_pdf.py` |

## Repository map

| Path | What |
| :--- | :--- |
| `apex_architecture_model.py` | The submitted pipeline |
| `evaluation/self_check.py` | Label-free comparison against Smart Tagging, with paired bootstrap |
| `experiments/` | Audit, isolated variants, failure cases, contact and tag-restart experiments |
| `tests/` | Synthetic pass, interception, deflection and restart scenes (`python -m unittest discover -s tests`) |
| `export_pdf.py` | Renders `TECHNICAL_DOCUMENT.md` to PDF |
| `evaluation/`, `starter/`, `games.py` | Competition scorer, validator and starter kit |

## Rules compliance

- **No hand labels** on the provided game; every label comes from the code above.
- **No ground truth used for training or selection.** The public window was used for format checks and a sanity check after each decision.
- **Deterministic.** Re-running the pipeline reproduces the submitted file exactly.
- **No match data in the repository.** `data/` and `evaluation/results/*.csv` are gitignored.

---
*Team Let'sC — More than a Hack 2027.*
