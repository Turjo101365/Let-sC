# Technical Document: APEX Possession Inference

**Team:** Let'sC · **Competition:** More than a Hack 2027, Phase 1 (Ball Possession) · **Match:** FC Barcelona (H) vs Athletic Club (A)

---

## 1. Approach

APEX is a **rule-based** system. Nothing in it is trained, and no labels of any kind are used to produce the file. It reads the automatic tactical tracking and turns the geometry of the players and the ball into one label per frame through five fixed steps:

1. **Proximity evidence.** On every frame where the ball is visible we take the nearest home and nearest away player. Three views of that distance are computed: the raw distance through a logistic with a 3 m scale; an *approach-adjusted* distance, shortened by how fast the player is closing on the ball over 0.25 s; and a soft home/away share of the two approach-adjusted distances (exp(-d / 2.5 m)). We call these the spatial, motion and interaction "experts", but they are formulas, not neural networks.
2. **Dead ball.** The ball is dead when the 50-frame moving average of mean player speed stays below 1.32 m/s for at least 60 frames (2.4 s), or when the tracked ball is outside the 105.3 x 68 m pitch.
3. **Controllable ball.** A ball moving faster than **8 m/s** over the ground (measured from a 5-frame centred average of its position) is a pass, shot or clearance in flight. Nobody can be controlling it on that frame, so the frame carries no evidence, exactly as if the ball were missing. 8 m/s is roughly the top speed of a player running with the ball.
4. **Possession with hysteresis.** A team is *controlling* a frame when the ball is controllable, its nearest player is within 3 m and closer than the opponent. Possession changes only after the other team has been controlling for **18 consecutive evidence frames** (frames without evidence neither count nor break the streak). When there is no evidence, possession stays where it was.
5. **Restart rule.** Every stretch of live play after a stoppage, and each half's kick-off, starts with the team of the player nearest the ball at its first sighting (within 2 s), instead of inheriting the team from before the stoppage.

**Why rules rather than a model.** The only labels on this game are the public 5 minutes, which the rules forbid for training or tuning, and when the rules were designed we had no labelled data from other games. A transparent rule set we can reason about was the honest option within that constraint. A learned model trained later on external matches was tested but not adopted (section 5, experiment C).

---

## 2. Assets used and dropped

| Asset | Use | Decision |
| :--- | :--- | :--- |
| Tactical ATD (`_atd.txt`, `_atd.xml`) | Player and ball positions and speeds; team of each track slot | **Used.** It is on the submission's frame timeline. |
| Tactical Smart Tagging | Label-free validation only (section 5) | **Used for checking, not as model input**, so it stays an independent check. |
| `games.py` half boundaries | Which frames are graded; second-half kick-off | **Used.** |
| Tactical homography | Considered as a per-frame reliability weight | **Dropped.** It was computed in an earlier version but never fed into a decision. |
| Broadcast ATD and homography | Considered for multi-view fusion | **Not used.** Its own unsynchronised timeline, and the broadcast tracking file was not in our download. |
| Videos | Not used | Not needed by the pipeline. |
| IDSSE Bundesliga data (external, other matches, CC BY 4.0) | Validation and training of experimental models | **Experiments only** (section 5, experiment C); not used by the submitted pipeline. |

---

## 3. Training data

**The submitted model uses no labelled data.** It is a fixed rule set; nothing in it was trained.

**External labelled data, used only in experiments that were not adopted:** the IDSSE dataset (Bassek et al., *Scientific Data* 2025). It contains seven 2022/23 Bundesliga matches, i.e. other games, with frame-level ball-status and possession labels from the official provider (DFL). Source: figshare doi:10.6084/m9.figshare.28196177; licence CC BY 4.0. It was used to validate experiments and to train candidate models (section 5, experiment C). See `experiments/EXTERNAL_DATA.md`. No IDSSE file is redistributed in this repository.

**No human data about the provided game was used.** The public 5-minute ground truth was used only to check that the file is aligned and well-formed, and for a sanity check after each decision (section 6). It did not choose between versions or settings.

The numeric settings of the original pipeline (3 m control radius, 18-frame confirmation, 1.32 m/s and 60-frame dead-ball rule, 0.25 s approach horizon) were set by hand in an earlier iteration and were not re-tuned. The 8 m/s control speed was set from the physical argument above; 6 and 10 m/s were run as a sensitivity check (section 5).

---

## 4. Possession rules implemented

- **Passes, shots and clearances in flight.** A fast ball (over 8 m/s) gives no team control, so a pass skimming past an opponent does not start a hand-over.
- **Loose balls.** Possession stays with the last controlling team until the opponent controls the ball for 18 evidence frames.
- **Challenges, deflections and touches.** A short spell of the opponent being nearest does not change possession.
- **Occlusion.** While the ball is not tracked, possession is held. Streaks are not reset by missing frames.
- **Restarts.** The team nearest the ball when play resumes owns the first possession after any stoppage.
- **Dead ball.** Always `D`, regardless of who will restart. Frames between the second-half tracking start and the kick-off are `D`.
- **Outside the halves.** `X` (ungraded).

---

## 5. How versions were compared without labels

RULES.md forbids choosing between versions on the public window, so every version is compared with `evaluation/self_check.py` against Metrica's **automatic** Smart Tagging (no human review). In the tags, `BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION` and `CREATING CHANCES` name the team in possession; the `DEFENDING IN ...` codes are exact mirrors. The script reports team agreement inside those phases, hand-overs found near the 24 `ATTACKING TRANSITION` tags, the team after 27 team-tagged corners, goal kicks and free kicks, switches per minute, and two tag-free signals: possession durations and "ping-pongs" (H-A-H with a middle spell under 2 s). A paired bootstrap over tags gives 95% intervals against the baseline.

**Decision rule**, fixed before the experiments: keep a change only if it improves the like-for-like signals (team agreement, restart accuracy, switches per minute) without lowering any of them. One caveat applies to team agreement: smoother files score higher on it, because a tagged phase mostly belongs to one team. So variants were also compared at equal switch rates.

### Audit of the previous version (restart rule only)

- **False hand-overs.** 113 of 491 possessions lasted under 2 s; **80 of them sat between two hand-overs** (H-A-H flicker).
- **Lateness.** Hand-overs are confirmed a median 17 frames (p90: 39) after the opponent first took control, with the ball visible for 99% of that wait.
- **Ball gaps are long, not short.** The raw ball track has only 186 gaps in the whole match, with a median of 230 frames. In live play, gaps under 3 s cover 1.9% of frames.
- **Restarts.** At 39 of 140 restarts no team was found in the first 2 s.

### Experiments (`experiments/variants.py`, each replacing one stage)

| Variant | Team agree | Restart right | Transitions found | Switches /min | Possessions | Under 2 s | Ping-pongs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Previous (restart rule) | 0.730 | 63% | 83% | 5.94 | 491 | 23% | 93 |
| Gap rebuild (12 / 25 frames) | identical | identical | identical | identical | 491 | 23% | 93 |
| Restart by majority, 2 s | 0.723 | 56% | 83% | 6.06 | 489 | 23% | 91 |
| Restart by majority, 3 s | 0.722 | 52% | 83% | 6.01 | 483 | 23% | 90 |
| HMM (Viterbi), K = 25 | 0.726 | 52% | 83% | 5.14 | 420 | 15% | 49 |
| HMM, K = 37 | 0.730 | 48% | 79% | 4.05 | 332 | 9% | 17 |
| Speed gate 10 m/s | 0.758 | 70% | 79% | 5.21 | 433 | 19% | 62 |
| **Speed gate 8 m/s (submitted)** | **0.776** | **74%** | 75% | **4.56** | 391 | 14% | 37 |
| Speed gate 6 m/s | 0.803 | 74% | 71% | 3.83 | 345 | 12% | 20 |
| Speed gate 8 + backdating | 0.765 | 78% | 79% | 4.58 | 391 | 15% | 42 |
| Speed gate 8 + HMM K = 25 | 0.770 | 56% | 79% | 3.73 | 339 | 12% | 28 |

Paired bootstrap, speed gate 8 m/s vs previous: team agreement +0.028 to +0.068, switches per minute -1.87 to -0.91 (both exclude zero); restart accuracy -0.07 to +0.30 and transitions found -0.21 to 0.00 (neither excludes zero).

**Gap rebuilding: no effect.** All 11 mid-length gaps (5-25 frames) end with the ball reappearing at an implied 30-110 m/s, i.e. a detector jump rather than a short loss. The plausibility check (at most 30 m/s) correctly refuses them, so nothing changes.

**Majority-vote restarts: rejected.** Worse than the first sighting: the restart taker is first to the ball, and a longer window picks up the duel that follows.

**HMM: rejected.** Viterbi segmentation with a per-hand-over cost K, hand-overs placed at the middle of the evidence-free gap. It trades flicker against missed transitions along one curve. At an equal switch rate it agrees with the tags less than the speed gate does (0.730 vs 0.803 at about 4 switches per minute), and its restart decisions are worse.

**Speed gate: kept, at 8 m/s.** All three thresholds improve team agreement and restart accuracy and reduce flicker. At an equal switch rate it beats the HMM clearly, so the gain is not just smoothing. We took 8 m/s from the physical argument rather than the 6 m/s that scores best, because team agreement rewards smoothing. **Cost:** 2 of 24 tagged transitions are no longer found; both are interceptions where the ball was still fast for about 37% of the surrounding second.

**Backdating: rejected again.** With the speed gate it finds more transitions and restarts but still lowers team agreement, which the decision rule does not allow.

### Remaining failure cases of the submitted version

- **7 team-tagged restarts still get the wrong team** (3 corners, 4 goal kicks): an opponent is nearest the ball at first sighting.
- **37 ping-pongs remain,** all with the ball visible: real duels where one side is nearest for 18+ frames.
- **Interceptions of fast passes** are found late or not at all (above).

### Further experiments, not adopted

Three more ideas were tested against the speed-gate model: A and B with the same decision rule and the same label-free evidence, C with external labelled matches. **None was adopted: the submitted pipeline is the 8 m/s speed-gate model described in section 1.** Code: `experiments/contact.py` and `experiments/contact_variants.py`; synthetic tests: `tests/test_contact.py`; full numbers: `experiments/CONTACT_RESULTS.md`.

**A. Contact-aware possession.**

- **Hypothesis.** Touches read from the ball's path could separate a fast ball flying past an opponent from a genuine interception. That would recover the fast interceptions the speed gate misses, without bringing back false hand-overs.
- **Method.** A touch is a fast ball (over 8 m/s) within 2 m of a player whose velocity vector changes by more than a threshold `tau`. `tau` is measured, not chosen: it is the 99th percentile of the same change for fast balls with nobody within 5 m, i.e. the tracking-noise floor. Rules on top of the speed-gate model:
  - **interception** — an opponent's touch kills the ball and that team out-controls the other over the next second;
  - **one-touch interception** — two consecutive opponent touches;
  - **teammate reception** — a touch by the team in possession resets any opponent streak;
  - **contested touch** — both teams within 0.5 m of each other; ignored.

  With these rules off, the decoder reproduces the submitted file exactly. Eleven synthetic scenes (pass, fly-by, interception, deflection, one-touch interception, contested ball, restarts) all pass.
- **Result.** The automatic ball track is too noisy for this. The noise floor is a relative velocity change of 0.73 at p99, so only 61 touches were detected in 106 minutes of live play.
  - No contact variant found more tagged transitions (all 18/24), and none removed a hand-over (changes ranged from 0 to +8).
  - Team agreement was unchanged or slightly lower: 0.769 to 0.776 vs 0.776. With all rules on, the paired-bootstrap interval was -0.020 to 0.000.
  - Lowering the threshold to the noise p95 recovered one interception (19/24), but added 16 hand-overs and 6 ping-pongs, and produced a turnover 63 frames before its tag.
- **Rejected:** it removed no false hand-overs, and its only recovered turnover came with several new false ones.

**B. Set-piece tags as restart input.**

- **Hypothesis.** The Smart Tags' team for corners, goal kicks and free kicks could fix restarts where an opponent stands nearest the ball.
- **Method.** A live stretch that starts inside a team-tagged set piece takes the tagged team. This applies to 24 of the 27 tags and changes 13 restarts. Set-piece tags are then model inputs, so restart accuracy against them is not independent. We relied instead on held-out attacking-phase tags, on 2-fold cross-validation by half, and on a case-by-case check against the tracking.
- **Result.**
  - Held-out team agreement rose from 0.776 to 0.782 (interval +0.001 to +0.013).
  - Cross-validation cannot measure the change: a tag alters only its own restart, so on the held-out half the model equals the baseline (0.741 both ways).
  - On the 13 changed restarts, ball control in the tracking over the next 5 s sided with the tag 5 times and with the baseline 5 times (3 had no evidence). The held-out phase tags in the next 15 s sided with the set-piece tag in 1 of 5 cases.
  - It is also unclear whether a corner tag's team is the taker or the defender, and both tag types come from the same automatic tagger.
- **Rejected:** the aggregate gain is small and not supported case by case by independent evidence.

**C. External labelled validation and a learned model (IDSSE).**

- **Hypothesis.** Labelled matches from another league could measure real accuracy, and a supervised model trained on them could beat the rules.
- **Method.**
  - Seven IDSSE matches, split strictly by whole match: leave-one-match-out cross-validation on five for every choice, and two test matches evaluated once after the configuration was frozen.
  - Tracking degraded to BAR-ATH's measured, label-free noise statistics (ball visibility runs, detector jumps, player dropout).
  - Candidates: the submitted rule, a retuned rule, logistic regression, and LightGBM on tracking-derived features with a Viterbi decoder.
  - Scored with the official scorer. Full report: `OVERFITTING_REPORT.md`.
- **Result.**
  - **On held-out Bundesliga matches, LightGBM beat the submitted rule:** spellF1 0.382 vs 0.293 in cross-validation, 0.373 vs 0.307 on the two test matches, ahead on every match and seed.
  - **No overfitting was detected,** and a shuffled-label test ruled out leakage.
  - **The submitted rule's hand-overs are about 14 frames late on real labels.**
  - **Domain shift is moderate.** A classifier separates IDSSE from Metrica tracking with AUC 0.755.
  - **On BAR-ATH, LightGBM was mixed:** it found more tagged turnovers but agreed significantly less with held-out team tags, and flickered more.
- **Not adopted:** the gain is validated on Bundesliga data, but not convincingly transferable to the competition's automatic tracking.

---

## 6. Sanity check on the public window

Scored with `evaluation/run_evaluation.py` only after each decision above was made. The window holds 17 possessions; EVALUATION.md shows that differences of this size there can reverse on the full match, so these are not a ranking.

| File | spellF1 | macroF1 | Possessions (GT 17) |
| :--- | :---: | :---: | :---: |
| **Submitted (speed gate + restart rule)** | 0.4842 | 0.8058 | 21 |
| Previous (restart rule) | 0.4878 | 0.8052 | 24 |
| Original APEX | 0.4545 | 0.7986 | 27 |
| baseline_smooth125 | 0.4242 | 0.5812 | 16 |
| baseline_smooth50 (published floor) | 0.3256 | 0.6207 | 26 |

On this window the speed gate is a tie with the previous version. The evidence for it is the full-match, label-free comparison in section 5.

---

## 7. Limitations

- **Dead-ball rule is a speed threshold.** Quick restarts with players still moving are missed, and slow spells of live play can be marked dead.
- **Fast interceptions.** Control only counts once the ball has slowed, so an interception of a hard pass is confirmed late.
- **No height information.** Aerial duels and bouncing balls in crowded boxes are judged by planar distance only.
- **Fixed lateness.** Hand-overs are reported when confirmed, at least 18 evidence frames after they start.
- **Restarts near opponents.** Corners and goal kicks where an opponent stands nearest the ball get the wrong team.
- **Hand-set constants.** No threshold was tuned on labelled data. Evaluated untuned on external Bundesliga labels, the rule scores below a learned model there (section 5, experiment C).

**With more time:** a cleaner ball track (ball detection from the video itself) would make contact detection viable, since the automatic track's noise floor is above most real touches (section 5, experiment A). Set-piece restarts need an independent way to validate them before the tags can be trusted as input (experiment B). Labelled data closer to Metrica's automatic tracking would let a learned model, or tuned thresholds, be validated for this match.

---

## 8. Reproduction

```
pip install -r requirements.txt
# unpack the match data into data/ (see data/README.md)
python apex_architecture_model.py                 # writes evaluation/results/Let'sC.csv
python evaluation/validate_submission.py "evaluation/results/Let'sC.csv"
python evaluation/self_check.py "evaluation/results/Let'sC.csv"
python experiments/variants.py                    # section 5 experiments, into data/cache/
python experiments/contact_variants.py            # experiments A and B, into data/cache/
python -m unittest discover -s tests              # synthetic contact and restart scenes
```

Runs in about 6 s on a laptop CPU. The output is deterministic and written with Unix line endings on every operating system; the submitted file's SHA-256 is `78d1e7c05f4d9cd062281d0bf98016f1f100d7efac7c6382e86f4c1074c3b1af`. `--control-speed 0` reproduces the previous version's labels.
