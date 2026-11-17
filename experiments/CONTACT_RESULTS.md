# Contact-aware possession and set-piece restarts: results

Baseline: the submitted model (speed gate 8 m/s + 18-frame hysteresis + restart rule), SHA-256
`78d1e7c0...`. Reproduce everything below with:

```
python -m unittest tests/test_contact.py -v
python experiments/contact_variants.py
```

No ground truth was read. Attacking-phase Smart Tags were never model inputs (held out for every
variant). Set-piece tags were inputs only to the `+tagrestart` variants.

**Decision: keep the submitted model.** Neither idea has enough independent evidence.

## 1. Contact-aware model (`experiments/contact.py`)

Touches are read from the ball's path. A fast ball (over 8 m/s) arriving within 2 m of a player
counts as touched when its velocity vector changes by more than `tau` times its speed. `tau` is
the tracking-noise floor: the same relative change for fast balls with nobody within 5 m.
Interceptions (an opponent kills the ball and out-controls the other team for 1 s), one-touch
interceptions (two consecutive opponent touches), teammate receptions and contested touches are
handled as specified. With every option off, the decoder reproduces the submitted file exactly.

**The ball track is too noisy for this.** The noise floor (n = 8,360 fast-ball frames with nobody
within 5 m) is a relative velocity change of 0.39 at p95 and **0.73 at p99**. Near players, the
signal is only about twice the noise (median 0.24 vs 0.12). At the honest p99 floor the detector
finds 61 touches in 106 minutes of live play (1.0 per minute); at p95 it finds 466 (7.3 per minute)
at a 5% false-alarm rate per candidate frame.

| Variant | Team agree (held out) | Tagged transitions | Hand-overs | Ping-pongs | Under 2 s | Possessions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (submitted) | 0.776 | 18/24 | 294 | 37 | 14.3% | 391 |
| Interception rule | 0.769 | 18/24 | 298 | 36 | 14.1% | 398 |
| Two-consecutive-touch rule | 0.776 | 18/24 | 296 | 38 | 14.5% | 393 |
| Teammate reset | 0.776 | 18/24 | 294 | 37 | 14.3% | 391 |
| All contact rules (p99, 2 m) | 0.769 | 18/24 | 300 | 37 | 14.2% | 400 |
| All rules, contact radius 1.5 / 2.5 m | 0.769 | 18/24 | 300 / 302 | 37 | 14.0 / 14.2% | 399 / 401 |
| All rules, p95 floor | 0.767 | **19/24** | 310 | 43 | 15.6% | 410 |

Paired bootstrap vs baseline, team agreement: all rules -0.020 to 0.000; p95 floor -0.026 to +0.002.

- **False hand-overs:** none removed. Hand-over counts change by 0 to +16 across the contact variants.
- **Tagged transitions:** only the p95 variant recovers one, the interception at 37:30. In
  exchange it adds 16 hand-overs and 6 ping-pongs, and moves the hand-over near 104:05 from 13
  frames after its tag to 63 frames before it: a false early turnover.
- **Restarts:** unchanged (74% on the 27 team-tagged set pieces).
- **Cost:** touch detection plus decoding takes about 0.04 s; negligible.

## 2. Set-piece tags as restart input

Each live stretch starting inside a team-tagged corner, goal kick or free kick window takes the
tagged team. 24 of 27 tags map to a restart, and 13 of those change the restart team.

| Variant | Team agree (held out) | Restart, cross-validated | Restart vs input tags (not independent) | Hand-overs |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 0.776 | 0.741 | n/a | 294 |
| + tag restarts | 0.782 | 0.741 | 0.926 | 296 |

Team agreement +0.001 to +0.013 (paired bootstrap).

- **Cross-validation by half cannot measure this method.** A tag changes only its own restart, so
  on the held-out half the model is the baseline. The 0.926 is the model copying its input.
- **Case by case, the independent evidence is split.** On the 13 changed restarts:
  - tracking control in the next 5 s agrees with the tag 5 times and with the baseline 5 times,
    and has no evidence 3 times;
  - held-out attacking-phase tags in the next 15 s agree with the set-piece tag 1 time in 5.
- **Caveats.** It is unclear whether a corner tag's Team is the taker or the defending side. Both
  kinds of tag come from the same automatic tagger, so their errors may be correlated.

## 3. Synthetic tests (`tests/test_contact.py`, 11 tests, all pass)

Pass received by a teammate · fast fly-by 1 m from an opponent (no touch, no hand-over) ·
interception handed over within 3 frames of the touch (the baseline is later) · deflection back
to the passing team (no hand-over) · one-touch interception to a teammate (hand-over at the first
touch) · contested touch (ignored by the contact rules) · restart to the player at the ball ·
set-piece tag overriding a restart · tag-to-restart mapping. The one-touch test caught an indexing
bug in the decoder, now fixed.

## What would change the conclusion

Touch detection needs a cleaner ball track: ball detection from the video itself, or ball height.
With the provided automatic tracking, the noise floor is above most real touches.
