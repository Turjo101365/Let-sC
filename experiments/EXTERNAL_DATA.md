# External data: IDSSE Bundesliga matches

Used **only** to train and validate experimental models on *other* matches (RULES.md: "Hand-labelled
data from other games is fine, in any amount"). No competition match data is redistributed, and no
IDSSE file is committed to this repository.

| | |
| :--- | :--- |
| Dataset | An integrated dataset of spatiotemporal and event data in elite soccer (IDSSE) |
| Authors | Bassek, M., Rein, R., Weber, H., Memmert, D. |
| Publication | *Scientific Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y |
| Files | figshare, doi:[10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177) |
| Licence | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Data owner | Deutsche Fußball Liga (DFL), released with DFL authorisation |
| Content | 7 matches, 2022/23 Bundesliga and 2. Bundesliga; TRACAB Gen5 tracking at 25 Hz |

## What is used, and how labels are formed

From each match we use only the match-information XML (team ids, pitch size) and the
positions XML (players, ball). Event files are not used.

Every ball frame carries `BallStatus` (0 = dead, 1 = alive) and `BallPossession` (1 = home,
2 = away). Ground truth is formed exactly as the competition describes its own: **D** where the
ball is dead, otherwise the team the tracking names as in possession.

These labels come from the official data provider for these other matches. The paper documents
the tracking pipeline (computer vision, operator-resolved identity swaps, quality control); it
does not state how possession was assigned.

## Split (strict, by whole match)

| Role | Matches |
| :--- | :--- |
| Leave-one-match-out cross-validation: all development and selection | J03WMX, J03WN1, J03WOH, J03WOY, J03WPY |
| Final test: untouched until the configuration is frozen, evaluated once | J03WQQ, J03WR9 |

## Getting the files

```
python experiments/download_idsse.py      # 14 files, about 2.6 GB, into data/external/idsse/
```

`data/` is gitignored, so the files never enter the repository. Parsed matches are cached in
`data/cache/idsse/` (also gitignored).

## Attribution

Contains data from the IDSSE dataset (Bassek et al., 2025), © Deutsche Fußball Liga, licensed
under CC BY 4.0. Derived figures in `reports/` are aggregate statistics; no tracking is
reproduced.
