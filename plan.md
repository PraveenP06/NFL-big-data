# Plan: Cut Recovery (Combine to NFL translation)
NFL Big Data Bowl 2027. Deadline: Jan 6, 2027, 11:59 PM UTC.

## Question
Does the cut-and-recover skill seen at the Combine show up in NFL games, and does it add anything beyond the 40, 3-cone, shuttle, height and weight?

## Deliverables
1. Game-side cut recovery metric for WRs (route breaks) and DBs (coverage breaks) from NFL game tracking.
2. Evidence the metric is a stable trait (split-half reliability).
3. Translation test: Combine measures (40, 10-yd split, shuttle, 3-cone, Combine cut metric, WR/DB drills) vs game cut recovery.
4. Secondary test: does it predict separation at pass forward beyond route, cushion and coverage?
5. Public Kaggle notebook + writeup (<= 2,000 words, < 10 figures/tables).

Out of scope: causal claims, OL/DL skills (other teammates), combined athleticism score unless components prove stable.

## Data facts that shape the plan
- 510 rookies (2023-25). Game tracking is NFL only (no college), 10 Hz, rookies only.
- 106 WRs, all with Combine route tracking. 115 DBs with game data, 122 with Combine DB drills.
- Only 5 receivers have two comeback attempts in Andrew's extractor; Combine cut reliability is unproven.
- Only rookies are tracked, so nearest-defender features are incomplete. About 8.9K man-coverage plays have both a rookie DB and rookie receiver tracked.
- Rookie-year routes only: about 62 WRs with >= 50 measured routes. Holdout 2025 is about 23-30 receivers.

## Phases (compressed to a few days)

### Phase 0: Setup and data (day 1 morning)
- Python env (pandas, pyarrow, scipy, statsmodels); CSVs to Parquet; raw data stays out of git (CC BY-NC).
- Download game tracking 2023-25. Loader joining player_play, games, players, combine_results.

### Phase 1: Game-side cut detection, WRs (day 1)
- WR routes with route_ran, no nullifying penalty, snap to pass_forward window.
- Normalize direction with play_direction; detect break from heading change over ~0.5 s.
- Metrics: entry speed, min speed, speed-loss fraction, 0.5 s speed gain, time to 90% (flag censored).
- Split by route type (IN, OUT, SLANT, CORNER, POST, HITCH).
- GATE: manually inspect trajectory plots per route type.

### Phase 2: Reliability (day 2 morning)
- Odd/even split-half per player, adjusted for entry speed, angle, route; empirical-Bayes shrinkage for low volume.
- GATE: drop or restate as null any route type with poor reliability.

### Phase 3: Combine side (day 2)
- Reuse Andrew's comeback/curl cut extractor (credit source; aschemmel3/NFL-Big-Data-Bowl-Cut-Recovery).
- Add 40, 10-yd split, 3-cone, shuttle, height, weight, route drills. Check Combine repeatability.

### Phase 4: Translation test (day 2-3)
- Regress game cut recovery on Combine measures, player-grouped CV, position controls.
- Four models on identical players: 40 only; traditional; traditional + Combine cut; traditional + all drills.
- Develop on 2023-24, keep 2025 untouched until design is fixed.
- Report explained variance vs reliability ceiling, player-level bootstrap CIs, family-wise permutation correction, list every test run.

### Phase 5: Separation as secondary outcome (day 3)
- Model separation_at_pass_forward with route, cushion, coverage, motion, formation, time to throw.
- Test added value of game cut recovery. Never treat missing separation as zero; report missingness.

### Phase 6: DB extension (day 3-4, cuttable)
- DB breaks on man-coverage plays from the DB's own movement (coverage_assignment).
- Link to Combine DB drills (backpedal-and-break, W-drill, box, gauntlet).
- Exploratory matchup test where rookie receiver is also tracked.
- GATE: if reliability or coverage is weak, keep as a short exploratory section.

### Phase 7: Visuals and writeup (day 4)
- Notebook with code in appendix.
- Figures: route-break speed curves, translation strength vs reliability ceiling, field heatmap of recovery by route depth (<10, 10-20, 20+), "hidden talent" table (descriptive only).

## Dependencies
Kaggle access (done), game tracking download (about 2 GB), Andrew's repo for the Combine extractor and agreement to merge approaches, memory-aware processing by season.

## Risks
- HIGH: Combine cut metric may be unreliable. Mitigate: agree up front that "the Combine can't see cut recovery" is a valid finding; lean on 40/shuttle/3-cone too.
- HIGH: small samples. Mitigate: pre-register hypotheses, intervals, shrinkage.
- MEDIUM: game cut detection noisy (coverage, QB timing). Mitigate: route-specific detectors, plot review, sensitivity grids.
- MEDIUM: incomplete defender tracking (rookies only).
- MEDIUM: playing-time confounds. Mitigate: route minimums, role controls.
- LOW: coordination with Andrew's pipeline.

## Complexity: HIGH
About 10 working days of effort for one person at normal pace; compressed to roughly 4 days by cutting Phase 6 if needed. Core path: Phases 0-5 and 7.
