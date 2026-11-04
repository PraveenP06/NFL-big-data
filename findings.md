# Findings: Combine tracking vs NFL performance for DL and OL

Short notes for writing this up later. Numbers are from the runs logged in `reports/dlol_log.txt` (local, gitignored). `DL.md` and `OL.md` have the longer confirmatory write-ups.

## The question and the sample

We asked whether NFL Combine tracking (10 Hz sensor data from the 40 and position drills) predicts how defensive linemen (DL) and offensive linemen (OL) play in NFL games, beyond the official Combine numbers. Only players drafted in 2023 to 2025 have both Combine tracking and game tracking, so the test sample is 99 DL and 103 OL (84 OL and 78 DL have a measured official 40). Players, not plays, are the sample size for a Combine-to-NFL link, which is why the power is limited.

## Game outcomes are real, stable player traits

Before testing any link we checked that each game outcome is a repeatable trait, by comparing a player's results across random halves of his plays. All four DL outcomes pass easily: NGS get-off (0.86), first-step distance at 0.5 s (0.93) and at 1.0 s (0.89), and speed at 0.5 s (0.91). For OL, snap reaction time (0.93) and first-0.5-second movement (0.93) pass, peak pressure probability allowed is adequate (0.64), and binary pressure allowed fails (0.42) and was dropped. The tracking-based first step agrees with the NGS get-off number (correlation 0.75 to 0.86). This is useful on its own: these are validated, tracking-based measures of linemen that did not exist as player ratings before.

## Combine sensor data: what is trustworthy

The sensor 40 top speed matches the official 40 time almost perfectly (r = -0.98 for DL, -0.90 for OL), so the sensor data is sound. 29 sensor measures are repeatable from attempt to attempt: 40-yard-dash speed, burst, acceleration, deceleration, path and time for both groups, the DL pass-rush drill, and only top speed and path length from the OL pass-rush drop. Most OL pass-rush-drop measures are not repeatable (some are negative). Most agility drills (hoop, body control, four-bag, pull drills, mirror drill, screen drill) were run once per player, so repeatability could not be checked and they were left out.

## Confirmatory result: no detectable link with the big model

The pre-planned test asked whether adding the sensor measures to the traditional Combine numbers improves predictions on players the model has not seen. It does not: 0 of 14 comparisons (8 DL, 6 OL) passed, and every change in out-of-sample R-squared was negative (-0.03 to -0.22). The traditional numbers themselves also did not beat a plain average, and the 40 alone was only faintly positive (R-squared 0.02 to 0.05 for DL get-off and OL first-0.5-second movement). With about 100 players per group, the test could only catch a true extra effect of R-squared 0.2 to 0.3, and then only about half the time. So the correct wording is "no detectable incremental link at this sample size", not "no link exists".

## A bug we caught and fixed

In the first translation run the OL peak-pressure outcome was a constant for every player. The mixed-model optimizer falsely reported "zero player variance" at a boundary, and the shrinkage step then pulled everyone to the mean. Four other optimizers and a classical variance estimate agreed the true value is small but not zero. We changed the tool to fit several optimizers and keep the best-likelihood fit, added tests, regenerated the scores and re-ran with identical settings. The conclusion did not change. The first run's files are kept in `reports/` for transparency.

## First exploratory finding: DL speed and pass-rush production, but mostly role

Simple one-to-one correlations are easier to detect than the big model. Using production per pass rush, the QB-hurry rate and tackle-for-loss rate (both stable enough per player to use) correlate strongly with Combine speed: r = -0.63 with the official 40. That headline number turned out to be mostly position and role, because edge rushers are faster (40 average 4.69 against 5.07) and hurry the quarterback more (10.6% against 6.1% of rushes). After adjusting for where each player lines up, the link for the 40 drops to about -0.16. What held up is the sensor 10-yard burst speed (r about +0.31 for hurry rate and +0.29 for tackle-for-loss rate, role-adjusted).

## Practical version: top third versus bottom third by burst speed

Ranking DL within role by sensor 10-yard burst speed, the top third produced 11.8 QB hurries per 100 pass rushes against 8.3 for the bottom third (gap +3.5, 95% interval +0.4 to +6.5), and 1.5 tackles for loss against 0.9 (gap +0.6, interval +0.1 to +1.1). The pattern holds for both edge and interior linemen and in the 2023 and 2024 draft classes, and it flips in 2025 (only 3 players in the top group). Ranking by the official 40 time gives a similar hurry gap (+3.2) but almost no tackle-for-loss gap (+0.3), so the sensor burst measure adds something the stopwatch time does not.

## Broad scan: peak acceleration in the 40 is the standout for DL

We then scanned every official measure and every repeatable sensor measure against every stable outcome (261 tests, adjusted for weight and role, corrected for the number of tests, and fixed in advance in the log). Three results are supported after correction: sensor 40 peak acceleration with QB-hurry rate (r = +0.42, 95% interval 0.21 to 0.59, q = 0.026), pressure rate (+0.42, q = 0.026) and tackle-for-loss rate (+0.40, q = 0.029). A fourth, burst speed at 1.0 s with tackle-for-loss rate (+0.37), is suggestive (q = 0.08). 26 tests were significant before correction against about 13 expected by chance, and all 21 with a clear football direction point the sensible way (better Combine, better play) while none point the opposite way. Other burst measures (10-yard burst speed, burst distance, 10-yard time) tell the same story at r about 0.27 to 0.32 without surviving the correction.

## What did not work (the negatives)

Offensive linemen: nothing survives the correction. The best OL result is hand size with peak pressure allowed (r = +0.29, q = 0.15), which has no clear football reading. Sack rate and plain tackle rate are too noisy per player to use (reliability 0.53 and 0.36), and for OL the sack-allowed rate is weak (0.36) while quick-pressure and unblocked-pressure events are too rare. Direction of effects: no significant result points the opposite way. Heavier DL get off the ball more slowly (r = +0.26) and a better broad jump goes with a better get-off (-0.24), as expected.

## Caveats to state in the writeup

The scan and the first-round correlations are exploratory, chosen after the confirmatory test came back null, and the confirmatory test remains the official result. The 2025 draft class (about 25 players with production data) does not show the peak-acceleration effect, which is too small a group to confirm or refute it. Pressure and production outcomes also depend on teammates, quarterback, scheme and role, and our adjustment cannot remove all of that. The permutation check in the confirmatory test shuffles outcomes completely, so it tests "no link at all" rather than "sensors add nothing beyond traditional". Combine numbers come from one day, and many drills have a single attempt. Only 2023 to 2025 classes can be linked, so the sample cannot grow without outside data.

## Multi-scope exploratory scans (Nov 4)

We ran five scoped scans (DL, OL, WR/TE, DB, value), each with its plan logged before running, the same stability gate, partial Spearman correlations controlling for weight and role, and a correction within the scope. All are exploratory. The Codex jobs hit their usage limit part-way, so Claude subagents finished four of them against the plans already in the log; the value scope's analysis had finished before the limit.

**DL deep dive (848 tests).** All eight DL outcomes are stable per player (get-off, three first-step measures, QB-hurry, pressure and tackle-for-loss rates, and a new rush-win rate). Nothing is supported after correction (smallest q = 0.45). The earlier lead survives as a consistent but modest correlation: 40-yard peak acceleration with hurry, pressure and rush-win rates at r about +0.36 (n = 77, 95% interval about 0.14 to 0.54). It is not explained by draft slot (controlling for the pick leaves it unchanged), it is no stronger for edge than interior linemen in the role-difference tests, and the 2025 class (about 20 players) shows about zero, which is too small to settle it. Burst index, pass-rush-drill acceleration and 10-yard burst point the same way without surviving. The three rate outcomes are near duplicates, so they are not independent evidence.

**OL deep dive (162 tests).** We built six new tracking-based pass-set outcomes (set depth at 1.0, 1.5 and 2.0 s, lateral shuffle, peak set speed, time to peak) and all are very stable per player (0.77 to 0.92). None relate to any Combine measure (smallest q = 0.91, 4 raw hits against about 8 by chance). Defender-separation outcomes were not possible because the tracking holds only about 1.6 tracked players per play, so the nearest defender cannot be identified. The stable part of these set outcomes may be team scheme rather than the player.

**WR/TE (123 tests).** Separation (overall and against zone), target rate and yards per route are stable; separation against man, catch rate and yards after the catch are not. No athletic measure predicts any receiver outcome (40, top speed, acceleration, burst, vertical and broad jump are all near zero). One result survives: heavier receivers get less separation (r = -0.40 overall, -0.45 against zone, q about 0.001), the same in each draft class. That is a body-size effect (partly WR versus TE), not a speed or burst one.

**DB (100 tests).** Only three DB outcomes are stable per player, and two of them (how often a DB is assigned to coverage and to man coverage) describe how he is used, not how well he plays. Assist rate is the only stable play outcome and nothing relates to it. The play data has no target or receiver fields for DBs, so receiver separation and yards allowed could not be linked, so there is no real coverage-quality outcome. No test is supported (smallest q = 0.95).

**Value: draft slot and careers (721 tests).** Faster OL tend to have a higher share of games started (40 time in drill, shuttle, 40: r about -0.33 to -0.42, same sign in the 2025 hold-out), suggestive but not significant after correction (q = 0.11 to 0.17 within the scope). The sensor data adds no detectable information beyond draft slot (hits when controlling for the draft pick are at chance), so the draft slot appears to already price in what the Combine shows. Few measures relate to draft position. This scope's analysis was fixed by the design in the launch prompt, but the Codex job did not write its own plan to the log before running.

**Master correction (1,426 tests).** Pooling every exploratory test into one family (DL pooled rows, OL, WR/TE, DB, value, and the OL rows from the first broad scan) and correcting once: 94 tests have raw p below 0.05 against about 71 expected by chance, and only 2 survive the global correction (both the WR/TE weight and separation result, q = 0.007 and 0.015). No athleticism result survives. The two leads that stay interesting are the DL peak-acceleration link to pass-rush production (q = 0.37) and the OL faster-40 link to games started (q = 0.22 to 0.37, with the same sign and a larger size in 2025). The table is `reports/master_results.csv` and the script is `scripts/exploratory/master_correction.py`.

## Suggested story for the writeup

Game-side linemen measures can be built from tracking and are stable traits. The Combine's sensor data is sound but only a handful of its measures are repeatable traits. A big model finds no incremental link at n of about 100, and we say how large a sample would be needed. Simple, role-adjusted comparisons point to one real-looking signal for DL: how hard a lineman accelerates and bursts in the first stretch of the 40 goes with pass-rush disruption (hurries, pressures, tackles for loss), while the plain 40 time mostly reflects who plays edge. For OL no Combine measure lined up with how they play in games, although faster OL tend to start more games (a suggestive lead that keeps its sign in 2025). Across the WR/TE, DB and value scans, the only result that survives a global correction is a body-size effect for receivers, so the honest headline is a measurement and sample-size story with a few leads, not a set of confirmed Combine predictors.

## Where the code and outputs are

Pipeline code is in `src/combine_dlol/` with tests in `tests/`. The exploratory scripts are in `scripts/exploratory/` (they read the processed data and write to `reports/`). Local outputs (not on GitHub): `reports/exploratory_scan.csv` (all 261 tests), `reports/exploratory_scan_outcome_gate.csv`, `reports/exploratory_correlations.csv`, `reports/exploratory_robustness.csv`, `reports/translation_*.csv/png`, `reports/phase1_*`, `reports/phase2_*`, and the full timestamped log `reports/dlol_log.txt`.
