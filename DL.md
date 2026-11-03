# Defensive Line (DL): Do Combine measures predict getting off the ball?

## The question

Do NFL Combine speed and burst measures (the 40-yard dash, the 10-yard split, and the pass-rush drill tracked with 10 Hz sensors) predict how fast a pass rusher gets off the ball in games, beyond what size and the traditional Combine numbers already tell us?

## Short answer

**We did not find a detectable link.** The game outcomes are real and very stable per player, and the Combine sensor measures are repeatable. But adding the sensor measures to a model of the traditional numbers made predictions on unseen players worse, not better, in all 8 comparisons. With only 99 players the test could only detect fairly large effects, so this is "not detected", not "does not exist".

## Data and who is in it

- Game side: 32,915 dropback pass-rush rows for DE, DT, NT and OLB players across the 2023-2025 tracking data. 32,822 matched to tracking. 115 players matched, **99 players have 50 or more plays** (32,309 plays), which is the minimum we required.
- Combine side: 118 DL players have Combine rows. **78 of the 99** also have a measured 40.
- The test sample is the 99 players (players, not plays, are what count for a Combine-to-NFL link).

## Game outcomes (all measured from the game data, Combine data not opened)

| Outcome | What it is |
|---|---|
| D-A | NGS `player_get_off` (how quickly the rusher gets off the snap), adjusted for context |
| D-B1 | Downfield displacement 0 to 0.5 s after the snap (tracking, offense-downfield frame) |
| D-B2 | Downfield displacement 0 to 1.0 s after the snap |
| D-B3 | Speed 0.5 s after the snap |

Each is adjusted for lined-up position, down, yards to go, formation, blitz, and alignment depth and width at the snap, then shrunk toward the average for players with fewer plays (empirical Bayes).

**Stability check (the gate).** An outcome is only used if a player's results in one random half of his plays agree with the other half. We required split-half Spearman-Brown reliability of at least 0.50 with a bootstrap lower bound of at least 0.30.

| Outcome | Reliability | 95% interval | Verdict |
|---|---|---|---|
| D-A get-off | 0.86 | 0.78 to 0.91 | adequate |
| D-B 0 to 0.5 s displacement | 0.93 | 0.86 to 0.96 | adequate |
| D-B 0 to 1.0 s displacement | 0.89 | 0.80 to 0.95 | adequate |
| D-B speed at 0.5 s | 0.91 | 0.85 to 0.94 | adequate |

- Permutation nulls (shuffled labels) sit around 0, far from the real values.
- Edge rushers (DE/OLB, 55 players) and interior linemen (DT/NT, 44) both pass.
- The tracking-based first step agrees with the NGS get-off: correlation 0.75 (0 to 0.5 s), 0.86 (0 to 1.0 s), 0.84 (speed). That tells us the two are measuring the same thing.
- Sample plots show rushers starting about 0 to 1 yard past the line and moving into the backfield after the snap, so the field frame is correct.
- I recomputed get-off reliability independently with a simpler adjustment and got 0.91, in line with 0.86.

## Combine predictors

- The sensor 40 top speed matches the official 40 time: **r = -0.98** (n = 93). Faster speed, lower time, as it should.
- **Repeatable** (attempt-to-attempt reliability of at least 0.30 with at least 25 players having 2 or more attempts): 40-yard-dash measures (top speed, burst distance and speed, sensor 10-yard time, acceleration, deceleration, path length and efficiency) and pass-rush-drill measures (top speed, burst, acceleration, deceleration, path length, time).
- **Not repeatable:** pass-rush path efficiency (0.26).
- **Could not be judged:** most agility drills (hoop, body control, four-bag, back-pedal and others), because players ran them once. These are exploratory only and were not used.
- The model with sensor measures (M3) has 21 features: the 9 traditional measures plus 12 repeatable sensor speed, burst and acceleration measures. M4 adds the other repeatable sensor metrics for 26 features.

## The test

Per group we compared models using predictions on players the model had not seen (repeated 5-fold cross-validation, 200 repeats):

- M0: just the average. M1: the 40 only. M2: traditional numbers (40, 10-yard split, 3-cone, shuttle, vertical, broad jump, height, weight, arm length). M3: M2 plus the repeatable sensor speed and burst measures. M4: M3 plus the other repeatable sensor metrics.
- The key number is delta R-squared: how much better M3 or M4 predicts than M2.
- Success was defined in advance as: delta R-squared above 0, the bootstrap interval excluding 0, and a family-wise p under 0.05 across all 14 comparisons (DL and OL together).

### Results (DL)

| Outcome | M3 vs M2 (95% interval) | M4 vs M2 (95% interval) |
|---|---|---|
| D-A get-off | -0.105 (-0.238 to 0.007) | -0.088 (-0.222 to 0.029) |
| 0 to 0.5 s displacement | -0.074 (-0.218 to 0.051) | -0.136 (-0.287 to -0.007) |
| 0 to 1.0 s displacement | -0.087 (-0.221 to 0.020) | -0.172 (-0.325 to -0.052) |
| Speed at 0.5 s | -0.079 (-0.218 to 0.045) | -0.165 (-0.327 to -0.033) |

Family-wise p-values are 0.98 to 1.00. **0 of 8 DL comparisons succeeded.**

Descriptive context (never used to declare success):
- The traditional-numbers model (M2) does worse than the plain average (M0) on every DL outcome (out-of-sample R-squared -0.09 to -0.20 against about -0.02).
- The 40 alone (M1) is slightly positive only for get-off (R-squared 0.021 against -0.022 for the average).
- Stronger or weaker shrinkage, a complete-case subset, elastic net, gradient boosting, adding draft slot, and leaving one draft class out all tell the same story: nothing beats the average reliably.
- I recomputed three of these values with plain scikit-learn as a check: M1 0.021, M2 -0.088, M3 -0.187, against the module's 0.021, -0.090, -0.195.

## Power: what could we have detected?

If one sensor measure really explained a given share of the outcome beyond the traditional numbers, the chance our test would flag it (for one comparison, not the stricter family) was:

| True extra R-squared | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| Chance of detecting (DL) | about 2% | 4 to 9% | 13 to 16% | 26 to 29% | 54 to 59% |

So with 99 players we only had decent odds of finding a large effect. Smaller true effects could easily be hiding.

## Caveats

- Get-off and first-step are stable per player, but they also reflect scheme, stance, alignment and who the player is rushing against. The adjustment cannot remove all of that.
- Only players drafted 2023 to 2025 with Combine data are in the test, so it is small and a bit selected.
- The permutation check shuffles the outcomes completely, so it tests "no Combine link at all" rather than "sensors add nothing beyond traditional". This was written down before any results.
- Many Combine drills were run once per player, so their repeatability could not be judged and they were not used.

## Where the files are

- Code: `src/combine_dlol/dl_outcomes.py`, `combine_predictors.py`, `translation.py` (with tests in `tests/`).
- Frozen plan and every run, with timestamps: `reports/dlol_log.txt` (local, not committed).
- Outputs (local, not committed): `reports/phase1_dl_*.csv/png`, `reports/phase2_combine_*`, `reports/translation_*`.
