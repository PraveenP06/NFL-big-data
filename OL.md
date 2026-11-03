# Offensive Line (OL): Do Combine measures predict pass protection?

## The question

Do NFL Combine speed and movement measures (the 40-yard dash and the pass-rush-drop drill, tracked with 10 Hz sensors) predict how well an offensive lineman reacts at the snap and how much pressure he allows in games, beyond size and the traditional Combine numbers?

## Short answer

**We did not find a detectable link.** Two of the OL game outcomes (snap reaction and first-half-second movement) are very stable per player, and peak pressure allowed is moderately stable. But adding the sensor measures to a model of the traditional numbers made predictions on unseen players worse in all 6 comparisons. With only 103 players the test could only detect fairly large effects, so this is "not detected", not "does not exist". Pressure allowed is also the hardest outcome to read as an individual skill.

## Data and who is in it

- Game side: 57,614 pass-play rows for tackles, guards and centers from the 2023 to 2025 data, 120 players. 56,627 matched to tracking. **103 players have 50 or more plays** (56,168 plays).
- Combine side: 121 OL players have Combine rows. **84 of the 103** also have a measured 40.
- The test sample is the 103 players.

## Game outcomes

| Outcome | What it is | Source |
|---|---|---|
| O-A | `peak_pressure_probability_allowed`, the highest chance of pressure the lineman allowed on the play | NGS, adjusted |
| O-A2 | `pressure_allowed` (yes or no) | NGS, adjusted, secondary |
| O-B1 | Snap reaction time: seconds from the snap until the player first moves (0.25 yards, or speed 0.75 yards per second) | tracking |
| O-B2 | Movement in the first 0.5 s after the snap | tracking |

All are adjusted for lined-up position, down, yards to go, formation, dropback duration and pass rushers encountered, then shrunk toward the average for players with fewer plays.

For snap reaction we dropped plays where the lineman was already moving before the snap and flagged the 0.7% of plays where no movement was seen within 1.0 s. Tracking is 10 frames per second, so reaction times come in steps of 0.1 s (median 0.3 s).

**Stability check (the gate).** Same rule as DL: split-half reliability of at least 0.50 with a bootstrap lower bound of at least 0.30.

| Outcome | Reliability | 95% interval | Verdict |
|---|---|---|---|
| Snap reaction time | 0.93 | 0.88 to 0.96 | adequate |
| First 0.5 s movement | 0.93 | 0.88 to 0.96 | adequate |
| Peak pressure probability allowed | 0.64 | 0.43 to 0.78 | adequate |
| Pressure allowed (yes or no) | 0.42 | 0.10 to 0.65 | weak, **dropped** from the test |

- Permutation nulls are around 0.
- Tackles and guards/centers separately: all adequate except peak pressure for guards and centers (0.60, lower bound 0.30, just under the line), which we only reported as secondary.
- The sample snap-reaction plots mark first movement where the displacement curve leaves zero, around 0.2 to 0.4 s after the snap.

### A bug I caught

In the first translation run, the OL peak-pressure outcome turned out to be a **constant for all 103 players**. The mixed model that sets how much to shrink each player toward the average falsely reported "zero player variance" because one optimizer stopped at a boundary and claimed it had converged. Four other optimizers and a classical variance estimate all agree the real value is small but not zero (reliability of an average player about 0.007 per play). The shrinkage then collapsed everyone to one number, so the first two OL peak-pressure comparisons were meaningless. I fixed the tool to try several optimizers and keep the best fit, added tests, regenerated the scores, and re-ran the identical frozen test. The split-half reliability and verdicts did not change. The first run's files are kept for transparency (`reports/translation_results_run1_degenerate_peak_pressure.csv`).

## Combine predictors

- The sensor 40 top speed matches the official 40 time: **r = -0.90** (n = 95).
- **Repeatable:** 40-yard-dash measures (speed, burst, acceleration, deceleration, path, time) and, from the pass-rush-drop drill, only top speed (reliability 0.37) and path length.
- **Not repeatable:** most pass-rush-drop measures (burst, acceleration, deceleration, path efficiency, time), several with negative attempt-to-attempt reliability.
- **Could not be judged:** most agility drills (pull drills, mirror drill, wave drill, screen drill) because players ran them once.
- M3 has 17 features (9 traditional plus 8 repeatable sensor speed and burst measures). M4 has 21.

## The test

The same design as DL: out-of-sample delta R-squared, M3 and M4 against the traditional model (M2), 200 repeated cross-validations, 1000 permutations, one family across all 14 comparisons (DL and OL together). Success required delta above 0, an interval excluding 0, and a family-wise p under 0.05.

### Results (OL, after the fix)

| Outcome | M3 vs M2 (95% interval) | M4 vs M2 (95% interval) |
|---|---|---|
| Peak pressure probability allowed | -0.050 (-0.110 to 0.004) | -0.169 (-0.359 to -0.039) |
| Snap reaction time | -0.100 (-0.244 to 0.010) | -0.215 (-0.477 to -0.054) |
| First 0.5 s movement | -0.028 (-0.125 to 0.042) | -0.115 (-0.268 to -0.020) |

Family-wise p-values are 0.71 to 1.00. **0 of 6 OL comparisons succeeded.** The closest is first-0.5s movement with M3 (-0.028, p = 0.71).

Descriptive context:
- The traditional-numbers model does worse than the plain average on every OL outcome (R-squared -0.12 to -0.26 against about -0.02).
- The 40 alone is slightly positive only for first-0.5s movement (0.033 against -0.022 for the average).
- Other shrinkage strengths, complete cases, elastic net, gradient boosting, adding draft slot, and leaving one draft class out all agree.

## Power: what could we have detected?

For a single comparison, the chance our test would flag a real extra effect:

| True extra R-squared | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| Chance of detecting (OL) | 2 to 5% | 15 to 17% | 23 to 29% | 42 to 53% | 72 to 76% |

With 103 players we could only find fairly large effects.

## Caveats

- **Pressure allowed is not purely the lineman.** It also depends on teammates, the quarterback (how long he holds the ball), scheme and the opposing rusher. The adjustment helps but cannot remove this. Team-level differences in the average residual are about half as large as player-level ones (SD 0.016 against 0.034), so it is not just team effects, but it is not clean either. Snap reaction and first-half-second movement are closer to the player himself.
- Snap reaction and first-0.5s movement come from the same movement trace, so they are correlated, not independent evidence.
- Only players drafted 2023 to 2025 with Combine data are in the test.
- The permutation check shuffles outcomes completely, so it tests "no Combine link at all", not strictly "sensors add nothing beyond traditional".
- Many Combine drills were run once per player, so their repeatability could not be judged.

## Where the files are

- Code: `src/combine_dlol/ol_outcomes.py`, `combine_predictors.py`, `reliability.py` (with the fix), `translation.py` (tests in `tests/`).
- Frozen plan and every run, with timestamps: `reports/dlol_log.txt` (local, not committed).
- Outputs (local, not committed): `reports/phase1_ol_*`, `reports/phase2_combine_*`, `reports/translation_*`.
