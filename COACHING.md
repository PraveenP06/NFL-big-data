# Which traits are worth training time? Coaching-unit results

This is exploratory work on 2023-2025 draftees. Everything below is an association, not a cause. Full tables: `reports/coaching_effects.csv` (692 rows), `reports/coaching_group_rollup.csv`, `reports/coaching_trainability.csv`, `reports/coaching_wr_block.csv`.

## How it was done (5 lines)
1. Each measure is tagged trainable, partially trainable or fixed, and given a realistic one-season improvement written down before looking at results (for example 0.10 s in the 40, 2 in in the vertical, 3 bench reps; sensor measures use half a standard deviation).
2. For each position group (DL pooled with an EDGE/IDL control, OL, DB, WR, TE) and each game outcome that passed the stability check, we fit a straight line adjusting for weight and role, and report the change in the outcome per realistic improvement (positive = better play), with a 2,000-draw player bootstrap 95% interval.
3. Each result is also expressed as a fraction of the spread between players on that outcome (SD) and compared with a smallest effect worth caring about, 0.20 SD. Verdicts: supported benefit, rules out effects larger than X, or cannot say.
4. A measure is "prioritize" if some outcome shows a clear benefit and the 2025 class points the same way, "deprioritize" if every tested outcome rules out effects above 0.20 SD, otherwise "unclear". Fixed body measures are listed as recruiting-only.
5. Evidence grades: supported = survives the project-wide multiple-testing correction; suggestive = nominally clear and not overturned, but not corrected; speculative = everything else. Rules were logged in `reports/dlol_log.txt` before any slope was computed (two small amendments are logged after the first run).

## Headline
- Of 692 measure-by-outcome rows, 540 could be tested (at least 30 players with all values); 152 could not (all of TE, and rarely-taken drills).
- Only 2 rows earn the grade "supported", and both are the same finding: heavier receivers are associated with less separation. For WR, 5 lb heavier goes with 0.023 yards less separation at the pass (95% CI -0.034 to -0.013, n=80) and 0.029 yards less in zone coverage (-0.043 to -0.016, n=74); the sign held in both 2023-24 and 2025.
- 33 more rows are "suggestive". Nominal p<0.05 occurred in 39 of 540 tests; chance alone would give about 27. Clear benefits and clear harms were equally common (15 each), which is what noise looks like.
- Most tested rows either rule out effects larger than 0.20 SD (285) or say "cannot say" (220). With about 100 players per group the data can rule out large effects but not small ones.

## Trainability and the improvement unit used
| Measure | Tag | Realistic improvement |
|---|---|---|
| forty | partially_trainable | 0.10 s faster |
| ten_yd_split | trainable | 0.05 s faster |
| three_cone | trainable | 0.15 s faster |
| short_shuttle | trainable | 0.10 s faster |
| vertical | trainable | 2 in higher |
| broad_jump | trainable | 4 in farther |
| bench_reps | trainable | 3 more reps |
| combine_weight | partially_trainable | 10 lb heavier (DL/OL), 5 lb heavier (DB/WR/TE) |
| combine_height | fixed | 0.5 SD of the measure within position group |
| arm_length | fixed | 0.5 SD of the measure within position group |
| wing_span | fixed | 0.5 SD of the measure within position group |
| hand_size | fixed | 0.5 SD of the measure within position group |
| forty_yard_dash_top_speed | partially_trainable | 0.5 SD of the measure within position group |
| forty_yard_dash_burst_speed_1_0s | trainable | 0.5 SD of the measure within position group |
| forty_yard_dash_sensor_ten_yd_time | trainable | 0.5 SD of the measure within position group |
| forty_yard_dash_sensor_ten_yd_burst_speed | trainable | 0.5 SD of the measure within position group |
| forty_yard_dash_peak_acceleration | partially_trainable | 0.5 SD of the measure within position group |
| forty_yard_dash_peak_deceleration | partially_trainable | 0.5 SD of the measure within position group |
| forty_yard_dash_path_efficiency | partially_trainable | 0.5 SD of the measure within position group |
| forty_yard_dash_time_in_drill | partially_trainable | 0.5 SD of the measure within position group |
| short_shuttle_time_in_drill | trainable | 0.5 SD of the measure within position group |
| three_cone_drill_time_in_drill | trainable | 0.5 SD of the measure within position group |

Sensor measures (40-yard top speed, 10-yard time and burst speed, peak acceleration and deceleration, path efficiency, shuttle and cone drill times, route-drill top speed) use 0.5 SD within position group; weight uses 10 lb for DL/OL and 5 lb for DB/WR/TE (judgment call). Sources and one-line reasons are in the CSV; anything not backed by literature is marked "judgment call".

## Results by position group
Effects are the associated change in the outcome per improvement, in outcome units, with the SD fraction in brackets. "Outcome shown" is the clearest outcome for prioritize rows, the weakest rule-out for deprioritize rows (so the other outcomes rule out at least as much), and the largest estimate for unclear rows.

### DL (EDGE and interior pooled)
**prioritize** (2)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| forty_yard_dash_sensor_ten_yd_time | 0.5 SD (= 0.0478) | tfl_rate | 0.000189 (0.114 SD) | 1.7e-06 to 0.000412 | 77 | suggestive |
| forty_yard_dash_peak_acceleration | 0.5 SD (= 0.278) | rush_win_rate | 0.00191 (0.124 SD) | 3.23e-06 to 0.00449 | 77 | suggestive |

**unclear** (16)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| forty | 0.10 s faster | hurry_rate | 0.00247 (0.153 SD) | -0.000162 to 0.00477 | 78 | speculative |
| ten_yd_split | 0.05 s faster | pressure_rate | 0.00348 (0.227 SD) | -0.000109 to 0.00681 | 78 | speculative |
| three_cone | 0.15 s faster | shrunk_d_b_displacement_1_0 | 0.0226 (0.194 SD) | -0.00834 to 0.0586 | 37 | speculative |
| short_shuttle | 0.10 s faster | shrunk_d_b_displacement_1_0 | 0.0286 (0.23 SD) | 0.000557 to 0.06 | 39 | suggestive |
| vertical | 2 in higher | shrunk_d_a | 0.00357 (0.0886 SD) | -0.00305 to 0.00907 | 87 | speculative |
| broad_jump | 4 in farther | pressure_rate | 0.00284 (0.2 SD) | -2.07e-05 to 0.00552 | 84 | speculative |
| bench_reps | 3 more reps | rush_win_rate | 0.00446 (0.284 SD) | 0.000547 to 0.00809 | 46 | suggestive |
| combine_weight | 10 lb heavier | shrunk_d_a | -0.0068 (-0.169 SD) | -0.0107 to -0.00261 | 99 | suggestive |
| forty_yard_dash_top_speed | 0.5 SD (= 0.303) | hurry_rate | 0.0021 (0.13 SD) | -0.00152 to 0.00502 | 77 | speculative |
| forty_yard_dash_burst_speed_1_0s | 0.5 SD (= 0.259) | rush_win_rate | 0.00164 (0.106 SD) | -0.000394 to 0.004 | 77 | speculative |
| forty_yard_dash_sensor_ten_yd_burst_speed | 0.5 SD (= 0.121) | tfl_rate | 0.00018 (0.109 SD) | -8.78e-06 to 0.000418 | 77 | speculative |
| forty_yard_dash_peak_deceleration | 0.5 SD (= 0.212) | pressure_rate | 0.000921 (0.0596 SD) | -0.000525 to 0.00313 | 77 | speculative |
| forty_yard_dash_path_efficiency | 0.5 SD (= 0.000717) | shrunk_d_b_displacement_0_5 | -0.00509 (-0.0826 SD) | -0.00856 to 0.0139 | 77 | speculative |
| forty_yard_dash_time_in_drill | 0.5 SD (= 0.126) | rush_win_rate | 0.00239 (0.155 SD) | -0.000439 to 0.00508 | 77 | speculative |
| short_shuttle_time_in_drill | 0.5 SD (= 0.202) | shrunk_d_b_displacement_0_5 | 0.00775 (0.119 SD) | -0.0174 to 0.0237 | 32 | speculative |
| three_cone_drill_time_in_drill | 0.5 SD (= 0.306) | tfl_rate | 0.000448 (0.281 SD) | 8.98e-05 to 0.000817 | 40 | suggestive |

### OL
**prioritize** (2)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| forty | 0.10 s faster | set_time_to_peak_speed_0_1_5s | 0.00315 (0.124 SD) | 4.79e-05 to 0.00667 | 84 | suggestive |
| combine_weight | 10 lb heavier | set_time_to_peak_speed_0_1_5s | 0.0046 (0.181 SD) | 0.000902 to 0.00956 | 103 | suggestive |

**deprioritize** (5)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| forty_yard_dash_top_speed | 0.5 SD (= 0.174) | set_time_to_peak_speed_0_1_5s | rules out > 0.175 SD | -0.000375 to 0.00446 | 84 | speculative |
| forty_yard_dash_sensor_ten_yd_time | 0.5 SD (= 0.0607) | set_time_to_peak_speed_0_1_5s | rules out > 0.175 SD | -0.00444 to 1.58e-05 | 84 | speculative |
| forty_yard_dash_peak_acceleration | 0.5 SD (= 0.313) | set_time_to_peak_speed_0_1_5s | rules out > 0.176 SD | -0.00447 to 0.000352 | 84 | speculative |
| forty_yard_dash_peak_deceleration | 0.5 SD (= 0.165) | snap_reaction_time | rules out > 0.198 SD | -0.00794 to 0.00165 | 84 | speculative |
| forty_yard_dash_time_in_drill | 0.5 SD (= 0.109) | snap_reaction_time | rules out > 0.166 SD | -0.00206 to 0.00665 | 84 | speculative |

**unclear** (11)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| ten_yd_split | 0.05 s faster | first_0_5s_displacement | 0.00571 (0.12 SD) | -0.00278 to 0.0161 | 84 | speculative |
| three_cone | 0.15 s faster | lateral_shuffle_0_1_5s | -0.0126 (-0.0962 SD) | -0.0419 to 0.0172 | 52 | speculative |
| short_shuttle | 0.10 s faster | lateral_shuffle_0_1_5s | -0.0125 (-0.106 SD) | -0.0295 to 0.00542 | 55 | speculative |
| vertical | 2 in higher | set_time_to_peak_speed_0_1_5s | 0.00215 (0.0888 SD) | -0.000862 to 0.00589 | 88 | speculative |
| broad_jump | 4 in farther | set_peak_speed_0_1_5s | 0.01 (0.125 SD) | -0.00475 to 0.0231 | 87 | speculative |
| bench_reps | 3 more reps | lateral_shuffle_0_1_5s | 0.0166 (0.168 SD) | 0.000243 to 0.0362 | 44 | suggestive |
| forty_yard_dash_burst_speed_1_0s | 0.5 SD (= 0.315) | set_time_to_peak_speed_0_1_5s | -0.00292 (-0.115 SD) | -0.00475 to -0.00059 | 84 | suggestive |
| forty_yard_dash_sensor_ten_yd_burst_speed | 0.5 SD (= 0.148) | set_time_to_peak_speed_0_1_5s | -0.00259 (-0.102 SD) | -0.00441 to -6.09e-05 | 84 | suggestive |
| forty_yard_dash_path_efficiency | 0.5 SD (= 0.00158) | set_time_to_peak_speed_0_1_5s | -0.00175 (-0.0686 SD) | -0.00625 to 0.00156 | 84 | speculative |
| short_shuttle_time_in_drill | 0.5 SD (= 0.384) | pass_set_depth_1_0s | 0.0128 (0.108 SD) | -0.00504 to 0.0455 | 46 | speculative |
| three_cone_drill_time_in_drill | 0.5 SD (= 0.574) | first_0_5s_displacement | 0.00332 (0.0779 SD) | -0.00388 to 0.0154 | 40 | speculative |

### DB
Only three DB outcomes pass the stability check (coverage assignment rate, man-coverage assignment rate, assist rate) and the first two have no better-or-worse direction, so they cannot earn "supported benefit".

**deprioritize** (6)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| forty_yard_dash_burst_speed_1_0s | 0.5 SD (= 0.353) | coverage_assignment_rate | rules out > 0.157 SD | -0.00397 to 0.00976 | 62 | speculative |
| forty_yard_dash_sensor_ten_yd_time | 0.5 SD (= 0.123) | assist_rate | rules out > 0.156 SD | -0.00125 to 0.000665 | 90 | speculative |
| forty_yard_dash_sensor_ten_yd_burst_speed | 0.5 SD (= 0.177) | coverage_assignment_rate | rules out > 0.155 SD | -0.00424 to 0.00964 | 62 | speculative |
| forty_yard_dash_peak_deceleration | 0.5 SD (= 0.231) | coverage_assignment_rate | rules out > 0.174 SD | -0.00468 to 0.0108 | 62 | speculative |
| forty_yard_dash_path_efficiency | 0.5 SD (= 0.00366) | coverage_assignment_rate | rules out > 0.194 SD | -0.0121 to 0.0089 | 62 | speculative |
| forty_yard_dash_time_in_drill | 0.5 SD (= 0.129) | coverage_assignment_rate | rules out > 0.138 SD | -0.00771 to 0.00857 | 62 | speculative |

**unclear** (9)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| forty | 0.10 s faster | man_coverage_assignment_rate | 0.00849 (0.0792 SD) | -0.0109 to 0.0263 | 63 | speculative |
| ten_yd_split | 0.05 s faster | assist_rate | -0.000686 (-0.0868 SD) | -0.00238 to 0.00126 | 92 | speculative |
| short_shuttle | 0.10 s faster | assist_rate | 0.000845 (0.0962 SD) | -0.000768 to 0.00241 | 30 | speculative |
| vertical | 2 in higher | coverage_assignment_rate | 0.00688 (0.109 SD) | -0.00245 to 0.0164 | 60 | speculative |
| broad_jump | 4 in farther | coverage_assignment_rate | 0.00754 (0.122 SD) | -0.00428 to 0.0205 | 55 | speculative |
| bench_reps | 3 more reps | coverage_assignment_rate | -0.0138 (-0.228 SD) | -0.0344 to 0.00798 | 30 | speculative |
| combine_weight | 5 lb heavier | coverage_assignment_rate | 0.00659 (0.112 SD) | -0.0043 to 0.0181 | 73 | speculative |
| forty_yard_dash_top_speed | 0.5 SD (= 0.138) | coverage_assignment_rate | -0.0041 (-0.0658 SD) | -0.013 to 0.00401 | 62 | speculative |
| forty_yard_dash_peak_acceleration | 0.5 SD (= 0.252) | coverage_assignment_rate | 0.00688 (0.11 SD) | 0.000702 to 0.0124 | 62 | suggestive |

**not tested (fewer than 30 complete cases)** (3): three_cone, short_shuttle_time_in_drill, three_cone_drill_time_in_drill

### WR
**prioritize** (3)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| vertical | 2 in higher | separation_at_pass_forward | 0.022 (0.179 SD) | 0.00524 to 0.0386 | 72 | suggestive |
| route_slot_sail_top_speed | 0.5 SD (= 0.29) | target_rate | 0.0026 (0.0886 SD) | 8.75e-05 to 0.00604 | 73 | suggestive |
| route_slot_strike_top_speed | 0.5 SD (= 0.305) | target_rate | 0.00328 (0.108 SD) | 0.000635 to 0.00606 | 75 | suggestive |

**deprioritize** (12)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| forty_yard_dash_top_speed | 0.5 SD (= 0.141) | separation_zone | rules out > 0.174 SD | -0.013 to 0.0293 | 67 | speculative |
| forty_yard_dash_burst_speed_1_0s | 0.5 SD (= 0.31) | target_rate | rules out > 0.162 SD | -0.00425 to 0.00269 | 72 | speculative |
| forty_yard_dash_sensor_ten_yd_time | 0.5 SD (= 0.0532) | rec_yards_per_route | rules out > 0.157 SD | -0.045 to 0.0301 | 72 | speculative |
| forty_yard_dash_sensor_ten_yd_burst_speed | 0.5 SD (= 0.138) | target_rate | rules out > 0.162 SD | -0.00424 to 0.0028 | 72 | speculative |
| forty_yard_dash_peak_deceleration | 0.5 SD (= 0.179) | rec_yards_per_route | rules out > 0.194 SD | -0.0162 to 0.0553 | 72 | speculative |
| forty_yard_dash_time_in_drill | 0.5 SD (= 0.0679) | rec_yards_per_route | rules out > 0.174 SD | -0.0498 to 0.0286 | 72 | speculative |
| route_comeback_top_speed | 0.5 SD (= 0.135) | rec_yards_per_route | rules out > 0.183 SD | -0.0259 to 0.0598 | 71 | speculative |
| route_curl_top_speed | 0.5 SD (= 0.153) | separation_zone | rules out > 0.168 SD | -0.0048 to 0.0276 | 70 | speculative |
| route_dagger_top_speed | 0.5 SD (= 0.146) | target_rate | rules out > 0.193 SD | -0.00048 to 0.00572 | 80 | speculative |
| route_over_shoulder_adjust_top_speed | 0.5 SD (= 0.296) | separation_zone | rules out > 0.178 SD | -0.0295 to 0.000493 | 71 | speculative |
| route_slant_top_speed | 0.5 SD (= 0.362) | separation_zone | rules out > 0.133 SD | -0.0126 to 0.0221 | 74 | speculative |
| route_speed_out_top_speed | 0.5 SD (= 0.208) | separation_zone | rules out > 0.174 SD | -0.0288 to 0.00665 | 74 | speculative |

**unclear** (9)

| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |
|---|---|---|---|---|---|---|
| forty | 0.10 s faster | rec_yards_per_route | 0.0279 (0.0991 SD) | -0.0542 to 0.108 | 74 | speculative |
| ten_yd_split | 0.05 s faster | separation_at_pass_forward | 0.0377 (0.27 SD) | -0.0238 to 0.105 | 74 | speculative |
| short_shuttle | 0.10 s faster | rec_yards_per_route | 0.0879 (0.323 SD) | -0.011 to 0.185 | 30 | speculative |
| broad_jump | 4 in farther | separation_at_pass_forward | 0.0193 (0.158 SD) | -0.000376 to 0.0385 | 73 | speculative |
| combine_weight | 5 lb heavier | separation_zone | -0.0293 (-0.177 SD) | -0.0428 to -0.0156 | 74 | supported |
| forty_yard_dash_peak_acceleration | 0.5 SD (= 0.444) | rec_yards_per_route | 0.0426 (0.149 SD) | -0.0126 to 0.0656 | 72 | speculative |
| forty_yard_dash_path_efficiency | 0.5 SD (= 0.00167) | rec_yards_per_route | -0.0211 (-0.0737 SD) | -0.0697 to -0.00533 | 72 | suggestive |
| route_go_top_speed | 0.5 SD (= 0.15) | rec_yards_per_route | 0.0228 (0.069 SD) | -0.0175 to 0.0685 | 71 | speculative |
| route_red_zone_fade_top_speed | 0.5 SD (= 0.247) | rec_yards_per_route | -0.0394 (-0.121 SD) | -0.0759 to 0.000623 | 72 | speculative |

**not tested (fewer than 30 complete cases)** (5): three_cone, bench_reps, short_shuttle_time_in_drill, three_cone_drill_time_in_drill, route_circus_top_speed

### TE
No TE measure had 30 players with all values (best case 27), so nothing is tested.

**not tested (fewer than 30 complete cases)** (26): forty, ten_yd_split, three_cone, short_shuttle, vertical, broad_jump, bench_reps, combine_weight, forty_yard_dash_top_speed, forty_yard_dash_burst_speed_1_0s, forty_yard_dash_sensor_ten_yd_time, forty_yard_dash_sensor_ten_yd_burst_speed, forty_yard_dash_peak_acceleration, forty_yard_dash_peak_deceleration, forty_yard_dash_path_efficiency, forty_yard_dash_time_in_drill, short_shuttle_time_in_drill, three_cone_drill_time_in_drill, route_block_explosion_top_speed, route_corner_top_speed, route_end_zone_fade_top_speed, route_flat_top_speed, route_hook_top_speed, route_in_top_speed, route_over_shoulder_adjust_top_speed, route_wheel_top_speed

Fixed measures (height, arm length, wing span, hand size) were fitted for every group and are in the CSV flagged recruiting-only; they are not ranked. In DL, longer wingspan, height and arm length went with slightly lower pressure rates (about -0.1 SD per half SD), which is suggestive-grade at most.

## WR speed and agility: what we can and cannot rule out
Outcomes are context-adjusted player scores; "SD" is the spread of that score between WRs. The median smallest effect we could detect with 80% power for separation was 0.14 SD, so only effects at least that large were findable.

| Measure | Improvement | n | separation at pass | target rate | yards per route |
|---|---|---|---|---|---|
| forty | 0.10 s faster (forty) | 74 | cannot say (-0.22 to +0.32 SD) | cannot say (-0.24 to +0.37 SD) | cannot say (-0.19 to +0.38 SD) |
| ten_yd_split | 0.05 s faster (ten_yd_split) | 74 | cannot say (-0.17 to +0.76 SD) | cannot say (-0.41 to +0.48 SD) | cannot say (-0.41 to +0.45 SD) |
| short_shuttle | 0.10 s faster (short_shuttle) | 30 | cannot say (-0.32 to +0.11 SD) | cannot say (-0.28 to +0.54 SD) | cannot say (-0.04 to +0.68 SD) |
| three_cone | n/a | 0 | not tested (n<30) | not tested (n<30) | not tested (n<30) |
| forty_yard_dash_top_speed | 0.5 SD (0.141) of forty_yard_dash_top_speed | 72 | rules out > 0.17 SD | rules out > 0.14 SD | rules out > 0.16 SD |
| forty_yard_dash_burst_speed_1_0s | 0.5 SD (0.31) of forty_yard_dash_burst_speed_1_0s | 72 | rules out > 0.10 SD | rules out > 0.16 SD | rules out > 0.15 SD |
| forty_yard_dash_sensor_ten_yd_burst_speed | 0.5 SD (0.138) of forty_yard_dash_sensor_ten_yd_burst_speed | 72 | rules out > 0.09 SD | rules out > 0.16 SD | rules out > 0.14 SD |

- The 40 sensor speed and burst measures rule out effects above about 0.1 to 0.17 SD on every WR outcome. The official 40 and 10-yard split cannot say: their intervals include benefits of 0.2 SD or more.
- Route-drill top speed: 24 of 30 tests rule out effects above 0.20 SD. Two show a clear benefit on target rate (slot sail and slot strike drills, suggestive grade, about 0.09 to 0.11 SD per half SD of drill speed); these are 2 of 30 tests, so treat them as leads. Most route drills are single-attempt, so repeatability is unverified.
- Short shuttle: n=30, cannot say. Three-cone: fewer than 30 WRs, not tested.
- Left/right-side ability could not be tested. Each route drill is run to one side only (post-corner is the only drill with both sides, 65 players, and its top speed is not repeatable), and game outcomes were not split or stability-checked by side.

## How not to misuse this
- This is not a cut-or-keep tool and must not be used to evaluate an individual player.
- Correlation is not causation. Training a trait is not shown to produce these differences; better players may simply test well, and teams with stronger programs may both train and play better.
- The samples are small (30 to about 100 players per test, draft classes 2023-2025). Most results are null or inconclusive, and "rules out" means "not larger than X in this sample".
- Drafted players only: draft pedigree, playing time, role and program quality all sit behind the numbers. A "weight_confounded" flag in the CSV marks effects that shrink by more than 40% when weight or draft pick is added; 27 tested rows carry it.
- Some outcomes depend on the project's assumed direction (for example lower is better for DL get-off scores); outcomes with no clear direction are reported unsigned.

## What the data cannot say
- No running backs, quarterbacks or linebackers: Combine tracking covers only DB, DL, OL, WR and TE.
- No TE results, and only three DB outcomes pass the stability check.
- Outcomes are tracking-based play measures (for example pressure rate, route separation), not wins, and exist only for 2023-2025 draftees.
- Many sensor drills are single-attempt, so their repeatability is unknown; the 40-yard sensor measures passed the repeatability check for DL, OL, WR and TE (not re-checked for DB).
- Realistic improvement sizes are judgment calls drawn from sports-science ranges; the trained-versus-untrained effect of a real program is not measured here.
