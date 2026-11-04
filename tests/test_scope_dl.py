from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from combine_dlol.scope_dl import (
    OFFICIAL,
    RESULT_COLUMNS,
    STATIC_OUTCOMES,
    _bootstrap_difference,
    _interaction_p,
    association_frame,
    bh_qvalues,
    build_composite,
    join_predictors_outcomes,
    orientation_label,
    partial_spearman,
    role_labels,
    run_analysis,
    run_rates,
)

COMPOSITE_COLUMNS = (
    "forty_yard_dash_burst_speed_1_0s",
    "forty_yard_dash_sensor_ten_yd_burst_speed",
    "forty_yard_dash_sensor_ten_yd_time",
    "forty_yard_dash_top_speed",
    "pass_rush_drill_top_speed",
    "pass_rush_drill_peak_acceleration",
    "pass_rush_drill_burst_speed_1_0s",
)


def test_partial_spearman_removes_a_shared_rank_covariate() -> None:
    values = np.arange(60, dtype=float)
    control = values.copy()
    rng = np.random.default_rng(21)
    predictor = control + rng.normal(0.0, 10.0, len(control))
    outcome = control + rng.normal(0.0, 10.0, len(control))
    result = partial_spearman(predictor, outcome, [control])
    assert abs(result.r) < 0.1
    assert 0.0 <= result.p <= 1.0
    assert result.ci_lo < result.r < result.ci_hi


def test_join_predictors_outcomes_coerces_identifier_and_retains_one_row_per_player() -> None:
    predictors = pd.DataFrame({"nfl_id": ["10", "20"], "forty": [4.6, 4.8]})
    outcomes = pd.DataFrame({"nfl_id": [10, 20], "rate": [0.1, 0.2]})
    joined = join_predictors_outcomes(predictors, outcomes)
    assert joined["nfl_id"].tolist() == [10, 20]
    assert joined["rate"].tolist() == [0.1, 0.2]


def test_role_labels_and_orientation_are_football_consistent() -> None:
    roles = role_labels(pd.Series(["DE", "OLB", "DT", "NT"]))
    assert roles.tolist() == ["EDGE", "EDGE", "IDL", "IDL"]
    assert orientation_label("forty", "shrunk_d_a", 0.3) == "expected"
    assert orientation_label("forty_yard_dash_top_speed", "pressure_rate", 0.3) == "expected"
    assert orientation_label("combine_height", "pressure_rate", 0.3) == "unclassified"


def test_composite_averages_available_oriented_zscores_and_is_missing_only_without_components() -> None:
    frame = pd.DataFrame({"speed": [1.0, 2.0, 3.0], "time": [3.0, 2.0, np.nan]})
    composite = build_composite(frame, {"speed": 1, "time": -1})
    assert np.isclose(float(composite.iloc[0]), -(np.sqrt(1.5) + 1.0) / 2.0)
    assert np.isclose(float(composite.iloc[1]), 0.5)
    assert np.isclose(float(composite.iloc[2]), np.sqrt(1.5))
    empty_row = pd.DataFrame({"speed": [1.0, np.nan], "time": [1.0, np.nan]})
    assert np.isnan(float(build_composite(empty_row, {"speed": 1, "time": -1}).iloc[1]))


def test_bh_qvalues_are_monotone_in_sorted_pvalues() -> None:
    qvalues = bh_qvalues(np.array([0.04, 0.001, 0.03, 0.5]))
    ordered = qvalues[np.argsort(np.array([0.04, 0.001, 0.03, 0.5]))]
    assert np.all(np.diff(ordered) >= 0.0)
    assert np.all((qvalues >= 0.0) & (qvalues <= 1.0))


def test_association_frame_adds_rank_partial_and_raw_correlations() -> None:
    rng = np.random.default_rng(45)
    frame = pd.DataFrame(
        {
            "predictor": np.arange(50, dtype=float),
            "outcome": np.arange(50, dtype=float),
            "combine_weight": rng.normal(280.0, 15.0, 50),
            "edge": np.tile([0, 1], 25),
            "draft_year": np.repeat([2023, 2024, 2025], [17, 17, 16]),
            "draft_overall_pick": rng.permutation(np.arange(1, 51, dtype=float)),
        }
    )
    row = association_frame(frame, "predictor", "outcome", ["combine_weight", "edge"], "official")
    assert row["n"] == 50
    assert row["r_raw"] > 0.99
    assert row["r_adj"] > 0.99
    assert row["partial_draft_r"] > 0.99


def test_time_in_drill_is_flipped_and_replication_needs_fifteen_players() -> None:
    assert orientation_label("pass_rush_drill_time_in_drill", "hurry_rate", 0.3) == "opposite"
    frame = pd.DataFrame(
        {
            "predictor": np.arange(40, dtype=float),
            "outcome": np.arange(40, dtype=float),
            "combine_weight": np.random.default_rng(3).normal(280.0, 10.0, 40),
            "draft_year": np.repeat([2023, 2024, 2025], [20, 14, 6]),
            "draft_overall_pick": np.arange(40, dtype=float),
        }
    )
    row = association_frame(frame, "predictor", "outcome", ["combine_weight"], "official")
    assert row["r_2023"] > 0.99
    assert np.isnan(row["r_2024"])
    assert np.isnan(row["r_hold_2025"])


def test_interaction_detects_role_specific_slope_and_bootstrap_brackets_difference() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=120)
    edge = np.tile([0.0, 1.0], 60)
    frame = pd.DataFrame(
        {"x": x, "y": x * edge + rng.normal(0, 0.3, 120), "combine_weight": rng.normal(size=120), "edge": edge}
    )
    assert _interaction_p(frame, "x", "y") < 0.001
    top, bottom = rng.normal(1.0, 1.0, 50), rng.normal(0.0, 1.0, 50)
    difference, lower, upper = _bootstrap_difference(top, bottom)
    assert lower < difference < upper


def test_partial_spearman_without_controls_equals_spearman() -> None:
    x = np.arange(30, dtype=float)
    result = partial_spearman(x, x[::-1].copy(), [])
    assert np.isclose(result.r, -1.0)


def _synthetic_root(root: Path) -> None:
    rng = np.random.default_rng(5)
    players = np.arange(1, 101)
    positions = np.where(players % 2 == 0, "DE", "DT")
    skill = rng.normal(0.0, 1.0, len(players))
    plays = []
    for index, player in enumerate(players):
        count = 70
        win_probability = 1.0 / (1.0 + np.exp(-(skill[index] - 1.0)))
        won = rng.random(count) < win_probability
        plays.append(
            pd.DataFrame(
                {
                    "nfl_id": player,
                    "nfl_position": positions[index],
                    "lined_up_position": positions[index],
                    "down": rng.integers(1, 4, count),
                    "yards_to_go": rng.integers(1, 10, count),
                    "blitzing": rng.integers(0, 2, count),
                    "offense_formation": rng.choice(["SHOTGUN", "SINGLEBACK"], count),
                    "player_get_off": rng.normal(size=count),
                    "pass_result": "C",
                    "play_nullified_by_penalty": "N",
                    "sack": won * (rng.random(count) < 0.3),
                    "tackle_for_loss": (rng.random(count) < 0.05).astype(float),
                    "time_to_pressure": np.where(won, 2.0, np.nan),
                    "time_to_qb_hurry": np.where(won, 2.1, np.nan),
                    "draft_overall_pick": float(index + 1),
                }
            )
        )
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    (root / "reports").mkdir()
    pd.concat(plays, ignore_index=True).to_parquet(processed / "player_play_joined.parquet", index=False)
    predictors = pd.DataFrame({"nfl_id": players.astype(str), "group": "DL", "draft_year": 2023 + players % 3})
    for column in OFFICIAL:
        predictors[column] = rng.normal(size=len(players))
    predictors["forty_yard_dash_peak_acceleration"] = skill + rng.normal(0.0, 0.5, len(players))
    for column in COMPOSITE_COLUMNS:
        predictors[column] = rng.normal(size=len(players))
    predictors.to_parquet(processed / "combine_predictors.parquet", index=False)
    outcomes = pd.DataFrame({"nfl_id": players})
    for column in STATIC_OUTCOMES:
        outcomes[column] = rng.normal(size=len(players))
    outcomes.to_parquet(processed / "dl_outcomes_scores.parquet", index=False)
    pd.DataFrame(
        {
            "predictor_column": ["forty_yard_dash_peak_acceleration", *COMPOSITE_COLUMNS],
            "group": "DL",
            "eligible_for_M4": True,
        }
    ).to_csv(root / "reports/phase2_combine_predictors.csv", index=False)
    reliability = pd.DataFrame(
        {
            "outcome": ["d_a", "d_b_displacement_0_5", "d_b_displacement_1_0", "d_b_speed_0_5"],
            "group": "all",
            "n_players": 100,
            "split_half_median": 0.8,
            "split_half_ci_lower": 0.6,
            "split_half_ci_upper": 0.9,
            "icc": 0.05,
            "verdict": "adequate",
        }
    )
    reliability.to_csv(root / "reports/phase1_dl_reliability.csv", index=False)


def test_full_pipeline_on_synthetic_root_writes_gate_results_and_quartiles(tmp_path: Path) -> None:
    _synthetic_root(tmp_path)
    run_rates(tmp_path)
    results, quartiles = run_analysis(tmp_path)
    gate = pd.read_csv(tmp_path / "reports/scope_dl_gate.csv")
    assert {"hurry_rate", "shrunk_d_a"} <= set(gate["outcome"])
    assert tuple(results.columns) == RESULT_COLUMNS
    assert {"pooled", "EDGE", "IDL", "interaction"} <= set(results["group"])
    assert results.loc[results["group"].ne("interaction"), "n"].min() >= 40
    assert (tmp_path / "reports/scope_dl_results.csv").exists()
    assert set(quartiles.columns) >= {"predictor", "outcome", "difference"}
