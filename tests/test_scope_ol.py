from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol.scope_ol import (
    _predictor_orientation,
    _reach_partial,
    associate,
    attach_tracking_features,
    benjamini_hochberg,
    extract_season_tracking,
    gate_outcome,
    main,
    orientation_label,
    partial_spearman,
    prepare_play_outcomes,
    record_infeasible_outcomes,
    tracking_features,
)


def _tracking() -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    for second in (0.0, 1.0, 1.5, 2.0):
        rows.extend(
            [
                {
                    "game_id": 1,
                    "play_id": 2,
                    "nfl_id": 10,
                    "seconds_since_snap": second,
                    "x_rel": -second,
                    "y_rel": 2.0 * second,
                    "s": 3.0 + second,
                    "is_ol": True,
                    "is_defender": False,
                },
                {
                    "game_id": 1,
                    "play_id": 2,
                    "nfl_id": 20,
                    "seconds_since_snap": second,
                    "x_rel": -second + 3.0,
                    "y_rel": 2.0 * second + 4.0,
                    "s": 2.0,
                    "is_ol": False,
                    "is_defender": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_tracking_features_measure_depth_speed_lateral_and_separation() -> None:
    feature = tracking_features(_tracking())
    row = feature.iloc[0]
    assert row["pass_set_depth_1_0s"] == 1.0
    assert row["pass_set_depth_1_5s"] == 1.5
    assert row["pass_set_depth_2_0s"] == 2.0
    assert row["lateral_shuffle_0_1_5s"] == 3.0
    assert row["set_peak_speed_0_1_5s"] == 4.5
    assert row["set_time_to_peak_speed_0_1_5s"] == 1.5
    assert row["nearest_defender_separation_1_0s"] == 5.0
    assert row["nearest_defender_separation_2_0s"] == 5.0


def test_attach_tracking_features_preserves_only_matched_ol_rows() -> None:
    outcomes = pd.DataFrame({"game_id": [1, 1], "play_id": [2, 3], "nfl_id": [10, 11]})
    features = pd.DataFrame({"game_id": [1], "play_id": [2], "nfl_id": [10], "value": [2.0]})
    joined = attach_tracking_features(outcomes, features)
    assert joined.shape == (2, 4)
    assert joined.loc[joined["nfl_id"].eq(10), "value"].iloc[0] == 2.0
    assert joined.loc[joined["nfl_id"].eq(11), "value"].isna().all()


def test_partial_spearman_removes_ranked_confounder() -> None:
    rng = np.random.default_rng(12)
    confounder = np.arange(80, dtype=float)
    predictor = confounder + rng.normal(0.0, 5.0, 80)
    outcome = confounder + rng.normal(0.0, 5.0, 80)
    statistic = partial_spearman(predictor, outcome, [confounder])
    assert abs(statistic.r) < 0.15
    assert statistic.p > 0.1


def test_bh_and_orientation_are_deterministic() -> None:
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(q, np.array([0.03, 0.04, 0.04]))
    assert orientation_label(0.2, -1, -1) == "expected"
    assert orientation_label(0.2, 1, -1) == "opposite"
    assert orientation_label(0.2, 0, -1) == "unclassified"


def _write_synthetic_root(root: Path) -> None:
    (root / "data/processed").mkdir(parents=True)
    (root / "reports").mkdir()
    rng = np.random.default_rng(5)
    players = 60
    plays = 60
    ids = np.arange(1000, 1000 + players)
    effect = rng.normal(0.0, 0.5, players)
    frame = pd.DataFrame(
        {
            "game_id": np.repeat(np.arange(players), plays),
            "play_id": np.tile(np.arange(plays), players),
            "nfl_id": np.repeat(ids, plays),
            "season": 2024,
            "lined_up_position": rng.choice(["LT", "LG", "C"], players * plays),
            "down": rng.integers(1, 4, players * plays),
            "yards_to_go": rng.integers(1, 15, players * plays),
            "offense_formation": rng.choice(["SHOTGUN", "SINGLEBACK"], players * plays),
            "dropback_duration": rng.uniform(2.0, 4.0, players * plays),
            "pass_rushers_encountered": rng.integers(0, 3, players * plays),
        }
    )
    frame["pass_set_depth_1_0s"] = np.repeat(effect, plays) + rng.normal(0.0, 1.0, players * plays)
    frame["pressure_allowed"] = rng.random(players * plays) < 0.1
    frame.to_parquet(root / "data/processed/scope_ol_play_outcomes.parquet", index=False)
    predictors = pd.DataFrame(
        {
            "nfl_id": ids.astype(str),
            "group": "OL",
            "nfl_position": np.where(np.arange(players) % 2 == 0, "T", "G"),
            "draft_year": 2023 + np.arange(players) % 3,
            "forty": 5.0 + rng.normal(0.0, 0.2, players) - 0.3 * effect,
            "combine_weight": rng.normal(310.0, 10.0, players),
            "arm_length": rng.normal(33.0, 1.0, players),
            "combine_height": rng.normal(77.0, 1.0, players),
            "forty_yard_dash_top_speed": rng.normal(6.0, 0.3, players),
            "pass_rush_drop_top_speed": rng.normal(5.0, 0.3, players),
            "pass_rush_drop_path_length": rng.normal(10.0, 1.0, players),
        }
    )
    predictors.to_parquet(root / "data/processed/combine_predictors.parquet", index=False)
    names = ["forty_yard_dash_top_speed", "pass_rush_drop_top_speed", "pass_rush_drop_path_length"]
    pd.DataFrame(
        {"predictor_column": names, "family": "sensor", "group": "OL", "eligible_for_M4": True}
    ).to_csv(root / "reports/phase2_combine_predictors.csv", index=False)


def test_gate_then_associate_produces_standard_results_schema(tmp_path: Path) -> None:
    _write_synthetic_root(tmp_path)
    gate = gate_outcome(tmp_path, "pass_set_depth_1_0s")
    assert gate.iloc[0]["verdict"] == "adequate"
    record_infeasible_outcomes(tmp_path)
    stored = pd.read_csv(tmp_path / "reports/scope_ol_gate.csv")
    assert set(stored["verdict"]) == {"adequate", "infeasible"}
    results = associate(tmp_path)
    assert set(results["predictor"]).issuperset({"forty", "combine_weight", "OL_DROP_INDEX", "OL_SPEED"})
    assert set(results["outcome"]) == {"pass_set_depth_1_0s"}
    forty = results.loc[results["predictor"].eq("forty")].iloc[0]
    assert forty["direction"] == "expected"
    assert forty["n"] == 60
    assert results["q_scope"].between(0.0, 1.0).all()
    assert list(results.columns[:19]) == [
        "scope", "group", "outcome", "predictor", "family", "n", "r_raw", "r_adj", "ci_lo", "ci_hi", "p",
        "q_scope", "oriented_r", "direction", "r_disc_2023_24", "r_hold_2025", "r_2023", "r_2024", "r_2025",
    ]


def test_gate_handles_boolean_outcome_and_rejects_unknown(tmp_path: Path) -> None:
    _write_synthetic_root(tmp_path)
    assert gate_outcome(tmp_path, "pressure_allowed").iloc[0]["verdict"] in {"weak", "poor", "adequate"}
    with pytest.raises(ValueError):
        gate_outcome(tmp_path, "nearest_defender_separation_1_0s")


def test_predictor_orientation_follows_time_and_speed_naming() -> None:
    assert _predictor_orientation("forty") == -1
    assert _predictor_orientation("forty_yard_dash_time_in_drill") == -1
    assert _predictor_orientation("forty_yard_dash_burst_distance_0_1s") == 1
    assert _predictor_orientation("vertical") == 1
    assert _predictor_orientation("hand_size") == 0


def test_main_dispatch_validates_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        main(["extract", "--root", str(tmp_path)])
    with pytest.raises(ValueError):
        main(["gate", "--outcome", "bogus", "--root", str(tmp_path)])
    (tmp_path / "reports").mkdir()
    main(["infeasible", "--root", str(tmp_path)])
    assert (tmp_path / "reports/scope_ol_gate.csv").is_file()


def _write_tracking_root(root: Path) -> None:
    (root / "data/processed").mkdir(parents=True)
    times = pd.date_range("2024-01-01 20:00:00", periods=25, freq="100ms")
    events = ["ball_snap" if index == 5 else None for index in range(25)]
    joined_columns = {
        "game_id": 1, "play_id": 1, "season": 2024, "team_abbr": "AAA", "possession_team": "AAA",
        "nfl_position": "T", "lined_up_position": "LT", "down": 1, "yards_to_go": 10,
        "offense_formation": "SHOTGUN", "dropback_duration": 3.0, "pass_rushers_encountered": 1,
        "peak_pressure_probability_allowed": 0.1, "pressure_allowed": False, "sack_allowed": 0,
        "play_nullified_by_penalty": "N", "yardline_side": "AAA", "yardline_number": 30,
        "play_direction": "right",
    }
    joined = pd.DataFrame([{**joined_columns, "nfl_id": 7}, {**joined_columns, "nfl_id": 8, "team_abbr": "BBB",
                                                           "nfl_position": "DE"}])
    joined.to_parquet(root / "data/processed/player_play_joined.parquet", index=False)
    tracking = pd.concat(
        [
            pd.DataFrame(
                {"game_id": 1, "play_id": 1, "nfl_id": nfl_id, "time": times.astype(str), "x": 50.0 - index * 0.1,
                 "y": 20.0, "s": 2.0, "event": events}
            )
            for nfl_id, index in ((7, np.arange(25)), (8, np.zeros(25)))
        ],
        ignore_index=True,
    )
    tracking.to_parquet(root / "data/processed/game_tracking_2024.parquet", index=False)


def test_extract_and_prepare_build_play_outcome_artifact(tmp_path: Path) -> None:
    _write_tracking_root(tmp_path)
    path = extract_season_tracking(tmp_path, 2024)
    features = pd.read_parquet(path)
    assert len(features) == 1
    assert features["pass_set_depth_1_0s"].iloc[0] > 0.0
    with pytest.raises(FileNotFoundError):
        prepare_play_outcomes(tmp_path)
    for season in (2023, 2025):
        features.iloc[0:0].to_parquet(tmp_path / f"data/processed/scope_ol_tracking_features_{season}.parquet")
    for season in (2023, 2024, 2025):
        pd.DataFrame(
            {"game_id": [1], "play_id": [1], "nfl_id": [7], "snap_reaction_time": [0.3],
             "first_0_5s_displacement": [-0.2]}
        ).iloc[: 1 if season == 2024 else 0].to_parquet(
            tmp_path / f"data/processed/ol_outcomes_snap_features_{season}.parquet"
        )
    output = pd.read_parquet(prepare_play_outcomes(tmp_path))
    assert len(output) == 1
    assert output["snap_reaction_time"].iloc[0] == 0.3


def test_reach_partial_needs_forty_rows() -> None:
    small = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 3.0], "z": [1.0, 3.0, 2.0]})
    assert np.isnan(_reach_partial(small, "x", "y", ["z"]))
