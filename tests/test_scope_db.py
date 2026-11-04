from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol.scope_db import (
    OUTCOMES,
    RESULT_COLUMNS,
    _outcome_frame,
    _predictor_orientation,
    _zscore,
    add_composites,
    association_rows,
    benjamini_hochberg,
    build_predictor_frame,
    build_sensor_attempts,
    eligible_players,
    gate_outcomes,
    join_predictors_scores,
    official_predictors,
    orientation_label,
    partial_spearman,
    role_from_position,
    sensor_player_features,
    sensor_repeatability,
)


def test_partial_spearman_removes_rank_linear_control() -> None:
    control = np.arange(12, dtype=float)
    pattern = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6], dtype=float)
    statistic = partial_spearman(control + pattern, control - pattern, [control])

    assert statistic.r_adj < -0.9
    assert statistic.r_raw > 0.0
    assert statistic.p < 0.001


def test_benjamini_hochberg_is_monotone_in_sorted_p_values() -> None:
    actual = benjamini_hochberg(np.array([0.04, 0.001, 0.02, 0.9]))

    assert actual.tolist() == pytest.approx([0.0533333333, 0.004, 0.04, 0.9])


def test_join_predictors_scores_casts_identifier_and_preserves_only_complete_scores() -> None:
    predictors = pd.DataFrame(
        {
            "nfl_id": ["1", "2", "3"],
            "combine_weight": [190.0, 200.0, 210.0],
            "nfl_position": ["CB", "FS", "SS"],
        }
    )
    scores = pd.DataFrame({"nfl_id": [1, 2], "tackle_rate": [0.2, 0.3]})

    actual = join_predictors_scores(predictors, scores)

    assert actual.nfl_id.tolist() == [1, 2]
    assert actual.tackle_rate.tolist() == [0.2, 0.3]
    assert actual.role.tolist() == ["CB", "S"]


def test_composites_use_oriented_z_scores_only_when_all_components_are_present() -> None:
    frame = pd.DataFrame(
        {
            "forty_yard_dash_top_speed": [8.0, 9.0, 10.0],
            "forty": [4.8, 4.6, 4.4],
            "vertical": [30.0, 34.0, 38.0],
            "broad_jump": [100.0, 110.0, 120.0],
            "arm_length": [30.0, 31.0, 32.0],
            "combine_height": [70.0, 71.0, 72.0],
        }
    )

    actual = add_composites(frame, set(frame.columns))

    assert actual["DB_SPEED"].tolist() == pytest.approx([-1.22474487, 0.0, 1.22474487])
    assert actual["DB_EXPLOSIVE"].tolist() == pytest.approx([-1.22474487, 0.0, 1.22474487])
    assert actual["DB_REACH"].tolist() == pytest.approx([-1.22474487, 0.0, 1.22474487])
    assert actual["DB_BURST"].isna().all()


def test_orientation_labels_follow_pre_specification() -> None:
    assert role_from_position("CB") == "CB"
    assert role_from_position("FS") == "S"
    assert orientation_label(0.2, "tackle_rate", "forty") == "expected"
    assert orientation_label(-0.2, "tackle_rate", "forty") == "opposite"
    assert orientation_label(0.2, "coverage_assignment_rate", "forty") == "unclassified"


def _player_play(players: int = 6, plays: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    count = players * plays
    ids = np.repeat(np.arange(1, players + 1), plays)
    assigned = rng.random(count) < 0.6
    tackle_probability = np.linspace(0.03, 0.5, players)[ids - 1]
    return pd.DataFrame(
        {
            "nfl_id": ids,
            "nfl_position": np.where(ids % 2 == 0, "CB", "FS"),
            "draft_year": 2024,
            "season": 2024,
            "play_nullified_by_penalty": np.where(rng.random(count) < 0.02, "Y", "N"),
            "pass_length": np.where(assigned, rng.integers(0, 30, count), np.nan),
            "down": rng.integers(1, 5, count),
            "yards_to_go": rng.integers(1, 15, count),
            "team_coverage_man_zone": np.where(assigned, "MAN_COVERAGE", None),
            "team_coverage_type": np.where(assigned, "COVER_1", None),
            "offense_formation": "SHOTGUN",
            "coverage_assignment_at_snap": np.where(assigned, np.where(rng.random(count) < 0.3, "MAN", "HOOK"), None),
            "tackle": np.where(rng.random(count) < tackle_probability, 1.0, np.nan),
            "assist": np.where(rng.random(count) < 0.05, 1.0, np.nan),
            "caused_forced_fumble": np.where(rng.random(count) < 0.01, 1.0, np.nan),
            "tackle_for_loss": np.where(rng.random(count) < 0.02, True, None),
        }
    )


def test_outcome_frame_defines_rates_and_conditional_man_rate() -> None:
    frame = _outcome_frame(_player_play())

    assert set(frame["nfl_position"]) <= {"CB", "FS"}
    assert frame["play_nullified_by_penalty"].eq("Y").sum() == 0
    assert frame.loc[~frame["assigned_coverage"], "man_coverage_assignment_rate"].isna().all()
    assert frame["pass_length"].notna().all()
    assert frame["team_coverage_type"].isin(["COVER_1", "NONE"]).all()
    assert frame["tackle_rate"].isin([0.0, 1.0]).all()


def test_eligible_players_use_assigned_plays_for_coverage_outcomes() -> None:
    frame = _outcome_frame(_player_play(players=3, plays=100))
    thin = frame.assign(assigned_coverage=frame["nfl_id"].eq(1) & frame["assigned_coverage"])

    assert eligible_players(thin, "man_coverage_assignment_rate").tolist() in ([1], [])
    assert len(eligible_players(frame, "tackle_rate")) == 3


def test_gate_outcomes_returns_every_outcome_with_scores_for_adequate_ones() -> None:
    gate, scores = gate_outcomes(_player_play())

    assert gate["outcome"].tolist() == list(OUTCOMES)
    assert set(gate["verdict"]) <= {"adequate", "weak", "poor"}
    assert "nfl_id" in scores.columns
    adequate = gate.loc[gate["verdict"].eq("adequate"), "outcome"]
    assert "tackle_rate" in set(adequate)
    assert set(adequate) <= set(scores.columns)
    assert scores["tackle_rate"].notna().sum() == 6


def test_gate_flags_too_few_players() -> None:
    gate, _ = gate_outcomes(_player_play(players=2))

    assert gate["verdict"].eq("poor").all()
    assert gate["n_players"].eq(0).all()


def test_official_predictors_keep_db_rows_one_per_player_and_attach_draft_pick() -> None:
    results = pd.DataFrame(
        {
            "nfl_id": [1, 2, 3],
            "combine_position": ["DB", "DB", "WR"],
            "forty": [4.5, 4.6, 4.4],
        }
    )
    players = pd.DataFrame(
        {"nfl_id": [1, 2, 3], "nfl_position": ["CB", "SS", "WR"], "draft_overall_pick": [10, 50, 20]}
    )

    actual = official_predictors(results, players)

    assert actual.nfl_id.tolist() == [1, 2]
    assert actual.draft_overall_pick.tolist() == [10, 50]
    assert actual.nfl_id.is_unique


def _attempts() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for player in range(1, 31):
        level = rng.normal()
        for attempt in ("1", "2", "3"):
            rows.append(
                {
                    "nfl_id": player,
                    "drill": "FORTY_YARD_DASH",
                    "attempt": attempt,
                    "top_speed": level + rng.normal(scale=0.1),
                    "noise_metric": rng.normal(),
                }
            )
    rows.append({"nfl_id": 99, "drill": "BOX_DRILL", "attempt": "1", "top_speed": 1.0, "noise_metric": 1.0})
    return pd.DataFrame(rows)


def test_sensor_repeatability_screens_by_attempt_level_reliability() -> None:
    table = sensor_repeatability(_attempts())
    forty = table.loc[table["drill"].eq("FORTY_YARD_DASH")].set_index("metric")

    assert forty.loc["top_speed", "status"] == "pass"
    assert forty.loc["top_speed", "predictor"] == "forty_yard_dash_top_speed"
    assert forty.loc["noise_metric", "status"] == "unreliable"
    box = table.loc[table["drill"].eq("BOX_DRILL")]
    assert box["status"].eq("insufficient_attempts").all()


def test_sensor_player_features_use_only_passing_metrics() -> None:
    attempts = _attempts()
    table = sensor_repeatability(attempts)

    features = sensor_player_features(attempts, table)

    assert "forty_yard_dash_top_speed" in features.columns
    assert "forty_yard_dash_noise_metric" not in features.columns
    assert features["nfl_id"].is_unique
    assert sensor_player_features(attempts, table.assign(status="unreliable")).columns.tolist() == ["nfl_id"]


def test_build_predictor_frame_lists_families_and_composites() -> None:
    attempts = _attempts()
    repeatability = sensor_repeatability(attempts)
    rng = np.random.default_rng(1)
    results = pd.DataFrame(
        {
            "nfl_id": np.arange(1, 31),
            "combine_position": "DB",
            "forty": rng.normal(4.5, 0.1, 30),
            "vertical": rng.normal(35, 2, 30),
            "broad_jump": rng.normal(120, 5, 30),
            "arm_length": rng.normal(31, 1, 30),
            "combine_height": rng.normal(71, 1, 30),
            "combine_weight": rng.normal(195, 8, 30),
        }
    )
    players = pd.DataFrame({"nfl_id": np.arange(1, 31), "nfl_position": "CB", "draft_overall_pick": np.arange(1, 31)})

    frame, predictors = build_predictor_frame(results, players, attempts, repeatability)

    assert frame["nfl_id"].is_unique
    assert ("forty", "official") not in predictors
    assert ("forty_yard_dash_top_speed", "sensor") not in predictors
    assert len(frame) == 30


def test_build_predictor_frame_keeps_predictors_with_forty_players() -> None:
    rng = np.random.default_rng(2)
    ids = np.arange(1, 61)
    attempts = pd.DataFrame(
        [
            {"nfl_id": int(i), "drill": "FORTY_YARD_DASH", "attempt": str(a), "top_speed": i + rng.normal(scale=0.5)}
            for i in ids
            for a in (1, 2)
        ]
    )
    repeatability = sensor_repeatability(attempts)
    results = pd.DataFrame(
        {
            "nfl_id": ids,
            "combine_position": "DB",
            "forty": rng.normal(4.5, 0.1, 60),
            "vertical": rng.normal(35, 2, 60),
            "broad_jump": rng.normal(120, 5, 60),
            "arm_length": rng.normal(31, 1, 60),
            "combine_height": rng.normal(71, 1, 60),
        }
    )
    players = pd.DataFrame({"nfl_id": ids, "nfl_position": "S", "draft_overall_pick": ids})

    frame, predictors = build_predictor_frame(results, players, attempts, repeatability)

    assert ("forty", "official") in predictors
    assert ("forty_yard_dash_top_speed", "sensor") in predictors
    assert {"DB_SPEED", "DB_EXPLOSIVE", "DB_REACH"} <= {name for name, family in predictors if family == "composite"}
    assert "DB_BURST" not in {name for name, _ in predictors}


def _analysis(players: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    pick = rng.permutation(players)
    speed = rng.normal(size=players)
    return pd.DataFrame(
        {
            "nfl_id": np.arange(players),
            "nfl_position": np.where(np.arange(players) % 3 == 0, "CB", "SS"),
            "role": np.where(np.arange(players) % 3 == 0, "CB", "S"),
            "draft_year": np.tile([2023, 2024, 2025], players // 3),
            "draft_overall_pick": pick,
            "combine_weight": rng.normal(200, 10, players),
            "forty": -speed + rng.normal(scale=0.2, size=players),
            "combine_height": rng.normal(71, 2, players),
            "tackle_rate": speed + rng.normal(scale=0.2, size=players),
            "coverage_assignment_rate": rng.normal(size=players),
        }
    )


def test_association_rows_orient_and_replicate() -> None:
    rows = association_rows(
        _analysis(),
        ["tackle_rate", "coverage_assignment_rate"],
        [("forty", "official"), ("combine_height", "official")],
    )

    assert list(rows.columns) == [*RESULT_COLUMNS, "partial_draft_r"]
    assert len(rows) == 4
    forty = rows.loc[rows["outcome"].eq("tackle_rate") & rows["predictor"].eq("forty")].iloc[0]
    assert forty["direction"] == "expected"
    assert forty["oriented_r"] > 0.5
    assert forty["r_hold_2025"] < -0.5
    assert forty["n"] == 90
    assert forty["q_scope"] <= 1.0
    height = rows.loc[rows["predictor"].eq("combine_height")]
    assert height["direction"].eq("unclassified").all()
    assert height["oriented_r"].isna().all()
    coverage = rows.loc[rows["outcome"].eq("coverage_assignment_rate")]
    assert coverage["direction"].eq("unclassified").all()


def test_association_rows_skip_small_samples_and_flag_opposite() -> None:
    small = association_rows(_analysis(30), ["tackle_rate"], [("forty", "official")])
    assert small.empty
    assert list(small.columns) == [*RESULT_COLUMNS, "partial_draft_r"]

    flipped = _analysis().assign(forty=lambda frame: -frame["forty"])
    rows = association_rows(flipped, ["tackle_rate"], [("forty", "official")])
    assert rows.iloc[0]["direction"] == "opposite"


def test_predictor_orientation_rules() -> None:
    assert _predictor_orientation("forty") == -1.0
    assert _predictor_orientation("forty_yard_dash_sensor_ten_yd_time") == -1.0
    assert _predictor_orientation("box_drill_time_to_top_speed") == -1.0
    assert _predictor_orientation("forty_yard_dash_peak_acceleration") == 1.0
    assert _predictor_orientation("vertical") == 1.0
    assert _predictor_orientation("DB_BURST") == 1.0
    assert _predictor_orientation("combine_weight") == 0.0
    assert _predictor_orientation("box_drill_path_length") == 0.0
    assert orientation_label(0.3, "tackle_rate", "arm_length") == "unclassified"


def test_benjamini_hochberg_ignores_missing_values() -> None:
    actual = benjamini_hochberg(np.array([np.nan, 0.01, 0.03]))

    assert np.isnan(actual[0])
    assert actual[1:].tolist() == pytest.approx([0.02, 0.03])
    assert np.isnan(benjamini_hochberg(np.array([np.nan]))).all()


def test_zscore_of_constant_column_is_missing() -> None:
    assert _zscore(pd.Series([1.0, 1.0, 1.0]), 1.0).isna().all()


def test_build_sensor_attempts_streams_raw_tracking(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    times = pd.date_range("2025-03-01T10:00:00", periods=40, freq="100ms").strftime("%Y-%m-%dT%H:%M:%S.%f").tolist()
    rows = [
        {
            "nfl_id": nfl_id,
            "entity_type": "PLAYER",
            "time": moment,
            "drill_name": "FORTY_YARD_DASH",
            "attempt": 1,
            "x": 0.15 * index**1.5,
            "y": 10.0,
        }
        for nfl_id in (7, 8)
        for index, moment in enumerate(times)
    ]
    pd.DataFrame(rows).to_csv(raw / "combine_tracking.csv", index=False)

    attempts = build_sensor_attempts(tmp_path, {"7"})

    assert attempts["nfl_id"].unique().tolist() == [7]
    assert attempts["drill"].unique().tolist() == ["FORTY_YARD_DASH"]
    assert attempts["top_speed"].iloc[0] > 0.0
    assert "sensor_ten_yd_time" in attempts.columns


def test_missing_draft_pick_does_not_shrink_main_sample() -> None:
    frame = _analysis().assign(draft_overall_pick=lambda data: data["draft_overall_pick"].where(data["nfl_id"] > 20))

    rows = association_rows(frame, ["tackle_rate"], [("forty", "official")])

    assert rows.iloc[0]["n"] == 90
    assert np.isfinite(rows.iloc[0]["partial_draft_r"])

    sparse = _analysis().assign(draft_overall_pick=lambda data: data["draft_overall_pick"].where(data["nfl_id"] > 70))
    assert np.isnan(association_rows(sparse, ["tackle_rate"], [("forty", "official")]).iloc[0]["partial_draft_r"])
