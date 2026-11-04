from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol import scope_db_drills as m
from combine_dlol.db_play_tracking import (
    SEASONS,
    PlayContext,
    Track,
    backpedal_share,
    build_tracking_features,
    cod_speed_retention,
    db_play_metrics,
    line_of_scrimmage,
    peak_decel_in_play,
    peak_speed_in_play,
    receiver_metrics,
    run_pursuit_peak_speed,
    snap_reaction_time,
    speed_1s,
    speed_gain_after_throw,
    tackle_depth,
    tag_roles,
    wrap_degrees,
)


def _track(
    seconds: np.ndarray,
    x: np.ndarray | float = 0.0,
    y: np.ndarray | float = 0.0,
    s: np.ndarray | float = 0.0,
    o: np.ndarray | float = 0.0,
    d: np.ndarray | float = 0.0,
) -> Track:
    size = len(seconds)

    def full(value: np.ndarray | float) -> np.ndarray:
        return np.broadcast_to(np.asarray(value, dtype=float), (size,)).copy()

    return Track(seconds, full(x), full(y), full(s), full(o), full(d))


T = np.arange(0.0, 6.01, 0.1)


def test_orientations_follow_pre_specification() -> None:
    assert m.predictor_orientation("forty") == -1.0
    assert m.predictor_orientation("w_drill_time_in_drill") == -1.0
    assert m.predictor_orientation("box_drill_cod_speed_loss_fraction") == -1.0
    assert m.predictor_orientation("vertical") == 1.0
    assert m.predictor_orientation("forty_yard_dash_peak_acceleration") == 1.0
    assert m.predictor_orientation("w_drill_cod_speed_recovery_0_5s") == 1.0
    assert m.predictor_orientation("box_drill_path_length") == 0.0
    assert m.outcome_orientation("snap_reaction_time") == -1.0
    assert m.outcome_orientation("backpedal_share") == 0.0
    assert m.orientation_label(0.2) == "expected"
    assert m.orientation_label(-0.2) == "opposite"
    assert m.orientation_label(float("nan")) == "unclassified"
    assert m.predictor_column("W_DRILL", "top_speed") == "w_drill_top_speed"


def test_wrap_degrees_is_shortest_angle() -> None:
    assert wrap_degrees(np.array([350.0 - 10.0, -190.0, 90.0])).tolist() == [20.0, 170.0, 90.0]


def test_line_of_scrimmage_handles_side_and_direction() -> None:
    assert line_of_scrimmage("KC", 30, "KC", "right") == (40.0, 1.0)
    assert line_of_scrimmage("DEN", 30, "KC", "right") == (80.0, 1.0)
    assert line_of_scrimmage("KC", 30, "KC", "left") == (80.0, -1.0)
    assert line_of_scrimmage("DEN", 30, "KC", "left") == (40.0, -1.0)


def _sprint_then_turn() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(0.0, 6.0, 0.1)
    x = np.where(time < 3.0, 6.0 * time, 18.0 + 3.0 * (time - 3.0) * 0.0)
    y = np.where(time < 3.0, 0.0, 6.0 * (time - 3.0))
    return time, x, y


def test_cod_metrics_find_the_sharpest_turn() -> None:
    time, x, y = _sprint_then_turn()
    metrics = m.cod_metrics(time, x, y)

    assert metrics["cod_speed_loss_fraction"] >= 0.0
    assert np.isfinite(metrics["cod_speed_recovery_0_5s"])


def test_cod_metrics_nan_without_a_turn_or_enough_frames() -> None:
    time = np.arange(0.0, 4.0, 0.1)
    straight = m.cod_metrics(time, 6.0 * time, np.zeros_like(time))
    short = m.cod_metrics(time[:4], time[:4], time[:4])

    assert np.isnan(straight["cod_speed_loss_fraction"])
    assert np.isnan(short["cod_speed_recovery_0_5s"])


def test_snap_reaction_time_requires_stillness_and_caps_at_three_seconds() -> None:
    moving_at_half = np.where(T < 0.5, 0.0, 3.0)
    reacts = _track(T, x=np.cumsum(moving_at_half) * 0.1, s=moving_at_half)
    assert snap_reaction_time(reacts, 0.0) == pytest.approx(0.5, abs=0.11)
    assert np.isnan(snap_reaction_time(_track(T, s=3.0), 0.0))
    assert snap_reaction_time(_track(T), 0.0) == 3.0
    assert np.isnan(snap_reaction_time(_track(T), 99.0))
    assert np.isnan(snap_reaction_time(_track(T[:3]), 0.0))


def test_speed_and_backpedal_features_use_pass_window() -> None:
    track = _track(T, s=3.0, o=0.0, d=180.0)
    assert speed_1s(track, 0.0, 2.0) == 3.0
    assert np.isnan(speed_1s(track, 0.0, 1.0))
    assert np.isnan(speed_1s(track, 0.0, float("nan")))
    assert backpedal_share(track, 0.0, 2.5) == 1.0
    assert backpedal_share(_track(T, s=3.0, o=0.0, d=0.0), 0.0, 2.5) == 0.0
    assert np.isnan(backpedal_share(track, 0.0, 0.5))
    assert np.isnan(backpedal_share(track, 0.0, float("nan")))


def test_speed_gain_after_throw_needs_half_second_window() -> None:
    speed = np.where(T < 2.0, 2.0, 2.0 + (T - 2.0) * 4.0)
    track = _track(T, s=speed)
    assert speed_gain_after_throw(track, 2.0, float("nan")) == pytest.approx(4.0)
    assert np.isnan(speed_gain_after_throw(track, 2.0, 2.3))
    assert np.isnan(speed_gain_after_throw(track, float("nan"), 3.0))


def test_play_window_outcomes() -> None:
    speed = np.where(T < 2.0, 4.0, 4.0 - (T - 2.0) * 2.0).clip(min=0.0)
    heading = np.where(T < 1.0, 0.0, 90.0)
    track = _track(T, s=np.maximum(speed, 0.0), d=heading)
    assert peak_speed_in_play(track, 0.0, 4.0) == pytest.approx(4.0)
    assert peak_decel_in_play(track, 0.0, 4.0) == pytest.approx(2.0, abs=0.01)
    assert cod_speed_retention(_track(T, s=3.0, d=heading), 0.0, 4.0) == pytest.approx(1.0)
    assert np.isnan(cod_speed_retention(_track(T, s=3.0, d=0.0), 0.0, 4.0))
    assert np.isnan(cod_speed_retention(track, 0.0, float("nan")))
    assert np.isnan(peak_decel_in_play(track, 0.0, 0.3))
    assert np.isnan(peak_speed_in_play(track, 0.0, float("nan")))
    assert np.isnan(cod_speed_retention(_track(T, s=0.5, d=heading), 0.0, 4.0))


def test_run_pursuit_and_tackle_depth() -> None:
    track = _track(T, x=np.linspace(50.0, 56.0, len(T)), s=np.linspace(1.0, 7.0, len(T)))
    assert run_pursuit_peak_speed(track, 0.0, 4.0) == pytest.approx(5.0, abs=0.2)
    assert np.isnan(run_pursuit_peak_speed(track, 0.0, 1.0))
    assert np.isnan(run_pursuit_peak_speed(track, 0.0, float("nan")))
    assert tackle_depth(track, 6.0, 52.0, 1.0) == pytest.approx(4.0)
    assert tackle_depth(track, 6.0, 52.0, -1.0) == pytest.approx(-4.0)
    assert np.isnan(tackle_depth(track, 99.0, 52.0, 1.0))


def test_receiver_metrics_measure_nearest_and_target_closing() -> None:
    db = _track(T, x=0.0, y=0.0)
    near = _track(T, x=np.linspace(8.0, 2.0, len(T)), y=0.0)
    far = _track(T, x=30.0, y=0.0)
    result = receiver_metrics(db, {1: near, 2: far}, 2, 1.0, 3.0)

    assert result["nearest_receiver_dist_pf"] == pytest.approx(7.0)
    assert result["nearest_receiver_dist_arrival"] == pytest.approx(5.0)
    assert result["closing_speed_nearest"] == pytest.approx(1.0)
    assert np.isnan(result["closing_speed_target"])
    closing = receiver_metrics(db, {1: near}, 1, 1.0, 3.0)
    assert closing["dist_to_target_arrival"] == pytest.approx(5.0)
    assert closing["closing_speed_target"] == pytest.approx(1.0)
    assert np.isnan(receiver_metrics(db, {}, None, 1.0, 3.0)["nearest_receiver_dist_pf"])
    no_arrival = receiver_metrics(db, {1: near}, None, 1.0, float("nan"))
    assert np.isfinite(no_arrival["nearest_receiver_dist_pf"])
    assert np.isnan(no_arrival["closing_speed_nearest"])


def test_db_play_metrics_separates_pass_and_run_outcomes() -> None:
    db = _track(T, x=np.linspace(50.0, 56.0, len(T)), s=3.0, o=0.0, d=180.0)
    events = {"ball_snap": 0.0, "pass_forward": 2.5, "pass_arrived": 4.0, "tackle": 6.0}
    passing = db_play_metrics(db, {}, PlayContext(events, 52.0, 1.0, True, None, False))
    running = db_play_metrics(db, {}, PlayContext(events, 52.0, 1.0, False, None, True))

    assert passing["backpedal_share"] == 1.0
    assert np.isnan(passing["run_pursuit_peak_speed"])
    assert np.isnan(passing["tackle_depth"])
    assert np.isnan(running["backpedal_share"])
    assert running["run_pursuit_peak_speed"] == pytest.approx(3.0)
    assert running["tackle_depth"] == pytest.approx(4.0)


def _tracking_fixture(root: Path) -> pd.DataFrame:
    base = pd.Timestamp("2024-09-08T18:00:00")
    rows: list[dict[str, object]] = []
    for i, second in enumerate(np.arange(0.0, 6.0, 0.1)):
        stamp = (base + pd.Timedelta(seconds=float(second))).strftime("%Y-%m-%dT%H:%M:%S.%f")
        event = {0: "ball_snap", 25: "pass_forward", 40: "pass_arrived"}.get(i)
        rows.append(
            dict(game_id=1, play_id=1, nfl_id=10, time=stamp, x=50.0, y=20.0, s=2.0, o=0.0, dir=180.0, event=event)
        )
        rows.append(
            dict(
                game_id=1,
                play_id=1,
                nfl_id=20,
                time=stamp,
                x=55.0 + i * 0.05,
                y=20.0,
                s=2.0,
                o=0.0,
                dir=0.0,
                event=event,
            )
        )
        rows.append(
            dict(game_id=1, play_id=1, nfl_id=30, time=stamp, x=40.0, y=20.0, s=0.0, o=0.0, dir=0.0, event=event)
        )
    tracking = pd.DataFrame(rows)
    path = root / "data" / "processed"
    path.mkdir(parents=True)
    for season in SEASONS:
        tracking.iloc[: 0 if season != 2024 else len(tracking)].to_parquet(path / f"game_tracking_{season}.parquet")
    return tracking


def _player_play_fixture() -> pd.DataFrame:
    common = dict(
        game_id=1,
        play_id=1,
        possession_team="KC",
        yardline_side="KC",
        yardline_number=40,
        play_direction="right",
        pass_result="C",
        tackle=np.nan,
        season=2024,
        play_nullified_by_penalty="N",
    )
    return pd.DataFrame(
        [
            dict(common, nfl_id=10, nfl_position="CB", team_abbr="DEN", target=False),
            dict(common, nfl_id=20, nfl_position="WR", team_abbr="KC", target=True),
            dict(common, nfl_id=30, nfl_position="T", team_abbr="KC", target=False),
        ]
    )


def test_tag_roles_and_build_tracking_features(tmp_path: Path) -> None:
    _tracking_fixture(tmp_path)
    player_play = _player_play_fixture()
    assert tag_roles(player_play)["role"].tolist() == ["db", "rec", "other"]

    features = build_tracking_features(tmp_path, player_play)

    assert features["nfl_id"].tolist() == [10]
    assert features["speed_at_pass_forward"].iloc[0] == pytest.approx(2.0)
    assert features["backpedal_share"].iloc[0] == 1.0
    assert np.isfinite(features["nearest_receiver_dist_pf"].iloc[0])
    assert np.isfinite(features["dist_to_target_arrival"].iloc[0])


def _attempt_csv(root: Path) -> None:
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2025-03-02T18:00:00")
    for player in range(1, 31):
        for attempt in (1, 2):
            for i in range(40):
                stamp = (base + pd.Timedelta(seconds=0.1 * i)).strftime("%Y-%m-%dT%H:%M:%S.%f")
                scale = 5.0 + 0.1 * player + 0.01 * attempt
                rows.append(
                    dict(
                        nfl_id=player,
                        entity_type="PLAYER",
                        time=stamp,
                        drill_name="W_DRILL" if attempt == 1 else "FORTY_YARD_DASH",
                        drill_type="SKILL_DRILLS_DB" if attempt == 1 else "FORTY_YARD_DASH",
                        attempt=attempt,
                        x=scale * 0.1 * i,
                        y=0.0,
                    )
                )
    rows.append(
        dict(
            nfl_id=99,
            entity_type="PLAYER",
            time=base.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            drill_name="W_DRILL",
            drill_type="SKILL_DRILLS_DB",
            attempt=1,
            x=0.0,
            y=0.0,
        )
    )
    rows.append(
        dict(
            nfl_id=1,
            entity_type="PLAYER",
            time=base.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            drill_name="CORNER_ROUTE",
            drill_type="SKILL_DRILLS_WR",
            attempt=1,
            x=0.0,
            y=0.0,
        )
    )
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(raw / "combine_tracking.csv", index=False)


def test_build_attempts_filters_players_and_drill_types(tmp_path: Path) -> None:
    _attempt_csv(tmp_path)

    attempts = m.build_attempts(tmp_path, {str(i) for i in range(1, 31)}, m.DB_DRILL_TYPES)

    assert set(attempts["drill"]) == {"W_DRILL", "FORTY_YARD_DASH"}
    assert attempts["nfl_id"].nunique() == 30
    assert {"cod_speed_loss_fraction", "sensor_ten_yd_time"} <= set(attempts.columns)
    empty = m.build_attempts(tmp_path, {"12345"}, m.TE_DRILL_TYPES)
    assert empty.empty


def test_classify_predictors_and_sensor_wide() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for player in range(1, 41):
        level = rng.normal()
        for attempt in ("1", "2", "3"):
            rows.append(
                dict(
                    nfl_id=player,
                    drill="FORTY_YARD_DASH",
                    attempt=attempt,
                    top_speed=level + rng.normal(scale=0.05),
                    noise=rng.normal(),
                )
            )
        rows.append(dict(nfl_id=player, drill="BOX_DRILL", attempt="1", top_speed=level, noise=rng.normal()))
    attempts = pd.DataFrame(rows)

    table = m.classify_predictors(attempts, ["FORTY_YARD_DASH", "BOX_DRILL"])
    status = table.set_index("predictor")["status"]

    assert status["forty_yard_dash_top_speed"] == "repeatable"
    assert status["forty_yard_dash_noise"] == "unreliable"
    assert status["box_drill_top_speed"] == "unverifiable"
    repeatable = m.sensor_wide(attempts, table, "repeatable")
    assert list(repeatable.columns) == ["nfl_id", "forty_yard_dash_top_speed"]
    assert m.sensor_wide(attempts, table.iloc[:0], "repeatable").columns.tolist() == ["nfl_id"]


def _play_level(players: int = 8, plays: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    ids = np.repeat(np.arange(1, players + 1), plays)
    skill = np.linspace(-1.0, 1.0, players)[ids - 1]
    count = len(ids)
    assigned = rng.random(count) < 0.6
    is_run = rng.random(count) < 0.4
    return pd.DataFrame(
        {
            "game_id": np.arange(count) // 7,
            "play_id": np.arange(count),
            "nfl_id": ids,
            "nfl_position": np.where(ids % 2 == 0, "CB", "FS"),
            "lined_up_position": np.where(ids % 2 == 0, "CB", "HIGH_SAFETY"),
            "draft_year": 2024,
            "season": 2024,
            "play_nullified_by_penalty": "N",
            "pass_result": np.where(is_run, None, "C"),
            "pass_length": np.where(is_run, np.nan, rng.integers(0, 20, count)),
            "down": rng.integers(1, 4, count),
            "yards_to_go": rng.integers(1, 12, count),
            "team_coverage_man_zone": np.where(assigned, "MAN_COVERAGE", None),
            "team_coverage_type": np.where(assigned, "COVER_1", None),
            "offense_formation": "SHOTGUN",
            "coverage_assignment_at_snap": np.where(assigned, np.where(rng.random(count) < 0.3, "MAN", "HOOK"), None),
            "tackle": np.where(rng.random(count) < 0.1, 1.0, np.nan),
            "assist": np.where(rng.random(count) < 0.05, 1.0, np.nan),
            "caused_forced_fumble": np.nan,
            "tackle_for_loss": None,
            "snap_signal": skill + rng.normal(scale=0.3, size=count),
            "noise": rng.normal(size=count),
        }
    )


def test_gate_outcome_scores_stable_and_rejects_noise() -> None:
    frame = _play_level().assign(is_pass="True")
    stable, score = m.gate_outcome(
        frame, "snap_signal", min_plays=30, group="DB", numeric=("down",), categorical=("lined_up_position", "is_pass")
    )
    noisy, none_score = m.gate_outcome(
        frame, "noise", min_plays=30, group="DB", numeric=("down",), categorical=("lined_up_position",)
    )

    assert stable["verdict"] == "adequate"
    assert score is not None and score["snap_signal"].notna().sum() == 8
    assert "C(lined_up_position)" in str(stable["formula"])
    assert "C(is_pass)" not in str(noisy["formula"])
    assert noisy["verdict"] != "adequate"
    assert none_score is None


def test_gate_outcome_flags_too_few_players_and_honours_basis() -> None:
    frame = _play_level(players=2)
    gate, score = m.gate_outcome(
        frame, "snap_signal", min_plays=30, group="DB", numeric=(), categorical=(), basis=frame["snap_signal"].gt(0)
    )

    assert gate["verdict"] == "poor"
    assert gate["n_players"] == 2 or gate["n_players"] < 3
    assert score is None


def test_outcome_frames_and_run_gates_cover_all_outcomes() -> None:
    player_play = _play_level()
    keys = player_play[["game_id", "play_id", "nfl_id"]]
    features = keys.assign(**{name: np.random.default_rng(2).normal(size=len(keys)) for name in m.G2_OUTCOMES})
    g1, g2 = m.outcome_frames(player_play, features)

    assert g1["run_support_rate"].notna().sum() == g1["pass_result"].isna().sum()
    assert set(g2["is_pass"]) == {"True", "False"}
    gate, scores = m.run_gates(g1, g2)

    assert gate["outcome"].tolist() == [*m.G1_OUTCOMES, *m.G2_OUTCOMES]
    assert set(gate["verdict"]) <= {"adequate", "weak", "poor"}
    assert "nfl_id" in scores.columns


def _db_frame(count: int = 70) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    weight = rng.normal(190.0, 12.0, count)
    forty = rng.normal(4.55, 0.1, count)
    return pd.DataFrame(
        {
            "nfl_id": np.arange(count),
            "combine_weight": weight,
            "role": np.where(np.arange(count) % 2 == 0, "CB", "S"),
            "draft_year": np.resize([2023, 2024, 2025], count),
            "forty": forty,
            "noise_metric": rng.normal(size=count),
            "constant": 1.0,
            "snap_reaction_time": forty + rng.normal(scale=0.05, size=count),
            "speed_1s": rng.normal(size=count),
        }
    )


def test_association_table_computes_oriented_partial_correlation_and_bh() -> None:
    frame = _db_frame()
    pairs = [
        ("forty", "official", "snap_reaction_time"),
        ("noise_metric", "sensor", "speed_1s"),
        ("constant", "x", "speed_1s"),
    ]

    table = m.association_table(frame, pairs, "pair_x")

    assert table["family_label"].eq("pair_x").all()
    assert len(table) == 2
    top = table.set_index("predictor").loc["forty"]
    assert top["r_adj"] > 0.5
    assert top["oriented_r"] > 0.5 and top["direction"] == "expected"
    assert table.set_index("predictor").loc["noise_metric", "direction"] == "unclassified"
    assert table["q_scope"].between(0, 1).all()
    assert not table["low_power"].all() or len(frame) < 60
    assert m.association_table(frame.iloc[:20], pairs, "pair_x").empty


def test_association_table_te_group_has_no_role_control_and_low_power_flag() -> None:
    frame = _db_frame(35).assign(role="TE")
    table = m.association_table(
        frame, [("forty", "sensor", "snap_reaction_time")], "te_only", min_n=30, use_role=False, group="TE"
    )

    assert table["scope"].tolist() == ["te_drills"]
    assert table["low_power"].tolist() == [True]


def test_build_families_respects_repeatability_status_and_gate() -> None:
    predictors = pd.DataFrame(
        {
            "drill": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "W_DRILL", "FORTY_YARD_DASH", "BOX_DRILL"],
            "predictor": [
                "back_pedal_and_transition_45_degree_reaction_top_speed",
                "w_drill_top_speed",
                "forty_yard_dash_top_speed",
                "box_drill_top_speed",
            ],
            "status": ["repeatable", "unverifiable", "repeatable", "unverifiable"],
        }
    )
    available = {name: "sensor" for name in predictors["predictor"]} | {"forty": "official", "ten_yd_split": "official"}
    adequate = ["speed_1s", "closing_speed_nearest", "run_support_rate"]

    families = m.build_families(predictors, available, adequate)

    assert families["pair_backpedal"] == [
        ("back_pedal_and_transition_45_degree_reaction_top_speed", "sensor", "speed_1s")
    ]
    assert families["pair_cod"] == []
    assert {pair[0] for pair in families["pair_cod_unverifiable"]} == {"w_drill_top_speed", "box_drill_top_speed"}
    assert {pair[2] for pair in families["pair_cod_unverifiable"]} == {"closing_speed_nearest"}
    assert {pair[0] for pair in families["pair_speed_burst"]} == {"forty_yard_dash_top_speed", "forty", "ten_yd_split"}
    assert len(families["all_by_all"]) == 4 * 3
    assert {pair[0] for pair in families["unverifiable_repeatability"]} == {"w_drill_top_speed", "box_drill_top_speed"}


def test_family_summary_counts_survivors_and_directions() -> None:
    results = pd.DataFrame(
        {
            "family_label": ["a", "a", "a", "b"],
            "p": [0.001, 0.2, 0.04, 0.5],
            "q_scope": [0.02, 0.4, 0.09, 0.5],
            "direction": ["expected", "opposite", "opposite", "unclassified"],
            "r_disc_2023_24": [0.3, -0.1, -0.2, 0.1],
            "r_hold_2025": [0.2, 0.1, -0.1, 0.1],
        }
    )

    summary = m.family_summary(results).set_index("family_label")

    assert summary.loc["a", "q_lt_10"] == 2
    assert summary.loc["a", "q_lt_05"] == 1
    assert summary.loc["a", "survivors_holdout_same_sign"] == 2
    assert summary.loc["a", "expected_all"] == 1 and summary.loc["a", "opposite_all"] == 2
    assert summary.loc["b", "expected_raw_p_lt_05"] == 0


def test_te_frame_and_families() -> None:
    combine = pd.DataFrame(
        {
            "nfl_id": [1, 2, 3],
            "combine_position": ["TE", "TE", "WR"],
            "combine_weight": [250, 255, 200],
            "draft_year": [2024, 2025, 2024],
        }
    )
    players = pd.DataFrame({"nfl_id": [1, 2, 3], "nfl_position": ["TE", "TE", "WR"]})
    sensors = pd.DataFrame({"nfl_id": [1, 2], "block_explosion_top_speed": [5.0, 6.0]})
    scores = pd.DataFrame({"nfl_id": [1, 2], "target_rate": [0.1, 0.2]})

    frame = m.te_frame(combine, players, sensors, scores)
    predictors = pd.DataFrame(
        {
            "predictor": ["block_explosion_top_speed", "hook_route_right_top_speed"],
            "status": ["unverifiable", "repeatable"],
        }
    )
    families = m.te_families(predictors, frame.columns, ["target_rate"])

    assert frame["nfl_id"].tolist() == [1, 2]
    assert frame["role"].eq("TE").all()
    assert families["te_unverifiable"] == [("block_explosion_top_speed", "sensor", "target_rate")]
    assert families["te_only"] == []


def _route_rows(count_tes: int = 6, routes: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    ids = np.repeat(np.arange(1, count_tes + 1), routes)
    size = len(ids)
    return pd.DataFrame(
        {
            "nfl_id": ids,
            "nfl_position": np.where(ids <= 4, "TE", "WR"),
            "route_ran": "HITCH",
            "play_nullified_by_penalty": "N",
            "target": rng.random(size) < 0.2,
            "rec_yards": rng.normal(5.0, 2.0, size),
            "yards_after_catch": rng.normal(3.0, 1.0, size),
            "expected_yards_after_catch": rng.normal(3.0, 1.0, size),
            "expected_points_added": rng.normal(size=size),
            "separation_at_pass_forward": np.repeat(np.linspace(1.0, 4.0, count_tes), routes)
            + rng.normal(0.0, 0.3, size),
            "team_coverage_man_zone": "ZONE_COVERAGE",
            "team_coverage_type": "COVER_3",
            "cushion": rng.normal(6.0, 1.0, size),
            "down": rng.integers(1, 4, size),
            "yards_to_go": rng.integers(1, 12, size),
            "offense_formation": "SHOTGUN",
            "pass_length": rng.integers(0, 15, size),
        }
    )


def test_te_outcome_scores_gate_only_tight_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "TE_MIN_ROUTES", 50)
    gate, scores = m.te_outcome_scores(_route_rows())

    assert set(gate["outcome"]) == set(m.TE_OUTCOMES)
    assert gate["group"].eq("TE").all()
    stable = gate.set_index("outcome").loc["separation_at_pass_forward"]
    assert stable["n_players"] == 4
    assert stable["verdict"] in {"adequate", "weak", "poor"}
    assert set(scores["nfl_id"]) == {1, 2, 3, 4}


def test_db_predictor_frame_keeps_only_players_with_enough_data() -> None:
    count = 50
    combine = pd.DataFrame(
        {
            "nfl_id": np.arange(count),
            "combine_position": "DB",
            "combine_weight": np.linspace(180, 210, count),
            "draft_year": 2024,
            "forty": np.linspace(4.4, 4.7, count),
            "vertical": np.nan,
        }
    )
    players = pd.DataFrame({"nfl_id": np.arange(count), "nfl_position": np.where(np.arange(count) % 2, "CB", "SS")})
    attempts = pd.DataFrame(
        {
            "nfl_id": np.arange(count),
            "drill": "BOX_DRILL",
            "attempt": "1",
            "top_speed": np.linspace(5, 9, count),
        }
    )
    table = pd.DataFrame(
        {
            "drill": ["BOX_DRILL"],
            "metric": ["top_speed"],
            "predictor": ["box_drill_top_speed"],
            "status": ["unverifiable"],
        }
    )

    frame, available = m._db_predictor_frame(combine, players, attempts, table)

    assert available == {"forty": "official", "box_drill_top_speed": "sensor_unverifiable"}
    assert set(frame["role"]) == {"CB", "S"}


def test_cached_builds_once(tmp_path: Path) -> None:
    calls: list[int] = []

    def build() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"a": [1]})

    path = tmp_path / "cache.parquet"
    first = m._cached(path, build)
    second = m._cached(path, build)

    assert len(calls) == 1
    assert first.equals(second)


def test_infeasible_outcomes_are_documented() -> None:
    names = {name for name, _ in m.INFEASIBLE}
    assert {"pass_coverage_yards_allowed", "te_blocking"} <= names
