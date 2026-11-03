from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import combine_dlol.ol_outcomes as ol
from combine_dlol.ol_outcomes import (
    ReliabilitySummary,
    build_population,
    detect_snap_movement,
    fit_adjusted_outcome,
    summarize_movement,
)


def _trajectory(onset: float = 0.3, pre_speed: float = 0.1) -> pd.DataFrame:
    seconds = np.round(np.arange(-0.3, 1.01, 0.1), 2)
    x = np.where(seconds >= onset, (seconds - onset) * 2.0, 0.0)
    speed = np.where(seconds >= onset, 1.0, pre_speed)
    return pd.DataFrame({"seconds_since_snap": seconds, "x_rel": x, "y_rel": 0.0, "s": speed})


def test_first_movement_detection_finds_known_onset_and_displacement() -> None:
    detected = detect_snap_movement(_trajectory())
    assert detected.included
    assert detected.reaction_time == 0.3
    assert detected.displacement_0_5 == 0.4
    assert not detected.censored


def test_first_movement_drops_pre_snap_motion_and_censors() -> None:
    moving = detect_snap_movement(_trajectory(pre_speed=0.8))
    still = _trajectory(onset=2.0)
    censored = detect_snap_movement(still)
    assert not moving.included
    assert moving.pre_snap_motion
    assert censored.included
    assert censored.censored
    assert censored.reaction_time == 1.0


def test_displacement_window_does_not_use_post_half_second_frames() -> None:
    trajectory = _trajectory(onset=2.0)
    trajectory.loc[trajectory["seconds_since_snap"].eq(0.6), "x_rel"] = 99.0
    detected = detect_snap_movement(trajectory)
    assert detected.displacement_0_5 == 0.0


def test_adjustment_is_orthogonal_to_context() -> None:
    rng = np.random.default_rng(7)
    context = np.repeat(np.arange(8), 30)
    frame = pd.DataFrame(
        {
            "outcome": 2.0 * context + rng.normal(0.0, 0.1, len(context)),
            "context": context,
            "group": np.where(context % 2, "a", "b"),
        }
    )
    adjusted = fit_adjusted_outcome(frame, "outcome", ["context"], ["group"])
    assert abs(float(np.corrcoef(adjusted["outcome_residual"], adjusted["context"])[0, 1])) < 1e-10


def test_synthetic_player_effect_and_noise_are_distinguished() -> None:
    rng = np.random.default_rng(9)
    players = np.repeat(np.arange(30), 12)
    effects = np.repeat(rng.normal(size=30), 12)
    planted = pd.DataFrame({"nfl_id": players, "value": effects + rng.normal(0.0, 0.2, len(players))})
    noise = pd.DataFrame({"nfl_id": players, "value": rng.normal(size=len(players))})
    assert summarize_movement(planted, "value", min_plays=10, n_splits=100, n_bootstrap=100).median > 0.8
    assert abs(summarize_movement(noise, "value", min_plays=10, n_splits=100, n_bootstrap=100).median) < 0.3


def test_population_and_vectorized_snap_extraction() -> None:
    population = pd.DataFrame(
        {
            "nfl_position": ["T", "G", "C", "DE"],
            "peak_pressure_probability_allowed": [0.1, 0.2, np.nan, 0.3],
            "play_nullified_by_penalty": ["N", "Y", "N", "N"],
        }
    )
    assert len(build_population(population)) == 1
    frames = []
    for player, speed in [(1, 0.1), (2, 0.8)]:
        for second in np.round(np.arange(-0.2, 0.6, 0.1), 2):
            frames.append(
                {
                    "game_id": 1,
                    "play_id": player,
                    "nfl_id": player,
                    "time": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=float(second)),
                    "event": "ball_snap" if second == 0.0 else None,
                    "x_rel": max(second, 0.0),
                    "y_rel": 0.0,
                    "s": speed if second < 0 else 1.0,
                }
            )
    extracted, diagnostics = ol.snap_outcomes(pd.DataFrame(frames))
    assert extracted["nfl_id"].tolist() == [1]
    assert int(diagnostics["pre_snap_motion"].sum()) == 1


def test_streaming_helpers_preserve_trajectory_boundaries(monkeypatch: object) -> None:
    first = pd.DataFrame({"game_id": [1, 1], "play_id": [1, 1], "nfl_id": [1, 1]})
    second = pd.DataFrame({"game_id": [1, 1], "play_id": [1, 2], "nfl_id": [1, 2]})
    result = list(ol._complete_trajectories([first, second]))
    assert sum(len(frame) for frame in result) == 4

    def fake_tracking(*args: object, **kwargs: object) -> list[pd.DataFrame]:
        return [pd.DataFrame({"game_id": [1], "play_id": [1], "nfl_id": [1]})]

    monkeypatch.setattr(ol, "iter_game_tracking", fake_tracking)
    monkeypatch.setattr(ol, "add_line_of_scrimmage", lambda tracking, context: tracking.assign(x_rel=0.0, y_rel=0.0))
    population = pd.DataFrame({"game_id": [1], "play_id": [1], "nfl_id": [1], "season": [2023]})
    assert len(list(ol._tracking_for_season(population, 2023))) == 1
    assert ol._tracking_columns()[-1] == "event"


def _summary() -> ReliabilitySummary:
    return ReliabilitySummary(0.6, 0.4, 0.8, 0.3, 0.8, 0.9, 0.95, 0.98, 0.0, "adequate")


def test_reporting_helpers_and_plots(monkeypatch: object, tmp_path: object) -> None:
    rows = []
    for player, position in [(1, "T"), (2, "T"), (3, "T"), (4, "G"), (5, "G"), (6, "C")]:
        for play in range(50):
            rows.append({"nfl_id": player, "nfl_position": position, "value_residual": float(play)})
    adjusted = pd.DataFrame(rows)
    monkeypatch.setattr(ol, "summarize_movement", lambda *args, **kwargs: _summary())
    table = ol._reliability_table({"value_residual": adjusted})
    assert set(table["subgroup"]) == {"all", "T", "G_C"}
    monkeypatch.setattr(ol, "mixed_model_icc", lambda *args, **kwargs: SimpleNamespace(icc=0.3))
    monkeypatch.setattr(
        ol,
        "empirical_bayes_shrink",
        lambda data, *args, **kwargs: pd.DataFrame(
            {
                "nfl_id": [1],
                "n": [50],
                "raw_mean": [0.0],
                "grand_mean": [0.0],
                "reliability": [1.0],
                "posterior_mean": [0.0],
            }
        ),
    )
    scores = ol._scores({"value_residual": adjusted})
    assert scores["outcome"].tolist() == ["value_residual"]
    plot_path = tmp_path / "reliability.png"
    ol._plot_reliability(table, plot_path)
    assert plot_path.is_file()


def test_run_phase_writes_requested_artifacts(monkeypatch: object, tmp_path: object) -> None:
    population = pd.DataFrame(
        {
            "game_id": [1], "play_id": [1], "nfl_id": [1], "nfl_position": ["T"], "forty": [5.0],
            "peak_pressure_probability_allowed": [0.2], "pressure_allowed": [False], "snap_reaction_time": [0.1],
            "first_0_5s_displacement": [0.0], "season": [2023], "lined_up_position": ["LT"], "down": [1],
            "yards_to_go": [10],
            "offense_formation": ["SHOTGUN"],
            "dropback_duration": [2.0],
            "pass_rushers_encountered": [4],
        }
    )
    snap = pd.DataFrame({"game_id": [1], "play_id": [1], "nfl_id": [1], "reaction_censored": [0]})
    monkeypatch.setattr(ol, "_analysis_population", lambda: population)
    diagnostics = pd.DataFrame({"pre_snap_motion": [0], "missing_snap": [0]})
    monkeypatch.setattr(ol, "_run_snap_extraction", lambda data: (snap, diagnostics))
    monkeypatch.setattr(ol, "fit_adjusted_outcome", lambda data, outcome: data.assign(**{f"{outcome}_residual": 0.0}))
    reliability = pd.DataFrame(
        {
            "outcome": ["x"],
            "subgroup": ["all"],
            "split_half_spearman_brown_median": [0.6],
            "bootstrap_ci_lower": [0.4],
            "bootstrap_ci_upper": [0.8],
        }
    )
    monkeypatch.setattr(ol, "_reliability_table", lambda data: reliability)
    monkeypatch.setattr(ol, "_scores", lambda data: pd.DataFrame({"nfl_id": [1], "outcome": ["x"]}))
    monkeypatch.setattr(ol, "_plot_reliability", lambda table, path: path.touch())
    monkeypatch.setattr(ol, "_plot_snap_examples", lambda data, path: path.touch())
    table, counts, scores = ol.run_phase_one(tmp_path)
    assert len(table) == len(counts) == len(scores) == 1
    assert (tmp_path / "data" / "processed" / "ol_outcomes_scores.parquet").is_file()
    assert (tmp_path / "reports" / "phase1_ol_counts.csv").is_file()


def test_snap_plot_and_extraction_runner(monkeypatch: object, tmp_path: object) -> None:
    rows = []
    for play in range(12):
        for second in np.round(np.arange(-0.2, 1.01, 0.1), 2):
            rows.append(
                {
                    "game_id": 1,
                    "play_id": play,
                    "nfl_id": play,
                    "time": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=float(second)),
                    "event": "ball_snap" if second == 0.0 else None,
                    "x_rel": max(second, 0.0),
                    "y_rel": 0.0,
                    "s": 0.1 if second < 0.0 else 1.0,
                }
            )
    tracking = pd.DataFrame(rows)
    monkeypatch.setattr(ol, "_tracking_for_season", lambda population, season: [tracking])
    features, diagnostics = ol._run_snap_extraction(pd.DataFrame())
    assert len(features) == 36
    assert diagnostics["pre_snap_motion"].sum() == 0
    plot_path = tmp_path / "snap.png"
    ol._plot_snap_examples(pd.DataFrame(), plot_path)
    assert plot_path.is_file()
