from __future__ import annotations

import numpy as np
import pandas as pd

import combine_dlol.dl_outcomes as outcomes
from combine_dlol.dl_outcomes import adjust_dl_outcome, first_step_metrics, summarize_outcome


def _tracking(x_values: list[float], events: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1] * len(x_values),
            "play_id": [2] * len(x_values),
            "nfl_id": [3] * len(x_values),
            "time": pd.date_range("2025-09-01 12:00:00", periods=len(x_values), freq="100ms"),
            "x_rel": x_values,
            "s": [0.0] * len(x_values),
            "event": events,
        }
    )


def test_first_step_metrics_recover_known_offense_downfield_motion() -> None:
    frame = _tracking([0.0 + 0.1 * value for value in range(11)], ["ball_snap"] + [None] * 10)
    metrics = first_step_metrics(frame).iloc[0]
    assert np.isclose(float(metrics["displacement_0_5"]), 0.5)
    assert np.isclose(float(metrics["displacement_1_0"]), 1.0)
    assert np.isclose(float(metrics["speed_0_5"]), 1.0)


def test_first_step_window_starts_at_ball_snap() -> None:
    frame = _tracking([-30.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [None, "ball_snap", None, None, None, None, None])
    frame["time"] = pd.date_range("2025-09-01 12:00:00", periods=len(frame), freq="100ms")
    metrics = first_step_metrics(frame).iloc[0]
    assert np.isclose(float(metrics["displacement_0_5"]), 1.0)
    assert np.isnan(float(metrics["displacement_1_0"]))


def test_adjustment_is_orthogonal_to_context() -> None:
    rng = np.random.default_rng(41)
    size = 300
    down = rng.integers(1, 5, size)
    yards = rng.uniform(1.0, 15.0, size)
    frame = pd.DataFrame(
        {
            "outcome": 1.8 * down + 0.7 * yards + rng.normal(0.0, 0.1, size),
            "lined_up_position": np.where(down < 3, "LE", "RE"),
            "down": down,
            "yards_to_go": yards,
            "offense_formation": np.where(yards < 8.0, "SHOTGUN", "SINGLEBACK"),
            "blitzing": np.where(down % 2 == 0, "Y", "N"),
            "snap_x_rel": rng.normal(0.5, 0.2, size),
            "snap_y_rel": rng.normal(26.0, 5.0, size),
        }
    )
    adjusted = adjust_dl_outcome(frame, "outcome")
    assert abs(float(np.corrcoef(adjusted["outcome_residual"], adjusted["yards_to_go"])[0, 1])) < 1e-8
    assert abs(float(adjusted.groupby("down")["outcome_residual"].mean().max())) < 1e-8


def test_reliability_separates_planted_effects_from_noise() -> None:
    rng = np.random.default_rng(42)
    players = np.repeat(np.arange(40), 50)
    effects = np.repeat(rng.normal(0.0, 1.0, 40), 50)
    planted = pd.DataFrame({"nfl_id": players, "value": effects + rng.normal(0.0, 0.6, len(players))})
    noise = pd.DataFrame({"nfl_id": players, "value": rng.normal(0.0, 1.0, len(players))})
    planted_summary = summarize_outcome(planted, "value", n_splits=100, n_bootstrap=100, n_permutations=40)
    noise_summary = summarize_outcome(noise, "value", n_splits=100, n_bootstrap=100, n_permutations=40)
    assert planted_summary["split_half_median"] > 0.75
    assert abs(noise_summary["split_half_median"]) < 0.30


def _residual_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(91)
    players = np.repeat(np.arange(1, 7), 50)
    positions = np.repeat(["DE", "OLB", "DE", "DT", "NT", "DT"], 50)
    indexes = np.arange(len(players))
    base = np.repeat(rng.normal(0.0, 0.4, 6), 50) + rng.normal(0.0, 0.2, len(players))
    complete = pd.DataFrame(
        {
            "game_id": indexes,
            "play_id": indexes,
            "nfl_id": players,
            "nfl_position": positions,
            "player_get_off": base,
            "displacement_0_5": base * 1.3,
            "displacement_1_0": base * 1.6,
            "speed_0_5": base * 1.9,
            "supplied_speed_0_5": -base * 1.9,
            "draft_year": 2023,
            "forty": 4.7,
            "season": 2023,
        }
    )
    residuals: dict[str, pd.DataFrame] = {}
    for label, outcome in outcomes.OUTCOMES.items():
        frame = complete.loc[:, ["game_id", "play_id", "nfl_id", "nfl_position"]].copy()
        frame[f"{outcome}_residual"] = complete[outcome]
        residuals[label] = frame
    return complete, complete.copy(), residuals


def test_population_scores_counts_and_validity() -> None:
    complete, population, residuals = _residual_frames()
    population["pass_result"] = "C"
    population["play_nullified_by_penalty"] = "N"
    selected = outcomes.prepare_dl_population(population)
    assert len(selected) == len(population)
    scores = outcomes._scores(residuals)
    assert {"raw_d_a", "shrunk_d_b_speed_0_5", "reliability_d_b_displacement_1_0"}.issubset(scores.columns)
    reliability = pd.DataFrame({"outcome": list(outcomes.OUTCOMES), "group": ["all"] * 4})
    validity = outcomes._validity(reliability, complete, residuals)
    assert validity.loc[validity["outcome"].eq("d_b_speed_0_5"), "validity_raw_absolute_pass"].item()
    counts = outcomes._counts(selected, complete, residuals)
    assert counts.loc[counts["measure"].eq("eligible_players_min_50"), "value"].item() == 6
    compared = outcomes._with_speed_comparison(counts, complete)
    assert compared.loc[compared["measure"].str.contains("coordinate"), "value"].item() < -0.99


def test_tracking_helper_and_plots(tmp_path, monkeypatch) -> None:
    complete, population, residuals = _residual_frames()
    population["yardline_side"] = "A"
    population["yardline_number"] = 20
    population["possession_team"] = "A"
    population["play_direction"] = "right"
    metric_population = population.iloc[:1].copy()
    tracked = _tracking([0.0 + 0.1 * value for value in range(11)], ["ball_snap"] + [None] * 10)
    tracked["nfl_id"] = int(metric_population["nfl_id"].iloc[0])
    tracked["game_id"] = int(metric_population["game_id"].iloc[0])
    tracked["play_id"] = int(metric_population["play_id"].iloc[0])
    tracked["x"] = tracked["x_rel"]
    tracked["y"] = 25.0

    def fake_iter(*args, **kwargs):
        yield tracked.loc[:, ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"]]

    def fake_los(frame, context):
        return frame.assign(x_rel=frame["x"], y_rel=frame["y"])

    monkeypatch.setattr(outcomes, "iter_game_tracking", fake_iter)
    monkeypatch.setattr(outcomes, "add_line_of_scrimmage", fake_los)
    metrics = outcomes._metric_frame_for_season(2023, metric_population, population, None)
    assert np.isclose(metrics["displacement_0_5"].item(), 0.5)
    assert outcomes._metric_frame_for_season(2024, metric_population, population, None).empty
    split_results = {
        label: outcomes.split_half_reliability(frame, "nfl_id", f"{outcome}_residual", 50, n_splits=10)
        for label, (outcome, frame) in (
            (label, (outcomes.OUTCOMES[label], residuals[label])) for label in outcomes.OUTCOMES
        )
    }
    monkeypatch.setattr(outcomes, "permutation_null", lambda *args, **kwargs: np.array([-0.1, 0.0, 0.1]))
    reliability_plot = tmp_path / "reliability.png"
    outcomes._plot_reliability(split_results, residuals, reliability_plot)
    assert reliability_plot.is_file()
    trajectory_plot = tmp_path / "trajectory.png"
    outcomes._plot_trajectories(complete, population, None, trajectory_plot)
    assert trajectory_plot.is_file()


def test_run_and_render_paths(tmp_path, monkeypatch) -> None:
    complete, population, residuals = _residual_frames()
    root = tmp_path
    (root / "data" / "processed").mkdir(parents=True)
    (root / "reports").mkdir()
    monkeypatch.setattr(
        outcomes,
        "_phase_data",
        lambda columns, path: (population, population, complete, residuals, set(complete["nfl_id"])),
    )
    reliability = pd.DataFrame({"outcome": list(outcomes.OUTCOMES), "group": ["all"] * 4, "verdict": ["adequate"] * 4})
    monkeypatch.setattr(outcomes, "_reliability_rows", lambda data: (reliability, {}))
    monkeypatch.setattr(outcomes, "_validity", lambda table, matched, data: table)
    monkeypatch.setattr(
        outcomes, "_counts", lambda population, matched, data: pd.DataFrame({"measure": ["base"], "value": [1]})
    )
    monkeypatch.setattr(outcomes, "_scores", lambda data: pd.DataFrame({"nfl_id": [1], "n_plays": [50]}))
    monkeypatch.setattr(outcomes, "_plot_reliability", lambda *args: None)
    monkeypatch.setattr(outcomes, "_plot_trajectories", lambda *args: None)
    written, counts = outcomes.run_phase(root)
    assert written.equals(reliability)
    assert (root / "data" / "processed" / "dl_outcomes_scores.parquet").is_file()
    assert counts.loc[counts["measure"].str.contains("coordinate"), "value"].item() < -0.99
    original_split = outcomes.split_half_reliability
    monkeypatch.setattr(
        outcomes,
        "split_half_reliability",
        lambda frame, player, value, min_n, n_splits: original_split(frame, player, value, min_n, n_splits=10),
    )
    rendered = outcomes.render_phase(root)
    assert len(rendered) == 2


def test_population_and_helper_edge_cases(monkeypatch) -> None:
    with np.testing.assert_raises(ValueError):
        outcomes.prepare_dl_population(pd.DataFrame({"nfl_id": [1]}))
    without_snap = _tracking([0.0, 0.1], [None, None])
    assert first_step_metrics(without_snap).empty
    with np.testing.assert_raises(ValueError):
        first_step_metrics(pd.DataFrame())
    assert np.isnan(outcomes._nearest_value(pd.DataFrame({"seconds_since_snap": [-0.1], "x_rel": [1.0]}), 0.0, "x_rel"))
    frame = pd.DataFrame({"nfl_id": [1, 2, 3], "nfl_position": ["DE", "DT", "OLB"]})
    monkeypatch.setattr(
        outcomes,
        "_metric_frame_for_season",
        lambda season, population, player_play, root: frame if season == 2023 else pd.DataFrame(),
    )
    combined = outcomes.tracking_outcomes(frame, frame)
    assert len(combined) == 3
    assert len(outcomes._subgroup(frame, ("DE", "OLB"))) == 2


def test_eligible_residuals_keeps_players_with_fifty_adjusted_plays() -> None:
    complete, _, _ = _residual_frames()
    complete["lined_up_position"] = np.where(complete["nfl_position"].eq("DE"), "EDGE", "INTERIOR")
    complete["down"] = np.tile([1, 2], len(complete) // 2)
    complete["yards_to_go"] = np.tile([5.0, 10.0], len(complete) // 2)
    complete["offense_formation"] = np.where(complete["down"].eq(1), "SHOTGUN", "SINGLEBACK")
    complete["blitzing"] = complete["down"].eq(1)
    complete["snap_x_rel"] = 0.5
    complete["snap_y_rel"] = 26.0
    eligible = outcomes._eligible_residuals(complete, "player_get_off")
    assert eligible["nfl_id"].nunique() == 6
