from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol import combine_predictors
from combine_dlol.combine_predictors import (
    OFFICIAL_MEASURES,
    aggregate_sensor_attempts,
    attempt_kinematics,
    build,
    canonicalize_mirror,
    coverage_table,
    extract_attempts,
    first_sustained_movement,
    forty_speed_correlations,
    path_efficiency,
    predictor_eligibility,
    prepare_official,
    repeatability_table,
)


def test_first_sustained_movement_trims_initial_stillness() -> None:
    speed = np.array([0.0, 0.2, 0.74, 0.76, 0.80, 0.82, 0.0])

    assert first_sustained_movement(speed) == 3


def test_straight_trajectory_recovers_speed_burst_and_efficiency() -> None:
    time = np.arange(0.0, 4.1, 0.1)
    x = 3.0 * time
    metrics = attempt_kinematics(time, x, np.zeros_like(x), "FORTY_YARD_DASH")

    assert metrics["top_speed"] == pytest.approx(3.0, abs=0.01)
    assert metrics["burst_distance_0_1s"] == pytest.approx(3.0, abs=0.01)
    assert metrics["burst_speed_1_0s"] == pytest.approx(3.0, abs=0.01)
    assert metrics["path_efficiency"] == pytest.approx(1.0, abs=0.01)
    assert metrics["sensor_ten_yd_time"] == pytest.approx(10.0 / 3.0, abs=0.01)
    assert metrics["sensor_ten_yd_burst_speed"] == pytest.approx(3.0, abs=0.01)
    assert metrics["top_speed"] >= 0.0


def test_semicircle_path_efficiency_is_two_over_pi() -> None:
    theta = np.linspace(0.0, np.pi, 101)

    assert path_efficiency(np.cos(theta), np.sin(theta)) == pytest.approx(2.0 / np.pi, abs=0.001)


def test_aggregate_sensor_attempts_uses_median_and_spread() -> None:
    attempts = pd.DataFrame(
        {
            "nfl_id": ["1", "1", "2"],
            "group": ["DL", "DL", "DL"],
            "drill": ["FORTY_YARD_DASH"] * 3,
            "attempt": ["a", "b", "a"],
            "top_speed": [4.0, 6.0, 5.0],
        }
    )

    actual = aggregate_sensor_attempts(attempts, pd.Series(["1", "2"], dtype="string"))

    first = actual.loc[actual["nfl_id"].eq("1")].iloc[0]
    assert first["forty_yard_dash_n_attempts"] == 2
    assert first["forty_yard_dash_top_speed"] == pytest.approx(5.0)
    assert first["forty_yard_dash_top_speed_spread"] == pytest.approx(1.0)


def test_forty_speed_correlations_are_negative_for_both_groups() -> None:
    features = pd.DataFrame(
        {
            "nfl_id": ["1", "2", "3", "4", "5", "6"],
            "group": ["DL", "DL", "DL", "OL", "OL", "OL"],
            "forty": [5.0, 4.5, 4.0, 5.2, 4.7, 4.2],
        }
    )
    attempts = pd.DataFrame(
        {
            "nfl_id": ["1", "2", "3", "4", "5", "6"],
            "group": ["DL", "DL", "DL", "OL", "OL", "OL"],
            "drill": ["FORTY_YARD_DASH"] * 6,
            "attempt": ["1"] * 6,
            "top_speed": [7.0, 8.0, 9.0, 6.0, 7.0, 8.0],
        }
    )

    actual = forty_speed_correlations(features, attempts)

    assert set(actual["group"]) == {"DL", "OL"}
    assert (actual["correlation"] < -0.9).all()


def _combine_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = pd.DataFrame(
        {
            "nfl_id": [1, 2, 3],
            "nfl_position": ["DE", "T", "WR"],
            "draft_year": [2023, 2023, 2023],
        }
    )
    results = pd.DataFrame(
        {
            "nfl_id": [1, 2, 3],
            **{measure: [1.0, 2.0, 3.0] for measure in OFFICIAL_MEASURES},
        }
    )
    rows: list[dict[str, object]] = []
    for nfl_id, drill, drill_type in [
        (1, "FORTY_YARD_DASH", "FORTY_YARD_DASH"),
        (1, "PASS_RUSH_DRILL", "SKILL_DRILLS_DL"),
        (2, "FORTY_YARD_DASH", "FORTY_YARD_DASH"),
        (2, "PASS_PRO_MIRROR_DRILL", "SKILL_DRILLS_OL"),
        (2, "PASS_PRO-MIRROR_DRILL", "SKILL_DRILLS_OL"),
    ]:
        for attempt in [1, 2]:
            for frame in range(16):
                rows.append(
                    {
                        "nfl_id": nfl_id,
                        "entity_type": "PLAYER",
                        "time": f"2023-03-05T19:39:{frame / 10:04.1f}",
                        "drill_type": drill_type,
                        "drill_name": drill,
                        "attempt": attempt,
                        "x": 1.2 * frame + 0.1 * attempt,
                        "y": 0.0,
                    }
                )
    rows.append(
        {
            "nfl_id": 1,
            "entity_type": "DRILL",
            "time": "2023-03-05T19:39:00.0",
            "drill_type": "FORTY_YARD_DASH",
            "drill_name": "FORTY_YARD_DASH",
            "attempt": 1,
            "x": 0.0,
            "y": 0.0,
        }
    )
    return players, results, pd.DataFrame(rows)


def test_prepare_extract_canonicalize_and_cover_synthetic_combine_data() -> None:
    players, results, tracking = _combine_frames()

    official = prepare_official(players, results)
    attempts = extract_attempts(tracking, official)
    canonical = canonicalize_mirror(attempts)
    coverage = coverage_table(canonical)

    assert set(official["group"]) == {"DL", "OL"}
    assert official["nfl_id"].tolist() == ["1", "2"]
    assert official["forty_missing"].eq(0).all()
    assert attempts["attempt"].nunique() == 2
    assert attempts["top_speed"].ge(0.0).all()
    assert "PASS_PRO-MIRROR_DRILL" not in set(canonical["drill"])
    assert set(coverage["drill"]) == {"FORTY_YARD_DASH", "PASS_PRO_MIRROR_DRILL", "PASS_RUSH_DRILL"}


def test_repeatability_and_eligibility_apply_frozen_gate() -> None:
    attempts = pd.DataFrame(
        {
            "nfl_id": np.repeat([str(value) for value in range(25)], 2),
            "group": ["DL"] * 50,
            "drill": ["FORTY_YARD_DASH"] * 50,
            "attempt": [1, 2] * 25,
            "top_speed": np.repeat(np.arange(25, dtype=float), 2) + np.tile([0.0, 0.1], 25),
        }
    )
    official = pd.DataFrame(
        {
            "nfl_id": ["1"],
            "group": ["DL"],
            **{measure: [1.0] for measure in OFFICIAL_MEASURES},
        }
    )

    repeatability = repeatability_table(attempts)
    eligibility = predictor_eligibility(official, repeatability)

    sensor = eligibility.loc[eligibility["family"].eq("sensor")].iloc[0]
    assert sensor["status"] == "pass"
    assert bool(sensor["eligible_for_M3"])
    assert bool(sensor["eligible_for_M4"])
    assert eligibility.loc[eligibility["family"].eq("official"), "eligible_for_M4"].eq(False).all()


def test_build_writes_required_outputs_and_main_requires_logged_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    players, results, tracking = _combine_frames()
    monkeypatch.setattr(combine_predictors, "load_players", lambda root: players)
    monkeypatch.setattr(combine_predictors, "load_combine_results", lambda root: results)
    monkeypatch.setattr(combine_predictors, "load_combine_tracking", lambda root: tracking)
    result = build(tmp_path, merge_mirror=True)

    assert (tmp_path / "data" / "processed" / "combine_predictors.parquet").is_file()
    assert (tmp_path / "reports" / "phase2_combine_repeatability.csv").is_file()
    assert (tmp_path / "reports" / "phase2_combine_coverage.csv").is_file()
    assert (tmp_path / "reports" / "phase2_combine_predictors.csv").is_file()
    assert (tmp_path / "reports" / "phase2_combine_repeatability.png").is_file()
    assert result.features["nfl_id"].is_unique
    assert not result.mirror_comparison.empty
    monkeypatch.setattr(combine_predictors, "build", lambda root, merge_mirror: result)

    assert combine_predictors.main(["--root", str(tmp_path), "--merge-mirror"]) == 0
    assert "PASS_PRO_MIRROR_DRILL" in capsys.readouterr().out
    with pytest.raises(ValueError, match="logged comparison"):
        combine_predictors.main(["--root", str(tmp_path)])
