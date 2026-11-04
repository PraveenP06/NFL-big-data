from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol.scope_wrte import (
    OUTCOMES,
    RESULT_COLUMNS,
    _association_rows,
    _context_formula,
    _outcome_definitions,
    _pair_repeatability,
    benjamini_hochberg,
    build_composite,
    combine_stages,
    join_player_scores,
    orientation_label,
    partial_spearman,
    predictor_orientation,
    sensor_player_table,
    sensor_selection,
    stage_outcome,
)


def test_partial_spearman_removes_a_shared_rank_control() -> None:
    generator = np.random.default_rng(8)
    control = generator.normal(size=200)
    value = control * 3.0 + generator.normal(scale=0.5, size=200)
    outcome = control * -2.0 + generator.normal(scale=0.5, size=200)
    assert abs(partial_spearman(value, outcome, [control])) < 0.15


def test_benjamini_hochberg_is_monotone_and_bounded() -> None:
    values = benjamini_hochberg(np.array([0.03, 0.001, 0.04, 0.8]))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert values[1] == pytest.approx(0.004)
    ordered = values[np.argsort(np.array([0.03, 0.001, 0.04, 0.8]))]
    assert np.all(np.diff(ordered) >= 0.0)


def test_build_composite_orients_and_averages_available_components() -> None:
    data = pd.DataFrame({"fast": [1.0, 2.0, 3.0], "time": [3.0, 2.0, np.nan]})
    result = build_composite(data, (("fast", 1), ("time", -1)))
    assert result.iloc[0] < result.iloc[1]
    assert result.iloc[2] > result.iloc[1]


def test_join_player_scores_validates_one_row_per_player() -> None:
    predictors = pd.DataFrame({"nfl_id": [1, 2], "forty": [4.4, 4.5]})
    outcomes = pd.DataFrame({"nfl_id": [1, 2], "separation": [2.0, 1.0]})
    joined = join_player_scores(predictors, outcomes)
    assert joined.shape == (2, 3)
    duplicated = pd.concat([outcomes, outcomes.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row"):
        join_player_scores(predictors, duplicated)


@pytest.mark.parametrize(
    ("oriented", "expected"),
    [(0.2, "expected"), (-0.2, "opposite"), (np.nan, "unclassified")],
)
def test_orientation_labels(oriented: float, expected: str) -> None:
    assert orientation_label(oriented) == expected


def _attempts(seed: int = 3) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for player in range(40):
        ability = generator.normal()
        for attempt in range(2):
            rows.append(
                {"nfl_id": str(player), "drill": "FORTY_YARD_DASH", "attempt": str(attempt), "metric": "top_speed",
                 "value": ability + generator.normal(scale=0.1)}
            )
            rows.append(
                {"nfl_id": str(player), "drill": "FORTY_YARD_DASH", "attempt": str(attempt), "metric": "path_length",
                 "value": float(player) * (1 - 2 * attempt)}
            )
    return pd.DataFrame(rows)


def test_sensor_selection_keeps_only_repeatable_metrics() -> None:
    screen, selected = sensor_selection(_attempts())
    assert selected == ["forty_yard_dash_top_speed"]
    assert set(screen["status"]) == {"pass", "fail"}


def test_sensor_player_table_has_one_row_per_player_with_attempt_means() -> None:
    attempts = _attempts()
    table = sensor_player_table(attempts, ["forty_yard_dash_top_speed"])
    assert table["nfl_id"].is_unique
    assert len(table) == 40
    expected = attempts.loc[attempts["metric"].eq("top_speed") & attempts["nfl_id"].eq("0"), "value"].mean()
    assert table.loc[table["nfl_id"].eq("0"), "forty_yard_dash_top_speed"].iloc[0] == pytest.approx(expected)


def test_pair_repeatability_requires_enough_players() -> None:
    few = _attempts().query("nfl_id in ['0','1','2']").query("metric == 'top_speed'")
    assert _pair_repeatability(few)[0] == 3
    assert np.isnan(_pair_repeatability(few)[1])


@pytest.mark.parametrize(
    ("column", "sign"),
    [
        ("WR_SPEED", 1),
        ("forty", -1),
        ("vertical", 1),
        ("forty_yard_dash_sensor_ten_yd_time", -1),
        ("forty_yard_dash_top_speed", 1),
        ("combine_height", 0),
        ("forty_yard_dash_peak_deceleration", 0),
    ],
)
def test_predictor_orientation(column: str, sign: int) -> None:
    assert predictor_orientation(column) == sign


def _players() -> pd.DataFrame:
    generator = np.random.default_rng(5)
    size = 90
    speed = generator.normal(size=size)
    return pd.DataFrame(
        {
            "nfl_id": [str(index) for index in range(size)],
            "draft_year": np.tile([2023, 2024, 2025], size // 3),
            "combine_weight": generator.normal(200, 15, size),
            "combine_height": generator.normal(74, 2, size),
            "role_wr": generator.integers(0, 2, size).astype(float),
            "forty_yard_dash_top_speed": speed,
            "WR_SPEED": speed + generator.normal(scale=0.3, size=size),
            "separation_at_pass_forward": speed + generator.normal(size=size),
        }
    )


def test_association_rows_detect_planted_signal_with_expected_direction() -> None:
    results, height = _association_rows(
        _players(), ["forty_yard_dash_top_speed", "WR_SPEED", "combine_weight"], ["separation_at_pass_forward"]
    )
    assert list(results.columns) == list(RESULT_COLUMNS)
    speed = results.loc[results["predictor"].eq("forty_yard_dash_top_speed")].iloc[0]
    assert speed["r_adj"] > 0.4 and speed["direction"] == "expected"
    assert results.loc[results["predictor"].eq("WR_SPEED"), "family"].iloc[0] == "composite"
    assert results.loc[results["predictor"].eq("combine_weight"), "direction"].iloc[0] == "unclassified"
    assert (results["q_scope"] >= results["p"]).all()
    assert len(height) == len(results)


def test_context_formula_drops_constant_terms() -> None:
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "down": [1, 2, 3], "cushion": [5.0, 5.0, 5.0]})
    formula, numeric, categorical = _context_formula("y", data)
    assert numeric == []
    assert categorical == ["down"]
    assert "cushion" not in formula


def test_outcome_definitions_split_man_and_zone_and_catches() -> None:
    data = pd.DataFrame(
        {
            "target": [True, True, False, True],
            "rec_yards": [10.0, np.nan, np.nan, 5.0],
            "yards_after_catch": [3.0, np.nan, np.nan, 1.0],
            "expected_yards_after_catch": [2.0, np.nan, np.nan, 2.0],
            "expected_points_added": [0.5, -0.5, 0.0, 0.2],
            "separation_at_pass_forward": [1.0, 2.0, 3.0, 4.0],
            "team_coverage_man_zone": ["MAN_COVERAGE", "ZONE_COVERAGE", "ZONE_COVERAGE", "MAN_COVERAGE"],
        }
    )
    definitions = {name: frame for name, frame, _ in _outcome_definitions(data)}
    assert definitions["catch_rate"]["catch_rate"].tolist() == [1.0, 0.0, 1.0]
    assert definitions["yac_over_expected"]["yac_over_expected"].tolist() == [1.0, -1.0]
    assert len(definitions["separation_man"]) == 2 and len(definitions["separation_zone"]) == 2


def _synthetic_root(root: Path) -> Path:
    generator = np.random.default_rng(11)
    players = 60
    ability = generator.normal(size=players)
    routes = []
    for player in range(players):
        for _ in range(90):
            targeted = generator.random() < 0.2
            routes.append(
                {
                    "nfl_id": player,
                    "nfl_position": "WR" if player % 3 else "TE",
                    "draft_year": 2023 + player % 3,
                    "route_ran": "GO" if generator.random() < 0.5 else "OUT",
                    "play_nullified_by_penalty": "N",
                    "target": targeted,
                    "rec_yards": float(generator.integers(0, 20)) if targeted and generator.random() < 0.6 else np.nan,
                    "yards_after_catch": float(generator.integers(0, 8)),
                    "expected_yards_after_catch": 3.0,
                    "expected_points_added": generator.normal(),
                    "separation_at_pass_forward": 2.0 + 1.5 * ability[player] + generator.normal(),
                    "team_coverage_man_zone": "MAN_COVERAGE" if generator.random() < 0.4 else "ZONE_COVERAGE",
                    "team_coverage_type": "COVER_3",
                    "cushion": generator.normal(5, 1),
                    "down": int(generator.integers(1, 4)),
                    "yards_to_go": float(generator.integers(1, 12)),
                    "offense_formation": "SHOTGUN",
                    "pass_length": float(generator.integers(0, 3)),
                }
            )
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    (root / "reports").mkdir()
    pd.DataFrame(routes).to_parquet(processed / "player_play_joined.parquet", index=False)
    attempts = _attempts()
    attempts = attempts.loc[attempts["nfl_id"].astype(int) < players]
    attempts.to_parquet(processed / "scope_wrte_sensor_attempts.parquet", index=False)
    archive = pd.DataFrame(
        {
            "nfl_id": [str(player) for player in range(players)],
            "combine_position": ["WR" if player % 3 else "TE" for player in range(players)],
            "draft_year": [2023 + player % 3 for player in range(players)],
            "combine_weight": generator.normal(200, 15, players),
            "combine_height": generator.normal(74, 2, players),
            "forty": 4.5 - 0.1 * ability + generator.normal(scale=0.05, size=players),
            "vertical": generator.normal(35, 3, players),
            "broad_jump": generator.normal(120, 8, players),
        }
    )
    source = root / "archive.parquet"
    archive.to_parquet(source, index=False)
    return source


def test_staged_pipeline_gates_outcomes_and_writes_results(tmp_path: Path) -> None:
    source = _synthetic_root(tmp_path)
    for outcome in ("separation_at_pass_forward", "target_rate"):
        stage_outcome(tmp_path, outcome)
    with pytest.raises(FileNotFoundError):
        combine_stages(tmp_path, source)
    for outcome in OUTCOMES:
        if outcome not in ("separation_at_pass_forward", "target_rate"):
            stage_outcome(tmp_path, outcome)
    results, gate = combine_stages(tmp_path, source)
    adequate = gate.loc[gate["verdict"].eq("adequate"), "outcome"].tolist()
    assert "separation_at_pass_forward" in adequate
    assert "target_rate" not in adequate
    assert set(results["outcome"]) <= set(adequate)
    forty = results.loc[
        results["outcome"].eq("separation_at_pass_forward") & results["predictor"].eq("forty")
    ].iloc[0]
    assert forty["r_adj"] < -0.3 and forty["direction"] == "expected"
    assert (tmp_path / "reports/scope_wrte_results.csv").exists()


def test_stage_outcome_rejects_unknown_outcome(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        stage_outcome(tmp_path, "nonsense")
