from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol import scope_value
from combine_dlol.scope_value import (
    bh_fdr,
    join_value_sources,
    oriented_sign,
    partial_spearman,
    sensor_composite,
)


def test_partial_spearman_removes_a_shared_control() -> None:
    control = np.arange(80, dtype=float)
    generator = np.random.default_rng(7)
    predictor = control + generator.normal(0.0, 3.0, len(control))
    outcome = control + generator.normal(0.0, 3.0, len(control))
    raw, adjusted = partial_spearman(predictor, outcome, control[:, None])
    assert raw > 0.9
    assert abs(adjusted) < 0.2


def test_bh_fdr_is_monotone_and_bounded() -> None:
    result = bh_fdr(np.array([0.01, 0.04, 0.03, 0.9]))
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert result[0] <= result[2] <= result[1]


def test_join_value_sources_normalizes_identifier_types() -> None:
    combine = pd.DataFrame({"nfl_id": ["10"], "combine_position": ["DL"], "combine_weight": [250.0]})
    players = pd.DataFrame({"nfl_id": [10], "draft_overall_pick": [12], "nfl_position": ["DE"]})
    career = pd.DataFrame({"nfl_id": [10], "career_games_active": [20]})
    joined = join_value_sources(combine, players, career)
    assert joined.loc[0, "draft_overall_pick"] == 12
    assert joined.loc[0, "career_games_active"] == 20


def test_orientation_and_composite_flip_times_only() -> None:
    assert oriented_sign("forty") == -1
    assert oriented_sign("pass_rush_peak_acceleration") == 1
    assert oriented_sign("path_efficiency") == 0
    values = pd.DataFrame({"forty": [4.4, 4.6], "burst_speed": [8.0, 7.0]})
    composite = sensor_composite(values, ("forty", "burst_speed"))
    assert composite.iloc[0] > composite.iloc[1]


def _synthetic_frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    weight = generator.normal(300.0, 20.0, n)
    forty = generator.normal(5.0, 0.2, n)
    pick = np.exp(generator.normal(4.0, 0.8, n))
    started = np.clip(0.5 - 0.5 * (forty - 5.0) + generator.normal(0.0, 0.05, n), 0.0, 1.0)
    return pd.DataFrame(
        {
            "nfl_id": [str(i) for i in range(n)],
            "display_name": [f"Player {i}" for i in range(n)],
            "combine_position": ["DL"] * n,
            "nfl_position": (["DE", "DT", "OLB"] * n)[:n],
            "draft_year": ([2023, 2024, 2025] * n)[:n],
            "draft_overall_pick": pick,
            "log_draft_pick": np.log(pick),
            "combine_weight": weight,
            "forty": forty,
            "games_started_share": started,
            "role": (["EDGE", "IDL"] * n)[:n],
        }
    )


def test_partial_spearman_recovers_planted_correlation_after_control() -> None:
    generator = np.random.default_rng(11)
    control = generator.normal(size=400)
    shared = generator.normal(size=400)
    predictor = shared + control
    outcome = 0.9 * shared + control
    raw, adjusted = partial_spearman(predictor, outcome, control[:, None])
    assert adjusted > 0.55
    assert adjusted > raw - 0.5
    assert abs(adjusted - np.corrcoef(shared, 0.9 * shared)[0, 1]) < 0.35


def test_bh_fdr_matches_statsmodels_and_handles_empty() -> None:
    from statsmodels.stats.multitest import multipletests

    p_values = np.array([0.001, 0.2, 0.03, 0.04, 0.5, 0.0005, 0.9])
    expected = multipletests(p_values, method="fdr_bh")[1]
    np.testing.assert_allclose(bh_fdr(p_values), expected)
    assert bh_fdr(np.array([])).size == 0


def test_oriented_sign_classifies_times_and_capacities() -> None:
    assert oriented_sign("Three_Cone") == -1
    assert oriented_sign("ten_yd_split") == -1
    assert oriented_sign("vertical") == 1
    assert oriented_sign("arm_length") == 1
    assert oriented_sign("mystery") == 0


def test_join_value_sources_keeps_unmatched_combine_rows() -> None:
    combine = pd.DataFrame({"nfl_id": [1, 2], "combine_position": ["DL", "OL"]})
    players = pd.DataFrame({"nfl_id": ["1"], "draft_overall_pick": [5], "ignored": [0]})
    career = pd.DataFrame({"nfl_id": ["1"], "career_games_active": [9]})
    joined = join_value_sources(combine, players, career)
    assert len(joined) == 2
    assert "ignored" not in joined
    assert pd.isna(joined.loc[1, "draft_overall_pick"])


def test_sensor_composite_is_mean_of_oriented_z_scores() -> None:
    frame = pd.DataFrame({"forty": [4.0, 5.0, 6.0], "vertical": [40.0, 35.0, 30.0], "flat": [1.0, 1.0, 1.0]})
    composite = sensor_composite(frame, ("forty", "vertical"))
    np.testing.assert_allclose(composite.to_numpy(), [1.2247449, 0.0, -1.2247449])
    skipped = sensor_composite(frame, ("forty", "flat"))
    np.testing.assert_allclose(skipped.to_numpy(), [1.2247449, 0.0, -1.2247449])


def test_role_splits_edge_interior_and_tackle() -> None:
    frame = pd.DataFrame(
        {
            "combine_position": ["DL", "DL", "OL", "OL", "WR", "DL"],
            "nfl_position": ["DE", "DT", "T", "G", "WR", None],
        }
    )
    assert list(scope_value._role(frame)) == ["EDGE", "IDL", "T", "IOL", "WR", "IDL"]


def test_controls_impute_median_and_append_role_dummies() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "role": ["x", "y", "x"]})
    matrix = scope_value._controls(frame, ("a", "missing"))
    assert matrix.shape == (3, 3)
    assert matrix[1, 0] == 2.0
    assert list(matrix[:, 1]) == [1.0, 0.0, 1.0]


def test_ci_p_value_bounds_and_degenerate_cases() -> None:
    low, high, p_value = scope_value._ci_p_value(0.5, 100, 2)
    assert low < 0.5 < high
    assert 0.0 < p_value < 0.001
    assert all(np.isnan(v) for v in scope_value._ci_p_value(1.0, 100, 2))
    assert all(np.isnan(v) for v in scope_value._ci_p_value(0.3, 4, 2))


def test_replication_needs_fifteen_rows_per_window() -> None:
    frame = _synthetic_frame(60)
    disc, hold, y23, y24, y25 = scope_value._replication(frame, "forty", "games_started_share")
    assert disc < -0.5 and hold < -0.5 and y23 < -0.5 and y24 < -0.5 and y25 < -0.5
    sparse = frame.loc[frame["draft_year"].eq(2023)].head(10)
    assert np.isnan(scope_value._replication(sparse, "forty", "games_started_share")[0])


def test_association_rows_label_direction_and_skip_thin_data() -> None:
    frame = _synthetic_frame()
    rows = scope_value._association_rows(
        frame, "DL", "games_started_share", ("forty", "combine_weight"), "career", ("combine_weight",), 1
    )
    by_predictor = {row["predictor"]: row for row in rows}
    assert by_predictor["forty"]["direction"] == "expected"
    assert by_predictor["forty"]["oriented_r"] > 0.5
    assert by_predictor["combine_weight"]["direction"] == "unclassified"
    opposite = scope_value._association_rows(frame, "DL", "games_started_share", ("forty",), "career", (), -1)
    assert opposite[0]["direction"] == "opposite"
    thin = scope_value._association_rows(frame.head(30), "DL", "games_started_share", ("forty",), "career", (), 1)
    assert thin == []


def test_association_rows_mark_unclassified_predictors() -> None:
    frame = _synthetic_frame().assign(mystery=lambda f: f["forty"] * 2.0)
    rows = scope_value._association_rows(frame, "DL", "games_started_share", ("mystery",), "career", (), 1)
    assert rows[0]["direction"] == "unclassified"
    assert np.isnan(rows[0]["oriented_r"])


def test_draft_pricing_and_market_gap_use_different_controls() -> None:
    frame = _synthetic_frame()
    pricing = scope_value._association_rows(
        frame, "DL", "log_draft_pick", ("forty",), "draft_pricing", ("forty", "combine_weight"), -1
    )
    gap = scope_value._association_rows(
        frame, "DL", "games_started_share", ("forty",), "market_gap", ("combine_weight", "log_draft_pick"), 1
    )
    assert abs(pricing[0]["r_adj"]) < 0.5
    assert gap[0]["family"] == "market_gap"
    assert gap[0]["direction"] == "expected"


def test_official_for_group_requires_forty_non_missing_values() -> None:
    frame = pd.DataFrame({"forty": np.arange(50.0), "vertical": [1.0] * 10 + [np.nan] * 40})
    assert scope_value._official_for_group(frame) == ("forty",)


def test_career_predictors_dedupe_and_add_composite() -> None:
    frame = pd.DataFrame({"forty": np.arange(50.0)})
    assert scope_value._career_predictors(frame, ("forty", "x")) == ("forty", "x", "sensor_composite")
    assert scope_value._career_predictors(frame, ()) == ("forty",)


def test_cv_rows_report_improvement_when_official_signal_exists() -> None:
    frame = _synthetic_frame().assign(
        snaps_per_active_game=lambda f: f["games_started_share"] * 50.0,
        ever_all_pro_or_pro_bowl=lambda f: (f["games_started_share"] > 0.5).astype(float),
    )
    rows = scope_value._cv_rows(frame, "DL", ("forty",), ())
    games = [r for r in rows if r["outcome"] == "games_started_share" and r["comparison"].startswith("official")]
    assert len(rows) == 6
    assert games[0]["delta_r2"] > 0.2
    assert games[0]["ci_lo"] <= games[0]["delta_r2"] <= games[0]["ci_hi"]
    assert scope_value._cv_rows(frame.head(10), "DL", ("forty",), ()) == []


def test_player_lists_rank_over_and_under_performers() -> None:
    base = _synthetic_frame(60).assign(forty_top_speed=1.0)
    outcomes = pd.DataFrame(
        {
            "nfl_id": base["nfl_id"],
            "shrunk_d_a": np.linspace(-1.0, 1.0, len(base)),
            "pressure_rate": np.linspace(1.0, -1.0, len(base)),
        }
    )
    lists = scope_value._player_lists(base, {"DL": outcomes, "OL": outcomes.iloc[0:0]})
    dl = lists.loc[lists["combine_position"].eq("DL")]
    assert len(dl) == 20
    assert list(dl["descriptive_group"].unique()) == ["falls_short", "exceeds"]
    short = dl.loc[dl["descriptive_group"].eq("falls_short"), "beyond_draft_residual"]
    exceeds = dl.loc[dl["descriptive_group"].eq("exceeds"), "beyond_draft_residual"]
    assert short.max() < exceeds.min()
    assert "forty_top_speed" in lists


def _write_synthetic_root(root: Path, home: Path) -> None:
    generator = np.random.default_rng(5)
    (root / "data/raw").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    (root / "reports").mkdir()
    (home / "dev/NFL-big-data-archive/processed").mkdir(parents=True)
    layout = [("DL", 0, 70, "DE"), ("OL", 100, 70, "T"), ("WR", 200, 40, "WR")]
    results, players, career = [], [], []
    for position, start, count, nfl_position in layout:
        for offset in range(count):
            nfl_id = start + offset
            quickness = generator.normal()
            pick = int(np.clip(60 - 15 * quickness + generator.normal(0, 10), 1, 250))
            results.append(
                {
                    "nfl_id": nfl_id,
                    "combine_position": position,
                    "forty": 5.0 - 0.1 * quickness + generator.normal(0, 0.02),
                    "vertical": 30.0 + 2.0 * quickness + generator.normal(0, 1.0),
                    "combine_weight": generator.normal(280.0, 25.0),
                }
            )
            players.append(
                {
                    "nfl_id": str(nfl_id),
                    "display_name": f"P{nfl_id}",
                    "nfl_position": nfl_position,
                    "draft_year": 2023 + offset % 3,
                    "draft_overall_pick": pick,
                    "draft_round": 1,
                }
            )
            games = 0 if offset % 7 == 0 else 40
            career.append(
                {
                    "nfl_id": nfl_id,
                    "career_games_active": games,
                    "career_games_started": games * min(1.0, max(0.0, 0.5 + 0.2 * quickness)),
                    "career_offensive_snaps": games * 30.0,
                    "career_defensive_snaps": 0.0,
                    "pro_bowl_original_ballot": int(quickness > 1.0),
                }
            )
    pd.DataFrame(results).to_csv(root / "data/raw/combine_results.csv", index=False)
    pd.DataFrame(players).to_csv(root / "data/raw/players.csv", index=False)
    pd.DataFrame(career).to_csv(root / "data/raw/player_career_successes.csv", index=False)
    line_ids = [row["nfl_id"] for row in results if row["combine_position"] in ("DL", "OL")]
    pd.DataFrame(
        {
            "nfl_id": line_ids,
            "group": ["DL" if i < 100 else "OL" for i in line_ids],
            "burst_speed_1_0s": generator.normal(size=len(line_ids)),
            "lateral_peak_acceleration": generator.normal(size=len(line_ids)),
        }
    ).to_parquet(root / "data/processed/combine_predictors.parquet")
    pd.DataFrame(
        [
            {"group": group, "family": "sensor", "status": "pass", "predictor_column": column}
            for group in ("DL", "OL")
            for column in ("burst_speed_1_0s", "lateral_peak_acceleration")
        ]
    ).to_csv(root / "reports/phase2_combine_predictors.csv", index=False)
    receivers = [row["nfl_id"] for row in results if row["combine_position"] == "WR"]
    pd.DataFrame(
        {
            "nfl_id": [str(i) for i in receivers],
            "combine_position": "WR",
            "dash_top_speed": generator.normal(size=len(receivers)),
            "dash_peak_acceleration": generator.normal(size=len(receivers)),
        }
    ).to_parquet(home / "dev/NFL-big-data-archive/processed/combine_features.parquet")
    skill = np.repeat(generator.normal(size=len(receivers)), 4)
    pd.DataFrame(
        {
            "nfl_id": np.repeat([str(i) for i in receivers], 4),
            "drill_name": "DASH",
            "attempt": np.tile(np.arange(4), len(receivers)),
            "top_speed": skill + generator.normal(0, 0.1, len(skill)),
            "peak_acceleration": generator.normal(size=len(skill)),
        }
    ).to_parquet(root / "data/processed/scope_value_sensor_attempts.parquet")
    pd.DataFrame(
        {
            "nfl_id": [str(i) for i in range(70)],
            **{
                name: generator.normal(size=70)
                for name in (
                    "shrunk_d_a",
                    "shrunk_d_b_displacement_0_5",
                    "shrunk_d_b_displacement_1_0",
                    "shrunk_d_b_speed_0_5",
                )
            },
        }
    ).to_parquet(root / "data/processed/dl_outcomes_scores.parquet")
    pd.DataFrame(
        [
            {"nfl_id": i, "outcome": outcome, "posterior_mean": generator.normal()}
            for i in range(100, 170)
            for outcome in ("pressure_allowed", "first_0_5s_displacement")
        ]
    ).to_parquet(root / "data/processed/ol_outcomes_scores.parquet")


def test_run_builds_all_four_tables_from_synthetic_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    root = tmp_path / "root"
    _write_synthetic_root(root, home)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    results, gate, models, lists = scope_value.run(root)
    assert set(results["family"]) >= {"career", "market_gap", "draft_pricing", "perf_beyond_draft"}
    assert results["q_scope"].between(0.0, 1.0).all()
    assert {"DL", "OL"} <= set(gate.loc[gate["verdict"].eq("adequate_existing_gate"), "group"])
    assert set(models["comparison"]) == {"official_minus_draft_pick", "sensor_minus_draft_pick"}
    assert len(lists) == 40
    repeatability = pd.read_csv(root / "reports/scope_value_repeatability.csv")
    assert repeatability.loc[repeatability["metric"].eq("top_speed"), "status"].iloc[0] == "pass"
    assert repeatability.loc[repeatability["metric"].eq("peak_acceleration"), "status"].iloc[0] == "unreliable"
    assert (root / "reports/scope_value_results.csv").exists()


def test_career_rates_are_missing_for_players_without_active_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    root = tmp_path / "root"
    _write_synthetic_root(root, home)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    base, sensors = scope_value._base_frame(root)
    idle = base.loc[base["career_games_active"].eq(0)]
    assert len(idle) > 0
    assert idle["games_started_share"].isna().all()
    assert idle["snaps_per_active_game"].isna().all()
    active = base.loc[base["career_games_active"].gt(0)]
    assert active["games_started_share"].between(0.0, 1.0).all()
    assert (active["snaps_per_active_game"] == 30.0).all()
    assert base.loc[base["pro_bowl_original_ballot"].gt(0), "ever_all_pro_or_pro_bowl"].eq(1.0).all()
    assert sensors["WR"] == ("dash_top_speed",)
    assert base["sensor_composite"].notna().any()


def test_main_runs_with_root_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    root = tmp_path / "root"
    _write_synthetic_root(root, home)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert scope_value.main(["--root", str(root)]) == 0
    assert (root / "reports/scope_value_player_list.csv").exists()
