from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol.scope_wrte_drills import (
    DRILL_MAP,
    MATCHED,
    RESULT_COLUMNS,
    ROUTE_FAMILIES,
    association_rows,
    break_metrics,
    build_attempts,
    build_tests,
    main,
    merged_drill,
    outcome_definitions,
    outcome_name,
    player_drill_table,
    predictor_name,
    repeatability_gate,
    run_scope,
    score_outcomes,
    summarize,
)


def _l_path(speed: float, seconds: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(0.0, seconds, 0.1)
    distance = speed * time
    half = len(time) // 2
    x = np.where(np.arange(len(time)) <= half, distance, distance[half])
    y = np.where(np.arange(len(time)) <= half, 0.0, distance - distance[half])
    return time, x, y


def test_merged_drill_collapses_name_variants() -> None:
    assert merged_drill("SPEED_OUT_LEFT") == merged_drill("SPEED_OUT_ROUTE_LEFT") == "SPEED_OUT"
    assert merged_drill("OVER_THE_SHOULDER_ADJUST") == "OVER_SHOULDER_ADJUST"
    assert merged_drill("POST_CORNER_ROUTE_LEFT") == merged_drill("POST_CORNER_ROUTE_RIGHT") == "POST_CORNER"
    assert merged_drill("FORTY_YARD_DASH") is None
    assert len(set(DRILL_MAP.values())) == 19


def test_every_matched_drill_is_a_known_drill_and_family() -> None:
    assert set(MATCHED) <= set(DRILL_MAP.values())
    assert set(MATCHED.values()) <= set(ROUTE_FAMILIES)


def test_break_metrics_detect_a_ninety_degree_turn() -> None:
    time, x, y = _l_path(8.0)
    values = break_metrics(time, x, y)
    assert values["break_speed"] == pytest.approx(7.0, abs=1.5)
    assert 0.5 < values["break_speed_retention"] <= 1.0


def test_break_metrics_scale_with_speed() -> None:
    slow = break_metrics(*_l_path(5.0))
    fast = break_metrics(*_l_path(9.0))
    assert fast["break_speed"] > slow["break_speed"]


def test_break_metrics_are_missing_for_a_straight_line_and_short_input() -> None:
    time = np.arange(0.0, 3.0, 0.1)
    straight = break_metrics(time, 8.0 * time, np.zeros_like(time))
    assert np.isnan(straight["break_speed"]) and np.isnan(straight["break_speed_retention"])
    assert np.isnan(break_metrics([0.0, 0.1], [0.0, 1.0], [0.0, 0.0])["break_speed"])
    assert np.isnan(break_metrics(time[:8], 8.0 * time[:8], np.zeros(8))["break_speed"])


def test_break_metrics_ignore_a_turn_while_nearly_stationary() -> None:
    time = np.arange(0.0, 3.0, 0.1)
    x = np.concatenate([np.linspace(0, 6, 15), np.full(15, 6.0)])
    y = np.concatenate([np.zeros(15), np.linspace(0, 6, 15)])
    assert not np.isnan(break_metrics(time, x, y)["break_speed"])
    jitter_x = np.cumsum(np.full(30, 0.01))
    assert np.isnan(break_metrics(time, jitter_x, np.zeros(30))["break_speed"])


def _attempt_table(multi: int, noise: float, seed: int = 1, drill: str = "SLANT") -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    rows = []
    for player in range(60):
        base = generator.normal()
        count = 2 if player < multi else 1
        for attempt in range(count):
            rows.append(
                {
                    "nfl_id": str(player),
                    "drill": drill,
                    "raw_drill": "SLANT_ROUTE_LEFT",
                    "attempt": f"a|{attempt}",
                    "metric": "top_speed",
                    "value": base + generator.normal(scale=noise),
                }
            )
    return pd.DataFrame(rows)


def test_repeatability_gate_assigns_three_statuses() -> None:
    passing = repeatability_gate(_attempt_table(multi=50, noise=0.2))
    assert passing["status"].tolist() == ["pass"]
    unverifiable = repeatability_gate(_attempt_table(multi=10, noise=0.2))
    assert unverifiable["status"].tolist() == ["unverifiable"]
    noisy = repeatability_gate(_attempt_table(multi=50, noise=50.0, seed=3))
    assert noisy["status"].tolist() == ["unrepeatable"]
    assert passing.loc[0, "predictor"] == "slant_top_speed"


def test_player_drill_table_keeps_only_usable_predictors_and_averages_attempts() -> None:
    attempts = pd.concat(
        [_attempt_table(50, 0.2), _attempt_table(50, 50.0, seed=3, drill="CURL")], ignore_index=True
    )
    gate = repeatability_gate(attempts)
    table = player_drill_table(attempts, gate)
    assert "slant_top_speed" in table and "curl_top_speed" not in table
    first = attempts.loc[(attempts["drill"] == "SLANT") & (attempts["nfl_id"] == "0"), "value"].mean()
    assert table.loc[table["nfl_id"] == "0", "slant_top_speed"].iloc[0] == pytest.approx(first)


def test_player_drill_table_is_empty_when_nothing_is_usable() -> None:
    attempts = _attempt_table(50, 50.0, seed=3)
    table = player_drill_table(attempts, repeatability_gate(attempts))
    assert table.empty


def test_build_tests_separates_matched_unmatched_and_tiers() -> None:
    gate = pd.DataFrame(
        {
            "drill": ["SLANT", "CURL", "CIRCUS"],
            "metric": ["top_speed", "top_speed", "top_speed"],
            "predictor": ["slant_top_speed", "curl_top_speed", "circus_top_speed"],
            "status": ["pass", "unverifiable", "unrepeatable"],
        }
    )
    tests = build_tests(gate, [outcome_name("sep", "INSIDE"), outcome_name("ypr", "GO")])
    labels = {(t["predictor"], t["outcome"]): t["family_label"] for t in tests}
    assert labels[("slant_top_speed", "sep_INSIDE")] == "headline_matched"
    assert labels[("slant_top_speed", "ypr_GO")] == "headline_unmatched"
    assert labels[("curl_top_speed", "sep_INSIDE")] == "unverifiable_unmatched"
    assert all(t["predictor"] != "circus_top_speed" for t in tests)


def _association_frame(n: int = 120, effect: float = 1.0, seed: int = 5) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    speed = generator.normal(size=n)
    return pd.DataFrame(
        {
            "nfl_id": [str(i) for i in range(n)],
            "slant_top_speed": speed,
            "slant_time_in_drill": -speed + generator.normal(scale=0.1, size=n),
            "slant_path_length": speed,
            "noise": generator.normal(size=n),
            "sep_INSIDE": effect * speed + generator.normal(size=n),
            "combine_weight": generator.normal(210, 20, size=n),
            "role_wr": (np.arange(n) % 4 != 0).astype(float),
            "draft_year": np.tile([2023, 2024, 2025], n // 3 + 1)[:n],
        }
    )


def _test(predictor: str, metric: str, label: str, outcome: str = "sep_INSIDE") -> dict[str, str]:
    return {"predictor": predictor, "metric": metric, "outcome": outcome, "family": "matched", "family_label": label}


def test_association_recovers_a_planted_effect_with_expected_direction() -> None:
    rows, skipped = association_rows(
        _association_frame(), [_test("slant_top_speed", "top_speed", "headline_matched")]
    )
    assert skipped == 0
    assert list(rows.columns) == list(RESULT_COLUMNS)
    row = rows.iloc[0]
    assert row["r_adj"] > 0.5 and row["p"] < 0.001 and row["q_scope"] < 0.01
    assert row["direction"] == "expected" and row["oriented_r"] == pytest.approx(row["r_adj"])
    assert row["ci_lo"] < row["r_adj"] < row["ci_hi"]
    assert row["r_hold_2025"] > 0.0 and row["n"] == 120


def test_lower_is_better_metric_is_oriented_so_faster_is_positive() -> None:
    rows, _ = association_rows(
        _association_frame(), [_test("slant_time_in_drill", "time_in_drill", "headline_matched")]
    )
    row = rows.iloc[0]
    assert row["r_adj"] < 0.0 and row["oriented_r"] > 0.0 and row["direction"] == "expected"


def test_unoriented_metric_is_unclassified() -> None:
    rows, _ = association_rows(_association_frame(), [_test("slant_path_length", "path_length", "headline_matched")])
    assert rows.iloc[0]["direction"] == "unclassified" and np.isnan(rows.iloc[0]["oriented_r"])


def test_opposite_direction_is_labelled() -> None:
    rows, _ = association_rows(
        _association_frame(effect=-1.0), [_test("slant_top_speed", "top_speed", "headline_matched")]
    )
    assert rows.iloc[0]["direction"] == "opposite"


def test_association_skips_small_samples_and_missing_columns() -> None:
    small = _association_frame(n=30)
    rows, skipped = association_rows(
        small,
        [_test("slant_top_speed", "top_speed", "headline_matched"), _test("absent", "top_speed", "headline_matched")],
    )
    assert rows.empty and skipped == 2
    assert list(rows.columns) == list(RESULT_COLUMNS)


def test_association_skips_constant_predictor() -> None:
    frame = _association_frame().assign(slant_top_speed=1.0)
    rows, skipped = association_rows(frame, [_test("slant_top_speed", "top_speed", "headline_matched")])
    assert rows.empty and skipped == 1


def test_role_control_is_dropped_when_role_is_constant() -> None:
    frame = _association_frame().assign(role_wr=1.0)
    rows, _ = association_rows(frame, [_test("slant_top_speed", "top_speed", "headline_matched")])
    assert np.isfinite(rows.iloc[0]["r_adj"])


def test_bh_is_applied_within_each_family_label_separately() -> None:
    frame = _association_frame()
    tests = [
        _test("slant_top_speed", "top_speed", "headline_matched"),
        _test("noise", "top_speed", "unverifiable_matched"),
        _test("slant_top_speed", "top_speed", "unverifiable_matched"),
    ]
    rows, _ = association_rows(frame, tests)
    alone = rows.loc[rows["family_label"] == "headline_matched", "q_scope"].iloc[0]
    assert alone == pytest.approx(rows.loc[rows["family_label"] == "headline_matched", "p"].iloc[0])
    pair = rows.loc[rows["family_label"] == "unverifiable_matched"]
    assert pair["q_scope"].max() == pytest.approx(pair["p"].max())


def test_summarize_counts_direction_and_significance() -> None:
    frame = _association_frame()
    rows, _ = association_rows(frame, [_test("slant_top_speed", "top_speed", "headline_matched")])
    summary = summarize(rows)
    assert summary.loc[0, "q_lt_0_10"] == 1 and summary.loc[0, "expected"] == 1
    assert summarize(rows.iloc[0:0]).empty


def _route_data(seed: int = 2, players: int = 90, routes: int = 220) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    types = np.array(["GO", "HITCH", "SLANT", "IN", "CROSS", "OUT", "FLAT", "POST", "CORNER"])
    weights = np.array([0.4, 0.2, 0.08, 0.08, 0.06, 0.05, 0.05, 0.04, 0.04])
    frames = []
    for player in range(players):
        skill = generator.normal()
        frames.append(
            pd.DataFrame(
                {
                    "nfl_id": str(player),
                    "nfl_position": "WR" if player % 4 else "TE",
                    "route_ran": generator.choice(types, size=routes, p=weights),
                    "play_nullified_by_penalty": "N",
                    "target": generator.random(routes) < 0.15 + 0.03 * skill,
                    "rec_yards": generator.normal(2.0 + skill, 3.0, size=routes),
                    "separation_at_pass_forward": 3.0 + skill + generator.normal(size=routes),
                    "team_coverage_man_zone": generator.choice(["MAN_COVERAGE", "ZONE_COVERAGE"], size=routes),
                    "yards_to_go": generator.integers(1, 12, size=routes).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_outcome_definitions_cover_five_families_and_three_measures() -> None:
    definitions = outcome_definitions(_route_data(players=5, routes=60))
    names = [name for name, _ in definitions]
    assert len(names) == 15 and "sep_INSIDE" in names and "ypr_DEEP_BREAK" in names
    frames = dict(definitions)
    assert set(frames["sep_INSIDE"]["route_ran"]) <= {"SLANT", "IN", "CROSS"}
    assert frames["target_rate_GO"]["target_rate"].isin([0.0, 1.0]).all()


def test_score_outcomes_gates_reliable_player_skill_and_rejects_sparse_data() -> None:
    scores, gate = score_outcomes(_route_data(), n_splits=60, n_bootstrap=60)
    verdicts = dict(zip(gate["outcome"], gate["verdict"], strict=True))
    assert verdicts["sep_GO"] == "adequate"
    assert "sep_GO" in scores and scores["sep_GO"].notna().sum() > 50
    sparse_scores, sparse_gate = score_outcomes(_route_data(players=6, routes=30), n_splits=20, n_bootstrap=20)
    assert (sparse_gate["verdict"] == "poor").all() and sparse_scores.empty


def test_score_outcomes_handles_empty_family_after_missing_context() -> None:
    data = _route_data(players=4, routes=20).assign(separation_at_pass_forward=np.nan)
    _, gate = score_outcomes(data, n_splits=20, n_bootstrap=20)
    assert gate.loc[gate["outcome"] == "sep_GO", "n_players"].iloc[0] == 0


def _write_root(tmp_path: Path, players: int = 90) -> Path:
    generator = np.random.default_rng(11)
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    route_data = _route_data(players=players)
    skills = route_data.groupby("nfl_id")["separation_at_pass_forward"].mean()
    combine = pd.DataFrame(
        {
            "draft_year": np.tile([2023, 2024, 2025], players // 3 + 1)[:players],
            "nfl_id": range(players),
            "combine_position": ["WR" if i % 4 else "TE" for i in range(players)],
            "combine_weight": generator.normal(210, 15, size=players),
        }
    )
    combine.to_csv(tmp_path / "data/raw/combine_results.csv", index=False)
    route_data.to_parquet(tmp_path / "data/processed/player_play_joined.parquet", index=False)
    records = []
    for player in range(players):
        speed = 6.0 + 1.5 * float(skills.loc[str(player)] - skills.mean()) + generator.normal(scale=0.1)
        for raw, attempts in (("SLANT_ROUTE_LEFT", 2), ("GO_ROUTE_RIGHT", 1), ("FORTY_YARD_DASH", 2)):
            for attempt in range(attempts):
                time, x, y = _l_path(speed + generator.normal(scale=0.1))
                stamps = pd.Timestamp("2025-03-01T12:00:00") + pd.to_timedelta(time, unit="s")
                records.append(
                    pd.DataFrame(
                        {
                            "nfl_id": player,
                            "entity_type": "PLAYER",
                            "time": stamps.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                            "drill_name": raw,
                            "attempt": attempt + 1,
                            "x": x,
                            "y": y,
                        }
                    )
                )
    pd.concat(records, ignore_index=True).to_csv(tmp_path / "data/raw/combine_tracking.csv", index=False)
    return tmp_path


def test_build_attempts_filters_drills_and_caches(tmp_path: Path) -> None:
    root = _write_root(tmp_path, players=12)
    attempts = build_attempts(root, {str(i) for i in range(12)})
    assert set(attempts["drill"]) == {"SLANT", "GO"}
    assert "break_speed" in set(attempts.loc[attempts["drill"] == "SLANT", "metric"])
    assert "break_speed" not in set(attempts.loc[attempts["drill"] == "GO", "metric"])
    (root / "data/raw/combine_tracking.csv").unlink()
    assert len(build_attempts(root, set())) == len(attempts)


def test_build_attempts_with_no_matching_drills_is_empty(tmp_path: Path) -> None:
    root = _write_root(tmp_path, players=6)
    assert build_attempts(root, {"nobody"}).empty


def test_run_scope_end_to_end_writes_families_and_finds_planted_speed_link(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    results, gate, repeatability, skipped = run_scope(root, n_splits=60, n_bootstrap=60)
    assert (root / "reports/scope_wrte_drills_results.csv").exists()
    assert (root / "reports/scope_wrte_drills_outcome_gate.csv").exists()
    assert (root / "reports/scope_wrte_drills_repeatability_gate.csv").exists()
    assert list(results.columns) == list(RESULT_COLUMNS)
    status = dict(zip(repeatability["predictor"], repeatability["status"], strict=True))
    assert status["slant_top_speed"] == "pass"
    assert status["go_top_speed"] == "unverifiable"
    assert set(results["family_label"]) <= {
        "headline_matched",
        "headline_unmatched",
        "unverifiable_matched",
        "unverifiable_unmatched",
    }
    planted = results.loc[(results["predictor"] == "slant_top_speed") & (results["outcome"] == "sep_GO")]
    assert not planted.empty and planted.iloc[0]["r_adj"] > 0.3
    assert (gate["verdict"] == "adequate").any() and skipped >= 0


def test_main_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _write_root(tmp_path, players=60)
    assert main(["--root", str(root)]) == 0
    assert "tests run=" in capsys.readouterr().out


def test_predictor_and_outcome_names() -> None:
    assert predictor_name("POST_CORNER", "top_speed") == "post_corner_top_speed"
    assert outcome_name("sep", "GO") == "sep_GO"
