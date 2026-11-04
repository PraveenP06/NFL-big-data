from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol import scope_ngs_agility as module
from combine_dlol.scope_ngs_agility import (
    add_draft_residuals,
    agility_index,
    association_row,
    benjamini_hochberg_by_family,
    drill_attempts,
    extra_kinematics,
    outcome_frame,
    outcome_sign,
    pairing_triples,
    part_a_rows,
    player_drill_table,
    predictor_sign,
    role_columns,
    stable_outcomes,
    summarize,
    tier_table,
)

GROUP_SIZES = {"DL": 60, "OL": 60, "DB": 60, "WR": 50, "TE": 45}
POSITIONS = {"DL": ("DE", "DT"), "OL": ("T", "G"), "DB": ("CB", "S"), "WR": ("WR",), "TE": ("TE",)}


def _roster() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    combine_rows, player_rows = [], []
    next_id = 1000
    for group, size in GROUP_SIZES.items():
        for index in range(size):
            skill = rng.normal()
            position = POSITIONS[group][index % len(POSITIONS[group])]
            combine_rows.append(
                {
                    "nfl_id": next_id,
                    "draft_year": 2023 + index % 3,
                    "combine_position": group,
                    "combine_weight": 200 + 3 * index + rng.normal(0, 5),
                    "ngs_athleticism_score": 60 + 8 * skill + rng.normal(0, 3),
                    "ngs_college_production_score": 60 + rng.normal(0, 8),
                    "ngs_final_score": 60 + 5 * skill + rng.normal(0, 3),
                }
            )
            player_rows.append(
                {"nfl_id": next_id, "nfl_position": position, "draft_overall_pick": 1 + index * 3 + (index % 5)}
            )
            next_id += 1
    return pd.DataFrame(combine_rows), pd.DataFrame(player_rows)


def _trajectory(amplitude: float, samples: int = 40) -> pd.DataFrame:
    seconds = np.arange(samples) * 0.1
    return pd.DataFrame(
        {
            "time": seconds,
            "x": 6.0 * seconds,
            "y": amplitude * np.sin(2.5 * seconds),
        }
    )


def _tracking(players: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    drills = {
        "DL": ["BODY_CONTROL_DRILL", "FOUR_BAG_AGILITY_DRILL", "SHORT_ZONE_BREAK_INSIDE", "SHORT_ZONE_BREAKS"],
        "OL": [
            "SCREEN_DRILL",
            "PASS_PRO-MIRROR_DRILL",
            "OL_PULL_DRILL_FOLD_BLOCK_RIGHT",
            "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE",
        ],
    }
    rows = []
    for group in ("DL", "OL"):
        ids = combine.loc[combine["combine_position"].eq(group), "nfl_id"]
        for nfl_id in ids:
            for drill in drills[group]:
                for attempt in (1, 2) if drill == "SHORT_ZONE_BREAKS" else (1,):
                    frame = _trajectory(abs(rng.normal(2.0, 0.6)), int(rng.integers(34, 46)))
                    rows.append(frame.assign(nfl_id=nfl_id, drill_name=drill, attempt=attempt, entity_type="PLAYER"))
    rows.append(_trajectory(1.0).assign(nfl_id=1000, drill_name="BODY_CONTROL_DRILL", attempt=1, entity_type="BALL"))
    rows.append(_trajectory(1.0).assign(nfl_id=1000, drill_name="UNLISTED", attempt=1, entity_type="PLAYER"))
    return pd.concat(rows, ignore_index=True)


def _outcome(ids: pd.Series, scale: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, scale, len(ids))


def _make_root(tmp_path: Path) -> Path:
    combine, players = _roster()
    raw = tmp_path / "data/raw"
    processed = tmp_path / "data/processed"
    reports = tmp_path / "reports"
    for directory in (raw, processed, reports):
        directory.mkdir(parents=True)
    combine.to_csv(raw / "combine_results.csv", index=False)
    players.to_csv(raw / "players.csv", index=False)
    _tracking(players, combine).to_csv(raw / "combine_tracking.csv", index=False)

    def ids(group: str) -> pd.Series:
        return combine.loc[combine["combine_position"].eq(group), "nfl_id"]

    gate_columns = ["outcome", "group", "n_players", "SB_median", "lower_ci", "upper_ci", "icc", "verdict", "formula"]

    def gate(rows: list[tuple[str, float, float, str]], group: str) -> pd.DataFrame:
        return pd.DataFrame([(o, group, 50, m, lo, 0.9, 0.1, v, "f") for o, m, lo, v in rows], columns=gate_columns)

    dl_static = ["shrunk_d_a", "shrunk_d_b_displacement_0_5", "shrunk_d_b_displacement_1_0", "shrunk_d_b_speed_0_5"]
    gate(
        [(o, 0.9, 0.8, "adequate") for o in dl_static]
        + [("hurry_rate", 0.7, 0.5, "adequate"), ("tfl_rate", 0.35, 0.0, "weak")],
        "DL",
    ).to_csv(reports / "scope_dl_gate.csv", index=False)
    gate(
        [
            ("pass_set_depth_1_0s", 0.9, 0.8, "adequate"),
            ("first_0_5s_displacement", 0.9, 0.8, "adequate"),
            ("set_peak_speed_0_1_5s", 0.9, 0.8, "adequate"),
            ("sack_allowed", 0.2, -0.2, "poor"),
        ],
        "OL",
    ).to_csv(reports / "scope_ol_gate.csv", index=False)
    gate([("coverage_assignment_rate", 0.9, 0.8, "adequate"), ("assist_rate", 0.6, 0.4, "adequate")], "DB").to_csv(
        reports / "scope_db_gate.csv", index=False
    )
    wrte = gate([("target_rate", 0.6, 0.4, "adequate"), ("catch_rate", 0.4, 0.05, "weak")], "WRTE")
    career = gate([("games_started_share", np.nan, np.nan, "career")], "DL")
    pd.concat([wrte, career]).to_csv(reports / "scope_wrte_gate.csv", index=False)
    career.to_csv(reports / "scope_db_gate_career.csv", index=False)

    dl_ids = ids("DL")
    dl = pd.DataFrame({"nfl_id": dl_ids})
    for index, name in enumerate(dl_static):
        dl[name] = _outcome(dl_ids, 1.0, 10 + index)
    dl.to_parquet(processed / "dl_outcomes_scores.parquet", index=False)
    pd.DataFrame({"nfl_id": dl_ids, "outcome": "hurry_rate", "score": _outcome(dl_ids, 0.1, 20)}).to_csv(
        processed / "scope_dl_rate_scores_progress.csv", index=False
    )
    for index, name in enumerate(("pass_set_depth_1_0s", "first_0_5s_displacement", "set_peak_speed_0_1_5s")):
        ol_ids = ids("OL")
        pd.DataFrame({"nfl_id": ol_ids, "posterior_mean": _outcome(ol_ids, 1.0, 30 + index)}).to_parquet(
            processed / f"scope_ol_scores_{name}.parquet", index=False
        )
    db_ids = ids("DB")
    pd.DataFrame(
        {
            "nfl_id": db_ids,
            "coverage_assignment_rate": _outcome(db_ids, 1.0, 40),
            "assist_rate": _outcome(db_ids, 1.0, 41),
        }
    ).to_parquet(processed / "scope_db_analysis.parquet", index=False)
    wrte_ids = pd.concat([ids("WR"), ids("TE")])
    pd.DataFrame({"nfl_id": wrte_ids.astype(str), "target_rate": _outcome(wrte_ids, 1.0, 50)}).to_parquet(
        processed / "scope_wrte_scores_target_rate.parquet", index=False
    )
    return tmp_path


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_root(tmp_path_factory.mktemp("ngs"))


@pytest.fixture(scope="module")
def results(synthetic_root: Path) -> pd.DataFrame:
    return module.run(synthetic_root)


def test_signs_follow_the_existing_scope_conventions() -> None:
    assert predictor_sign("ngs_final_score") == 1
    assert predictor_sign("AGILITY_INDEX") == 1
    assert predictor_sign("body_control_drill_time_in_drill") == -1
    assert predictor_sign("body_control_drill_direction_change_loss") == -1
    assert predictor_sign("body_control_drill_path_length") == 0
    assert predictor_sign("mystery") == 0
    assert outcome_sign("DL", "shrunk_d_a") == -1
    assert outcome_sign("DL", "hurry_rate") == 1
    assert outcome_sign("OL", "snap_reaction_time") == 0
    assert outcome_sign("DB", "coverage_assignment_rate") == 0
    assert outcome_sign("DB", "assist_rate") == 1
    assert outcome_sign("TE", "target_rate") == 1


def test_gate_keeps_only_stable_game_outcomes(synthetic_root: Path) -> None:
    gate = module.gate_table(synthetic_root)
    assert "games_started_share" not in set(gate["outcome"])
    assert stable_outcomes(gate, "OL") == ["pass_set_depth_1_0s", "first_0_5s_displacement", "set_peak_speed_0_1_5s"]
    assert "tfl_rate" not in stable_outcomes(gate, "DL")
    assert stable_outcomes(gate, "TE") == ["target_rate"]
    weak = gate.loc[gate["verdict"].isin(["weak", "poor"])]
    assert not weak.empty and not weak["stable"].any()


def test_benjamini_is_applied_inside_each_family() -> None:
    frame = pd.DataFrame({"family_label": ["a", "a", "b"], "p": [0.01, 0.04, 0.5]})
    q = benjamini_hochberg_by_family(frame)
    assert q.tolist() == pytest.approx([0.02, 0.04, 0.5])


def test_draft_residual_removes_the_draft_slot_trend() -> None:
    pick = np.arange(1, 61, dtype=float)
    frame = pd.DataFrame(
        {
            "draft_overall_pick": pick,
            "ngs_athleticism_score": 90 - 8 * np.log(pick),
            "ngs_college_production_score": np.nan,
            "ngs_final_score": 50.0,
        }
    )
    out = add_draft_residuals(frame)
    assert out["ngs_athleticism_resid"].abs().max() < 1e-8
    assert out["ngs_college_production_resid"].isna().all()
    assert out["ngs_final_resid"].abs().max() < 1e-8


def test_role_columns_by_group() -> None:
    frame = pd.DataFrame({"nfl_position": ["DE", "DT", "OLB"], "combine_position": ["DL", "DL", "DL"]})
    assert role_columns("DL", frame)["role"].tolist() == ["EDGE", "IDL", "EDGE"]
    ol = role_columns("OL", pd.DataFrame({"nfl_position": ["T", "G"], "combine_position": ["OL", "OL"]}))
    assert ol["role_ctl"].tolist() == [1.0, 0.0]
    db = role_columns("DB", pd.DataFrame({"nfl_position": ["CB", "S"], "combine_position": ["DB", "DB"]}))
    assert db["role"].tolist() == ["CB", "S"]
    wr = role_columns("WR", pd.DataFrame({"nfl_position": ["WR", "WR"], "combine_position": ["WR", "WR"]}))
    assert wr["role_ctl"].isna().all()
    mixed = role_columns("WRTE", pd.DataFrame({"nfl_position": ["WR", "TE"], "combine_position": ["WR", "TE"]}))
    assert mixed["role_ctl"].tolist() == [1.0, 0.0]


def _association_frame(size: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=size)
    return pd.DataFrame(
        {
            "x": x,
            "y": x + rng.normal(size=size),
            "combine_weight": rng.normal(size=size),
            "log_pick": rng.normal(size=size),
            "draft_year": np.resize([2023, 2024, 2025], size),
        }
    )


def test_association_row_reports_oriented_effect_and_replication() -> None:
    row = association_row(
        _association_frame(90),
        predictor="x",
        outcome="y",
        controls=["combine_weight"],
        meta={"part": "A", "family_label": "f", "group": "G", "family": "t"},
        predictor_orientation=1,
        outcome_orientation=1,
    )
    assert row["status"] == "ok"
    assert row["direction"] == "expected"
    assert row["r_adj"] > 0.4 and row["oriented_r"] == pytest.approx(row["r_adj"])
    assert row["ci_lo"] < row["r_adj"] < row["ci_hi"]
    assert np.isfinite(row["partial_draft_r"]) and np.isfinite(row["r_hold_2025"])
    flipped = association_row(
        _association_frame(90),
        predictor="x",
        outcome="y",
        controls=[],
        extras=["log_pick"],
        meta={},
        predictor_orientation=1,
        outcome_orientation=-1,
    )
    assert flipped["direction"] == "opposite"
    assert np.isnan(flipped["partial_draft_r"])
    neutral = association_row(
        _association_frame(90),
        predictor="x",
        outcome="y",
        controls=[],
        meta={},
        predictor_orientation=0,
        outcome_orientation=1,
    )
    assert neutral["direction"] == "unclassified"


def test_association_row_flags_small_and_degenerate_samples() -> None:
    small = association_row(
        _association_frame(30),
        predictor="x",
        outcome="y",
        controls=[],
        meta={},
        predictor_orientation=1,
        outcome_orientation=1,
    )
    assert small["status"] == "underpowered_n<40" and small["n"] == 30 and np.isnan(small["p"])
    constant = _association_frame(60).assign(y=1.0)
    flat = association_row(
        constant, predictor="x", outcome="y", controls=[], meta={}, predictor_orientation=1, outcome_orientation=1
    )
    assert flat["status"] == "degenerate"


def test_incremental_rows_name_the_conditioning_set() -> None:
    combine, players = _roster()
    ids = combine.loc[combine["combine_position"].eq("DL"), "nfl_id"]
    outcomes = pd.DataFrame({"nfl_id": ids, "hurry_rate": _outcome(ids, 1.0, 9)})
    frame = module.build_group_frame(combine, players, outcomes, "DL")
    rows = pd.DataFrame(part_a_rows(frame, "DL", ["hurry_rate"]))
    incremental = rows.loc[rows["family_label"].eq("A_incremental_DL"), "predictor"].tolist()
    assert incremental == [
        "ngs_college_production_score|athl",
        "ngs_college_production_score|athl+pick",
        "ngs_athleticism_score|pick",
        "ngs_college_production_score|pick",
    ]
    assert len(rows.loc[rows["family_label"].eq("A_main_DL")]) == 6


def test_tier_table_compares_top_and_bottom_thirds_within_role() -> None:
    size = 60
    score = np.tile(np.arange(30, dtype=float), 2)
    frame = pd.DataFrame(
        {
            "role": ["A"] * 30 + ["B"] * 30,
            "draft_year": np.resize([2023, 2024, 2025], size),
            "s": score,
            "o": score * 2.0,
        }
    )
    tiers = tier_table(frame, "G", ["s"], ["o"])
    assert tiers["n_top"].iloc[0] == 22 and tiers["n_bottom"].iloc[0] == 20
    assert tiers["difference"].iloc[0] > 25
    assert tiers["ci_lo"].iloc[0] > 0
    assert tier_table(frame.assign(o=np.nan), "G", ["s"], ["o"]).empty
    assert tier_table(frame.head(6), "G", ["s"], ["o"]).empty


def test_extra_kinematics_separates_straight_from_zigzag_paths() -> None:
    straight = _trajectory(0.0)
    zigzag = _trajectory(3.0)
    flat = extra_kinematics(straight["time"], straight["x"], straight["y"])
    wavy = extra_kinematics(zigzag["time"], zigzag["x"], zigzag["y"])
    assert flat["lateral_speed_mean"] < 0.05
    assert wavy["lateral_speed_mean"] > flat["lateral_speed_mean"] + 1.0
    assert wavy["lateral_speed_peak"] >= wavy["lateral_speed_mean"]
    assert 0.0 <= wavy["direction_change_loss"] <= 1.0
    assert extra_kinematics([0.0, 0.1], [0.0, 1.0], [0.0, 0.0]) == {}
    assert extra_kinematics([0.0, 0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]) == {}


def test_extra_kinematics_leaves_loss_undefined_for_slow_paths() -> None:
    seconds = np.arange(12) * 0.1
    values = extra_kinematics(seconds, 0.2 * seconds, np.zeros(12))
    assert np.isnan(values.get("direction_change_loss", np.nan))


def test_drill_attempts_assign_groups_alias_variants_and_pool_zone_breaks() -> None:
    combine, players = _roster()
    attempts = drill_attempts(_tracking(players, combine), players)
    assert set(attempts["group"]) == {"DL", "OL"}
    assert "PASS_PRO-MIRROR_DRILL" not in set(attempts["drill"])
    assert "PASS_PRO_MIRROR_DRILL" in set(attempts["drill"])
    assert set(attempts.loc[attempts["group"].eq("DL"), "drill"]) == {
        "BODY_CONTROL_DRILL",
        "FOUR_BAG_AGILITY_DRILL",
        "SHORT_ZONE_BREAKS",
    }
    assert attempts["lateral_speed_mean"].notna().all()
    wide, meta = player_drill_table(attempts)
    zone = meta.loc[meta["drill"].eq("SHORT_ZONE_BREAKS") & meta["metric"].eq("time_in_drill")].iloc[0]
    assert zone["share_single_attempt"] == 0.0
    assert wide["body_control_drill_lateral_speed_mean"].notna().sum() == GROUP_SIZES["DL"]
    gate = module.drill_gate_table(meta, wide)
    status = gate.set_index("drill")["repeatability_status"]
    assert status["BODY_CONTROL_DRILL"] == "unverifiable_single_attempt"
    assert status["SHORT_ZONE_BREAKS"] == "multi_attempt_not_gated"


def test_agility_index_needs_four_components_and_is_fixed() -> None:
    combine, players = _roster()
    wide, meta = player_drill_table(drill_attempts(_tracking(players, combine), players))
    index = agility_index(wide, meta, "DL")
    assert index.notna().sum() >= GROUP_SIZES["DL"]
    assert index.dropna().abs().max() < 4
    sparse = wide.copy()
    for column in sparse.columns[1:]:
        sparse.loc[sparse.index[:5], column] = np.nan
    assert agility_index(sparse, meta, "DL").iloc[:5].isna().all()
    assert agility_index(wide, meta.assign(n_players=1), "DL").isna().all()


def test_pairing_triples_cover_prespecified_cells() -> None:
    dl = pairing_triples("DL")
    assert len(dl) == 80
    assert ("run_the_hoop_drill_lateral_speed_mean", "lateral_speed_mean", "rush_win_rate") in dl
    ol = pairing_triples("OL")
    assert len(ol) == 48
    assert ("screen_drill_top_speed", "top_speed", "set_peak_speed_0_1_5s") in ol
    assert pairing_triples("DB") == set()


def test_summarize_counts_directions_and_holdout_agreement() -> None:
    frame = pd.DataFrame(
        {
            "family_label": ["f"] * 4,
            "status": ["ok", "ok", "ok", "underpowered_n<40"],
            "p": [0.01, 0.2, 0.03, np.nan],
            "q_scope": [0.05, 0.4, 0.06, np.nan],
            "direction": ["expected", "opposite", "opposite", "unclassified"],
            "r_hold_2025": [0.1, -0.1, 0.2, np.nan],
            "r_adj": [0.3, -0.1, 0.25, np.nan],
        }
    )
    summary = summarize(frame).iloc[0]
    assert (summary["n_tests"], summary["n_q_lt_0.10"], summary["n_p_lt_0.05"]) == (3, 2, 2)
    assert (summary["n_expected"], summary["n_opposite"]) == (1, 2)
    assert summary["holdout_same_sign_nominal"] == 2


def test_outcome_frame_rebuilds_missing_score_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data/processed").mkdir(parents=True)
    calls: list[str] = []

    def fake_gate(root: Path, outcome: str) -> None:
        calls.append(outcome)
        pd.DataFrame({"nfl_id": [1, 2], "posterior_mean": [0.1, 0.2]}).to_parquet(
            root / f"data/processed/scope_ol_scores_{outcome}.parquet", index=False
        )

    def fake_stage(root: Path, outcome: str) -> None:
        calls.append(outcome)
        pd.DataFrame({"nfl_id": ["1", "2"], outcome: [0.3, 0.4]}).to_parquet(
            root / f"data/processed/scope_wrte_scores_{outcome}.parquet", index=False
        )

    monkeypatch.setattr(module.scope_ol, "gate_outcome", fake_gate)
    monkeypatch.setattr(module.scope_wrte, "stage_outcome", fake_stage)
    ol = outcome_frame(tmp_path, "OL", ["snap_reaction_time"])
    wrte = outcome_frame(tmp_path, "WRTE", ["target_rate"])
    assert calls == ["snap_reaction_time", "target_rate"]
    assert ol["snap_reaction_time"].tolist() == [0.1, 0.2]
    assert wrte["nfl_id"].tolist() == [1, 2]
    assert outcome_frame(tmp_path, "OL", ["snap_reaction_time"]).shape == (2, 2)


def test_outcome_frame_rebuilds_db_and_dl_rates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"nfl_id": [1, 2], "shrunk_d_a": [0.1, 0.2]}).to_parquet(
        processed / "dl_outcomes_scores.parquet", index=False
    )

    def fake_db(root: Path) -> None:
        pd.DataFrame({"nfl_id": [3], "assist_rate": [0.5]}).to_parquet(
            root / "data/processed/scope_db_analysis.parquet", index=False
        )

    def fake_rates(root: Path) -> None:
        pd.DataFrame({"nfl_id": [1, 2], "outcome": "hurry_rate", "score": [0.7, 0.8]}).to_csv(
            root / "data/processed/scope_dl_rate_scores_progress.csv", index=False
        )

    monkeypatch.setattr(module.scope_db, "run_scope", fake_db)
    monkeypatch.setattr(module.scope_dl, "run_rates", fake_rates)
    assert outcome_frame(tmp_path, "DB", ["assist_rate"])["assist_rate"].tolist() == [0.5]
    dl = outcome_frame(tmp_path, "DL", ["shrunk_d_a", "hurry_rate"])
    assert dl["hurry_rate"].tolist() == [0.7, 0.8]


def test_run_writes_every_report_with_part_and_family_columns(synthetic_root: Path, results: pd.DataFrame) -> None:
    reports = synthetic_root / "reports"
    for name in ("results", "gate", "drill_gate", "headline", "scatter", "tiers", "summary"):
        assert (reports / f"scope_ngs_agility_{name}.csv").is_file()
    assert list(results.columns) == list(module.RESULT_COLUMNS)
    assert set(results["part"]) == {"A", "B"}
    labels = set(results["family_label"])
    assert {"A_main_DL", "A_incremental_OL", "B_pairing_DL", "B_allbyall_OL", "B_index_DL", "B_index_OL"} <= labels
    assert results["q_scope"].between(0, 1).all() or results["q_scope"].isna().any()
    ok = results.loc[results["status"].eq("ok")]
    assert ok["q_scope"].notna().all()


def test_run_reports_pairings_as_a_subset_of_all_by_all(results: pd.DataFrame) -> None:
    pairing = results.loc[results["family_label"].eq("B_pairing_DL")]
    allbyall = results.loc[results["family_label"].eq("B_allbyall_DL"), ["predictor", "outcome", "r_adj"]]
    merged = pairing.merge(allbyall, on=["predictor", "outcome"], suffixes=("", "_all"))
    assert len(merged) == len(pairing) > 0
    assert np.allclose(merged["r_adj"], merged["r_adj_all"], equal_nan=True)


def test_headline_scatter_and_tiers_have_player_level_values(synthetic_root: Path) -> None:
    reports = synthetic_root / "reports"
    headline = pd.read_csv(reports / "scope_ngs_agility_headline.csv")
    assert set(headline["predictor"]) == set(module.NGS_SCORES)
    scatter = pd.read_csv(reports / "scope_ngs_agility_scatter.csv")
    assert {"nfl_id", "group", "ngs_final_score", "ngs_final_resid", "role"} <= set(scatter.columns)
    assert set(scatter["group"]) == set(module.PART_A_GROUPS)
    tiers = pd.read_csv(reports / "scope_ngs_agility_tiers.csv")
    assert {"top_mean", "bottom_mean", "difference", "ci_lo", "ci_hi"} <= set(tiers.columns)


def test_small_tight_end_group_is_reported_not_tested_when_under_forty(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    combine = pd.read_csv(root / "data/raw/combine_results.csv")
    keep = combine.loc[~(combine["combine_position"].eq("TE") & combine["nfl_id"].gt(combine["nfl_id"].max() - 15))]
    keep.to_csv(root / "data/raw/combine_results.csv", index=False)
    out = module.run(root)
    te = out.loc[out["family_label"].eq("A_main_TE")]
    assert te["status"].eq("underpowered_n<40").all()
    assert te["q_scope"].isna().all()


def test_main_runs_the_pipeline(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    module.main(["--root", str(root)])
    assert (root / "reports/scope_ngs_agility_results.csv").is_file()
