from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol import external_validation as ev


def make_combine(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": 2005,
            "draft_year": 2005.0,
            "draft_ovr": [float(i + 1) for i in range(n)],
            "pfr_id": [f"P{i}" for i in range(n)],
            "player_name": [f"Player {i} Jr." for i in range(n)],
            "pos": ["OT", "WR", "CB", "DE", "RB", "QB"][:n],
            "ht": "6-2",
            "wt": 250.0,
            "forty": 4.8,
            "bench": 20.0,
            "vertical": 33.0,
            "broad_jump": 120.0,
            "cone": 7.2,
            "shuttle": 4.4,
        }
    )


def make_draft(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pfr_player_id": [f"P{i}" for i in range(n)],
            "season": 2005,
            "pick": [i + 1 for i in range(n)],
            "pfr_player_name": [f"Player {i}" for i in range(n)],
            "position": ["T", "WR", "CB", "DE", "RB", "QB"][:n],
            "car_av": 10.0,
            "w_av": 8.0,
            "games": 100.0,
            "seasons_started": 5.0,
            "probowls": 1.0,
            "allpro": [0, 0, 1, 0, 0, 0][:n],
        }
    )


def synthetic_table(n: int = 400, effect: float = 0.5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = rng.integers(1, 250, n).astype(float)
    forty = rng.normal(5.0, 0.2, n)
    wt = rng.normal(310, 15, n)
    outcome = 10 - 2 * np.log(pick) - effect * (forty - 5.0) * 10 + rng.normal(0, 1, n)
    blanks = {c: np.nan for c in (*ev.MEASURES, *ev.OUTCOMES) if c not in {"forty", "shuttle", "wt", "ht_in"}}
    return pd.DataFrame(
        {
            **blanks,
            "pfr_id": [f"X{i}" for i in range(n)],
            "pos": rng.choice(["OT", "OG", "C"], n),
            "draft_season": rng.integers(2000, 2022, n),
            "pick": pick,
            "log_pick": np.log(pick),
            "wt": wt,
            "forty": forty,
            "shuttle": rng.normal(4.7, 0.2, n),
            "seasons_started": outcome,
            "games": outcome * 10,
            "ht_in": 77.0,
        }
    )


def test_parse_height_and_name_normalization() -> None:
    assert ev.parse_height("6-4") == 76.0
    assert np.isnan(ev.parse_height(None))
    assert ev.normalize_name("Odell Beckham Jr.") == ev.normalize_name("odell beckham")


def test_build_player_table_drops_missing_keys_and_duplicates() -> None:
    combine = pd.concat([make_combine(), make_combine().iloc[:1], make_combine().iloc[:1].assign(pfr_id=np.nan)])
    draft = pd.concat([make_draft(), make_draft().iloc[:1], make_draft().iloc[:1].assign(pfr_player_id=np.nan)])
    table = ev.build_player_table(combine, draft)
    assert len(table) == 6
    assert table["pfr_id"].is_unique
    assert table.loc[0, "ht_in"] == 74.0
    assert table.loc[0, "log_pick"] == 0.0


def test_join_report_counts_mismatches() -> None:
    combine = make_combine().assign(pfr_id=lambda d: d["pfr_id"].where(d.index != 5, np.nan))
    draft = make_draft().assign(pfr_player_name=lambda d: d["pfr_player_name"].where(d.index != 1, "Someone Else"))
    table = ev.build_player_table(combine, draft)
    report = ev.join_report(combine, draft, table).set_index("metric")["value"]
    assert report["combine_rows_missing_pfr_id"] == 1
    assert report["joined_players"] == 5
    assert report["name_mismatch"] == 1
    assert report["position_mismatch"] == 1
    assert report["pick_mismatch"] == 0


def test_partial_spearman_detects_signal_and_ignores_pick_confound() -> None:
    frame = synthetic_table().assign(x=lambda d: d["forty"], y=lambda d: d["seasons_started"], grp=lambda d: d["pos"])
    result = ev.partial_spearman(frame)
    assert result["r"] < -0.3
    assert result["ci_lo"] < result["r"] < result["ci_hi"]
    assert result["p"] < 0.001


def test_partial_spearman_null_and_small_samples() -> None:
    rng = np.random.default_rng(3)
    frame = synthetic_table(300, effect=0.0).assign(x=rng.normal(size=300), grp="OT")
    frame["y"] = frame["seasons_started"]
    assert abs(ev.partial_spearman(frame, use_weight=False)["r"]) < 0.2
    tiny = frame.head(4)
    assert np.isnan(ev.partial_spearman(tiny)["r"])
    constant = frame.assign(x=1.0)
    assert np.isnan(ev.partial_spearman(constant)["r"])


def test_slope_bootstrap_recovers_coaching_unit_effect() -> None:
    frame = synthetic_table(500, effect=0.5).assign(
        x=lambda d: d["forty"], y=lambda d: d["seasons_started"], grp=lambda d: d["pos"]
    )
    out = ev.slope_with_bootstrap(frame, unit=0.1, use_weight=True, reps=200)
    assert out["slope_lo"] < out["slope"] < out["slope_hi"]
    assert out["slope"] == pytest.approx(-0.5, abs=0.2)
    assert np.isnan(ev.slope_with_bootstrap(frame.head(5), 0.1, True)["slope"])


def test_holm_and_bh_adjustments() -> None:
    assert ev.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ev.bh_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert ev.bh_adjust([]) == []


def test_sign_direction_and_shrinkage() -> None:
    assert ev.sign_of(-0.2) == -1
    assert ev.sign_of(float("nan")) == 0
    assert ev.direction_label(-0.1, -1) == "expected"
    assert ev.direction_label(0.1, -1) == "opposite"
    assert ev.direction_label(0.1, 0) == "no_prior"
    assert ev.shrinkage_flag(0.05, 0.3)
    assert ev.shrinkage_flag(-0.2, 0.3)
    assert not ev.shrinkage_flag(0.25, 0.3)
    assert not ev.shrinkage_flag(float("nan"), 0.3)


def test_season_rates_apply_snap_threshold_and_collapse_teams() -> None:
    rec = pd.DataFrame(
        {
            "pfr_id": ["A", "A", "A", "B", "C"],
            "season": [2019, 2019, 2019, 2019, 2019],
            "tm": ["2TM", "NYG", "DAL", "NYG", "NYG"],
            "yds": [800, 300, 500, 400, 100],
            "tgt": [100, 40, 60, 50, 10],
        }
    )
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ["A", "A", "B", "C"],
            "season": 2019,
            "game_type": ["REG", "REG", "REG", "REG"],
            "offense_snaps": [100, 100, 100, 500],
            "defense_snaps": [100, 100, 100, 500],
        }
    )
    rates = ev.season_rates(rec, snaps, "rec").set_index("pfr_id")["rate"]
    assert rates["A"] == pytest.approx(8.0)
    assert "B" not in rates.index
    assert rates["C"] == pytest.approx(10.0)

    defense = rec.rename(columns={"yds": "prss"})
    pressure = ev.season_rates(defense, snaps, "def").set_index("pfr_id")["rate"]
    assert pressure["A"] == pytest.approx(400.0)
    assert ev.average_rates(ev.season_rates(rec, snaps, "rec"), (2020, 2024)).empty


def test_run_confirmatory_on_synthetic_data() -> None:
    table = synthetic_table(600)
    table["pos"] = table["pos"].map({"OT": "OT", "OG": "OG", "C": "C"})
    seasons = {
        "rec": pd.DataFrame(columns=["pfr_id", "season", "rate"]),
        "def": pd.DataFrame(columns=["pfr_id", "season", "rate"]),
    }
    hyps = tuple(h for h in ev.HYPOTHESES if h.id in {"E3", "E4", "E3b"})
    rows = [ev.evaluate_hypothesis(h, table, seasons) for h in hyps]
    e3 = rows[0]
    assert e3["sign_matches"]
    assert e3["replicates"]
    assert e3["n"] == 600
    assert e3["slope_per_unit"] < 0


def test_run_confirmatory_holm_and_rate_hypotheses(monkeypatch: pytest.MonkeyPatch) -> None:
    table = synthetic_table(300)
    table["pos"] = "WR"
    table["vertical"] = np.random.default_rng(5).normal(35, 3, 300)
    rates = pd.DataFrame(
        {
            "pfr_id": np.repeat(table["pfr_id"], 2).to_numpy(),
            "season": np.tile([2019, 2022], 300),
            "rate": np.repeat(7 + 0.2 * (table["vertical"] - 35).to_numpy(), 2)
            + np.random.default_rng(6).normal(0, 0.5, 600),
        }
    )
    seasons = {"rec": rates, "def": rates}
    result = ev.run_confirmatory(table, seasons)
    e2 = result.set_index("hypothesis").loc["E2"]
    assert e2["r"] > 0.3
    assert e2["replicates"]
    assert e2["holm_p"] < 0.05
    assert bool(e2["confirmed"])
    assert result.loc[~result["primary"], "holm_p"].isna().all()
    assert len(result) == len(ev.HYPOTHESES)


def test_run_atlas_respects_min_n_and_flags() -> None:
    table = synthetic_table(500)
    table["draft_season"] = np.where(np.arange(500) % 2 == 0, 2005, 2015)
    table["pos"] = "OT"
    for col in ("car_av", "w_av", "probowls"):
        table[col] = table["seasons_started"]
    table["ht_in"] = 77.0
    table["allpro"] = (table["seasons_started"] > 8).astype(float)
    for col in ("vertical", "broad_jump", "cone", "bench"):
        table[col] = np.nan
    atlas = ev.run_atlas(table, reps=20)
    assert set(atlas["group"]) == {"OL"}
    assert set(atlas["measure"]) == {"forty", "shuttle", "wt"}
    assert {"q_bh", "significant", "pick_shrinkage_flag", "rank_in_group"} <= set(atlas.columns)
    forty = atlas[(atlas["measure"] == "forty") & (atlas["outcome"] == "seasons_started")].iloc[0]
    assert forty["direction"] == "expected"
    assert forty["replicates"]
    summary = ev.atlas_summary(atlas)
    assert list(summary["family"]) == ["all", "significant"]
    assert summary.loc[0, "expected"] > 0


def test_run_atlas_empty_when_cells_too_small() -> None:
    atlas = ev.run_atlas(synthetic_table(50).assign(pos="OT"), reps=5)
    assert atlas.empty
    assert ev.atlas_summary(atlas).empty


def test_end_to_end_writes_reports(tmp_path: Path) -> None:
    ext = tmp_path / "data" / "external"
    ext.mkdir(parents=True)
    table = synthetic_table(260).assign(pos="OT")
    ids = table["pfr_id"]
    combine = pd.DataFrame(
        {
            "season": table["draft_season"],
            "draft_year": table["draft_season"].astype(float),
            "draft_ovr": table["pick"],
            "pfr_id": ids,
            "player_name": "N " + ids,
            "pos": "OT",
            "ht": "6-5",
            "wt": table["wt"],
            "forty": table["forty"],
            "bench": 25.0,
            "vertical": 30.0,
            "broad_jump": 110.0,
            "cone": 7.5,
            "shuttle": table["shuttle"],
        }
    )
    draft = pd.DataFrame(
        {
            "pfr_player_id": ids,
            "season": table["draft_season"],
            "pick": table["pick"],
            "pfr_player_name": "N " + ids,
            "position": "T",
            "car_av": table["seasons_started"],
            "w_av": table["seasons_started"],
            "games": table["games"],
            "seasons_started": table["seasons_started"],
            "probowls": 0,
            "allpro": 0,
        }
    )
    combine.to_csv(ext / "combine_combine.csv", index=False)
    draft.to_csv(ext / "draft_picks_draft_picks.csv", index=False)
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ids[:50],
            "season": 2019,
            "game_type": "REG",
            "offense_snaps": 500,
            "defense_snaps": 500,
        }
    )
    snaps.to_csv(ext / "snap_2019.csv", index=False)
    rec = pd.DataFrame({"pfr_id": ids[:50], "season": 2019, "tm": "NYG", "yds": 300, "tgt": 40})
    defense = pd.DataFrame({"pfr_id": ids[:50], "season": 2019, "tm": "NYG", "prss": 20})
    rec.to_csv(ext / "pfr_advstats_advstats_season_rec.csv", index=False)
    defense.to_csv(ext / "pfr_advstats_advstats_season_def.csv", index=False)
    ev.main(["--root", str(tmp_path)])
    for name in ("external_confirmatory", "external_atlas", "external_join_report"):
        assert (tmp_path / "reports" / f"{name}.csv").exists()
    join = pd.read_csv(tmp_path / "reports" / "external_join_report.csv")
    assert "joined_players_with_rec_rate" in set(join["metric"])
