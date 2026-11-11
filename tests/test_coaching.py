from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol import coaching

ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (ROOT / "data/processed/combine_predictors.parquet").is_file() and (ROOT / "data/raw/players.csv").is_file()
REPS = 60


def _group_frame(
    n: int = 80, slope: float = 0.0, weight_slope: float = 0.0, seed: int = 4, group: str = "DL"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weight = rng.normal(300, 20, n)
    forty = 5.0 + 0.004 * (weight - 300) + rng.normal(0, 0.2, n)
    role = (rng.random(n) < 0.5).astype(float)
    outcome = -slope * (forty - 5.0) / 0.1 * 0.1 + weight_slope * (weight - 300) + rng.normal(0, 1.0, n)
    return pd.DataFrame(
        {
            "nfl_id": np.arange(n) + (1000 if group == "DL" else 5000),
            "group": group,
            "role": np.where(role > 0, "EDGE", "IDL"),
            "role_ctl": role,
            "draft_year": rng.choice([2023, 2024, 2025], n),
            "draft_overall_pick": rng.integers(1, 250, n).astype(float),
            "log_pick": np.log(rng.integers(1, 250, n).astype(float)),
            "nfl_position": "DE",
            "forty": forty,
            "combine_weight": weight,
            "pressure_rate": outcome,
            "hurry_rate": rng.normal(size=n),
        }
    )


def _controls(frame: pd.DataFrame, column: str) -> list[str]:
    return [name for name in ("combine_weight", "role_ctl") if name != column and frame[name].notna().any()]


def _row(frame: pd.DataFrame, outcome_sign: int = 1, outcome: str = "pressure_rate", lookup: dict | None = None):
    measure = coaching.measure_for("forty")
    return coaching.effect_row(
        frame,
        group="DL",
        measure=measure,
        outcome=outcome,
        unit=0.10,
        outcome_sign=outcome_sign,
        controls=_controls(frame, "forty"),
        lookup=lookup or {},
        reps=REPS,
    )


def test_trainability_table_covers_every_official_measure_and_sensor_family() -> None:
    table = coaching.trainability_table()
    for column in coaching.OFFICIAL_COLUMNS:
        assert column in set(table["measure"])
    assert set(table["trainability"]) == {"trainable", "partially_trainable", "fixed"}
    families = set(table["family"])
    assert families == {"official", "sensor", "route_drill"}
    assert table["justification"].str.len().gt(10).all()
    assert table["source"].str.len().gt(3).all()


def test_realistic_improvement_units_are_the_prespecified_values() -> None:
    table = coaching.trainability_table().set_index("measure")["realistic_improvement"]
    assert table["forty"] == 0.10
    assert table["ten_yd_split"] == 0.05
    assert table["three_cone"] == 0.15
    assert table["short_shuttle"] == 0.10
    assert table["vertical"] == 2.0
    assert table["broad_jump"] == 4.0
    assert table["bench_reps"] == 3.0


def test_fixed_measures_are_not_rank_eligible() -> None:
    for column in ("combine_height", "arm_length", "wing_span", "hand_size"):
        assert not coaching.measure_for(column).rank_eligible
    assert coaching.measure_for("forty").rank_eligible


def test_write_trainability_creates_csv(tmp_path: Path) -> None:
    written = coaching.write_trainability(tmp_path / "t.csv")
    assert len(pd.read_csv(tmp_path / "t.csv")) == len(written)


def test_improvement_unit_uses_group_weight_fixed_value_or_half_a_standard_deviation() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert coaching.improvement_unit(coaching.measure_for("combine_weight"), "OL", values) == 10.0
    assert coaching.improvement_unit(coaching.measure_for("combine_weight"), "WR", values) == 5.0
    assert coaching.improvement_unit(coaching.measure_for("vertical"), "DL", values) == 2.0
    expected = 0.5 * values.std(ddof=1)
    assert coaching.improvement_unit(coaching.measure_for("arm_length"), "DL", values) == pytest.approx(expected)


def test_route_measures_are_derived_from_the_template() -> None:
    measure = coaching.measure_for("route_go_top_speed")
    assert measure.family == "route_drill"
    assert measure.unit is None
    assert measure.column == "route_go_top_speed"


def test_fit_ols_recovers_the_slope_and_matches_lstsq() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 1))
    y = 2.0 * x[:, 0] + rng.normal(0, 0.5, 200)
    frame = pd.DataFrame({"x": x[:, 0] * 3 + 10, "y": y})
    fit = coaching.fit_subset(frame, "x", "y", [])
    assert fit.beta == pytest.approx(2.0 / 3.0, abs=0.1)
    design = np.column_stack([np.ones(200), frame["x"]])
    assert fit.beta == pytest.approx(np.linalg.lstsq(design, y, rcond=None)[0][1])
    assert fit.p < 0.001
    assert fit.se > 0
    assert fit.residual_sd == pytest.approx(0.5, abs=0.1)


def test_fit_ols_has_a_large_p_value_when_there_is_no_relationship() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    assert coaching.fit_subset(frame, "x", "y", []).p > 0.01


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    frame = _group_frame(slope=0.5)
    controls = _controls(frame, "forty")
    design, spread = coaching._design(frame, "forty", controls)
    y = frame["pressure_rate"].to_numpy()
    first = coaching.bootstrap_slopes(y, design, spread, 100, 7)
    second = coaching.bootstrap_slopes(y, design, spread, 100, 7)
    assert np.array_equal(first, second)
    point = coaching.fit_ols(y, design, spread).beta
    assert np.percentile(first, 2.5) < point < np.percentile(first, 97.5)


def test_shrink_reports_fraction_lost_and_handles_zero_reference() -> None:
    assert coaching.shrink(2.0, 1.0) == pytest.approx(0.5)
    assert coaching.shrink(-2.0, -2.0) == pytest.approx(0.0)
    assert coaching.shrink(2.0, -1.0) == pytest.approx(1.5)
    assert np.isnan(coaching.shrink(0.0, 1.0))
    assert np.isnan(coaching.shrink(1.0, float("nan")))


@pytest.mark.parametrize(
    ("ci_lo", "ci_hi", "signed", "expected"),
    [
        (0.1, 0.5, True, "supported_benefit"),
        (-0.5, -0.1, True, "supported_opposite"),
        (-0.1, 0.1, True, "no_effect_larger_than_X"),
        (-0.1, 0.5, True, "cannot_say"),
        (-0.5, 0.1, True, "cannot_say"),
        (0.1, 0.15, False, "unsigned_association"),
        (-0.1, 0.1, False, "no_effect_larger_than_X"),
    ],
)
def test_verdict_rules(ci_lo: float, ci_hi: float, signed: bool, expected: str) -> None:
    verdict, bound = coaching.verdict_for(ci_lo, ci_hi, signed, sesoi=0.2)
    assert verdict == expected
    assert bound == max(abs(ci_lo), abs(ci_hi))


def test_evidence_grade_levels() -> None:
    kwargs = {"holdout_agrees": True, "q_group": 0.5, "ci_lo": -1.0, "ci_hi": 1.0}
    assert coaching.evidence_grade(master_q=0.04, p=0.5, **kwargs) == "supported"
    assert coaching.evidence_grade(master_q=0.5, p=0.5, **kwargs) == "speculative"
    assert coaching.evidence_grade(master_q=np.nan, p=0.5, **{**kwargs, "ci_lo": 0.1}) == "suggestive"
    assert coaching.evidence_grade(master_q=np.nan, p=0.04, **{**kwargs, "q_group": 0.05}) == "suggestive"
    assert coaching.evidence_grade(master_q=np.nan, p=0.04, **{**kwargs, "holdout_agrees": False}) == "speculative"
    assert coaching.evidence_grade(master_q=np.nan, p=np.nan, **kwargs) == "speculative"


def test_master_lookup_takes_the_best_q_over_group_aliases(tmp_path: Path) -> None:
    path = tmp_path / "master.csv"
    pd.DataFrame(
        {
            "group": ["WRTE", "WR", "DB"],
            "outcome": ["sep", "sep", "sep"],
            "predictor": ["combine_weight"] * 3,
            "q_master": [0.04, 0.2, 0.01],
        }
    ).to_csv(path, index=False)
    lookup = coaching.master_lookup(path)
    assert coaching.master_q(lookup, "WR", "sep", "combine_weight") == pytest.approx(0.04)
    assert coaching.master_q(lookup, "TE", "sep", "combine_weight") == pytest.approx(0.04)
    assert np.isnan(coaching.master_q(lookup, "OL", "sep", "combine_weight"))
    assert coaching.master_lookup(tmp_path / "missing.csv") == {}


def test_effect_row_flags_a_real_beneficial_association_in_coaching_units() -> None:
    row = _row(_group_frame(n=400, slope=1.0, seed=1))
    assert row["status"] == "ok"
    assert row["verdict"] == "supported_benefit"
    assert row["ci_lo"] < 0.1 < row["ci_hi"]
    assert row["ci_lo"] > 0
    assert row["effect_sd"] == pytest.approx(row["effect"] / row["outcome_sd"])
    assert row["mde"] > 0
    assert row["sesoi"] == pytest.approx(0.2 * row["outcome_sd"])


def test_effect_row_sign_flips_with_outcome_orientation() -> None:
    frame = _group_frame(n=200, slope=1.0)
    positive = _row(frame, outcome_sign=1)
    negative = _row(frame, outcome_sign=-1)
    assert negative["effect"] == pytest.approx(-positive["effect"])
    assert negative["verdict"] == "supported_opposite"


def test_effect_row_unsigned_outcome_never_earns_supported_benefit() -> None:
    row = _row(_group_frame(n=200, slope=1.0), outcome_sign=0)
    assert row["verdict"] == "unsigned_association"
    assert not row["outcome_signed"]


def test_effect_row_reports_no_effect_with_bound_when_data_are_tight() -> None:
    row = _row(_group_frame(n=400, slope=0.0), outcome="hurry_rate")
    assert row["verdict"] == "no_effect_larger_than_X"
    assert row["max_effect_bound"] <= row["sesoi"]


def test_effect_row_small_sample_is_kept_but_not_tested() -> None:
    row = _row(_group_frame(n=20))
    assert row["status"].startswith("not_tested")
    assert row["verdict"] == "not_tested"
    assert np.isnan(row["effect"])


def test_effect_row_constant_predictor_is_degenerate() -> None:
    frame = _group_frame().assign(forty=5.0)
    assert _row(frame)["status"] == "degenerate"


def test_weight_confounding_is_flagged_when_weight_explains_the_association() -> None:
    row = _row(_group_frame(n=300, slope=0.0, weight_slope=0.08, seed=9))
    assert row["weight_shrink"] > 0.4
    assert row["weight_confounded"]
    assert "weight" in row["confounded_by"]
    assert "weight" in row["confounder_note"]


def test_effect_row_reports_holdout_and_draft_sensitivity() -> None:
    row = _row(_group_frame(n=240, slope=1.0))
    assert row["n_2025"] >= coaching.MIN_YEAR_N
    assert row["holdout_sign_agrees"] in (True, False)
    assert np.isfinite(row["effect_with_draft"])
    assert row["n_disc"] + row["n_2025"] == row["n"]


def test_effect_row_attaches_master_q() -> None:
    lookup = {("DL", "pressure_rate", "forty"): 0.03}
    assert _row(_group_frame(), lookup=lookup)["master_q"] == pytest.approx(0.03)


def test_weight_predictor_skips_the_weight_adjustment() -> None:
    frame = _group_frame(n=100)
    measure = coaching.measure_for("combine_weight")
    row = coaching.effect_row(
        frame,
        group="DL",
        measure=measure,
        outcome="pressure_rate",
        unit=10.0,
        outcome_sign=1,
        controls=_controls(frame, "combine_weight"),
        lookup={},
        reps=REPS,
    )
    assert np.isnan(row["weight_shrink"])


def _table() -> pd.DataFrame:
    dl = _group_frame(n=120, slope=1.0)
    ol = _group_frame(n=100, slope=0.0, seed=8, group="OL").rename(columns={"pressure_rate": "pass_set_depth_1_0s"})
    ol = ol.assign(hurry_rate=np.nan)
    table = pd.concat([dl, ol], ignore_index=True)
    return table.assign(
        pass_set_depth_1_0s=table["pass_set_depth_1_0s"].where(table["group"].eq("OL")),
        pressure_rate=table["pressure_rate"].where(table["group"].eq("DL")),
        hurry_rate=table["hurry_rate"].where(table["group"].eq("DL")),
        route_go_top_speed=np.where(table["group"].eq("DL"), np.nan, np.linspace(7, 9, len(table))),
    )


def test_compute_effects_covers_every_measure_outcome_pair_and_grades_all_rows(tmp_path: Path) -> None:
    table = _table()
    master = tmp_path / "master.csv"
    pd.DataFrame({"group": ["DL"], "outcome": ["pressure_rate"], "predictor": ["forty"], "q_master": [0.01]}).to_csv(
        master, index=False
    )
    effects = coaching.compute_effects(table, master, reps=REPS)
    assert set(effects["group"]) == {"DL", "OL"}
    dl = effects.loc[effects["group"].eq("DL")]
    assert set(dl["outcome"]) == {"pressure_rate", "hurry_rate"}
    assert set(dl["measure"]) == {"forty", "combine_weight"}
    assert effects["grade"].isin(["supported", "suggestive", "speculative"]).all()
    top = dl.loc[dl["measure"].eq("forty") & dl["outcome"].eq("pressure_rate")].iloc[0]
    assert top["grade"] == "supported"
    assert effects["q_group"].notna().any()
    assert list(effects.columns) == list(coaching.RESULT_COLUMNS)


def _effects_rows(verdicts: list[str], agrees: list[bool], eligible: bool = True) -> pd.DataFrame:
    n = len(verdicts)
    return pd.DataFrame(
        {
            "group": "DL",
            "measure": "forty",
            "trainability": "partially_trainable",
            "rank_eligible": eligible,
            "status": "ok",
            "verdict": verdicts,
            "holdout_sign_agrees": agrees,
            "outcome": [f"o{i}" for i in range(n)],
            "effect": 0.1,
            "ci_lo": [-0.1] * n,
            "ci_hi": [0.1] * n,
            "effect_sd": [0.1 + i / 100 for i in range(n)],
            "ci_lo_sd": -0.1,
            "ci_hi_sd": [0.1 + i / 100 for i in range(n)],
            "outcome_sd": 1.0,
            "mde": 0.2,
            "n": 80,
            "unit_label": "0.10 s faster",
            "improvement_unit": 0.1,
            "grade": ["suggestive"] + ["speculative"] * (n - 1),
            "weight_confounded": False,
        }
    )


def test_decide_prioritizes_only_when_holdout_agrees() -> None:
    assert coaching.decide(_effects_rows(["supported_benefit", "cannot_say"], [True, False]), True) == "prioritize"
    assert coaching.decide(_effects_rows(["supported_benefit", "cannot_say"], [False, False]), True) == "unclear"


def test_decide_deprioritizes_only_when_every_outcome_is_ruled_out() -> None:
    assert coaching.decide(_effects_rows(["no_effect_larger_than_X"] * 2, [True, True]), True) == "deprioritize"
    assert coaching.decide(_effects_rows(["no_effect_larger_than_X", "cannot_say"], [True, True]), True) == "unclear"


def test_decide_handles_fixed_and_untested_measures() -> None:
    assert coaching.decide(_effects_rows(["cannot_say"], [False], eligible=False), False) == "recruiting_only"
    untested = _effects_rows(["not_tested"], [False]).assign(status="not_tested_n<30")
    assert coaching.decide(untested, True) == "not_tested"


def test_headline_row_picks_the_relevant_outcome_per_decision() -> None:
    rows = _effects_rows(["supported_benefit", "supported_benefit", "cannot_say"], [True, True, True])
    assert coaching.headline_row(rows, "prioritize")["outcome"] == "o1"
    assert coaching.headline_row(rows, "deprioritize")["outcome"] == "o2"
    assert coaching.headline_row(rows, "unclear")["outcome"] == "o2"
    untested = rows.assign(status="not_tested_n<30")
    assert coaching.headline_row(untested, "unclear")["outcome"] == "o0"


def test_rollup_and_markdown_include_every_group_measure() -> None:
    effects = pd.concat(
        [
            _effects_rows(["supported_benefit", "cannot_say"], [True, False]),
            _effects_rows(["no_effect_larger_than_X"], [True]).assign(measure="vertical"),
            _effects_rows(["cannot_say"], [False]).assign(measure="combine_weight"),
            _effects_rows(["not_tested"], [False]).assign(measure="bench_reps", status="not_tested_n<30"),
            _effects_rows(["cannot_say"], [False], eligible=False).assign(measure="arm_length"),
        ],
        ignore_index=True,
    )
    summary = coaching.rollup(effects)
    decisions = dict(zip(summary["measure"], summary["decision"], strict=True))
    assert decisions == {
        "forty": "prioritize",
        "vertical": "deprioritize",
        "combine_weight": "unclear",
        "bench_reps": "not_tested",
        "arm_length": "recruiting_only",
    }
    text = coaching.group_markdown(summary, "DL")
    assert "**prioritize** (1)" in text
    assert "rules out >" in text
    assert "lb heavier" in text
    assert "not tested" in text
    assert coaching.group_markdown(summary, "TE") == ""


def test_statement_covers_each_verdict() -> None:
    base = {
        "status": "ok",
        "n": 70,
        "ci_lo": -0.1,
        "ci_hi": 0.1,
        "outcome": "target_rate",
        "mde": 0.2,
        "unit_label": "0.10 s faster",
        "measure": "forty",
        "max_effect_bound": 0.1,
        "effect": 0.05,
        "outcome_sd": 0.5,
        "improvement_unit": 0.1,
    }
    assert "rule out" in coaching.statement(pd.Series({**base, "verdict": "no_effect_larger_than_X"}))
    assert "Cannot say" in coaching.statement(pd.Series({**base, "verdict": "cannot_say"}))
    assert "supported_benefit" in coaching.statement(pd.Series({**base, "verdict": "supported_benefit"}))
    assert "not tested" in coaching.statement(pd.Series({**base, "status": "not_tested_n<30"}))
    sd_based = {**base, "unit_label": "0.5 SD of the measure", "measure": "route_go_top_speed", "verdict": "cannot_say"}
    assert "0.5 SD (0.1) of route_go_top_speed" in coaching.statement(pd.Series(sd_based))
    weight = {**base, "measure": "combine_weight", "improvement_unit": 5.0, "verdict": "cannot_say"}
    assert "5 lb heavier" in coaching.statement(pd.Series(weight))


def test_wr_block_selects_wide_receiver_speed_and_agility_rows() -> None:
    effects = pd.DataFrame(
        {
            "group": ["WR", "WR", "DL", "TE"],
            "measure": ["forty", "vertical", "forty", "route_go_top_speed"],
            "family": ["official", "official", "official", "route_drill"],
            "outcome": ["target_rate"] * 4,
            "status": ["ok", "ok", "ok", "not_tested_n<30"],
            "verdict": ["cannot_say", "cannot_say", "cannot_say", "not_tested"],
            "n": [70, 70, 70, 20],
            "ci_lo": -0.1,
            "ci_hi": 0.1,
            "mde": 0.2,
            "unit_label": "x",
            "max_effect_bound": 0.1,
            "effect": 0.0,
            "outcome_sd": 1.0,
            "improvement_unit": 0.1,
        }
    )
    block = coaching.wr_block(effects)
    assert set(zip(block["group"], block["measure"], strict=True)) == {("WR", "forty")}
    assert block["statement"].str.len().gt(0).all()


def test_validation_passes_on_a_clean_table_and_fails_on_duplicates() -> None:
    table = pd.DataFrame(
        {
            "nfl_id": range(5),
            "group": coaching.GROUPS,
            "draft_year": [2023, 2024, 2025, 2023, 2024],
            "combine_weight": 250.0,
            "pressure_rate": [1.0, np.nan, np.nan, np.nan, np.nan],
            "pass_set_depth_1_0s": [np.nan, 1.0, np.nan, np.nan, np.nan],
            "assist_rate": [np.nan, np.nan, 1.0, np.nan, np.nan],
            "rec_yards_per_route": [np.nan, np.nan, np.nan, 1.0, 1.0],
        }
    )
    assert coaching.assert_valid(table)["passed"].all()
    duplicated = pd.concat([table, table.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="no duplicate players"):
        coaching.assert_valid(duplicated)


def test_data_dictionary_describes_every_family_of_column() -> None:
    table = _table()
    dictionary = coaching.data_dictionary(table)
    assert {"nfl_id", "forty", "route_go_top_speed", "pressure_rate"} <= set(dictionary["column"])
    assert dictionary["description"].str.len().gt(0).all()


def test_attempts_to_wide_averages_attempts_and_names_columns() -> None:
    attempts = pd.DataFrame(
        {
            "nfl_id": ["1", "1", "2"],
            "drill": ["FORTY_YARD_DASH", "FORTY_YARD_DASH", "SHORT_SHUTTLE"],
            "metric": ["top_speed", "top_speed", "time_in_drill"],
            "value": [8.0, 10.0, 4.5],
        }
    )
    wide = coaching.attempts_to_wide(attempts).set_index("nfl_id")
    assert wide.loc["1", "forty_yard_dash_top_speed"] == pytest.approx(9.0)
    assert wide.loc["2", "short_shuttle_time_in_drill"] == pytest.approx(4.5)


def test_db_attempts_long_melts_metrics_and_drops_missing() -> None:
    attempts = pd.DataFrame(
        {
            "nfl_id": [1, 1],
            "drill": ["FORTY_YARD_DASH", "FORTY_YARD_DASH"],
            "attempt": [1, 2],
            "top_speed": [9.0, np.nan],
            "time_in_drill": [4.5, 4.6],
        }
    )
    long = coaching.db_attempts_long(attempts)
    assert len(long) == 3
    assert set(long["metric"]) == {"top_speed", "time_in_drill"}


def test_route_speed_table_is_empty_when_caches_are_missing(tmp_path: Path) -> None:
    assert coaching.route_speed_table(tmp_path).empty


@pytest.mark.skipif(not HAS_DATA, reason="local gitignored data not available")
def test_real_player_table_is_valid_and_has_one_row_per_player() -> None:
    table = coaching.load_player_table(ROOT)
    report = coaching.assert_valid(table)
    assert report["passed"].all()
    assert table["nfl_id"].is_unique
    assert set(table["group"]) == set(coaching.GROUPS)
    assert len(table) == table["nfl_id"].nunique()
    assert table.groupby("group").size().min() >= 40
    assert table["combine_weight"].notna().all()


@pytest.mark.skipif(not HAS_DATA, reason="local gitignored data not available")
def test_real_run_writes_all_outputs(tmp_path: Path) -> None:
    results = coaching.run(ROOT, reps=40, output=tmp_path)
    for name in (
        "coaching_effects.csv",
        "coaching_group_rollup.csv",
        "coaching_trainability.csv",
        "coaching_wr_block.csv",
        "coaching_data_dictionary.csv",
    ):
        assert (tmp_path / name).is_file()
    effects = results["effects"]
    assert effects.duplicated(["group", "measure", "outcome"]).sum() == 0
    assert (effects["status"].eq("ok") | effects["verdict"].eq("not_tested")).all()
    assert set(results["rollup"]["decision"]) <= {
        "prioritize",
        "deprioritize",
        "unclear",
        "not_tested",
        "recruiting_only",
    }


def test_main_runs_end_to_end_on_a_real_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Path] = {}
    monkeypatch.setattr(coaching, "run", lambda root, reps=0, output=None: called.update(root=root) or {})
    coaching.main(["--root", str(tmp_path)])
    assert called["root"] == tmp_path.resolve()
