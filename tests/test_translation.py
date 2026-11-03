from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combine_dlol import translation


def _group_data(seed: int, planted: bool) -> translation.GroupData:
    generator = np.random.default_rng(seed)
    count = 80
    traditional = generator.normal(size=(count, 2))
    sensor = generator.normal(size=(count, 1))
    outcome = (
        (4.0 * sensor[:, 0] + generator.normal(scale=0.35, size=count)) if planted else generator.normal(size=count)
    )
    outcomes = pd.DataFrame(
        {name: outcome + generator.normal(scale=0.03, size=count) for name in translation.OUTCOMES["DL"]}
    )
    return translation.GroupData(
        group="DL",
        ids=np.arange(count),
        outcomes=outcomes,
        features={
            "M1": traditional[:, :1],
            "M2": traditional,
            "M3": np.column_stack((traditional, sensor)),
            "M4": np.column_stack((traditional, sensor)),
        },
        draft_pick=generator.integers(1, 250, size=count).astype(float),
        draft_year=np.repeat([2020, 2021, 2022, 2023], count // 4).astype(float),
        reliability={name: 0.9 for name in translation.OUTCOMES["DL"]},
    )


def _settings() -> translation.Settings:
    return translation.Settings(repeats=10, permutations=50, permutation_repeats=4, bootstraps=100, seed=91)


def test_operator_matches_naive_sklearn_ridge_loop() -> None:
    features = np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0], [6.0, 5.0]])
    outcome = np.array([0.3, -1.0, 1.2, 2.2, 1.7, 3.1])
    train = np.array([0, 1, 2, 3])
    test = np.array([4, 5])

    operator = translation.prediction_operator(features, train, test)
    actual = operator @ outcome
    expected = translation.naive_ridge_predictions(features, train, test, outcome)

    assert np.allclose(actual, expected, atol=1e-10)


def test_in_fold_imputation_does_not_use_poisoned_test_rows() -> None:
    train = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0]])
    ordinary_test = np.array([[4.0, np.nan]])
    poisoned_test = np.array([[4.0, 1_000_000.0]])

    ordinary_train, ordinary = translation._processed_design(train, ordinary_test)
    poisoned_train, poisoned = translation._processed_design(train, poisoned_test)

    assert np.array_equal(ordinary_train, poisoned_train)
    assert ordinary[0, 1] == pytest.approx(0.0)
    assert poisoned[0, 1] > 100_000.0


def test_joint_permutation_preserves_outcome_rows() -> None:
    outcomes = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    shuffled = translation.joint_row_permutation(outcomes, np.random.default_rng(5))

    assert {tuple(row) for row in shuffled} == {tuple(row) for row in outcomes}
    assert np.all(shuffled[:, 1] == shuffled[:, 0] * 10.0)


def test_family_wise_p_value_uses_maximum_distribution() -> None:
    maxima = np.array([0.01, 0.20, 0.10, 0.30])

    assert translation.family_p_value(maxima, 0.15) == pytest.approx(3.0 / 5.0)
    assert translation.family_p_value(maxima, 0.31) == pytest.approx(1.0 / 5.0)


def test_batched_delta_r2_matches_single_outcome_calculation() -> None:
    base = np.array([[2.0, 3.0], [1.0, 4.0], [3.0, 2.0]])
    model = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]])
    outcomes = np.array([[1.0, 3.0], [2.0, 1.0], [4.0, 2.0]])

    actual = translation.delta_r2_batch(base, model, outcomes)
    expected = np.array(
        [translation.delta_r2(base[:, column], model[:, column], outcomes[:, column]) for column in range(2)]
    )

    assert np.allclose(actual, expected)


def test_planted_sensor_effect_is_detected_with_familywise_control() -> None:
    results = translation.primary_results({"DL": _group_data(22, planted=True)}, _settings())
    target = results.loc[(results["outcome"] == "shrunk_d_a") & (results["comparison"] == "M3_vs_M2")].iloc[0]

    assert target["delta_r2"] > 0.0
    assert target["p_fwer"] < 0.05


def test_pure_noise_has_nonpositive_average_delta_and_large_familywise_p() -> None:
    data = _group_data(37, planted=False)
    results = translation.primary_results({"DL": data}, _settings())
    base = translation.build_operators(data.features["M2"], 10, 5, 91)
    sensor = translation.build_operators(data.features["M3"], 10, 5, 91)
    generator = np.random.default_rng(17)
    deltas = [
        translation.delta_r2(
            translation.averaged_squared_errors(base, outcome),
            translation.averaged_squared_errors(sensor, outcome),
            outcome,
        )
        for outcome in generator.normal(size=(80, 30)).T
    ]

    assert np.mean(deltas) <= 0.0
    assert results["p_fwer"].min() > 0.05


def test_ol_pressure_allowed_is_excluded() -> None:
    assert "pressure_allowed_residual" not in translation.OUTCOMES["OL"]
    assert set(translation.OUTCOMES["OL"]) == {
        "peak_pressure_probability_allowed_residual",
        "snap_reaction_time_residual",
        "first_0_5s_displacement_residual",
    }


def test_string_predictor_ids_join_integer_outcomes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    predictors = pd.DataFrame({"nfl_id": ["10", "11"], "group": ["DL", "DL"], "forty": [4.8, 4.9]})
    scores = pd.DataFrame({"nfl_id": [10, 11], **{name: [0.1, 0.2] for name in translation.OUTCOMES["DL"]}})
    joined = pd.DataFrame({"nfl_id": [10, 10, 11], "draft_overall_pick": [4, 4, 9], "draft_year": [2020, 2020, 2021]})
    metadata = pd.DataFrame(
        {
            "predictor_column": ["forty"],
            "group": ["DL"],
            "family": ["official"],
            "eligible_for_M3": [False],
            "eligible_for_M4": [False],
        }
    )

    def read_parquet(path: Path) -> pd.DataFrame:
        if path.name == "combine_predictors.parquet":
            return predictors
        if path.name == "dl_outcomes_scores.parquet":
            return scores
        return joined

    monkeypatch.setattr(translation.pd, "read_parquet", read_parquet)
    monkeypatch.setattr(
        translation.pd,
        "read_csv",
        lambda path: (
            metadata
            if path.name == "phase2_combine_predictors.csv"
            else pd.DataFrame({"group": ["all"], "outcome": ["d_a"], "split_half_median": [0.9]})
        ),
    )
    data = translation.load_group_data(tmp_path, "DL")

    assert data.ids.tolist() == [10, 11]
    assert data.features["M1"].shape == (2, 1)


def test_descriptive_power_and_plots_cover_reduced_settings(tmp_path: Path) -> None:
    data = _group_data(49, planted=True)
    settings = translation.Settings(
        repeats=2,
        permutations=1,
        permutation_repeats=1,
        bootstraps=10,
        power_simulations=1,
        power_bootstraps=10,
        power_permutations=5,
        seed=101,
    )

    descriptive = translation.descriptive_results({"DL": data}, settings)
    power = translation.power_results({"DL": data}, settings)
    results = pd.DataFrame(
        {
            "group": ["DL"],
            "outcome": ["shrunk_d_a"],
            "comparison": ["M3_vs_M2"],
            "delta_r2": [0.1],
            "ci_lower": [0.02],
            "ci_upper": [0.2],
            "p_fwer": [0.01],
        }
    )
    forest = tmp_path / "forest.png"
    power_path = tmp_path / "power.png"
    translation.forest_plot(results, forest)
    translation.power_plot(power, power_path)

    assert "outcome_reliability" in set(descriptive["metric"])
    assert len(power) == len(translation.OUTCOMES["DL"]) * 6
    assert forest.exists()
    assert power_path.exists()


def test_run_writes_all_requested_outputs_and_cli_keeps_frozen_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    results = pd.DataFrame(
        {
            "group": ["DL"],
            "outcome": ["shrunk_d_a"],
            "comparison": ["M3_vs_M2"],
            "delta_r2": [0.1],
            "ci_lower": [0.02],
            "ci_upper": [0.2],
            "p_fwer": [0.01],
        }
    )
    descriptive = pd.DataFrame({"group": ["DL"], "outcome": ["shrunk_d_a"], "metric": ["x"], "value": [1.0]})
    power = pd.DataFrame(
        {
            "group": ["DL"],
            "outcome": ["shrunk_d_a"],
            "planted_partial_r2": [0.02],
            "detection_probability": [0.5],
        }
    )
    monkeypatch.setattr(translation, "load_group_data", lambda _, group: _group_data(61 if group == "DL" else 62, True))
    monkeypatch.setattr(translation, "primary_results", lambda _, __: results)
    monkeypatch.setattr(translation, "descriptive_results", lambda _, __: descriptive)
    monkeypatch.setattr(translation, "power_results", lambda _, __: power)

    translation.run(tmp_path, translation.Settings())

    assert translation.parse_args(["--root", str(tmp_path)]).root == tmp_path
    assert translation.Settings().repeats == 200
    assert translation.Settings().permutations == 1000
    assert {path.name for path in reports.iterdir()} == {
        "translation_results.csv",
        "translation_descriptive.csv",
        "translation_power.csv",
        "translation_forest.png",
        "translation_power.png",
    }
