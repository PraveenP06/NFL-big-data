from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from combine_dlol.reliability import (
    adjusted_residuals,
    empirical_bayes_shrink,
    mean_reliability,
    mixed_model_icc,
    permutation_null,
    player_bootstrap_ci,
    split_half_reliability,
    verdict,
)


def _planted_data(seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    players = np.repeat(np.arange(48), 12)
    effects = rng.normal(0.0, 1.0, 48)
    values = np.repeat(effects, 12) + rng.normal(0.0, 0.75, len(players))
    return pd.DataFrame({"player": players, "value": values})


def test_reliability_recovers_planted_player_effects() -> None:
    data = _planted_data()
    result = split_half_reliability(data, "player", "value", min_n=10, n_splits=200, seed=3)
    model = mixed_model_icc(data, "player", "value")
    lower, upper = player_bootstrap_ci(result, n_bootstrap=300, seed=4)
    assert result.median > 0.80
    assert model.icc > 0.50
    assert lower > 0.65
    assert upper >= lower


def test_noise_and_permutation_null_are_near_zero() -> None:
    rng = np.random.default_rng(8)
    data = pd.DataFrame({"player": np.repeat(np.arange(50), 10), "value": rng.normal(size=500)})
    result = split_half_reliability(data, "player", "value", min_n=10, n_splits=150, seed=9)
    null = permutation_null(data, "player", "value", min_n=10, n_permutations=80, n_splits=20, seed=10)
    assert abs(result.median) < 0.30
    assert abs(float(np.nanmedian(null))) < 0.20


def test_split_halves_are_deterministic_and_drop_odd_observations() -> None:
    data = pd.DataFrame({"player": np.repeat(np.arange(4), 5), "value": np.arange(20, dtype=float)})
    first = split_half_reliability(data, "player", "value", min_n=5, n_splits=20, seed=12)
    second = split_half_reliability(data, "player", "value", min_n=5, n_splits=20, seed=12)
    pd.testing.assert_frame_equal(first.correlations, second.correlations)


def test_empirical_bayes_shrinks_low_counts_more() -> None:
    rng = np.random.default_rng(11)
    low = pd.DataFrame({"player": ["low"] * 2, "value": [2.0, 2.0]})
    high = pd.DataFrame({"player": ["high"] * 40, "value": 2.0 + rng.normal(0.0, 0.2, 40)})
    rest = pd.DataFrame({"player": np.repeat(np.arange(24), 10), "value": rng.normal(0.0, 1.0, 240)})
    data = pd.concat([low, high, rest], ignore_index=True)
    model = mixed_model_icc(data, "player", "value")
    scores = empirical_bayes_shrink(data, "player", "value", model, min_n=2).set_index("player")
    assert abs(float(scores.loc["low", "posterior_mean"]) - float(scores.loc["low", "grand_mean"])) < abs(
        float(scores.loc["high", "posterior_mean"]) - float(scores.loc["high", "grand_mean"])
    )
    assert float(scores.loc["low", "reliability"]) < float(scores.loc["high", "reliability"])
    assert mean_reliability(model.icc, 2) < mean_reliability(model.icc, 40)


def test_adjustment_and_verdict_thresholds() -> None:
    data = pd.DataFrame(
        {
            "outcome": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "speed": [0, 1, 2, 3, 4, 5],
            "group": ["a", "a", "a", "b", "b", "b"],
        }
    )
    adjusted = adjusted_residuals(data, "outcome", ["speed"], ["group"])
    assert "outcome_residual" in adjusted
    assert verdict(0.50, 0.30) == "adequate"
    assert verdict(0.49, 0.90) == "weak"
    assert verdict(0.80, 0.29) == "weak"
    assert verdict(0.29, 0.90) == "poor"


def test_mixed_model_icc_rejects_false_boundary_fit_from_one_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _planted_data()
    real_fit = sm.MixedLM.fit

    def flaky_fit(self: sm.MixedLM, *args: object, **kwargs: object) -> object:
        result = real_fit(self, *args, **kwargs)
        if kwargs.get("method") != "lbfgs":
            return result
        return SimpleNamespace(
            cov_re=np.zeros((1, 1)), scale=float(result.scale), converged=True, llf=float(result.llf) - 50.0
        )

    honest = mixed_model_icc(data, "player", "value")
    monkeypatch.setattr(sm.MixedLM, "fit", flaky_fit)
    repaired = mixed_model_icc(data, "player", "value")
    assert honest.icc > 0.3
    assert repaired.icc == pytest.approx(honest.icc, rel=1e-3)


def test_empirical_bayes_keeps_player_differences_when_variance_is_small_but_real() -> None:
    rng = np.random.default_rng(5)
    players = np.repeat(np.arange(60), 300)
    effects = rng.normal(0.0, 0.034, 60)
    values = np.repeat(effects, 300) + rng.normal(0.0, 0.2175, len(players))
    data = pd.DataFrame({"player": players, "value": values})
    scores = empirical_bayes_shrink(data, "player", "value", mixed_model_icc(data, "player", "value"))
    assert scores["posterior_mean"].std() > 0.01
