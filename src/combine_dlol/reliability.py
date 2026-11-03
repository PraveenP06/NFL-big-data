from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import rankdata
from statsmodels.regression.mixed_linear_model import MixedLMResults


@dataclass(frozen=True)
class MixedModelICC:
    player_variance: float
    residual_variance: float
    icc: float


@dataclass(frozen=True)
class SplitHalfReliability:
    correlations: pd.DataFrame
    representative: pd.DataFrame

    @property
    def median(self) -> float:
        return float(self.correlations["spearman_brown"].median())


def _spearman_brown(correlation: float) -> float:
    if not np.isfinite(correlation) or np.isclose(correlation, -1.0):
        return float("nan")
    return float(2.0 * correlation / (1.0 + correlation))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.isclose(np.std(first), 0.0) or np.isclose(np.std(second), 0.0):
        return float("nan")
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = np.sqrt(np.dot(centered_first, centered_first) * np.dot(centered_second, centered_second))
    return float(np.dot(centered_first, centered_second) / denominator)


def _eligible_groups(
    residuals: pd.DataFrame, id_col: str, value_col: str, min_n: int
) -> list[tuple[object, np.ndarray]]:
    if min_n < 2:
        raise ValueError("min_n must be at least 2")
    data = residuals.loc[:, [id_col, value_col]].dropna().copy()
    counts = data.groupby(id_col, sort=False).size()
    eligible_ids = counts[counts.ge(min_n)].index
    selected = data.loc[data[id_col].isin(eligible_ids)]
    groups: list[tuple[object, np.ndarray]] = []
    for player, values in selected.groupby(id_col, sort=False)[value_col]:
        array = values.to_numpy(dtype=float)
        if len(array) % 2:
            array = array[:-1]
        if len(array) >= 2:
            groups.append((player, array))
    if len(groups) < 3:
        raise ValueError("at least three eligible players are required")
    return groups


def split_half_reliability(
    residuals: pd.DataFrame,
    id_col: str,
    value_col: str,
    min_n: int,
    n_splits: int = 2000,
    seed: int = 20261103,
) -> SplitHalfReliability:
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    groups = _eligible_groups(residuals, id_col, value_col, min_n)
    rng = np.random.default_rng(seed)
    player_ids = np.array([player for player, _ in groups], dtype=object)
    rows: list[dict[str, float]] = []
    halves: list[tuple[np.ndarray, np.ndarray]] = []
    for split in range(n_splits):
        split_values = [rng.permutation(values) for _, values in groups]
        first = np.array([values[: len(values) // 2].mean() for values in split_values], dtype=float)
        second = np.array([values[len(values) // 2 :].mean() for values in split_values], dtype=float)
        correlation = _correlation(rankdata(first), rankdata(second))
        rows.append({"split": float(split), "spearman": correlation, "spearman_brown": _spearman_brown(correlation)})
        halves.append((first, second))
    correlations = pd.DataFrame(rows)
    finite = correlations["spearman_brown"].dropna()
    if finite.empty:
        representative = pd.DataFrame(columns=[id_col, "half_a", "half_b"])
    else:
        target = float(finite.median())
        index = int(np.nanargmin((correlations["spearman_brown"] - target).abs().to_numpy(dtype=float)))
        first, second = halves[index]
        representative = pd.DataFrame({id_col: player_ids, "half_a": first, "half_b": second})
    return SplitHalfReliability(correlations=correlations, representative=representative)


def player_bootstrap_ci(
    split_result: SplitHalfReliability,
    n_bootstrap: int = 1000,
    seed: int = 20261104,
) -> tuple[float, float]:
    values = split_result.representative.loc[:, ["half_a", "half_b"]].to_numpy(dtype=float)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = values[rng.integers(0, len(values), len(values))]
        estimates[index] = _spearman_brown(_correlation(rankdata(sample[:, 0]), rankdata(sample[:, 1])))
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return float("nan"), float("nan")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper)


def mixed_model_icc(residuals: pd.DataFrame, id_col: str, value_col: str) -> MixedModelICC:
    data = residuals.loc[:, [id_col, value_col]].dropna().copy()
    if data[id_col].nunique() < 2:
        raise ValueError("at least two players are required")
    response = data[value_col].to_numpy(dtype=float)
    groups = pd.factorize(data[id_col], sort=False)[0]
    model = sm.MixedLM(response, np.ones((len(data), 1)), groups=groups)
    fitted: MixedLMResults | None = None
    failures: list[Exception] = []
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = model.fit(reml=True, method=method, disp=False)
            fitted = candidate
            if candidate.converged:
                break
        except (np.linalg.LinAlgError, ValueError) as error:
            failures.append(error)
    if fitted is None:
        raise RuntimeError(f"MixedLM did not fit: {failures}")
    result = fitted
    player_variance = max(float(np.asarray(result.cov_re)[0, 0]), 0.0)
    residual_variance = max(float(result.scale), 0.0)
    total = player_variance + residual_variance
    return MixedModelICC(player_variance, residual_variance, player_variance / total if total else float("nan"))


def mean_reliability(icc: float, n: int | np.ndarray) -> float | np.ndarray:
    counts = np.asarray(n, dtype=float)
    reliability = counts * icc / (1.0 + (counts - 1.0) * icc)
    return float(reliability) if counts.ndim == 0 else reliability


def permutation_null(
    residuals: pd.DataFrame,
    id_col: str,
    value_col: str,
    min_n: int,
    n_permutations: int = 500,
    n_splits: int = 100,
    seed: int = 20261105,
) -> np.ndarray:
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    rng = np.random.default_rng(seed)
    labels = residuals[id_col].to_numpy()
    values = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        shuffled = residuals.assign(**{id_col: rng.permutation(labels)})
        values[index] = split_half_reliability(
            shuffled, id_col, value_col, min_n, n_splits=n_splits, seed=int(rng.integers(0, 2**32 - 1))
        ).median
    return values


def empirical_bayes_shrink(
    residuals: pd.DataFrame,
    id_col: str,
    value_col: str,
    mixed_model: MixedModelICC,
    min_n: int = 1,
) -> pd.DataFrame:
    data = residuals.loc[:, [id_col, value_col]].dropna().copy()
    summary = data.groupby(id_col, as_index=False)[value_col].agg(["size", "mean"]).reset_index()
    summary = summary.rename(columns={"size": "n", "mean": "raw_mean"})
    summary = summary.loc[summary["n"].ge(min_n)].copy()
    grand_mean = float(data[value_col].mean())
    n_values = summary["n"].to_numpy(dtype=float)
    denominator = n_values * mixed_model.player_variance + mixed_model.residual_variance
    reliabilities = np.divide(
        n_values * mixed_model.player_variance,
        denominator,
        out=np.zeros_like(n_values),
        where=denominator > 0,
    )
    summary["grand_mean"] = grand_mean
    summary["reliability"] = reliabilities
    summary["posterior_mean"] = grand_mean + reliabilities * (summary["raw_mean"] - grand_mean)
    columns = [id_col, "n", "raw_mean", "grand_mean", "reliability", "posterior_mean"]
    return summary.loc[:, columns].reset_index(drop=True)


def adjusted_residuals(
    df: pd.DataFrame,
    outcome: str,
    covariates: Sequence[str],
    categorical: Sequence[str],
) -> pd.DataFrame:
    required = [outcome, *covariates, *categorical]
    data = df.dropna(subset=required).copy()
    categorical_set = set(categorical)
    variables = [*covariates, *categorical]
    terms = [f"C(Q('{column}'))" if column in categorical_set else f"Q('{column}')" for column in variables]
    formula = f"Q('{outcome}') ~ " + (" + ".join(terms) if terms else "1")
    model = smf.ols(formula, data=data).fit()
    return data.assign(**{f"{outcome}_residual": model.resid.to_numpy()})


def verdict(median: float, lower_ci: float) -> str:
    if median >= 0.50 and lower_ci >= 0.30:
        return "adequate"
    if median >= 0.30:
        return "weak"
    return "poor"
