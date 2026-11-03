from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

from matplotlib import pyplot as plt

TRADITIONAL_FEATURES = (
    "forty",
    "ten_yd_split",
    "three_cone",
    "short_shuttle",
    "vertical",
    "broad_jump",
    "combine_height",
    "combine_weight",
    "arm_length",
)
OUTCOMES: Mapping[str, tuple[str, ...]] = {
    "DL": (
        "shrunk_d_a",
        "shrunk_d_b_displacement_0_5",
        "shrunk_d_b_displacement_1_0",
        "shrunk_d_b_speed_0_5",
    ),
    "OL": (
        "peak_pressure_probability_allowed_residual",
        "snap_reaction_time_residual",
        "first_0_5s_displacement_residual",
    ),
}


@dataclass(frozen=True)
class Settings:
    repeats: int = 200
    folds: int = 5
    permutations: int = 1000
    permutation_repeats: int = 10
    bootstraps: int = 2000
    power_simulations: int = 200
    power_bootstraps: int = 500
    power_permutations: int = 200
    seed: int = 20261103


@dataclass(frozen=True)
class GroupData:
    group: str
    ids: np.ndarray
    outcomes: pd.DataFrame
    features: Mapping[str, np.ndarray]
    draft_pick: np.ndarray
    draft_year: np.ndarray
    reliability: Mapping[str, float]


@dataclass(frozen=True)
class Operators:
    matrices: tuple[np.ndarray, ...]
    alpha: float
    n_features: int


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _as_float(value: Any) -> float:
    return float(value)


def _feature_names(metadata: pd.DataFrame, group: str, level: str) -> tuple[str, ...]:
    selected = metadata.loc[metadata["group"].eq(group)].copy()
    sensor = selected.loc[selected["family"].eq("sensor")]
    m3 = tuple(sensor.loc[sensor["eligible_for_M3"].astype(bool), "predictor_column"])
    m4 = tuple(sensor.loc[sensor["eligible_for_M4"].astype(bool), "predictor_column"])
    if level == "M0":
        return ()
    if level == "M1":
        return ("forty",)
    if level == "M2":
        return TRADITIONAL_FEATURES
    if level == "M3":
        return (*TRADITIONAL_FEATURES, *m3)
    if level == "M4":
        return (*TRADITIONAL_FEATURES, *m3, *tuple(value for value in m4 if value not in m3))
    raise ValueError(f"unknown model level: {level}")


def _outcome_table(group: str, scores: pd.DataFrame) -> pd.DataFrame:
    if group == "DL":
        values = scores.loc[:, ["nfl_id", *OUTCOMES[group]]].copy()
        values["nfl_id"] = pd.to_numeric(values["nfl_id"], errors="raise").astype("int64")
        return values
    wide = scores.loc[scores["outcome"].isin(OUTCOMES[group]), ["nfl_id", "outcome", "posterior_mean"]].copy()
    wide["nfl_id"] = pd.to_numeric(wide["nfl_id"], errors="raise").astype("int64")
    result = wide.pivot(index="nfl_id", columns="outcome", values="posterior_mean").reset_index()
    return result.loc[:, ["nfl_id", *OUTCOMES[group]]]


def _draft_table(joined: pd.DataFrame) -> pd.DataFrame:
    values = joined.loc[:, ["nfl_id", "draft_overall_pick", "draft_year"]].copy()
    values["nfl_id"] = pd.to_numeric(values["nfl_id"], errors="coerce").astype("Int64")
    values = values.dropna(subset=["nfl_id"]).sort_values("nfl_id")
    return values.groupby("nfl_id", as_index=False).agg({"draft_overall_pick": "first", "draft_year": "first"})


def _reliability(root: Path, group: str) -> dict[str, float]:
    if group == "DL":
        data = pd.read_csv(root / "reports/phase1_dl_reliability.csv")
        rows = data.loc[data["group"].eq("all")]
        return {f"shrunk_{row.outcome}": _as_float(row.split_half_median) for row in rows.itertuples()}
    data = pd.read_csv(root / "reports/phase1_ol_reliability.csv")
    rows = data.loc[data["subgroup"].eq("all")]
    return {str(row.outcome): _as_float(row.split_half_spearman_brown_median) for row in rows.itertuples()}


def load_group_data(root: Path, group: str) -> GroupData:
    predictors = pd.read_parquet(root / "data/processed/combine_predictors.parquet")
    predictors = predictors.loc[predictors["group"].eq(group)].copy()
    predictors["nfl_id"] = pd.to_numeric(predictors["nfl_id"], errors="raise").astype("int64")
    scores_path = "dl_outcomes_scores.parquet" if group == "DL" else "ol_outcomes_scores.parquet"
    outcomes = _outcome_table(group, pd.read_parquet(root / "data/processed" / scores_path))
    merged = outcomes.merge(predictors, on="nfl_id", how="inner", validate="one_to_one")
    draft = _draft_table(pd.read_parquet(root / "data/processed/player_play_joined.parquet")).rename(
        columns={"draft_year": "joined_draft_year"}
    )
    merged = merged.merge(draft, on="nfl_id", how="left", validate="one_to_one").sort_values("nfl_id")
    metadata = pd.read_csv(root / "reports/phase2_combine_predictors.csv")
    feature_values = {
        level: _numeric_frame(merged, _feature_names(metadata, group, level)) for level in ("M1", "M2", "M3", "M4")
    }
    return GroupData(
        group=group,
        ids=merged["nfl_id"].to_numpy(dtype=int),
        outcomes=pd.DataFrame(merged.loc[:, OUTCOMES[group]].reset_index(drop=True)),
        features=feature_values,
        draft_pick=pd.to_numeric(merged["draft_overall_pick"], errors="coerce").to_numpy(dtype=float),
        draft_year=pd.to_numeric(merged.get("draft_year", merged["joined_draft_year"]), errors="coerce").to_numpy(
            dtype=float
        ),
        reliability=_reliability(root, group),
    )


def _processed_design(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_values = np.asarray(train, dtype=float)
    test_values = np.asarray(test, dtype=float)
    missing = np.isnan(train_values)
    medians = np.nanmedian(train_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_filled = np.where(missing, medians, train_values)
    test_filled = np.where(np.isnan(test_values), medians, test_values)
    indicator_columns = np.flatnonzero(np.any(missing, axis=0))
    if len(indicator_columns):
        train_filled = np.column_stack((train_filled, missing[:, indicator_columns].astype(float)))
        test_filled = np.column_stack((test_filled, np.isnan(test_values[:, indicator_columns]).astype(float)))
    mean = train_filled.mean(axis=0)
    scale = train_filled.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (train_filled - mean) / scale, (test_filled - mean) / scale


def prediction_operator(
    features: np.ndarray, train_index: np.ndarray, test_index: np.ndarray, alpha: float = 10.0
) -> np.ndarray:
    count = len(features)
    if features.shape[1] == 0:
        operator = np.zeros((len(test_index), count), dtype=float)
        operator[:, train_index] = 1.0 / len(train_index)
        return operator
    train, test = _processed_design(features[train_index], features[test_index])
    train_centered = train - train.mean(axis=0)
    test_centered = test - train.mean(axis=0)
    gram = train_centered.T @ train_centered + alpha * np.eye(train_centered.shape[1])
    local = np.full((len(test_index), len(train_index)), 1.0 / len(train_index))
    local += test_centered @ np.linalg.solve(gram, train_centered.T)
    operator = np.zeros((len(test_index), count), dtype=float)
    operator[:, train_index] = local
    return operator


def build_operators(features: np.ndarray, repeats: int, folds: int, seed: int, alpha: float = 10.0) -> Operators:
    if len(features) < folds:
        raise ValueError("players must be at least the number of folds")
    matrices: list[np.ndarray] = []
    for repeat in range(repeats):
        matrix = np.zeros((len(features), len(features)), dtype=float)
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for train, test in splitter.split(features):
            matrix[test] = prediction_operator(features, train, test, alpha)
        matrices.append(matrix)
    return Operators(tuple(matrices), alpha, features.shape[1])


def averaged_squared_errors(operators: Operators, outcome: np.ndarray) -> np.ndarray:
    values = np.asarray(outcome, dtype=float)
    errors = np.stack([(matrix @ values - values) ** 2 for matrix in operators.matrices])
    return errors.mean(axis=0)


def delta_r2(base_errors: np.ndarray, model_errors: np.ndarray, outcome: np.ndarray) -> float:
    sst = float(np.sum((outcome - outcome.mean()) ** 2))
    if sst <= 0.0:
        return float("nan")
    return float((base_errors.sum() - model_errors.sum()) / sst)


def delta_r2_batch(base_errors: np.ndarray, model_errors: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    numerator = base_errors.sum(axis=0) - model_errors.sum(axis=0)
    centered = outcomes - outcomes.mean(axis=0, keepdims=True)
    denominator = (centered**2).sum(axis=0)
    return numerator / denominator


def bootstrap_delta(
    base_errors: np.ndarray, model_errors: np.ndarray, outcome: np.ndarray, samples: int, seed: int
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(outcome), size=(samples, len(outcome)))
    numerator = (base_errors[indices] - model_errors[indices]).sum(axis=1)
    sampled = outcome[indices]
    denominator = ((sampled - sampled.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    values = numerator / denominator
    quantiles = np.quantile(values, [0.025, 0.975])
    return float(quantiles[0]), float(quantiles[1])


def joint_row_permutation(outcomes: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    return outcomes[generator.permutation(len(outcomes))]


def family_maxima(
    groups: Mapping[str, GroupData], operators: Mapping[str, Mapping[str, Operators]], settings: Settings
) -> np.ndarray:
    generator = np.random.default_rng(settings.seed + 1)
    maxima = np.empty(settings.permutations, dtype=float)
    for permutation in range(settings.permutations):
        maximum = -np.inf
        for group, data in groups.items():
            shuffled = joint_row_permutation(data.outcomes.to_numpy(dtype=float), generator)
            errors = {
                level: np.column_stack(
                    [
                        averaged_squared_errors(operators[group][level], shuffled[:, column])
                        for column in range(shuffled.shape[1])
                    ]
                )
                for level in ("M2", "M3", "M4")
            }
            for column in range(shuffled.shape[1]):
                for level in ("M3", "M4"):
                    maximum = max(
                        maximum, delta_r2(errors["M2"][:, column], errors[level][:, column], shuffled[:, column])
                    )
        maxima[permutation] = maximum
    return maxima


def family_p_value(maxima: np.ndarray, observed: float) -> float:
    return float((1 + np.count_nonzero(maxima >= observed)) / (len(maxima) + 1))


def _r2(errors: np.ndarray, outcome: np.ndarray) -> float:
    sst = float(np.sum((outcome - outcome.mean()) ** 2))
    return float(1.0 - errors.sum() / sst) if sst > 0.0 else float("nan")


def primary_results(groups: Mapping[str, GroupData], settings: Settings) -> pd.DataFrame:
    operators = {
        group: {
            level: build_operators(data.features[level], settings.repeats, settings.folds, settings.seed)
            for level in ("M0", "M1", "M2", "M3", "M4")
            if level != "M0"
        }
        for group, data in groups.items()
    }
    for group, data in groups.items():
        operators[group]["M0"] = build_operators(
            np.empty((len(data.ids), 0)), settings.repeats, settings.folds, settings.seed
        )
    permutation_operators = {
        group: {
            level: build_operators(
                data.features[level], settings.permutation_repeats, settings.folds, settings.seed + 10000
            )
            for level in ("M2", "M3", "M4")
        }
        for group, data in groups.items()
    }
    maxima = family_maxima(groups, permutation_operators, settings)
    rows: list[dict[str, object]] = []
    for group, data in groups.items():
        for outcome_name in OUTCOMES[group]:
            outcome = data.outcomes[outcome_name].to_numpy(dtype=float)
            errors = {
                level: averaged_squared_errors(operators[group][level], outcome)
                for level in ("M0", "M1", "M2", "M3", "M4")
            }
            for level in ("M3", "M4"):
                value = delta_r2(errors["M2"], errors[level], outcome)
                lower, upper = bootstrap_delta(
                    errors["M2"], errors[level], outcome, settings.bootstraps, settings.seed + len(rows)
                )
                p_value = family_p_value(maxima, value)
                rows.append(
                    {
                        "group": group,
                        "outcome": outcome_name,
                        "comparison": f"{level}_vs_M2",
                        "delta_r2": value,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "p_fwer": p_value,
                        **{
                            f"r2_{model.lower()}": _r2(errors[model], outcome)
                            for model in ("M0", "M1", "M2", "M3", "M4")
                        },
                        "n_players": len(outcome),
                        "n_features": data.features[level].shape[1],
                    }
                )
    return pd.DataFrame(rows)


def naive_ridge_predictions(
    features: np.ndarray, train_index: np.ndarray, test_index: np.ndarray, outcome: np.ndarray, alpha: float = 10.0
) -> np.ndarray:
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    pipeline.fit(features[train_index], outcome[train_index])
    return pipeline.predict(features[test_index])


def _repeated_r2(
    estimator: Any, features: np.ndarray, outcome: np.ndarray, repeats: int, folds: int, seed: int
) -> float:
    errors = np.zeros(len(outcome), dtype=float)
    for repeat in range(repeats):
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        predicted = np.empty(len(outcome), dtype=float)
        for train, test in splitter.split(features):
            model = clone(estimator)
            model.fit(features[train], outcome[train])
            predicted[test] = model.predict(features[test])
        errors += (predicted - outcome) ** 2 / repeats
    return _r2(errors, outcome)


def _draft_features(features: np.ndarray, draft: np.ndarray) -> np.ndarray:
    return np.column_stack((features, draft))


def _leave_one_draft_out(features: np.ndarray, outcome: np.ndarray, years: np.ndarray, alpha: float) -> float:
    predicted = np.full(len(outcome), np.nan)
    for year in np.unique(years[np.isfinite(years)]):
        test = np.flatnonzero(years == year)
        train = np.flatnonzero(years != year)
        if len(test) and len(train):
            predicted[test] = naive_ridge_predictions(features, train, test, outcome, alpha)
    observed = np.isfinite(predicted)
    return _r2((predicted[observed] - outcome[observed]) ** 2, outcome[observed])


def descriptive_results(groups: Mapping[str, GroupData], settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, data in groups.items():
        operators = {
            level: build_operators(data.features[level], settings.repeats, settings.folds, settings.seed)
            for level in ("M0", "M1", "M2", "M3", "M4")
            if level != "M0"
        }
        operators["M0"] = build_operators(np.empty((len(data.ids), 0)), settings.repeats, settings.folds, settings.seed)
        complete = np.isfinite(data.features["M1"][:, 0])
        for outcome_name in OUTCOMES[group]:
            outcome = data.outcomes[outcome_name].to_numpy(dtype=float)
            errors = {level: averaged_squared_errors(operators[level], outcome) for level in operators}
            for level in ("M1", "M2"):
                rows.append(
                    _description(
                        group, outcome_name, f"{level}_vs_M0_delta_r2", delta_r2(errors["M0"], errors[level], outcome)
                    )
                )
            for alpha in (1.0, 100.0):
                alternative_m2 = build_operators(
                    data.features["M2"], settings.repeats, settings.folds, settings.seed, alpha
                )
                alternative_errors = averaged_squared_errors(alternative_m2, outcome)
                for level in ("M3", "M4"):
                    alternative = build_operators(
                        data.features[level], settings.repeats, settings.folds, settings.seed, alpha
                    )
                    rows.append(
                        _description(
                            group,
                            outcome_name,
                            f"{level}_alpha_{int(alpha)}_vs_M2_delta_r2",
                            delta_r2(alternative_errors, averaged_squared_errors(alternative, outcome), outcome),
                        )
                    )
            elastic = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(),
                ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=settings.seed),
            )
            boosting = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                GradientBoostingRegressor(random_state=settings.seed),
            )
            rows.append(
                _description(
                    group,
                    outcome_name,
                    "elastic_net_m4_r2",
                    _repeated_r2(elastic, data.features["M4"], outcome, 20, settings.folds, settings.seed),
                )
            )
            rows.append(
                _description(
                    group,
                    outcome_name,
                    "gradient_boosting_m4_r2",
                    _repeated_r2(boosting, data.features["M4"], outcome, 20, settings.folds, settings.seed),
                )
            )
            for level in ("M2", "M4"):
                with_draft = build_operators(
                    _draft_features(data.features[level], data.draft_pick),
                    settings.repeats,
                    settings.folds,
                    settings.seed,
                )
                rows.append(
                    _description(
                        group,
                        outcome_name,
                        f"draft_added_to_{level}_delta_r2",
                        delta_r2(errors[level], averaged_squared_errors(with_draft, outcome), outcome),
                    )
                )
                rows.append(
                    _description(
                        group,
                        outcome_name,
                        f"leave_one_draft_class_out_{level}_r2",
                        _leave_one_draft_out(data.features[level], outcome, data.draft_year, 10.0),
                    )
                )
            if int(complete.sum()) >= settings.folds:
                local = {
                    level: build_operators(
                        data.features[level][complete], settings.repeats, settings.folds, settings.seed
                    )
                    for level in ("M2", "M3", "M4")
                }
                for level in ("M3", "M4"):
                    rows.append(
                        _description(
                            group,
                            outcome_name,
                            f"complete_case_{level}_vs_M2_delta_r2",
                            delta_r2(
                                averaged_squared_errors(local["M2"], outcome[complete]),
                                averaged_squared_errors(local[level], outcome[complete]),
                                outcome[complete],
                            ),
                        )
                    )
            rows.append(_description(group, outcome_name, "outcome_reliability", data.reliability[outcome_name]))
    return pd.DataFrame(rows)


def _description(group: str, outcome: str, metric: str, value: float) -> dict[str, object]:
    return {"group": group, "outcome": outcome, "metric": metric, "value": value}


def _orthogonal_sensor_component(m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
    sensor = m3[:, m2.shape[1] :]
    filled_sensor, _ = _processed_design(sensor, sensor)
    filled_m2, _ = _processed_design(m2, m2)
    residual = filled_sensor - filled_m2 @ np.linalg.lstsq(filled_m2, filled_sensor, rcond=None)[0]
    component = np.linalg.svd(residual, full_matrices=False)[0][:, 0]
    scale = component.std()
    return component / scale if scale > 0.0 else component


def power_results(groups: Mapping[str, GroupData], settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    planted_values = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
    for group, data in groups.items():
        operators = {
            level: build_operators(data.features[level], settings.permutation_repeats, settings.folds, settings.seed)
            for level in ("M2", "M3")
        }
        signal = _orthogonal_sensor_component(data.features["M2"], data.features["M3"])
        for outcome_name in OUTCOMES[group]:
            observed = data.outcomes[outcome_name].to_numpy(dtype=float)
            residual = observed - np.mean([matrix @ observed for matrix in operators["M2"].matrices], axis=0)
            for planted in planted_values:
                detected = 0
                generator = np.random.default_rng(settings.seed + int(planted * 1000))
                for simulation in range(settings.power_simulations):
                    noise = residual[generator.permutation(len(residual))]
                    planted_outcome = noise + np.sqrt(planted) * np.std(noise) * signal
                    base = averaged_squared_errors(operators["M2"], planted_outcome)
                    model = averaged_squared_errors(operators["M3"], planted_outcome)
                    delta = delta_r2(base, model, planted_outcome)
                    lower, _ = bootstrap_delta(
                        base, model, planted_outcome, settings.power_bootstraps, settings.seed + simulation
                    )
                    shuffled = np.column_stack(
                        [
                            joint_row_permutation(planted_outcome[:, None], generator).ravel()
                            for _ in range(settings.power_permutations)
                        ]
                    )
                    nulls = delta_r2_batch(
                        averaged_squared_errors(operators["M2"], shuffled),
                        averaged_squared_errors(operators["M3"], shuffled),
                        shuffled,
                    )
                    p_value = (1 + np.count_nonzero(nulls >= delta)) / (len(nulls) + 1)
                    detected += int(delta > 0.0 and lower > 0.0 and p_value < 0.05)
                rows.append(
                    {
                        "group": group,
                        "outcome": outcome_name,
                        "reliability": data.reliability[outcome_name],
                        "planted_partial_r2": planted,
                        "detection_probability": detected / settings.power_simulations,
                        "comparison": "M3_vs_M2_one_comparison",
                    }
                )
    return pd.DataFrame(rows)


def forest_plot(results: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, max(4, len(results) * 0.42)))
    values = results.reset_index(drop=True)
    y = np.arange(len(values))
    success = (
        (values["delta_r2"] > 0.0)
        & ((values["ci_lower"] > 0.0) | (values["ci_upper"] < 0.0))
        & (values["p_fwer"] < 0.05)
    )
    colors = np.where(success, "#18864b", "#8b8b8b")
    delta = values["delta_r2"].to_numpy(dtype=float)
    lower = values["ci_lower"].to_numpy(dtype=float)
    upper = values["ci_upper"].to_numpy(dtype=float)
    for index, color in enumerate(colors):
        axis.errorbar(
            delta[index],
            y[index],
            xerr=np.array(
                [
                    [delta[index] - lower[index]],
                    [upper[index] - delta[index]],
                ]
            ),
            fmt="none",
            ecolor=color,
        )
    axis.scatter(delta, y, c=colors)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(y, values["group"] + " " + values["outcome"] + " " + values["comparison"])
    axis.set_xlabel("Out-of-sample delta R2")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def power_plot(results: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for (group, outcome), data in results.groupby(["group", "outcome"]):
        axis.plot(data["planted_partial_r2"], data["detection_probability"], marker="o", label=f"{group} {outcome}")
    axis.set_xlabel("Planted partial R2")
    axis.set_ylabel("Detection probability")
    axis.set_ylim(0.0, 1.0)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(root: Path, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = {group: load_group_data(root, group) for group in OUTCOMES}
    results = primary_results(groups, settings)
    descriptive = descriptive_results(groups, settings)
    power = power_results(groups, settings)
    reports = root / "reports"
    results.to_csv(reports / "translation_results.csv", index=False)
    descriptive.to_csv(reports / "translation_descriptive.csv", index=False)
    power.to_csv(reports / "translation_power.csv", index=False)
    forest_plot(results, reports / "translation_forest.png")
    power_plot(power, reports / "translation_power.png")
    return results, descriptive, power


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    run(args.root.resolve(), Settings())


if __name__ == "__main__":
    main()
