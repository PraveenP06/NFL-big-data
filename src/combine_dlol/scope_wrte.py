from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as t_distribution

from combine_dlol.reliability import (
    empirical_bayes_shrink,
    mixed_model_icc,
    player_bootstrap_ci,
    split_half_reliability,
    verdict,
)

OFFICIAL = (
    "forty",
    "ten_yd_split",
    "three_cone",
    "short_shuttle",
    "vertical",
    "broad_jump",
    "combine_height",
    "combine_weight",
    "arm_length",
    "wing_span",
    "hand_size",
    "bench_reps",
)
ARCHIVE_FEATURES = Path("/Users/praveen/dev/NFL-big-data-archive/processed/combine_features.parquet")
RESULT_COLUMNS = (
    "scope",
    "group",
    "outcome",
    "predictor",
    "family",
    "n",
    "r_raw",
    "r_adj",
    "ci_lo",
    "ci_hi",
    "p",
    "q_scope",
    "oriented_r",
    "direction",
    "r_disc_2023_24",
    "r_hold_2025",
    "r_2023",
    "r_2024",
    "r_2025",
)
OUTCOMES = (
    "separation_at_pass_forward",
    "target_rate",
    "rec_yards_per_route",
    "catch_rate",
    "yards_after_catch",
    "yac_over_expected",
    "expected_points_added",
    "separation_man",
    "separation_zone",
)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    if len(p_values) == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    output = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank, index in zip(range(len(p_values), 0, -1), order[::-1], strict=True):
        running = min(running, float(p_values[index]) * len(p_values) / rank)
        output[index] = running
    return output


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: Sequence[np.ndarray]) -> float:
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    design = np.column_stack([np.ones(len(x)), *[rankdata(value) for value in controls]])
    x_residual = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    y_residual = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def build_composite(data: pd.DataFrame, components: Sequence[tuple[str, int]]) -> pd.Series:
    values: list[pd.Series] = []
    for column, sign in components:
        if column not in data:
            continue
        series = pd.to_numeric(data[column], errors="coerce")
        standard_deviation = float(series.std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation == 0.0:
            continue
        values.append(sign * (series - series.mean()) / standard_deviation)
    if not values:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.concat(values, axis=1).mean(axis=1, skipna=True)


def join_player_scores(predictors: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if predictors["nfl_id"].duplicated().any() or outcomes["nfl_id"].duplicated().any():
        raise ValueError("predictors and outcomes must have one row per player")
    return predictors.merge(outcomes, on="nfl_id", how="inner", validate="one_to_one")


def orientation_label(oriented_r: float) -> str:
    if not np.isfinite(oriented_r):
        return "unclassified"
    return "expected" if oriented_r > 0.0 else "opposite"


def _root(root: Path | None) -> Path:
    return Path.cwd() if root is None else root


def _metric_column(drill: str, metric: str) -> str:
    prefix = "".join(character.lower() if character.isalnum() else "_" for character in drill).strip("_")
    return f"{prefix}_{metric}"


def predictor_orientation(column: str) -> int:
    if column.startswith("WR_"):
        return 1
    if column in {"combine_height", "combine_weight", "arm_length", "wing_span", "hand_size"}:
        return 0
    if column in {"forty", "ten_yd_split", "three_cone", "short_shuttle"} or "time" in column:
        return -1
    if any(value in column for value in ("speed", "acceleration", "vertical", "broad_jump", "bench_reps", "distance")):
        return 1
    return 0


def _pair_repeatability(values: pd.DataFrame) -> tuple[int, float]:
    pairs: list[tuple[float, float]] = []
    generator = np.random.default_rng(20261103)
    for _, player_values in values.groupby("nfl_id", observed=True)["value"]:
        finite = player_values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) >= 2:
            selected = generator.choice(finite, size=2, replace=False)
            pairs.append((float(selected[0]), float(selected[1])))
    if len(pairs) < 25:
        return len(pairs), float("nan")
    matrix = np.asarray(pairs, dtype=float)
    correlation = float(spearmanr(matrix[:, 0], matrix[:, 1]).statistic)
    if not np.isfinite(correlation) or correlation <= -1.0:
        return len(pairs), float("nan")
    return len(pairs), float(2.0 * correlation / (1.0 + correlation))


def _sensor_attempts(root: Path, identifiers: set[str]) -> pd.DataFrame:
    cache = root / "data/processed/scope_wrte_sensor_attempts.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from combine_dlol.combine_predictors import attempt_kinematics

    segments: list[pd.DataFrame] = []
    tracking = root / "data/raw/combine_tracking.csv"
    usecols = ["nfl_id", "entity_type", "time", "drill_name", "attempt", "x", "y"]
    for chunk in pd.read_csv(tracking, usecols=usecols, chunksize=100_000, dtype={"nfl_id": "string"}):
        selected = chunk.loc[(chunk["entity_type"] == "PLAYER") & chunk["nfl_id"].isin(identifiers)].copy()
        if selected.empty:
            continue
        selected["seconds"] = (
            pd.to_datetime(selected["time"], utc=True, errors="coerce") - pd.Timestamp("1970-01-01", tz="UTC")
        ).dt.total_seconds()
        selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
        selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
        selected = selected.dropna(subset=["seconds", "x", "y", "attempt"])
        segments.append(selected.loc[:, ["nfl_id", "drill_name", "attempt", "seconds", "x", "y"]])
    selected_data = pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, attempt in selected_data.groupby(["nfl_id", "drill_name", "attempt"], observed=True, sort=False):
        player, drill, attempt_id = keys
        values = attempt_kinematics(
            attempt["seconds"].to_numpy(dtype=float),
            attempt["x"].to_numpy(dtype=float),
            attempt["y"].to_numpy(dtype=float),
            str(drill),
        )
        for metric, value in values.items():
            rows.append(
                {
                    "nfl_id": str(player),
                    "drill": str(drill),
                    "attempt": str(attempt_id),
                    "metric": metric,
                    "value": value,
                }
            )
    result = pd.DataFrame(rows, columns=["nfl_id", "drill", "attempt", "metric", "value"])
    result.to_parquet(cache, index=False)
    return result


def sensor_selection(attempts: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    selected: list[str] = []
    for (drill, metric), metric_values in attempts.groupby(["drill", "metric"], observed=True, sort=True):
        column = _metric_column(str(drill), str(metric))
        n_players, reliability = _pair_repeatability(metric_values)
        status = "pass" if n_players >= 25 and np.isfinite(reliability) and reliability >= 0.30 else "fail"
        rows.append({"predictor": column, "n_players": n_players, "spearman_brown": reliability, "status": status})
        if status == "pass":
            selected.append(column)
    return pd.DataFrame(rows), selected


def sensor_player_table(attempts: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    names = [
        _metric_column(str(drill), str(metric))
        for drill, metric in zip(attempts["drill"], attempts["metric"], strict=True)
    ]
    labelled = attempts.assign(column=names)
    selected = labelled.loc[labelled["column"].isin(columns)]
    table = selected.pivot_table(index="nfl_id", columns="column", values="value", aggfunc="mean")
    return table.reset_index().rename_axis(columns=None)


def _load_predictors(root: Path, source: Path = ARCHIVE_FEATURES) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    archive = pd.read_parquet(source)
    predictors = archive.loc[archive["combine_position"].isin(["WR", "TE"])].copy()
    predictors["nfl_id"] = predictors["nfl_id"].astype(str)
    if predictors["nfl_id"].duplicated().any():
        raise ValueError("archived Combine features must have one row per nfl_id")
    predictors["role_wr"] = predictors["combine_position"].eq("WR").astype(float)
    attempts = _sensor_attempts(root, set(predictors["nfl_id"]))
    screen, sensor = sensor_selection(attempts)
    sensor_table = sensor_player_table(attempts, sensor)
    predictors = predictors.drop(columns=[c for c in sensor if c in predictors]).merge(
        sensor_table, on="nfl_id", how="left", validate="one_to_one"
    )
    screen.to_csv(root / "data/processed/scope_wrte_sensor_repeatability.csv", index=False)
    available_official = [
        column for column in OFFICIAL if column in predictors and predictors[column].notna().sum() >= 40
    ]
    usable = set(sensor)
    forty_components = (
        ("forty_yard_dash_top_speed", 1),
        ("forty", -1),
        ("forty_yard_dash_peak_acceleration", 1),
        ("forty_yard_dash_burst_speed_1_0s", 1),
        ("forty_yard_dash_sensor_ten_yd_burst_speed", 1),
        ("forty_yard_dash_sensor_ten_yd_time", -1),
    )
    agility_components = (
        ("three_cone", -1),
        ("short_shuttle", -1),
        *tuple((column, 1) for column in sensor if "route" in column and column.endswith("top_speed")),
    )
    explosive_components = (("vertical", 1), ("broad_jump", 1))
    composite_definitions = {
        "WR_SPEED": forty_components,
        "WR_BURST": forty_components[2:],
        "WR_AGILITY": agility_components,
        "WR_EXPLOSIVE": explosive_components,
    }
    composites: list[str] = []
    for name, components in composite_definitions.items():
        allowed = tuple(
            (column, sign)
            for column, sign in components
            if column in predictors and (column in usable or column in OFFICIAL)
        )
        predictors[name] = build_composite(predictors, allowed)
        if predictors[name].notna().sum() >= 40:
            composites.append(name)
    return predictors, [*available_official, *sensor, *composites], screen


def _context_formula(outcome: str, data: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    numeric = [
        column
        for column in ("cushion", "yards_to_go")
        if column in data and data[column].notna().any() and data[column].nunique(dropna=True) > 1
    ]
    categorical = [
        column
        for column in (
            "route_ran",
            "team_coverage_man_zone",
            "team_coverage_type",
            "down",
            "offense_formation",
            "pass_length",
        )
        if column in data and data[column].notna().any() and data[column].nunique(dropna=True) > 1
    ]
    terms = [*numeric, *[f"C(Q('{column}'))" for column in categorical]]
    formula = f"Q('{outcome}') ~ " + (" + ".join(terms) if terms else "1")
    return formula, numeric, categorical


def _outcome_definitions(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame, int]]:
    route = data.copy()
    route["target_rate"] = pd.to_numeric(route["target"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    route["rec_yards_per_route"] = pd.to_numeric(route["rec_yards"], errors="coerce").fillna(0.0)
    target = route.loc[route["target_rate"].eq(1.0)].copy()
    target["catch_rate"] = pd.to_numeric(target["rec_yards"], errors="coerce").notna().astype(float)
    received = target.loc[target["catch_rate"].eq(1.0)].copy()
    received["yac_over_expected"] = pd.to_numeric(received["yards_after_catch"], errors="coerce") - pd.to_numeric(
        received["expected_yards_after_catch"], errors="coerce"
    )
    definitions = [
        ("separation_at_pass_forward", route, 50),
        ("target_rate", route, 50),
        ("rec_yards_per_route", route, 50),
        ("catch_rate", target, 25),
        ("yards_after_catch", received, 25),
        ("yac_over_expected", received, 25),
    ]
    if "expected_points_added" in target:
        definitions.append(("expected_points_added", target, 25))
    man = route.loc[route["team_coverage_man_zone"].astype(str).str.upper().str.contains("MAN")].copy()
    zone = route.loc[route["team_coverage_man_zone"].astype(str).str.upper().str.contains("ZONE")].copy()
    definitions.extend(
        [
            ("separation_man", man.rename(columns={"separation_at_pass_forward": "separation_man"}), 50),
            ("separation_zone", zone.rename(columns={"separation_at_pass_forward": "separation_zone"}), 50),
        ]
    )
    return definitions


def _outcome_scores(root: Path, requested: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "nfl_id",
        "nfl_position",
        "draft_year",
        "route_ran",
        "play_nullified_by_penalty",
        "target",
        "rec_yards",
        "yards_after_catch",
        "expected_yards_after_catch",
        "expected_points_added",
        "separation_at_pass_forward",
        "team_coverage_man_zone",
        "team_coverage_type",
        "cushion",
        "down",
        "yards_to_go",
        "offense_formation",
        "pass_length",
    ]
    source = root / "data/processed/player_play_joined.parquet"
    present = set(pq.ParquetFile(source).schema_arrow.names)
    data = pd.read_parquet(source, columns=[column for column in columns if column in present])
    data = data.loc[
        data["nfl_position"].isin(["WR", "TE"]) & data["route_ran"].notna() & data["play_nullified_by_penalty"].ne("Y")
    ].copy()
    data["nfl_id"] = data["nfl_id"].astype(str)
    scores: list[pd.DataFrame] = []
    gate_rows: list[dict[str, object]] = []
    for outcome, subset, minimum in _outcome_definitions(data):
        if requested is not None and outcome != requested:
            continue
        subset = subset.dropna(subset=[outcome]).copy()
        formula, numeric, categorical = _context_formula(outcome, subset)
        needed = [outcome, *numeric, *categorical]
        model_data = subset.dropna(subset=needed).copy()
        if model_data.empty:
            gate_rows.append(
                {
                    "outcome": outcome,
                    "group": "WRTE",
                    "n_players": 0,
                    "SB_median": np.nan,
                    "lower_ci": np.nan,
                    "upper_ci": np.nan,
                    "icc": np.nan,
                    "verdict": "poor",
                    "formula": formula,
                }
            )
            continue
        residual = smf.ols(formula, data=model_data).fit().resid
        residual_column = f"{outcome}_residual"
        residual_data = model_data.loc[:, ["nfl_id"]].assign(**{residual_column: residual.to_numpy()})
        counts = residual_data.groupby("nfl_id", observed=True).size()
        eligible = residual_data.loc[residual_data["nfl_id"].isin(counts[counts >= minimum].index)].copy()
        if eligible["nfl_id"].nunique() < 3:
            gate_rows.append(
                {
                    "outcome": outcome,
                    "group": "WRTE",
                    "n_players": int(eligible["nfl_id"].nunique()),
                    "SB_median": np.nan,
                    "lower_ci": np.nan,
                    "upper_ci": np.nan,
                    "icc": np.nan,
                    "verdict": "poor",
                    "formula": formula,
                }
            )
            continue
        split = split_half_reliability(eligible, "nfl_id", residual_column, minimum, n_splits=2000, seed=20261103)
        lower, upper = player_bootstrap_ci(split, n_bootstrap=1000, seed=20261104)
        reliability = mixed_model_icc(eligible, "nfl_id", residual_column)
        status = verdict(split.median, lower)
        gate_rows.append(
            {
                "outcome": outcome,
                "group": "WRTE",
                "n_players": int(eligible["nfl_id"].nunique()),
                "SB_median": split.median,
                "lower_ci": lower,
                "upper_ci": upper,
                "icc": reliability.icc,
                "verdict": status,
                "formula": formula,
            }
        )
        if status == "adequate":
            shrunk = empirical_bayes_shrink(eligible, "nfl_id", residual_column, reliability, min_n=minimum)
            scores.append(shrunk.loc[:, ["nfl_id", "posterior_mean"]].rename(columns={"posterior_mean": outcome}))
    gate = pd.DataFrame(gate_rows)
    if not scores:
        return pd.DataFrame(columns=["nfl_id"]), gate
    combined = scores[0]
    for score in scores[1:]:
        combined = combined.merge(score, on="nfl_id", how="outer", validate="one_to_one")
    return combined, gate


def _association_rows(
    data: pd.DataFrame, predictors: Sequence[str], outcomes: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    height_rows: list[dict[str, object]] = []
    for outcome in outcomes:
        for predictor in predictors:
            required = [predictor, outcome, "combine_weight", "role_wr", "combine_height", "draft_year"]
            model = data.loc[:, list(dict.fromkeys(required))].dropna().copy()
            if len(model) < 40 or model[predictor].nunique() < 5:
                continue
            controls = (
                [model["role_wr"].to_numpy(dtype=float)]
                if predictor == "combine_weight"
                else [model["combine_weight"].to_numpy(dtype=float), model["role_wr"].to_numpy(dtype=float)]
            )
            adjusted = partial_spearman(
                model[predictor].to_numpy(dtype=float), model[outcome].to_numpy(dtype=float), controls
            )
            raw = float(spearmanr(model[predictor], model[outcome]).statistic)
            degrees = len(model) - 2 - len(controls)
            statistic = adjusted * np.sqrt(degrees / (1.0 - adjusted**2))
            p_value = float(2.0 * t_distribution.sf(abs(statistic), degrees))
            standard_error = 1.0 / np.sqrt(len(model) - 3 - len(controls))
            low, high = np.tanh(np.arctanh(adjusted) + np.array([-1.96, 1.96]) * standard_error)
            family = "composite" if predictor.startswith("WR_") else "official" if predictor in OFFICIAL else "sensor"
            oriented = adjusted * predictor_orientation(predictor)
            replication: list[float] = []
            for years in ((2023, 2024), (2025,), (2023,), (2024,), (2025,)):
                subset = model.loc[model["draft_year"].isin(years)]
                replication.append(
                    float(spearmanr(subset[predictor], subset[outcome]).statistic)
                    if len(subset) >= 15
                    else float("nan")
                )
            rows.append(
                {
                    "scope": "wrte",
                    "group": "WRTE",
                    "outcome": outcome,
                    "predictor": predictor,
                    "family": family,
                    "n": len(model),
                    "r_raw": raw,
                    "r_adj": adjusted,
                    "ci_lo": float(low),
                    "ci_hi": float(high),
                    "p": p_value,
                    "oriented_r": oriented if predictor_orientation(predictor) else np.nan,
                    "direction": orientation_label(oriented if predictor_orientation(predictor) else np.nan),
                    "r_disc_2023_24": replication[0],
                    "r_hold_2025": replication[1],
                    "r_2023": replication[2],
                    "r_2024": replication[3],
                    "r_2025": replication[4],
                }
            )
            height_controls = (
                [*controls, model["combine_height"].to_numpy(dtype=float)]
                if predictor != "combine_height"
                else controls
            )
            height_rows.append(
                {
                    "outcome": outcome,
                    "predictor": predictor,
                    "r_adj_weight_role_height": partial_spearman(
                        model[predictor].to_numpy(dtype=float), model[outcome].to_numpy(dtype=float), height_controls
                    ),
                    "n": len(model),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_scope"] = benjamini_hochberg(result["p"].to_numpy(dtype=float))
        result = pd.DataFrame(result.loc[:, RESULT_COLUMNS])
    return result, pd.DataFrame(height_rows)


def stage_outcome(root: Path, outcome: str) -> pd.DataFrame:
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome: {outcome}")
    scores, gate = _outcome_scores(root, outcome)
    scores.to_parquet(root / f"data/processed/scope_wrte_scores_{outcome}.parquet", index=False)
    gate.to_csv(root / f"data/processed/scope_wrte_gate_{outcome}.csv", index=False)
    return gate


def combine_stages(root: Path, source: Path = ARCHIVE_FEATURES) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_paths = [root / f"data/processed/scope_wrte_gate_{outcome}.csv" for outcome in OUTCOMES]
    missing = [str(path) for path in gate_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing staged gates: {missing}")
    gate = pd.concat([pd.read_csv(path) for path in gate_paths], ignore_index=True)
    predictors, predictor_columns, _ = _load_predictors(root, source)
    gate.to_csv(root / "reports/scope_wrte_gate.csv", index=False)
    gated = gate.loc[gate["verdict"].eq("adequate"), "outcome"].tolist()
    score_paths = {outcome: root / f"data/processed/scope_wrte_scores_{outcome}.parquet" for outcome in gated}
    missing_scores = [str(path) for path in score_paths.values() if not path.exists()]
    if missing_scores:
        raise FileNotFoundError(f"missing staged scores: {missing_scores}")
    scores = pd.DataFrame({"nfl_id": pd.Series(dtype=str)})
    for path in score_paths.values():
        score = pd.read_parquet(path)
        scores = score if scores.empty else scores.merge(score, on="nfl_id", how="outer", validate="one_to_one")
    combined = join_player_scores(predictors, scores) if gated else pd.DataFrame()
    results, height = (
        _association_rows(combined, predictor_columns, gated)
        if gated
        else (pd.DataFrame(columns=RESULT_COLUMNS), pd.DataFrame())
    )
    results.to_csv(root / "reports/scope_wrte_results.csv", index=False)
    height.to_parquet(root / "data/processed/scope_wrte_height_adjusted.parquet", index=False)
    return results, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage-outcome", choices=OUTCOMES)
    parser.add_argument("--combine", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage_outcome:
        print(stage_outcome(arguments.root, arguments.stage_outcome).to_string(index=False))
    elif arguments.combine:
        results, gate = combine_stages(arguments.root)
        print(gate.to_string(index=False))
        print(f"tests={len(results)} raw_p_lt_0_05={int((results['p'] < 0.05).sum()) if not results.empty else 0}")
    else:
        parser.error("choose --stage-outcome or --combine")


if __name__ == "__main__":
    main()
