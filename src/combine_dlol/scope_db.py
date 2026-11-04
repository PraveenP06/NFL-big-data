from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as student_t

from combine_dlol.combine_predictors import attempt_kinematics
from combine_dlol.reliability import (
    adjusted_residuals,
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
OUTCOMES = (
    "coverage_assignment_rate",
    "man_coverage_assignment_rate",
    "tackle_rate",
    "assist_rate",
    "forced_fumble_rate",
    "tackle_for_loss_rate",
)
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


@dataclass(frozen=True)
class PartialSpearman:
    r_raw: float
    r_adj: float
    ci_lo: float
    ci_hi: float
    p: float


def role_from_position(position: object) -> str:
    return "CB" if str(position) == "CB" else "S"


def _rank_residual(values: np.ndarray, controls: Sequence[np.ndarray]) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), *[rankdata(control) for control in controls]])
    ranked = rankdata(values)
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: Sequence[np.ndarray]) -> PartialSpearman:
    raw = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    x_residual = _rank_residual(x, controls)
    y_residual = _rank_residual(y, controls)
    adjusted = float(np.corrcoef(x_residual, y_residual)[0, 1])
    control_count = len(controls)
    degrees = len(x) - control_count - 2
    if degrees <= 0 or not np.isfinite(adjusted) or abs(adjusted) >= 1:
        p_value = 0.0 if abs(adjusted) == 1 else float("nan")
    else:
        statistic = adjusted * np.sqrt(degrees / (1.0 - adjusted**2))
        p_value = float(2.0 * student_t.sf(abs(statistic), degrees))
    standard_error = 1.0 / np.sqrt(len(x) - 3 - control_count)
    clipped = float(np.clip(adjusted, -0.999999, 0.999999))
    lower, upper = np.tanh(np.arctanh(clipped) + np.array([-1.96, 1.96]) * standard_error)
    return PartialSpearman(raw, adjusted, float(lower), float(upper), p_value)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count == 0:
        return result
    indices = np.flatnonzero(finite)
    order = indices[np.argsort(values[indices])]
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for rank, index in zip(range(count, 0, -1), order[::-1], strict=True):
        running = min(running, float(values[index]) * count / rank)
        adjusted[rank - 1] = running
    result[order] = adjusted
    return result


def join_predictors_scores(predictors: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    left = predictors.copy()
    right = scores.copy()
    left["nfl_id"] = pd.to_numeric(left["nfl_id"], errors="raise").astype(int)
    right["nfl_id"] = pd.to_numeric(right["nfl_id"], errors="raise").astype(int)
    merged = left.merge(right, on="nfl_id", how="inner", validate="one_to_one")
    return merged.assign(role=merged["nfl_position"].map(role_from_position))


def _zscore(values: pd.Series, orientation: float) -> pd.Series:
    standard_deviation = float(values.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation == 0.0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return orientation * (values - float(values.mean())) / standard_deviation


def _composite(frame: pd.DataFrame, components: Sequence[tuple[str, float]], eligible: set[str]) -> pd.Series:
    names = [name for name, _ in components]
    if not names or not set(names).issubset(eligible) or not set(names).issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    scaled = pd.concat([_zscore(frame[name], sign) for name, sign in components], axis=1)
    return scaled.mean(axis=1).where(scaled.notna().all(axis=1))


def add_composites(frame: pd.DataFrame, eligible: set[str]) -> pd.DataFrame:
    result = frame.copy()
    cod_sensor = [
        column
        for column in sorted(eligible)
        if ("back_pedal" in column or "transition" in column)
        and (column.endswith("top_speed") or column.endswith("peak_acceleration"))
    ]
    result["DB_SPEED"] = _composite(
        result,
        (("forty_yard_dash_top_speed", 1.0), ("forty", -1.0)),
        eligible,
    )
    result["DB_BURST"] = _composite(
        result,
        (
            ("forty_yard_dash_peak_acceleration", 1.0),
            ("forty_yard_dash_burst_speed_1_0s", 1.0),
            ("forty_yard_dash_sensor_ten_yd_time", -1.0),
        ),
        eligible,
    )
    result["DB_COD"] = _composite(
        result,
        (("three_cone", -1.0), ("short_shuttle", -1.0), *[(column, 1.0) for column in cod_sensor]),
        eligible,
    )
    result["DB_EXPLOSIVE"] = _composite(result, (("vertical", 1.0), ("broad_jump", 1.0)), eligible)
    result["DB_REACH"] = _composite(result, (("arm_length", 1.0), ("combine_height", 1.0)), eligible)
    return result


def orientation_label(oriented_r: float, outcome: str, predictor: str) -> str:
    if not np.isfinite(oriented_r) or _outcome_orientation(outcome) == 0.0 or _predictor_orientation(predictor) == 0.0:
        return "unclassified"
    return "expected" if oriented_r > 0 else "opposite"


_LOWER_IS_BETTER_OFFICIAL = {"forty", "ten_yd_split", "three_cone", "short_shuttle"}
_LOWER_IS_BETTER_SUFFIXES = ("_time", "_time_in_drill", "_time_to_top_speed")
_HIGHER_IS_BETTER_OFFICIAL = {"vertical", "broad_jump", "bench_reps", "DB_SPEED", "DB_BURST", "DB_COD", "DB_EXPLOSIVE"}
_HIGHER_IS_BETTER_SUFFIXES = (
    "_top_speed",
    "_burst_distance_0_1s",
    "_burst_speed_1_0s",
    "_peak_acceleration",
    "_peak_deceleration",
    "_turn_speed_gain_0_5s",
    "_ten_yd_burst_speed",
)


def _predictor_orientation(predictor: str) -> float:
    if predictor in _LOWER_IS_BETTER_OFFICIAL or predictor.endswith(_LOWER_IS_BETTER_SUFFIXES):
        return -1.0
    if predictor in _HIGHER_IS_BETTER_OFFICIAL or predictor.endswith(_HIGHER_IS_BETTER_SUFFIXES):
        return 1.0
    return 0.0


def _outcome_orientation(outcome: str) -> float:
    return 0.0 if "coverage" in outcome else 1.0


def _raw_spearman(frame: pd.DataFrame, predictor: str, outcome: str) -> float:
    if len(frame) < 15:
        return float("nan")
    return float(np.corrcoef(rankdata(frame[predictor]), rankdata(frame[outcome]))[0, 1])


def _partial_draft_r(observed: pd.DataFrame, predictor: str, outcome: str) -> float:
    drafted = observed.dropna(subset=["draft_overall_pick"])
    if len(drafted) < 40:
        return float("nan")
    return partial_spearman(
        drafted[predictor].to_numpy(dtype=float),
        drafted[outcome].to_numpy(dtype=float),
        [
            drafted["combine_weight"].to_numpy(dtype=float),
            drafted["role"].eq("CB").astype(float).to_numpy(),
            drafted["draft_overall_pick"].to_numpy(dtype=float),
        ],
    ).r_adj


def association_rows(
    frame: pd.DataFrame,
    outcomes: Sequence[str],
    predictors: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        for predictor, family in predictors:
            columns = [predictor, outcome, "combine_weight", "role", "draft_year"]
            observed = frame.loc[:, list(dict.fromkeys(columns))].dropna().copy()
            observed["draft_overall_pick"] = frame.loc[observed.index, "draft_overall_pick"]
            if len(observed) < 40 or observed[predictor].nunique() < 5 or observed[outcome].nunique() < 5:
                continue
            role = observed["role"].eq("CB").astype(float).to_numpy()
            statistic = partial_spearman(
                observed[predictor].to_numpy(dtype=float),
                observed[outcome].to_numpy(dtype=float),
                [observed["combine_weight"].to_numpy(dtype=float), role],
            )
            partial_draft_r = _partial_draft_r(observed, predictor, outcome)
            orientation = _predictor_orientation(predictor) * _outcome_orientation(outcome)
            oriented = statistic.r_adj * orientation if orientation else float("nan")
            replicated = {
                "r_disc_2023_24": _raw_spearman(observed[observed.draft_year.isin([2023, 2024])], predictor, outcome),
                "r_hold_2025": _raw_spearman(observed[observed.draft_year.eq(2025)], predictor, outcome),
                "r_2023": _raw_spearman(observed[observed.draft_year.eq(2023)], predictor, outcome),
                "r_2024": _raw_spearman(observed[observed.draft_year.eq(2024)], predictor, outcome),
                "r_2025": _raw_spearman(observed[observed.draft_year.eq(2025)], predictor, outcome),
            }
            rows.append(
                {
                    "scope": "db",
                    "group": "DB",
                    "outcome": outcome,
                    "predictor": predictor,
                    "family": family,
                    "n": len(observed),
                    "r_raw": statistic.r_raw,
                    "r_adj": statistic.r_adj,
                    "ci_lo": statistic.ci_lo,
                    "ci_hi": statistic.ci_hi,
                    "p": statistic.p,
                    "q_scope": np.nan,
                    "oriented_r": oriented,
                    "direction": orientation_label(oriented, outcome, predictor),
                    "partial_draft_r": partial_draft_r,
                    **replicated,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[*RESULT_COLUMNS, "partial_draft_r"])
    result["q_scope"] = benjamini_hochberg(result["p"].to_numpy(dtype=float))
    return result.loc[:, [*RESULT_COLUMNS, "partial_draft_r"]]


def _sensor_value_columns(attempts: pd.DataFrame) -> list[str]:
    fixed = {"nfl_id", "drill", "attempt"}
    return [str(column) for column in attempts.columns if column not in fixed]


def sensor_repeatability(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drill, drill_frame in attempts.groupby("drill", observed=True, sort=True):
        for metric in _sensor_value_columns(attempts):
            values = drill_frame.loc[:, ["nfl_id", metric]].dropna()
            count = int((values.groupby("nfl_id", observed=True).size() >= 2).sum())
            record: dict[str, object] = {
                "drill": str(drill),
                "metric": metric,
                "predictor": f"{str(drill).lower()}_{metric}",
                "n_players": count,
                "reliability": np.nan,
                "status": "insufficient_attempts",
            }
            if count >= 25:
                split = split_half_reliability(values, "nfl_id", metric, min_n=2, n_splits=2000, seed=20261103)
                reliability = split.median
                record["reliability"] = reliability
                record["status"] = "pass" if reliability >= 0.30 else "unreliable"
            rows.append(record)
    return pd.DataFrame(rows)


COVERAGE_OUTCOMES = ("coverage_assignment_rate", "man_coverage_assignment_rate")
MIN_PLAYS = 50
CATEGORICALS = ("team_coverage_man_zone", "team_coverage_type", "offense_formation")
COVARIATES = ("pass_length", "down", "yards_to_go")
OUTCOME_COLUMNS = (
    "coverage_assignment_at_snap",
    "tackle",
    "assist",
    "caused_forced_fumble",
    "tackle_for_loss",
)
FORMULA = (
    "{outcome} ~ pass_length + down + yards_to_go + C(team_coverage_man_zone) "
    "+ C(team_coverage_type) + C(offense_formation); missing pass_length (non-pass plays) set to 0 and "
    "missing categories set to NONE; man_coverage_assignment_rate is conditional on an assigned coverage"
)


def _outcome_frame(player_play: pd.DataFrame) -> pd.DataFrame:
    positions = {"CB", "FS", "SS", "DB"}
    frame = player_play.loc[
        player_play["nfl_position"].isin(positions)
        & player_play["draft_year"].isin([2023, 2024, 2025])
        & player_play["season"].isin([2023, 2024, 2025])
        & player_play["play_nullified_by_penalty"].ne("Y")
    ].copy()
    assigned = frame["coverage_assignment_at_snap"].notna()
    frame["assigned_coverage"] = assigned
    frame["coverage_assignment_rate"] = assigned.astype(float)
    frame["man_coverage_assignment_rate"] = frame["coverage_assignment_at_snap"].eq("MAN").astype(float).where(assigned)
    frame["tackle_rate"] = frame["tackle"].fillna(0.0).gt(0.0).astype(float)
    frame["assist_rate"] = frame["assist"].fillna(0.0).gt(0.0).astype(float)
    frame["forced_fumble_rate"] = frame["caused_forced_fumble"].fillna(0.0).gt(0.0).astype(float)
    frame["tackle_for_loss_rate"] = frame["tackle_for_loss"].fillna(False).astype(bool).astype(float)
    frame["pass_length"] = frame["pass_length"].fillna(0.0)
    for column in CATEGORICALS:
        frame[column] = frame[column].astype("object").where(frame[column].notna(), "NONE")
    return frame


def eligible_players(frame: pd.DataFrame, outcome: str) -> pd.Index:
    basis = frame["assigned_coverage"] if outcome in COVERAGE_OUTCOMES else frame[outcome].notna()
    counts = frame.loc[basis].groupby("nfl_id", observed=True).size()
    return counts.index[counts.ge(MIN_PLAYS)]


def _empty_gate_row(outcome: str) -> dict[str, object]:
    return {
        "outcome": outcome,
        "group": "DB",
        "n_players": 0,
        "SB_median": np.nan,
        "lower_ci": np.nan,
        "upper_ci": np.nan,
        "icc": np.nan,
        "verdict": "poor",
        "formula": "not enough players",
    }


def _gate_one(frame: pd.DataFrame, outcome: str) -> tuple[dict[str, object], pd.DataFrame | None]:
    players = eligible_players(frame, outcome)
    rows = frame.loc[frame["nfl_id"].isin(players) & frame[outcome].notna()]
    if len(players) < 3:
        return _empty_gate_row(outcome), None
    split = split_half_reliability(rows, "nfl_id", outcome, min_n=2, n_splits=2000, seed=20261103)
    lower, upper = player_bootstrap_ci(split, n_bootstrap=1000, seed=20261104)
    median = split.median
    try:
        icc = mixed_model_icc(rows, "nfl_id", outcome).icc
    except RuntimeError:
        icc = float("nan")
    status = verdict(median, lower)
    gate_row: dict[str, object] = {
        "outcome": outcome,
        "group": "DB",
        "n_players": int(len(players)),
        "SB_median": median,
        "lower_ci": lower,
        "upper_ci": upper,
        "icc": icc,
        "verdict": status,
        "formula": FORMULA.format(outcome=outcome),
    }
    if status != "adequate":
        return gate_row, None
    adjusted = adjusted_residuals(rows, outcome, COVARIATES, CATEGORICALS)
    name = f"{outcome}_residual"
    try:
        model = mixed_model_icc(adjusted, "nfl_id", name)
    except RuntimeError:
        return gate_row, None
    shrunk = empirical_bayes_shrink(adjusted, "nfl_id", name, model, min_n=2)
    return gate_row, shrunk.loc[:, ["nfl_id", "posterior_mean"]].rename(columns={"posterior_mean": outcome})


def gate_outcomes(player_play: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _outcome_frame(player_play)
    results = [_gate_one(frame, outcome) for outcome in OUTCOMES]
    scores = pd.DataFrame({"nfl_id": frame["nfl_id"].drop_duplicates()})
    for _, score in results:
        if score is not None:
            scores = scores.merge(score, on="nfl_id", how="left", validate="one_to_one")
    return pd.DataFrame([row for row, _ in results]), scores


def official_predictors(results: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    db = results.loc[results["combine_position"].eq("DB")].copy()
    db["nfl_id"] = pd.to_numeric(db["nfl_id"], errors="raise").astype(int)
    meta = players.loc[:, ["nfl_id", "nfl_position", "draft_overall_pick"]].copy()
    meta["nfl_id"] = pd.to_numeric(meta["nfl_id"], errors="raise").astype(int)
    return db.merge(meta, on="nfl_id", how="left", validate="one_to_one")


def sensor_player_features(attempts: pd.DataFrame, repeatability: pd.DataFrame) -> pd.DataFrame:
    passing = repeatability.loc[repeatability["status"].eq("pass")]
    wide = pd.DataFrame({"nfl_id": attempts["nfl_id"].drop_duplicates().astype(int)})
    columns: list[pd.Series] = []
    for drill, metric, predictor in zip(passing["drill"], passing["metric"], passing["predictor"], strict=True):
        subset = attempts.loc[attempts["drill"].eq(str(drill)), ["nfl_id", str(metric)]].dropna()
        columns.append(subset.groupby("nfl_id")[str(metric)].median().rename(str(predictor)))
    if not columns:
        return wide
    return wide.merge(pd.concat(columns, axis=1).reset_index(), on="nfl_id", how="left", validate="one_to_one")


def build_predictor_frame(
    results: pd.DataFrame,
    players: pd.DataFrame,
    attempts: pd.DataFrame,
    repeatability: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    official = official_predictors(results, players)
    sensors = sensor_player_features(attempts, repeatability)
    frame = official.merge(sensors, on="nfl_id", how="left", validate="one_to_one")
    eligible_official = [column for column in OFFICIAL if column in frame and frame[column].notna().sum() >= 40]
    sensor_columns = [column for column in sensors.columns if column != "nfl_id" and frame[column].notna().sum() >= 40]
    frame = add_composites(frame, set(eligible_official).union(sensor_columns))
    composites = [
        column
        for column in ("DB_SPEED", "DB_BURST", "DB_COD", "DB_EXPLOSIVE", "DB_REACH")
        if frame[column].notna().sum() >= 40
    ]
    predictors = [
        *((column, "official") for column in eligible_official),
        *((column, "sensor") for column in sensor_columns),
        *((column, "composite") for column in composites),
    ]
    return frame, predictors


def build_sensor_attempts(root: Path, identifiers: set[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    columns = ("nfl_id", "entity_type", "time", "drill_name", "attempt", "x", "y")
    source = root / "data" / "raw" / "combine_tracking.csv"
    for chunk in pd.read_csv(source, usecols=list(columns), chunksize=100_000, low_memory=False):
        selected = chunk.loc[chunk["entity_type"].eq("PLAYER") & chunk["nfl_id"].astype(str).isin(identifiers)].copy()
        if selected.empty:
            continue
        parsed_time = pd.to_datetime(selected["time"], utc=True, errors="coerce")
        selected["time"] = (parsed_time - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
        selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
        chunks.append(selected.dropna(subset=["time", "x", "y", "attempt", "drill_name"]))
    tracking = pd.concat(chunks, ignore_index=True)
    rows: list[dict[str, object]] = []
    for (nfl_id, drill, attempt), values in tracking.groupby(["nfl_id", "drill_name", "attempt"], observed=True):
        metrics = attempt_kinematics(
            values["time"].to_numpy(dtype=float),
            values["x"].to_numpy(dtype=float),
            values["y"].to_numpy(dtype=float),
            str(drill),
        )
        rows.append({"nfl_id": int(str(nfl_id)), "drill": str(drill), "attempt": str(attempt), **metrics})
    return pd.DataFrame(rows)


def run_scope(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed = root / "data" / "processed"
    reports = root / "reports"
    raw = root / "data" / "raw"
    combine = pd.read_csv(raw / "combine_results.csv")
    players = pd.read_csv(raw / "players.csv")
    attempts_path = processed / "scope_db_sensor_attempts.parquet"
    if attempts_path.exists():
        attempts = pd.read_parquet(attempts_path)
    else:
        identifiers = set(combine.loc[combine["combine_position"].eq("DB"), "nfl_id"].astype(str))
        attempts = build_sensor_attempts(root, identifiers)
        attempts.to_parquet(attempts_path, index=False)
    repeat_path = processed / "scope_db_sensor_repeatability.parquet"
    if repeat_path.exists():
        repeatability = pd.read_parquet(repeat_path)
    else:
        repeatability = sensor_repeatability(attempts)
        repeatability.to_parquet(repeat_path, index=False)
    predictor_frame, predictors = build_predictor_frame(combine, players, attempts, repeatability)
    columns = [
        "nfl_id",
        "nfl_position",
        "draft_year",
        "season",
        "play_nullified_by_penalty",
        *COVARIATES,
        *CATEGORICALS,
        *OUTCOME_COLUMNS,
    ]
    player_play = pd.read_parquet(processed / "player_play_joined.parquet", columns=list(dict.fromkeys(columns)))
    gate, scores = gate_outcomes(player_play)
    gate.to_csv(reports / "scope_db_gate.csv", index=False)
    joined = join_predictors_scores(predictor_frame, scores)
    joined.to_parquet(processed / "scope_db_analysis.parquet", index=False)
    gated = gate.loc[gate["verdict"].eq("adequate"), "outcome"].tolist()
    result = association_rows(joined, gated, predictors)
    result.loc[:, RESULT_COLUMNS].to_csv(reports / "scope_db_results.csv", index=False)
    result.to_parquet(processed / "scope_db_results_with_draft.parquet", index=False)
    return gate, result, repeatability


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    gate, result, repeatability = run_scope(arguments.root)
    print(gate.to_string(index=False))
    print(f"repeatable sensor metrics: {(repeatability['status'] == 'pass').sum()}")
    print(f"associations: {len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
