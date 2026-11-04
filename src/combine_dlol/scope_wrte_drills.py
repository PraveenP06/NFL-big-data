from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from combine_dlol.combine_predictors import (
    _clean_trajectory,
    attempt_kinematics,
    centered_smoothed_speed,
    first_sustained_movement,
)
from combine_dlol.reliability import (
    empirical_bayes_shrink,
    mixed_model_icc,
    player_bootstrap_ci,
    split_half_reliability,
    verdict,
)
from combine_dlol.scope_db import benjamini_hochberg, partial_spearman
from combine_dlol.scope_wrte import (
    RESULT_COLUMNS as BASE_RESULT_COLUMNS,
)
from combine_dlol.scope_wrte import (
    _context_formula,
    _pair_repeatability,
    orientation_label,
)

RESULT_COLUMNS = (*BASE_RESULT_COLUMNS, "family_label")

DRILL_MAP = {
    "SLANT_ROUTE_LEFT": "SLANT",
    "SLOT_STRIKE_ROUTE_LEFT": "SLOT_STRIKE",
    "CURL_ROUTE_RIGHT": "CURL",
    "COMEBACK_ROUTE_RIGHT": "COMEBACK",
    "HOOK_ROUTE_RIGHT": "HOOK",
    "GO_ROUTE_RIGHT": "GO",
    "RED_ZONE_FADE_RIGHT": "RED_ZONE_FADE",
    "END_ZONE_FADE_RIGHT": "END_ZONE_FADE",
    "OVER_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
    "SLOT_SAIL_ROUTE_RIGHT": "SLOT_SAIL",
    "POST_CORNER_ROUTE_LEFT": "POST_CORNER",
    "POST_CORNER_ROUTE_RIGHT": "POST_CORNER",
    "CORNER_ROUTE": "CORNER",
    "IN_ROUTE_LEFT": "IN",
    "DAGGER_ROUTE_LEFT": "DAGGER",
    "SPEED_OUT_LEFT": "SPEED_OUT",
    "SPEED_OUT_ROUTE_LEFT": "SPEED_OUT",
    "FLAT_ROUTE_LEFT": "FLAT",
    "WHEEL_ROUTE_LEFT": "WHEEL",
    "CIRCUS_ROUTE_LEFT": "CIRCUS",
    "BLOCK_EXPLOSION": "BLOCK_EXPLOSION",
}
NO_BREAK_DRILLS = frozenset(
    {"GO", "RED_ZONE_FADE", "END_ZONE_FADE", "OVER_SHOULDER_ADJUST", "CIRCUS", "BLOCK_EXPLOSION"}
)
ROUTE_FAMILIES = {
    "GO": ("GO",),
    "HITCH": ("HITCH",),
    "INSIDE": ("SLANT", "IN", "CROSS"),
    "OUTSIDE_SHORT": ("OUT", "FLAT", "WHEEL"),
    "DEEP_BREAK": ("POST", "CORNER"),
}
MATCHED = {
    "SLANT": "INSIDE",
    "SLOT_STRIKE": "INSIDE",
    "IN": "INSIDE",
    "DAGGER": "INSIDE",
    "CURL": "HITCH",
    "COMEBACK": "HITCH",
    "HOOK": "HITCH",
    "GO": "GO",
    "RED_ZONE_FADE": "GO",
    "END_ZONE_FADE": "GO",
    "OVER_SHOULDER_ADJUST": "GO",
    "SLOT_SAIL": "GO",
    "POST_CORNER": "DEEP_BREAK",
    "CORNER": "DEEP_BREAK",
    "SPEED_OUT": "OUTSIDE_SHORT",
    "FLAT": "OUTSIDE_SHORT",
    "WHEEL": "OUTSIDE_SHORT",
}
MEASURES = ("sep", "target_rate", "ypr")
METRIC_ORIENTATION = {
    "time_in_drill": -1,
    "top_speed": 1,
    "burst_distance_0_1s": 1,
    "burst_speed_1_0s": 1,
    "peak_acceleration": 1,
    "peak_deceleration": 1,
    "path_length": 0,
    "path_efficiency": 1,
    "break_speed": 1,
    "break_speed_retention": 1,
}
MIN_PLAYERS = 40
MIN_MULTI_ATTEMPT_PLAYERS = 25
MIN_REPEATABILITY = 0.30
MIN_ROUTES = 50
BREAK_WINDOW_SECONDS = 0.4
BREAK_SPEED_SECONDS = 0.25
MIN_TURN_DEGREES = 45.0
MIN_HEADING_SPEED = 1.0
ATTEMPT_COLUMNS = ["nfl_id", "drill", "raw_drill", "attempt", "metric", "value"]
GATE_COLUMNS = ["outcome", "group", "n_players", "SB_median", "lower_ci", "upper_ci", "icc", "verdict", "formula"]


def merged_drill(raw_name: str) -> str | None:
    return DRILL_MAP.get(raw_name)


def predictor_name(drill: str, metric: str) -> str:
    return f"{drill.lower()}_{metric}"


def outcome_name(measure: str, family: str) -> str:
    return f"{measure}_{family}"


def break_metrics(
    time: Sequence[float] | np.ndarray, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> dict[str, float]:
    empty = {"break_speed": float("nan"), "break_speed_retention": float("nan")}
    clean_time, clean_x, clean_y = _clean_trajectory(time, x, y)
    if len(clean_time) < 5:
        return empty
    speed = centered_smoothed_speed(clean_time, clean_x, clean_y)
    start = first_sustained_movement(speed)
    seconds = clean_time[start:]
    speed = speed[start:]
    if len(seconds) < 5:
        return empty
    kernel = np.array([0.25, 0.5, 0.25])
    velocity_x = np.convolve(np.pad(np.gradient(clean_x[start:], seconds), 1, mode="edge"), kernel, mode="valid")
    velocity_y = np.convolve(np.pad(np.gradient(clean_y[start:], seconds), 1, mode="edge"), kernel, mode="valid")
    heading = np.arctan2(velocity_y, velocity_x)
    step = float(np.median(np.diff(seconds)))
    offset = int(round(BREAK_WINDOW_SECONDS / step))
    if offset < 1 or len(seconds) <= 2 * offset:
        return empty
    centers = np.arange(offset, len(seconds) - offset)
    turn = np.abs((heading[centers + offset] - heading[centers - offset] + np.pi) % (2.0 * np.pi) - np.pi)
    moving = (speed[centers - offset] >= MIN_HEADING_SPEED) & (speed[centers + offset] >= MIN_HEADING_SPEED)
    turn = np.where(moving, turn, -1.0)
    best = int(np.argmax(turn))
    if np.degrees(turn[best]) < MIN_TURN_DEGREES:
        return empty
    index = int(centers[best])
    half = max(1, int(round(BREAK_SPEED_SECONDS / step)))
    through = float(np.mean(speed[max(0, index - half) : index + half + 1]))
    before = float(np.max(speed[: index + 1]))
    return {"break_speed": through, "break_speed_retention": through / before if before > 0.0 else float("nan")}


def _attempt_metrics(raw_drill: str, drill: str, time: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    values = attempt_kinematics(time, x, y, raw_drill)
    wanted = {name: value for name, value in values.items() if name in METRIC_ORIENTATION}
    if drill not in NO_BREAK_DRILLS and wanted:
        wanted.update(break_metrics(time, x, y))
    return wanted


def build_attempts(root: Path, identifiers: set[str]) -> pd.DataFrame:
    cache = root / "data/processed/scope_wrte_drills_attempts.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    usecols = ["nfl_id", "entity_type", "time", "drill_name", "attempt", "x", "y"]
    segments: list[pd.DataFrame] = []
    source = root / "data/raw/combine_tracking.csv"
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=100_000, dtype={"nfl_id": "string"}):
        keep = chunk["entity_type"].eq("PLAYER") & chunk["nfl_id"].isin(identifiers)
        keep &= chunk["drill_name"].isin(DRILL_MAP)
        selected = chunk.loc[keep].copy()
        if selected.empty:
            continue
        stamps = pd.to_datetime(selected["time"], utc=True, errors="coerce")
        selected["seconds"] = (stamps - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
        selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
        segments.append(selected.dropna(subset=["seconds", "x", "y", "attempt"]))
    rows: list[dict[str, object]] = []
    if segments:
        tracking = pd.concat(segments, ignore_index=True)
        for keys, attempt in tracking.groupby(["nfl_id", "drill_name", "attempt"], observed=True, sort=False):
            player, raw_drill, attempt_id = keys
            drill = DRILL_MAP[str(raw_drill)]
            values = _attempt_metrics(
                str(raw_drill),
                drill,
                attempt["seconds"].to_numpy(dtype=float),
                attempt["x"].to_numpy(dtype=float),
                attempt["y"].to_numpy(dtype=float),
            )
            rows.extend(
                {
                    "nfl_id": str(player),
                    "drill": drill,
                    "raw_drill": str(raw_drill),
                    "attempt": f"{raw_drill}|{attempt_id}",
                    "metric": metric,
                    "value": value,
                }
                for metric, value in values.items()
            )
    result = pd.DataFrame(rows, columns=ATTEMPT_COLUMNS)
    result.to_parquet(cache, index=False)
    return result


def repeatability_gate(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (drill, metric), values in attempts.groupby(["drill", "metric"], observed=True, sort=True):
        n_players, reliability = _pair_repeatability(values)
        if n_players < MIN_MULTI_ATTEMPT_PLAYERS:
            status = "unverifiable"
        elif np.isfinite(reliability) and reliability >= MIN_REPEATABILITY:
            status = "pass"
        else:
            status = "unrepeatable"
        rows.append(
            {
                "drill": str(drill),
                "metric": str(metric),
                "predictor": predictor_name(str(drill), str(metric)),
                "n_players_multi_attempt": n_players,
                "spearman_brown": reliability,
                "status": status,
            }
        )
    return pd.DataFrame(
        rows, columns=["drill", "metric", "predictor", "n_players_multi_attempt", "spearman_brown", "status"]
    )


def player_drill_table(attempts: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    usable = gate.loc[gate["status"].isin(["pass", "unverifiable"]), "predictor"]
    labelled = attempts.assign(
        column=[predictor_name(str(d), str(m)) for d, m in zip(attempts["drill"], attempts["metric"], strict=True)]
    )
    selected = labelled.loc[labelled["column"].isin(set(usable))]
    if selected.empty:
        return pd.DataFrame({"nfl_id": pd.Series(dtype=str)})
    table = selected.pivot_table(index="nfl_id", columns="column", values="value", aggfunc="mean")
    return table.reset_index().rename_axis(columns=None)


def _route_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["target_rate"] = pd.to_numeric(frame["target"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    frame["ypr"] = pd.to_numeric(frame["rec_yards"], errors="coerce").fillna(0.0)
    return frame


def outcome_definitions(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frame = _route_frame(data)
    source = {"sep": "separation_at_pass_forward", "target_rate": "target_rate", "ypr": "ypr"}
    definitions: list[tuple[str, pd.DataFrame]] = []
    for family, routes in ROUTE_FAMILIES.items():
        subset = frame.loc[frame["route_ran"].isin(routes)]
        for measure in MEASURES:
            name = outcome_name(measure, family)
            definitions.append((name, subset.assign(**{name: subset[source[measure]]})))
    return definitions


def _gate_row(outcome: str, n_players: int, formula: str, **values: float | str) -> dict[str, object]:
    row: dict[str, object] = {
        "outcome": outcome,
        "group": "WRTE",
        "n_players": n_players,
        "SB_median": np.nan,
        "lower_ci": np.nan,
        "upper_ci": np.nan,
        "icc": np.nan,
        "verdict": "poor",
        "formula": formula,
    }
    row.update(values)
    return row


def score_outcomes(
    data: pd.DataFrame, n_splits: int = 2000, n_bootstrap: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores: list[pd.DataFrame] = []
    gate_rows: list[dict[str, object]] = []
    for outcome, subset in outcome_definitions(data):
        subset = subset.dropna(subset=[outcome])
        formula, numeric, categorical = _context_formula(outcome, subset)
        model_data = subset.dropna(subset=[outcome, *numeric, *categorical])
        if model_data.empty:
            gate_rows.append(_gate_row(outcome, 0, formula))
            continue
        residual = smf.ols(formula, data=model_data).fit().resid
        column = f"{outcome}_residual"
        residuals = model_data.loc[:, ["nfl_id"]].assign(**{column: residual.to_numpy()})
        counts = residuals.groupby("nfl_id", observed=True).size()
        eligible = residuals.loc[residuals["nfl_id"].isin(counts[counts >= MIN_ROUTES].index)]
        n_eligible = int(eligible["nfl_id"].nunique())
        if n_eligible < 3:
            gate_rows.append(_gate_row(outcome, n_eligible, formula))
            continue
        split = split_half_reliability(eligible, "nfl_id", column, MIN_ROUTES, n_splits=n_splits, seed=20261103)
        lower, upper = player_bootstrap_ci(split, n_bootstrap=n_bootstrap, seed=20261104)
        reliability = mixed_model_icc(eligible, "nfl_id", column)
        status = verdict(split.median, lower)
        gate_rows.append(
            _gate_row(
                outcome,
                n_eligible,
                formula,
                SB_median=split.median,
                lower_ci=lower,
                upper_ci=upper,
                icc=reliability.icc,
                verdict=status,
            )
        )
        if status == "adequate":
            shrunk = empirical_bayes_shrink(eligible, "nfl_id", column, reliability, min_n=MIN_ROUTES)
            scores.append(shrunk.loc[:, ["nfl_id", "posterior_mean"]].rename(columns={"posterior_mean": outcome}))
    gate = pd.DataFrame(gate_rows, columns=GATE_COLUMNS)
    combined = pd.DataFrame({"nfl_id": pd.Series(dtype=str)})
    for score in scores:
        combined = score if combined.empty else combined.merge(score, on="nfl_id", how="outer", validate="one_to_one")
    return combined, gate


def build_tests(repeatability: pd.DataFrame, gated_outcomes: Sequence[str]) -> list[dict[str, str]]:
    tests: list[dict[str, str]] = []
    for row in repeatability.itertuples(index=False):
        if row.status not in ("pass", "unverifiable"):
            continue
        tier = "headline" if row.status == "pass" else "unverifiable"
        matched_family = MATCHED.get(str(row.drill))
        for outcome in gated_outcomes:
            family = next(name for name in ROUTE_FAMILIES if outcome.endswith(f"_{name}"))
            kind = "matched" if matched_family == family else "unmatched"
            tests.append(
                {
                    "predictor": str(row.predictor),
                    "metric": str(row.metric),
                    "outcome": outcome,
                    "family": kind,
                    "family_label": f"{tier}_{kind}",
                }
            )
    return tests


def _year_correlation(model: pd.DataFrame, predictor: str, outcome: str, years: tuple[int, ...]) -> float:
    subset = model.loc[model["draft_year"].isin(years)]
    if len(subset) < 15:
        return float("nan")
    return float(spearmanr(subset[predictor], subset[outcome]).statistic)


def association_rows(frame: pd.DataFrame, tests: Iterable[dict[str, str]]) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    skipped = 0
    for test in tests:
        predictor, outcome = test["predictor"], test["outcome"]
        if predictor not in frame or outcome not in frame:
            skipped += 1
            continue
        model = frame.loc[:, [predictor, outcome, "combine_weight", "role_wr", "draft_year"]].dropna()
        if len(model) < MIN_PLAYERS or model[predictor].nunique() < 5 or model[outcome].nunique() < 5:
            skipped += 1
            continue
        controls = [model["combine_weight"].to_numpy(dtype=float)]
        if model["role_wr"].nunique() > 1:
            controls.append(model["role_wr"].to_numpy(dtype=float))
        result = partial_spearman(model[predictor].to_numpy(float), model[outcome].to_numpy(float), controls)
        orientation = METRIC_ORIENTATION[test["metric"]]
        oriented = result.r_adj * orientation if orientation else float("nan")
        rows.append(
            {
                "scope": "wrte_drills",
                "group": "WRTE",
                "outcome": outcome,
                "predictor": predictor,
                "family": test["family"],
                "n": len(model),
                "r_raw": result.r_raw,
                "r_adj": result.r_adj,
                "ci_lo": result.ci_lo,
                "ci_hi": result.ci_hi,
                "p": result.p,
                "q_scope": np.nan,
                "oriented_r": oriented,
                "direction": orientation_label(oriented),
                "r_disc_2023_24": _year_correlation(model, predictor, outcome, (2023, 2024)),
                "r_hold_2025": _year_correlation(model, predictor, outcome, (2025,)),
                "r_2023": _year_correlation(model, predictor, outcome, (2023,)),
                "r_2024": _year_correlation(model, predictor, outcome, (2024,)),
                "r_2025": _year_correlation(model, predictor, outcome, (2025,)),
                "family_label": test["family_label"],
            }
        )
    result_frame = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    for _, index in result_frame.groupby("family_label").groups.items():
        result_frame.loc[index, "q_scope"] = benjamini_hochberg(result_frame.loc[index, "p"].to_numpy(dtype=float))
    return result_frame, skipped


def load_players(root: Path) -> pd.DataFrame:
    combine = pd.read_csv(root / "data/raw/combine_results.csv")
    players = combine.loc[combine["combine_position"].isin(["WR", "TE"])].copy()
    players["nfl_id"] = players["nfl_id"].astype(str)
    players["combine_weight"] = pd.to_numeric(players["combine_weight"], errors="coerce")
    players["role_wr"] = players["combine_position"].eq("WR").astype(float)
    return players.loc[:, ["nfl_id", "draft_year", "combine_weight", "role_wr"]]


def load_route_data(root: Path) -> pd.DataFrame:
    columns = [
        "nfl_id",
        "nfl_position",
        "route_ran",
        "play_nullified_by_penalty",
        "target",
        "rec_yards",
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
    keep = data["nfl_position"].isin(["WR", "TE"]) & data["route_ran"].notna()
    data = data.loc[keep & data["play_nullified_by_penalty"].ne("Y")].copy()
    data["nfl_id"] = data["nfl_id"].astype(str)
    return data


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["family_label", "tests", "raw_p_lt_0_05", "q_lt_0_10", "expected", "opposite"])
    grouped = results.groupby("family_label")
    return pd.DataFrame(
        {
            "tests": grouped.size(),
            "raw_p_lt_0_05": grouped["p"].apply(lambda s: int((s < 0.05).sum())),
            "q_lt_0_10": grouped["q_scope"].apply(lambda s: int((s < 0.10).sum())),
            "expected": grouped["direction"].apply(lambda s: int((s == "expected").sum())),
            "opposite": grouped["direction"].apply(lambda s: int((s == "opposite").sum())),
        }
    ).reset_index()


def run_scope(
    root: Path, n_splits: int = 2000, n_bootstrap: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    reports = root / "reports"
    players = load_players(root)
    attempts = build_attempts(root, set(players["nfl_id"]))
    repeatability = repeatability_gate(attempts)
    repeatability.to_csv(reports / "scope_wrte_drills_repeatability_gate.csv", index=False)
    scores, gate = score_outcomes(load_route_data(root), n_splits=n_splits, n_bootstrap=n_bootstrap)
    gate.to_csv(reports / "scope_wrte_drills_outcome_gate.csv", index=False)
    gated = gate.loc[gate["verdict"].eq("adequate"), "outcome"].tolist()
    predictors = player_drill_table(attempts, repeatability)
    frame = players.merge(predictors, on="nfl_id", how="inner", validate="one_to_one")
    if gated:
        frame = frame.merge(scores, on="nfl_id", how="inner", validate="one_to_one")
    tests = build_tests(repeatability, gated)
    results, skipped = association_rows(frame, tests)
    results.to_csv(reports / "scope_wrte_drills_results.csv", index=False)
    return results, gate, repeatability, skipped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    results, gate, repeatability, skipped = run_scope(arguments.root)
    print(gate.drop(columns="formula").to_string(index=False))
    print(repeatability["status"].value_counts().to_string())
    print(f"tests run={len(results)} skipped(n<{MIN_PLAYERS})={skipped}")
    print(summarize(results).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
