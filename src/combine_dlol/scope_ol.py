from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t

from combine_dlol.data import add_line_of_scrimmage
from combine_dlol.reliability import (
    adjusted_residuals,
    empirical_bayes_shrink,
    mixed_model_icc,
    player_bootstrap_ci,
    split_half_reliability,
    verdict,
)

KEYS = ("game_id", "play_id", "nfl_id")
OFFICIAL = (
    "forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump",
    "combine_height", "combine_weight", "arm_length", "wing_span", "hand_size", "bench_reps",
)
FEATURE_OUTCOMES = (
    "pass_set_depth_1_0s", "pass_set_depth_1_5s", "pass_set_depth_2_0s",
    "lateral_shuffle_0_1_5s", "set_peak_speed_0_1_5s", "set_time_to_peak_speed_0_1_5s",
    "nearest_defender_separation_1_0s", "nearest_defender_separation_2_0s",
    "pressure_allowed", "sack_allowed",
)
INFEASIBLE_OUTCOMES = ("nearest_defender_separation_1_0s", "nearest_defender_separation_2_0s")
GATED_OUTCOMES = tuple(name for name in FEATURE_OUTCOMES if name not in INFEASIBLE_OUTCOMES)
BASELINE_OUTCOMES = ("peak_pressure_probability_allowed", "snap_reaction_time", "first_0_5s_displacement")
INFEASIBLE_REASON = (
    "infeasible: game tracking holds only draft-class players (about 1.6 tracked players per play), "
    "so the nearest defender cannot be identified"
)
OUTCOME_ORIENTATION = {
    "pass_set_depth_1_0s": 1, "pass_set_depth_1_5s": 1, "pass_set_depth_2_0s": 1,
    "lateral_shuffle_0_1_5s": 0, "set_peak_speed_0_1_5s": 1,
    "set_time_to_peak_speed_0_1_5s": -1, "nearest_defender_separation_1_0s": 1,
    "nearest_defender_separation_2_0s": 1, "pressure_allowed": -1, "sack_allowed": -1,
    "peak_pressure_probability_allowed": -1, "snap_reaction_time": 0, "first_0_5s_displacement": 0,
}
ADJUSTMENT_NUMERIC = ("down", "yards_to_go", "dropback_duration", "pass_rushers_encountered")
ADJUSTMENT_CATEGORICAL = ("lined_up_position", "offense_formation")
FORMULA = (
    "OLS residual: outcome ~ C(lined_up_position) + down + yards_to_go + "
    "C(offense_formation) + dropback_duration + pass_rushers_encountered"
)
CONTROLS = ("combine_weight", "arm_length", "combine_height")
COMPOSITES = {
    "OL_QUICKNESS": (
        ("forty_yard_dash_peak_acceleration", 1), ("forty_yard_dash_burst_speed_1_0s", 1),
        ("forty_yard_dash_sensor_ten_yd_burst_speed", 1), ("forty_yard_dash_sensor_ten_yd_time", -1),
    ),
    "OL_SPEED": (("forty_yard_dash_top_speed", 1), ("forty", -1)),
    "OL_DROP_INDEX": (("pass_rush_drop_top_speed", 1), ("pass_rush_drop_path_length", 1)),
}


@dataclass(frozen=True)
class PartialStatistic:
    r: float
    ci_lo: float
    ci_hi: float
    p: float


def _snapshot_at(data: pd.DataFrame, second: float) -> pd.DataFrame:
    window = data.loc[np.abs(data["seconds_since_snap"] - second).le(0.051)].copy()
    if window.empty:
        return window
    window["_distance"] = np.abs(window["seconds_since_snap"] - second)
    return window.sort_values([*KEYS, "_distance"], kind="stable").groupby(list(KEYS), as_index=False).head(1)


def _separation(tracking: pd.DataFrame, second: float) -> pd.DataFrame:
    ol = _snapshot_at(tracking.loc[tracking["is_ol"]], second)
    defenders = _snapshot_at(tracking.loc[tracking["is_defender"]], second)
    name = f"nearest_defender_separation_{str(second).replace('.', '_')}s"
    if ol.empty or defenders.empty:
        return pd.DataFrame(columns=[*KEYS, name])
    match_keys = ["game_id", "play_id", "seconds_since_snap"]
    merged = ol.loc[:, [*KEYS, "seconds_since_snap", "x_rel", "y_rel"]].merge(
        defenders.loc[:, [*match_keys, "x_rel", "y_rel"]], on=match_keys, how="inner", suffixes=("_ol", "_def")
    )
    distance = np.hypot(merged["x_rel_ol"] - merged["x_rel_def"], merged["y_rel_ol"] - merged["y_rel_def"])
    return merged.assign(**{name: distance}).groupby(list(KEYS), as_index=False).agg(**{name: (name, "min")})


def tracking_features(tracking: pd.DataFrame) -> pd.DataFrame:
    required = {*KEYS, "seconds_since_snap", "x_rel", "y_rel", "s", "is_ol", "is_defender"}
    if not required.issubset(tracking.columns):
        raise ValueError("tracking lacks required columns")
    ol = tracking.loc[tracking["is_ol"]].copy()
    snap = _snapshot_at(ol, 0.0).loc[:, [*KEYS, "x_rel", "y_rel"]].rename(
        columns={"x_rel": "_snap_x", "y_rel": "_snap_y"}
    )
    result = snap.copy()
    for second in (1.0, 1.5, 2.0):
        point = _snapshot_at(ol, second).loc[:, [*KEYS, "x_rel", "y_rel"]]
        point = point.rename(columns={"x_rel": f"_x_{second}", "y_rel": f"_y_{second}"})
        result = result.merge(point, on=list(KEYS), how="left", validate="one_to_one")
    for second in (1.0, 1.5, 2.0):
        name = f"pass_set_depth_{str(second).replace('.', '_')}s"
        result[name] = -(result[f"_x_{second}"] - result["_snap_x"])
    result["lateral_shuffle_0_1_5s"] = (result["_y_1.5"] - result["_snap_y"]).abs()
    early = ol.loc[ol["seconds_since_snap"].between(0.0, 1.5)].copy()
    early = early.sort_values([*KEYS, "seconds_since_snap"], kind="stable")
    peaks = early.groupby(list(KEYS), as_index=False).agg(set_peak_speed_0_1_5s=("s", "max"))
    peak_time = early.merge(peaks, on=list(KEYS), how="inner", validate="many_to_one")
    peak_time = peak_time.loc[np.isclose(peak_time["s"], peak_time["set_peak_speed_0_1_5s"])]
    peak_time = peak_time.groupby(list(KEYS), as_index=False).agg(
        set_time_to_peak_speed_0_1_5s=("seconds_since_snap", "min")
    )
    result = result.merge(peaks, on=list(KEYS), how="left", validate="one_to_one")
    result = result.merge(peak_time, on=list(KEYS), how="left", validate="one_to_one")
    for second in (1.0, 2.0):
        result = result.merge(_separation(tracking, second), on=list(KEYS), how="left", validate="one_to_one")
    return result.loc[:, [*KEYS, *FEATURE_OUTCOMES[:8]]]


def attach_tracking_features(outcomes: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return outcomes.merge(features, on=list(KEYS), how="left", validate="one_to_one")


def partial_spearman(x: Sequence[float], y: Sequence[float], controls: Sequence[Sequence[float]]) -> PartialStatistic:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    control_values = [np.asarray(control, dtype=float) for control in controls]
    if len(x_values) != len(y_values) or any(len(control) != len(x_values) for control in control_values):
        raise ValueError("partial Spearman inputs must have equal length")
    design = np.column_stack([np.ones(len(x_values)), *[rankdata(control) for control in control_values]])
    x_residual = rankdata(x_values) - design @ np.linalg.lstsq(design, rankdata(x_values), rcond=None)[0]
    y_residual = rankdata(y_values) - design @ np.linalg.lstsq(design, rankdata(y_values), rcond=None)[0]
    r_value = float(np.corrcoef(x_residual, y_residual)[0, 1])
    n_controls = len(control_values)
    degrees = len(x_values) - 2 - n_controls
    statistic = r_value * np.sqrt(degrees / max(1.0 - r_value * r_value, np.finfo(float).eps))
    p_value = float(2.0 * t.sf(abs(statistic), degrees))
    standard_error = 1.0 / np.sqrt(len(x_values) - 3 - n_controls)
    bounded = np.clip(r_value, -0.999999, 0.999999)
    lower, upper = np.tanh(np.arctanh(bounded) + np.array([-1.96, 1.96]) * standard_error)
    return PartialStatistic(r_value, float(lower), float(upper), p_value)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    ordering = np.argsort(values)
    output = np.empty(len(values), dtype=float)
    running = 1.0
    for rank, index in zip(range(len(values), 0, -1), ordering[::-1], strict=True):
        running = min(running, values[index] * len(values) / rank)
        output[index] = running
    return output


def orientation_label(r_value: float, predictor_orientation: int, outcome_orientation: int) -> str:
    if predictor_orientation == 0 or outcome_orientation == 0 or not np.isfinite(r_value):
        return "unclassified"
    return "expected" if r_value * predictor_orientation * outcome_orientation > 0 else "opposite"


def _ol_context(root: Path) -> pd.DataFrame:
    columns = [
        "game_id", "play_id", "nfl_id", "season", "nfl_position", "team_abbr", "possession_team",
        "lined_up_position", "down", "yards_to_go", "offense_formation", "dropback_duration",
        "pass_rushers_encountered", "peak_pressure_probability_allowed", "pressure_allowed", "sack_allowed",
        "play_nullified_by_penalty", "yardline_side", "yardline_number", "play_direction",
    ]
    data = pd.read_parquet(root / "data/processed/player_play_joined.parquet", columns=columns)
    included = data["nfl_position"].isin(["T", "G", "C"]) & data["play_nullified_by_penalty"].ne("Y")
    return data.loc[included & data["dropback_duration"].notna() & data["pass_rushers_encountered"].notna()].copy()


def extract_season_tracking(root: Path, season: int) -> Path:
    context = _ol_context(root)
    ol = context.loc[context["season"].eq(season)].copy()
    play_keys = ol.loc[:, ["game_id", "play_id"]].drop_duplicates()
    player_columns = ["game_id", "play_id", "nfl_id", "team_abbr"]
    all_players = pd.read_parquet(root / "data/processed/player_play_joined.parquet", columns=player_columns)
    all_players = all_players.merge(play_keys, on=["game_id", "play_id"], how="inner", validate="many_to_one")
    all_players = all_players.drop_duplicates(list(KEYS))
    tracking = pd.read_parquet(
        root / f"data/processed/game_tracking_{season}.parquet",
        columns=["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"],
    )
    tracking = tracking.merge(all_players, on=list(KEYS), how="inner", validate="many_to_one")
    geometry = ol.loc[
        :, ["game_id", "play_id", "yardline_side", "yardline_number", "possession_team", "play_direction"]
    ]
    tracking = add_line_of_scrimmage(tracking, geometry)
    snap = tracking.loc[tracking["event"].eq("ball_snap"), ["game_id", "play_id", "time"]]
    snap = snap.drop_duplicates(["game_id", "play_id"])
    snap = snap.rename(columns={"time": "_snap_time"})
    tracking = tracking.merge(snap, on=["game_id", "play_id"], how="inner", validate="many_to_one")
    seconds = (pd.to_datetime(tracking["time"]) - pd.to_datetime(tracking["_snap_time"])).dt.total_seconds()
    ol_keys = ol.loc[:, list(KEYS)].drop_duplicates().assign(is_ol=True)
    tracking = tracking.merge(ol_keys, on=list(KEYS), how="left", validate="many_to_one")
    tracking["is_ol"] = tracking["is_ol"].fillna(False)
    tracking["is_defender"] = tracking["team_abbr"].ne(tracking["possession_team"]) & ~tracking["is_ol"]
    features = tracking_features(tracking.assign(seconds_since_snap=seconds))
    output = root / f"data/processed/scope_ol_tracking_features_{season}.parquet"
    features.to_parquet(output, index=False)
    return output


def prepare_play_outcomes(root: Path) -> Path:
    context = _ol_context(root)
    feature_paths = [
        root / f"data/processed/scope_ol_tracking_features_{season}.parquet" for season in (2023, 2024, 2025)
    ]
    if not all(path.is_file() for path in feature_paths):
        raise FileNotFoundError("run season extraction before preparing play outcomes")
    features = pd.concat([pd.read_parquet(path) for path in feature_paths], ignore_index=True)
    output = attach_tracking_features(context, features)
    snap_paths = [
        root / f"data/processed/ol_outcomes_snap_features_{season}.parquet" for season in (2023, 2024, 2025)
    ]
    snap = pd.concat([pd.read_parquet(path) for path in snap_paths], ignore_index=True)
    snap_columns = [*KEYS, "snap_reaction_time", "first_0_5s_displacement"]
    output = output.merge(snap.loc[:, snap_columns], on=list(KEYS), how="left")
    output.to_parquet(root / "data/processed/scope_ol_play_outcomes.parquet", index=False)
    return root / "data/processed/scope_ol_play_outcomes.parquet"


def _store_gate_rows(root: Path, rows: pd.DataFrame) -> None:
    path = root / "reports/scope_ol_gate.csv"
    existing = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=rows.columns)
    kept = existing.loc[~existing["outcome"].isin(rows["outcome"])]
    pd.concat([kept, rows], ignore_index=True).to_csv(path, index=False)


def record_infeasible_outcomes(root: Path) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "outcome": list(INFEASIBLE_OUTCOMES), "group": "OL", "n_players": 0, "SB_median": float("nan"),
            "lower_ci": float("nan"), "upper_ci": float("nan"), "icc": float("nan"),
            "verdict": "infeasible", "formula": INFEASIBLE_REASON,
        }
    )
    _store_gate_rows(root, rows)
    return rows


def gate_outcome(root: Path, outcome: str) -> pd.DataFrame:
    if outcome not in (*GATED_OUTCOMES, *BASELINE_OUTCOMES):
        raise ValueError(f"unknown outcome: {outcome}")
    data = pd.read_parquet(root / "data/processed/scope_ol_play_outcomes.parquet")
    numeric = pd.to_numeric(data[outcome], errors="coerce").astype(float)
    residualized = adjusted_residuals(
        data.assign(**{outcome: numeric}), outcome, ADJUSTMENT_NUMERIC, ADJUSTMENT_CATEGORICAL
    )
    residual_name = f"{outcome}_residual"
    counts = residualized.groupby("nfl_id")[residual_name].count()
    usable = residualized.loc[residualized["nfl_id"].isin(counts.loc[counts.ge(50)].index)].copy()
    split = split_half_reliability(usable, "nfl_id", residual_name, 50, n_splits=2000, seed=20261103)
    lower, upper = player_bootstrap_ci(split, n_bootstrap=1000, seed=20261104)
    model = mixed_model_icc(usable, "nfl_id", residual_name)
    decision = verdict(split.median, lower)
    score = empirical_bayes_shrink(usable, "nfl_id", residual_name, model, min_n=50).assign(outcome=outcome)
    score.to_parquet(root / f"data/processed/scope_ol_scores_{outcome}.parquet", index=False)
    row = pd.DataFrame(
        [{
            "outcome": outcome, "group": "OL", "n_players": usable["nfl_id"].nunique(),
            "SB_median": split.median, "lower_ci": lower, "upper_ci": upper, "icc": model.icc,
            "verdict": decision, "formula": FORMULA,
        }]
    )
    _store_gate_rows(root, row)
    residualized.loc[:, [*KEYS, "season", residual_name]].to_parquet(
        root / f"data/processed/scope_ol_adjusted_{outcome}.parquet", index=False
    )
    return row


def _predictors(root: Path) -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    data = pd.read_parquet(root / "data/processed/combine_predictors.parquet")
    data["nfl_id"] = data["nfl_id"].astype(int)
    data = data.loc[data["group"].eq("OL")].copy()
    meta = pd.read_csv(root / "reports/phase2_combine_predictors.csv")
    sensor = meta.loc[
        meta["group"].eq("OL") & meta["family"].eq("sensor") & meta["eligible_for_M4"].astype(bool), "predictor_column"
    ].tolist()
    families = {name: "official" for name in OFFICIAL if name in data.columns}
    families.update({name: "sensor" for name in sensor})
    eligible = set(sensor)
    for composite, terms in COMPOSITES.items():
        names = [name for name, _ in terms]
        sensor_names = [name for name in names if name != "forty"]
        if all(name in data.columns for name in names) and all(name in eligible for name in sensor_names):
            values = []
            for name, sign in terms:
                series = pd.to_numeric(data[name], errors="coerce")
                values.append(sign * (series - series.mean()) / series.std(ddof=0))
            data[composite] = pd.concat(values, axis=1).mean(axis=1, skipna=False)
            families[composite] = "composite"
    orientation = {name: _predictor_orientation(name) for name in families}
    return data, families, orientation


def _predictor_orientation(name: str) -> int:
    if name in {"forty", "ten_yd_split", "three_cone", "short_shuttle"} or name.endswith(("_time", "time_in_drill")):
        return -1
    if name in {"vertical", "broad_jump", "bench_reps", "OL_QUICKNESS", "OL_SPEED", "OL_DROP_INDEX"}:
        return 1
    if any(token in name for token in ("top_speed", "burst", "peak_acceleration")):
        return 1
    return 0


def _raw_spearman(data: pd.DataFrame, predictor: str, outcome: str) -> float:
    if len(data) < 15 or data[predictor].nunique() < 2 or data[outcome].nunique() < 2:
        return float("nan")
    return float(spearmanr(data[predictor], data[outcome]).statistic)


def _reach_partial(data: pd.DataFrame, predictor: str, outcome: str, controls: Sequence[str]) -> float:
    if len(data) < 40:
        return float("nan")
    return partial_spearman(
        data[predictor].astype(float).tolist(),
        data[outcome].astype(float).tolist(),
        [data[name].astype(float).tolist() for name in controls],
    ).r


def associate(root: Path) -> pd.DataFrame:
    gate = pd.read_csv(root / "reports/scope_ol_gate.csv")
    accepted = gate.loc[gate["verdict"].eq("adequate") & gate["outcome"].isin(GATED_OUTCOMES), "outcome"].tolist()
    predictors, families, orientations = _predictors(root)
    role = predictors["nfl_position"].eq("T").astype(float)
    predictors = predictors.assign(role_t=role)
    rows: list[dict[str, float | int | str]] = []
    for outcome in accepted:
        score = pd.read_parquet(root / f"data/processed/scope_ol_scores_{outcome}.parquet")
        score = score.loc[:, ["nfl_id", "posterior_mean"]].rename(columns={"posterior_mean": outcome})
        base = predictors.merge(score, on="nfl_id", how="inner", validate="one_to_one")
        for predictor, family in families.items():
            required = list(dict.fromkeys([predictor, outcome, *CONTROLS, "role_t", "draft_year"]))
            core = list(dict.fromkeys([predictor, outcome, "combine_weight", "role_t", "draft_year"]))
            data = base.loc[:, required].dropna(subset=core).copy()
            if len(data) < 40 or data[predictor].nunique() < 5:
                continue
            x_values = data[predictor].astype(float).tolist()
            y_values = data[outcome].astype(float).tolist()
            main_names = [name for name in ("combine_weight", "role_t") if name != predictor]
            reach_names = [name for name in (*main_names, "arm_length", "combine_height") if name != predictor]
            statistic = partial_spearman(x_values, y_values, [data[name].astype(float).tolist() for name in main_names])
            reach_data = data.dropna(subset=reach_names)
            reach_r = _reach_partial(reach_data, predictor, outcome, reach_names)
            periods = ([2023, 2024], [2025], [2023], [2024], [2025])
            replication = [
                _raw_spearman(data.loc[data["draft_year"].isin(years)], predictor, outcome) for years in periods
            ]
            oriented = statistic.r * orientations[predictor] * OUTCOME_ORIENTATION[outcome]
            rows.append(
                {
                    "scope": "ol", "group": "OL", "outcome": outcome, "predictor": predictor, "family": family,
                    "n": len(data), "r_raw": _raw_spearman(data, predictor, outcome), "r_adj": statistic.r,
                    "ci_lo": statistic.ci_lo, "ci_hi": statistic.ci_hi, "p": statistic.p, "q_scope": float("nan"),
                    "oriented_r": oriented
                    if orientations[predictor] and OUTCOME_ORIENTATION[outcome]
                    else float("nan"),
                    "direction": orientation_label(statistic.r, orientations[predictor], OUTCOME_ORIENTATION[outcome]),
                    "r_disc_2023_24": replication[0], "r_hold_2025": replication[1], "r_2023": replication[2],
                    "r_2024": replication[3], "r_2025": replication[4], "partial_reach_r": reach_r,
                }
            )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["q_scope"] = benjamini_hochberg(results["p"].to_numpy())
    columns = [
        "scope", "group", "outcome", "predictor", "family", "n", "r_raw", "r_adj", "ci_lo", "ci_hi", "p",
        "q_scope", "oriented_r", "direction", "r_disc_2023_24", "r_hold_2025", "r_2023", "r_2024", "r_2025",
        "partial_reach_r",
    ]
    results = results.reindex(columns=columns)
    results.to_csv(root / "reports/scope_ol_results.csv", index=False)
    return results


def main(arguments: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "prepare", "gate", "infeasible", "associate"])
    parser.add_argument("--season", type=int)
    parser.add_argument("--outcome")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(arguments)
    if args.stage == "extract":
        if args.season not in {2023, 2024, 2025}:
            raise ValueError("extract requires --season 2023, 2024, or 2025")
        extract_season_tracking(args.root, args.season)
    elif args.stage == "prepare":
        prepare_play_outcomes(args.root)
    elif args.stage == "gate":
        if args.outcome not in (*GATED_OUTCOMES, *BASELINE_OUTCOMES):
            raise ValueError("gate requires a prespecified --outcome")
        gate_outcome(args.root, args.outcome)
    elif args.stage == "infeasible":
        record_infeasible_outcomes(args.root)
    else:
        associate(args.root)


if __name__ == "__main__":
    main()
