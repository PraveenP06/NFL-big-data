from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from combine_dlol.data import add_line_of_scrimmage, iter_game_tracking, load_player_play
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

MOVEMENT_DISPLACEMENT_YARDS = 0.25
MOVEMENT_SPEED_YARDS_PER_SECOND = 0.75
CENSOR_SECONDS = 1.0
PRE_SNAP_SECONDS = 0.3
HALF_SECOND = 0.5
RANDOM_SEED = 20261103
MIN_PLAYS = 50
ADJUSTMENT_NUMERIC = ("down", "yards_to_go", "dropback_duration", "pass_rushers_encountered")
ADJUSTMENT_CATEGORICAL = ("lined_up_position", "offense_formation")
ADJUSTMENT_FORMULA = (
    "OLS residual: outcome ~ C(lined_up_position) + down + yards_to_go + "
    "C(offense_formation) + dropback_duration + pass_rushers_encountered"
)


@dataclass(frozen=True)
class MovementDetection:
    reaction_time: float
    displacement_0_5: float
    censored: bool
    included: bool
    pre_snap_motion: bool


@dataclass(frozen=True)
class ReliabilitySummary:
    median: float
    ci_lower: float
    ci_upper: float
    icc: float
    reliability_n10: float
    reliability_n25: float
    reliability_n50: float
    reliability_n100: float
    permutation_median: float
    verdict: str


def build_population(player_play: pd.DataFrame) -> pd.DataFrame:
    required = [
        "nfl_position",
        "peak_pressure_probability_allowed",
        "play_nullified_by_penalty",
    ]
    if any(column not in player_play for column in required):
        raise ValueError("player-play data lacks a required OL population field")
    mask = (
        player_play["nfl_position"].isin(["T", "G", "C"])
        & player_play["peak_pressure_probability_allowed"].notna()
        & player_play["play_nullified_by_penalty"].ne("Y")
    )
    return player_play.loc[mask].copy()


def fit_adjusted_outcome(
    data: pd.DataFrame,
    outcome: str,
    numeric: Sequence[str] = ADJUSTMENT_NUMERIC,
    categorical: Sequence[str] = ADJUSTMENT_CATEGORICAL,
) -> pd.DataFrame:
    numeric_outcome = pd.to_numeric(data[outcome], errors="coerce").astype(float)
    return adjusted_residuals(data.assign(**{outcome: numeric_outcome}), outcome, numeric, categorical)


def detect_snap_movement(trajectory: pd.DataFrame) -> MovementDetection:
    needed = {"seconds_since_snap", "x_rel", "y_rel", "s"}
    if not needed.issubset(trajectory.columns):
        raise ValueError("trajectory lacks a required movement field")
    ordered = trajectory.loc[:, list(needed)].dropna().sort_values("seconds_since_snap", kind="stable")
    snap = ordered.loc[np.isclose(ordered["seconds_since_snap"].to_numpy(dtype=float), 0.0)]
    pre = ordered.loc[
        ordered["seconds_since_snap"].ge(-PRE_SNAP_SECONDS) & ordered["seconds_since_snap"].lt(0.0)
    ]
    if len(snap) != 1 or pre.empty:
        return MovementDetection(np.nan, np.nan, False, False, False)
    pre_motion = bool(pre["s"].ge(MOVEMENT_SPEED_YARDS_PER_SECOND).any())
    if pre_motion:
        return MovementDetection(np.nan, np.nan, False, False, True)
    post = ordered.loc[
        ordered["seconds_since_snap"].ge(0.0) & ordered["seconds_since_snap"].le(CENSOR_SECONDS)
    ].copy()
    half_window = post.loc[post["seconds_since_snap"].le(HALF_SECOND)]
    if post.empty or half_window.empty or float(half_window["seconds_since_snap"].max()) < HALF_SECOND - 0.011:
        return MovementDetection(np.nan, np.nan, False, False, False)
    snap_x = float(snap.iloc[0]["x_rel"])
    snap_y = float(snap.iloc[0]["y_rel"])
    displacement = np.hypot(post["x_rel"].to_numpy(dtype=float) - snap_x, post["y_rel"].to_numpy(dtype=float) - snap_y)
    moved = (displacement >= MOVEMENT_DISPLACEMENT_YARDS) | (
        post["s"].to_numpy(dtype=float) >= MOVEMENT_SPEED_YARDS_PER_SECOND
    )
    first = post.loc[moved, "seconds_since_snap"]
    censored = first.empty
    reaction = CENSOR_SECONDS if censored else float(first.iloc[0])
    half = half_window.iloc[-1]
    return MovementDetection(reaction, float(half["x_rel"]) - snap_x, censored, True, False)


def _trajectory_with_seconds(group: pd.DataFrame) -> pd.DataFrame | None:
    if "seconds_since_snap" in group:
        return group
    snaps = group.loc[group["event"].eq("ball_snap"), "time"]
    if snaps.empty:
        return None
    timestamps = pd.to_datetime(group["time"], errors="coerce")
    snap_time = pd.to_datetime(snaps.iloc[0], errors="coerce")
    if pd.isna(snap_time) or timestamps.isna().any():
        return None
    return group.assign(seconds_since_snap=(timestamps - snap_time).dt.total_seconds())


def snap_outcomes(tracking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["game_id", "play_id", "nfl_id"]
    timestamps = pd.to_datetime(tracking["time"], errors="coerce")
    timed = tracking.assign(_timestamp=timestamps)
    snap_times = (
        timed.loc[timed["event"].eq("ball_snap"), [*keys, "_timestamp"]]
        .drop_duplicates(keys)
        .rename(columns={"_timestamp": "_snap_timestamp"})
    )
    prepared = timed.merge(snap_times, on=keys, how="left", validate="many_to_one")
    seconds = (prepared["_timestamp"] - prepared["_snap_timestamp"]).dt.total_seconds()
    prepared = prepared.assign(seconds_since_snap=seconds)
    group_keys = prepared.loc[:, keys].drop_duplicates()
    snap = prepared.loc[np.isclose(prepared["seconds_since_snap"], 0.0), [*keys, "x_rel", "y_rel"]]
    snap = snap.drop_duplicates(keys).rename(columns={"x_rel": "_snap_x", "y_rel": "_snap_y"})
    pre = prepared.loc[
        prepared["seconds_since_snap"].ge(-PRE_SNAP_SECONDS) & prepared["seconds_since_snap"].lt(0.0),
        [*keys, "seconds_since_snap", "s"],
    ]
    pre_stats = pre.groupby(keys, as_index=False).agg(
        _pre_min=("seconds_since_snap", "min"), _pre_max_speed=("s", "max")
    )
    half = prepared.loc[
        prepared["seconds_since_snap"].ge(0.0) & prepared["seconds_since_snap"].le(HALF_SECOND),
        [*keys, "seconds_since_snap", "x_rel"],
    ]
    half = half.sort_values("seconds_since_snap", kind="stable").groupby(keys, as_index=False).tail(1)
    half = half.rename(columns={"seconds_since_snap": "_half_time", "x_rel": "_half_x"})
    post = prepared.loc[
        prepared["seconds_since_snap"].ge(0.0) & prepared["seconds_since_snap"].le(CENSOR_SECONDS),
        [*keys, "seconds_since_snap", "x_rel", "y_rel", "s"],
    ].merge(snap, on=keys, how="inner", validate="many_to_one")
    distance = np.hypot(post["x_rel"] - post["_snap_x"], post["y_rel"] - post["_snap_y"])
    moved = (distance >= MOVEMENT_DISPLACEMENT_YARDS) | post["s"].ge(MOVEMENT_SPEED_YARDS_PER_SECOND)
    first = post.loc[moved, [*keys, "seconds_since_snap"]].groupby(keys, as_index=False).min()
    first = first.rename(columns={"seconds_since_snap": "_first_movement"})
    result = group_keys.merge(snap, on=keys, how="left").merge(pre_stats, on=keys, how="left")
    result = result.merge(half.loc[:, [*keys, "_half_time", "_half_x"]], on=keys, how="left")
    result = result.merge(first, on=keys, how="left")
    valid_window = result["_half_time"].ge(HALF_SECOND - 0.011)
    still = result["_pre_max_speed"].lt(MOVEMENT_SPEED_YARDS_PER_SECOND)
    has_snap = result["_snap_x"].notna()
    included = has_snap & still & valid_window
    diagnostics = pd.DataFrame(
        {
            "missing_snap": (~has_snap | ~valid_window | ~result["_pre_min"].notna()).astype(int),
            "pre_snap_motion": (has_snap & result["_pre_max_speed"].ge(MOVEMENT_SPEED_YARDS_PER_SECOND)).astype(int),
            "censored": (included & result["_first_movement"].isna()).astype(int),
        }
    )
    output = result.loc[included, keys].copy()
    output["snap_reaction_time"] = result.loc[included, "_first_movement"].fillna(CENSOR_SECONDS).to_numpy()
    output["first_0_5s_displacement"] = (result.loc[included, "_half_x"] - result.loc[included, "_snap_x"]).to_numpy()
    output["reaction_censored"] = result.loc[included, "_first_movement"].isna().astype(int).to_numpy()
    return output, diagnostics


def summarize_movement(
    residuals: pd.DataFrame,
    value: str,
    min_plays: int = MIN_PLAYS,
    n_splits: int = 2000,
    n_bootstrap: int = 1000,
    n_permutations: int = 0,
) -> ReliabilitySummary:
    split = split_half_reliability(residuals, "nfl_id", value, min_plays, n_splits=n_splits, seed=RANDOM_SEED)
    lower, upper = player_bootstrap_ci(split, n_bootstrap=n_bootstrap, seed=RANDOM_SEED + 1)
    model = mixed_model_icc(residuals, "nfl_id", value)
    null_median = float("nan")
    if n_permutations:
        null = permutation_null(
            residuals,
            "nfl_id",
            value,
            min_plays,
            n_permutations=n_permutations,
            n_splits=100,
            seed=RANDOM_SEED + 2,
        )
        null_median = float(np.nanmedian(null))
    return ReliabilitySummary(
        median=split.median,
        ci_lower=lower,
        ci_upper=upper,
        icc=model.icc,
        reliability_n10=float(mean_reliability(model.icc, 10)),
        reliability_n25=float(mean_reliability(model.icc, 25)),
        reliability_n50=float(mean_reliability(model.icc, 50)),
        reliability_n100=float(mean_reliability(model.icc, 100)),
        permutation_median=null_median,
        verdict=verdict(split.median, lower),
    )


def _tracking_columns() -> tuple[str, ...]:
    return ("game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event")


def _tracking_for_season(population: pd.DataFrame, season: int) -> Iterable[pd.DataFrame]:
    context = population.loc[population["season"].eq(season)].copy()
    player_ids = context["nfl_id"].dropna().astype(int).unique().tolist()
    keys = ["game_id", "play_id", "nfl_id"]
    for chunk in iter_game_tracking(season, _tracking_columns(), player_ids=player_ids, chunk_size=5_000_000):
        selected = chunk.merge(context.loc[:, keys].drop_duplicates(), on=keys, how="inner")
        if not selected.empty:
            yield add_line_of_scrimmage(selected, context)


def _complete_trajectories(chunks: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
    pending = pd.DataFrame()
    keys = ["game_id", "play_id", "nfl_id"]
    for chunk in chunks:
        joined = pd.concat([pending, chunk], ignore_index=True) if not pending.empty else chunk
        last = joined.iloc[-1]
        last_mask = np.logical_and.reduce([joined[key].eq(last[key]).to_numpy() for key in keys])
        complete = joined.loc[~last_mask]
        if not complete.empty:
            yield complete
        pending = joined.loc[last_mask].copy()
    if not pending.empty:
        yield pending


def _run_snap_extraction(population: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for season in (2023, 2024, 2025):
        for chunk in _complete_trajectories(_tracking_for_season(population, season)):
            extracted, details = snap_outcomes(chunk)
            if not extracted.empty:
                features.append(extracted)
            if not details.empty:
                diagnostics.append(details)
    feature_frame = pd.concat(features, ignore_index=True) if features else pd.DataFrame()
    diagnostic_frame = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    return feature_frame, diagnostic_frame


def _analysis_population() -> pd.DataFrame:
    columns = [
        "game_id",
        "play_id",
        "nfl_id",
        "season",
        "nfl_position",
        "peak_pressure_probability_allowed",
        "pressure_allowed",
        "play_nullified_by_penalty",
        "lined_up_position",
        "down",
        "yards_to_go",
        "offense_formation",
        "dropback_duration",
        "pass_rushers_encountered",
        "yardline_side",
        "yardline_number",
        "possession_team",
        "play_direction",
        "forty",
    ]
    return build_population(load_player_play(columns))


def _summary_row(
    outcome: str, subgroup: str, summary: ReliabilitySummary, n_players: int, n_plays: int
) -> dict[str, float | int | str]:
    return {
        "outcome": outcome,
        "subgroup": subgroup,
        "eligible_players": n_players,
        "eligible_plays": n_plays,
        "split_half_spearman_brown_median": summary.median,
        "bootstrap_ci_lower": summary.ci_lower,
        "bootstrap_ci_upper": summary.ci_upper,
        "mixedlm_icc": summary.icc,
        "mean_reliability_n10": summary.reliability_n10,
        "mean_reliability_n25": summary.reliability_n25,
        "mean_reliability_n50": summary.reliability_n50,
        "mean_reliability_n100": summary.reliability_n100,
        "permutation_null_median": summary.permutation_median,
        "verdict": summary.verdict,
        "adjustment_formula": ADJUSTMENT_FORMULA,
    }


def _eligible(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    counts = data.groupby("nfl_id")[outcome].count()
    return data.loc[data["nfl_id"].isin(counts.loc[counts.ge(MIN_PLAYS)].index)].copy()


def _reliability_table(adjusted_by_outcome: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for outcome, adjusted in adjusted_by_outcome.items():
        groups = {
            "all": adjusted,
            "T": adjusted.loc[adjusted["nfl_position"].eq("T")],
            "G_C": adjusted.loc[adjusted["nfl_position"].isin(["G", "C"])],
        }
        for subgroup, data in groups.items():
            usable = _eligible(data, outcome)
            if usable["nfl_id"].nunique() < 3:
                continue
            summary = summarize_movement(usable, outcome, n_permutations=500)
            rows.append(_summary_row(outcome, subgroup, summary, usable["nfl_id"].nunique(), len(usable)))
    return pd.DataFrame(rows)


def _scores(adjusted_by_outcome: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result: list[pd.DataFrame] = []
    for outcome, adjusted in adjusted_by_outcome.items():
        usable = _eligible(adjusted, outcome)
        model = mixed_model_icc(usable, "nfl_id", outcome)
        score = empirical_bayes_shrink(usable, "nfl_id", outcome, model, min_n=MIN_PLAYS)
        score = score.assign(outcome=outcome, mixedlm_icc=model.icc)
        result.append(score)
    return pd.concat(result, ignore_index=True)


def _plot_reliability(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = table.loc[table["subgroup"].eq("all")].copy()
    labels = all_rows["outcome"].str.replace("_residual", "", regex=False).tolist()
    values = all_rows["split_half_spearman_brown_median"].to_numpy(dtype=float)
    lowers = all_rows["bootstrap_ci_lower"].to_numpy(dtype=float)
    uppers = all_rows["bootstrap_ci_upper"].to_numpy(dtype=float)
    positions = np.arange(len(all_rows))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.errorbar(positions, values, yerr=[values - lowers, uppers - values], fmt="o", capsize=4)
    axis.axhline(0.5, color="black", linewidth=1)
    axis.axhline(0.3, color="gray", linewidth=1, linestyle="--")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("split-half Spearman-Brown reliability")
    axis.set_ylim(-0.2, 1.05)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_snap_examples(population: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RANDOM_SEED)
    candidates: list[pd.DataFrame] = []
    included = 0
    for season in (2023, 2024, 2025):
        for chunk in _complete_trajectories(_tracking_for_season(population, season)):
            for _, group in chunk.groupby(["game_id", "play_id", "nfl_id"], sort=False):
                trajectory = _trajectory_with_seconds(group)
                if trajectory is None:
                    continue
                detected = detect_snap_movement(trajectory)
                if detected.included:
                    selected = trajectory.assign(detected_reaction=detected.reaction_time)
                    included += 1
                    if len(candidates) < 12:
                        candidates.append(selected)
                    else:
                        replacement = int(rng.integers(0, included))
                        if replacement < len(candidates):
                            candidates[replacement] = selected
    figure, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True)
    for axis, trajectory in zip(axes.flat, candidates, strict=False):
        at_snap = trajectory.loc[np.isclose(trajectory["seconds_since_snap"], 0.0)].iloc[0]
        displacement = np.hypot(trajectory["x_rel"] - at_snap["x_rel"], trajectory["y_rel"] - at_snap["y_rel"])
        axis.plot(trajectory["seconds_since_snap"], displacement)
        axis.axvline(float(trajectory["detected_reaction"].iloc[0]), color="tab:red")
        axis.set_xlim(-0.3, 1.0)
    for axis in axes.flat:
        axis.set_xlabel("seconds since snap")
        axis.set_ylabel("displacement (yd)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_phase_one(root: Path = Path(".")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    population = _analysis_population()
    snap, diagnostics = _run_snap_extraction(population)
    combined = population.merge(snap, on=["game_id", "play_id", "nfl_id"], how="inner", validate="one_to_one")
    outcomes = [
        "peak_pressure_probability_allowed",
        "pressure_allowed",
        "snap_reaction_time",
        "first_0_5s_displacement",
    ]
    adjusted = {f"{outcome}_residual": fit_adjusted_outcome(combined, outcome) for outcome in outcomes}
    table = _reliability_table(adjusted)
    scores = _scores(adjusted)
    counts = pd.DataFrame(
        [
            {
                "population_plays": len(population),
                "population_players": population["nfl_id"].nunique(),
                "tracking_plays": len(combined),
                "tracking_players": combined["nfl_id"].nunique(),
                "pre_snap_motion_dropped": int(diagnostics.get("pre_snap_motion", pd.Series(dtype=int)).sum()),
                "missing_snap_or_window_dropped": int(diagnostics.get("missing_snap", pd.Series(dtype=int)).sum()),
                "reaction_censored_share": float(combined["reaction_censored"].mean()),
                "eligible_ol_any_outcome": int(scores["nfl_id"].nunique()),
                "eligible_ol_nonnull_forty": int(
                    population.loc[population["nfl_id"].isin(scores["nfl_id"]), ["nfl_id", "forty"]]
                    .drop_duplicates("nfl_id")["forty"]
                    .notna()
                    .sum()
                ),
            }
        ]
    )
    reports = root / "reports"
    processed = root / "data" / "processed"
    reports.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(processed / "ol_outcomes_scores.parquet", index=False)
    table.to_csv(reports / "phase1_ol_reliability.csv", index=False)
    counts.to_csv(reports / "phase1_ol_counts.csv", index=False)
    _plot_reliability(table, reports / "phase1_ol_reliability.png")
    _plot_snap_examples(population, reports / "phase1_ol_snap.png")
    return table, counts, scores


if __name__ == "__main__":
    run_phase_one()
