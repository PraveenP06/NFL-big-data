from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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

DL_POSITIONS = ("DE", "DT", "NT", "OLB")
CONTEXT_CATEGORICAL = ("lined_up_position", "down", "offense_formation", "blitzing")
CONTEXT_CONTINUOUS = ("yards_to_go", "snap_x_rel", "snap_y_rel")
OUTCOMES = {
    "d_a": "player_get_off",
    "d_b_displacement_0_5": "displacement_0_5",
    "d_b_displacement_1_0": "displacement_1_0",
    "d_b_speed_0_5": "speed_0_5",
}
PLOT_SEED = 20261106


def _keys() -> list[str]:
    return ["game_id", "play_id", "nfl_id"]


def prepare_dl_population(player_play: pd.DataFrame) -> pd.DataFrame:
    required = [*_keys(), "nfl_position", "player_get_off", "pass_result", "play_nullified_by_penalty"]
    missing = set(required).difference(player_play.columns)
    if missing:
        raise ValueError(f"DL population is missing columns: {sorted(missing)}")
    selected = player_play.loc[
        player_play["nfl_position"].isin(DL_POSITIONS)
        & player_play["player_get_off"].notna()
        & player_play["pass_result"].notna()
        & player_play["play_nullified_by_penalty"].ne("Y")
    ].copy()
    return selected.drop_duplicates(_keys()).reset_index(drop=True)


def _nearest_value(frame: pd.DataFrame, target: float, column: str) -> float:
    candidate = frame.loc[frame["seconds_since_snap"].ge(0.0)].copy()
    if candidate.empty:
        return float("nan")
    distance = (candidate["seconds_since_snap"] - target).abs()
    position = int(np.argmin(distance.to_numpy(dtype=float)))
    if float(distance.iloc[position]) > 0.051:
        return float("nan")
    return float(candidate.iloc[position][column])


def _first_step_for_play(frame: pd.DataFrame) -> dict[str, float | int]:
    sorted_frame = frame.sort_values("time").copy()
    event = sorted_frame["event"].fillna("").astype(str).str.lower()
    snaps = sorted_frame.loc[event.eq("ball_snap")]
    if snaps.empty:
        return {}
    snap = snaps.iloc[0]
    times = pd.to_datetime(sorted_frame["time"], utc=True)
    snap_time = pd.to_datetime(snap["time"], utc=True)
    sorted_frame["seconds_since_snap"] = (times - snap_time).dt.total_seconds()
    snap_x = _nearest_value(sorted_frame, 0.0, "x_rel")
    x_04 = _nearest_value(sorted_frame, 0.4, "x_rel")
    x_05 = _nearest_value(sorted_frame, 0.5, "x_rel")
    x_06 = _nearest_value(sorted_frame, 0.6, "x_rel")
    x_10 = _nearest_value(sorted_frame, 1.0, "x_rel")
    time_04 = _nearest_value(sorted_frame, 0.4, "seconds_since_snap")
    time_06 = _nearest_value(sorted_frame, 0.6, "seconds_since_snap")
    coordinate_speed = float("nan")
    if np.isfinite(x_04) and np.isfinite(x_06) and time_06 > time_04:
        coordinate_speed = (x_06 - x_04) / (time_06 - time_04)
    supplied_speed = _nearest_value(sorted_frame, 0.5, "s") if "s" in sorted_frame else float("nan")
    output: dict[str, float | int] = {
        "game_id": int(snap["game_id"]),
        "play_id": int(snap["play_id"]),
        "nfl_id": int(snap["nfl_id"]),
        "snap_x_rel": snap_x,
        "snap_y_rel": float(snap["y_rel"]) if "y_rel" in sorted_frame else float("nan"),
        "displacement_0_5": x_05 - snap_x if np.isfinite(x_05) and np.isfinite(snap_x) else float("nan"),
        "displacement_1_0": x_10 - snap_x if np.isfinite(x_10) and np.isfinite(snap_x) else float("nan"),
        "speed_0_5": coordinate_speed,
        "supplied_speed_0_5": supplied_speed,
    }
    return output


def first_step_metrics(tracking: pd.DataFrame) -> pd.DataFrame:
    required = {*_keys(), "time", "x_rel", "event"}
    missing = required.difference(tracking.columns)
    if missing:
        raise ValueError(f"tracking is missing columns: {sorted(missing)}")
    rows = [_first_step_for_play(group) for _, group in tracking.groupby(_keys(), sort=False)]
    rows = [row for row in rows if row]
    return pd.DataFrame(rows)


def adjust_dl_outcome(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    return adjusted_residuals(frame, outcome, CONTEXT_CONTINUOUS, CONTEXT_CATEGORICAL)


def _metric_frame_for_season(
    season: int,
    population: pd.DataFrame,
    player_play: pd.DataFrame,
    root: Path | None,
) -> pd.DataFrame:
    season_population = population.loc[population["season"].eq(season)].copy()
    if season_population.empty:
        return pd.DataFrame()
    key_frame = season_population.loc[:, _keys()].drop_duplicates()
    play_context = player_play.loc[
        :,
        [
            "game_id",
            "play_id",
            "yardline_side",
            "yardline_number",
            "possession_team",
            "play_direction",
        ],
    ].drop_duplicates(["game_id", "play_id"])
    collected: list[pd.DataFrame] = []
    columns = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"]
    player_ids = season_population["nfl_id"].drop_duplicates().astype(int).tolist()
    for batch in iter_game_tracking(season, columns=columns, player_ids=player_ids, root=root):
        selected = batch.merge(key_frame, on=_keys(), how="inner", validate="many_to_one")
        if not selected.empty:
            collected.append(add_line_of_scrimmage(selected, play_context))
    if not collected:
        return pd.DataFrame()
    return first_step_metrics(pd.concat(collected, ignore_index=True))


def tracking_outcomes(
    population: pd.DataFrame,
    player_play: pd.DataFrame,
    root: Path | None = None,
) -> pd.DataFrame:
    frames = [_metric_frame_for_season(season, population, player_play, root) for season in (2023, 2024, 2025)]
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def summarize_outcome(
    residuals: pd.DataFrame,
    value_column: str,
    n_splits: int = 2000,
    n_bootstrap: int = 1000,
    n_permutations: int = 500,
) -> dict[str, float | str]:
    split = split_half_reliability(residuals, "nfl_id", value_column, min_n=50, n_splits=n_splits)
    lower, upper = player_bootstrap_ci(split, n_bootstrap=n_bootstrap)
    model = mixed_model_icc(residuals, "nfl_id", value_column)
    null = permutation_null(
        residuals,
        "nfl_id",
        value_column,
        min_n=50,
        n_permutations=n_permutations,
    )
    return {
        "n_players": float(residuals["nfl_id"].nunique()),
        "n_plays": float(len(residuals)),
        "split_half_median": split.median,
        "split_half_ci_lower": lower,
        "split_half_ci_upper": upper,
        "icc": model.icc,
        "mean_reliability_n10": float(mean_reliability(model.icc, 10)),
        "mean_reliability_n25": float(mean_reliability(model.icc, 25)),
        "mean_reliability_n50": float(mean_reliability(model.icc, 50)),
        "mean_reliability_n100": float(mean_reliability(model.icc, 100)),
        "permutation_null_median": float(np.nanmedian(null)),
        "permutation_null_lower": float(np.nanquantile(null, 0.025)),
        "permutation_null_upper": float(np.nanquantile(null, 0.975)),
        "verdict": verdict(split.median, lower),
    }


def _eligible_residuals(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    adjusted = adjust_dl_outcome(frame, outcome)
    residual = f"{outcome}_residual"
    counts = adjusted.groupby("nfl_id")[residual].transform("size")
    return adjusted.loc[counts.ge(50)].copy()


def _subgroup(frame: pd.DataFrame, positions: Sequence[str]) -> pd.DataFrame:
    return frame.loc[frame["nfl_position"].isin(positions)].copy()


def _scores(residuals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for label, outcome in OUTCOMES.items():
        frame = residuals[label]
        model = mixed_model_icc(frame, "nfl_id", f"{outcome}_residual")
        score = empirical_bayes_shrink(frame, "nfl_id", f"{outcome}_residual", model, min_n=50).rename(
            columns={
                "n": f"n_plays_{label}",
                "raw_mean": f"raw_{label}",
                "reliability": f"reliability_{label}",
                "posterior_mean": f"shrunk_{label}",
            }
        )
        score = score.loc[:, ["nfl_id", f"n_plays_{label}", f"raw_{label}", f"reliability_{label}", f"shrunk_{label}"]]
        merged = score if merged is None else merged.merge(score, on="nfl_id", how="outer", validate="one_to_one")
    assert merged is not None
    n_columns = [column for column in merged.columns if column.startswith("n_plays_")]
    merged["n_plays"] = merged[n_columns].min(axis=1)
    excluded = {"nfl_id", "n_plays", *n_columns}
    ordered = ["nfl_id", "n_plays", *[column for column in merged.columns if column not in excluded]]
    return merged.loc[:, ordered].sort_values("nfl_id").reset_index(drop=True)


def _reliability_rows(residuals: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, float | str]] = []
    split_results: dict[str, Any] = {}
    for label, outcome in OUTCOMES.items():
        frame = residuals[label]
        groups = (
            ("all", frame),
            ("EDGE", _subgroup(frame, ("DE", "OLB"))),
            ("IDL", _subgroup(frame, ("DT", "NT"))),
        )
        for group, selected in groups:
            summary = summarize_outcome(selected, f"{outcome}_residual")
            summary.update({"outcome": label, "group": group})
            rows.append(summary)
        split_results[label] = split_half_reliability(frame, "nfl_id", f"{outcome}_residual", min_n=50, n_splits=2000)
    return pd.DataFrame(rows), split_results


def _validity(reliability: pd.DataFrame, matched: pd.DataFrame, residuals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | bool | str]] = []
    for label, outcome in OUTCOMES.items():
        if label == "d_a":
            continue
        raw = float(cast(Any, matched.loc[:, ["player_get_off", outcome]].dropna().corr().iloc[0, 1]))
        adjusted = (
            residuals["d_a"]
            .loc[:, _keys() + ["player_get_off_residual"]]
            .merge(
                residuals[label].loc[:, _keys() + [f"{outcome}_residual"]],
                on=_keys(),
                how="inner",
                validate="one_to_one",
            )
        )
        correlation_value = adjusted.loc[:, ["player_get_off_residual", f"{outcome}_residual"]].corr().iloc[0, 1]
        residual_correlation = float(cast(Any, correlation_value))
        rows.append(
            {
                "outcome": label,
                "group": "all",
                "validity_raw_pearson": raw,
                "validity_adjusted_pearson": residual_correlation,
                "validity_raw_absolute_pass": abs(raw) >= 0.5,
            }
        )
    return reliability.merge(pd.DataFrame(rows), on=["outcome", "group"], how="left", validate="one_to_one")


def _counts(population: pd.DataFrame, matched: pd.DataFrame, residuals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    distribution = matched.groupby("nfl_id").size()
    eligible = set(residuals["d_a"]["nfl_id"])
    eligible_rows = matched.loc[matched["nfl_id"].isin(eligible)]
    forty_players = eligible_rows.loc[
        eligible_rows["draft_year"].isin([2023, 2024, 2025]) & eligible_rows["forty"].notna(), "nfl_id"
    ].nunique()
    rows = [
        {"measure": "candidate_dl_pass_rows", "value": len(population)},
        {"measure": "matched_tracking_rows", "value": len(matched)},
        {"measure": "matched_players", "value": matched["nfl_id"].nunique()},
        {"measure": "eligible_players_min_50", "value": len(eligible)},
        {"measure": "eligible_players_draft_2023_2025_with_nonnull_forty", "value": forty_players},
    ]
    for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows.append(
            {
                "measure": f"matched_plays_per_player_q{int(quantile * 100):02d}",
                "value": distribution.quantile(quantile),
            }
        )
    return pd.DataFrame(rows)


def _plot_reliability(split_results: dict[str, Any], residuals: dict[str, pd.DataFrame], path: Path) -> None:
    figure, axes = plt.subplots(len(OUTCOMES), 2, figsize=(10, 14), constrained_layout=True)
    for row, (label, outcome) in enumerate(OUTCOMES.items()):
        split = split_results[label]
        scatter = axes[row, 0]
        representative = split.representative
        scatter.scatter(representative["half_a"], representative["half_b"], s=12, alpha=0.65)
        scatter.set(title=f"{label}: split-half means", xlabel="half A residual", ylabel="half B residual")
        null = permutation_null(residuals[label], "nfl_id", f"{outcome}_residual", 50, n_permutations=500)
        histogram = axes[row, 1]
        histogram.hist(null[np.isfinite(null)], bins=25, color="0.65")
        histogram.axvline(split.median, color="tab:red")
        histogram.set(title=f"{label}: permutation null", xlabel="Spearman-Brown reliability", ylabel="count")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_trajectories(matched: pd.DataFrame, player_play: pd.DataFrame, root: Path | None, path: Path) -> None:
    candidate_columns = _keys() + ["nfl_position", "player_get_off", "season"]
    candidates = matched.loc[matched["nfl_position"].isin(["DE", "DT"]), candidate_columns]
    rng = np.random.default_rng(PLOT_SEED)
    samples: list[pd.DataFrame] = []
    for position in ("DE", "DT"):
        subset = candidates.loc[candidates["nfl_position"].eq(position)]
        if not subset.empty:
            count = min(6, len(subset))
            samples.append(subset.iloc[rng.choice(len(subset), size=count, replace=False)])
    selected = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.flat)
    play_context_columns = [
        "game_id",
        "play_id",
        "yardline_side",
        "yardline_number",
        "possession_team",
        "play_direction",
    ]
    play_context = player_play.loc[:, play_context_columns].drop_duplicates(["game_id", "play_id"])
    trajectories: list[pd.DataFrame] = []
    for season in (2023, 2024, 2025):
        season_selected = selected.loc[selected["season"].eq(season)]
        if season_selected.empty:
            continue
        key_frame = season_selected.loc[:, _keys()]
        identifiers = season_selected["nfl_id"].astype(int).tolist()
        tracking_columns = ["game_id", "play_id", "nfl_id", "time", "x", "y", "event"]
        for batch in iter_game_tracking(season, columns=tracking_columns, player_ids=identifiers, root=root):
            subset = batch.merge(key_frame, on=_keys(), how="inner", validate="many_to_one")
            if not subset.empty:
                trajectories.append(add_line_of_scrimmage(subset, play_context))
    if trajectories:
        tracked = pd.concat(trajectories, ignore_index=True)
        for axis, (_, sample) in zip(flat_axes, selected.iterrows(), strict=False):
            annotations = sample.to_frame().T.loc[:, _keys() + ["nfl_position", "player_get_off"]]
            trajectory = tracked.merge(annotations, on=_keys(), how="inner")
            if not trajectory.empty:
                times = pd.to_datetime(trajectory["time"], utc=True)
                snap_times = trajectory.loc[trajectory["event"].eq("ball_snap"), "time"]
                if not snap_times.empty:
                    seconds = (times - pd.to_datetime(snap_times.iloc[0], utc=True)).dt.total_seconds()
                    window = trajectory.loc[seconds.between(-0.1, 1.05)]
                    axis.plot(seconds.loc[window.index], window["x_rel"])
                axis.set_title(f"{sample['nfl_position']} get-off {sample['player_get_off']:.3f}")
    for axis in flat_axes:
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set(xlabel="seconds since snap", ylabel="x_rel yards")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_phase(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        *_keys(),
        "nfl_position",
        "player_get_off",
        "pass_result",
        "play_nullified_by_penalty",
        "lined_up_position",
        "down",
        "yards_to_go",
        "offense_formation",
        "blitzing",
        "yardline_side",
        "yardline_number",
        "possession_team",
        "play_direction",
        "season",
        "draft_year",
        "forty",
    ]
    player_play, population, complete, residuals, shared_players = _phase_data(columns, root)
    reliability, split_results = _reliability_rows(residuals)
    reliability = _validity(reliability, complete, residuals)
    counts = _counts(population, complete, residuals)
    scores = _scores(residuals)
    processed = (Path.cwd() if root is None else root) / "data" / "processed"
    reports = (Path.cwd() if root is None else root) / "reports"
    scores.to_parquet(processed / "dl_outcomes_scores.parquet", index=False)
    reliability.to_csv(reports / "phase1_dl_reliability.csv", index=False)
    counts.to_csv(reports / "phase1_dl_counts.csv", index=False)
    _plot_reliability(split_results, residuals, reports / "phase1_dl_reliability.png")
    plot_matched = complete.loc[complete["nfl_id"].isin(shared_players)]
    _plot_trajectories(plot_matched, player_play, root, reports / "phase1_dl_firststep.png")
    counts = _with_speed_comparison(counts, complete)
    counts.to_csv(reports / "phase1_dl_counts.csv", index=False)
    return reliability, counts


def _phase_data(
    columns: Sequence[str], root: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], set[Any]]:
    player_play = load_player_play(columns, root=root)
    population = prepare_dl_population(player_play)
    metrics = tracking_outcomes(population, player_play, root)
    if metrics.empty:
        raise ValueError("no tracking outcomes were produced for the frozen DL population")
    matched = population.merge(metrics, on=_keys(), how="inner", validate="one_to_one")
    required_complete = [
        "player_get_off",
        "displacement_0_5",
        "displacement_1_0",
        "speed_0_5",
        *CONTEXT_CATEGORICAL,
        *CONTEXT_CONTINUOUS,
    ]
    complete = matched.dropna(subset=required_complete).copy()
    residuals = {label: _eligible_residuals(complete, outcome) for label, outcome in OUTCOMES.items()}
    shared_players = set.intersection(*(set(frame["nfl_id"]) for frame in residuals.values()))
    selected_residuals = {
        label: frame.loc[frame["nfl_id"].isin(shared_players)].copy() for label, frame in residuals.items()
    }
    return player_play, population, complete, selected_residuals, shared_players


def _with_speed_comparison(counts: pd.DataFrame, complete: pd.DataFrame) -> pd.DataFrame:
    comparison_value = complete.loc[:, ["speed_0_5", "supplied_speed_0_5"]].dropna().corr().iloc[0, 1]
    comparison = float(cast(Any, comparison_value))
    row = pd.DataFrame([{"measure": "coordinate_vs_supplied_speed_0_5_pearson", "value": comparison}])
    return pd.concat([counts.loc[~counts["measure"].eq(row.iloc[0]["measure"])], row], ignore_index=True)


def render_phase(root: Path | None = None) -> pd.DataFrame:
    columns = [
        *_keys(),
        "nfl_position",
        "player_get_off",
        "pass_result",
        "play_nullified_by_penalty",
        "lined_up_position",
        "down",
        "yards_to_go",
        "offense_formation",
        "blitzing",
        "yardline_side",
        "yardline_number",
        "possession_team",
        "play_direction",
        "season",
        "draft_year",
        "forty",
    ]
    player_play, _, complete, residuals, shared_players = _phase_data(columns, root)
    split_results: dict[str, Any] = {}
    for label, outcome in OUTCOMES.items():
        split_results[label] = split_half_reliability(
            residuals[label], "nfl_id", f"{outcome}_residual", min_n=50, n_splits=2000
        )
    reports = (Path.cwd() if root is None else root) / "reports"
    _plot_reliability(split_results, residuals, reports / "phase1_dl_reliability.png")
    plot_matched = complete.loc[complete["nfl_id"].isin(shared_players)]
    _plot_trajectories(plot_matched, player_play, root, reports / "phase1_dl_firststep.png")
    counts = pd.read_csv(reports / "phase1_dl_counts.csv")
    counts = _with_speed_comparison(counts, complete)
    counts.to_csv(reports / "phase1_dl_counts.csv", index=False)
    return counts


def main() -> None:
    run_phase()


if __name__ == "__main__":
    main()
