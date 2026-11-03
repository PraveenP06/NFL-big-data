from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from combine_dlol.data import load_combine_results, load_combine_tracking, load_players
from combine_dlol.reliability import mixed_model_icc, player_bootstrap_ci, split_half_reliability

GROUP_POSITIONS: dict[str, frozenset[str]] = {
    "DL": frozenset({"DE", "DT", "NT", "OLB"}),
    "OL": frozenset({"T", "G", "C"}),
}
OFFICIAL_MEASURES = (
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
STANDARD_DRILLS = frozenset({"FORTY_YARD_DASH", "THREE_CONE_DRILL", "SHORT_SHUTTLE"})
DL_DRILLS = frozenset(
    {
        "PASS_RUSH_DRILL",
        "RUN_THE_HOOP_DRILL",
        "BODY_CONTROL_DRILL",
        "FOUR_BAG_AGILITY_DRILL",
        "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION",
        "BACK_PEDAL_AND_REACT",
        "RUN_AND_CLUB_DRILL",
    }
)
OL_DRILLS = frozenset(
    {
        "PASS_RUSH_DROP",
        "PASS_PRO_MIRROR_DRILL",
        "PASS_PRO-MIRROR_DRILL",
        "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE",
        "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT",
        "OL_PULL_DRILL_FOLD_BLOCK_RIGHT",
        "OL_PULL_DRILL_LONG_PULL_RIGHT",
        "SCREEN_DRILL",
    }
)
BASE_METRICS = (
    "time_in_drill",
    "top_speed",
    "burst_distance_0_1s",
    "burst_speed_1_0s",
    "peak_acceleration",
    "peak_deceleration",
    "path_length",
    "path_efficiency",
)
FORTY_METRICS = ("sensor_ten_yd_time", "sensor_ten_yd_burst_speed")
SPEED_ACCELERATION_BURST_METRICS = frozenset(
    {
        "top_speed",
        "burst_distance_0_1s",
        "burst_speed_1_0s",
        "peak_acceleration",
        "peak_deceleration",
        "sensor_ten_yd_time",
        "sensor_ten_yd_burst_speed",
    }
)


@dataclass(frozen=True)
class BuildResult:
    features: pd.DataFrame
    attempts: pd.DataFrame
    repeatability: pd.DataFrame
    coverage: pd.DataFrame
    predictors: pd.DataFrame
    forty_correlations: pd.DataFrame
    mirror_comparison: pd.DataFrame


def slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _clean_trajectory(
    time: Sequence[float] | np.ndarray[Any, Any],
    x: Sequence[float] | np.ndarray[Any, Any],
    y: Sequence[float] | np.ndarray[Any, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = pd.DataFrame({"time": time, "x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    values = values.sort_values("time").drop_duplicates("time", keep="last")
    clean_time = values["time"].to_numpy(dtype=float)
    clean_x = values["x"].to_numpy(dtype=float)
    clean_y = values["y"].to_numpy(dtype=float)
    if len(clean_time) < 3 or np.any(np.diff(clean_time) <= 0.0):
        return np.array([]), np.array([]), np.array([])
    return clean_time, clean_x, clean_y


def centered_smoothed_speed(
    time: Sequence[float] | np.ndarray[Any, Any],
    x: Sequence[float] | np.ndarray[Any, Any],
    y: Sequence[float] | np.ndarray[Any, Any],
) -> np.ndarray:
    clean_time, clean_x, clean_y = _clean_trajectory(time, x, y)
    if len(clean_time) < 3:
        return np.array([])
    velocity_x = np.gradient(clean_x, clean_time)
    velocity_y = np.gradient(clean_y, clean_time)
    raw_speed = np.hypot(velocity_x, velocity_y)
    return np.convolve(np.pad(raw_speed, 1, mode="edge"), np.array([0.25, 0.5, 0.25]), mode="valid")


def first_sustained_movement(speed: Sequence[float] | np.ndarray[Any, Any], threshold: float = 0.75) -> int:
    values = np.asarray(speed, dtype=float)
    moving = values >= threshold
    for index in range(max(0, len(moving) - 2)):
        if bool(np.all(moving[index : index + 3])):
            return index
    return 0


def _path_length(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if len(x) >= 2 else 0.0


def path_efficiency(
    x: Sequence[float] | np.ndarray[Any, Any], y: Sequence[float] | np.ndarray[Any, Any]
) -> float:
    values_x = np.asarray(x, dtype=float)
    values_y = np.asarray(y, dtype=float)
    length = _path_length(values_x, values_y)
    if len(values_x) < 2 or length <= 0.0:
        return float("nan")
    return float(np.hypot(values_x[-1] - values_x[0], values_y[-1] - values_y[0]) / length)


def _interpolated_path_distance(time: np.ndarray, x: np.ndarray, y: np.ndarray, seconds: float) -> float:
    target = time[0] + seconds
    if target > time[-1]:
        return float("nan")
    cumulative = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    return float(np.interp(target, time, cumulative))


def _interpolated_speed(time: np.ndarray, speed: np.ndarray, seconds: float) -> float:
    target = time[0] + seconds
    if target > time[-1]:
        return float("nan")
    return float(np.interp(target, time, speed))


def _ten_yard_split(time: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    displacement = np.hypot(x - x[0], y - y[0])
    reached = np.flatnonzero(displacement >= 10.0)
    if len(reached) == 0:
        return float("nan"), float("nan")
    index = int(reached[0])
    if index == 0:
        return 0.0, float("nan")
    distance_delta = displacement[index] - displacement[index - 1]
    if distance_delta <= 0.0:
        return float("nan"), float("nan")
    fraction = (10.0 - displacement[index - 1]) / distance_delta
    elapsed = (time[index - 1] + fraction * (time[index] - time[index - 1])) - time[0]
    if elapsed <= 0.0:
        return float("nan"), float("nan")
    return float(elapsed), float(10.0 / elapsed)


def attempt_kinematics(
    time: Sequence[float] | np.ndarray[Any, Any],
    x: Sequence[float] | np.ndarray[Any, Any],
    y: Sequence[float] | np.ndarray[Any, Any],
    drill: str,
) -> dict[str, float]:
    clean_time, clean_x, clean_y = _clean_trajectory(time, x, y)
    speed = centered_smoothed_speed(clean_time, clean_x, clean_y)
    if len(speed) < 3:
        return {}
    start = first_sustained_movement(speed)
    trimmed_time = clean_time[start:]
    trimmed_x = clean_x[start:]
    trimmed_y = clean_y[start:]
    trimmed_speed = speed[start:]
    if len(trimmed_speed) < 3:
        return {}
    acceleration = np.gradient(trimmed_speed, trimmed_time)
    smoothed_acceleration = np.convolve(
        np.pad(acceleration, 1, mode="edge"), np.array([0.25, 0.5, 0.25]), mode="valid"
    )
    values = {
        "time_in_drill": float(trimmed_time[-1] - trimmed_time[0]),
        "top_speed": float(np.max(trimmed_speed)),
        "burst_distance_0_1s": _interpolated_path_distance(trimmed_time, trimmed_x, trimmed_y, 1.0),
        "burst_speed_1_0s": _interpolated_speed(trimmed_time, trimmed_speed, 1.0),
        "peak_acceleration": float(max(0.0, np.max(smoothed_acceleration))),
        "peak_deceleration": float(max(0.0, np.max(-smoothed_acceleration))),
        "path_length": _path_length(trimmed_x, trimmed_y),
        "path_efficiency": path_efficiency(trimmed_x, trimmed_y),
    }
    if drill == "FORTY_YARD_DASH":
        split, burst = _ten_yard_split(trimmed_time, trimmed_x, trimmed_y)
        values["sensor_ten_yd_time"] = split
        values["sensor_ten_yd_burst_speed"] = burst
    return values


def _player_group(position: object) -> str | None:
    value = str(position)
    for group, positions in GROUP_POSITIONS.items():
        if value in positions:
            return group
    return None


def prepare_official(players: pd.DataFrame, combine_results: pd.DataFrame) -> pd.DataFrame:
    player_columns = ["nfl_id", "nfl_position", "draft_year"]
    player_data = players.loc[:, player_columns].copy()
    player_data["nfl_id"] = player_data["nfl_id"].astype("string")
    player_data["group"] = player_data["nfl_position"].map(_player_group)
    player_data = player_data.loc[player_data["group"].notna()].copy()
    if player_data["nfl_id"].duplicated().any():
        raise ValueError("players must contain one row per nfl_id")
    result_columns = ["nfl_id", *OFFICIAL_MEASURES]
    result_data = combine_results.loc[:, result_columns].copy()
    result_data["nfl_id"] = result_data["nfl_id"].astype("string")
    if result_data["nfl_id"].duplicated().any():
        raise ValueError("combine results must contain one row per nfl_id")
    official = player_data.merge(result_data, on="nfl_id", how="left", validate="one_to_one")
    for measure in OFFICIAL_MEASURES:
        official[measure] = pd.to_numeric(official[measure], errors="coerce")
        official[f"{measure}_missing"] = official[measure].isna().astype("int8")
    return official.sort_values("nfl_id").reset_index(drop=True)


def _allowed_drill(group: str, drill: str, drill_type: str) -> bool:
    if drill in STANDARD_DRILLS:
        return drill_type == drill
    if group == "DL" and drill in DL_DRILLS:
        return drill_type == "SKILL_DRILLS_DL" or (drill == "PASS_RUSH_DRILL" and drill_type == "SKILL_DRILLS_LB")
    return group == "OL" and drill in OL_DRILLS and drill_type == "SKILL_DRILLS_OL"


def _tracking_seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    origin = pd.Timestamp("1970-01-01", tz="UTC")
    return (timestamps - origin).dt.total_seconds()


def extract_attempts(tracking: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    player_groups = official.set_index("nfl_id")["group"]
    selected = tracking.loc[tracking["entity_type"].eq("PLAYER")].copy()
    selected["nfl_id"] = selected["nfl_id"].astype("string")
    selected["group"] = selected["nfl_id"].map(player_groups)
    selected["drill"] = selected["drill_name"].astype("string")
    selected["drill_type"] = selected["drill_type"].astype("string")
    selected = selected.loc[selected["group"].notna()].copy()
    allowed = [
        _allowed_drill(str(group), str(drill), str(drill_type))
        for group, drill, drill_type in zip(selected["group"], selected["drill"], selected["drill_type"], strict=True)
    ]
    selected = selected.loc[allowed].copy()
    selected["seconds"] = _tracking_seconds(selected["time"])
    selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
    selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
    selected = selected.dropna(subset=["seconds", "x", "y", "attempt"])
    rows: list[dict[str, object]] = []
    group_columns = ["nfl_id", "group", "drill", "attempt"]
    for keys, values in selected.groupby(group_columns, sort=True, observed=True):
        nfl_id, group, drill, attempt = keys
        metrics = attempt_kinematics(
            values["seconds"].to_numpy(dtype=float),
            values["x"].to_numpy(dtype=float),
            values["y"].to_numpy(dtype=float),
            str(drill),
        )
        if metrics:
            row: dict[str, object] = {"nfl_id": nfl_id, "group": group, "drill": drill, "attempt": attempt}
            row.update(metrics)
            rows.append(row)
    columns = ["nfl_id", "group", "drill", "attempt", *BASE_METRICS, *FORTY_METRICS]
    return pd.DataFrame(rows, columns=columns)


def mirror_comparison(attempts: pd.DataFrame) -> pd.DataFrame:
    variants = attempts.loc[attempts["drill"].isin(["PASS_PRO_MIRROR_DRILL", "PASS_PRO-MIRROR_DRILL"])].copy()
    variants["displacement"] = variants["path_length"] * variants["path_efficiency"]
    columns = ["time_in_drill", "path_length", "displacement"]
    return variants.groupby("drill", observed=True)[columns].median().reset_index()


def canonicalize_mirror(attempts: pd.DataFrame) -> pd.DataFrame:
    return attempts.assign(drill=attempts["drill"].replace({"PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL"}))


def _metric_columns(attempts: pd.DataFrame) -> tuple[str, ...]:
    return tuple(metric for metric in (*BASE_METRICS, *FORTY_METRICS) if metric in attempts.columns)


def aggregate_sensor_attempts(attempts: pd.DataFrame, player_ids: pd.Series) -> pd.DataFrame:
    wide = pd.DataFrame({"nfl_id": player_ids.astype("string")})
    for drill, drill_attempts in attempts.groupby("drill", sort=True, observed=True):
        prefix = slug(str(drill))
        counts = drill_attempts.groupby("nfl_id", observed=True)["attempt"].nunique().rename(f"{prefix}_n_attempts")
        wide = wide.merge(counts.reset_index(), on="nfl_id", how="left", validate="one_to_one")
        for metric in _metric_columns(drill_attempts):
            values = drill_attempts.loc[drill_attempts[metric].notna(), ["nfl_id", metric]]
            median = values.groupby("nfl_id", observed=True)[metric].median().rename(f"{prefix}_{metric}")
            spread = values.groupby("nfl_id", observed=True)[metric].std(ddof=0).rename(f"{prefix}_{metric}_spread")
            wide = wide.merge(median.reset_index(), on="nfl_id", how="left", validate="one_to_one")
            wide = wide.merge(spread.reset_index(), on="nfl_id", how="left", validate="one_to_one")
    return wide


def repeatability_table(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (group, drill), drill_attempts in attempts.groupby(["group", "drill"], sort=True, observed=True):
        for metric in _metric_columns(drill_attempts):
            values = drill_attempts.loc[drill_attempts[metric].notna(), ["nfl_id", metric]].copy()
            n_players = int((values.groupby("nfl_id", observed=True).size() >= 2).sum())
            row: dict[str, object] = {
                "group": str(group),
                "drill": str(drill),
                "metric": metric,
                "predictor_column": f"{slug(str(drill))}_{metric}",
                "n_players": n_players,
                "pair_correlation": float("nan"),
                "icc": float("nan"),
                "reliability": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "status": "insufficient_attempts",
            }
            if n_players >= 25:
                split = split_half_reliability(values, "nfl_id", metric, min_n=2)
                lower, upper = player_bootstrap_ci(split)
                icc = mixed_model_icc(values, "nfl_id", metric).icc
                reliability = split.median
                row.update(
                    {
                        "pair_correlation": float(split.correlations["spearman"].median()),
                        "icc": icc,
                        "reliability": reliability,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "status": "pass" if reliability >= 0.30 else "unreliable",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["group", "drill", "metric"]).reset_index(drop=True)


def coverage_table(attempts: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        attempts.groupby(["group", "drill"], observed=True)
        .agg(players=("nfl_id", "nunique"), attempts=("attempt", "size"))
        .reset_index()
    )
    return coverage.sort_values(["group", "drill"]).reset_index(drop=True)


def predictor_eligibility(official: pd.DataFrame, repeatability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_official in official.groupby("group", sort=True, observed=True):
        for measure in OFFICIAL_MEASURES:
            rows.append(
                {
                    "predictor_column": measure,
                    "family": "official",
                    "group": str(group),
                    "n_players": int(group_official[measure].notna().sum()),
                    "reliability": float("nan"),
                    "status": "included_m2",
                    "eligible_for_M3": False,
                    "eligible_for_M4": False,
                }
            )
            rows.append(
                {
                    "predictor_column": f"{measure}_missing",
                    "family": "official",
                    "group": str(group),
                    "n_players": int(len(group_official)),
                    "reliability": float("nan"),
                    "status": "included_m2",
                    "eligible_for_M3": False,
                    "eligible_for_M4": False,
                }
            )
    for row in repeatability.itertuples(index=False):
        passing = str(row.status) == "pass"
        metric = str(row.metric)
        rows.append(
            {
                "predictor_column": str(row.predictor_column),
                "family": "sensor",
                "group": str(row.group),
                "n_players": int(str(row.n_players)),
                "reliability": float(str(row.reliability)),
                "status": str(row.status),
                "eligible_for_M3": passing and metric in SPEED_ACCELERATION_BURST_METRICS,
                "eligible_for_M4": passing,
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "family", "predictor_column"]).reset_index(drop=True)


def forty_speed_correlations(features: pd.DataFrame, attempts: pd.DataFrame) -> pd.DataFrame:
    forty = attempts.loc[attempts["drill"].eq("FORTY_YARD_DASH")]
    speed = forty.groupby(["nfl_id", "group"], observed=True)["top_speed"].median().rename("sensor_top_speed")
    joined = features.merge(speed.reset_index(), on=["nfl_id", "group"], how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for group, values in joined.groupby("group", sort=True, observed=True):
        observed = values.loc[:, ["forty", "sensor_top_speed"]].dropna()
        correlation = float(str(observed.corr(method="pearson").iloc[0, 1])) if len(observed) >= 3 else float("nan")
        rows.append({"group": str(group), "correlation": correlation, "n_players": int(len(observed))})
    return pd.DataFrame(rows)


def _repeatability_figure(repeatability: pd.DataFrame, path: Path) -> None:
    plotted = repeatability.loc[repeatability["reliability"].notna()].copy()
    figure, axis = plt.subplots(figsize=(max(12.0, len(plotted) * 0.32), 7.0))
    values = plotted["reliability"].to_numpy(dtype=float)
    lower = plotted["ci_lower"].to_numpy(dtype=float)
    upper = plotted["ci_upper"].to_numpy(dtype=float)
    errors = np.vstack((np.maximum(0.0, values - lower), np.maximum(0.0, upper - values)))
    colors = ["#2a9d8f" if status == "pass" else "#e76f51" for status in plotted["status"]]
    axis.bar(np.arange(len(plotted)), values, yerr=errors, capsize=2, color=colors)
    axis.axhline(0.30, color="#264653", linewidth=1)
    axis.set_xticks(
        np.arange(len(plotted)),
        [f"{row.group}\n{slug(str(row.drill))}\n{row.metric}" for row in plotted.itertuples(index=False)],
        rotation=90,
        fontsize=6,
    )
    axis.set_ylabel("Spearman-Brown repeatability")
    axis.set_title("DL and OL Combine sensor repeatability")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build(root: Path, merge_mirror: bool) -> BuildResult:
    official = prepare_official(load_players(root), load_combine_results(root))
    attempts = extract_attempts(load_combine_tracking(root), official)
    comparison = mirror_comparison(attempts)
    canonical_attempts = canonicalize_mirror(attempts) if merge_mirror else attempts
    sensor = aggregate_sensor_attempts(canonical_attempts, official["nfl_id"])
    features = official.merge(sensor, on="nfl_id", how="left", validate="one_to_one")
    if features["nfl_id"].duplicated().any():
        raise ValueError("combine predictors must contain one row per nfl_id")
    if (canonical_attempts["top_speed"].dropna() < 0.0).any():
        raise ValueError("sensor speeds must be non-negative")
    repeatability = repeatability_table(canonical_attempts)
    coverage = coverage_table(canonical_attempts)
    predictors = predictor_eligibility(official, repeatability)
    correlations = forty_speed_correlations(features, canonical_attempts)
    processed = root / "data" / "processed"
    reports = root / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    features.to_parquet(processed / "combine_predictors.parquet", index=False)
    repeatability.to_csv(reports / "phase2_combine_repeatability.csv", index=False)
    coverage.to_csv(reports / "phase2_combine_coverage.csv", index=False)
    predictors.to_csv(reports / "phase2_combine_predictors.csv", index=False)
    _repeatability_figure(repeatability, reports / "phase2_combine_repeatability.png")
    return BuildResult(features, canonical_attempts, repeatability, coverage, predictors, correlations, comparison)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--merge-mirror", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.merge_mirror:
        raise ValueError("PASS_PRO mirror variants require a logged comparison before merging")
    result = build(arguments.root, merge_mirror=True)
    print(result.mirror_comparison.to_csv(index=False).strip())
    print(result.forty_correlations.to_csv(index=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
