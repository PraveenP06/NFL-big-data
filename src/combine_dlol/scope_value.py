from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as student_t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from combine_dlol.reliability import player_bootstrap_ci, split_half_reliability

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
GROUPS = ("DL", "OL", "WR", "TE", "DB")


def oriented_sign(name: str) -> int:
    lowered = name.lower()
    if any(
        token in lowered for token in ("forty", "split", "cone", "shuttle", "time", "loss_fraction", "deceleration")
    ):
        return -1
    if any(
        token in lowered
        for token in (
            "speed",
            "acceleration",
            "vertical",
            "broad",
            "bench",
            "distance",
            "height",
            "wing",
            "arm",
            "hand",
        )
    ):
        return 1
    return 0


def join_value_sources(combine: pd.DataFrame, players: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    left = combine.copy().assign(nfl_id=combine["nfl_id"].astype("string"))
    player_columns = [
        column
        for column in ("nfl_id", "display_name", "nfl_position", "draft_year", "draft_overall_pick", "draft_round")
        if column in players
    ]
    player_frame = players.loc[:, player_columns].copy().assign(nfl_id=players["nfl_id"].astype("string"))
    career_frame = career.copy().assign(nfl_id=career["nfl_id"].astype("string"))
    return left.merge(player_frame, on="nfl_id", how="left", suffixes=("", "_player")).merge(
        career_frame, on="nfl_id", how="left"
    )


def sensor_composite(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame.loc[:, list(columns)].astype(float)
    oriented = pd.DataFrame({column: values[column] * oriented_sign(column) for column in columns})
    standardized = (oriented - oriented.mean()) / oriented.std(ddof=0).replace(0.0, np.nan)
    return standardized.mean(axis=1, skipna=True)


def partial_spearman(predictor: np.ndarray, outcome: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
    predictor_rank = rankdata(predictor)
    outcome_rank = rankdata(outcome)
    design = np.column_stack((np.ones(len(predictor_rank)), np.asarray(controls, dtype=float)))
    predictor_residual = predictor_rank - design @ np.linalg.lstsq(design, predictor_rank, rcond=None)[0]
    outcome_residual = outcome_rank - design @ np.linalg.lstsq(design, outcome_rank, rcond=None)[0]
    raw = float(np.corrcoef(predictor_rank, outcome_rank)[0, 1])
    adjusted = float(np.corrcoef(predictor_residual, outcome_residual)[0, 1])
    return raw, adjusted


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    if len(p_values) == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    sorted_values = p_values[order]
    adjusted = np.minimum.accumulate((sorted_values * len(p_values) / np.arange(1, len(p_values) + 1))[::-1])[::-1]
    result = np.empty(len(p_values), dtype=float)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _role(frame: pd.DataFrame) -> pd.Series:
    position = frame["nfl_position"].fillna("unknown").astype(str)
    return pd.Series(
        np.where(
            frame["combine_position"].eq("DL"),
            np.where(position.isin(["DE", "OLB"]), "EDGE", "IDL"),
            np.where(frame["combine_position"].eq("OL"), np.where(position.eq("T"), "T", "IOL"), position),
        ),
        index=frame.index,
    )


def _controls(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    numeric = frame.loc[:, [name for name in names if name in frame]].astype(float)
    filled = numeric.fillna(numeric.median()).fillna(0.0)
    role = pd.get_dummies(frame["role"], dtype=float)
    return np.column_stack((filled.to_numpy(dtype=float), role.to_numpy(dtype=float)))


def _ci_p_value(r_value: float, n: int, control_count: int) -> tuple[float, float, float]:
    degrees = n - control_count - 2
    if degrees <= 0 or abs(r_value) >= 1.0:
        return float("nan"), float("nan"), float("nan")
    z_value = np.arctanh(r_value)
    standard_error = 1.0 / np.sqrt(n - control_count - 3)
    statistic = r_value * np.sqrt(degrees / (1.0 - r_value**2))
    return (
        float(np.tanh(z_value - 1.96 * standard_error)),
        float(np.tanh(z_value + 1.96 * standard_error)),
        float(2.0 * student_t.sf(abs(statistic), degrees)),
    )


def _replication(frame: pd.DataFrame, predictor: str, outcome: str) -> tuple[float, float, float, float, float]:
    values: list[float] = []
    for years in ((2023, 2024), (2025,), (2023,), (2024,), (2025,)):
        subset = frame.loc[frame["draft_year"].isin(years), [predictor, outcome]].dropna()
        value = (
            float(np.corrcoef(rankdata(subset[predictor]), rankdata(subset[outcome]))[0, 1])
            if len(subset) >= 15
            else float("nan")
        )
        values.append(value)
    return values[0], values[1], values[2], values[3], values[4]


def _association_rows(
    frame: pd.DataFrame,
    group: str,
    outcome: str,
    predictors: Sequence[str],
    family: str,
    controls: Sequence[str],
    outcome_sign: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for predictor in predictors:
        required = [predictor, outcome, "draft_year", "role"]
        subset = (
            frame.loc[:, list(dict.fromkeys(required + list(controls)))]
            .dropna(subset=[predictor, outcome, "draft_year", "role"])
            .copy()
        )
        if len(subset) < 40 or subset[predictor].nunique() < 5 or subset[outcome].nunique() < 2:
            continue
        control_values = _controls(subset, controls)
        raw, adjusted = partial_spearman(
            subset[predictor].to_numpy(float), subset[outcome].to_numpy(float), control_values
        )
        low, high, p_value = _ci_p_value(adjusted, len(subset), control_values.shape[1])
        sign = oriented_sign(predictor) * outcome_sign
        oriented = adjusted * sign if sign else float("nan")
        direction = "expected" if oriented > 0 else "opposite" if oriented < 0 else "unclassified"
        discovery, holdout, first, second, third = _replication(subset, predictor, outcome)
        rows.append(
            {
                "scope": "value",
                "group": group,
                "outcome": outcome,
                "predictor": predictor,
                "family": family,
                "n": len(subset),
                "r_raw": raw,
                "r_adj": adjusted,
                "ci_lo": low,
                "ci_hi": high,
                "p": p_value,
                "q_scope": float("nan"),
                "oriented_r": oriented,
                "direction": direction,
                "r_disc_2023_24": discovery,
                "r_hold_2025": holdout,
                "r_2023": first,
                "r_2024": second,
                "r_2025": third,
            }
        )
    return rows


def _official_for_group(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in OFFICIAL if column in frame and int(frame[column].notna().sum()) >= 40)


def _raw_sensor_attempts(root: Path, ids: set[str]) -> pd.DataFrame:
    cache = root / "data/processed/scope_value_sensor_attempts.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts: list[pd.DataFrame] = []
    columns = ["nfl_id", "drill_name", "attempt", "s", "a"]
    for chunk in pd.read_csv(root / "data/raw/combine_tracking.csv", usecols=columns, chunksize=200_000):
        kept = chunk.loc[chunk["nfl_id"].astype("string").isin(ids)].copy()
        if not kept.empty:
            parts.append(kept)
    tracking = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    tracking["nfl_id"] = tracking["nfl_id"].astype("string")
    summary = tracking.groupby(["nfl_id", "drill_name", "attempt"], as_index=False).agg(
        top_speed=("s", "max"), peak_acceleration=("a", "max")
    )
    summary.to_parquet(cache, index=False)
    return summary


def _archive_repeatable(root: Path, archive: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    selected = archive.loc[archive["combine_position"].isin(["WR", "TE", "DB"])].copy()
    attempts = _raw_sensor_attempts(root, set(selected["nfl_id"].astype("string")))
    selected_ids = selected.set_index("nfl_id")["combine_position"].astype(str).to_dict()
    attempts["group"] = attempts["nfl_id"].map(selected_ids)
    metric_rows: list[dict[str, object]] = []
    keep: dict[str, list[str]] = {group: [] for group in ("WR", "TE", "DB")}
    for (group, drill), data in attempts.groupby(["group", "drill_name"], observed=True):
        for metric in ("top_speed", "peak_acceleration"):
            wide = data.pivot_table(index="nfl_id", columns="attempt", values=metric, aggfunc="mean")
            eligible = wide.dropna(thresh=2)
            reliability = float("nan")
            lower = float("nan")
            upper = float("nan")
            n_players = len(eligible)
            if n_players >= 25:
                split = split_half_reliability(data.loc[:, ["nfl_id", metric]], "nfl_id", metric, min_n=2)
                reliability = split.median
                lower, upper = player_bootstrap_ci(split)
            column = f"{str(drill).lower()}_{metric}"
            passed = n_players >= 25 and np.isfinite(reliability) and reliability >= 0.30 and column in selected
            metric_rows.append(
                {
                    "group": group,
                    "drill": drill,
                    "metric": metric,
                    "predictor_column": column,
                    "n_players": n_players,
                    "reliability": reliability,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "status": "pass" if passed else "unreliable",
                }
            )
            if passed:
                keep[str(group)].append(column)
    pd.DataFrame(metric_rows).to_csv(root / "reports/scope_value_repeatability.csv", index=False)
    return selected, {group: tuple(columns) for group, columns in keep.items()}


def _base_frame(root: Path) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    results = pd.read_csv(root / "data/raw/combine_results.csv")
    players = pd.read_csv(root / "data/raw/players.csv")
    career = pd.read_csv(root / "data/raw/player_career_successes.csv")
    dlol = pd.read_parquet(root / "data/processed/combine_predictors.parquet")
    archive = pd.read_parquet(Path.home() / "dev/NFL-big-data-archive/processed/combine_features.parquet")
    archive, archive_sensors = _archive_repeatable(root, archive)
    official = join_value_sources(results.loc[results["combine_position"].isin(GROUPS)], players, career)
    sensor = pd.concat((dlol, archive), ignore_index=True, sort=False)
    sensor["nfl_id"] = sensor["nfl_id"].astype("string")
    merged = official.merge(
        sensor.drop(columns=[column for column in OFFICIAL if column in sensor], errors="ignore"),
        on="nfl_id",
        how="left",
        suffixes=("", "_sensor"),
    )
    merged["combine_position"] = merged["combine_position"].fillna(merged.get("group"))
    merged["draft_year"] = pd.to_numeric(merged["draft_year"], errors="coerce")
    merged["role"] = _role(merged)
    career_columns = [
        "career_offensive_snaps",
        "career_defensive_snaps",
        "career_games_active",
        "career_games_started",
        "ap_all_pro_1st_team",
        "ap_all_pro_2nd_team",
        "pro_bowl_original_ballot",
    ]
    for column in career_columns:
        source = merged[column] if column in merged else pd.Series(0.0, index=merged.index)
        merged[column] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    merged["games_started_share"] = np.where(
        merged["career_games_active"] > 0, merged["career_games_started"] / merged["career_games_active"], np.nan
    )
    merged["snaps_per_active_game"] = np.where(
        merged["career_games_active"] > 0,
        (merged["career_offensive_snaps"] + merged["career_defensive_snaps"]) / merged["career_games_active"],
        np.nan,
    )
    merged["ever_all_pro_or_pro_bowl"] = (
        merged[["ap_all_pro_1st_team", "ap_all_pro_2nd_team", "pro_bowl_original_ballot"]].max(axis=1) > 0
    ).astype(float)
    metadata = pd.read_csv(root / "reports/phase2_combine_predictors.csv")
    sensors: dict[str, tuple[str, ...]] = {}
    for group in ("DL", "OL"):
        rows = metadata.loc[
            (metadata["group"] == group) & (metadata["family"] == "sensor") & (metadata["status"] == "pass"),
            "predictor_column",
        ]
        sensors[group] = tuple(column for column in rows if column in merged and merged[column].notna().sum() >= 40)
    sensors.update(archive_sensors)
    for group, columns in sensors.items():
        if columns:
            selector = merged["combine_position"].eq(group)
            merged.loc[selector, "sensor_composite"] = sensor_composite(merged.loc[selector], columns)
    return merged, sensors


def _career_predictors(frame: pd.DataFrame, sensors: Sequence[str]) -> tuple[str, ...]:
    official = _official_for_group(frame)
    composite = ("sensor_composite",) if sensors else ()
    return tuple(dict.fromkeys((*official, *sensors, *composite)))


def _game_outcomes(root: Path, base: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    values: dict[str, pd.DataFrame] = {}
    gate_rows: list[dict[str, object]] = []
    dl = pd.read_parquet(root / "data/processed/dl_outcomes_scores.parquet")
    dl["nfl_id"] = dl["nfl_id"].astype("string")
    dl_outcomes = {
        "shrunk_d_a": -1,
        "shrunk_d_b_displacement_0_5": -1,
        "shrunk_d_b_displacement_1_0": -1,
        "shrunk_d_b_speed_0_5": -1,
    }
    values["DL"] = dl.loc[:, ["nfl_id", *dl_outcomes]].copy()
    for outcome in dl_outcomes:
        gate_rows.append(
            {
                "outcome": outcome,
                "group": "DL",
                "n_players": int(dl[outcome].notna().sum()),
                "SB_median": np.nan,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "icc": np.nan,
                "verdict": "adequate_existing_gate",
                "formula": "existing EB-shrunk context-adjusted score",
            }
        )
    ol = pd.read_parquet(root / "data/processed/ol_outcomes_scores.parquet")
    pivot = ol.pivot(index="nfl_id", columns="outcome", values="posterior_mean").reset_index()
    pivot["nfl_id"] = pivot["nfl_id"].astype("string")
    values["OL"] = pivot
    for outcome in pivot.columns.drop("nfl_id"):
        gate_rows.append(
            {
                "outcome": str(outcome),
                "group": "OL",
                "n_players": int(pivot[outcome].notna().sum()),
                "SB_median": np.nan,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "icc": np.nan,
                "verdict": "adequate_existing_gate",
                "formula": "existing EB-shrunk context-adjusted score",
            }
        )
    return values, gate_rows


def _cv_rows(
    frame: pd.DataFrame, group: str, official: Sequence[str], sensors: Sequence[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    predictor_sets = {
        "draft_pick": ("log_draft_pick",),
        "official": ("log_draft_pick", *official),
        "sensor": ("log_draft_pick", *official, *sensors),
    }
    outcomes = ("games_started_share", "snaps_per_active_game", "ever_all_pro_or_pro_bowl")
    for outcome in outcomes:
        data = frame.dropna(subset=[outcome, "log_draft_pick"])
        if len(data) < 20:
            continue
        scores: dict[str, np.ndarray] = {}
        for name, columns in predictor_sets.items():
            matrix = data.loc[:, list(dict.fromkeys(columns))].to_numpy(dtype=float)
            predictions = np.zeros(len(data), dtype=float)
            splitter = RepeatedKFold(n_splits=5, n_repeats=10, random_state=20261103)
            counts = np.zeros(len(data), dtype=float)
            for train, test in splitter.split(matrix):
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
                model.fit(matrix[train], data[outcome].to_numpy(float)[train])
                predictions[test] += model.predict(matrix[test])
                counts[test] += 1.0
            scores[name] = predictions / counts
        target = data[outcome].to_numpy(float)
        baseline_error = (target - scores["draft_pick"]) ** 2
        for name in ("official", "sensor"):
            model_error = (target - scores[name]) ** 2
            delta = 1.0 - model_error.mean() / baseline_error.mean()
            rng = np.random.default_rng(20261103)
            boot = np.array(
                [
                    1.0 - model_error[index].mean() / baseline_error[index].mean()
                    for index in (rng.integers(0, len(data), len(data)) for _ in range(1000))
                ]
            )
            low, high = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "group": group,
                    "outcome": outcome,
                    "comparison": f"{name}_minus_draft_pick",
                    "n": len(data),
                    "delta_r2": delta,
                    "ci_lo": low,
                    "ci_hi": high,
                }
            )
    return rows


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base, sensors = _base_frame(root)
    base["draft_overall_pick"] = pd.to_numeric(base["draft_overall_pick"], errors="coerce")
    base["log_draft_pick"] = np.log(base["draft_overall_pick"])
    rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for group in GROUPS:
        frame = base.loc[base["combine_position"].eq(group)].copy()
        sensor_predictors = sensors.get(group, ())
        predictors = _career_predictors(frame, sensor_predictors)
        composite = ("sensor_composite",) if sensor_predictors else ()
        sensor_with_composite = (*sensor_predictors, *composite)
        official = _official_for_group(frame)
        rows.extend(
            _association_rows(
                frame,
                group,
                "log_draft_pick",
                sensor_with_composite,
                "draft_pricing",
                (*official, "combine_weight"),
                -1,
            )
        )
        for outcome in ("games_started_share", "snaps_per_active_game", "ever_all_pro_or_pro_bowl"):
            rows.extend(_association_rows(frame, group, outcome, predictors, "career", ("combine_weight",), 1))
            rows.extend(
                _association_rows(
                    frame, group, outcome, predictors, "market_gap", ("combine_weight", "log_draft_pick"), 1
                )
            )
            valid = frame[outcome].notna()
            zeros = int((frame.loc[valid, outcome] == 0).sum())
            gate_rows.append(
                {
                    "outcome": outcome,
                    "group": group,
                    "n_players": int(valid.sum()),
                    "SB_median": np.nan,
                    "lower_ci": np.nan,
                    "upper_ci": np.nan,
                    "icc": np.nan,
                    "verdict": "career",
                    "formula": f"career outcome; zeros={zeros}; active-game denominator required for rates",
                }
            )
        model_rows.extend(_cv_rows(frame, group, official, sensor_with_composite))
    game, game_gate = _game_outcomes(root, base)
    gate_rows.extend(game_gate)
    for group, outcomes in game.items():
        frame = base.loc[base["combine_position"].eq(group)].merge(outcomes, on="nfl_id", how="inner")
        for outcome in outcomes.columns.drop("nfl_id"):
            rows.extend(
                _association_rows(
                    frame,
                    group,
                    str(outcome),
                    _career_predictors(frame, sensors.get(group, ())),
                    "perf_beyond_draft",
                    ("combine_weight", "log_draft_pick"),
                    -1 if "pressure" in str(outcome) or "reaction" in str(outcome) else 1,
                )
            )
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    for family in results["family"].unique():
        indices = results.index[results["family"].eq(family)]
        results.loc[indices, "q_scope"] = bh_fdr(results.loc[indices, "p"].to_numpy(dtype=float))
    gate = pd.DataFrame(
        gate_rows,
        columns=("outcome", "group", "n_players", "SB_median", "lower_ci", "upper_ci", "icc", "verdict", "formula"),
    )
    models = pd.DataFrame(model_rows)
    lists = _player_lists(base, game)
    results.to_csv(root / "reports/scope_value_results.csv", index=False)
    gate.to_csv(root / "reports/scope_value_gate.csv", index=False)
    models.to_csv(root / "reports/scope_value_models.csv", index=False)
    lists.to_csv(root / "reports/scope_value_player_list.csv", index=False)
    return results, gate, models, lists


def _player_lists(base: pd.DataFrame, game: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for group in ("DL", "OL"):
        outcomes = game[group]
        frame = (
            base.loc[base["combine_position"].eq(group)]
            .merge(outcomes, on="nfl_id", how="inner")
            .dropna(subset=["log_draft_pick"])
        )
        names = [column for column in outcomes.columns if column != "nfl_id"]
        scored = frame.loc[:, names].copy()
        for name in names:
            scored[name] = scored[name] * (-1 if "pressure" in name or "reaction" in name else 1)
        frame["game_composite"] = ((scored - scored.mean()) / scored.std(ddof=0)).mean(axis=1)
        design = np.column_stack((np.ones(len(frame)), frame["log_draft_pick"].to_numpy(float)))
        frame["beyond_draft_residual"] = (
            frame["game_composite"] - design @ np.linalg.lstsq(design, frame["game_composite"], rcond=None)[0]
        )
        sensor_columns = [
            str(column)
            for column in frame
            if str(column).endswith(("top_speed", "peak_acceleration", "burst_speed_1_0s"))
        ][:5]
        selected = pd.concat(
            (frame.nsmallest(10, "beyond_draft_residual"), frame.nlargest(10, "beyond_draft_residual"))
        )
        rows.append(
            selected.loc[
                :,
                [
                    "nfl_id",
                    "display_name",
                    "combine_position",
                    "draft_overall_pick",
                    "beyond_draft_residual",
                    *sensor_columns,
                ],
            ].assign(
                descriptive_rank=np.r_[np.arange(1, 11), np.arange(1, 11)],
                descriptive_group=["falls_short"] * 10 + ["exceeds"] * 10,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(arguments)
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
