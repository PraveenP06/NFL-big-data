from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import rankdata

from combine_dlol.combine_predictors import (
    _clean_trajectory,
    attempt_kinematics,
    centered_smoothed_speed,
    first_sustained_movement,
)
from combine_dlol.db_play_tracking import build_tracking_features
from combine_dlol.db_play_tracking import seconds as _seconds
from combine_dlol.db_play_tracking import wrap_degrees as _wrap_degrees
from combine_dlol.reliability import (
    adjusted_residuals,
    empirical_bayes_shrink,
    mixed_model_icc,
    player_bootstrap_ci,
    split_half_reliability,
    verdict,
)
from combine_dlol.scope_db import (
    RESULT_COLUMNS,
    _outcome_frame,
    benjamini_hochberg,
    partial_spearman,
    role_from_position,
    sensor_repeatability,
)
from combine_dlol.scope_wrte import _context_formula, _outcome_definitions

MIN_PREDICTOR_N = 40
MIN_TE_N = 30
MIN_REPEAT_PLAYERS = 25

DB_DRILLS = (
    "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL",
    "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION",
    "BOX_DRILL",
    "W_DRILL",
    "TERYL_AUSTIN_DRILL_1",
    "TERYL_AUSTIN_DRILL_2",
    "LINE_DRILL",
    "LINE",
    "GAUNTLET_DRILL",
)
STANDARD_DRILLS = ("FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL")
DB_DRILL_TYPES = frozenset({"SKILL_DRILLS_DB", *STANDARD_DRILLS})
TE_DRILL_TYPES = frozenset({"SKILL_DRILLS_TE"})
OFFICIAL_REFERENCE = ("forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump")

BACKPEDAL_DRILLS = ("BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION")
COD_DRILLS = ("W_DRILL", "BOX_DRILL", "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2")
PAIR_OUTCOMES = {
    "pair_backpedal": (
        "snap_reaction_time",
        "speed_1s",
        "backpedal_share",
        "speed_at_pass_forward",
        "speed_gain_after_throw",
    ),
    "pair_cod": (
        "cod_speed_retention",
        "peak_decel_in_play",
        "closing_speed_nearest",
        "closing_speed_target",
        "nearest_receiver_dist_arrival",
        "dist_to_target_arrival",
    ),
    "pair_speed_burst": (
        "closing_speed_nearest",
        "closing_speed_target",
        "run_support_rate",
        "run_pursuit_peak_speed",
        "peak_speed_in_play",
    ),
}

G1_OUTCOMES = (
    "coverage_assignment_rate",
    "man_coverage_assignment_rate",
    "assist_rate",
    "tackle_rate",
    "tackle_for_loss_rate",
    "forced_fumble_rate",
    "run_support_rate",
)
G2_OUTCOMES = (
    "snap_reaction_time",
    "speed_1s",
    "backpedal_share",
    "speed_at_pass_forward",
    "speed_gain_after_throw",
    "nearest_receiver_dist_pf",
    "nearest_receiver_dist_arrival",
    "closing_speed_nearest",
    "dist_to_target_arrival",
    "closing_speed_target",
    "cod_speed_retention",
    "peak_decel_in_play",
    "peak_speed_in_play",
    "run_pursuit_peak_speed",
    "tackle_depth",
)
G2_MIN_PLAYS = {"tackle_depth": 15}
G2_DEFAULT_MIN_PLAYS = 30
G1_MIN_PLAYS = 50
G1_NUMERIC = ("pass_length", "down", "yards_to_go")
G1_CATEGORICAL = ("team_coverage_man_zone", "team_coverage_type", "offense_formation")
G2_CATEGORICAL = (*G1_CATEGORICAL, "lined_up_position", "is_pass")
TE_OUTCOMES = ("separation_at_pass_forward", "target_rate", "rec_yards_per_route")
TE_MIN_ROUTES = 50
OUTCOME_ORIENTATION = {
    "snap_reaction_time": -1.0,
    "nearest_receiver_dist_pf": -1.0,
    "nearest_receiver_dist_arrival": -1.0,
    "dist_to_target_arrival": -1.0,
    "tackle_depth": -1.0,
    "speed_1s": 1.0,
    "speed_at_pass_forward": 1.0,
    "speed_gain_after_throw": 1.0,
    "closing_speed_nearest": 1.0,
    "closing_speed_target": 1.0,
    "cod_speed_retention": 1.0,
    "peak_decel_in_play": 1.0,
    "peak_speed_in_play": 1.0,
    "run_pursuit_peak_speed": 1.0,
    "run_support_rate": 1.0,
    "assist_rate": 1.0,
    "tackle_rate": 1.0,
    "tackle_for_loss_rate": 1.0,
    "forced_fumble_rate": 1.0,
    "separation_at_pass_forward": 1.0,
    "target_rate": 1.0,
    "rec_yards_per_route": 1.0,
}
LOWER_OFFICIAL = frozenset({"forty", "ten_yd_split", "three_cone", "short_shuttle"})
HIGHER_OFFICIAL = frozenset({"vertical", "broad_jump"})
LOWER_SUFFIXES = ("_time_in_drill", "_sensor_ten_yd_time", "_cod_speed_loss_fraction")
HIGHER_SUFFIXES = (
    "_top_speed",
    "_burst_distance_0_1s",
    "_burst_speed_1_0s",
    "_peak_acceleration",
    "_peak_deceleration",
    "_sensor_ten_yd_burst_speed",
    "_cod_speed_recovery_0_5s",
)
FAMILY_COLUMNS = (*RESULT_COLUMNS, "family_label", "low_power")
INFEASIBLE = (
    ("ball_skill_pass_breakup_interception", "player_play has no PBU, interception or ball-skill field"),
    ("pass_coverage_yards_allowed", "no defender-to-receiver matchup field; MAN assignment carries no receiver id"),
    ("te_blocking", "player_play has no blocking credit, pancake or run-block field"),
)


def predictor_orientation(name: str) -> float:
    if name in LOWER_OFFICIAL or name.endswith(LOWER_SUFFIXES):
        return -1.0
    if name in HIGHER_OFFICIAL or name.endswith(HIGHER_SUFFIXES):
        return 1.0
    return 0.0


def outcome_orientation(name: str) -> float:
    return OUTCOME_ORIENTATION.get(name, 0.0)


def orientation_label(oriented_r: float) -> str:
    if not np.isfinite(oriented_r) or oriented_r == 0.0:
        return "unclassified"
    return "expected" if oriented_r > 0 else "opposite"


def slug(drill: str) -> str:
    return drill.lower()


def predictor_column(drill: str, metric: str) -> str:
    return f"{slug(drill)}_{metric}"


def cod_metrics(
    time: Sequence[float] | np.ndarray, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> dict[str, float]:
    empty = {"cod_speed_loss_fraction": float("nan"), "cod_speed_recovery_0_5s": float("nan")}
    clean_t, clean_x, clean_y = _clean_trajectory(time, x, y)
    speed = centered_smoothed_speed(clean_t, clean_x, clean_y)
    if len(speed) < 6:
        return empty
    start = first_sustained_movement(speed)
    t, px, py, v = clean_t[start:], clean_x[start:], clean_y[start:], speed[start:]
    if len(t) < 6:
        return empty
    heading = np.degrees(np.arctan2(np.gradient(py, t), np.gradient(px, t)))
    end = np.searchsorted(t, t + 0.5)
    candidates = np.flatnonzero((end < len(t)) & (v >= 2.0))
    if len(candidates) == 0:
        return empty
    change = _wrap_degrees(heading[end[candidates]] - heading[candidates])
    best = int(np.argmax(change))
    if change[best] < 60.0:
        return empty
    first, last = int(candidates[best]), int(end[candidates[best]])
    trough = float(v[first : last + 1].min())
    loss = (float(v[first]) - trough) / float(v[first])
    recover_at = t[last] + 0.5
    recovery = float(np.interp(recover_at, t, v)) - trough if recover_at <= t[-1] else float("nan")
    return {"cod_speed_loss_fraction": loss, "cod_speed_recovery_0_5s": recovery}


def build_attempts(root: Path, identifiers: set[str], drill_types: frozenset[str]) -> pd.DataFrame:
    columns = ["nfl_id", "entity_type", "time", "drill_name", "drill_type", "attempt", "x", "y"]
    chunks: list[pd.DataFrame] = []
    source = root / "data" / "raw" / "combine_tracking.csv"
    for chunk in pd.read_csv(source, usecols=columns, chunksize=100_000, low_memory=False):
        keep = chunk["entity_type"].eq("PLAYER") & chunk["nfl_id"].astype(str).isin(identifiers)
        selected = chunk.loc[keep & chunk["drill_type"].isin(drill_types)].copy()
        if selected.empty:
            continue
        selected["time"] = _seconds(selected["time"])
        selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
        selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
        chunks.append(selected.dropna(subset=["time", "x", "y", "attempt", "drill_name"]))
    if not chunks:
        return pd.DataFrame(columns=["nfl_id", "drill", "attempt"])
    tracking = pd.concat(chunks, ignore_index=True)
    rows: list[dict[str, object]] = []
    for (nfl_id, drill, attempt), values in tracking.groupby(["nfl_id", "drill_name", "attempt"], observed=True):
        t, px, py = (values[c].to_numpy(dtype=float) for c in ("time", "x", "y"))
        metrics = attempt_kinematics(t, px, py, str(drill))
        if not metrics:
            continue
        rows.append(
            {
                "nfl_id": int(str(nfl_id)),
                "drill": str(drill),
                "attempt": str(attempt),
                **metrics,
                **cod_metrics(t, px, py),
            }
        )
    return pd.DataFrame(rows)


def classify_predictors(attempts: pd.DataFrame, drills: Sequence[str]) -> pd.DataFrame:
    scoped = attempts.loc[attempts["drill"].isin(drills)]
    table = sensor_repeatability(scoped).rename(columns={"status": "repeat_status"})
    table["status"] = table["repeat_status"].map(
        {"pass": "repeatable", "unreliable": "unreliable", "insufficient_attempts": "unverifiable"}
    )
    return table.drop(columns=["repeat_status"])


def sensor_wide(attempts: pd.DataFrame, table: pd.DataFrame, status: str) -> pd.DataFrame:
    wide = pd.DataFrame({"nfl_id": attempts["nfl_id"].drop_duplicates().astype(int)})
    columns: list[pd.Series] = []
    for row in table.loc[table["status"].eq(status)].itertuples(index=False):
        metric = str(row.metric)
        subset = attempts.loc[attempts["drill"].eq(str(row.drill)), ["nfl_id", metric]].dropna()
        columns.append(subset.groupby("nfl_id")[metric].median().rename(str(row.predictor)))
    if not columns:
        return wide
    return wide.merge(pd.concat(columns, axis=1).reset_index(), on="nfl_id", how="left", validate="one_to_one")


# ---------------------------------------------------------------- stability gate


def _gate_row(outcome: str, group: str, formula: str, players: int) -> dict[str, object]:
    return {
        "outcome": outcome,
        "group": group,
        "n_players": players,
        "SB_median": np.nan,
        "lower_ci": np.nan,
        "upper_ci": np.nan,
        "icc": np.nan,
        "verdict": "poor",
        "formula": formula,
    }


def _varying(frame: pd.DataFrame, names: Sequence[str]) -> list[str]:
    return [name for name in names if name in frame and frame[name].nunique(dropna=True) > 1]


def gate_outcome(
    frame: pd.DataFrame,
    outcome: str,
    *,
    min_plays: int,
    group: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
    basis: pd.Series | None = None,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    valid = frame.loc[frame[outcome].notna()]
    counted = valid if basis is None else valid.loc[basis.reindex(valid.index).fillna(False)]
    counts = counted.groupby("nfl_id", observed=True).size()
    players = counts.index[counts.ge(min_plays)]
    rows = valid.loc[valid["nfl_id"].isin(players)]
    numeric_terms = _varying(rows, numeric)
    categorical_terms = _varying(rows, categorical)
    formula = f"{outcome} ~ " + " + ".join([*numeric_terms, *[f"C({name})" for name in categorical_terms]] or ["1"])
    if len(players) < 3:
        return _gate_row(outcome, group, formula, len(players)), None
    split = split_half_reliability(rows, "nfl_id", outcome, min_n=2, n_splits=2000, seed=20261103)
    lower, upper = player_bootstrap_ci(split, n_bootstrap=1000, seed=20261104)
    try:
        icc = mixed_model_icc(rows, "nfl_id", outcome).icc
    except RuntimeError:
        icc = float("nan")
    status = verdict(split.median, lower)
    gate = {
        **_gate_row(outcome, group, formula, len(players)),
        "SB_median": split.median,
        "lower_ci": lower,
        "upper_ci": upper,
        "icc": icc,
        "verdict": status,
    }
    if status != "adequate":
        return gate, None
    adjusted = adjusted_residuals(rows, outcome, numeric_terms, categorical_terms)
    name = f"{outcome}_residual"
    try:
        model = mixed_model_icc(adjusted, "nfl_id", name)
    except RuntimeError:
        return gate, None
    shrunk = empirical_bayes_shrink(adjusted, "nfl_id", name, model, min_n=2)
    return gate, shrunk.loc[:, ["nfl_id", "posterior_mean"]].rename(columns={"posterior_mean": outcome})


def outcome_frames(player_play: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g1 = _outcome_frame(player_play)
    run = g1["pass_result"].isna()
    credited = g1["tackle"].fillna(0.0).gt(0.0) | g1["assist"].fillna(0.0).gt(0.0)
    g1["run_support_rate"] = credited.astype(float).where(run)
    keys = ["game_id", "play_id", "nfl_id"]
    g2 = g1.merge(features, on=keys, how="inner", validate="1:1")
    g2["is_pass"] = g2["pass_result"].notna().astype(str)
    g2["lined_up_position"] = g2["lined_up_position"].astype("object").fillna("NONE")
    return g1, g2


def run_gates(g1: pd.DataFrame, g2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    scores = pd.DataFrame({"nfl_id": g1["nfl_id"].drop_duplicates()})
    jobs: list[tuple[pd.DataFrame, str, int, Sequence[str], pd.Series | None]] = []
    for outcome in G1_OUTCOMES:
        basis = g1["assigned_coverage"] if outcome.endswith("coverage_assignment_rate") else None
        jobs.append((g1, outcome, G1_MIN_PLAYS, G1_CATEGORICAL, basis))
    jobs.extend(
        (g2, outcome, G2_MIN_PLAYS.get(outcome, G2_DEFAULT_MIN_PLAYS), G2_CATEGORICAL, None) for outcome in G2_OUTCOMES
    )
    for frame, outcome, minimum, categorical, basis in jobs:
        gate, score = gate_outcome(
            frame,
            outcome,
            min_plays=minimum,
            group="DB",
            numeric=G1_NUMERIC,
            categorical=categorical,
            basis=basis,
        )
        rows.append(gate)
        if score is not None:
            scores = scores.merge(score, on="nfl_id", how="left", validate="one_to_one")
    return pd.DataFrame(rows), scores


# ---------------------------------------------------------------- associations


def _raw_spearman(frame: pd.DataFrame, predictor: str, outcome: str) -> float:
    if len(frame) < 15:
        return float("nan")
    return float(np.corrcoef(rankdata(frame[predictor]), rankdata(frame[outcome]))[0, 1])


def association_table(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str, str]],
    family_label: str,
    *,
    min_n: int = MIN_PREDICTOR_N,
    use_role: bool = True,
    group: str = "DB",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for predictor, predictor_family, outcome in pairs:
        columns = list(dict.fromkeys([predictor, outcome, "combine_weight", "role", "draft_year"]))
        observed = frame.loc[:, columns].dropna()
        if len(observed) < min_n or observed[predictor].nunique() < 5 or observed[outcome].nunique() < 5:
            continue
        controls = [observed["combine_weight"].to_numpy(dtype=float)]
        if use_role:
            controls.append(observed["role"].eq("CB").astype(float).to_numpy())
        statistic = partial_spearman(
            observed[predictor].to_numpy(dtype=float), observed[outcome].to_numpy(dtype=float), controls
        )
        sign = predictor_orientation(predictor) * outcome_orientation(outcome)
        oriented = statistic.r_adj * sign if sign else float("nan")
        years = observed["draft_year"]
        rows.append(
            {
                "scope": "db_drills" if group == "DB" else "te_drills",
                "group": group,
                "outcome": outcome,
                "predictor": predictor,
                "family": predictor_family,
                "n": len(observed),
                "r_raw": statistic.r_raw,
                "r_adj": statistic.r_adj,
                "ci_lo": statistic.ci_lo,
                "ci_hi": statistic.ci_hi,
                "p": statistic.p,
                "q_scope": np.nan,
                "oriented_r": oriented,
                "direction": orientation_label(oriented),
                "r_disc_2023_24": _raw_spearman(observed[years.isin([2023, 2024])], predictor, outcome),
                "r_hold_2025": _raw_spearman(observed[years.eq(2025)], predictor, outcome),
                "r_2023": _raw_spearman(observed[years.eq(2023)], predictor, outcome),
                "r_2024": _raw_spearman(observed[years.eq(2024)], predictor, outcome),
                "r_2025": _raw_spearman(observed[years.eq(2025)], predictor, outcome),
                "family_label": family_label,
                "low_power": group != "DB" or len(observed) < 60,
            }
        )
    result = pd.DataFrame(rows, columns=list(FAMILY_COLUMNS))
    if not result.empty:
        result["q_scope"] = benjamini_hochberg(result["p"].to_numpy(dtype=float))
    return result


def build_families(
    predictors: pd.DataFrame,
    available: dict[str, str],
    adequate: Sequence[str],
) -> dict[str, list[tuple[str, str, str]]]:
    repeatable = predictors.loc[predictors["status"].eq("repeatable")]
    unverifiable = predictors.loc[predictors["status"].eq("unverifiable")]

    def columns(table: pd.DataFrame, drills: Sequence[str] | None = None) -> list[str]:
        subset = table if drills is None else table.loc[table["drill"].isin(drills)]
        return [name for name in subset["predictor"] if name in available]

    official = [name for name in OFFICIAL_REFERENCE if name in available]
    pools = {
        "pair_backpedal": columns(repeatable, BACKPEDAL_DRILLS),
        "pair_cod": columns(repeatable, COD_DRILLS),
        "pair_backpedal_unverifiable": columns(unverifiable, BACKPEDAL_DRILLS),
        "pair_cod_unverifiable": columns(unverifiable, COD_DRILLS),
        "pair_speed_burst": [
            *columns(repeatable, ("FORTY_YARD_DASH",)),
            *(n for n in ("forty", "ten_yd_split") if n in official),
        ],
    }
    families: dict[str, list[tuple[str, str, str]]] = {
        label: [
            (name, available[name], outcome)
            for name in pools[label]
            for outcome in PAIR_OUTCOMES[label.removesuffix("_unverifiable")]
            if outcome in adequate
        ]
        for label in pools
    }
    families["all_by_all"] = [
        (name, available[name], outcome) for name in [*columns(repeatable), *official] for outcome in adequate
    ]
    families["unverifiable_repeatability"] = [
        (name, available[name], outcome) for name in columns(unverifiable) for outcome in adequate
    ]
    return families


def family_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, group in results.groupby("family_label", sort=True):
        survivors = group.loc[group["q_scope"].lt(0.10)]
        classified = group.loc[group["direction"].isin(["expected", "opposite"])]
        agree = survivors["r_hold_2025"].notna() & (
            np.sign(survivors["r_hold_2025"]) == np.sign(survivors["r_disc_2023_24"])
        )
        rows.append(
            {
                "family_label": label,
                "tests": len(group),
                "raw_p_lt_05": int(group["p"].lt(0.05).sum()),
                "chance_expected": 0.05 * len(group),
                "min_q": float(group["q_scope"].min()),
                "q_lt_10": len(survivors),
                "q_lt_05": int(group["q_scope"].lt(0.05).sum()),
                "expected_all": int(classified["direction"].eq("expected").sum()),
                "opposite_all": int(classified["direction"].eq("opposite").sum()),
                "expected_raw_p_lt_05": int(group.loc[group["p"].lt(0.05), "direction"].eq("expected").sum()),
                "opposite_raw_p_lt_05": int(group.loc[group["p"].lt(0.05), "direction"].eq("opposite").sum()),
                "survivors_holdout_same_sign": int(agree.sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- TE family


def te_outcome_scores(player_play: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = player_play.loc[
        player_play["nfl_position"].isin(["WR", "TE"])
        & player_play["route_ran"].notna()
        & player_play["play_nullified_by_penalty"].ne("Y")
    ].copy()
    gates: list[dict[str, object]] = []
    scores = pd.DataFrame({"nfl_id": routes.loc[routes["nfl_position"].eq("TE"), "nfl_id"].drop_duplicates()})
    for outcome, subset, _minimum in _outcome_definitions(routes):
        if outcome not in TE_OUTCOMES:
            continue
        subset = subset.dropna(subset=[outcome])
        formula, numeric, categorical = _context_formula(outcome, subset)
        model_data = subset.dropna(subset=[outcome, *numeric, *categorical])
        residual = f"{outcome}_residual"
        fitted = model_data.assign(**{residual: smf.ols(formula, data=model_data).fit().resid.to_numpy()})
        tight = fitted.loc[fitted["nfl_position"].eq("TE")]
        counts = tight.groupby("nfl_id").size()
        eligible = tight.loc[tight["nfl_id"].isin(counts[counts.ge(TE_MIN_ROUTES)].index)]
        players = int(eligible["nfl_id"].nunique())
        if players < 3:
            gates.append(_gate_row(outcome, "TE", formula, players))
            continue
        split = split_half_reliability(eligible, "nfl_id", residual, min_n=2, n_splits=2000, seed=20261103)
        lower, upper = player_bootstrap_ci(split, n_bootstrap=1000, seed=20261104)
        model = mixed_model_icc(eligible, "nfl_id", residual)
        status = verdict(split.median, lower)
        gates.append(
            {
                **_gate_row(outcome, "TE", formula, players),
                "SB_median": split.median,
                "lower_ci": lower,
                "upper_ci": upper,
                "icc": model.icc,
                "verdict": status,
            }
        )
        if status == "adequate":
            shrunk = empirical_bayes_shrink(eligible, "nfl_id", residual, model, min_n=2)
            score = shrunk.loc[:, ["nfl_id", "posterior_mean"]].rename(columns={"posterior_mean": outcome})
            scores = scores.merge(score, on="nfl_id", how="left", validate="one_to_one")
    return pd.DataFrame(gates), scores


def te_frame(combine: pd.DataFrame, players: pd.DataFrame, sensors: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    tight = combine.loc[combine["combine_position"].eq("TE")].copy()
    meta = players.loc[:, ["nfl_id", "nfl_position"]]
    frame = tight.merge(meta, on="nfl_id", how="left", validate="one_to_one")
    frame = frame.merge(sensors, on="nfl_id", how="left", validate="one_to_one")
    frame["nfl_id"] = frame["nfl_id"].astype(int)
    scored = frame.merge(
        scores.assign(nfl_id=scores["nfl_id"].astype(int)), on="nfl_id", how="inner", validate="one_to_one"
    )
    return scored.assign(role="TE")


def te_families(
    predictors: pd.DataFrame, columns: Iterable[str], adequate: Sequence[str]
) -> dict[str, list[tuple[str, str, str]]]:
    present = set(columns)
    result: dict[str, list[tuple[str, str, str]]] = {}
    for label, status in (("te_only", "repeatable"), ("te_unverifiable", "unverifiable")):
        names = [name for name in predictors.loc[predictors["status"].eq(status), "predictor"] if name in present]
        result[label] = [(name, "sensor", outcome) for name in names for outcome in adequate]
    return result


# ---------------------------------------------------------------- orchestration


def _load_combine(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = root / "data" / "raw"
    return pd.read_csv(raw / "combine_results.csv"), pd.read_csv(raw / "players.csv")


def _cached(path: Path, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    frame = build()
    frame.to_parquet(path, index=False)
    return frame


def _db_predictor_frame(
    combine: pd.DataFrame, players: pd.DataFrame, attempts: pd.DataFrame, table: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    db = combine.loc[combine["combine_position"].eq("DB")].copy()
    meta = players.loc[:, ["nfl_id", "nfl_position"]]
    frame = db.merge(meta, on="nfl_id", how="left", validate="one_to_one")
    sensors = sensor_wide(attempts, table, "repeatable").merge(
        sensor_wide(attempts, table, "unverifiable"), on="nfl_id", how="outer", validate="one_to_one"
    )
    frame = frame.merge(sensors, on="nfl_id", how="left", validate="one_to_one")
    frame = frame.copy()
    frame["role"] = frame["nfl_position"].map(role_from_position)
    available = {name: "official" for name in OFFICIAL_REFERENCE if name in frame}
    for row in table.itertuples(index=False):
        if row.status in ("repeatable", "unverifiable") and row.predictor in frame:
            available[str(row.predictor)] = "sensor_" + str(row.status)
    return frame, {k: v for k, v in available.items() if frame[k].notna().sum() >= MIN_PREDICTOR_N}


def run_scope(root: Path) -> dict[str, pd.DataFrame]:
    processed, reports = root / "data" / "processed", root / "reports"
    combine, players = _load_combine(root)
    combine["nfl_id"] = combine["nfl_id"].astype(int)
    players["nfl_id"] = players["nfl_id"].astype(int)
    db_ids = set(combine.loc[combine["combine_position"].eq("DB"), "nfl_id"].astype(str))
    te_ids = set(combine.loc[combine["combine_position"].eq("TE"), "nfl_id"].astype(str))
    db_attempts = _cached(
        processed / "scope_db_drills_attempts.parquet", lambda: build_attempts(root, db_ids, DB_DRILL_TYPES)
    )
    te_attempts = _cached(
        processed / "scope_db_drills_te_attempts.parquet", lambda: build_attempts(root, te_ids, TE_DRILL_TYPES)
    )
    db_table = classify_predictors(db_attempts, [*DB_DRILLS, *STANDARD_DRILLS])
    te_table = classify_predictors(te_attempts, sorted(te_attempts["drill"].unique()))
    pd.concat([db_table.assign(group="DB"), te_table.assign(group="TE")]).to_csv(
        reports / "scope_db_drills_predictor_gate.csv", index=False
    )

    player_play = pd.read_parquet(processed / "player_play_joined.parquet")
    features = _cached(
        processed / "scope_db_drills_tracking_features.parquet", lambda: build_tracking_features(root, player_play)
    )
    g1, g2 = outcome_frames(player_play, features)
    gate, scores = run_gates(g1, g2)
    te_gate, te_scores = te_outcome_scores(player_play)
    pd.concat([gate, te_gate]).to_csv(reports / "scope_db_drills_outcome_gate.csv", index=False)
    pd.DataFrame(INFEASIBLE, columns=["outcome", "reason"]).to_csv(
        reports / "scope_db_drills_infeasible.csv", index=False
    )

    adequate = gate.loc[gate["verdict"].eq("adequate"), "outcome"].tolist()
    frame, available = _db_predictor_frame(combine, players, db_attempts, db_table)
    frame = frame.merge(scores, on="nfl_id", how="inner", validate="one_to_one")
    tables = [
        association_table(frame, pairs, label) for label, pairs in build_families(db_table, available, adequate).items()
    ]
    te_adequate = te_gate.loc[te_gate["verdict"].eq("adequate"), "outcome"].tolist()
    te_sensors = sensor_wide(te_attempts, te_table, "repeatable").merge(
        sensor_wide(te_attempts, te_table, "unverifiable"), on="nfl_id", how="outer", validate="one_to_one"
    )
    te_data = te_frame(combine, players, te_sensors, te_scores)
    for label, pairs in te_families(te_table, te_data.columns, te_adequate).items():
        tables.append(association_table(te_data, pairs, label, min_n=MIN_TE_N, use_role=False, group="TE"))
    results = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    results.to_csv(reports / "scope_db_drills_results.csv", index=False)
    summary = family_summary(results)
    summary.to_csv(reports / "scope_db_drills_summary.csv", index=False)
    return {"gate": gate, "te_gate": te_gate, "predictors": db_table, "results": results, "summary": summary}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    outputs = run_scope(arguments.root)
    print(outputs["gate"].drop(columns=["formula"]).to_string(index=False))
    print(outputs["te_gate"].drop(columns=["formula"]).to_string(index=False))
    print(outputs["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
