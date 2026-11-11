from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from combine_dlol import scope_dl, scope_ngs_agility, scope_wrte_drills

GROUPS = ("DL", "OL", "DB", "WR", "TE")
SEED = 20261111
BOOTSTRAP_REPS = 2000
MIN_N = 30
MIN_YEAR_N = 15
SESOI_SD = 0.20
SD_UNIT = 0.5
POWER_MULTIPLIER = 1.959964 + 0.841621
SHRINK_THRESHOLD = 0.40
SHRINK_REFERENCE_P = 0.10
SUGGESTIVE_Q = 0.10
SUPPORTED_Q = 0.10
MIN_ROUTE_PLAYERS = 30
WEIGHT_UNIT = {"DL": 10.0, "OL": 10.0, "DB": 5.0, "WR": 5.0, "TE": 5.0}
MASTER_ALIASES = {
    "DL": ("pooled", "DL"),
    "OL": ("OL",),
    "DB": ("DB",),
    "WR": ("WR", "WRTE"),
    "TE": ("TE", "WRTE"),
}
OFFICIAL_COLUMNS = (
    "forty",
    "ten_yd_split",
    "three_cone",
    "short_shuttle",
    "vertical",
    "broad_jump",
    "bench_reps",
    "combine_weight",
    "combine_height",
    "arm_length",
    "wing_span",
    "hand_size",
)
SENSOR_COLUMNS = (
    "forty_yard_dash_top_speed",
    "forty_yard_dash_burst_speed_1_0s",
    "forty_yard_dash_sensor_ten_yd_time",
    "forty_yard_dash_sensor_ten_yd_burst_speed",
    "forty_yard_dash_peak_acceleration",
    "forty_yard_dash_peak_deceleration",
    "forty_yard_dash_path_efficiency",
    "forty_yard_dash_time_in_drill",
    "short_shuttle_time_in_drill",
    "three_cone_drill_time_in_drill",
)
ROUTE_PREFIX = "route_"
ROUTE_SUFFIX = "_top_speed"
KEY_COLUMNS = ("nfl_id", "group", "role", "role_ctl", "draft_year", "draft_overall_pick", "log_pick", "nfl_position")
FIXED = "fixed"
TRAINABLE = "trainable"
PARTIAL = "partially_trainable"
LITERATURE_SPRINT = "Rumpf et al. 2016 J Strength Cond Res (sprint training review); Seitz et al. 2014 Sports Med"
LITERATURE_COD = "Sheppard and Young 2006 J Sports Sci; Brughelli et al. 2008 Sports Med"
LITERATURE_GROWTH = "Standard human growth biology (growth plates close by early adulthood)"
JUDGMENT = "judgment call"


@dataclass(frozen=True)
class Measure:
    column: str
    family: str
    trainability: str
    justification: str
    source: str
    unit: float | None
    unit_label: str
    direction: int

    @property
    def rank_eligible(self) -> bool:
        return self.trainability != FIXED


def _sd_measure(column: str, family: str, trainability: str, why: str, source: str, direction: int) -> Measure:
    return Measure(
        column, family, trainability, why, source, None, "0.5 SD of the measure within position group", direction
    )


TRAINABILITY: tuple[Measure, ...] = (
    Measure(
        "forty",
        "official",
        PARTIAL,
        "Acceleration responds to sprint and strength training; top-end speed has a larger genetic ceiling.",
        LITERATURE_SPRINT,
        0.10,
        "0.10 s faster",
        -1,
    ),
    Measure(
        "ten_yd_split",
        "official",
        TRAINABLE,
        "Start and first-step acceleration respond to resisted sprints, strength work and start technique.",
        LITERATURE_SPRINT,
        0.05,
        "0.05 s faster",
        -1,
    ),
    Measure(
        "three_cone",
        "official",
        TRAINABLE,
        "Change-of-direction technique and eccentric strength improve with coaching (part is drill familiarity).",
        LITERATURE_COD,
        0.15,
        "0.15 s faster",
        -1,
    ),
    Measure(
        "short_shuttle",
        "official",
        TRAINABLE,
        "Change-of-direction technique and braking strength improve with coaching and repetition.",
        LITERATURE_COD,
        0.10,
        "0.10 s faster",
        -1,
    ),
    Measure(
        "vertical",
        "official",
        TRAINABLE,
        "Plyometric and strength training reliably raise jump height in athletes.",
        "Markovic 2007 Br J Sports Med (plyometric meta-analysis)",
        2.0,
        "2 in higher",
        1,
    ),
    Measure(
        "broad_jump",
        "official",
        TRAINABLE,
        "Horizontal jump power responds to plyometric and strength work (carry-over from vertical is a judgment call).",
        f"Markovic 2007 Br J Sports Med; {JUDGMENT} for horizontal transfer",
        4.0,
        "4 in farther",
        1,
    ),
    Measure(
        "bench_reps",
        "official",
        TRAINABLE,
        "Repetitions at a fixed load rise with resistance training in the first years of a program.",
        "ACSM 2009 position stand on resistance training progression",
        3.0,
        "3 more reps",
        1,
    ),
    Measure(
        "combine_weight",
        "official",
        PARTIAL,
        "Body mass can be moved over months, but weight is not lean mass and heavier may be slower.",
        f"{JUDGMENT} (body-composition caveat)",
        None,
        "10 lb heavier (DL/OL), 5 lb heavier (DB/WR/TE)",
        1,
    ),
    _sd_measure(
        "combine_height", "official", FIXED, "Adult stature is fixed once growth plates close.", LITERATURE_GROWTH, 1
    ),
    _sd_measure("arm_length", "official", FIXED, "Limb length is skeletal and fixed in adults.", LITERATURE_GROWTH, 1),
    _sd_measure("wing_span", "official", FIXED, "Wingspan is skeletal and fixed in adults.", LITERATURE_GROWTH, 1),
    _sd_measure("hand_size", "official", FIXED, "Hand size is skeletal and fixed in adults.", LITERATURE_GROWTH, 1),
    _sd_measure(
        "forty_yard_dash_top_speed",
        "sensor",
        PARTIAL,
        "Maximal velocity can be improved but has a larger genetic and tendon-stiffness ceiling than acceleration.",
        LITERATURE_SPRINT,
        1,
    ),
    _sd_measure(
        "forty_yard_dash_burst_speed_1_0s",
        "sensor",
        TRAINABLE,
        "Speed one second into the run is an acceleration quality that responds to start and strength training.",
        f"{LITERATURE_SPRINT}; {JUDGMENT} on the 1.0 s window",
        1,
    ),
    _sd_measure(
        "forty_yard_dash_sensor_ten_yd_time",
        "sensor",
        TRAINABLE,
        "Sensor 10-yard time is the same acceleration quality as the official 10-yard split.",
        LITERATURE_SPRINT,
        -1,
    ),
    _sd_measure(
        "forty_yard_dash_sensor_ten_yd_burst_speed",
        "sensor",
        TRAINABLE,
        "Average speed over the first 10 yards is the same acceleration quality as the 10-yard split.",
        LITERATURE_SPRINT,
        1,
    ),
    _sd_measure(
        "forty_yard_dash_peak_acceleration",
        "sensor",
        PARTIAL,
        "Peak acceleration depends on force output and start posture; trainable but bounded by traction and body size.",
        f"{LITERATURE_SPRINT}; {JUDGMENT}",
        1,
    ),
    _sd_measure(
        "forty_yard_dash_peak_deceleration",
        "sensor",
        PARTIAL,
        "Braking ability responds to eccentric strength work and technique, but a single-run peak is noisy.",
        JUDGMENT,
        1,
    ),
    _sd_measure(
        "forty_yard_dash_path_efficiency",
        "sensor",
        PARTIAL,
        "Running a straighter line is partly technique; there is little room in a straight 40.",
        JUDGMENT,
        1,
    ),
    _sd_measure(
        "forty_yard_dash_time_in_drill",
        "sensor",
        PARTIAL,
        "Sensor 40 time carries both acceleration (trainable) and top speed (more fixed).",
        LITERATURE_SPRINT,
        -1,
    ),
    _sd_measure(
        "short_shuttle_time_in_drill",
        "sensor",
        TRAINABLE,
        "Sensor short-shuttle time is the same change-of-direction quality as the official shuttle.",
        LITERATURE_COD,
        -1,
    ),
    _sd_measure(
        "three_cone_drill_time_in_drill",
        "sensor",
        TRAINABLE,
        "Sensor three-cone time is the same change-of-direction quality as the official cone drill.",
        LITERATURE_COD,
        -1,
    ),
    _sd_measure(
        f"{ROUTE_PREFIX}<drill>{ROUTE_SUFFIX}",
        "route_drill",
        PARTIAL,
        "Route-drill top speed mixes sprint speed with route technique; most drills have one attempt.",
        f"{LITERATURE_SPRINT}; {JUDGMENT}",
        1,
    ),
)
MEASURE_BY_COLUMN = {measure.column: measure for measure in TRAINABILITY}
ROUTE_TEMPLATE = MEASURE_BY_COLUMN[f"{ROUTE_PREFIX}<drill>{ROUTE_SUFFIX}"]

CONFOUNDER_BASE = "draft pedigree; program quality; playing time and role usage"
RESULT_COLUMNS = (
    "group",
    "measure",
    "family",
    "trainability",
    "rank_eligible",
    "outcome",
    "outcome_signed",
    "n",
    "improvement_unit",
    "unit_label",
    "effect",
    "ci_lo",
    "ci_hi",
    "outcome_sd",
    "effect_sd",
    "ci_lo_sd",
    "ci_hi_sd",
    "p",
    "q_group",
    "residual_sd",
    "mde",
    "mde_sd",
    "sesoi",
    "max_effect_bound",
    "verdict",
    "n_draft",
    "effect_with_draft",
    "draft_shrink",
    "effect_without_weight",
    "weight_shrink",
    "weight_confounded",
    "confounded_by",
    "n_disc",
    "effect_disc_2023_24",
    "n_2025",
    "effect_2025",
    "holdout_sign_agrees",
    "master_q",
    "grade",
    "confounder_note",
    "status",
)


@dataclass(frozen=True)
class Fit:
    beta: float
    se: float
    p: float
    n: int
    residual_sd: float


def route_measure(drill: str) -> Measure:
    column = f"{ROUTE_PREFIX}{drill}{ROUTE_SUFFIX}"
    return Measure(
        column,
        ROUTE_TEMPLATE.family,
        ROUTE_TEMPLATE.trainability,
        ROUTE_TEMPLATE.justification,
        ROUTE_TEMPLATE.source,
        None,
        ROUTE_TEMPLATE.unit_label,
        ROUTE_TEMPLATE.direction,
    )


def measure_for(column: str) -> Measure:
    if column.startswith(ROUTE_PREFIX) and column.endswith(ROUTE_SUFFIX):
        return route_measure(column[len(ROUTE_PREFIX) : -len(ROUTE_SUFFIX)])
    return MEASURE_BY_COLUMN[column]


def trainability_table() -> pd.DataFrame:
    rows = [
        {
            "measure": measure.column,
            "family": measure.family,
            "trainability": measure.trainability,
            "rank_eligible": measure.rank_eligible,
            "justification": measure.justification,
            "source": measure.source,
            "realistic_improvement": measure.unit if measure.unit is not None else np.nan,
            "improvement_unit": measure.unit_label,
            "improvement_direction": "lower is better" if measure.direction < 0 else "higher is more",
        }
        for measure in TRAINABILITY
    ]
    return pd.DataFrame(rows)


def write_trainability(path: Path) -> pd.DataFrame:
    table = trainability_table()
    table.to_csv(path, index=False)
    return table


def improvement_unit(measure: Measure, group: str, values: pd.Series) -> float:
    if measure.column == "combine_weight":
        return WEIGHT_UNIT[group]
    if measure.unit is not None:
        return measure.unit
    return float(SD_UNIT * values.dropna().std(ddof=1))


def route_columns(table: pd.DataFrame) -> list[str]:
    return [name for name in table.columns if name.startswith(ROUTE_PREFIX) and name.endswith(ROUTE_SUFFIX)]


def route_speed_table(root: Path) -> pd.DataFrame:
    processed = root / "data/processed/scope_wrte_drills_attempts.parquet"
    gate_path = root / "reports/scope_wrte_drills_repeatability_gate.csv"
    if not processed.is_file() or not gate_path.is_file():
        return pd.DataFrame({"nfl_id": pd.Series(dtype=int)})
    wide = scope_wrte_drills.player_drill_table(pd.read_parquet(processed), pd.read_csv(gate_path))
    speed = wide.loc[:, ["nfl_id", *[name for name in wide.columns if name.endswith(ROUTE_SUFFIX)]]]
    renamed = speed.rename(columns={name: f"{ROUTE_PREFIX}{name}" for name in speed.columns if name != "nfl_id"})
    return renamed.assign(nfl_id=pd.to_numeric(renamed["nfl_id"]).astype(int))


DRILL_PREFIX = {
    "FORTY_YARD_DASH": "forty_yard_dash",
    "SHORT_SHUTTLE": "short_shuttle",
    "THREE_CONE_DRILL": "three_cone_drill",
}
SENSOR_METRICS = (
    "time_in_drill",
    "top_speed",
    "burst_speed_1_0s",
    "peak_acceleration",
    "peak_deceleration",
    "path_efficiency",
    "sensor_ten_yd_time",
    "sensor_ten_yd_burst_speed",
)


def attempts_to_wide(attempts: pd.DataFrame) -> pd.DataFrame:
    selected = attempts.loc[attempts["drill"].isin(DRILL_PREFIX) & attempts["metric"].isin(SENSOR_METRICS)]
    labelled = selected.assign(
        column=[f"{DRILL_PREFIX[str(d)]}_{m}" for d, m in zip(selected["drill"], selected["metric"], strict=True)]
    )
    wide = labelled.pivot_table(index="nfl_id", columns="column", values="value", aggfunc="mean")
    return wide.reset_index().rename_axis(columns=None)


def db_attempts_long(attempts: pd.DataFrame) -> pd.DataFrame:
    metrics = [name for name in SENSOR_METRICS if name in attempts]
    melted = attempts.melt(id_vars=["nfl_id", "drill"], value_vars=metrics, var_name="metric", value_name="value")
    return melted.dropna(subset=["value"])


def sensor_table(root: Path) -> pd.DataFrame:
    processed = root / "data/processed"
    linemen = pd.read_parquet(processed / "combine_predictors.parquet", columns=["nfl_id", *SENSOR_COLUMNS])
    receivers = attempts_to_wide(pd.read_parquet(processed / "scope_wrte_sensor_attempts.parquet"))
    backs = attempts_to_wide(db_attempts_long(pd.read_parquet(processed / "scope_db_sensor_attempts.parquet")))
    frames = [frame.assign(nfl_id=pd.to_numeric(frame["nfl_id"]).astype(int)) for frame in (linemen, receivers, backs)]
    combined = pd.concat(frames, ignore_index=True).reindex(columns=["nfl_id", *SENSOR_COLUMNS])
    return combined.drop_duplicates(subset="nfl_id")


def load_player_table(root: Path) -> pd.DataFrame:
    combine = pd.read_csv(root / "data/raw/combine_results.csv")
    players = pd.read_csv(root / "data/raw/players.csv")
    gate = scope_ngs_agility.gate_table(root)
    frames = []
    for group in GROUPS:
        outcomes = scope_ngs_agility.stable_outcomes(gate, group)
        scores = scope_ngs_agility.outcome_frame(root, scope_ngs_agility.GATE_GROUP[group], outcomes)
        frame = scope_ngs_agility.build_group_frame(combine, players, scores, group)
        frames.append(frame.assign(group=group))
    stacked = pd.concat(frames, ignore_index=True)
    merged = stacked.merge(sensor_table(root), on="nfl_id", how="left", validate="one_to_one")
    routes = route_speed_table(root)
    merged = merged.merge(routes, on="nfl_id", how="left", validate="one_to_one")
    outcome_names = sorted({name for group in GROUPS for name in scope_ngs_agility.stable_outcomes(gate, group)})
    keep = [
        *KEY_COLUMNS,
        *OFFICIAL_COLUMNS,
        *SENSOR_COLUMNS,
        *route_columns(merged),
        *[name for name in outcome_names if name in merged],
    ]
    return merged.loc[:, list(dict.fromkeys(keep))].reset_index(drop=True)


def outcome_columns(table: pd.DataFrame) -> list[str]:
    known = {*KEY_COLUMNS, *OFFICIAL_COLUMNS, *SENSOR_COLUMNS, *route_columns(table)}
    return [name for name in table.columns if name not in known]


def group_outcomes(table: pd.DataFrame, group: str) -> list[str]:
    rows = table.loc[table["group"].eq(group)]
    return [name for name in outcome_columns(table) if rows[name].notna().sum() > 0]


def validate_player_table(table: pd.DataFrame) -> pd.DataFrame:
    sizes = table.groupby("group").size()
    checks = [
        (
            "no duplicate players",
            not table["nfl_id"].duplicated().any(),
            f"{int(table['nfl_id'].duplicated().sum())} duplicates",
        ),
        ("groups are DL/OL/DB/WR/TE only", set(table["group"]) <= set(GROUPS), ",".join(sorted(set(table["group"])))),
        ("every group has rows", all(sizes.get(group, 0) > 0 for group in GROUPS), str(sizes.to_dict())),
        ("draft years 2023-2025", set(table["draft_year"].dropna().astype(int)) <= {2023, 2024, 2025}, "draft_year"),
        ("no null keys", not table[["nfl_id", "group", "draft_year"]].isna().any().any(), "nfl_id/group/draft_year"),
        (
            "combine_weight present for most players",
            float(table["combine_weight"].notna().mean()) >= 0.9,
            f"{float(table['combine_weight'].notna().mean()):.3f} non-null",
        ),
        ("every group has an outcome column", all(group_outcomes(table, group) for group in GROUPS), "stable outcomes"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def assert_valid(table: pd.DataFrame) -> pd.DataFrame:
    report = validate_player_table(table)
    failed = report.loc[~report["passed"], "check"].tolist()
    if failed:
        raise ValueError(f"player table failed validation: {failed}")
    return report


def data_dictionary(table: pd.DataFrame) -> pd.DataFrame:
    fixed = {
        "nfl_id": ("Player identifier", "id"),
        "group": ("Position group: DL (EDGE+IDL), OL, DB, WR, TE", "label"),
        "role": ("Role within group (EDGE/IDL, T/IOL, CB/S, WR/TE)", "label"),
        "role_ctl": ("0/1 role control (EDGE, T, CB); missing when the group has one role", "0/1"),
        "draft_year": ("NFL draft class", "year"),
        "draft_overall_pick": ("Overall draft pick; missing for undrafted players", "pick"),
        "log_pick": ("Natural log of overall pick", "log pick"),
        "nfl_position": ("Listed NFL position", "label"),
    }
    rows = [(name, text, unit, "players.csv / combine_results.csv") for name, (text, unit) in fixed.items()]
    rows += [
        (m.column, m.justification, m.unit_label, "official combine") for m in TRAINABILITY if m.family == "official"
    ]
    rows += [
        (m.column, m.justification, "sensor units", "combine_tracking sensors")
        for m in TRAINABILITY
        if m.family == "sensor"
    ]
    rows += [
        (name, "Route-drill top speed (one drill)", "sensor units", "combine_tracking route drills")
        for name in route_columns(table)
    ]
    rows += [
        (name, "Stable game outcome score (context-adjusted)", "outcome units", "game tracking")
        for name in outcome_columns(table)
    ]
    return pd.DataFrame(rows, columns=["column", "description", "units", "source"])


def _design(sub: pd.DataFrame, predictor: str, controls: Sequence[str]) -> tuple[np.ndarray, float]:
    columns = [predictor, *controls]
    values = sub.loc[:, columns].to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    spread = values.std(axis=0, ddof=1)
    scaled = centered / np.where(spread > 0, spread, 1.0)
    return np.column_stack([np.ones(len(sub)), scaled]), float(spread[0])


def fit_ols(y: np.ndarray, x: np.ndarray, predictor_sd: float) -> Fit:
    n, k = x.shape
    gram_inverse = np.linalg.pinv(x.T @ x)
    coefficients = gram_inverse @ x.T @ y
    residuals = y - x @ coefficients
    leverage = np.clip(np.einsum("ij,jk,ik->i", x, gram_inverse, x), 0.0, 0.999)
    meat = (x * (residuals / (1.0 - leverage))[:, None]).T @ (x * (residuals / (1.0 - leverage))[:, None])
    covariance = gram_inverse @ meat @ gram_inverse
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    statistic = coefficients[1] / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(statistic), df=max(n - k, 1))) if np.isfinite(statistic) else np.nan
    residual_sd = float(np.sqrt(residuals @ residuals / max(n - k, 1)))
    return Fit(float(coefficients[1] / predictor_sd), se / predictor_sd, p, n, residual_sd)


def bootstrap_slopes(y: np.ndarray, x: np.ndarray, predictor_sd: float, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(y), size=(reps, len(y)))
    xb = x[index]
    yb = y[index]
    gram = np.einsum("rni,rnj->rij", xb, xb)
    moment = np.einsum("rni,rn->ri", xb, yb)
    coefficients = np.einsum("rij,rj->ri", np.linalg.pinv(gram), moment)
    return coefficients[:, 1] / predictor_sd


def fit_subset(sub: pd.DataFrame, predictor: str, outcome: str, controls: Sequence[str]) -> Fit:
    x, spread = _design(sub, predictor, controls)
    return fit_ols(sub[outcome].to_numpy(dtype=float), x, spread)


def shrink(reference: float, adjusted: float) -> float:
    if not np.isfinite(reference) or reference == 0 or not np.isfinite(adjusted):
        return float("nan")
    return float(1.0 - adjusted * np.sign(reference) / abs(reference))


def verdict_for(ci_lo: float, ci_hi: float, signed: bool, sesoi: float) -> tuple[str, float]:
    bound = max(abs(ci_lo), abs(ci_hi))
    excludes_zero = ci_lo > 0 or ci_hi < 0
    if excludes_zero and not signed:
        return "unsigned_association", bound
    if ci_lo > 0:
        return "supported_benefit", bound
    if ci_hi < 0:
        return "supported_opposite", bound
    if bound <= sesoi:
        return "no_effect_larger_than_X", bound
    return "cannot_say", bound


def evidence_grade(
    *, master_q: float, p: float, holdout_agrees: bool, q_group: float, ci_lo: float, ci_hi: float
) -> str:
    if np.isfinite(master_q) and master_q < SUPPORTED_Q:
        return "supported"
    nominal = np.isfinite(p) and p < 0.05 and holdout_agrees and np.isfinite(q_group) and q_group < SUGGESTIVE_Q
    excludes_zero = np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0)
    return "suggestive" if nominal or excludes_zero else "speculative"


def master_lookup(path: Path) -> dict[tuple[str, str, str], float]:
    if not path.is_file():
        return {}
    master = pd.read_csv(path, usecols=["group", "outcome", "predictor", "q_master"])
    best = master.groupby(["group", "outcome", "predictor"], as_index=False)["q_master"].min()
    return {(str(r.group), str(r.outcome), str(r.predictor)): float(r.q_master) for r in best.itertuples()}


def master_q(lookup: dict[tuple[str, str, str], float], group: str, outcome: str, predictor: str) -> float:
    values = [
        lookup[(alias, outcome, predictor)] for alias in MASTER_ALIASES[group] if (alias, outcome, predictor) in lookup
    ]
    return min(values) if values else float("nan")


def confounder_note(group: str, measure: Measure, confounded_by: str) -> str:
    parts = [CONFOUNDER_BASE]
    if group in ("DL", "OL", "DB"):
        parts.append("role (position within group)")
    if measure.column != "combine_weight":
        parts.append("body size (weight controlled, not lean mass)")
    if confounded_by:
        parts.append(f"effect shrinks more than 40% with {confounded_by}")
    return "; ".join(parts)


def _blank_row(group: str, measure: Measure, outcome: str, signed: bool, n: int, status: str) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(RESULT_COLUMNS, np.nan)
    row.update(
        group=group,
        measure=measure.column,
        family=measure.family,
        trainability=measure.trainability,
        rank_eligible=measure.rank_eligible,
        outcome=outcome,
        outcome_signed=signed,
        n=n,
        status=status,
        verdict="not_tested",
        grade="speculative",
        weight_confounded=False,
        confounded_by="",
        holdout_sign_agrees=False,
        unit_label=measure.unit_label,
        confounder_note=confounder_note(group, measure, ""),
    )
    return row


def _scaled(value: float, scale: float) -> float:
    return value * scale if np.isfinite(value) else float("nan")


def _slope_effect(sub: pd.DataFrame, predictor: str, outcome: str, controls: Sequence[str], scale: float) -> float:
    if len(sub) < MIN_YEAR_N:
        return float("nan")
    return _scaled(fit_subset(sub, predictor, outcome, controls).beta, scale)


def effect_row(
    frame: pd.DataFrame,
    *,
    group: str,
    measure: Measure,
    outcome: str,
    unit: float,
    outcome_sign: int,
    controls: Sequence[str],
    lookup: dict[tuple[str, str, str], float],
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    signed = outcome_sign != 0
    needed = [measure.column, outcome, *controls]
    sub = frame.dropna(subset=needed)
    if len(sub) < MIN_N:
        return _blank_row(group, measure, outcome, signed, len(sub), f"not_tested_n<{MIN_N}")
    if sub[measure.column].nunique() < 5 or sub[outcome].nunique() < 5:
        return _blank_row(group, measure, outcome, signed, len(sub), "degenerate")
    scale = unit * measure.direction * (outcome_sign if signed else 1)
    design, spread = _design(sub, measure.column, controls)
    y = sub[outcome].to_numpy(dtype=float)
    fit = fit_ols(y, design, spread)
    draws = bootstrap_slopes(y, design, spread, reps, SEED) * scale
    ci_lo, ci_hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    outcome_sd = float(sub[outcome].std(ddof=1))
    sesoi = SESOI_SD * outcome_sd
    effect = fit.beta * scale
    verdict, bound = verdict_for(ci_lo, ci_hi, signed, sesoi)
    mde = POWER_MULTIPLIER * fit.se * abs(scale)
    row = _blank_row(group, measure, outcome, signed, fit.n, "ok")
    row.update(
        improvement_unit=unit,
        effect=effect,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        outcome_sd=outcome_sd,
        effect_sd=effect / outcome_sd,
        ci_lo_sd=ci_lo / outcome_sd,
        ci_hi_sd=ci_hi / outcome_sd,
        p=fit.p,
        residual_sd=fit.residual_sd,
        mde=mde,
        mde_sd=mde / outcome_sd,
        sesoi=sesoi,
        max_effect_bound=bound,
        verdict=verdict,
    )
    row.update(_sensitivity(sub, measure, outcome, controls, scale, effect))
    row["master_q"] = master_q(lookup, group, outcome, measure.column)
    note = row["confounded_by"]
    row["confounder_note"] = confounder_note(group, measure, note)
    return row


def _sensitivity(
    sub: pd.DataFrame, measure: Measure, outcome: str, controls: Sequence[str], scale: float, effect: float
) -> dict[str, Any]:
    predictor = measure.column
    drafted = sub.dropna(subset=["log_pick"])
    out: dict[str, Any] = {"n_draft": len(drafted)}
    flags: list[str] = []
    if len(drafted) >= MIN_N:
        reference = fit_subset(drafted, predictor, outcome, controls)
        adjusted = fit_subset(drafted, predictor, outcome, [*controls, "log_pick"])
        out["effect_with_draft"] = adjusted.beta * scale
        out["draft_shrink"] = shrink(reference.beta, adjusted.beta)
        if out["draft_shrink"] > SHRINK_THRESHOLD and reference.p < SHRINK_REFERENCE_P:
            flags.append("draft pick")
    without_weight = [name for name in controls if name != "combine_weight"]
    if "combine_weight" in controls:
        reference_w = fit_subset(sub, predictor, outcome, without_weight)
        out["effect_without_weight"] = reference_w.beta * scale
        out["weight_shrink"] = shrink(reference_w.beta, effect / scale)
        if out["weight_shrink"] > SHRINK_THRESHOLD and reference_w.p < SHRINK_REFERENCE_P:
            flags.append("weight")
    disc = sub.loc[sub["draft_year"].isin([2023, 2024])]
    latest = sub.loc[sub["draft_year"].eq(2025)]
    effect_2025 = _slope_effect(latest, predictor, outcome, controls, scale)
    out.update(
        n_disc=len(disc),
        effect_disc_2023_24=_slope_effect(disc, predictor, outcome, controls, scale),
        n_2025=len(latest),
        effect_2025=effect_2025,
        holdout_sign_agrees=bool(np.isfinite(effect_2025) and effect_2025 * effect > 0),
        confounded_by="+".join(flags),
        weight_confounded=bool(flags),
    )
    return out


def group_effects(
    table: pd.DataFrame, group: str, lookup: dict[tuple[str, str, str], float], reps: int = BOOTSTRAP_REPS
) -> list[dict[str, Any]]:
    frame = table.loc[table["group"].eq(group)]
    columns = [
        c
        for c in (*OFFICIAL_COLUMNS, *SENSOR_COLUMNS, *route_columns(table))
        if c in frame and frame[c].notna().sum() > 0
    ]
    outcomes = group_outcomes(table, group)
    rows: list[dict[str, Any]] = []
    for column in columns:
        measure = measure_for(column)
        unit = improvement_unit(measure, group, frame[column])
        controls = scope_ngs_agility.controls_for(frame, column)
        rows.extend(
            effect_row(
                frame,
                group=group,
                measure=measure,
                outcome=outcome,
                unit=unit,
                outcome_sign=scope_ngs_agility.outcome_sign(group, outcome),
                controls=controls,
                lookup=lookup,
                reps=reps,
            )
            for outcome in outcomes
        )
    return rows


def add_group_q(effects: pd.DataFrame) -> pd.Series:
    q = pd.Series(np.nan, index=effects.index, dtype=float)
    for _, index in effects.groupby("group").groups.items():
        q.loc[index] = scope_dl.bh_qvalues(effects.loc[index, "p"].to_numpy(dtype=float))
    return q


def _numbers(frame: pd.DataFrame, column: str) -> list[float]:
    return frame[column].astype(float).tolist()


def apply_grades(effects: pd.DataFrame) -> pd.DataFrame:
    graded = effects.assign(q_group=add_group_q(effects))
    columns = ("master_q", "p", "q_group", "ci_lo", "ci_hi")
    values = {name: _numbers(graded, name) for name in columns}
    agrees = graded["holdout_sign_agrees"].astype(bool).tolist()
    statuses = graded["status"].tolist()
    grades = [
        evidence_grade(
            master_q=values["master_q"][i],
            p=values["p"][i],
            holdout_agrees=agrees[i],
            q_group=values["q_group"][i],
            ci_lo=values["ci_lo"][i],
            ci_hi=values["ci_hi"][i],
        )
        if statuses[i] == "ok"
        else "speculative"
        for i in range(len(graded))
    ]
    return graded.assign(grade=grades)


def compute_effects(table: pd.DataFrame, master_path: Path, reps: int = BOOTSTRAP_REPS) -> pd.DataFrame:
    lookup = master_lookup(master_path)
    rows = [row for group in GROUPS for row in group_effects(table, group, lookup, reps)]
    effects = pd.DataFrame(rows).reindex(columns=list(RESULT_COLUMNS))
    return apply_grades(effects)


GRADE_ORDER = {"supported": 2, "suggestive": 1, "speculative": 0}


def decide(rows: pd.DataFrame, rank_eligible: bool) -> str:
    if not rank_eligible:
        return "recruiting_only"
    tested = rows.loc[rows["status"].eq("ok")]
    if tested.empty:
        return "not_tested"
    confirmed = tested.loc[tested["verdict"].eq("supported_benefit") & tested["holdout_sign_agrees"].astype(bool)]
    if not confirmed.empty:
        return "prioritize"
    if tested["verdict"].eq("no_effect_larger_than_X").all():
        return "deprioritize"
    return "unclear"


def headline_row(rows: pd.DataFrame, decision: str) -> pd.Series:
    tested = rows.loc[rows["status"].eq("ok")]
    if tested.empty:
        return rows.iloc[0]
    if decision == "prioritize":
        pool = tested.loc[tested["verdict"].eq("supported_benefit")]
        return pool.iloc[int(np.argmax(pool["effect_sd"].abs().to_numpy()))]
    if decision == "deprioritize":
        bound_sd = tested[["ci_lo_sd", "ci_hi_sd"]].abs().max(axis=1)
        return tested.iloc[int(np.argmax(bound_sd.to_numpy()))]
    return tested.iloc[int(np.argmax(tested["effect_sd"].abs().to_numpy()))]


def rollup(effects: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (group, measure), rows in effects.groupby(["group", "measure"], sort=False):
        eligible = bool(rows["rank_eligible"].iloc[0])
        decision = decide(rows, eligible)
        top = headline_row(rows, decision)
        tested = rows.loc[rows["status"].eq("ok")]
        records.append(
            {
                "group": group,
                "measure": measure,
                "trainability": rows["trainability"].iloc[0],
                "decision": decision,
                "outcomes_tested": len(tested),
                "n_supported_benefit": int(tested["verdict"].eq("supported_benefit").sum()),
                "n_no_effect_larger_than_X": int(tested["verdict"].eq("no_effect_larger_than_X").sum()),
                "n_cannot_say": int(tested["verdict"].eq("cannot_say").sum()),
                "n_other": int(
                    (~tested["verdict"].isin(["supported_benefit", "no_effect_larger_than_X", "cannot_say"])).sum()
                ),
                "headline_outcome": top["outcome"],
                "headline_verdict": top["verdict"],
                "headline_effect": top["effect"],
                "headline_ci_lo": top["ci_lo"],
                "headline_ci_hi": top["ci_hi"],
                "headline_effect_sd": top["effect_sd"],
                "headline_outcome_sd": top["outcome_sd"],
                "headline_mde": top["mde"],
                "headline_n": top["n"],
                "headline_unit": top["unit_label"],
                "headline_unit_value": top["improvement_unit"],
                "headline_grade": top["grade"],
                "best_grade": max(rows["grade"], key=lambda g: GRADE_ORDER[g]),
                "weight_confounded_any": bool(rows["weight_confounded"].astype(bool).any()),
            }
        )
    return pd.DataFrame(records)


WR_BLOCK_MEASURES = (
    "forty",
    "ten_yd_split",
    "forty_yard_dash_top_speed",
    "forty_yard_dash_sensor_ten_yd_burst_speed",
    "forty_yard_dash_burst_speed_1_0s",
    "short_shuttle",
    "three_cone",
)
WR_BLOCK_OUTCOMES = ("separation_at_pass_forward", "target_rate", "rec_yards_per_route")


def unit_description(row: pd.Series) -> str:
    label = str(row["unit_label"])
    value = row["improvement_unit"]
    if label.startswith("0.5 SD"):
        return f"0.5 SD ({_format(value)}) of {row['measure']}"
    if row["measure"] == "combine_weight":
        return f"{_format(value)} lb heavier"
    return f"{label} ({row['measure']})"


def statement(row: pd.Series) -> str:
    if row["status"] != "ok":
        return "not tested: fewer than 30 complete cases"
    unit = unit_description(row)
    interval = f"95% CI {_format(row['ci_lo'])} to {_format(row['ci_hi'])}"
    power = f"smallest effect detectable with 80% power {_format(row['mde'])}, n={int(row['n'])}"
    outcome = row["outcome"]
    if row["verdict"] == "no_effect_larger_than_X":
        bound = row["max_effect_bound"]
        size = f"{_format(bound)} {outcome} ({_format(bound / row['outcome_sd'])} SD)"
        return f"We can rule out an association larger than {size} per {unit}; {interval}; {power}"
    if row["verdict"] == "cannot_say":
        span = f"{_format(row['ci_lo'])} to {_format(row['ci_hi'])}"
        return f"Cannot say: per {unit} the data allow {outcome} changes from {span}; {power}"
    return f"{row['verdict']}: {_format(row['effect'])} {outcome} per {unit}; {interval}; {power}"


def wr_block(effects: pd.DataFrame) -> pd.DataFrame:
    routes = effects.loc[effects["family"].eq("route_drill")]["measure"].unique().tolist()
    selected = effects.loc[
        effects["group"].eq("WR")
        & effects["measure"].isin([*WR_BLOCK_MEASURES, *routes])
        & effects["outcome"].isin(WR_BLOCK_OUTCOMES)
    ]
    return selected.assign(statement=[statement(row) for _, row in selected.iterrows()]).reset_index(drop=True)


def _format(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3g}"


def unit_text(row: pd.Series) -> str:
    label = str(row["headline_unit"])
    value = row["headline_unit_value"]
    if label.startswith("0.5 SD"):
        return f"0.5 SD (= {_format(value)})"
    if row["measure"] == "combine_weight":
        return f"{_format(value)} lb heavier"
    return label


def markdown_row(row: pd.Series) -> str:
    interval = f"{_format(row['headline_ci_lo'])} to {_format(row['headline_ci_hi'])}"
    effect = f"{_format(row['headline_effect'])} ({_format(row['headline_effect_sd'])} SD)"
    if row["decision"] == "deprioritize":
        bound = max(
            abs(row["headline_ci_lo"] / row["headline_outcome_sd"]),
            abs(row["headline_ci_hi"] / row["headline_outcome_sd"]),
        )
        effect = f"rules out > {_format(bound)} SD"
    return (
        f"| {row['measure']} | {unit_text(row)} | {row['headline_outcome']} | {effect} | {interval} | "
        f"{int(row['headline_n'])} | {row['headline_grade']} |"
    )


def group_markdown(summary: pd.DataFrame, group: str) -> str:
    header = (
        "| Measure | Improvement | Outcome shown | Associated change | 95% CI (outcome units) | n | Grade |\n"
        "|---|---|---|---|---|---|---|"
    )
    blocks = []
    for decision in ("prioritize", "deprioritize", "unclear"):
        rows = summary.loc[summary["group"].eq(group) & summary["decision"].eq(decision)]
        if rows.empty:
            continue
        lines = [markdown_row(row) for _, row in rows.iterrows()]
        blocks.append(f"**{decision}** ({len(rows)})\n\n{header}\n" + "\n".join(lines))
    untested = summary.loc[summary["group"].eq(group) & summary["decision"].eq("not_tested"), "measure"].tolist()
    if untested:
        blocks.append(f"**not tested (fewer than 30 complete cases)** ({len(untested)}): " + ", ".join(untested))
    return "\n\n".join(blocks)


def run(root: Path, reps: int = BOOTSTRAP_REPS, output: Path | None = None) -> dict[str, pd.DataFrame]:
    reports = output if output is not None else root / "reports"
    table = load_player_table(root)
    validation = assert_valid(table)
    effects = compute_effects(table, root / "reports/master_results.csv", reps)
    summary = rollup(effects)
    block = wr_block(effects)
    write_trainability(reports / "coaching_trainability.csv")
    effects.to_csv(reports / "coaching_effects.csv", index=False)
    summary.to_csv(reports / "coaching_group_rollup.csv", index=False)
    block.to_csv(reports / "coaching_wr_block.csv", index=False)
    data_dictionary(table).to_csv(reports / "coaching_data_dictionary.csv", index=False)
    validation.to_csv(reports / "coaching_table_validation.csv", index=False)
    return {"effects": effects, "rollup": summary, "wr_block": block, "validation": validation}


def main(arguments: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(arguments)
    run(args.root.resolve())


if __name__ == "__main__":
    main()
