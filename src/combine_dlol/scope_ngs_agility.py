from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from combine_dlol import scope_db, scope_dl, scope_ol, scope_wrte
from combine_dlol.combine_predictors import (
    BASE_METRICS,
    GROUP_POSITIONS,
    _clean_trajectory,
    _tracking_seconds,
    attempt_kinematics,
    centered_smoothed_speed,
    first_sustained_movement,
    slug,
)

MIN_N = 40
STABLE_MEDIAN = 0.50
STABLE_LOWER = 0.30
NGS_SCORES = ("ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score")
NGS_RESIDUALS = {
    "ngs_athleticism_score": "ngs_athleticism_resid",
    "ngs_college_production_score": "ngs_college_production_resid",
    "ngs_final_score": "ngs_final_resid",
}
PART_A_GROUPS = ("DL", "OL", "DB", "WR", "TE", "WRTE")
GROUP_COMBINE_POSITIONS = {
    "DL": ("DL",),
    "OL": ("OL",),
    "DB": ("DB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "WRTE": ("WR", "TE"),
}
GATE_GROUP = {"DL": "DL", "OL": "OL", "DB": "DB", "WR": "WRTE", "TE": "WRTE", "WRTE": "WRTE"}
INCREMENTAL = (
    ("prod|athl", "ngs_college_production_score", ("ngs_athleticism_score",)),
    ("prod|athl+pick", "ngs_college_production_score", ("ngs_athleticism_score", "log_pick")),
    ("athl|pick", "ngs_athleticism_score", ("log_pick",)),
    ("prod|pick", "ngs_college_production_score", ("log_pick",)),
)
RESULT_COLUMNS = (
    "part",
    "family_label",
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
    "partial_draft_r",
    "status",
    "controls",
)

DRILL_ALIAS = {
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",
    "SHORT_ZONE_BREAK_FLAT_WHEEL": "SHORT_ZONE_BREAKS",
    "SHORT_ZONE_BREAK_FORWARD": "SHORT_ZONE_BREAKS",
    "SHORT_ZONE_BREAK_INSIDE": "SHORT_ZONE_BREAKS",
}
PART_B_DRILLS = {
    "DL": (
        "BODY_CONTROL_DRILL",
        "FOUR_BAG_AGILITY_DRILL",
        "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION",
        "RUN_AND_CLUB_DRILL",
        "RUN_THE_HOOP_DRILL",
        "BACK_PEDAL_AND_REACT",
        "SHORT_ZONE_BREAKS",
    ),
    "OL": (
        "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE",
        "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT",
        "OL_PULL_DRILL_FOLD_BLOCK_RIGHT",
        "OL_PULL_DRILL_LONG_PULL_RIGHT",
        "PASS_PRO_MIRROR_DRILL",
        "SCREEN_DRILL",
    ),
}
EXTRA_METRICS = ("lateral_speed_mean", "lateral_speed_peak", "direction_change_loss")
PART_B_METRICS = (*BASE_METRICS, *EXTRA_METRICS)
METRIC_ORIENTATION = {
    "time_in_drill": -1,
    "top_speed": 1,
    "burst_distance_0_1s": 1,
    "burst_speed_1_0s": 1,
    "peak_acceleration": 1,
    "peak_deceleration": 1,
    "path_length": 0,
    "path_efficiency": 0,
    "lateral_speed_mean": 1,
    "lateral_speed_peak": 1,
    "direction_change_loss": -1,
}
PAIRING_OUTCOME_SIGN = {"lateral_shuffle_0_1_5s": 1, "first_0_5s_displacement": 1, "snap_reaction_time": -1}
_LATERAL_DL = ("lateral_speed_mean", "lateral_speed_peak", "burst_speed_1_0s", "time_in_drill", "peak_acceleration")
_PULLS = (
    "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT",
    "OL_PULL_DRILL_FOLD_BLOCK_RIGHT",
    "OL_PULL_DRILL_LONG_PULL_RIGHT",
)
PAIRINGS: dict[str, tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...]] = {
    "DL": (
        (
            ("FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "RUN_THE_HOOP_DRILL", "FOUR_BAG_AGILITY_DRILL"),
            _LATERAL_DL,
            ("shrunk_d_a", "shrunk_d_b_displacement_0_5", "shrunk_d_b_speed_0_5", "rush_win_rate"),
        ),
        (
            ("BODY_CONTROL_DRILL",),
            ("lateral_speed_mean", "top_speed", "direction_change_loss", "peak_deceleration"),
            ("rush_win_rate", "pressure_rate", "hurry_rate"),
        ),
        (
            ("RUN_AND_CLUB_DRILL",),
            ("top_speed", "peak_acceleration", "burst_speed_1_0s", "time_in_drill"),
            ("tfl_rate", "shrunk_d_b_displacement_1_0"),
        ),
    ),
    "OL": (
        (
            ("FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE", "PASS_PRO_MIRROR_DRILL"),
            ("lateral_speed_mean", "lateral_speed_peak", "time_in_drill", "peak_deceleration", "direction_change_loss"),
            ("lateral_shuffle_0_1_5s", "pass_set_depth_1_0s", "pass_set_depth_1_5s"),
        ),
        (
            _PULLS,
            ("time_in_drill", "burst_distance_0_1s", "burst_speed_1_0s", "peak_acceleration"),
            ("first_0_5s_displacement",),
        ),
        (
            ("SCREEN_DRILL",),
            ("time_in_drill", "top_speed", "peak_acceleration"),
            ("first_0_5s_displacement", "set_peak_speed_0_1_5s"),
        ),
    ),
}


def benjamini_hochberg_by_family(results: pd.DataFrame) -> pd.Series:
    q = pd.Series(np.nan, index=results.index, dtype=float)
    for _, index in results.groupby("family_label").groups.items():
        q.loc[index] = scope_dl.bh_qvalues(results.loc[index, "p"].to_numpy(dtype=float))
    return q


def predictor_sign(predictor: str) -> int:
    if predictor in NGS_SCORES or predictor in NGS_RESIDUALS.values() or predictor == "AGILITY_INDEX":
        return 1
    for metric, sign in sorted(METRIC_ORIENTATION.items(), key=lambda item: -len(item[0])):
        if predictor.endswith(f"_{metric}"):
            return sign
    return 0


def outcome_sign(group: str, outcome: str) -> int:
    key = GATE_GROUP.get(group, group)
    if key == "DL":
        return int(scope_dl._outcome_orientation(outcome))
    if key == "OL":
        return int(scope_ol.OUTCOME_ORIENTATION.get(outcome, 0))
    if key == "DB":
        return int(scope_db._outcome_orientation(outcome))
    return 1


def _read_or_build(path: Path, builder: Callable[[], object], reader: Callable[[Path], pd.DataFrame]) -> pd.DataFrame:
    if not path.is_file():
        builder()
    return reader(path)


def gate_table(root: Path) -> pd.DataFrame:
    sources = (
        ("scope_dl_gate.csv", "DL"),
        ("scope_ol_gate.csv", "OL"),
        ("scope_db_gate.csv", "DB"),
        ("scope_wrte_gate.csv", "WRTE"),
    )
    frames = []
    for name, group in sources:
        gate = pd.read_csv(root / "reports" / name)
        frames.append(gate.assign(group=group, source=name))
    table = pd.concat(frames, ignore_index=True)
    table = table.loc[table["verdict"].ne("career"), ["outcome", "group", "source", "SB_median", "lower_ci", "verdict"]]
    table["stable"] = (
        table["verdict"].eq("adequate") & table["SB_median"].ge(STABLE_MEDIAN) & table["lower_ci"].ge(STABLE_LOWER)
    )
    return table.reset_index(drop=True)


def stable_outcomes(gate: pd.DataFrame, group: str) -> list[str]:
    key = GATE_GROUP.get(group, group)
    return gate.loc[gate["group"].eq(key) & gate["stable"], "outcome"].tolist()


def _rate_scores(root: Path) -> pd.DataFrame:
    path = root / "data/processed/scope_dl_rate_scores_progress.csv"
    scores = _read_or_build(path, lambda: scope_dl.run_rates(root), pd.read_csv)
    return scores.pivot(index="nfl_id", columns="outcome", values="score").reset_index()


def outcome_frame(root: Path, key: str, outcomes: Sequence[str]) -> pd.DataFrame:
    processed = root / "data/processed"
    if key == "DL":
        static = [name for name in outcomes if name in scope_dl.STATIC_OUTCOMES]
        frame = pd.read_parquet(processed / "dl_outcomes_scores.parquet", columns=["nfl_id", *static])
        rates = _rate_scores(root)
        frame = frame.merge(
            rates.loc[:, ["nfl_id", *[name for name in outcomes if name in rates]]], on="nfl_id", how="outer"
        )
        return frame.astype({"nfl_id": int})
    if key == "DB":
        analysis = _read_or_build(
            processed / "scope_db_analysis.parquet", lambda: scope_db.run_scope(root), pd.read_parquet
        )
        return analysis.loc[:, ["nfl_id", *outcomes]].astype({"nfl_id": int})
    frame = pd.DataFrame({"nfl_id": pd.Series(dtype=int)})
    for outcome in outcomes:
        if key == "OL":
            path = processed / f"scope_ol_scores_{outcome}.parquet"
            score = _read_or_build(path, partial(scope_ol.gate_outcome, root, outcome), pd.read_parquet)
            score = score.loc[:, ["nfl_id", "posterior_mean"]].rename(columns={"posterior_mean": outcome})
        else:
            path = processed / f"scope_wrte_scores_{outcome}.parquet"
            score = _read_or_build(path, partial(scope_wrte.stage_outcome, root, outcome), pd.read_parquet)
            score = score.loc[:, ["nfl_id", outcome]]
        score = score.assign(nfl_id=pd.to_numeric(score["nfl_id"]).astype(int))
        frame = score if frame.empty else frame.merge(score, on="nfl_id", how="outer", validate="one_to_one")
    return frame


def role_columns(group: str, frame: pd.DataFrame) -> pd.DataFrame:
    position = frame["nfl_position"].astype(str)
    if group == "DL":
        role = scope_dl.role_labels(frame["nfl_position"]).fillna("IDL")
        control = role.eq("EDGE")
    elif group == "OL":
        control = position.eq("T")
        role = control.map({True: "T", False: "IOL"})
    elif group == "DB":
        control = position.eq("CB")
        role = control.map({True: "CB", False: "S"})
    else:
        control = frame["combine_position"].eq("WR")
        role = frame["combine_position"].astype(str)
    result = frame.assign(role=role.to_numpy(), role_ctl=control.astype(float).to_numpy())
    if result["role_ctl"].nunique() < 2:
        result["role_ctl"] = np.nan
    return result


def add_draft_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_pick"] = np.log(pd.to_numeric(out["draft_overall_pick"], errors="coerce"))
    for score, name in NGS_RESIDUALS.items():
        out[name] = np.nan
        valid = out.loc[:, [score, "log_pick"]].dropna()
        if len(valid) < 10 or valid["log_pick"].nunique() < 2:
            continue
        slope, intercept = np.polyfit(valid["log_pick"], valid[score], 1)
        out.loc[valid.index, name] = valid[score] - (intercept + slope * valid["log_pick"])
    return out


def build_group_frame(combine: pd.DataFrame, players: pd.DataFrame, outcomes: pd.DataFrame, group: str) -> pd.DataFrame:
    selected = combine.loc[combine["combine_position"].isin(GROUP_COMBINE_POSITIONS[group])]
    meta = players.loc[:, ["nfl_id", "nfl_position", "draft_overall_pick"]]
    frame = selected.merge(meta, on="nfl_id", how="left", validate="one_to_one")
    frame = role_columns(group, frame)
    frame = add_draft_residuals(frame)
    return frame.merge(outcomes, on="nfl_id", how="left", validate="one_to_one")


def controls_for(frame: pd.DataFrame, predictor: str) -> list[str]:
    candidates = ["combine_weight"]
    if frame["role_ctl"].notna().any():
        candidates.append("role_ctl")
    return [name for name in candidates if name != predictor]


def _empty_row(**fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {name: np.nan for name in RESULT_COLUMNS}
    row.update(scope="ngs_agility", q_scope=np.nan, direction="unclassified", status="ok")
    row.update(fields)
    return row


def association_row(
    frame: pd.DataFrame,
    *,
    predictor: str,
    outcome: str,
    controls: Sequence[str],
    extras: Sequence[str] = (),
    meta: dict[str, Any],
    predictor_orientation: int,
    outcome_orientation: int,
) -> dict[str, Any]:
    needed = list(dict.fromkeys([predictor, outcome, *controls, *extras]))
    sub = frame.loc[:, list(dict.fromkeys([*needed, "draft_year", "log_pick"]))].dropna(subset=needed)
    row = _empty_row(**meta, predictor=predictor, outcome=outcome, n=len(sub), controls="+".join([*controls, *extras]))
    if len(sub) < MIN_N:
        row["status"] = "underpowered_n<40"
        return row
    if sub[predictor].nunique() < 5 or sub[outcome].nunique() < 5:
        row["status"] = "degenerate"
        return row
    statistic = scope_dl._partial_from_frame(sub, predictor, outcome, [*controls, *extras])
    row.update(
        r_raw=scope_dl._raw_spearman(sub, predictor, outcome),
        r_adj=statistic.r,
        ci_lo=statistic.ci_lo,
        ci_hi=statistic.ci_hi,
        p=statistic.p,
    )
    if "log_pick" not in extras:
        drafted = sub.dropna(subset=["log_pick"])
        if len(drafted) >= MIN_N:
            row["partial_draft_r"] = scope_dl._partial_from_frame(
                drafted, predictor, outcome, [*controls, *extras, "log_pick"]
            ).r
    for label, years in (
        ("r_disc_2023_24", (2023, 2024)),
        ("r_hold_2025", (2025,)),
        ("r_2023", (2023,)),
        ("r_2024", (2024,)),
        ("r_2025", (2025,)),
    ):
        row[label] = scope_dl._raw_spearman(sub.loc[sub["draft_year"].isin(years)], predictor, outcome, 15)
    sign = predictor_orientation * outcome_orientation
    if sign and np.isfinite(statistic.r):
        row["oriented_r"] = statistic.r * sign
        row["direction"] = "expected" if statistic.r * sign > 0 else "opposite"
    return row


def part_a_rows(frame: pd.DataFrame, group: str, outcomes: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictors = [*NGS_SCORES, *NGS_RESIDUALS.values()]
    for outcome in outcomes:
        out_sign = outcome_sign(group, outcome)
        for predictor in predictors:
            if predictor not in frame:
                continue
            family = "ngs_resid" if predictor in NGS_RESIDUALS.values() else "ngs_score"
            rows.append(
                association_row(
                    frame,
                    predictor=predictor,
                    outcome=outcome,
                    controls=controls_for(frame, predictor),
                    meta={
                        "part": "A",
                        "family_label": f"A_main_{group}",
                        "group": group,
                        "family": family,
                    },
                    predictor_orientation=1,
                    outcome_orientation=out_sign,
                )
            )
        for label, predictor, extras in INCREMENTAL:
            rows.append(_incremental_row(frame, group, predictor, label, extras, outcome, out_sign))
    return rows


def _incremental_row(
    frame: pd.DataFrame,
    group: str,
    predictor: str,
    label: str,
    extras: Sequence[str],
    outcome: str,
    out_sign: int,
) -> dict[str, Any]:
    working = frame.assign(**{f"{predictor}|{label.split('|')[1]}": frame[predictor]})
    return association_row(
        working,
        predictor=f"{predictor}|{label.split('|')[1]}",
        outcome=outcome,
        controls=controls_for(frame, predictor),
        extras=extras,
        meta={"part": "A", "family_label": f"A_incremental_{group}", "group": group, "family": "incremental"},
        predictor_orientation=1,
        outcome_orientation=out_sign,
    )


def tier_table(frame: pd.DataFrame, group: str, predictors: Sequence[str], outcomes: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for outcome in outcomes:
            sub = frame.loc[:, ["role", "draft_year", predictor, outcome]].dropna()
            if sub.empty:
                continue
            percentile = sub.groupby("role")[predictor].rank(pct=True)
            top = sub.loc[percentile.ge(2 / 3)]
            bottom = sub.loc[percentile.le(1 / 3)]
            if len(top) < 5 or len(bottom) < 5:
                continue
            difference, lower, upper = scope_dl._bootstrap_difference(
                top[outcome].to_numpy(dtype=float), bottom[outcome].to_numpy(dtype=float)
            )
            row: dict[str, Any] = {
                "group": group,
                "predictor": predictor,
                "outcome": outcome,
                "n_top": len(top),
                "n_bottom": len(bottom),
                "top_mean": float(top[outcome].mean()),
                "bottom_mean": float(bottom[outcome].mean()),
                "difference": difference,
                "ci_lo": lower,
                "ci_hi": upper,
            }
            for year in (2023, 2024, 2025):
                row[f"d_{year}"] = float(
                    top.loc[top["draft_year"].eq(year), outcome].mean()
                    - bottom.loc[bottom["draft_year"].eq(year), outcome].mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def extra_kinematics(
    time: Sequence[float] | np.ndarray[Any, Any],
    x: Sequence[float] | np.ndarray[Any, Any],
    y: Sequence[float] | np.ndarray[Any, Any],
) -> dict[str, float]:
    clean_time, clean_x, clean_y = _clean_trajectory(time, x, y)
    speed = centered_smoothed_speed(clean_time, clean_x, clean_y)
    if len(speed) < 3:
        return {}
    start = first_sustained_movement(speed)
    t, px, py, speed = clean_time[start:], clean_x[start:], clean_y[start:], speed[start:]
    if len(speed) < 5:
        return {}
    vx, vy = np.gradient(px, t), np.gradient(py, t)
    path = np.column_stack([px - px.mean(), py - py.mean()])
    axis = np.linalg.svd(path, full_matrices=False)[2][0]
    perpendicular = vx * -axis[1] + vy * axis[0]
    smoothed = np.abs(np.convolve(np.pad(perpendicular, 1, mode="edge"), np.array([0.25, 0.5, 0.25]), mode="valid"))
    heading = np.arctan2(vy, vx)
    turn_rate = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)
    step_speed = speed[1:]
    eligible = step_speed >= 1.0
    loss = float("nan")
    if int(eligible.sum()) >= 5:
        cutoff = np.quantile(turn_rate[eligible], 0.8)
        turning = eligible & (turn_rate >= cutoff)
        loss = float(1.0 - step_speed[turning].mean() / float(np.max(speed)))
    return {
        "lateral_speed_mean": float(smoothed.mean()),
        "lateral_speed_peak": float(smoothed.max()),
        "direction_change_loss": loss,
    }


def drill_attempts(tracking: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    wanted = {name for names in PART_B_DRILLS.values() for name in names} | set(DRILL_ALIAS)
    group_of = {position: group for group, positions in GROUP_POSITIONS.items() for position in positions}
    owner = players.assign(group=players["nfl_position"].map(group_of)).set_index("nfl_id")["group"]
    selected = tracking.loc[tracking["entity_type"].eq("PLAYER") & tracking["drill_name"].isin(wanted)].copy()
    selected["group"] = selected["nfl_id"].map(owner)
    selected["drill"] = selected["drill_name"].replace(DRILL_ALIAS)
    allowed = [
        drill in PART_B_DRILLS.get(group, ()) for group, drill in zip(selected["group"], selected["drill"], strict=True)
    ]
    selected = selected.loc[allowed].copy()
    selected["seconds"] = _tracking_seconds(selected["time"])
    selected = selected.dropna(subset=["seconds", "x", "y", "attempt"])
    rows: list[dict[str, Any]] = []
    for (nfl_id, group, drill, name, attempt), values in selected.groupby(
        ["nfl_id", "group", "drill", "drill_name", "attempt"], sort=True
    ):
        t = values["seconds"].to_numpy(dtype=float)
        x = values["x"].to_numpy(dtype=float)
        y = values["y"].to_numpy(dtype=float)
        base = attempt_kinematics(t, x, y, str(drill))
        if not base:
            continue
        rows.append(
            {"nfl_id": nfl_id, "group": group, "drill": drill, "source_drill": name, "attempt": attempt}
            | base
            | extra_kinematics(t, x, y)
        )
    return pd.DataFrame(rows, columns=["nfl_id", "group", "drill", "source_drill", "attempt", *PART_B_METRICS])


def player_drill_table(attempts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.DataFrame({"nfl_id": sorted(attempts["nfl_id"].unique())})
    meta_rows: list[dict[str, Any]] = []
    for (group, drill), values in attempts.groupby(["group", "drill"], sort=True):
        counts = values.groupby("nfl_id").size()
        for metric in PART_B_METRICS:
            column = f"{slug(str(drill))}_{metric}"
            mean = values.groupby("nfl_id")[metric].mean().rename(column).reset_index()
            wide = wide.merge(mean, on="nfl_id", how="left", validate="one_to_one")
            meta_rows.append(
                {
                    "group": group,
                    "drill": drill,
                    "metric": metric,
                    "column": column,
                    "n_players": int(counts.size),
                    "share_single_attempt": float((counts == 1).mean()),
                }
            )
    return wide, pd.DataFrame(meta_rows)


def drill_gate_table(meta: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    summary = meta.groupby(["group", "drill"], as_index=False).agg(
        n_players=("n_players", "first"), share_single_attempt=("share_single_attempt", "first")
    )
    summary["repeatability_status"] = np.where(
        summary["share_single_attempt"].ge(0.5), "unverifiable_single_attempt", "multi_attempt_not_gated"
    )
    summary["eligible_n>=40"] = summary["n_players"].ge(MIN_N)
    summary["exploratory"] = True
    return summary


def agility_index(table: pd.DataFrame, meta: pd.DataFrame, group: str) -> pd.Series:
    parts = []
    eligible = meta.loc[meta["group"].eq(group) & meta["n_players"].ge(MIN_N)]
    for drill in eligible["drill"].unique():
        for metric, sign in (("lateral_speed_mean", 1), ("time_in_drill", -1)):
            column = f"{slug(drill)}_{metric}"
            if column in table:
                parts.append(sign * scope_dl._zscore(table[column]))
    if not parts:
        return pd.Series(np.nan, index=table.index, dtype=float)
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1).where(stacked.notna().sum(axis=1).ge(4))


def pairing_triples(group: str) -> set[tuple[str, str, str]]:
    return {
        (f"{slug(drill)}_{metric}", metric, outcome)
        for drills, metrics, outcomes in PAIRINGS.get(group, ())
        for drill in drills
        for metric in metrics
        for outcome in outcomes
    }


def part_b_rows(frame: pd.DataFrame, meta: pd.DataFrame, group: str, outcomes: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = meta.loc[meta["group"].eq(group), "column"].tolist()
    pairs = {(column, outcome) for column, _, outcome in pairing_triples(group)}
    for outcome in outcomes:
        base_sign = outcome_sign(group, outcome)
        for column in columns:
            if column not in frame:
                continue
            controls = controls_for(frame, column)
            sign = predictor_sign(column)
            common = {"predictor": column, "outcome": outcome, "controls": controls, "extras": ()}
            common_meta = {"part": "B", "group": group, "family": "drill_metric"}
            rows.append(
                association_row(
                    frame,
                    **common,
                    meta={**common_meta, "family_label": f"B_allbyall_{group}"},
                    predictor_orientation=sign,
                    outcome_orientation=base_sign,
                )
            )
            if (column, outcome) in pairs:
                pairing_sign = base_sign or PAIRING_OUTCOME_SIGN.get(outcome, 0)
                rows.append(
                    association_row(
                        frame,
                        **common,
                        meta={**common_meta, "family_label": f"B_pairing_{group}"},
                        predictor_orientation=sign,
                        outcome_orientation=pairing_sign,
                    )
                )
        if "AGILITY_INDEX" in frame:
            rows.append(
                association_row(
                    frame,
                    predictor="AGILITY_INDEX",
                    outcome=outcome,
                    controls=controls_for(frame, "AGILITY_INDEX"),
                    meta={"part": "B", "group": group, "family": "agility_index", "family_label": f"B_index_{group}"},
                    predictor_orientation=1,
                    outcome_orientation=base_sign,
                )
            )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    tested = results.loc[results["status"].eq("ok")]
    rows: list[dict[str, Any]] = []
    for label, values in tested.groupby("family_label", sort=True):
        significant = values.loc[values["q_scope"].lt(0.10)]
        classified = values.loc[values["direction"].isin(["expected", "opposite"])]
        nominal = values.loc[values["p"].lt(0.05)]
        rows.append(
            {
                "family_label": label,
                "n_tests": len(values),
                "n_q_lt_0.10": len(significant),
                "n_p_lt_0.05": len(nominal),
                "expected_by_chance_p05": 0.05 * len(values),
                "n_expected": int(classified["direction"].eq("expected").sum()),
                "n_opposite": int(classified["direction"].eq("opposite").sum()),
                "n_nominal_expected": int(nominal["direction"].eq("expected").sum()),
                "n_nominal_opposite": int(nominal["direction"].eq("opposite").sum()),
                "holdout_same_sign_nominal": int((np.sign(nominal["r_hold_2025"]) == np.sign(nominal["r_adj"])).sum()),
                "holdout_defined_nominal": int(nominal["r_hold_2025"].notna().sum()),
                "max_abs_r": float(values["r_adj"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _scatter(frame: pd.DataFrame, group: str, outcomes: Sequence[str]) -> pd.DataFrame:
    columns = [
        "nfl_id",
        "draft_year",
        "nfl_position",
        "role",
        "draft_overall_pick",
        "combine_weight",
        *NGS_SCORES,
        *NGS_RESIDUALS.values(),
        *[name for name in outcomes if name in frame],
    ]
    return frame.loc[:, [column for column in columns if column in frame]].assign(group=group)


def run(root: Path) -> pd.DataFrame:
    raw = root / "data/raw"
    reports = root / "reports"
    combine = pd.read_csv(raw / "combine_results.csv")
    players = pd.read_csv(raw / "players.csv")
    gate = gate_table(root)
    gate.to_csv(reports / "scope_ngs_agility_gate.csv", index=False)
    rows: list[dict[str, Any]] = []
    scatter: list[pd.DataFrame] = []
    tiers: list[pd.DataFrame] = []
    frames: dict[str, pd.DataFrame] = {}
    for group in PART_A_GROUPS:
        outcomes = stable_outcomes(gate, group)
        outcome_scores = outcome_frame(root, GATE_GROUP[group], outcomes)
        frame = build_group_frame(combine, players, outcome_scores, group)
        frames[group] = frame
        outcomes = [name for name in outcomes if name in frame]
        rows.extend(part_a_rows(frame, group, outcomes))
        scatter.append(_scatter(frame, group, outcomes))
        tiers.append(tier_table(frame, group, [*NGS_SCORES, *NGS_RESIDUALS.values()], outcomes))
    attempts = drill_attempts(pd.read_csv(raw / "combine_tracking.csv"), players)
    table, meta = player_drill_table(attempts)
    drill_gate_table(meta, table).to_csv(reports / "scope_ngs_agility_drill_gate.csv", index=False)
    for group in ("DL", "OL"):
        frame = frames[group].merge(table, on="nfl_id", how="left", validate="one_to_one")
        index = agility_index(frame, meta, group)
        frame = frame.copy().assign(AGILITY_INDEX=index)
        rows.extend(part_b_rows(frame, meta, group, stable_outcomes(gate, group)))
    results = pd.DataFrame(rows).reindex(columns=list(RESULT_COLUMNS))
    results["q_scope"] = benjamini_hochberg_by_family(results)
    results.to_csv(reports / "scope_ngs_agility_results.csv", index=False)
    headline = results.loc[results["family_label"].str.startswith("A_main") & results["predictor"].isin(NGS_SCORES)]
    headline.to_csv(reports / "scope_ngs_agility_headline.csv", index=False)
    pd.concat(scatter, ignore_index=True).to_csv(reports / "scope_ngs_agility_scatter.csv", index=False)
    pd.concat(tiers, ignore_index=True).to_csv(reports / "scope_ngs_agility_tiers.csv", index=False)
    summarize(results).to_csv(reports / "scope_ngs_agility_summary.csv", index=False)
    return results


def main(arguments: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(arguments)
    run(args.root.resolve())


if __name__ == "__main__":
    main()
