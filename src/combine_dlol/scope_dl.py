from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import rankdata, spearmanr, t

from combine_dlol.reliability import (
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
RATE_FORMULA = "{outcome} ~ C(lined_up_position) + C(down) + yards_to_go + C(blitzing) + C(offense_formation)"
STATIC_OUTCOMES = (
    "shrunk_d_a",
    "shrunk_d_b_displacement_0_5",
    "shrunk_d_b_displacement_1_0",
    "shrunk_d_b_speed_0_5",
)
RATE_OUTCOMES = ("hurry_rate", "pressure_rate", "tfl_rate", "rush_win_rate")
COMPOSITES: dict[str, dict[str, int]] = {
    "BURST_INDEX": {
        "forty_yard_dash_peak_acceleration": 1,
        "forty_yard_dash_burst_speed_1_0s": 1,
        "forty_yard_dash_sensor_ten_yd_burst_speed": 1,
        "forty_yard_dash_sensor_ten_yd_time": -1,
    },
    "SPEED_INDEX": {"forty_yard_dash_top_speed": 1, "forty": -1},
    "DRILL_INDEX": {
        "pass_rush_drill_top_speed": 1,
        "pass_rush_drill_peak_acceleration": 1,
        "pass_rush_drill_burst_speed_1_0s": 1,
    },
    "SIZE_INDEX": {"combine_weight": 1, "combine_height": 1},
}
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
    "partial_draft_r",
)


@dataclass(frozen=True)
class PartialResult:
    r: float
    p: float
    ci_lo: float
    ci_hi: float


def role_labels(positions: pd.Series) -> pd.Series:
    return positions.map({"DE": "EDGE", "OLB": "EDGE", "DT": "IDL", "NT": "IDL"})


def join_predictors_outcomes(predictors: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    left = predictors.assign(nfl_id=pd.to_numeric(predictors["nfl_id"], errors="raise").astype(int))
    right = outcomes.assign(nfl_id=pd.to_numeric(outcomes["nfl_id"], errors="raise").astype(int))
    if left["nfl_id"].duplicated().any() or right["nfl_id"].duplicated().any():
        raise ValueError("predictors and outcomes must each have one row per nfl_id")
    return left.merge(right, on="nfl_id", how="inner", validate="one_to_one")


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    standard_deviation = float(numeric.std(ddof=0))
    if not np.isfinite(standard_deviation) or np.isclose(standard_deviation, 0.0):
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - float(numeric.mean())) / standard_deviation


def build_composite(frame: pd.DataFrame, components: dict[str, int]) -> pd.Series:
    values = pd.DataFrame({column: _zscore(frame[column]) * sign for column, sign in components.items()})
    return values.mean(axis=1, skipna=True).where(values.notna().any(axis=1))


def _rank_matrix(size: int, controls: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([np.ones(size), *(rankdata(control) for control in controls)])


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> PartialResult:
    if len(x) != len(y) or len(x) < 5 or any(len(control) != len(x) for control in controls):
        return PartialResult(float("nan"), float("nan"), float("nan"), float("nan"))
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    matrix = _rank_matrix(len(x), controls)
    residual_x = ranked_x - matrix @ np.linalg.lstsq(matrix, ranked_x, rcond=None)[0]
    residual_y = ranked_y - matrix @ np.linalg.lstsq(matrix, ranked_y, rcond=None)[0]
    denominator = float(np.sqrt(np.dot(residual_x, residual_x) * np.dot(residual_y, residual_y)))
    if np.isclose(denominator, 0.0):
        return PartialResult(float("nan"), float("nan"), float("nan"), float("nan"))
    correlation = float(np.dot(residual_x, residual_y) / denominator)
    k = len(controls)
    degrees = len(x) - 2 - k
    if degrees <= 0 or abs(correlation) >= 1.0:
        return PartialResult(correlation, 0.0 if abs(correlation) >= 1.0 else float("nan"), correlation, correlation)
    statistic = correlation * np.sqrt(degrees / (1.0 - correlation**2))
    p_value = float(2.0 * t.sf(abs(statistic), degrees))
    standard_error = 1.0 / np.sqrt(len(x) - 3 - k)
    transformed = np.arctanh(correlation)
    ci_lo, ci_hi = np.tanh([transformed - 1.96 * standard_error, transformed + 1.96 * standard_error])
    return PartialResult(correlation, p_value, float(ci_lo), float(ci_hi))


def _partial_from_frame(frame: pd.DataFrame, predictor: str, outcome: str, controls: list[str]) -> PartialResult:
    columns = list(dict.fromkeys([predictor, outcome, *controls]))
    selected = frame.loc[:, columns].dropna()
    if len(selected) < 5:
        return PartialResult(float("nan"), float("nan"), float("nan"), float("nan"))
    return partial_spearman(
        selected[predictor].to_numpy(dtype=float),
        selected[outcome].to_numpy(dtype=float),
        [selected[column].to_numpy(dtype=float) for column in controls],
    )


def _raw_spearman(frame: pd.DataFrame, predictor: str, outcome: str, minimum: int = 3) -> float:
    selected = frame.loc[:, [predictor, outcome]].dropna()
    if len(selected) < minimum:
        return float("nan")
    return float(spearmanr(selected[predictor], selected[outcome]).statistic)


def association_frame(
    frame: pd.DataFrame, predictor: str, outcome: str, controls: list[str], family: str
) -> dict[str, float | int | str]:
    columns = list(dict.fromkeys([predictor, outcome, *controls, "draft_year", "draft_overall_pick"]))
    selected = frame.loc[:, [column for column in columns if column in frame]].dropna(
        subset=[predictor, outcome, *controls]
    )
    result = _partial_from_frame(selected, predictor, outcome, controls)
    draft_controls = [*controls, "draft_overall_pick"]
    draft_result = _partial_from_frame(
        selected.dropna(subset=["draft_overall_pick"]), predictor, outcome, draft_controls
    )
    return {
        "family": family,
        "n": len(selected),
        "r_raw": _raw_spearman(selected, predictor, outcome),
        "r_adj": result.r,
        "ci_lo": result.ci_lo,
        "ci_hi": result.ci_hi,
        "p": result.p,
        "partial_draft_r": draft_result.r,
        "r_disc_2023_24": _raw_spearman(selected[selected["draft_year"].isin([2023, 2024])], predictor, outcome, 15),
        "r_hold_2025": _raw_spearman(selected[selected["draft_year"].eq(2025)], predictor, outcome, 15),
        "r_2023": _raw_spearman(selected[selected["draft_year"].eq(2023)], predictor, outcome, 15),
        "r_2024": _raw_spearman(selected[selected["draft_year"].eq(2024)], predictor, outcome, 15),
        "r_2025": _raw_spearman(selected[selected["draft_year"].eq(2025)], predictor, outcome, 15),
    }


def _predictor_orientation(predictor: str) -> int:
    timed = {"forty", "ten_yd_split", "three_cone", "short_shuttle"}
    if predictor in timed or predictor.endswith(("_time", "time_in_drill")):
        return -1
    if predictor in {"vertical", "broad_jump", "bench_reps", "BURST_INDEX", "SPEED_INDEX", "DRILL_INDEX"}:
        return 1
    if any(token in predictor for token in ("top_speed", "burst", "acceleration")):
        return 1
    return 0


def _outcome_orientation(outcome: str) -> int:
    if outcome in STATIC_OUTCOMES:
        return -1
    if outcome in RATE_OUTCOMES:
        return 1
    return 0


def orientation_label(predictor: str, outcome: str, correlation: float) -> str:
    sign = _predictor_orientation(predictor) * _outcome_orientation(outcome)
    if sign == 0 or not np.isfinite(correlation):
        return "unclassified"
    return "expected" if correlation * sign > 0.0 else "opposite"


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    qvalues = np.full(len(pvalues), np.nan, dtype=float)
    finite_indexes = np.flatnonzero(np.isfinite(pvalues))
    if not len(finite_indexes):
        return qvalues
    ordered = finite_indexes[np.argsort(pvalues[finite_indexes])]
    previous = 1.0
    count = len(ordered)
    for rank, index in zip(range(count, 0, -1), ordered[::-1], strict=True):
        previous = min(previous, float(pvalues[index]) * count / rank)
        qvalues[index] = previous
    return qvalues


def _interaction_p(frame: pd.DataFrame, predictor: str, outcome: str) -> float:
    selected = frame.loc[:, [predictor, outcome, "combine_weight", "edge"]].dropna()
    if len(selected) < 6 or predictor == "combine_weight":
        return float("nan")
    predictor_rank = rankdata(selected[predictor].to_numpy(dtype=float))
    outcome_rank = rankdata(selected[outcome].to_numpy(dtype=float))
    weight_rank = rankdata(selected["combine_weight"].to_numpy(dtype=float))
    edge = selected["edge"].to_numpy(dtype=float)
    matrix = np.column_stack([np.ones(len(selected)), predictor_rank, edge, weight_rank, predictor_rank * edge])
    coefficients, _, _, _ = np.linalg.lstsq(matrix, outcome_rank, rcond=None)
    degrees = len(selected) - matrix.shape[1]
    if degrees <= 0:
        return float("nan")
    residuals = outcome_rank - matrix @ coefficients
    covariance = float(np.dot(residuals, residuals) / degrees) * np.linalg.pinv(matrix.T @ matrix)
    standard_error = float(np.sqrt(max(covariance[-1, -1], 0.0)))
    if np.isclose(standard_error, 0.0):
        return float("nan")
    return float(2.0 * t.sf(abs(coefficients[-1] / standard_error), degrees))


def _rate_rows(root: Path) -> pd.DataFrame:
    columns = [
        "nfl_id",
        "nfl_position",
        "lined_up_position",
        "down",
        "yards_to_go",
        "blitzing",
        "offense_formation",
        "player_get_off",
        "pass_result",
        "play_nullified_by_penalty",
        "sack",
        "tackle_for_loss",
        "time_to_pressure",
        "time_to_qb_hurry",
    ]
    frame = pd.read_parquet(root / "data/processed/player_play_joined.parquet", columns=columns)
    selected = frame.loc[
        frame["nfl_position"].isin(["DE", "DT", "NT", "OLB"])
        & frame["player_get_off"].notna()
        & frame["pass_result"].notna()
        & frame["play_nullified_by_penalty"].ne("Y")
    ].copy()
    selected["hurry_rate"] = selected["time_to_qb_hurry"].notna().astype(float)
    selected["pressure_rate"] = selected["time_to_pressure"].notna().astype(float)
    selected["tfl_rate"] = selected["tackle_for_loss"].fillna(0).gt(0).astype(float)
    selected["rush_win_rate"] = (selected["time_to_pressure"].notna() | selected["sack"].fillna(0).gt(0)).astype(float)
    return selected


def _rate_residuals(root: Path) -> pd.DataFrame:
    cache = root / "data/processed/scope_dl_rate_residuals.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raw = _rate_rows(root)
    values: list[pd.DataFrame] = []
    required = ["lined_up_position", "down", "yards_to_go", "blitzing", "offense_formation"]
    for outcome in RATE_OUTCOMES:
        usable = raw.dropna(subset=[*required, outcome]).copy()
        residual = smf.ols(RATE_FORMULA.format(outcome=outcome), data=usable).fit().resid
        adjusted = usable.assign(value=residual.to_numpy())
        counts = adjusted.groupby("nfl_id")["value"].transform("size")
        values.append(
            adjusted.loc[counts.ge(50), ["nfl_id", outcome]].assign(
                outcome=outcome, value=adjusted.loc[counts.ge(50), "value"]
            )
        )
    result = pd.concat(values, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache, index=False)
    return result


def _rate_gate_and_scores(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals = _rate_residuals(root)
    progress_path = root / "data/processed/scope_dl_rate_gate_progress.csv"
    progress = pd.read_csv(progress_path) if progress_path.exists() else pd.DataFrame()
    score_path = root / "data/processed/scope_dl_rate_scores_progress.csv"
    scores = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame(columns=["nfl_id", "outcome", "score"])
    raw = _rate_rows(root)
    for outcome in RATE_OUTCOMES:
        if outcome in set(progress.get("outcome", pd.Series(dtype=str))):
            continue
        raw_outcome = raw.loc[:, ["nfl_id", outcome]].dropna()
        eligible = raw_outcome.groupby("nfl_id")[outcome].transform("size").ge(50)
        gate_frame = raw_outcome.loc[eligible]
        split = split_half_reliability(gate_frame, "nfl_id", outcome, min_n=50, n_splits=2000)
        lower, upper = player_bootstrap_ci(split, n_bootstrap=1000)
        adjusted = residuals.loc[residuals["outcome"].eq(outcome), ["nfl_id", "value"]]
        model = mixed_model_icc(adjusted, "nfl_id", "value")
        row = pd.DataFrame(
            [
                {
                    "outcome": outcome,
                    "group": "DL",
                    "n_players": gate_frame["nfl_id"].nunique(),
                    "SB_median": split.median,
                    "lower_ci": lower,
                    "upper_ci": upper,
                    "icc": model.icc,
                    "verdict": verdict(split.median, lower),
                    "formula": RATE_FORMULA.format(outcome=outcome),
                }
            ]
        )
        progress = pd.concat([progress, row], ignore_index=True)
        progress.to_csv(progress_path, index=False)
        if row["verdict"].item() == "adequate":
            shrunk = empirical_bayes_shrink(adjusted, "nfl_id", "value", model, min_n=50)
            score = (
                shrunk.loc[:, ["nfl_id", "posterior_mean"]]
                .rename(columns={"posterior_mean": "score"})
                .assign(outcome=outcome)
            )
            scores = pd.concat([scores, score], ignore_index=True)
            scores.to_csv(score_path, index=False)
    return progress, scores


def _static_gate(root: Path) -> pd.DataFrame:
    reliability = pd.read_csv(root / "reports/phase1_dl_reliability.csv")
    mapping = {
        "d_a": "shrunk_d_a",
        "d_b_displacement_0_5": "shrunk_d_b_displacement_0_5",
        "d_b_displacement_1_0": "shrunk_d_b_displacement_1_0",
        "d_b_speed_0_5": "shrunk_d_b_speed_0_5",
    }
    rows = reliability.loc[reliability["group"].eq("all")].copy()
    rows["outcome"] = rows["outcome"].map(mapping)
    rows["group"] = "DL"
    rows["SB_median"] = rows["split_half_median"]
    rows["lower_ci"] = rows["split_half_ci_lower"]
    rows["upper_ci"] = rows["split_half_ci_upper"]
    rows["formula"] = (
        "outcome ~ C(lined_up_position) + C(down) + yards_to_go + C(offense_formation) + "
        "C(blitzing) + snap_x_rel + snap_y_rel"
    )
    return rows.loc[
        :, ["outcome", "group", "n_players", "SB_median", "lower_ci", "upper_ci", "icc", "verdict", "formula"]
    ]


def run_rates(root: Path) -> pd.DataFrame:
    rate_gate, _ = _rate_gate_and_scores(root)
    gate = pd.concat([_static_gate(root), rate_gate], ignore_index=True)
    gate.to_csv(root / "reports/scope_dl_gate.csv", index=False)
    return gate


def _player_metadata(root: Path) -> pd.DataFrame:
    columns = ["nfl_id", "nfl_position", "draft_overall_pick"]
    frame = pd.read_parquet(root / "data/processed/player_play_joined.parquet", columns=columns)
    positions = (
        frame.dropna(subset=["nfl_position"])
        .groupby(["nfl_id", "nfl_position"], as_index=False)
        .size()
        .sort_values(["nfl_id", "size", "nfl_position"], ascending=[True, False, True])
        .drop_duplicates("nfl_id")
        .loc[:, ["nfl_id", "nfl_position"]]
    )
    picks = frame.groupby("nfl_id", as_index=False)["draft_overall_pick"].first()
    return positions.merge(picks, on="nfl_id", how="left", validate="one_to_one")


def _base_frame(root: Path) -> pd.DataFrame:
    predictors = pd.read_parquet(root / "data/processed/combine_predictors.parquet")
    predictors = predictors.loc[predictors["group"].eq("DL")].copy()
    predictors["nfl_id"] = predictors["nfl_id"].astype(int)
    outcomes = pd.read_parquet(root / "data/processed/dl_outcomes_scores.parquet", columns=["nfl_id", *STATIC_OUTCOMES])
    base = join_predictors_outcomes(predictors, outcomes)
    rate_scores = pd.read_csv(root / "data/processed/scope_dl_rate_scores_progress.csv")
    for outcome, score in rate_scores.groupby("outcome", sort=False):
        base = base.merge(
            score.loc[:, ["nfl_id", "score"]].rename(columns={"score": outcome}),
            on="nfl_id",
            how="left",
            validate="one_to_one",
        )
    metadata = _player_metadata(root)
    base = base.drop(columns=["nfl_position"], errors="ignore").merge(
        metadata, on="nfl_id", how="left", validate="one_to_one"
    )
    base["role"] = role_labels(base["nfl_position"])
    base["edge"] = base["role"].eq("EDGE").astype(float)
    for name, components in COMPOSITES.items():
        base[name] = build_composite(base, components)
    return base


def _predictors(root: Path, base: pd.DataFrame) -> list[tuple[str, str]]:
    metadata = pd.read_csv(root / "reports/phase2_combine_predictors.csv")
    sensors = metadata.loc[
        metadata["group"].eq("DL") & metadata["eligible_for_M4"].astype(bool), "predictor_column"
    ].tolist()
    candidates = [(predictor, "official") for predictor in OFFICIAL] + [(predictor, "sensor") for predictor in sensors]
    candidates.extend((predictor, "composite") for predictor in COMPOSITES)
    return [
        (predictor, family)
        for predictor, family in candidates
        if predictor in base and base[predictor].notna().sum() >= 40
    ]


def _result_rows(root: Path) -> pd.DataFrame:
    base = _base_frame(root)
    gate = pd.read_csv(root / "reports/scope_dl_gate.csv")
    outcomes = gate.loc[gate["verdict"].eq("adequate"), "outcome"].tolist()
    rows: list[dict[str, Any]] = []
    for predictor, family in _predictors(root, base):
        for outcome in outcomes:
            for group, selected, controls in (
                ("pooled", base, ["combine_weight", "edge"]),
                ("EDGE", base.loc[base["role"].eq("EDGE")], ["combine_weight"]),
                ("IDL", base.loc[base["role"].eq("IDL")], ["combine_weight"]),
            ):
                control_columns = [column for column in controls if column != predictor]
                association = association_frame(selected, predictor, outcome, control_columns, family)
                if int(association["n"]) < 40:
                    continue
                correlation = float(association["r_adj"])
                rows.append(
                    {
                        "scope": "dl",
                        "group": group,
                        "outcome": outcome,
                        "predictor": predictor,
                        **association,
                        "oriented_r": correlation * _predictor_orientation(predictor) * _outcome_orientation(outcome)
                        if _predictor_orientation(predictor) and _outcome_orientation(outcome)
                        else float("nan"),
                        "direction": orientation_label(predictor, outcome, correlation),
                    }
                )
            interaction = _interaction_p(
                base.loc[:, [predictor, outcome, "combine_weight", "edge"]].dropna(), predictor, outcome
            )
            rows.append(
                {
                    "scope": "dl",
                    "group": "interaction",
                    "outcome": outcome,
                    "predictor": predictor,
                    "family": family,
                    "n": len(base.loc[:, [predictor, outcome, "combine_weight", "edge"]].dropna()),
                    "r_raw": float("nan"),
                    "r_adj": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "p": interaction,
                    "oriented_r": float("nan"),
                    "direction": "unclassified",
                    "r_disc_2023_24": float("nan"),
                    "r_hold_2025": float("nan"),
                    "r_2023": float("nan"),
                    "r_2024": float("nan"),
                    "r_2025": float("nan"),
                    "partial_draft_r": float("nan"),
                }
            )
    results = pd.DataFrame(rows)
    results["q_scope"] = bh_qvalues(results["p"].to_numpy(dtype=float))
    return pd.DataFrame(results.loc[:, list(RESULT_COLUMNS)])


def _bootstrap_difference(top: np.ndarray, bottom: np.ndarray, seed: int = 20261103) -> tuple[float, float, float]:
    difference = float(top.mean() - bottom.mean())
    rng = np.random.default_rng(seed)
    estimates = np.empty(1000, dtype=float)
    for index in range(1000):
        estimates[index] = float(
            top[rng.integers(0, len(top), len(top))].mean() - bottom[rng.integers(0, len(bottom), len(bottom))].mean()
        )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return difference, float(lower), float(upper)


def _quartiles(root: Path, results: pd.DataFrame) -> pd.DataFrame:
    supported = results.loc[
        results["group"].ne("interaction") & results["family"].eq("composite") & results["q_scope"].lt(0.10),
        "predictor",
    ].unique()
    if not len(supported):
        return pd.DataFrame(
            columns=[
                "predictor",
                "outcome",
                "n_top",
                "n_bottom",
                "difference",
                "ci_lo",
                "ci_hi",
                "d_2023",
                "d_2024",
                "d_2025",
            ]
        )
    base = _base_frame(root)
    rows: list[dict[str, float | int | str]] = []
    for predictor in supported:
        for outcome in [name for name in RATE_OUTCOMES if name in base]:
            selected = base.loc[:, ["role", "draft_year", predictor, outcome]].dropna().copy()
            selected["quartile"] = selected.groupby("role")[predictor].rank(pct=True)
            top = selected.loc[selected["quartile"].ge(0.75), outcome].to_numpy(dtype=float)
            bottom = selected.loc[selected["quartile"].le(0.25), outcome].to_numpy(dtype=float)
            if not len(top) or not len(bottom):
                continue
            difference, lower, upper = _bootstrap_difference(top, bottom)
            class_differences = {
                year: float(
                    selected.loc[(selected["draft_year"].eq(year)) & selected["quartile"].ge(0.75), outcome].mean()
                    - selected.loc[(selected["draft_year"].eq(year)) & selected["quartile"].le(0.25), outcome].mean()
                )
                for year in (2023, 2024, 2025)
            }
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_top": len(top),
                    "n_bottom": len(bottom),
                    "difference": difference,
                    "ci_lo": lower,
                    "ci_hi": upper,
                    "d_2023": class_differences[2023],
                    "d_2024": class_differences[2024],
                    "d_2025": class_differences[2025],
                }
            )
    return pd.DataFrame(rows)


def run_analysis(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = _result_rows(root)
    results.to_csv(root / "reports/scope_dl_results.csv", index=False)
    quartiles = _quartiles(root, results)
    quartiles.to_csv(root / "reports/scope_dl_quartiles.csv", index=False)
    return results, quartiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("rates", "analyze", "all"), default="all")
    parser.add_argument("--root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if arguments.stage in {"rates", "all"}:
        run_rates(root)
    if arguments.stage in {"analyze", "all"}:
        run_analysis(root)


if __name__ == "__main__":
    main()
