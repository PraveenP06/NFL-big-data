from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261111
BOOTSTRAP_REPS = 1000
MIN_SNAPS = 150
MIN_ATLAS_N = 100
MIN_HALF_N_CONFIRMATORY = 30
MIN_HALF_N_ATLAS = 50
SHRINK_RATIO = 0.5
CAREER_YEARS = (2000, 2021)
EARLY_YEARS = (2000, 2012)
LATE_YEARS = (2013, 2021)
RATE_YEARS = (2018, 2024)
RATE_EARLY = (2018, 2020)
RATE_LATE = (2021, 2024)

OL_POS = frozenset({"OT", "OG", "C", "OL", "T", "G"})
DL_POS = frozenset({"DE", "DT", "NT", "DL"})
EDGE_DL_POS = DL_POS | {"OLB"}
DB_POS = frozenset({"CB", "S", "DB", "FS", "SS"})
LB_POS = frozenset({"ILB", "LB", "MLB"})
GROUPS: dict[str, frozenset[str]] = {
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "RB": frozenset({"RB"}),
    "QB": frozenset({"QB"}),
    "OL": OL_POS,
    "DL": DL_POS,
    "LB": LB_POS,
    "CB": frozenset({"CB"}),
    "S": frozenset({"S", "FS", "SS"}),
}
ROLE = {"DE": "EDGE", "OLB": "EDGE", "DT": "IDL", "NT": "IDL", "DL": "DL"}
MEASURES = ("forty", "vertical", "broad_jump", "cone", "shuttle", "bench", "wt", "ht_in")
OUTCOMES = ("car_av", "w_av", "games", "seasons_started", "probowls", "allpro")
UNITS = {
    "forty": 0.1,
    "cone": 0.1,
    "shuttle": 0.1,
    "vertical": 2.0,
    "broad_jump": 4.0,
    "bench": 3.0,
    "wt": 10.0,
    "ht_in": 1.0,
}
UNIT_LABEL = {
    "forty": "per 0.1 s",
    "cone": "per 0.1 s",
    "shuttle": "per 0.1 s",
    "vertical": "per 2 in",
    "broad_jump": "per 4 in",
    "bench": "per 3 reps",
    "wt": "per 10 lb",
    "ht_in": "per 1 in",
}
EXPECTED_SIGN = {"forty": -1, "cone": -1, "shuttle": -1, "vertical": 1, "broad_jump": 1, "bench": 1}


@dataclass(frozen=True)
class Hypothesis:
    id: str
    description: str
    positions: frozenset[str]
    predictor: str
    outcome: str
    source: str
    expected_sign: int
    dummy: str
    tracking_ref: str
    tracking_value: float
    primary: bool = True


HYPOTHESES = (
    Hypothesis(
        "E1",
        "WR/TE heavier -> lower rec yds/target",
        frozenset({"WR", "TE"}),
        "wt",
        "yds_tgt",
        "rec",
        -1,
        "pos",
        "WR weight-separation r -0.40 to -0.45 (tracking, different outcome)",
        -0.425,
    ),
    Hypothesis(
        "E2",
        "WR vertical -> higher rec yds/target",
        frozenset({"WR"}),
        "vertical",
        "yds_tgt",
        "rec",
        1,
        "pos",
        "WR vertical -> separation slope +0.022 per 2 in",
        0.022,
    ),
    Hypothesis(
        "E3",
        "OL faster forty -> more seasons started",
        OL_POS,
        "forty",
        "seasons_started",
        "career",
        -1,
        "pos",
        "OL forty -> games started r -0.33",
        -0.33,
    ),
    Hypothesis(
        "E4",
        "OL faster shuttle -> more seasons started",
        OL_POS,
        "shuttle",
        "seasons_started",
        "career",
        -1,
        "pos",
        "OL shuttle r -0.42",
        -0.42,
    ),
    Hypothesis(
        "E5",
        "DL/edge broad jump -> higher pressure rate",
        EDGE_DL_POS,
        "broad_jump",
        "prss_rate",
        "def",
        1,
        "role",
        "DL broad jump/get-off r -0.24 (hold-out -0.47); get-off time lower is better",
        -0.24,
    ),
    Hypothesis(
        "E6",
        "DL/edge heavier -> lower pressure rate within role",
        EDGE_DL_POS,
        "wt",
        "prss_rate",
        "def",
        -1,
        "role",
        "DL weight -> slower get-off (direction only)",
        float("nan"),
    ),
    Hypothesis(
        "E7",
        "DB broad jump -> more seasons started",
        DB_POS,
        "broad_jump",
        "seasons_started",
        "career",
        1,
        "pos",
        "DB broad jump -> games started r +0.24 (hold-out +0.36)",
        0.24,
    ),
    Hypothesis(
        "E3b",
        "OL faster forty -> more games (supplementary)",
        OL_POS,
        "forty",
        "games",
        "career",
        -1,
        "pos",
        "OL forty -> games started r -0.33",
        -0.33,
        primary=False,
    ),
)


def parse_height(value: object) -> float:
    match = re.fullmatch(r"\s*(\d+)-(\d+)\s*", str(value))
    if match is None:
        return float("nan")
    return float(int(match.group(1)) * 12 + int(match.group(2)))


def normalize_name(name: object) -> str:
    text = re.sub(r"[^a-z ]", "", str(name).lower())
    tokens = [t for t in text.split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(tokens)


def prepare_combine(combine: pd.DataFrame) -> pd.DataFrame:
    out = combine.copy()
    out["ht_in"] = out["ht"].map(parse_height)
    return out


def build_player_table(combine: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    left = prepare_combine(combine).dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="first")
    right = draft.dropna(subset=["pfr_player_id"]).drop_duplicates("pfr_player_id", keep="first")
    keep = ["pfr_player_id", "season", "pick", "pfr_player_name", "position", *OUTCOMES]
    merged = left.merge(right[keep], left_on="pfr_id", right_on="pfr_player_id", how="inner", suffixes=("_c", ""))
    merged = merged.rename(columns={"season": "draft_season"})
    merged["log_pick"] = np.log(merged["pick"].astype(float))
    merged["allpro"] = (merged["allpro"].astype(float) > 0).astype(float)
    return merged.reset_index(drop=True)


def join_report(combine: pd.DataFrame, draft: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    has_id = combine["pfr_id"].notna()
    drafted = combine["draft_ovr"].notna()
    unmatched = combine[has_id & ~combine["pfr_id"].isin(draft["pfr_player_id"].dropna())]
    names_equal = table["player_name"].map(normalize_name) == table["pfr_player_name"].map(normalize_name)
    pick_equal = table["draft_ovr"].astype(float) == table["pick"].astype(float)
    year_equal = table["draft_year"].astype(float) == table["draft_season"].astype(float)
    pos_equal = table["pos"] == table["position"]
    in_scope = table["draft_season"].between(*CAREER_YEARS)
    mapped = table["pos"].isin(set().union(*GROUPS.values()) | EDGE_DL_POS | DB_POS)
    rows = [
        ("combine_rows", len(combine), ""),
        ("combine_rows_with_pfr_id", int(has_id.sum()), "missing id rows cannot be joined"),
        ("combine_rows_missing_pfr_id", int((~has_id).sum()), "mostly undrafted; NaN keys dropped before merge"),
        ("combine_rows_marked_drafted", int(drafted.sum()), "draft_ovr non-null"),
        ("combine_duplicate_pfr_ids", int(combine["pfr_id"].dropna().duplicated().sum()), "first row kept"),
        ("draft_duplicate_pfr_ids", int(draft["pfr_player_id"].dropna().duplicated().sum()), "first row kept"),
        ("joined_players", len(table), "combine x draft on pfr_id"),
        ("join_rate_of_ids", len(table) / max(int(has_id.sum()), 1), "joined / combine rows with id"),
        ("join_rate_of_drafted", len(table) / max(int(drafted.sum()), 1), "joined / combine rows with draft_ovr"),
        ("combine_ids_not_in_draft", len(unmatched), ";".join(unmatched["player_name"].head(10).astype(str))),
        ("name_mismatch", int((~names_equal).sum()), "normalized combine vs draft name"),
        ("pick_mismatch", int((~pick_equal).sum()), "combine draft_ovr vs draft pick; draft pick used"),
        ("draft_year_mismatch", int((~year_equal).sum()), "combine draft_year vs draft season; draft season used"),
        ("position_mismatch", int((~pos_equal).sum()), "combine pos vs draft position; combine pos used"),
        ("in_career_cohort_2000_2021", int(in_scope.sum()), "draft years 2000-2021"),
        ("in_cohort_unmapped_position", int((in_scope & ~mapped).sum()), "positions outside frozen mapping"),
    ]
    unmapped = table.loc[in_scope & ~mapped, "pos"].value_counts()
    rows.extend((f"unmapped_pos_{pos}", int(count), "excluded") for pos, count in unmapped.items())
    return pd.DataFrame(rows, columns=["metric", "value", "detail"])


def season_snaps(snaps: pd.DataFrame, column: str) -> pd.DataFrame:
    regular = snaps[snaps["game_type"] == "REG"]
    grouped = regular.groupby(["pfr_player_id", "season"]).agg(snaps=(column, "sum")).reset_index()
    return grouped.rename(columns={"pfr_player_id": "pfr_id"})


def collapse_teams(adv: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    single = adv[~adv["tm"].astype(str).str.fullmatch(r"\d+TM")]
    return single.groupby(["pfr_id", "season"], as_index=False)[list(columns)].sum()


def season_rates(adv: pd.DataFrame, snaps: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "rec":
        base = collapse_teams(adv, ["yds", "tgt"])
        merged = base.merge(season_snaps(snaps, "offense_snaps"), on=["pfr_id", "season"], how="inner")
        merged = merged[(merged["snaps"] >= MIN_SNAPS) & (merged["tgt"] > 0)]
        return merged.assign(rate=merged["yds"] / merged["tgt"])[["pfr_id", "season", "rate"]]
    base = collapse_teams(adv, ["prss"])
    merged = base.merge(season_snaps(snaps, "defense_snaps"), on=["pfr_id", "season"], how="inner")
    merged = merged[merged["snaps"] >= MIN_SNAPS]
    return merged.assign(rate=100.0 * merged["prss"] / merged["snaps"])[["pfr_id", "season", "rate"]]


def average_rates(seasons: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    inside = seasons[seasons["season"].between(*years)]
    return inside.groupby("pfr_id").agg(rate=("rate", "mean")).reset_index()


def dummies(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    coded = pd.get_dummies(frame[column], prefix="d", drop_first=True, dtype=float)
    return coded.loc[:, coded.nunique() > 1]


def regress_residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), controls]) if controls.size else np.ones((len(values), 1))
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def fisher_ci(r: float, n: int, k: int) -> tuple[float, float]:
    se = 1.0 / np.sqrt(max(n - 3 - k, 1))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.tanh(z - 1.959964 * se)), float(np.tanh(z + 1.959964 * se))


def control_matrix(frame: pd.DataFrame, use_pick: bool, use_weight: bool, ranked: bool) -> np.ndarray:
    parts: list[np.ndarray] = []
    if use_pick:
        col = frame["log_pick"]
        parts.append((col.rank() if ranked else col).to_numpy(float)[:, None])
    if use_weight:
        col = frame["wt"]
        parts.append((col.rank() if ranked else col).to_numpy(float)[:, None])
    coded = dummies(frame, "grp")
    if coded.shape[1]:
        parts.append(coded.to_numpy(float))
    return np.hstack(parts) if parts else np.empty((len(frame), 0))


def complete_cases(frame: pd.DataFrame, use_weight: bool) -> pd.DataFrame:
    needed = ["y", "x", "log_pick"] + (["wt"] if use_weight else [])
    return frame.dropna(subset=needed).reset_index(drop=True)


def partial_spearman(frame: pd.DataFrame, use_pick: bool = True, use_weight: bool = True) -> dict[str, float]:
    data = complete_cases(frame, use_weight)
    n = len(data)
    empty = {"n": float(n), "r": np.nan, "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    controls = control_matrix(data, use_pick, use_weight, ranked=True) if n else np.empty((0, 0))
    k = controls.shape[1]
    if n - 3 - k < 1:
        return empty
    ry = regress_residual(data["y"].rank().to_numpy(float), controls)
    rx = regress_residual(data["x"].rank().to_numpy(float), controls)
    if ry.std() < 1e-9 or rx.std() < 1e-9:
        return empty
    r = float(np.corrcoef(ry, rx)[0, 1])
    df = n - 2 - k
    t = r * np.sqrt(df / max(1 - r * r, 1e-12))
    lo, hi = fisher_ci(r, n, k)
    return {"n": float(n), "r": r, "p": float(2 * stats.t.sf(abs(t), df)), "ci_lo": lo, "ci_hi": hi}


def ols_slope_matrix(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), x, controls])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])


def slope_with_bootstrap(
    frame: pd.DataFrame, unit: float, use_weight: bool, reps: int = BOOTSTRAP_REPS, seed: int = SEED
) -> dict[str, float]:
    data = complete_cases(frame, use_weight)
    if len(data) < 10:
        return {"slope": np.nan, "slope_lo": np.nan, "slope_hi": np.nan}
    controls = control_matrix(data, True, use_weight, ranked=False)
    y = data["y"].to_numpy(float)
    x = data["x"].to_numpy(float)
    point = ols_slope_matrix(y, x, controls)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [ols_slope_matrix(y[i], x[i], controls[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(reps))]
    )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"slope": point * unit, "slope_lo": float(lo) * unit, "slope_hi": float(hi) * unit}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order):
        running = min(running, p[idx] * m / (m - rank))
        adjusted[idx] = running
    return adjusted.tolist()


def sign_of(value: float) -> int:
    return 0 if np.isnan(value) or value == 0 else int(np.sign(value))


def half_result(frame: pd.DataFrame, mask: pd.Series, use_weight: bool, min_n: int) -> dict[str, float]:
    sub = frame[mask]
    result = partial_spearman(sub, use_weight=use_weight)
    return result if result["n"] >= min_n else {**result, "r": np.nan}


def outcome_frames(
    h: Hypothesis, table: pd.DataFrame, seasons: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if h.source == "career":
        base = table[table["draft_season"].between(*CAREER_YEARS)]
        base = base.assign(y=base[h.outcome].astype(float))
        in_early = base["draft_season"].between(*EARLY_YEARS)
        in_late = base["draft_season"].between(*LATE_YEARS)
        return base, base[in_early], base[in_late]
    full, early, late = (rate_frame(table, seasons[h.source], years) for years in (RATE_YEARS, RATE_EARLY, RATE_LATE))
    return full, early, late


def rate_frame(table: pd.DataFrame, season_df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    rates = average_rates(season_df, years).rename(columns={"rate": "y"})
    return table.merge(rates, on="pfr_id")


def prepare_frame(frame: pd.DataFrame, h: Hypothesis) -> pd.DataFrame:
    out = frame[frame["pos"].isin(h.positions)].copy()
    out["x"] = out[h.predictor]
    out["grp"] = out["pos"].map(ROLE) if h.dummy == "role" else out["pos"]
    return out


def evaluate_hypothesis(h: Hypothesis, table: pd.DataFrame, seasons: dict[str, pd.DataFrame]) -> dict[str, object]:
    full, early, late = (prepare_frame(f, h) for f in outcome_frames(h, table, seasons))
    use_weight = h.predictor != "wt"
    main = partial_spearman(full, use_weight=use_weight)
    no_pick = partial_spearman(full, use_pick=False, use_weight=use_weight)
    slope = slope_with_bootstrap(full, UNITS[h.predictor], use_weight)
    first = partial_spearman(early, use_weight=use_weight)
    second = partial_spearman(late, use_weight=use_weight)
    first_r = first["r"] if first["n"] >= MIN_HALF_N_CONFIRMATORY else np.nan
    second_r = second["r"] if second["n"] >= MIN_HALF_N_CONFIRMATORY else np.nan
    return {
        "hypothesis": h.id,
        "description": h.description,
        "primary": h.primary,
        "predictor": h.predictor,
        "outcome": h.outcome,
        "expected_sign": h.expected_sign,
        "n": int(main["n"]),
        "r": main["r"],
        "ci_lo": main["ci_lo"],
        "ci_hi": main["ci_hi"],
        "p": main["p"],
        "sign_matches": sign_of(main["r"]) == h.expected_sign,
        "r_no_pick_control": no_pick["r"],
        "unit": UNIT_LABEL[h.predictor],
        "slope_per_unit": slope["slope"],
        "slope_ci_lo": slope["slope_lo"],
        "slope_ci_hi": slope["slope_hi"],
        "early_n": int(first["n"]),
        "early_r": first_r,
        "late_n": int(second["n"]),
        "late_r": second_r,
        "replicates": sign_of(first_r) == h.expected_sign and sign_of(second_r) == h.expected_sign,
        "tracking_ref": h.tracking_ref,
        "tracking_value": h.tracking_value,
    }


def run_confirmatory(table: pd.DataFrame, seasons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = pd.DataFrame([evaluate_hypothesis(h, table, seasons) for h in HYPOTHESES])
    primary = result["primary"].to_numpy(bool)
    result["holm_p"] = np.nan
    result.loc[primary, "holm_p"] = holm_adjust(result.loc[primary, "p"].fillna(1.0).tolist())
    result["confirmed"] = result["sign_matches"] & (result["holm_p"] < 0.05)
    return result


def shrinkage_flag(r_adjusted: float, r_unadjusted: float) -> bool:
    if np.isnan(r_adjusted) or np.isnan(r_unadjusted):
        return False
    flipped = sign_of(r_adjusted) != sign_of(r_unadjusted)
    return flipped or abs(r_adjusted) < SHRINK_RATIO * abs(r_unadjusted)


def atlas_cell(
    group: str, measure: str, outcome: str, frame: pd.DataFrame, reps: int = BOOTSTRAP_REPS
) -> dict[str, object] | None:
    use_weight = measure != "wt"
    cell = frame.assign(x=frame[measure], y=frame[outcome].astype(float), grp=frame["pos"])
    main = partial_spearman(cell, use_weight=use_weight)
    if main["n"] < MIN_ATLAS_N or np.isnan(main["r"]):
        return None
    no_pick = partial_spearman(cell, use_pick=False, use_weight=use_weight)
    early = half_result(cell, cell["draft_season"].between(*EARLY_YEARS), use_weight, MIN_HALF_N_ATLAS)
    late = half_result(cell, cell["draft_season"].between(*LATE_YEARS), use_weight, MIN_HALF_N_ATLAS)
    slope = slope_with_bootstrap(cell, UNITS[measure], use_weight, reps=reps)
    expected = EXPECTED_SIGN.get(measure, 0)
    return {
        "group": group,
        "measure": measure,
        "outcome": outcome,
        "n": int(main["n"]),
        "r": main["r"],
        "ci_lo": main["ci_lo"],
        "ci_hi": main["ci_hi"],
        "p": main["p"],
        "unit": UNIT_LABEL[measure],
        "slope_per_unit": slope["slope"],
        "slope_ci_lo": slope["slope_lo"],
        "slope_ci_hi": slope["slope_hi"],
        "r_no_pick_control": no_pick["r"],
        "pick_shrinkage_flag": shrinkage_flag(main["r"], no_pick["r"]),
        "early_r": early["r"],
        "late_r": late["r"],
        "replicates": sign_of(early["r"]) != 0 and sign_of(early["r"]) == sign_of(late["r"]) == sign_of(main["r"]),
        "expected_sign": expected,
        "direction": direction_label(main["r"], expected),
    }


def direction_label(r: float, expected: int) -> str:
    if expected == 0:
        return "no_prior"
    return "expected" if sign_of(r) == expected else "opposite"


def run_atlas(table: pd.DataFrame, reps: int = BOOTSTRAP_REPS) -> pd.DataFrame:
    cohort = table[table["draft_season"].between(*CAREER_YEARS)]
    rows = [
        atlas_cell(group, measure, outcome, cohort[cohort["pos"].isin(positions)], reps)
        for group, positions in GROUPS.items()
        for measure in MEASURES
        for outcome in OUTCOMES
    ]
    result = pd.DataFrame([r for r in rows if r is not None])
    if result.empty:
        return result
    result["q_bh"] = bh_adjust(result["p"].tolist())
    result["significant"] = result["q_bh"] < 0.05
    result["rank_in_group"] = result.groupby("group")["p"].rank(method="first").astype(int)
    return result.sort_values(["group", "rank_in_group"]).reset_index(drop=True)


def atlas_summary(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return atlas
    directional = atlas[atlas["expected_sign"] != 0]
    keys = ["all", "significant"]
    rows = []
    for key in keys:
        subset = directional if key == "all" else directional[directional["significant"]]
        rows.append(
            {
                "family": key,
                "expected": int((subset["direction"] == "expected").sum()),
                "opposite": int((subset["direction"] == "opposite").sum()),
            }
        )
    return pd.DataFrame(rows)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    ext = root / "data" / "external"
    combine = pd.read_csv(ext / "combine_combine.csv")
    draft = pd.read_csv(ext / "draft_picks_draft_picks.csv")
    snaps = pd.concat([pd.read_csv(p) for p in sorted(ext.glob("snap_20*.csv"))], ignore_index=True)
    rec = pd.read_csv(ext / "pfr_advstats_advstats_season_rec.csv")
    defense = pd.read_csv(ext / "pfr_advstats_advstats_season_def.csv")
    seasons = {"rec": season_rates(rec, snaps, "rec"), "def": season_rates(defense, snaps, "def")}
    return combine, draft, seasons


def rate_match_rows(table: pd.DataFrame, seasons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [(f"qualified_{name}_player_seasons", len(df), f"snaps>={MIN_SNAPS}") for name, df in seasons.items()]
    for name, df in seasons.items():
        matched = table["pfr_id"].isin(df["pfr_id"]).sum()
        rows.append((f"joined_players_with_{name}_rate", int(matched), "of joined combine-draft players"))
    return pd.DataFrame(rows, columns=["metric", "value", "detail"])


def run(root: Path) -> None:
    combine, draft, seasons = load_inputs(root)
    table = build_player_table(combine, draft)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    pd.concat([join_report(combine, draft, table), rate_match_rows(table, seasons)]).to_csv(
        reports / "external_join_report.csv", index=False
    )
    run_confirmatory(table, seasons).to_csv(reports / "external_confirmatory.csv", index=False)
    atlas = run_atlas(table)
    atlas.to_csv(reports / "external_atlas.csv", index=False)
    print(atlas_summary(atlas).to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="External large-sample validation of combine leads.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args(argv).root)


if __name__ == "__main__":
    main()
