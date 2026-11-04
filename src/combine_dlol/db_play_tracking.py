from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DB_POSITIONS = frozenset({"CB", "FS", "SS", "DB"})
RECEIVER_POSITIONS = frozenset({"WR", "TE", "RB", "FB"})
SEASONS = (2023, 2024, 2025)
FRAME_TOLERANCE = 0.11


def wrap_degrees(delta: np.ndarray) -> np.ndarray:
    return np.abs((delta + 180.0) % 360.0 - 180.0)


def seconds(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


# ---------------------------------------------------------------- tracking-based game outcomes


@dataclass(frozen=True)
class Track:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    o: np.ndarray
    d: np.ndarray


def line_of_scrimmage(side: str, number: float, possession: str, direction: str) -> tuple[float, float]:
    own_side = side == possession
    if direction == "right":
        return (10.0 + number if own_side else 110.0 - number), 1.0
    return (110.0 - number if own_side else 10.0 + number), -1.0


def _index(track: Track, when: float) -> int | None:
    if not np.isfinite(when) or len(track.t) == 0:
        return None
    index = int(np.argmin(np.abs(track.t - when)))
    return index if abs(track.t[index] - when) <= FRAME_TOLERANCE else None


def _value_at(track: Track, values: np.ndarray, when: float) -> float:
    index = _index(track, when)
    return float("nan") if index is None else float(values[index])


def _window(track: Track, start: float, end: float) -> np.ndarray:
    return np.flatnonzero((track.t >= start - 1e-9) & (track.t <= end + 1e-9))


def snap_reaction_time(track: Track, snap: float) -> float:
    start = _index(track, snap)
    if start is None or track.s[start] >= 0.75 or track.t[-1] < snap + 0.5:
        return float("nan")
    displacement = np.hypot(track.x[start:] - track.x[start], track.y[start:] - track.y[start])
    moving = (displacement >= 0.25) | (track.s[start:] >= 0.75)
    elapsed = track.t[start:] - snap
    hits = np.flatnonzero(moving & (elapsed <= 3.0))
    if len(hits):
        return float(elapsed[hits[0]])
    return 3.0 if track.t[-1] >= snap + 3.0 else float("nan")


def speed_1s(track: Track, snap: float, forward: float) -> float:
    if not np.isfinite(forward) or forward - snap < 1.5:
        return float("nan")
    return _value_at(track, track.s, snap + 1.0)


def backpedal_share(track: Track, snap: float, forward: float) -> float:
    if not np.isfinite(forward):
        return float("nan")
    index = _window(track, snap, min(forward, snap + 3.0))
    if len(index) < 10:
        return float("nan")
    facing_away = wrap_degrees(track.o[index] - track.d[index]) >= 135.0
    return float(np.mean((track.s[index] >= 1.5) & facing_away))


def speed_gain_after_throw(track: Track, forward: float, arrival: float) -> float:
    if not np.isfinite(forward):
        return float("nan")
    stop = forward + 1.0 if not np.isfinite(arrival) else min(forward + 1.0, arrival)
    index = _window(track, forward, stop)
    if stop - forward < 0.5 or len(index) < 3:
        return float("nan")
    return float(track.s[index].max() - track.s[index[0]])


def _distance(first: Track, second: Track, when: float) -> float:
    i, j = _index(first, when), _index(second, when)
    if i is None or j is None:
        return float("nan")
    return float(np.hypot(first.x[i] - second.x[j], first.y[i] - second.y[j]))


def _nearest(db: Track, receivers: dict[int, Track], when: float) -> tuple[int | None, float]:
    best: tuple[int | None, float] = (None, float("nan"))
    for identifier, receiver in receivers.items():
        distance = _distance(db, receiver, when)
        if np.isfinite(distance) and (best[0] is None or distance < best[1]):
            best = (identifier, distance)
    return best


def _closing(db: Track, receiver: Track, forward: float, arrival: float) -> tuple[float, float]:
    at_forward = _distance(db, receiver, forward)
    at_arrival = _distance(db, receiver, arrival)
    flight = arrival - forward
    if not (np.isfinite(at_forward) and np.isfinite(at_arrival)) or at_forward > 10.0 or flight < 0.5:
        return float("nan"), float("nan")
    return at_arrival, (at_forward - at_arrival) / flight


def receiver_metrics(
    db: Track, receivers: dict[int, Track], target_id: int | None, forward: float, arrival: float
) -> dict[str, float]:
    result = dict.fromkeys(
        (
            "nearest_receiver_dist_pf",
            "nearest_receiver_dist_arrival",
            "closing_speed_nearest",
            "dist_to_target_arrival",
            "closing_speed_target",
        ),
        float("nan"),
    )
    if not np.isfinite(forward) or not receivers:
        return result
    nearest_id, result["nearest_receiver_dist_pf"] = _nearest(db, receivers, forward)
    if not np.isfinite(arrival):
        return result
    result["nearest_receiver_dist_arrival"] = _nearest(db, receivers, arrival)[1]
    if nearest_id is not None:
        result["closing_speed_nearest"] = _closing(db, receivers[nearest_id], forward, arrival)[1]
    if target_id is not None and target_id in receivers:
        result["dist_to_target_arrival"], result["closing_speed_target"] = _closing(
            db, receivers[target_id], forward, arrival
        )
    return result


def _turn_window(track: Track, snap: float, arrival: float) -> np.ndarray:
    if not np.isfinite(arrival):
        return np.array([], dtype=int)
    return _window(track, snap, min(arrival, snap + 5.0))


def cod_speed_retention(track: Track, snap: float, arrival: float) -> float:
    index = _turn_window(track, snap, arrival)
    if len(index) < 8:
        return float("nan")
    time, speed, heading = track.t[index], track.s[index], track.d[index]
    end = np.searchsorted(time, time + 0.5 - 1e-9)
    candidates = np.flatnonzero((end < len(time)) & (speed >= 2.0))
    if len(candidates) == 0:
        return float("nan")
    change = wrap_degrees(heading[end[candidates]] - heading[candidates])
    best = int(np.argmax(change))
    if change[best] < 60.0:
        return float("nan")
    first = int(candidates[best])
    return float(np.clip(speed[end[first]] / speed[first], 0.0, 2.0))


def peak_decel_in_play(track: Track, snap: float, arrival: float) -> float:
    index = _turn_window(track, snap, arrival)
    if len(index) < 8:
        return float("nan")
    time, speed = track.t[index], track.s[index]
    rate = (speed[3:] - speed[:-3]) / (time[3:] - time[:-3])
    return float(max(0.0, -rate.min()))


def peak_speed_in_play(track: Track, snap: float, arrival: float) -> float:
    index = _turn_window(track, snap, arrival)
    return float(track.s[index].max()) if len(index) >= 8 else float("nan")


def run_pursuit_peak_speed(track: Track, snap: float, stop: float) -> float:
    if not np.isfinite(stop) or stop - snap < 2.0:
        return float("nan")
    index = _window(track, snap, min(stop, snap + 8.0))
    return float(track.s[index].max()) if len(index) else float("nan")


def tackle_depth(track: Track, tackle_time: float, los_x: float, sign: float) -> float:
    x = _value_at(track, track.x, tackle_time)
    return float((x - los_x) * sign) if np.isfinite(x) else float("nan")


@dataclass(frozen=True)
class PlayContext:
    events: dict[str, float]
    los_x: float
    sign: float
    is_pass: bool
    target_id: int | None
    credited_tackle: bool


def db_play_metrics(db: Track, receivers: dict[int, Track], context: PlayContext) -> dict[str, float]:
    nan = float("nan")
    snap = context.events.get("ball_snap", nan)
    forward = context.events.get("pass_forward", nan) if context.is_pass else nan
    arrival = context.events.get("pass_arrived", nan) if context.is_pass else nan
    stops = [context.events[name] for name in ("tackle", "out_of_bounds", "touchdown") if name in context.events]
    metrics: dict[str, float] = {
        "snap_reaction_time": snap_reaction_time(db, snap),
        "speed_1s": speed_1s(db, snap, forward),
        "backpedal_share": backpedal_share(db, snap, forward),
        "speed_at_pass_forward": _value_at(db, db.s, forward),
        "speed_gain_after_throw": speed_gain_after_throw(db, forward, arrival),
        "cod_speed_retention": cod_speed_retention(db, snap, arrival),
        "peak_decel_in_play": peak_decel_in_play(db, snap, arrival),
        "peak_speed_in_play": peak_speed_in_play(db, snap, arrival),
        "run_pursuit_peak_speed": nan
        if context.is_pass or not np.isfinite(snap)
        else run_pursuit_peak_speed(db, snap, min(stops) if stops else nan),
        "tackle_depth": tackle_depth(db, context.events.get("tackle", nan), context.los_x, context.sign)
        if context.credited_tackle
        else nan,
    }
    metrics.update(receiver_metrics(db, receivers, context.target_id, forward, arrival))
    return metrics


def _tracks(group: pd.DataFrame) -> dict[int, Track]:
    result: dict[int, Track] = {}
    for identifier, values in group.groupby("nfl_id", sort=False):
        ordered = values.sort_values("seconds").drop_duplicates("seconds")
        result[int(str(identifier))] = Track(
            *(ordered[c].to_numpy(dtype=float) for c in ("seconds", "x", "y", "s", "o", "dir"))
        )
    return result


def _play_rows(frame: pd.DataFrame) -> Iterable[dict[str, object]]:
    for (game_id, play_id), group in frame.groupby(["game_id", "play_id"], sort=False):
        first = group.iloc[0]
        events = {
            str(name): float(value)
            for name, value in group.dropna(subset=["event"]).groupby("event")["seconds"].min().items()
        }
        los_x, sign = line_of_scrimmage(
            str(first["yardline_side"]),
            float(first["yardline_number"]),
            str(first["possession_team"]),
            str(first["play_direction"]),
        )
        tracks = _tracks(group)
        offense = group.loc[group["role"].eq("rec")]
        targets = offense.loc[offense["target"].fillna(False).astype(bool), "nfl_id"]
        target_id = int(targets.iloc[0]) if len(targets) else None
        receivers = {identifier: tracks[identifier] for identifier in offense["nfl_id"].astype(int).unique()}
        is_pass = bool(pd.notna(first["pass_result"]))
        for _, row in group.loc[group["role"].eq("db")].drop_duplicates("nfl_id").iterrows():
            identifier = int(row["nfl_id"])
            credited = bool(pd.notna(row["tackle"]) and float(row["tackle"]) > 0.0)
            context = PlayContext(events, los_x, sign, is_pass, target_id, credited)
            yield {
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": identifier,
                **db_play_metrics(tracks[identifier], receivers, context),
            }


def tag_roles(frame: pd.DataFrame) -> pd.DataFrame:
    defense = frame["team_abbr"].ne(frame["possession_team"])
    role = np.where(
        frame["nfl_position"].isin(DB_POSITIONS) & defense,
        "db",
        np.where(frame["nfl_position"].isin(RECEIVER_POSITIONS) & ~defense, "rec", "other"),
    )
    return frame.assign(role=role)


def build_tracking_features(root: Path, player_play: pd.DataFrame) -> pd.DataFrame:
    meta_columns = [
        "game_id",
        "play_id",
        "nfl_id",
        "nfl_position",
        "team_abbr",
        "possession_team",
        "yardline_side",
        "yardline_number",
        "play_direction",
        "pass_result",
        "target",
        "tackle",
        "season",
    ]
    meta = tag_roles(player_play.loc[player_play["play_nullified_by_penalty"].ne("Y"), meta_columns])
    meta = meta.loc[meta["role"].ne("other")]
    outputs: list[pd.DataFrame] = []
    for season in SEASONS:
        path = root / "data" / "processed" / f"game_tracking_{season}.parquet"
        season_meta = meta.loc[meta["season"].eq(season)]
        identifiers = sorted(int(v) for v in season_meta["nfl_id"].unique())
        columns = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "o", "dir", "event"]
        tracking = pd.read_parquet(path, columns=columns, filters=[("nfl_id", "in", identifiers)])
        keyed = tracking.merge(
            season_meta.drop(columns=["season"]), on=["game_id", "play_id", "nfl_id"], how="inner", validate="m:1"
        )
        keyed["seconds"] = seconds(keyed["time"])
        keyed = keyed.dropna(subset=["seconds", "x", "y", "s", "o", "dir"])
        db_plays = keyed.loc[keyed["role"].eq("db"), ["game_id", "play_id"]].drop_duplicates()
        keyed = keyed.merge(db_plays, on=["game_id", "play_id"], how="inner")
        outputs.append(pd.DataFrame(list(_play_rows(keyed))))
    return pd.concat(outputs, ignore_index=True)
