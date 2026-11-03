from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TRACKING_COLUMNS = (
    "game_id",
    "gameId",
    "play_id",
    "playId",
    "nfl_id",
    "nflId",
    "frame_id",
    "frameId",
    "time",
    "club",
    "play_direction",
    "playDirection",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "o",
    "dir",
    "event",
)


def _project_root(root: Path | None) -> Path:
    return Path.cwd() if root is None else root


def _raw_directory(root: Path | None) -> Path:
    return _project_root(root) / "data" / "raw"


def _processed_directory(root: Path | None) -> Path:
    return _project_root(root) / "data" / "processed"


def _raw_file(root: Path | None, name: str) -> Path:
    path = _raw_directory(root) / name
    if not path.is_file():
        raise FileNotFoundError(f"required raw data file is missing: {path}")
    return path


def _read_csv(root: Path | None, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    path = _raw_file(root, name)
    return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)


def load_players(root: Path | None = None) -> pd.DataFrame:
    return _read_csv(root, "players.csv")


def load_combine_results(root: Path | None = None) -> pd.DataFrame:
    return _read_csv(root, "combine_results.csv")


def load_combine_tracking(root: Path | None = None) -> pd.DataFrame:
    return _read_csv(root, "combine_tracking.csv")


def load_games(root: Path | None = None) -> pd.DataFrame:
    return _read_csv(root, "games.csv")


def load_player_career_successes(root: Path | None = None) -> pd.DataFrame:
    return _read_csv(root, "player_career_successes.csv")


def _parquet_has_columns(path: Path, columns: Sequence[str] | None) -> bool:
    if not path.is_file():
        return False
    if columns is None:
        return True
    available = set(pq.ParquetFile(path).schema_arrow.names)
    return set(columns).issubset(available)


def _csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _selected_columns(path: Path, columns: Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None
    available = set(_csv_columns(path))
    selected = [column for column in columns if column in available]
    if not selected:
        raise ValueError(f"none of the requested columns are present in {path}: {list(columns)}")
    return selected


def _stream_csv_to_parquet(source: Path, target: Path, columns: Sequence[str] | None, chunk_size: int) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".incomplete")
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in pd.read_csv(
            source, usecols=_selected_columns(source, columns), chunksize=chunk_size, low_memory=False
        ):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"raw CSV has no rows: {source}")
    temporary.replace(target)
    return target


def _stream_player_play_joined(root: Path | None, target: Path, chunk_size: int) -> Path:
    source = _raw_file(root, "player_play.csv")
    games = load_games(root)
    players = load_players(root)
    combine = load_combine_results(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".incomplete")
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in pd.read_csv(source, chunksize=chunk_size, low_memory=False):
            joined = chunk.merge(games, on="game_id", how="left", validate="many_to_one", suffixes=("", "_game"))
            joined = joined.merge(players, on="nfl_id", how="left", validate="many_to_one", suffixes=("", "_player"))
            if {"nfl_id", "draft_year"}.issubset(combine.columns) and "draft_year" in joined:
                joined = joined.merge(
                    combine,
                    on=["nfl_id", "draft_year"],
                    how="left",
                    validate="many_to_one",
                    suffixes=("", "_combine"),
                )
            table = pa.Table.from_pandas(joined, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"raw CSV has no rows: {source}")
    temporary.replace(target)
    return target


def rebuild_processed(
    root: Path | None = None,
    tracking_columns: Sequence[str] | None = None,
    player_play_columns: Sequence[str] | None = None,
    chunk_size: int = 100_000,
) -> dict[str, tuple[str, ...]]:
    processed = _processed_directory(root)
    schemas: dict[str, tuple[str, ...]] = {}
    player_play_source = _raw_directory(root) / "player_play.csv"
    player_play_target = processed / "player_play_joined.parquet"
    if player_play_source.is_file() and not player_play_target.exists():
        if player_play_columns is None:
            _stream_player_play_joined(root, player_play_target, chunk_size)
        else:
            _stream_csv_to_parquet(player_play_source, player_play_target, player_play_columns, chunk_size)
    if player_play_target.is_file():
        schemas["player_play_joined"] = tuple(pq.ParquetFile(player_play_target).schema_arrow.names)
    for season in (2023, 2024, 2025):
        source = _raw_directory(root) / "nfl-big-data-bowl-2027" / f"game_tracking_{season}.csv"
        target = processed / f"game_tracking_{season}.parquet"
        if source.is_file() and not target.exists():
            _stream_csv_to_parquet(source, target, tracking_columns or TRACKING_COLUMNS, chunk_size)
        if target.is_file():
            schemas[f"game_tracking_{season}"] = tuple(pq.ParquetFile(target).schema_arrow.names)
    return schemas


def load_player_play(columns: Sequence[str] | None = None, root: Path | None = None) -> pd.DataFrame:
    parquet = _processed_directory(root) / "player_play_joined.parquet"
    if _parquet_has_columns(parquet, columns):
        return pd.read_parquet(parquet, columns=list(columns) if columns is not None else None)
    if not parquet.exists():
        rebuild_processed(root)
        if _parquet_has_columns(parquet, columns):
            return pd.read_parquet(parquet, columns=list(columns) if columns is not None else None)
    source = _raw_file(root, "player_play.csv")
    return pd.read_csv(source, usecols=_selected_columns(source, columns), low_memory=False)


def _tracking_parquet(season: int, root: Path | None, columns: Sequence[str] | None, chunk_size: int) -> Path:
    target = _processed_directory(root) / f"game_tracking_{season}.parquet"
    if _parquet_has_columns(target, columns):
        return target
    source = _raw_directory(root) / "nfl-big-data-bowl-2027" / f"game_tracking_{season}.csv"
    if not source.is_file():
        raise FileNotFoundError(f"required tracking data file is missing: {source}")
    if target.exists():
        available = pq.ParquetFile(target).schema_arrow.names
        raise ValueError(
            f"tracking parquet schema does not contain requested columns {list(columns or ())}: {available}"
        )
    return _stream_csv_to_parquet(source, target, columns or TRACKING_COLUMNS, chunk_size)


def iter_game_tracking(
    season: int,
    columns: Sequence[str] | None = None,
    player_ids: Sequence[int] | None = None,
    root: Path | None = None,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    if season not in (2023, 2024, 2025):
        raise ValueError(f"unsupported tracking season: {season}")
    output_columns = list(columns) if columns is not None else None
    requested = None if output_columns is None else list(output_columns)
    if player_ids is not None and requested is not None and "nfl_id" not in requested and "nflId" not in requested:
        target = _processed_directory(root) / f"game_tracking_{season}.parquet"
        if target.is_file():
            available = pq.ParquetFile(target).schema_arrow.names
        else:
            source = _raw_directory(root) / "nfl-big-data-bowl-2027" / f"game_tracking_{season}.csv"
            available = _csv_columns(source) if source.is_file() else []
        identifier = "nfl_id" if "nfl_id" in available else "nflId"
        requested.append(identifier)
    path = _tracking_parquet(season, root, requested, chunk_size)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_size, columns=requested):
        frame = batch.to_pandas()
        if player_ids is not None:
            id_column = _column_name(frame, ("nfl_id", "nflId"))
            frame = frame.loc[frame[id_column].isin(player_ids)].copy()
        if output_columns is not None:
            frame = frame.loc[:, output_columns]
        if not frame.empty:
            yield cast(pd.DataFrame, frame)


def _column_name(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    matches = [candidate for candidate in candidates if candidate in frame.columns]
    if not matches:
        raise ValueError(f"required column missing; expected one of {list(candidates)}, found {list(frame.columns)}")
    return matches[0]


def add_line_of_scrimmage(tracking: pd.DataFrame, player_play: pd.DataFrame) -> pd.DataFrame:
    tracking_game = _column_name(tracking, ("game_id", "gameId"))
    tracking_play = _column_name(tracking, ("play_id", "playId"))
    play_game = _column_name(player_play, ("game_id", "gameId"))
    play_play = _column_name(player_play, ("play_id", "playId"))
    yardline_side = _column_name(player_play, ("yardline_side", "yardlineSide"))
    yardline_number = _column_name(player_play, ("yardline_number", "yardlineNumber"))
    possession = _column_name(player_play, ("possession_team", "possessionTeam"))
    direction = _column_name(player_play, ("play_direction", "playDirection"))
    x_column = _column_name(tracking, ("x",))
    y_column = _column_name(tracking, ("y",))
    fields = [play_game, play_play, yardline_side, yardline_number, possession, direction]
    plays = player_play.loc[:, fields].copy()
    plays = plays.rename(columns={play_game: "_game", play_play: "_play"}).drop_duplicates(["_game", "_play"])
    is_possession_side = plays[yardline_side].eq(plays[possession])
    yardline = plays[yardline_number].astype(float)
    line_when_right = yardline.where(is_possession_side, 100.0 - yardline) + 10.0
    line = line_when_right.where(plays[direction].astype(str).str.lower().eq("right"), 120.0 - line_when_right)
    plays = plays.assign(line_of_scrimmage_x=line)
    merged = tracking.merge(
        plays,
        how="left",
        left_on=[tracking_game, tracking_play],
        right_on=["_game", "_play"],
        validate="many_to_one",
    )
    if merged["line_of_scrimmage_x"].isna().any():
        raise ValueError("line of scrimmage is missing for one or more tracking rows")
    moving_right = merged[direction].astype(str).str.lower().eq("right")
    x_values = merged[x_column].astype(float)
    y_values = merged[y_column].astype(float)
    output = merged.assign(
        x_rel=(x_values - merged["line_of_scrimmage_x"]).where(moving_right, merged["line_of_scrimmage_x"] - x_values),
        y_rel=y_values.where(moving_right, 53.3333333333 - y_values),
    )
    return output.drop(columns=["_game", "_play"])
