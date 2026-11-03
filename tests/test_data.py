from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from combine_dlol.data import (
    add_line_of_scrimmage,
    iter_game_tracking,
    load_player_play,
    load_players,
    rebuild_processed,
)

RAW_ROOT = Path.cwd() / "data" / "raw"


def _write_fixture(root: Path) -> pd.DataFrame:
    raw = root / "data" / "raw" / "nfl-big-data-bowl-2027"
    raw.mkdir(parents=True)
    players = pd.DataFrame({"nfl_id": [1, 2], "display_name": ["One", "Two"]})
    players.to_csv(root / "data" / "raw" / "players.csv", index=False)
    tracking = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "play_id": [10, 10, 11, 11],
            "nfl_id": [1, 2, 1, 2],
            "frame_id": [1, 1, 1, 1],
            "x": [30.0, 31.0, 80.0, 81.0],
            "y": [20.0, 21.0, 22.0, 23.0],
        }
    )
    tracking.to_csv(raw / "game_tracking_2023.csv", index=False)
    return tracking


def test_streaming_loader_matches_fixture_rows(tmp_path: Path) -> None:
    expected = _write_fixture(tmp_path)
    rebuild_processed(tmp_path, tracking_columns=list(expected.columns))
    streamed = pd.concat(
        list(iter_game_tracking(2023, root=tmp_path, columns=["game_id", "nfl_id", "x"], chunk_size=2))
    )
    actual = streamed.sort_values(["game_id", "nfl_id"]).reset_index(drop=True)
    wanted = expected.loc[:, ["game_id", "nfl_id", "x"]].sort_values(["game_id", "nfl_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, wanted)
    expected_players = pd.DataFrame({"nfl_id": [1, 2], "display_name": ["One", "Two"]})
    pd.testing.assert_frame_equal(load_players(tmp_path), expected_players)


def test_streaming_filter_preserves_requested_columns(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    frames = list(iter_game_tracking(2023, root=tmp_path, columns=["game_id", "x"], player_ids=[1], chunk_size=2))
    result = pd.concat(frames, ignore_index=True)
    assert result.columns.tolist() == ["game_id", "x"]
    assert result["x"].tolist() == [30.0, 80.0]


def test_missing_file_error_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="players.csv"):
        load_players(tmp_path)
    with pytest.raises(FileNotFoundError, match="game_tracking_2024.csv"):
        list(iter_game_tracking(2024, root=tmp_path))


def test_missing_player_play_parquet_is_streamed_and_joined(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"game_id": [1], "home_team_abbr": ["A"]}).to_csv(raw / "games.csv", index=False)
    pd.DataFrame({"nfl_id": [9], "draft_year": [2020], "nfl_position": ["DE"]}).to_csv(
        raw / "players.csv", index=False
    )
    pd.DataFrame({"nfl_id": [9], "draft_year": [2020], "forty": [4.7]}).to_csv(raw / "combine_results.csv", index=False)
    pd.DataFrame({"game_id": [1], "play_id": [2], "nfl_id": [9]}).to_csv(raw / "player_play.csv", index=False)
    actual = load_player_play(["game_id", "nfl_position", "home_team_abbr", "forty"], root=tmp_path)
    expected = pd.DataFrame({"game_id": [1], "nfl_position": ["DE"], "home_team_abbr": ["A"], "forty": [4.7]})
    pd.testing.assert_frame_equal(actual, expected)


def test_line_of_scrimmage_normalizes_both_directions() -> None:
    player_play = pd.DataFrame(
        {
            "game_id": [1, 1],
            "play_id": [10, 11],
            "yardline_side": ["A", "B"],
            "yardline_number": [25, 25],
            "possession_team": ["A", "A"],
            "play_direction": ["right", "left"],
        }
    )
    tracking = pd.DataFrame(
        {
            "game_id": [1, 1],
            "play_id": [10, 11],
            "nfl_id": [99, 99],
            "x": [34.0, 36.0],
            "y": [10.0, 10.0],
        }
    )
    result = add_line_of_scrimmage(tracking, player_play)
    assert result["line_of_scrimmage_x"].tolist() == [35.0, 35.0]
    assert result["x_rel"].tolist() == [-1.0, -1.0]
    assert result["y_rel"].tolist() == [10.0, pytest.approx(53.3333333333 - 10.0)]


@pytest.mark.skipif(not RAW_ROOT.is_dir(), reason="data/raw is not available in this checkout")
def test_real_dl_ball_snap_positions_are_near_the_los() -> None:
    assert not load_players().empty
    fields = [
        "game_id",
        "play_id",
        "nfl_id",
        "nfl_position",
        "yardline_side",
        "yardline_number",
        "possession_team",
        "play_direction",
    ]
    player_play = load_player_play(columns=fields)
    dls = player_play.loc[player_play["nfl_position"].isin(["DE", "DT", "NT", "OLB"])].dropna()
    dls = dls.drop_duplicates(["game_id", "play_id", "nfl_id"])
    player_ids = dls["nfl_id"].drop_duplicates().head(50).tolist()
    frames: list[pd.DataFrame] = []
    for frame in iter_game_tracking(
        2023, columns=["game_id", "play_id", "nfl_id", "x", "y", "event"], player_ids=player_ids
    ):
        snap = frame.loc[frame["event"].eq("ball_snap")]
        if not snap.empty:
            frames.append(snap)
        if sum(len(item) for item in frames) >= 100:
            break
    tracking = pd.concat(frames, ignore_index=True).merge(
        dls[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"], how="inner"
    )
    plays = player_play.merge(tracking[["game_id", "play_id"]].drop_duplicates(), on=["game_id", "play_id"])
    plays = plays.loc[:, ["game_id", "play_id", *fields[4:]]].drop_duplicates()
    normalized = add_line_of_scrimmage(tracking, plays)
    assert len(normalized) >= 25
    assert float(normalized["x_rel"].between(0.0, 3.0).mean()) >= 0.80
