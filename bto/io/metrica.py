"""Loader for Metrica Sports sample tracking CSVs (SPEC S5).

Metrica raw tracking CSV layout: 3 header rows then one row per frame.
  row1: team name repeated over the x-column of each player pair
  row2: jersey number, aligned with the x-column of each player pair
  row3: 'Period,Frame,Time [s],Player<N>,,Player<N>,,...,Ball,'
Each player occupies an (x, y) column pair of coordinates normalized to
[0, 1], origin TOP-left, pitch 105x68m. The last (x, y) pair is the ball.
Both team files carry a Ball column with identical values; only the home
file's ball is used.

track_id convention: 'H' + jersey number for home players, 'A' + jersey
number for away players (e.g. 'H11', 'A25').

Coordinates are converted to core.py's meter/bottom-left convention:
  x_m = x_norm * PITCH_LENGTH
  y_m = (1 - y_norm) * PITCH_WIDTH   # flip: source origin is top-left

Rows with NaN coordinates mean the player is off the pitch for that frame
and are dropped from that Frame's player list.
"""

import warnings

import numpy as np
import pandas as pd

from bto.core import PITCH_LENGTH, PITCH_WIDTH, AWAY, HOME, Frame, PlayerPos


def _read_team(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Parse one Metrica team CSV.

    Returns (period, time_s, jersey_track_ids, xy) where xy has shape
    (n_frames, n_players, 2) in normalized [0, 1] coords (NaN if off pitch),
    plus the trailing ball xy is returned separately by the caller via the
    last two data columns.
    """
    header = pd.read_csv(csv_path, header=None, nrows=2)
    jersey_row = header.iloc[1]
    # player columns are all columns except the first 3 (Period, Frame, Time)
    # and the last 2 (Ball x, y); jersey numbers sit on the x-column of each
    # player pair (odd count of NaN cols in between are the y columns).
    n_cols = header.shape[1]
    player_x_cols = [c for c in range(3, n_cols - 2, 2) if pd.notna(jersey_row[c])]
    jerseys = [int(jersey_row[c]) for c in player_x_cols]

    data = pd.read_csv(csv_path, skiprows=3, header=None)
    period = data[0].to_numpy()
    time_s = data[2].to_numpy(dtype=float)

    xy = np.full((len(data), len(player_x_cols), 2), np.nan)
    for i, c in enumerate(player_x_cols):
        xy[:, i, 0] = data[c].to_numpy(dtype=float)
        xy[:, i, 1] = data[c + 1].to_numpy(dtype=float)

    ball_xy = data[[n_cols - 2, n_cols - 1]].to_numpy(dtype=float)
    return period, time_s, jerseys, xy, ball_xy


def _to_meters(xy: np.ndarray) -> np.ndarray:
    out = np.full_like(xy, np.nan)
    out[..., 0] = xy[..., 0] * PITCH_LENGTH
    out[..., 1] = (1.0 - xy[..., 1]) * PITCH_WIDTH
    return out


def _infer_attacking(
    period: np.ndarray, xy_m: np.ndarray, jerseys: list[str], team: str
) -> dict[int, int]:
    """Per-period attacking direction for one team's players.

    GK ~= the player whose mean x (over the first 100 frames of the period)
    is most extreme from the pitch center; whichever half that GK sits in
    is the team's own defensive half, so the team attacks the other end.
    """
    result: dict[int, int] = {}
    for per in np.unique(period):
        idx = np.where(period == per)[0][:100]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(xy_m[idx, :, 0], axis=0)  # per-player mean x (may be all-NaN for a sub)
        if np.all(np.isnan(means)):
            continue
        gk_i = int(np.nanargmax(np.abs(means - PITCH_LENGTH / 2)))
        gk_x = means[gk_i]
        result[int(per)] = 1 if gk_x < PITCH_LENGTH / 2 else -1
    return result


def load_metrica(home_csv: str, away_csv: str, downsample: int = 1) -> list[Frame]:
    h_period, h_time, h_jerseys, h_xy, ball_xy = _read_team(home_csv)
    a_period, a_time, a_jerseys, a_xy, _ = _read_team(away_csv)

    n = min(len(h_period), len(a_period))
    h_period, h_time, h_xy = h_period[:n], h_time[:n], h_xy[:n]
    a_period, a_xy = a_period[:n], a_xy[:n]
    ball_xy = ball_xy[:n]

    h_xy_m = _to_meters(h_xy)
    a_xy_m = _to_meters(a_xy)
    ball_m = _to_meters(ball_xy[:, None, :])[:, 0, :]

    h_attack = _infer_attacking(h_period, h_xy_m, h_jerseys, HOME)
    a_attack = _infer_attacking(a_period, a_xy_m, a_jerseys, AWAY)

    h_track_ids = [f"H{j}" for j in h_jerseys]
    a_track_ids = [f"A{j}" for j in a_jerseys]

    frames: list[Frame] = []
    for i in range(0, n, downsample):
        players: list[PlayerPos] = []
        for j, tid in enumerate(h_track_ids):
            x, y = h_xy_m[i, j]
            if not (np.isnan(x) or np.isnan(y)):
                players.append(PlayerPos(tid, HOME, float(x), float(y)))
        for j, tid in enumerate(a_track_ids):
            x, y = a_xy_m[i, j]
            if not (np.isnan(x) or np.isnan(y)):
                players.append(PlayerPos(tid, AWAY, float(x), float(y)))

        bx, by = ball_m[i]
        ball = None if (np.isnan(bx) or np.isnan(by)) else (float(bx), float(by))

        per = int(h_period[i])
        attacking = {HOME: h_attack.get(per, 1), AWAY: a_attack.get(per, -1)}

        frames.append(Frame(t=float(h_time[i]), players=players, ball=ball, attacking=attacking))

    return frames


if __name__ == "__main__":
    home_csv = "data/metrica/Sample_Game_1_RawTrackingData_Home_Team.csv"
    away_csv = "data/metrica/Sample_Game_1_RawTrackingData_Away_Team.csv"

    frames = load_metrica(home_csv, away_csv, downsample=25)
    assert len(frames) > 100, f"expected >100 frames after downsampling, got {len(frames)}"

    for f in frames:
        for p in f.players:
            # small overshoot beyond the touchline/goal line is real sensor
            # noise in the source data (normalized coords range ~[-0.05, 1.05])
            assert -6.0 <= p.x <= PITCH_LENGTH + 6.0, p
            assert -4.0 <= p.y <= PITCH_WIDTH + 4.0, p
        if f.ball is not None:
            assert -5.0 <= f.ball[0] <= PITCH_LENGTH + 5.0
            assert -5.0 <= f.ball[1] <= PITCH_WIDTH + 5.0
        teams = {p.team for p in f.players}
        assert HOME in teams and AWAY in teams, "both teams must be present"

    # attacking direction must flip between periods for each team
    p1, p2 = frames[0].attacking, frames[-1].attacking
    assert p1[HOME] == -p2[HOME], (p1, p2)
    assert p1[AWAY] == -p2[AWAY], (p1, p2)

    sample = frames[len(frames) // 2]
    print(f"sample frame: t={sample.t:.2f}s attacking={sample.attacking} "
          f"n_players={len(sample.players)} ball={sample.ball}")
    print("OK")
