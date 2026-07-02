"""Shared synthetic-frame builders for the M1 pattern-engine test suite.

Every helper here builds `list[Frame]` (bto.core) directly -- no CV, no
loaders -- so pattern-engine tests can script unambiguous scenarios well
inside detector thresholds.
"""

from math import hypot

import pytest

from bto.core import AWAY, HOME, Frame, PlayerPos

DT = 0.04  # 25 Hz, matches Metrica sample rate
ATTACK_RL = {HOME: 1, AWAY: -1}  # home attacks +x, away attacks -x


def make_frame(t: float, players: list[PlayerPos], ball, attacking=None) -> Frame:
    """Build one Frame; defaults to the standard home(+1)/away(-1) attacking split."""
    return Frame(t=t, players=players, ball=ball, attacking=attacking or dict(ATTACK_RL))


@pytest.fixture
def make_frame_fn():
    return make_frame


def hold_sequence(
    n: int,
    holder_id: str,
    holder_team: str,
    holder_xy: tuple[float, float],
    fillers: list[PlayerPos] | None = None,
    dt: float = DT,
    t0: float = 0.0,
    attacking=None,
) -> list[Frame]:
    """n frames where `holder_id` sits on the ball the whole time (well inside
    the default 2m control radius: exactly on the ball)."""
    fillers = fillers or []
    hx, hy = holder_xy
    frames = []
    for k in range(n):
        t = t0 + k * dt
        players = [PlayerPos(holder_id, holder_team, hx, hy), *fillers]
        frames.append(make_frame(t, players, (hx, hy), attacking))
    return frames


def handoff_sequence(
    n_a: int,
    n_gap: int,
    n_b: int,
    a_id: str = "A",
    b_id: str = "B",
    team: str = HOME,
    a_xy: tuple[float, float] = (30.0, 34.0),
    b_xy: tuple[float, float] = (60.0, 34.0),
    dt: float = DT,
    fillers: list[PlayerPos] | None = None,
    attacking=None,
) -> list[Frame]:
    """Scripted hand-off: A holds the ball for n_a frames, then the ball goes
    missing (in flight / undetected) for n_gap frames, then B holds it for
    n_b frames. A and B are both stationary far apart so the nearest-player
    logic in possession() is unambiguous."""
    fillers = fillers or []
    frames = []
    t = 0.0
    for k in range(n_a):
        players = [
            PlayerPos(a_id, team, *a_xy),
            PlayerPos(b_id, team, *b_xy),
            *fillers,
        ]
        frames.append(make_frame(t, players, a_xy, attacking))
        t += dt
    for k in range(n_gap):
        players = [
            PlayerPos(a_id, team, *a_xy),
            PlayerPos(b_id, team, *b_xy),
            *fillers,
        ]
        frames.append(make_frame(t, players, None, attacking))
        t += dt
    for k in range(n_b):
        players = [
            PlayerPos(a_id, team, *a_xy),
            PlayerPos(b_id, team, *b_xy),
            *fillers,
        ]
        frames.append(make_frame(t, players, b_xy, attacking))
        t += dt
    return frames


def far_fillers(near_xy: tuple[float, float] = (52.5, 34.0), n_home: int = 0, n_away: int = 0):
    """Players parked far from `near_xy` so they never interfere with
    possession/distance thresholds in a scripted scenario."""
    out = []
    for i in range(n_home):
        out.append(PlayerPos(f"HF{i}", HOME, 2.0, 2.0 + i))
    for i in range(n_away):
        out.append(PlayerPos(f"AF{i}", AWAY, 103.0, 66.0 - i))
    return out


def dist(a: PlayerPos, b: PlayerPos) -> float:
    return hypot(a.x - b.x, a.y - b.y)
