"""Ball possession inference (SPEC C9 logic, coordinate side).

Nearest player within a control radius, sustained over time. Tolerates short
ball dropouts (SPEC S9: Kalman/possession logic tolerates short gaps).
"""

from dataclasses import dataclass
from math import hypot

from bto.core import Frame


@dataclass
class Spell:
    track_id: str
    team: str
    t_start: float
    t_end: float
    i_start: int  # frame indices into the input list
    i_end: int


def possession(
    frames: list[Frame],
    control_radius: float = 2.0,
    min_frames: int = 3,
    max_gap: int = 12,  # ~0.5s at 25Hz: ball missing / in flight
) -> list[Spell]:
    """Segment frames into possession spells.

    A spell starts when the same player is nearest to the ball and within
    control_radius for min_frames consecutive frames, and survives gaps of up
    to max_gap frames where the ball is missing or in flight.
    """
    # per-frame candidate holder: (track_id, team) or None
    holders: list[tuple[str, str] | None] = []
    for f in frames:
        if f.ball is None:
            holders.append(None)
            continue
        bx, by = f.ball
        best, best_d = None, control_radius
        for p in f.players:
            d = hypot(p.x - bx, p.y - by)
            if d <= best_d:
                best, best_d = (p.track_id, p.team), d
        holders.append(best)

    spells: list[Spell] = []
    cur: tuple[str, str] | None = None
    start = 0
    last_seen = 0
    run = 0  # consecutive frames of a not-yet-confirmed candidate
    cand: tuple[str, str] | None = None
    cand_start = 0

    def close(i_end: int) -> None:
        nonlocal cur
        if cur is not None:
            spells.append(
                Spell(cur[0], cur[1], frames[start].t, frames[i_end].t, start, i_end)
            )
            cur = None

    for i, h in enumerate(holders):
        if cur is not None:
            if h == cur:
                last_seen = i
            elif h is None and i - last_seen <= max_gap:
                pass  # bridge the gap
            else:
                close(last_seen)
                run, cand = 0, None
        if cur is None and h is not None:
            if h == cand:
                run += 1
            else:
                cand, cand_start, run = h, i, 1
            if run >= min_frames:
                cur, start, last_seen = cand, cand_start, i
    close(last_seen)
    return spells
