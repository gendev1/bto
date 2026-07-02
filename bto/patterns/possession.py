"""Ball possession inference (SPEC C9 logic, coordinate side).

Nearest player within a control radius, sustained over time. Tolerates short
ball dropouts (SPEC S9: Kalman/possession logic tolerates short gaps) and
tracker identity swaps (M3 report issue #1): if the nearest-in-radius player
changes track_id but is the SAME TEAM and within handoff_dist of the holder's
last known position, it is treated as the same holder (ByteTrack tid swap
between overlapping players). A real pass moves the ball > handoff_dist away
from the holder before a teammate controls it, so genuine transfers still
split spells; an opponent candidate always splits.
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
    handoff_dist: float = 2.0,  # tid-swap tolerance around holder's last position
) -> list[Spell]:
    """Segment frames into possession spells.

    A spell starts when the same holder is nearest to the ball and within
    control_radius for min_frames consecutive frames, and survives gaps of up
    to max_gap frames where the ball is missing or in flight. "Same holder"
    means same track_id, OR same team within handoff_dist of the holder's
    last known position (tracker identity swap); the spell keeps the original
    track_id but the tracked position follows whichever track matched.
    """
    # per-frame candidate holder: (track_id, team, x, y) or None
    holders: list[tuple[str, str, float, float] | None] = []
    for f in frames:
        if f.ball is None:
            holders.append(None)
            continue
        bx, by = f.ball
        best, best_d = None, control_radius
        for p in f.players:
            d = hypot(p.x - bx, p.y - by)
            if d <= best_d:
                best, best_d = p, d
        holders.append((best.track_id, best.team, best.x, best.y) if best else None)

    def same_holder(h, tid: str, team: str, xy: tuple[float, float]) -> bool:
        return h[1] == team and (
            h[0] == tid or hypot(h[2] - xy[0], h[3] - xy[1]) <= handoff_dist
        )

    spells: list[Spell] = []
    cur: tuple[str, str] | None = None  # (original track_id, team) of the spell
    cur_xy: tuple[float, float] = (0.0, 0.0)  # holder's last known position
    start = 0
    last_seen = 0
    run = 0  # consecutive frames of a not-yet-confirmed candidate
    cand: tuple[str, str] | None = None
    cand_xy: tuple[float, float] = (0.0, 0.0)
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
            if h is not None and same_holder(h, cur[0], cur[1], cur_xy):
                last_seen, cur_xy = i, (h[2], h[3])
            elif h is None and i - last_seen <= max_gap:
                pass  # bridge the gap
            else:
                close(last_seen)
                run, cand = 0, None
        if cur is None and h is not None:
            if cand is not None and same_holder(h, cand[0], cand[1], cand_xy):
                run += 1
            else:
                cand, cand_start, run = (h[0], h[1]), i, 1
            cand_xy = (h[2], h[3])
            if run >= min_frames:
                cur, cur_xy, start, last_seen = cand, cand_xy, cand_start, i
    close(last_seen)
    return spells


# --------------------------------------------------------------------------
# Self-check: scripted tid swap must not split; a real 6m pass must split.
if __name__ == "__main__":
    from bto.core import HOME, PlayerPos

    def frame(t, players, ball):
        return Frame(t=t, players=players, ball=ball, attacking={HOME: 1, "away": -1})

    # 1) tid swap mid-possession at the same spot -> one spell, original tid
    fs = []
    for k in range(20):
        tid = "A" if k < 10 else "A2"  # ByteTrack swap at frame 10
        fs.append(frame(k * 0.04, [PlayerPos(tid, HOME, 30.0, 34.0)], (30.0, 34.0)))
    sp = possession(fs)
    assert len(sp) == 1 and sp[0].track_id == "A" and sp[0].i_end == 19, sp

    # 2) real pass: ball leaves A, teammate B controls it 6m away -> two spells
    fs = []
    a, b = PlayerPos("A", HOME, 30.0, 34.0), PlayerPos("B", HOME, 36.0, 34.0)
    for k in range(10):
        fs.append(frame(k * 0.04, [a, b], (30.0, 34.0)))
    fs.append(frame(10 * 0.04, [a, b], (33.0, 34.0)))  # in flight, nobody in radius
    for k in range(11, 21):
        fs.append(frame(k * 0.04, [a, b], (36.0, 34.0)))
    sp = possession(fs)
    assert [s.track_id for s in sp] == ["A", "B"], sp
    print("possession self-check OK:", "swap kept 1 spell; 6m pass split into 2")
