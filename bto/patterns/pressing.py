"""Pressing intensity detection (SPEC S6 "Pressing intensity").

For the current ball carrier (from possession()), count opposing players
within `radius` meters and their mean closing speed (negative d(distance)/dt
toward the carrier, computed from position deltas between consecutive
frames). Consecutive positive frames are merged into a single Detection.

geometry dict keys:
    'carrier'   -- (x, y) of the ball carrier at the last frame of the window
    'pressers'  -- list of (x, y) of opposing players within radius
    'intensity' -- float, pressers count + closing-speed weighting
    'level'     -- 'low' | 'medium' | 'high'
"""

from math import hypot

from bto.core import Detection, Frame, other
from bto.patterns.possession import possession

_LEVEL_MEDIUM = 3.0
_LEVEL_HIGH = 5.0


def _pos(frame: Frame, track_id: str) -> tuple[float, float] | None:
    for p in frame.players:
        if p.track_id == track_id:
            return (p.x, p.y)
    return None


def _level(intensity: float) -> str:
    if intensity >= _LEVEL_HIGH:
        return "high"
    if intensity >= _LEVEL_MEDIUM:
        return "medium"
    return "low"


def detect_pressing(
    frames: list[Frame],
    radius: float = 10.0,
    window_s: float = 1.0,
) -> list[Detection]:
    """Detect pressing episodes on the ball carrier.

    window_s controls the trailing window (in seconds) used to estimate
    closing speed at each frame.
    """
    if not frames:
        return []

    spells = possession(frames)
    # map frame index -> (carrier track_id, team)
    carrier_at: dict[int, tuple[str, str]] = {}
    for spell in spells:
        for i in range(spell.i_start, spell.i_end + 1):
            carrier_at[i] = (spell.track_id, spell.team)

    per_frame: list[dict | None] = [None] * len(frames)

    for i, frame in enumerate(frames):
        carrier = carrier_at.get(i)
        if carrier is None:
            continue
        carrier_id, carrier_team = carrier
        c_pos = _pos(frame, carrier_id)
        if c_pos is None:
            continue

        # find a previous frame within window_s to estimate closing speed
        j = i
        while j > 0 and frame.t - frames[j - 1].t <= window_s:
            j -= 1
        prev_frame = frames[j]
        dt = frame.t - prev_frame.t
        prev_c_pos = _pos(prev_frame, carrier_id)

        opp_team = other(carrier_team)
        pressers = []
        closing_speeds = []
        for p in frame.team_players(opp_team):
            d = hypot(p.x - c_pos[0], p.y - c_pos[1])
            if d > radius:
                continue
            pressers.append((p.x, p.y))
            if dt > 0 and prev_c_pos is not None:
                prev_p_pos = _pos(prev_frame, p.track_id)
                if prev_p_pos is not None:
                    prev_d = hypot(
                        prev_p_pos[0] - prev_c_pos[0], prev_p_pos[1] - prev_c_pos[1]
                    )
                    closing_speed = (prev_d - d) / dt  # positive = closing in
                    closing_speeds.append(closing_speed)

        if len(pressers) < 2:
            continue
        mean_closing = (
            sum(closing_speeds) / len(closing_speeds) if closing_speeds else 0.0
        )
        if mean_closing <= 0:
            continue

        clamped = max(0.0, min(mean_closing, 8.0))
        intensity = len(pressers) + clamped
        per_frame[i] = {
            "carrier_id": carrier_id,
            "carrier_pos": c_pos,
            "pressers": pressers,
            "presser_ids": [p.track_id for p in frame.team_players(opp_team)
                            if hypot(p.x - c_pos[0], p.y - c_pos[1]) <= radius],
            "intensity": intensity,
        }

    detections: list[Detection] = []
    i = 0
    n = len(frames)
    while i < n:
        if per_frame[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and per_frame[j + 1] is not None:
            j += 1
        # merged segment [i, j]
        last = per_frame[j]
        intensity = max(per_frame[k]["intensity"] for k in range(i, j + 1))
        players = [last["carrier_id"]] + last["presser_ids"]
        detections.append(
            Detection(
                type="press",
                players=players,
                geometry={
                    "carrier": last["carrier_pos"],
                    "pressers": last["pressers"],
                    "intensity": intensity,
                    "level": _level(intensity),
                },
                confidence=min(1.0, intensity / (_LEVEL_HIGH + 2.0)),
                t_start=frames[i].t,
                t_end=frames[j].t,
            )
        )
        i = j + 1

    return detections


if __name__ == "__main__":
    from bto.core import PlayerPos

    # Scripted 3-man converging press: carrier 'a1' stands still at (50, 34)
    # for team home; three away players start 9m out and close in at 4 m/s
    # over 1s (25 frames), triggering a 'high' intensity press.
    dt = 0.04
    n = 25
    starts = [(41.0, 34.0), (59.0, 34.0), (50.0, 25.0)]
    frames = []
    for k in range(n):
        t = k * dt
        players = [PlayerPos("a1", "home", 50.0, 34.0)]
        for idx, (sx, sy) in enumerate(starts):
            # move toward (50,34) at ~4 m/s
            dx, dy = 50.0 - sx, 34.0 - sy
            dist = hypot(dx, dy)
            step = min(4.0 * t, dist * 0.9)
            if dist > 0:
                x = sx + dx / dist * step
                y = sy + dy / dist * step
            else:
                x, y = sx, sy
            players.append(PlayerPos(f"d{idx}", "away", x, y))
        frames.append(
            Frame(t=t, players=players, ball=(50.0, 34.0), attacking={"home": 1, "away": -1})
        )

    dets = detect_pressing(frames, radius=10.0, window_s=1.0)
    assert dets, "expected at least one press detection"
    assert any(d.geometry["level"] == "high" for d in dets), [
        d.geometry for d in dets
    ]
    assert all(d.type == "press" for d in dets)
    print("pressing.py self-check OK:", [(d.geometry["level"], round(d.geometry["intensity"], 2)) for d in dets])
