"""Canonical coordinate stream (SPEC S3).

Key design rule: the pattern engine consumes ONLY these types - never pixels.
Coordinates are meters on a 105x68 pitch, origin at the bottom-left corner
(x: 0..105 along the length, y: 0..68). Time is seconds from kickoff.
"""

from dataclasses import dataclass

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

HOME = "home"
AWAY = "away"


@dataclass(frozen=True)
class PlayerPos:
    track_id: str
    team: str  # HOME | AWAY (GK/ref filtered out or folded in by loaders)
    x: float
    y: float


@dataclass
class Frame:
    t: float
    players: list[PlayerPos]
    ball: tuple[float, float] | None  # None when ball position is unknown
    # team -> +1 (attacking toward x=105) or -1 (toward x=0); flips at halftime
    attacking: dict[str, int]

    def team_players(self, team: str) -> list[PlayerPos]:
        return [p for p in self.players if p.team == team]


@dataclass
class Detection:
    """Output contract for every SPEC S6 detector."""

    type: str
    players: list[str]  # track_ids involved
    geometry: dict  # shape data in pitch meters (renderer projects to pixels)
    confidence: float  # 0..1
    t_start: float
    t_end: float


def other(team: str) -> str:
    return AWAY if team == HOME else HOME


def track_velocity(
    frames: list[Frame], i: int, track_id: str, window_s: float = 0.4
) -> tuple[float, float] | None:
    """Smoothed (vx, vy) in m/s of track_id at frame i.

    Positions are already bridge-smoothed; this is the shared smoothed-
    velocity helper for closing-speed/run/matchup estimation. Displacement of
    track_id between frame i and the earliest frame within the trailing
    window_s, divided by dt; None if the track is missing at either end or
    dt <= 0.
    """
    if not frames or i < 0 or i >= len(frames):
        return None
    f = frames[i]
    j = i
    while j > 0 and f.t - frames[j - 1].t <= window_s:
        j -= 1
    dt = f.t - frames[j].t
    if dt <= 0:
        return None
    cur = next((p for p in f.players if p.track_id == track_id), None)
    prev = next((p for p in frames[j].players if p.track_id == track_id), None)
    if cur is None or prev is None:
        return None
    return ((cur.x - prev.x) / dt, (cur.y - prev.y) / dt)
