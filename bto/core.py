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
