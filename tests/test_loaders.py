"""Tests for bto/io loaders against the real sample datasets.

Both blocks skip (not fail) when the corresponding data isn't present, so
this file is safe to run before `scripts/get_*.sh` has been run.
"""

from pathlib import Path

import pytest

from bto.core import AWAY, HOME, PITCH_LENGTH, PITCH_WIDTH

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
METRICA_HOME = DATA_DIR / "metrica" / "Sample_Game_1_RawTrackingData_Home_Team.csv"
METRICA_AWAY = DATA_DIR / "metrica" / "Sample_Game_1_RawTrackingData_Away_Team.csv"
SKILLCORNER_DIR = DATA_DIR / "skillcorner" / "1886347"


@pytest.mark.skipif(
    not (METRICA_HOME.exists() and METRICA_AWAY.exists()),
    reason="Metrica sample tracking CSVs not present",
)
def test_load_metrica_bounds_teams_and_attacking_flip():
    metrica = pytest.importorskip("bto.io.metrica")

    frames = metrica.load_metrica(str(METRICA_HOME), str(METRICA_AWAY), downsample=50)
    assert len(frames) > 10

    for f in frames:
        for p in f.players:
            assert -6.0 <= p.x <= PITCH_LENGTH + 6.0
            assert -4.0 <= p.y <= PITCH_WIDTH + 4.0
        if f.ball is not None:
            assert -5.0 <= f.ball[0] <= PITCH_LENGTH + 5.0
            assert -5.0 <= f.ball[1] <= PITCH_WIDTH + 5.0
        teams = {p.team for p in f.players}
        assert teams <= {HOME, AWAY}
        assert set(f.attacking) == {HOME, AWAY}
        assert f.attacking[HOME] in (1, -1)
        assert f.attacking[AWAY] == -f.attacking[HOME]

    # attacking direction must flip between the first and last loaded frame
    # (different halves of the match)
    p1, p2 = frames[0].attacking, frames[-1].attacking
    assert p1[HOME] == -p2[HOME]
    assert p1[AWAY] == -p2[AWAY]

    # time is monotonically non-decreasing
    for a, b in zip(frames, frames[1:]):
        assert b.t >= a.t


@pytest.mark.skipif(
    not SKILLCORNER_DIR.exists(), reason="SkillCorner sample match directory not present"
)
def test_load_skillcorner_bounds_teams_and_attacking_flip():
    skillcorner = pytest.importorskip("bto.io.skillcorner")

    frames = skillcorner.load_skillcorner(str(SKILLCORNER_DIR))
    assert len(frames) > 10

    # downsample manually (loader has no downsample kwarg) to keep the
    # per-frame assertions fast on a full ~90-minute match.
    sample = frames[::50]

    for f in sample:
        for p in f.players:
            assert -3.0 <= p.x <= PITCH_LENGTH + 5.0
            assert -3.0 <= p.y <= PITCH_WIDTH + 5.0
        if f.ball is not None:
            assert -5.0 <= f.ball[0] <= PITCH_LENGTH + 7.0
            assert -5.0 <= f.ball[1] <= PITCH_WIDTH + 7.0
        teams = {p.team for p in f.players}
        assert teams <= {HOME, AWAY}
        assert set(f.attacking) == {HOME, AWAY}
        assert f.attacking[HOME] == -f.attacking[AWAY]

    # attacking direction must flip between periods somewhere in the match
    directions = {f.attacking[HOME] for f in frames}
    assert directions <= {1, -1}

    for a, b in zip(frames, frames[1:]):
        assert b.t >= a.t
