"""Tests for the frozen possession() contract (bto/patterns/possession.py)."""

from bto.core import AWAY, HOME, PlayerPos
from bto.patterns.possession import possession
from tests.conftest import DT, handoff_sequence, make_frame


def test_handoff_two_spells_correct_ids_and_times():
    # A holds 10 frames, ball transits (in-flight/missing) 5 frames (well
    # under the default max_gap=12), then B holds 10 frames.
    frames = handoff_sequence(n_a=10, n_gap=5, n_b=10, a_id="A", b_id="B", team=HOME)

    spells = possession(frames)

    assert [s.track_id for s in spells] == ["A", "B"]
    assert all(s.team == HOME for s in spells)

    a_spell, b_spell = spells
    assert a_spell.i_start == 0
    assert a_spell.t_start == frames[0].t
    # A's spell should end at/around frame 9 (last frame A held before the gap)
    assert a_spell.i_end == 9
    assert a_spell.t_end == frames[9].t

    # B's spell should start once B is confirmed holder (min_frames=3 after
    # the gap ends at index 15), and run to the final frame.
    assert b_spell.i_start >= 15
    assert b_spell.i_end == len(frames) - 1
    assert b_spell.t_end == frames[-1].t

    # spells must not overlap
    assert a_spell.i_end < b_spell.i_start


def test_short_ball_dropout_does_not_split_spell():
    # Same holder throughout; ball just disappears briefly (gap well under
    # max_gap=12) then reappears with the SAME holder -- must stay one spell.
    frames = handoff_sequence(n_a=10, n_gap=6, n_b=10, a_id="A", b_id="B", team=HOME)
    # Rewrite so it's actually a single holder across the dropout: replace the
    # "B holds" tail with "A holds" again.
    fixed = []
    for f in frames:
        if f.ball == (60.0, 34.0):
            fixed.append(make_frame(f.t, f.players, (30.0, 34.0), f.attacking))
        else:
            fixed.append(f)

    spells = possession(fixed)

    assert len(spells) == 1
    assert spells[0].track_id == "A"
    assert spells[0].i_start == 0
    assert spells[0].i_end == len(fixed) - 1


def test_gap_exceeding_max_gap_splits_into_two_spells_same_player():
    # A gap longer than max_gap=12 (default) breaks continuity even for the
    # same eventual holder, since the bridge can no longer reach across it.
    frames = handoff_sequence(n_a=10, n_gap=20, n_b=10, a_id="A", b_id="B", team=HOME)
    spells = possession(frames, max_gap=12)
    assert len(spells) == 2
    assert spells[0].track_id == "A"
    assert spells[1].track_id == "B"


def test_no_ball_produces_no_spells():
    fillers = [PlayerPos("H1", HOME, 50.0, 34.0), PlayerPos("A1", AWAY, 55.0, 34.0)]
    frames = [make_frame(k * DT, fillers, None) for k in range(20)]
    assert possession(frames) == []
