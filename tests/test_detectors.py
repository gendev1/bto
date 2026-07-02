"""Tests for M1 §6 detectors, built on synthetic frames only.

Each sibling module is imported with pytest.importorskip so a missing or
broken sibling module SKIPS this file's tests for it instead of failing the
whole suite.
"""

from math import hypot

import numpy as np
import pytest

from bto.core import AWAY, HOME, PlayerPos
from tests.conftest import DT, far_fillers, make_frame

formation = pytest.importorskip("bto.patterns.formation")
matchups_mod = pytest.importorskip("bto.patterns.matchups")
offside_mod = pytest.importorskip("bto.patterns.offside")
passing_mod = pytest.importorskip("bto.patterns.passing")
pressing_mod = pytest.importorskip("bto.patterns.pressing")
runs_mod = pytest.importorskip("bto.patterns.runs")


# ---------------------------------------------------------------------------
# formation: textbook 4-4-2 lines cluster correctly
# ---------------------------------------------------------------------------


def _line_team(prefix, team, sign, line_spec, gk_depth=5.0):
    """line_spec: [(depth_m, n_players), ...] defense->attack, plus a GK."""

    def x_of(d):
        return d if sign > 0 else 105.0 - d

    players = [PlayerPos(f"{prefix}gk", team, x_of(gk_depth), 34.0)]
    for li, (d, n) in enumerate(line_spec):
        for pi, y in enumerate(np.linspace(10.0, 58.0, n)):
            players.append(PlayerPos(f"{prefix}{li}{pi}", team, x_of(d), float(y)))
    return players


def test_detect_formation_442_clusters_correctly():
    home = _line_team("H", HOME, 1, [(15.0, 4), (35.0, 4), (60.0, 2)])
    away = _line_team("A", AWAY, -1, [(15.0, 4), (35.0, 4), (60.0, 2)])
    frames = [
        make_frame(k * DT, home + away, None, {HOME: 1, AWAY: -1}) for k in range(150)
    ]

    dets = formation.detect_formation(frames, window_s=5.0)
    home_dets = [d for d in dets if d.players[0].startswith("H")]
    assert home_dets, "expected at least one formation Detection for home"
    assert home_dets[0].geometry["label"] == "4-4-2"
    assert home_dets[0].type == "formation"
    assert "Hgk" not in home_dets[0].players  # GK dropped


def test_detect_block_deep_compact_team_reads_low():
    # Whole team squeezed into the defensive third (depth well under 35m).
    home = _line_team("H", HOME, 1, [(10.0, 4), (18.0, 4), (25.0, 2)], gk_depth=4.0)
    away = _line_team("A", AWAY, -1, [(15.0, 4), (35.0, 4), (60.0, 2)])
    frames = [
        make_frame(k * DT, home + away, None, {HOME: 1, AWAY: -1}) for k in range(60)
    ]

    dets = formation.detect_block(frames, window_s=2.0)
    home_dets = [d for d in dets if d.players[0].startswith("H")]
    assert home_dets
    assert home_dets[0].geometry["block"] == "low"


# ---------------------------------------------------------------------------
# matchups: isolated 2v2 fires; isolations: 1v1 fires only when region empty
# ---------------------------------------------------------------------------


def test_detect_matchups_isolated_2v2_fires():
    frames = []
    for k in range(30):
        t = k * DT
        players = [
            PlayerPos("H1", HOME, 90.0, 8.0),
            PlayerPos("H2", HOME, 92.0, 12.0),
            PlayerPos("A1", AWAY, 91.0, 9.0),
            PlayerPos("A2", AWAY, 91.5, 11.0),
            *far_fillers(),
        ]
        frames.append(make_frame(t, players, (91.0, 10.0), {HOME: 1, AWAY: -1}))

    dets = matchups_mod.detect_matchups(frames, radius=8.0, n_max=3, min_duration_s=1.0)
    two_v_two = [d for d in dets if d.type == "2v2" and set(d.players) == {"H1", "H2", "A1", "A2"}]
    assert two_v_two, [(d.type, d.players) for d in dets]
    assert two_v_two[0].t_end - two_v_two[0].t_start >= 1.0 - 1e-9


def test_detect_isolations_1v1_fires_only_when_region_empty():
    frames = []
    for k in range(30):
        t = k * DT
        ball_with_iso = k < 15
        players = [
            # clean 1v1, far from everyone: 2m apart -> engaged (< engage_dist=5)
            PlayerPos("H3", HOME, 10.0, 60.0),
            PlayerPos("A3", AWAY, 12.0, 60.0),
            # crowded duel: same tight marking, but teammates nearby -> must not fire
            PlayerPos("H4", HOME, 50.0, 34.0),
            PlayerPos("A4", AWAY, 52.0, 34.0),
            PlayerPos("H5", HOME, 54.0, 35.0),
            PlayerPos("A5", AWAY, 48.0, 33.0),
        ]
        ball = (10.0, 60.0) if ball_with_iso else (52.0, 34.0)
        frames.append(make_frame(t, players, ball, {HOME: 1, AWAY: -1}))

    dets = matchups_mod.detect_isolations(frames, iso_radius=10.0, engage_dist=5.0)
    iso_sets = [set(d.players) for d in dets]
    assert {"H3", "A3"} in iso_sets
    for d in dets:
        assert not ({"H4", "A4"} & set(d.players)), "crowded carrier must not isolate"


# ---------------------------------------------------------------------------
# triangles: open triangle fires, blocked lane doesn't
# ---------------------------------------------------------------------------


def test_detect_triangles_open_fires():
    frames = []
    for i in range(15):
        t = i * 0.1
        players = [
            PlayerPos("C", HOME, 50.0, 34.0),
            PlayerPos("P1", HOME, 55.0, 40.0),
            PlayerPos("P2", HOME, 45.0, 40.0),
            *far_fillers(n_home=0, n_away=2),
        ]
        frames.append(make_frame(t, players, (50.0, 34.0), {HOME: 1, AWAY: -1}))

    dets = passing_mod.detect_triangles(frames)
    assert len(dets) == 1
    assert dets[0].type == "triangle"
    assert set(dets[0].players) == {"C", "P1", "P2"}


def test_detect_triangles_blocked_lane_does_not_fire():
    frames = []
    for i in range(15):
        t = i * 0.1
        players = [
            PlayerPos("C", HOME, 50.0, 34.0),
            PlayerPos("P1", HOME, 55.0, 40.0),
            PlayerPos("P2", HOME, 45.0, 40.0),
            PlayerPos("O3", AWAY, 52.5, 37.0),  # sits right on the C-P1 lane
        ]
        frames.append(make_frame(t, players, (50.0, 34.0), {HOME: 1, AWAY: -1}))

    assert passing_mod.detect_triangles(frames) == []


# ---------------------------------------------------------------------------
# back passes: backward transfer fires, forward doesn't
# ---------------------------------------------------------------------------


def _backpass_seq(b_pos):
    seq = []
    for i in range(12):
        t = i * 0.1
        actors = [PlayerPos("A", HOME, 30.0, 34.0), PlayerPos("B", HOME, *b_pos)]
        if i < 5:
            ball = (30.0, 34.0)
        elif i < 7:
            ball = (25.0, 50.0)  # in flight, far from either actor
        else:
            ball = b_pos
        seq.append(make_frame(t, actors, ball, {HOME: 1, AWAY: -1}))
    return seq


def test_detect_back_pass_backward_fires():
    frames = _backpass_seq((20.0, 34.0))  # behind A given attacking=+1
    dets = passing_mod.detect_back_passes(frames)
    assert len(dets) == 1
    assert dets[0].type == "back_pass"
    assert dets[0].players == ["A", "B"]


def test_detect_back_pass_forward_does_not_fire():
    frames = _backpass_seq((40.0, 34.0))  # ahead of A -> not a back pass
    assert passing_mod.detect_back_passes(frames) == []


# ---------------------------------------------------------------------------
# runs: fast outside run -> overlap
# ---------------------------------------------------------------------------


def test_detect_runs_fast_outside_run_is_overlap():
    n = 75
    frames = []
    for k in range(n):
        t = k * DT
        c_x = 40.0 + 0.3 * k * DT
        c_y = 55.0
        r_x = 35.0 + 6.0 * k * DT  # sprints at 6 m/s, well above speed_min=4
        r_y = 62.0  # ends closer to the touchline than C
        players = [
            PlayerPos("C", HOME, c_x, c_y),
            PlayerPos("R", HOME, r_x, r_y),
            PlayerPos("D1", AWAY, 70.0, 34.0),
        ]
        frames.append(make_frame(t, players, (c_x, c_y), {HOME: 1, AWAY: -1}))

    dets = runs_mod.detect_runs(frames, speed_min=4.0, window_s=1.0)
    overlaps = [d for d in dets if d.type == "overlap" and set(d.players) == {"R", "C"}]
    assert overlaps, dets


# ---------------------------------------------------------------------------
# pressing: 3 closing defenders -> press
# ---------------------------------------------------------------------------


def test_detect_pressing_three_closing_defenders_fires():
    dt = DT
    n = 25
    starts = [(41.0, 34.0), (59.0, 34.0), (50.0, 25.0)]
    frames = []
    for k in range(n):
        t = k * dt
        players = [PlayerPos("a1", HOME, 50.0, 34.0)]
        for idx, (sx, sy) in enumerate(starts):
            dx, dy = 50.0 - sx, 34.0 - sy
            d = hypot(dx, dy)
            step = min(4.0 * t, d * 0.9)
            if d > 0:
                x, y = sx + dx / d * step, sy + dy / d * step
            else:
                x, y = sx, sy
            players.append(PlayerPos(f"d{idx}", AWAY, x, y))
        frames.append(make_frame(t, players, (50.0, 34.0), {HOME: 1, AWAY: -1}))

    dets = pressing_mod.detect_pressing(frames, radius=10.0, window_s=1.0)
    assert dets
    assert any(d.geometry["level"] == "high" for d in dets)
    assert all(d.type == "press" for d in dets)


# ---------------------------------------------------------------------------
# offside_line: lands on 2nd-rearmost defender
# ---------------------------------------------------------------------------


def test_offside_line_lands_on_second_rearmost_defender():
    dt = DT
    n = 30
    xs = [2.0, 10.0, 15.0, 20.0, 25.0]  # away's own goal at x=105 (attacking=-1)
    frames = []
    for k in range(n):
        t = k * dt
        players = [PlayerPos(f"h{i}", HOME, 60.0 + i, 34.0) for i in range(3)]
        players += [PlayerPos(f"a{i}", AWAY, x, 34.0) for i, x in enumerate(xs)]
        frames.append(make_frame(t, players, (50.0, 34.0), {HOME: 1, AWAY: -1}))

    dets = offside_mod.offside_line(frames, smooth_s=0.5)
    away_dets = [d for d in dets if d.geometry["team_defending"] == AWAY]
    assert away_dets
    for d in away_dets:
        assert abs(d.geometry["x"] - 20.0) < 1e-6
        assert d.geometry["approximate"] is True
        assert d.type == "offside_line"
