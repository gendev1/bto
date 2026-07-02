"""Passing triangle and back-pass detectors (SPEC S6).

Two detectors on the coordinate stream, both pure `list[Frame] -> list[Detection]`:

- ``detect_triangles``: three same-team players around the ball carrier, pairwise
  distances in ``[dmin, dmax]``, with every side's passing lane clear of opponents
  (no opponent within ``lane_clear`` meters of the segment, checked via point-to-
  segment distance vectorized over opponents with numpy). Only evaluated for the
  team currently in possession (per ``possession()``), and only trios that include
  the ball carrier, to keep the combinatorics small. Consecutive frames reporting
  the same trio (as a ``frozenset`` of track_ids) are merged into one Detection.
  geometry = ``{'vertices': [(x, y) x3]}`` (carrier first, then the two teammates).

- ``detect_back_passes``: consecutive possession spells A -> B (from
  ``possession()``), same team, release-to-reception gap < 3s, ball travel
  >= 3m, and B's reception x is behind A's release x relative to that team's
  attacking direction (``Frame.attacking``). geometry = ``{'from': (x, y),
  'to': (x, y)}`` using A's position at release and B's position at reception.

  Precision gates (M3 precision pass): the displacement must be predominantly
  backward (``|dx| / dist >= min_backward_ratio``, rejecting lateral passes
  with a marginal x-decrease) and the implied ball speed between release and
  reception must be physically plausible (``<= max_ball_speed`` m/s, rejecting
  stale / non-adjacent touch pairings stitched into one fake long pass).
  A second back pass on the same unordered player pair within ``cooldown_s``
  of a kept one is suppressed (possession churn between the same two players
  reads as one event to a viewer, not a stream of arrows). When the ball is
  visible at release/reception, the from/to player must be within
  ``carrier_ball_max`` of it (stale spell tids otherwise anchor the arrow on
  a player nowhere near the actual ball). Confidence blends
  backward dominance with speed plausibility (a slow straight back pass
  scores ~1.0; a fast, marginal one lands near the renderer's floor).
"""

from itertools import combinations
from math import hypot

import numpy as np

from bto.core import Detection, Frame, other
from bto.patterns.possession import possession


def _point_seg_dist(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    """Vectorized point-to-segment distance: points (px, py) vs segment a->b."""
    abx, aby = bx - ax, by - ay
    seg_len2 = abx * abx + aby * aby
    if seg_len2 < 1e-9:
        return np.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    projx = ax + t * abx
    projy = ay + t * aby
    return np.hypot(px - projx, py - projy)


def detect_triangles(
    frames: list[Frame],
    dmin: float = 5.0,
    dmax: float = 22.0,  # compactness cap: kills touchline-to-touchline "triangles"
    # (15m cut real Metrica triangles to <10% of reference; 22m keeps ~2/3 of
    # them while confidence -- monotone in compactness -- downweights the
    # stretched ones below the renderer's callout floor)
    lane_clear: float = 2.0,
    min_duration_s: float = 1.0,
    carrier_ball_max: float = 3.0,
) -> list[Detection]:
    spells = possession(frames)
    carrier: list[tuple[str, str] | None] = [None] * len(frames)
    for s in spells:
        for i in range(s.i_start, s.i_end + 1):
            carrier[i] = (s.track_id, s.team)

    active: dict[frozenset, dict] = {}
    detections: list[Detection] = []

    def close(info: dict) -> None:
        t_start = frames[info["start"]].t
        t_end = frames[info["last"]].t
        if t_end - t_start >= min_duration_s:
            # compactness confidence: a tight local triangle (sides near dmin)
            # scores ~1.0, one stretched to the dmax cap scores the 0.2 floor.
            conf = min(1.0, max(0.2, 1.5 * (1.0 - info["max_side"] / dmax)))
            detections.append(
                Detection(
                    type="triangle",
                    players=list(info["players_order"]),
                    geometry={"vertices": info["vertices"]},
                    confidence=conf,
                    t_start=t_start,
                    t_end=t_end,
                )
            )

    for i, f in enumerate(frames):
        found: dict[frozenset, tuple[tuple[str, str, str], list[tuple[float, float]], float]] = {}
        c = carrier[i]
        if c is not None:
            cid, team = c
            pos_by_id = {p.track_id: p for p in f.players}
            cpos = pos_by_id.get(cid)
            # spell tid can go stale across tracker id swaps: only trust
            # frames where the looked-up carrier is actually on the ball.
            if cpos is not None and (
                f.ball is None
                or hypot(cpos.x - f.ball[0], cpos.y - f.ball[1]) > carrier_ball_max
            ):
                cpos = None
            if cpos is not None:
                cx, cy = cpos.x, cpos.y
                teammates = [p for p in f.team_players(team) if p.track_id != cid]
                # only teammates already within [dmin, dmax] of the carrier are
                # candidate triangle vertices -- keeps the combinatorics small.
                near = [p for p in teammates if dmin <= hypot(p.x - cx, p.y - cy) <= dmax]
                opponents = f.team_players(other(team))
                opp_xy = np.array([[p.x, p.y] for p in opponents], dtype=float) if opponents else np.zeros((0, 2))

                for p1, p2 in combinations(near, 2):
                    d12 = hypot(p1.x - p2.x, p1.y - p2.y)
                    if not (dmin <= d12 <= dmax):
                        continue
                    max_side = max(
                        d12,
                        hypot(p1.x - cx, p1.y - cy),
                        hypot(p2.x - cx, p2.y - cy),
                    )
                    edges = [
                        (cx, cy, p1.x, p1.y),
                        (cx, cy, p2.x, p2.y),
                        (p1.x, p1.y, p2.x, p2.y),
                    ]
                    if opp_xy.shape[0] > 0:
                        blocked = False
                        for ax, ay, bx, by in edges:
                            dseg = _point_seg_dist(opp_xy[:, 0], opp_xy[:, 1], ax, ay, bx, by)
                            if np.any(dseg < lane_clear):
                                blocked = True
                                break
                        if blocked:
                            continue
                    key = frozenset((cid, p1.track_id, p2.track_id))
                    found[key] = (
                        (cid, p1.track_id, p2.track_id),
                        [(cx, cy), (p1.x, p1.y), (p2.x, p2.y)],
                        max_side,
                    )

        for key in list(active.keys()):
            if key not in found:
                close(active.pop(key))
        for key, (players_order, vertices, max_side) in found.items():
            if key in active:
                info = active[key]
                info["last"] = i
                info["vertices"] = vertices  # keep geometry fresh at close
                info["max_side"] = max(info["max_side"], max_side)
            else:
                active[key] = {
                    "start": i,
                    "last": i,
                    "players_order": players_order,
                    "vertices": vertices,
                    "max_side": max_side,
                }

    for info in active.values():
        close(info)

    return detections


def detect_back_passes(
    frames: list[Frame],
    min_backward_ratio: float = 0.6,
    max_ball_speed: float = 28.0,
    cooldown_s: float = 8.0,
    carrier_ball_max: float = 5.0,
) -> list[Detection]:
    spells = possession(frames)
    detections: list[Detection] = []

    # median inter-frame dt: floor for the release->reception gap so a
    # possession flicker (gap ~ 0) doesn't imply infinite ball speed.
    dts = sorted(f2.t - f1.t for f1, f2 in zip(frames, frames[1:]) if f2.t > f1.t)
    med_dt = dts[len(dts) // 2] if dts else 0.08
    last_kept: dict[frozenset, float] = {}  # unordered pair -> t of kept det

    for a, b in zip(spells, spells[1:]):
        if a.team != b.team or a.track_id == b.track_id:
            continue
        gap = b.t_start - a.t_end
        if gap < 0 or gap >= 3.0:
            continue

        a_frame, b_frame = frames[a.i_end], frames[b.i_start]
        a_pos = next((p for p in a_frame.players if p.track_id == a.track_id), None)
        b_pos = next((p for p in b_frame.players if p.track_id == b.track_id), None)
        if a_pos is None or b_pos is None:
            continue

        # anchor sanity (same idea as detect_triangles' carrier check): spell
        # tids go stale across tracker id swaps, leaving 'from'/'to' anchored
        # on a player 15-25m from the actual ball -- the arrow then draws a
        # backward pass that never happened. When the ball is visible at
        # release/reception, the anchored player must actually be on it.
        if a_frame.ball is not None and hypot(
            a_pos.x - a_frame.ball[0], a_pos.y - a_frame.ball[1]
        ) > carrier_ball_max:
            continue
        if b_frame.ball is not None and hypot(
            b_pos.x - b_frame.ball[0], b_pos.y - b_frame.ball[1]
        ) > carrier_ball_max:
            continue

        if a_frame.ball is not None and b_frame.ball is not None:
            traveled = hypot(b_frame.ball[0] - a_frame.ball[0], b_frame.ball[1] - a_frame.ball[1])
        else:
            traveled = hypot(b_pos.x - a_pos.x, b_pos.y - a_pos.y)
        if traveled < 3.0:
            continue

        direction = a_frame.attacking.get(a.team, 1)
        disp = hypot(b_pos.x - a_pos.x, b_pos.y - a_pos.y)
        back_dx = (a_pos.x - b_pos.x) * direction  # > 0 means toward own goal
        if disp < 1e-6 or back_dx / disp < min_backward_ratio:
            continue  # lateral (or forward) pass, not predominantly backward

        # implied speed cap: reject stale / non-adjacent touch pairings. Check
        # BOTH the ball's travel and the drawn from->to player displacement --
        # a mistracked pairing can have a plausible ball delta but an
        # impossibly long player-to-player arrow.
        speed = max(traveled, disp) / max(gap, med_dt)
        if speed > max_ball_speed:
            continue

        # cooldown: the same unordered pair trading the ball back and forth
        # is one story, not many; keep the earlier detection.
        pair = frozenset((a.track_id, b.track_id))
        if pair in last_kept and a.t_end - last_kept[pair] < cooldown_s:
            continue
        last_kept[pair] = a.t_end

        ratio = back_dx / disp
        confidence = min(1.0, max(0.2, 0.6 * ratio + 0.4 * (1.0 - speed / max_ball_speed)))
        detections.append(
            Detection(
                type="back_pass",
                players=[a.track_id, b.track_id],
                geometry={"from": (a_pos.x, a_pos.y), "to": (b_pos.x, b_pos.y)},
                confidence=confidence,
                t_start=a.t_end,
                t_end=b.t_start,
            )
        )

    return detections


if __name__ == "__main__":
    from bto.core import PlayerPos

    def _fillers(t: float) -> list[PlayerPos]:
        # far-away players so team_players() has more than the actors, without
        # interfering with distances/lanes.
        return [
            PlayerPos("H_far", "home", 5.0, 5.0),
            PlayerPos("A_far1", "away", 90.0, 60.0),
            PlayerPos("A_far2", "away", 95.0, 5.0),
        ]

    ATTACK = {"home": 1, "away": -1}

    # --- triangle: clean case ---------------------------------------------
    tri_frames = []
    for i in range(15):
        t = i * 0.1
        players = [
            PlayerPos("C", "home", 50.0, 34.0),
            PlayerPos("P1", "home", 55.0, 40.0),
            PlayerPos("P2", "home", 45.0, 40.0),
            *_fillers(t),
        ]
        tri_frames.append(Frame(t=t, players=players, ball=(50.0, 34.0), attacking=ATTACK))

    tri_dets = detect_triangles(tri_frames)
    assert len(tri_dets) == 1, tri_dets
    d = tri_dets[0]
    assert d.type == "triangle"
    assert set(d.players) == {"C", "P1", "P2"}
    assert d.t_end - d.t_start >= 1.0
    assert len(d.geometry["vertices"]) == 3

    # --- triangle: blocked lane kills it ------------------------------------
    blocked_frames = []
    for i in range(15):
        t = i * 0.1
        players = [
            PlayerPos("C", "home", 50.0, 34.0),
            PlayerPos("P1", "home", 55.0, 40.0),
            PlayerPos("P2", "home", 45.0, 40.0),
            PlayerPos("O3", "away", 52.5, 37.0),  # sits on the C-P1 lane midpoint
            *_fillers(t),
        ]
        blocked_frames.append(Frame(t=t, players=players, ball=(50.0, 34.0), attacking=ATTACK))

    assert detect_triangles(blocked_frames) == []

    # --- back pass: A holds, ball transits, B (deeper) receives ------------
    def _backpass_seq(b_pos: tuple[float, float]) -> list[Frame]:
        # 0.5s release->reception gap over 10m = 20 m/s, inside the 28 m/s cap
        seq = []
        for i in range(15):
            t = i * 0.1
            actors = [PlayerPos("A", "home", 30.0, 34.0), PlayerPos("B", "home", *b_pos)]
            if i < 5:
                ball = (30.0, 34.0)  # A holds
            elif i < 9:
                ball = (25.0, 50.0)  # in flight, far from everyone
            else:
                ball = b_pos  # B holds
            players = [*actors, *_fillers(t)]
            seq.append(Frame(t=t, players=players, ball=ball, attacking=ATTACK))
        return seq

    back_frames = _backpass_seq((20.0, 34.0))  # behind A (attacking +1: lower x = deeper)
    back_dets = detect_back_passes(back_frames)
    assert len(back_dets) == 1, back_dets
    bd = back_dets[0]
    assert bd.type == "back_pass"
    assert bd.players == ["A", "B"]
    assert bd.geometry["from"] == (30.0, 34.0)
    assert bd.geometry["to"] == (20.0, 34.0)

    forward_frames = _backpass_seq((40.0, 34.0))  # ahead of A -> not a back pass
    assert detect_back_passes(forward_frames) == []

    print("passing.py self-check OK")
