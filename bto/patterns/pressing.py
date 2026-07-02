"""Pressing intensity detection (SPEC S6 "Pressing intensity"; M6 relational
rework).

detect_pressing now reads its candidate pressers off the relational layer
(bto.patterns.edges): a pressing episode is built from currently-ALIVE
'press' edges on the ball carrier (a defender within edges.R_PRESS and
sustained-CLOSING for edges.PRESS_SUSTAIN_S, per edge lifecycle rules)
instead of a raw per-frame radius scan. Concurrent alive press edges on the
same carrier are the NvN converging press; per-frame intensity blends their
count, mean lifetime strength, and mean closing speed (still measured fresh
per frame from the coordinate stream, since strength is a whole-edge-life
summary and closing speed is inherently instantaneous). Consecutive
qualifying frames merge into a single Detection exactly as before.

Precision gates (M3 precision pass, still enforced, now partly inherited
from the edge birth condition itself): a track carrying the carrier's own
track_id -- or standing within 0.3 m of the carrier (a duplicate/ghost box
whose team label flipped) -- is never counted as a presser (edges.GHOST_DIST
already excludes this at edge birth), the nearest presser must be within
`engage_dist` meters, the mean closing speed must be >= `min_closing` m/s,
an episode must persist >= `min_duration_s`, and 'high' level additionally
requires the nearest presser within 3 m.

geometry dict keys:
    'carrier'   -- (x, y) of the ball carrier at the last frame of the window
    'pressers'  -- list of (x, y) of opposing players within radius
    'intensity' -- float, pressers count + strength + closing-speed weighting
    'level'     -- 'low' | 'medium' | 'high'
    'track'     -- [(t, x_presser, y_presser, x_carrier, y_carrier), ...],
                   merged from the contributing press edges' own samples,
                   sorted by time -- carrier path + presser endpoints in one
                   moving-geometry series (SPEC M6 render convention).
"""

from math import hypot

from bto.core import Detection, Frame
from bto.patterns.edges import Edge, track_edges

_LEVEL_MEDIUM = 3.0
_LEVEL_HIGH = 5.0
_HIGH_NEAREST_MAX = 3.0  # 'high' needs someone genuinely on the carrier


def _pos(frame: Frame, track_id: str) -> tuple[float, float] | None:
    for p in frame.players:
        if p.track_id == track_id:
            return (p.x, p.y)
    return None


def _level(intensity: float, nearest_d: float = 0.0) -> str:
    if intensity >= _LEVEL_HIGH and nearest_d <= _HIGH_NEAREST_MAX:
        return "high"
    if intensity >= _LEVEL_MEDIUM:
        return "medium"
    return "low"


def detect_pressing(
    frames: list[Frame],
    radius: float = 7.0,  # 6.0 cut real Metrica presses below 60% of reference
    window_s: float = 1.0,
    engage_dist: float = 6.0,
    min_closing: float = 1.0,
    merge_gap_s: float = 0.5,
    min_duration_s: float = 0.3,  # 0.4 over-cut on clean data; 0.3 still kills flicker
    carrier_ball_max: float = 3.0,
    edges: list[Edge] | None = None,
) -> list[Detection]:
    """Detect pressing episodes on the ball carrier from alive press edges.

    window_s controls the trailing window (in seconds) used to estimate
    closing speed at each frame. engage_dist is the maximum distance of the
    nearest presser for a frame to qualify; min_closing the minimum mean
    closing speed (m/s). Qualifying runs on the same carrier separated by
    gaps <= merge_gap_s are merged into one episode (temporal hysteresis),
    and episodes shorter than min_duration_s are dropped -- a single-frame
    press chip is imperceptible flicker, not information.

    edges: precomputed track_edges(frames) output; computed internally when
    None (callers running several M6 detectors together, e.g. run_all,
    should compute edges once and pass it to each to avoid recomputation).
    """
    if not frames:
        return []
    if edges is None:
        edges = track_edges(frames)
    press_edges = [e for e in edges if e.kind == "press"]
    if not press_edges:
        return []

    per_frame: list[dict | None] = [None] * len(frames)

    for i, frame in enumerate(frames):
        alive = [e for e in press_edges if e.i_start <= i <= e.i_end]
        if not alive:
            continue
        carrier_id = alive[0].b
        alive = [e for e in alive if e.b == carrier_id]  # one carrier per frame
        c_pos = _pos(frame, carrier_id)
        if c_pos is None:
            continue
        if frame.ball is None or hypot(
            c_pos[0] - frame.ball[0], c_pos[1] - frame.ball[1]
        ) > carrier_ball_max:
            continue

        # find a previous frame within window_s to estimate closing speed
        j = i
        while j > 0 and frame.t - frames[j - 1].t <= window_s:
            j -= 1
        prev_frame = frames[j]
        dt = frame.t - prev_frame.t
        prev_c_pos = _pos(prev_frame, carrier_id)

        pressers = []
        presser_ids = []
        presser_dists = []
        closing_speeds = []
        strengths = []
        contributing: list[Edge] = []
        for e in alive:
            p_pos = _pos(frame, e.a)
            if p_pos is None:
                continue
            d = hypot(p_pos[0] - c_pos[0], p_pos[1] - c_pos[1])
            if d > radius:
                continue
            pressers.append(p_pos)
            presser_ids.append(e.a)
            presser_dists.append(d)
            strengths.append(e.strength)
            contributing.append(e)
            if dt > 0 and prev_c_pos is not None:
                prev_p_pos = _pos(prev_frame, e.a)
                if prev_p_pos is not None:
                    prev_d = hypot(
                        prev_p_pos[0] - prev_c_pos[0], prev_p_pos[1] - prev_c_pos[1]
                    )
                    closing_speed = (prev_d - d) / dt  # positive = closing in
                    closing_speeds.append(closing_speed)

        if len(pressers) < 2:
            continue
        if min(presser_dists) > engage_dist:
            continue  # nobody actually on the carrier, just same-zone players
        mean_closing = (
            sum(closing_speeds) / len(closing_speeds) if closing_speeds else 0.0
        )
        if mean_closing < min_closing:
            continue

        clamped = max(0.0, min(mean_closing, 8.0))
        mean_strength = sum(strengths) / len(strengths)
        # count + closing speed + a lifetime-strength nudge (well-sustained,
        # tight edges count for slightly more than a fresh, marginal one)
        intensity = len(pressers) + clamped + mean_strength
        per_frame[i] = {
            "carrier_id": carrier_id,
            "carrier_pos": c_pos,
            "pressers": pressers,
            "presser_ids": presser_ids,
            "intensity": intensity,
            "nearest_d": min(presser_dists),
            "edges": contributing,
        }

    # merge qualifying frames into episodes: consecutive frames always merge;
    # gaps <= merge_gap_s merge too when the carrier is the same (hysteresis).
    episodes: list[tuple[int, int]] = []  # (first, last) frame indices
    n = len(frames)
    for i in range(n):
        if per_frame[i] is None:
            continue
        if episodes:
            pi, pj = episodes[-1]
            same_carrier = per_frame[pj]["carrier_id"] == per_frame[i]["carrier_id"]
            if frames[i].t - frames[pj].t <= merge_gap_s and same_carrier:
                episodes[-1] = (pi, i)
                continue
        episodes.append((i, i))

    detections: list[Detection] = []
    for i, j in episodes:
        if frames[j].t - frames[i].t < min_duration_s:
            continue  # sub-perceptual flicker
        last = per_frame[j]
        active = [per_frame[k] for k in range(i, j + 1) if per_frame[k] is not None]
        intensity = max(a["intensity"] for a in active)
        nearest_d = min(a["nearest_d"] for a in active)
        players = [last["carrier_id"]] + last["presser_ids"]
        edge_ids = {id(e): e for a in active for e in a["edges"]}
        track = sorted({s for e in edge_ids.values() for s in e.samples}, key=lambda s: s[0])
        # confidence: intensity scaled by how tight the press actually got --
        # a loose "same zone" ring scores under the renderer's callout floor.
        confidence = min(1.0, max(0.0, intensity / (_LEVEL_HIGH + 2.0))) * min(
            1.0, max(0.25, 1.0 - (nearest_d - 1.0) / radius)
        )
        detections.append(
            Detection(
                type="press",
                players=players,
                geometry={
                    "carrier": last["carrier_pos"],
                    "pressers": last["pressers"],
                    "intensity": intensity,
                    "level": _level(intensity, nearest_d),
                    "track": track,
                },
                confidence=confidence,
                t_start=frames[i].t,
                t_end=frames[j].t,
            )
        )

    return detections


if __name__ == "__main__":
    from bto.core import PlayerPos

    # Scripted 3-man converging press: carrier 'a1' stands still at (50, 34)
    # for team home; three away players start 6.5m out (inside edges.R_PRESS
    # once they close a little, so their press edges can actually birth) and
    # close in at 4 m/s, triggering a 'high' press with a run comfortably
    # past the persistence gate. (Pre-M6 self-check started pressers 9m out;
    # M6 press edges only birth inside edges.R_PRESS=5m, so starts are pulled
    # in to guarantee edge candidacy within this clip's runway.)
    dt = 0.04
    n = 75  # 3.0s
    starts = [(44.5, 34.0), (55.5, 34.0), (50.0, 27.5)]
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
    assert all("track" in d.geometry for d in dets)
    assert any(len(d.geometry["track"]) >= 2 for d in dets)
    print(
        "pressing.py self-check OK:",
        [(d.geometry["level"], round(d.geometry["intensity"], 2)) for d in dets],
    )
