"""Overlap / underlap run detection (SPEC S6, socker-plays-ref "Easy" rows).

Overlap: a teammate (R) of the ball carrier (C) sprints from level-or-behind
C to ahead of C along the attacking direction, ending *wider* than C (closer
to the nearest touchline) -- the classic fullback-outside-the-winger run.

Underlap: same shape but R ends in the half-space -- between C's y-position
and the pitch center -- rather than wider.

geometry dict keys: 'path' (list of sampled (x, y) positions of R over the
run, in pitch meters) and 'carrier' ((x, y) of C at the end of the run).
"""

from bto.core import Detection, Frame, PITCH_WIDTH
from bto.patterns.possession import possession

_LEVEL_EPS = 0.5  # meters: "behind or level" tolerance along attacking axis


def _infer_dt(frames: list[Frame]) -> float:
    diffs = [b.t - a.t for a, b in zip(frames, frames[1:]) if b.t > a.t]
    if not diffs:
        return 0.04  # 25 Hz fallback
    diffs.sort()
    return diffs[len(diffs) // 2]


def _pos(frame: Frame, track_id: str) -> tuple[float, float] | None:
    for p in frame.players:
        if p.track_id == track_id:
            return (p.x, p.y)
    return None


def _touchline_dist(y: float) -> float:
    return min(y, PITCH_WIDTH - y)


def _classify_lateral(r_y: float, c_y: float) -> str | None:
    center = PITCH_WIDTH / 2.0
    same_side = (r_y - center) * (c_y - center) > 0
    if same_side and _touchline_dist(r_y) < _touchline_dist(c_y):
        return "overlap"
    lo, hi = min(center, c_y), max(center, c_y)
    if lo <= r_y <= hi and abs(r_y - c_y) > 1e-9:
        return "underlap"
    return None


def _check_window(
    frames: list[Frame],
    i: int,
    j: int,
    r_id: str,
    c_id: str,
    team: str,
    speed_min: float,
    smooth_n: int,
) -> tuple[str, float] | None:
    r_raw = [_pos(frames[k], r_id) for k in range(i, j + 1)]
    if any(p is None for p in r_raw):
        return None
    c_start = _pos(frames[i], c_id)
    c_end = _pos(frames[j], c_id)
    if c_start is None or c_end is None:
        return None

    direction = frames[i].attacking.get(team, 1)

    # smooth R's path with a trailing moving average to kill jitter
    smoothed: list[tuple[float, float]] = []
    for k in range(len(r_raw)):
        lo = max(0, k - smooth_n + 1)
        chunk = r_raw[lo : k + 1]
        sx = sum(p[0] for p in chunk) / len(chunk)
        sy = sum(p[1] for p in chunk) / len(chunk)
        smoothed.append((sx, sy))

    speeds = []
    for k in range(1, len(smoothed)):
        dt = frames[i + k].t - frames[i + k - 1].t
        if dt <= 0:
            continue
        dx = smoothed[k][0] - smoothed[k - 1][0]
        dy = smoothed[k][1] - smoothed[k - 1][1]
        speeds.append((dx * dx + dy * dy) ** 0.5 / dt)
    if not speeds:
        return None
    avg_speed = sum(speeds) / len(speeds)
    if avg_speed < speed_min:
        return None

    r_start_adv = direction * r_raw[0][0]
    c_start_adv = direction * c_start[0]
    if r_start_adv > c_start_adv + _LEVEL_EPS:
        return None  # R must start behind or level with C

    r_end_adv = direction * smoothed[-1][0]
    c_end_adv = direction * c_end[0]
    if r_end_adv <= c_end_adv:
        return None  # R must end ahead of C

    kind = _classify_lateral(smoothed[-1][1], c_end[1])
    if kind is None:
        return None
    return kind, avg_speed


def detect_runs(
    frames: list[Frame],
    speed_min: float = 4.0,
    window_s: float = 2.0,
) -> list[Detection]:
    if len(frames) < 2:
        return []

    dt = _infer_dt(frames)
    window_n = max(2, round(window_s / dt))
    smooth_n = max(1, round(0.4 / dt))

    spells = possession(frames)
    detections: list[Detection] = []

    for spell in spells:
        c_id, team = spell.track_id, spell.team
        lo, hi = spell.i_start, spell.i_end
        if hi - lo < window_n:
            continue

        teammate_ids: set[str] = set()
        for f in frames[lo : hi + 1]:
            for p in f.team_players(team):
                if p.track_id != c_id:
                    teammate_ids.add(p.track_id)

        for r_id in teammate_ids:
            windows: list[tuple[int, int, str, float]] = []
            for i in range(lo, hi - window_n + 1):
                j = i + window_n
                res = _check_window(frames, i, j, r_id, c_id, team, speed_min, smooth_n)
                if res is not None:
                    kind, speed = res
                    windows.append((i, j, kind, speed))

            windows.sort(key=lambda w: w[0])
            merged: list[tuple[int, int, str, float]] = []
            for w in windows:
                if merged and w[0] <= merged[-1][1]:
                    pi, pj, pk, ps = merged[-1]
                    merged[-1] = (pi, max(pj, w[1]), pk if ps >= w[3] else w[2], max(ps, w[3]))
                else:
                    merged.append(w)

            for i, j, kind, speed in merged:
                stride = max(1, (j - i) // 10)
                path = [
                    _pos(frames[k], r_id) for k in range(i, j + 1, stride)
                ]
                carrier = _pos(frames[j], c_id)
                confidence = min(1.0, speed / (speed_min * 2.0))
                detections.append(
                    Detection(
                        type=kind,
                        players=[r_id, c_id],
                        geometry={"path": path, "carrier": carrier},
                        confidence=confidence,
                        t_start=frames[i].t,
                        t_end=frames[j].t,
                    )
                )

    return detections


if __name__ == "__main__":
    from bto.core import PlayerPos

    def make_frame(t, players, ball, attacking):
        return Frame(t=t, players=players, ball=ball, attacking=attacking)

    dt = 0.04  # 25 Hz
    n = 75  # 3 seconds
    att = {"home": 1, "away": -1}

    # --- overlap case: carrier (C) holds ball wide-ish, fullback (R) sprints
    # outside/around him from behind to ahead, ending closer to the touchline.
    frames_overlap = []
    for k in range(n):
        t = k * dt
        c_x = 40.0 + 0.3 * k * dt  # carrier drifts forward slowly
        c_y = 55.0  # carrier stays wide-ish
        r_x = 35.0 + 6.0 * k * dt  # fullback sprints at 6 m/s
        r_y = 62.0  # fullback runs even wider (closer to touchline y=68)
        players = [
            PlayerPos("C", "home", c_x, c_y),
            PlayerPos("R", "home", r_x, r_y),
            PlayerPos("D1", "away", 70.0, 34.0),
        ]
        frames_overlap.append(make_frame(t, players, (c_x, c_y), att))

    dets = detect_runs(frames_overlap, speed_min=4.0, window_s=1.0)
    overlap_dets = [d for d in dets if d.type == "overlap" and set(d.players) == {"R", "C"}]
    assert overlap_dets, f"expected an overlap detection, got {dets}"
    assert overlap_dets[0].confidence > 0.5

    # --- underlap case: same sprint speed, but R ends in the half-space
    # between C's y and the pitch center (34), not wider than C.
    frames_underlap = []
    for k in range(n):
        t = k * dt
        c_x = 40.0 + 0.3 * k * dt
        c_y = 60.0  # carrier out wide near the touchline
        r_x = 35.0 + 6.0 * k * dt
        r_y = 60.0 - 10.0 * (k / n)  # cuts inside toward center, ends at ~50
        players = [
            PlayerPos("C", "home", c_x, c_y),
            PlayerPos("R", "home", r_x, r_y),
            PlayerPos("D1", "away", 70.0, 34.0),
        ]
        frames_underlap.append(make_frame(t, players, (c_x, c_y), att))

    dets_u = detect_runs(frames_underlap, speed_min=4.0, window_s=1.0)
    underlap_dets = [d for d in dets_u if d.type == "underlap" and set(d.players) == {"R", "C"}]
    assert underlap_dets, f"expected an underlap detection, got {dets_u}"

    # --- slow jog: same shape as overlap but well under speed_min -> nothing
    frames_slow = []
    for k in range(n):
        t = k * dt
        c_x = 40.0 + 0.3 * k * dt
        c_y = 55.0
        r_x = 35.0 + 1.0 * k * dt  # jog at 1 m/s
        r_y = 62.0
        players = [
            PlayerPos("C", "home", c_x, c_y),
            PlayerPos("R", "home", r_x, r_y),
            PlayerPos("D1", "away", 70.0, 34.0),
        ]
        frames_slow.append(make_frame(t, players, (c_x, c_y), att))

    dets_slow = detect_runs(frames_slow, speed_min=4.0, window_s=1.0)
    assert not dets_slow, f"slow jog should not fire, got {dets_slow}"

    print("runs.py self-check OK:", len(overlap_dets), "overlap,", len(underlap_dets), "underlap")
