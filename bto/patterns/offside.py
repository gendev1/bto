"""Approximate offside line (SPEC S6 "Approximate offside line").

Per frame, per defending team, the offside line is approximated as the x
position of the SECOND-rearmost defender relative to the goal that team
defends (the rearmost defender is usually the goalkeeper). The raw per-frame
line is smoothed with a trailing moving average over `smooth_s` seconds, then
resampled into ~`_SEGMENT_S`-second segments emitted as one Detection each.

NEVER present this as an offside call (SPEC S9) -- it is an approximation
for overlay purposes only; geometry carries 'approximate': True and the
renderer must label it accordingly (dashed line, "approx offside").

geometry dict keys:
    'x'               -- smoothed offside-line x (pitch meters) at segment end
    'team_defending'  -- team whose defensive line this is
    'approximate'     -- always True
    'samples'         -- list of (t, x) smoothed samples within the segment
"""

from bto.core import AWAY, Detection, Frame, HOME

# Resample cadence: one Detection per ~2s. Was 1.0s; the always-on line
# chained a fresh detection pair every second for the whole clip, which read
# as spam even though the geometry is accurate (M3 precision pass).
_SEGMENT_S = 2.0


def _second_rearmost_x(frame: Frame, team: str) -> float | None:
    direction = frame.attacking.get(team, 1)
    players = frame.team_players(team)
    if len(players) < 2:
        return None
    # "rearmost" = closest to own goal along the direction this team
    # defends, i.e. smallest advancement in the attacking direction.
    xs = sorted(players, key=lambda p: direction * p.x)
    return xs[1].x  # second-rearmost


def _infer_dt(frames: list[Frame]) -> float:
    diffs = [b.t - a.t for a, b in zip(frames, frames[1:]) if b.t > a.t]
    if not diffs:
        return 0.04
    diffs.sort()
    return diffs[len(diffs) // 2]


def offside_line(frames: list[Frame], smooth_s: float = 0.5) -> list[Detection]:
    """Compute a smoothed, resampled approximate offside line per team."""
    if not frames:
        return []

    dt = _infer_dt(frames)
    smooth_n = max(1, round(smooth_s / dt)) if dt > 0 else 1

    detections: list[Detection] = []
    for team in (HOME, AWAY):
        raw: list[tuple[float, float] | None] = []  # (t, x) or None
        for f in frames:
            x = _second_rearmost_x(f, team)
            raw.append((f.t, x) if x is not None else None)

        # trailing moving average smoothing, skipping missing values
        smoothed: list[tuple[float, float] | None] = []
        for k in range(len(raw)):
            lo = max(0, k - smooth_n + 1)
            chunk = [v for v in raw[lo : k + 1] if v is not None]
            if not chunk:
                smoothed.append(None)
                continue
            avg_x = sum(v[1] for v in chunk) / len(chunk)
            smoothed.append((raw[k][0] if raw[k] is not None else frames[k].t, avg_x))

        # resample into contiguous ~_SEGMENT_S segments
        seg_samples: list[tuple[float, float]] = []
        seg_start_t: float | None = None
        for k, s in enumerate(smoothed):
            if s is None:
                if seg_samples:
                    detections.append(_make_detection(team, seg_samples))
                    seg_samples = []
                    seg_start_t = None
                continue
            t, x = s
            if seg_start_t is None:
                seg_start_t = t
            seg_samples.append((t, x))
            if t - seg_start_t >= _SEGMENT_S:
                detections.append(_make_detection(team, seg_samples))
                seg_samples = []
                seg_start_t = None
        if seg_samples:
            detections.append(_make_detection(team, seg_samples))

    detections.sort(key=lambda d: d.t_start)
    return detections


def _make_detection(team: str, samples: list[tuple[float, float]]) -> Detection:
    final_x = samples[-1][1]
    return Detection(
        type="offside_line",
        players=[],
        geometry={
            "x": final_x,
            "team_defending": team,
            "approximate": True,
            "samples": samples,
        },
        confidence=0.5,  # always approximate -- see SPEC S9
        t_start=samples[0][0],
        t_end=samples[-1][0],
    )


if __name__ == "__main__":
    from bto.core import PlayerPos

    # Synthetic line of 5 'away' defenders at increasing x. Away attacks
    # toward x=0 (attacking=-1), so away's own goal is at the x=105 end:
    # "rearmost" (closest to own goal) = largest x = the GK at x=25.
    # Second-rearmost should be the defender at x=20.
    dt = 0.04
    n = 30
    xs = [2.0, 10.0, 15.0, 20.0, 25.0]  # ..., 2nd-rearmost=20, GK(rearmost)=25
    frames = []
    for k in range(n):
        t = k * dt
        players = [PlayerPos(f"h{i}", "home", 60.0 + i, 34.0) for i in range(3)]
        players += [PlayerPos(f"a{i}", "away", x, 34.0) for i, x in enumerate(xs)]
        frames.append(
            Frame(t=t, players=players, ball=(50.0, 34.0), attacking={"home": 1, "away": -1})
        )

    dets = offside_line(frames, smooth_s=0.5)
    away_dets = [d for d in dets if d.geometry["team_defending"] == "away"]
    assert away_dets, "expected offside line detections for away"
    for d in away_dets:
        assert abs(d.geometry["x"] - 20.0) < 1e-6, d.geometry
        assert d.geometry["approximate"] is True
        assert d.type == "offside_line"

    print("offside.py self-check OK:", [(d.geometry["team_defending"], d.geometry["x"], round(d.t_start, 2), round(d.t_end, 2)) for d in dets])
