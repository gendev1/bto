"""Formation shape + defensive block detectors (SPEC S6 'Formation shape';
socker-plays-ref Easy rows 'Compact block' and 'Defensive line height').

Both detectors work per team on rolling (tumbling) time windows: player
positions are averaged over the window, the rearmost player on the team's
attacking axis is dropped as the GK, and the remaining outfield players are
clustered into 2-4 lines by 1-D gap clustering on attacking-axis depth
(largest gaps split lines).

Depth convention: attacking-axis meters, 0 = own goal line,
PITCH_LENGTH = opponent goal line (i.e. depth = x when attacking +1,
105 - x when attacking -1).

Geometry keys (pitch meters unless noted):

type='formation' (one Detection per team per window):
  label      : formation string ordered defense->attack, e.g. '4-3-3'
  lines      : list of lines defense->attack, each a list of (x, y) mean
               outfield player positions (raw pitch coords)
  hull       : convex hull of the outfield mean positions, CCW (x, y) list
  shift_from : previous window's label; present only when the label changed

type='block' (one Detection per team per window):
  width       : max - min y of outfield players
  depth       : max - min attacking-axis depth of outfield players
  line_height : mean attacking-axis depth of the rearmost outfield line
  line_x      : same rearmost line as mean raw pitch x (renderer convenience)
  block       : 'low' | 'mid' | 'high' -- line_height by thirds of the pitch

Confidence is min(1, n_outfield / 10): 1.0 with all 10 outfield players
visible, scaling down linearly when broadcast tracking loses players.
"""

import numpy as np

from bto.core import AWAY, HOME, PITCH_LENGTH, Detection, Frame


def _windows(frames: list[Frame], window_s: float):
    """Yield (i0, i1) index ranges chunking frames into window_s buckets."""
    i0 = 0
    n = len(frames)
    while i0 < n:
        end_t = frames[i0].t + window_s
        i1 = i0 + 1
        while i1 < n and frames[i1].t < end_t:
            i1 += 1
        yield i0, i1
        i0 = i1


def _team_window(
    frames: list[Frame],
    i0: int,
    i1: int,
    team: str,
    min_presence: float = 0.5,
    max_players: int = 11,
):
    """Mean position per track over frames[i0:i1].

    Returns (ids, xy, depth) with xy an (n, 2) array of mean pitch positions
    and depth the attacking-axis projection, both sorted by depth ascending
    (rearmost first). ids is the matching track_id list.

    Ghost/duplicate tracker ids (broadcast id churn) are filtered out: a track
    must be present in at least min_presence of the window's frames, and at
    most max_players tracks (the most-present ones) are kept -- an 11-a-side
    team can never contribute more than 11 real players to a window.
    """
    sums: dict[str, list[float]] = {}
    for f in frames[i0:i1]:
        for p in f.players:
            if p.team != team:
                continue
            s = sums.setdefault(p.track_id, [0.0, 0.0, 0.0])
            s[0] += p.x
            s[1] += p.y
            s[2] += 1.0
    n_frames = max(i1 - i0, 1)
    sums = {tid: s for tid, s in sums.items() if s[2] / n_frames >= min_presence}
    if len(sums) > max_players:
        keep = sorted(sums, key=lambda tid: (-sums[tid][2], tid))[:max_players]
        sums = {tid: sums[tid] for tid in keep}
    if not sums:
        return [], np.empty((0, 2)), np.empty(0)
    ids = sorted(sums)
    xy = np.array([[sums[i][0] / sums[i][2], sums[i][1] / sums[i][2]] for i in ids])
    sign = frames[i0].attacking.get(team, 1)
    depth = xy[:, 0] * sign + (0.0 if sign > 0 else PITCH_LENGTH)
    order = np.argsort(depth)
    return [ids[i] for i in order], xy[order], depth[order]


def _line_splits(depth: np.ndarray, min_gap: float, max_lines: int = 4) -> list[int]:
    """Split points into lines at the largest depth gaps.

    depth must be sorted ascending. Returns sorted split indices s such that
    lines are depth[0:s1], depth[s1:s2], ... Always at least one split when
    n >= 2 (2-4 lines), never more than max_lines - 1.
    """
    gaps = np.diff(depth)
    if len(gaps) == 0:
        return []
    cuts = np.where(gaps >= min_gap)[0]
    if len(cuts) == 0:
        cuts = np.array([np.argmax(gaps)])
    elif len(cuts) > max_lines - 1:
        cuts = cuts[np.argsort(gaps[cuts])[-(max_lines - 1) :]]
    return sorted(int(c) + 1 for c in cuts)


def _lines(xy: np.ndarray, depth: np.ndarray, min_gap: float) -> list[np.ndarray]:
    """Group depth-sorted outfield positions into 2-4 lines, defense->attack."""
    splits = _line_splits(depth, min_gap)
    bounds = [0] + splits + [len(depth)]
    return [xy[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def _hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Convex hull, monotone chain. Returns CCW vertices (no repeat)."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def detect_formation(
    frames: list[Frame],
    window_s: float = 5.0,
    min_line_gap: float = 5.0,
    min_outfield: int = 6,
) -> list[Detection]:
    """Formation shape per team per window; see module docstring for geometry.

    A window only emits when at least min_outfield stable outfield tracks are
    visible after GK drop -- a formation label built from a handful of players
    (or from ghost-track soup) is meaningless to a viewer.
    """
    out: list[Detection] = []
    for team in (HOME, AWAY):
        prev_label: str | None = None
        for i0, i1 in _windows(frames, window_s):
            ids, xy, depth = _team_window(frames, i0, i1, team)
            if len(ids) < min_outfield + 1:  # GK + min_outfield outfield players
                continue
            ids, xy, depth = ids[1:], xy[1:], depth[1:]  # drop rearmost = GK
            lines = _lines(xy, depth, min_line_gap)
            label = "-".join(str(len(ln)) for ln in lines)
            geometry = {
                "label": label,
                "lines": [
                    [(float(x), float(y)) for x, y in ln[np.argsort(ln[:, 1])]]
                    for ln in lines
                ],
                "hull": _hull(xy),
            }
            if prev_label is not None and label != prev_label:
                geometry["shift_from"] = prev_label
            prev_label = label
            out.append(
                Detection(
                    type="formation",
                    players=ids,
                    geometry=geometry,
                    confidence=min(1.0, len(ids) / 10.0),
                    t_start=frames[i0].t,
                    t_end=frames[i1 - 1].t,
                )
            )
    return out


def detect_block(
    frames: list[Frame],
    window_s: float = 2.0,
    min_line_gap: float = 5.0,
    min_outfield: int = 6,
) -> list[Detection]:
    """Block compactness + line height per team per window (both teams --
    possession is not known here); see module docstring for geometry.

    Like detect_formation, a window only emits with at least min_outfield
    stable outfield tracks after GK drop -- 'block' geometry from 2-3 players
    on a close-up frame is noise, not team shape.
    """
    third = PITCH_LENGTH / 3.0
    out: list[Detection] = []
    for team in (HOME, AWAY):
        for i0, i1 in _windows(frames, window_s):
            ids, xy, depth = _team_window(frames, i0, i1, team)
            if len(ids) < min_outfield + 1:  # GK + min_outfield outfield
                continue
            ids, xy, depth = ids[1:], xy[1:], depth[1:]  # drop rearmost = GK
            splits = _line_splits(depth, min_line_gap)
            k = splits[0] if splits else len(depth)  # rearmost line size
            line_height = float(depth[:k].mean())
            block = "low" if line_height < third else "mid" if line_height < 2 * third else "high"
            geometry = {
                "width": float(xy[:, 1].max() - xy[:, 1].min()),
                "depth": float(depth[-1] - depth[0]),
                "line_height": line_height,
                "line_x": float(xy[:k, 0].mean()),
                "block": block,
            }
            out.append(
                Detection(
                    type="block",
                    players=ids,
                    geometry=geometry,
                    confidence=min(1.0, len(ids) / 10.0),
                    t_start=frames[i0].t,
                    t_end=frames[i1 - 1].t,
                )
            )
    return out


if __name__ == "__main__":
    from bto.core import PlayerPos

    def team_players(prefix: str, team: str, sign: int, line_spec):
        """line_spec: [(depth_m, n_players), ...] defense->attack. Adds a GK."""

        def x_of(d):
            return d if sign > 0 else PITCH_LENGTH - d

        players = [PlayerPos(f"{prefix}gk", team, x_of(5.0), 34.0)]
        for li, (d, n) in enumerate(line_spec):
            for pi, y in enumerate(np.linspace(12.0, 56.0, n)):
                players.append(PlayerPos(f"{prefix}{li}{pi}", team, x_of(d), float(y)))
        return players

    home_442 = team_players("H", HOME, 1, [(20, 4), (40, 4), (60, 2)])
    home_433 = team_players("H", HOME, 1, [(20, 4), (45, 3), (65, 3)])
    away_433 = team_players("A", AWAY, -1, [(20, 4), (40, 3), (60, 3)])
    attacking = {HOME: 1, AWAY: -1}

    frames = [
        Frame(
            t=i / 25.0,
            players=(home_442 if i < 125 else home_433) + away_433,
            ball=None,
            attacking=attacking,
        )
        for i in range(250)  # two 5 s windows at 25 Hz
    ]

    dets = detect_formation(frames)
    home = [d for d in dets if d.players[0].startswith("H")]
    away = [d for d in dets if d.players[0].startswith("A")]
    assert [d.geometry["label"] for d in home] == ["4-4-2", "4-3-3"], home
    assert [d.geometry["label"] for d in away] == ["4-3-3", "4-3-3"], away
    assert home[1].geometry["shift_from"] == "4-4-2"
    assert "shift_from" not in home[0].geometry
    assert all(len(d.geometry["hull"]) >= 4 for d in dets)
    assert all(d.confidence == 1.0 for d in dets)
    assert home[0].t_start == 0.0 and home[0].t_end == 124 / 25.0
    assert len(home[0].players) == 10 and "Hgk" not in home[0].players

    blocks = detect_block(frames[:125])
    hb = next(d for d in blocks if d.players[0].startswith("H"))
    assert hb.geometry["block"] == "low", hb.geometry
    assert abs(hb.geometry["line_height"] - 20.0) < 1e-9
    assert abs(hb.geometry["line_x"] - 20.0) < 1e-9
    assert abs(hb.geometry["width"] - 44.0) < 1e-9
    assert abs(hb.geometry["depth"] - 40.0) < 1e-9

    # a team squeezed high up the pitch reads 'high'
    home_high = team_players("H", HOME, 1, [(75, 4), (85, 4), (95, 2)])
    hi_frames = [
        Frame(t=i / 25.0, players=home_high + away_433, ball=None, attacking=attacking)
        for i in range(60)
    ]
    hi = next(d for d in detect_block(hi_frames) if d.players[0].startswith("H"))
    assert hi.geometry["block"] == "high", hi.geometry

    # confidence scales down when players are missing (7 outfield -> 0.7)
    few = [
        Frame(t=i / 25.0, players=home_442[:8] + away_433, ball=None, attacking=attacking)
        for i in range(60)
    ]
    fd = next(d for d in detect_formation(few) if d.players[0].startswith("H"))
    assert abs(fd.confidence - 0.7) < 1e-9

    print("formation self-check OK")
