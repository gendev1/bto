"""NvN local matchups + 1v1 isolation detection (SPEC S6).

Pure geometry on the coordinate stream; detect_isolations additionally uses
possession() to know who the ball carrier is.

geometry keys:
  detect_matchups:   pairs=[(xa, ya, xd, yd), ...] closest cross-team links
                      (first-team coords first: the attacking team when
                      possession() covers the frame, else HOME);
                      region=(min_x, min_y, max_x, max_y) bounding box of
                      every player involved.
  detect_isolations: attacker=(x, y), defender=(x, y).
"""

from math import hypot

from bto.core import AWAY, Detection, Frame, HOME, other
from bto.patterns.possession import Spell, possession


def _attacking_team(spells: list[Spell], frame_idx: int) -> str | None:
    """Team in possession at frame_idx per possession spells, else None."""
    for s in spells:
        if s.i_start <= frame_idx <= s.i_end:
            return s.team
    return None


def _components(home, away, radius):
    """Bipartite proximity connected components.

    home, away: list[PlayerPos]. Edges only between opposite teams (distance
    < radius); components are then read off via union-find. Returns
    [(home_subset, away_subset), ...] for components that contain at least
    one player from each team (pure single-team clusters are not matchups).
    """
    nodes = [("h", p) for p in home] + [("a", p) for p in away]
    n = len(nodes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(home)):
        pi = home[i]
        for j in range(len(home), n):
            pj = nodes[j][1]
            if hypot(pi.x - pj.x, pi.y - pj.y) < radius:
                union(i, j)

    groups: dict[int, tuple[list, list]] = {}
    for i, (side, p) in enumerate(nodes):
        h_list, a_list = groups.setdefault(find(i), ([], []))
        (h_list if side == "h" else a_list).append(p)
    return [(h, a) for h, a in groups.values() if h and a]


def _pairs_geometry(first, second):
    """Closest cross-team links, first-team coords listed first per pair."""
    pairs = []
    seen = set()
    for p in first:
        q = min(second, key=lambda q: hypot(p.x - q.x, p.y - q.y))
        key = (p.track_id, q.track_id)
        if key not in seen:
            seen.add(key)
            pairs.append((p.x, p.y, q.x, q.y))
    for q in second:
        p = min(first, key=lambda p: hypot(p.x - q.x, p.y - q.y))
        key = (p.track_id, q.track_id)
        if key not in seen:
            seen.add(key)
            pairs.append((p.x, p.y, q.x, q.y))
    return pairs


def _region(players):
    xs = [p.x for p in players]
    ys = [p.y for p in players]
    return (min(xs), min(ys), max(xs), max(ys))


def _matchup_confidence(first, second, radius):
    """Tighter cross-team spacing -> higher confidence, in [0, 1]."""
    dists = [hypot(p.x - q.x, p.y - q.y) for p in first for q in second]
    avg = sum(dists) / len(dists)
    return max(0.0, min(1.0, 1.0 - avg / radius))


def _candidates(frame: Frame, radius: float, n_max: int, attacking_team):
    home = frame.team_players(HOME)
    away = frame.team_players(AWAY)
    out = []
    for h_list, a_list in _components(home, away, radius):
        n, m = len(h_list), len(a_list)
        if max(n, m) > n_max or abs(n - m) > 1:
            continue
        if attacking_team == AWAY:
            first, second, type_ = a_list, h_list, f"{m}v{n}"
        else:
            first, second, type_ = h_list, a_list, f"{n}v{m}"
        players = frozenset(p.track_id for p in h_list + a_list)
        out.append((players, type_, first, second))
    return out


def detect_matchups(
    frames: list[Frame], radius: float = 8.0, n_max: int = 3, min_duration_s: float = 1.0
) -> list[Detection]:
    """NvN local matchups: bipartite proximity connected components.

    A component with n home / m away players (max(n, m) <= n_max, sizes
    differing by at most 1) is a candidate matchup for that frame. Runs of
    consecutive frames sharing the exact same player set and type are merged
    into one Detection; matchups shorter than min_duration_s are dropped.
    """
    spells = possession(frames)
    active: dict[frozenset, dict] = {}
    detections: list[Detection] = []

    def close(key):
        st = active.pop(key)
        if st["t_end"] - st["t_start"] >= min_duration_s:
            confidences = st["confidences"]
            detections.append(
                Detection(
                    type=st["type"],
                    players=sorted(key),
                    geometry=st["geometry"],
                    confidence=sum(confidences) / len(confidences),
                    t_start=st["t_start"],
                    t_end=st["t_end"],
                )
            )

    for i, frame in enumerate(frames):
        attacker = _attacking_team(spells, i)
        seen_keys = set()
        for key, type_, first, second in _candidates(frame, radius, n_max, attacker):
            seen_keys.add(key)
            geometry = {
                "pairs": _pairs_geometry(first, second),
                "region": _region(first + second),
            }
            confidence = _matchup_confidence(first, second, radius)
            st = active.get(key)
            if st is not None and st["type"] == type_:
                st["t_end"] = frame.t
                st["geometry"] = geometry
                st["confidences"].append(confidence)
            else:
                if st is not None:
                    close(key)
                active[key] = {
                    "type": type_,
                    "t_start": frame.t,
                    "t_end": frame.t,
                    "geometry": geometry,
                    "confidences": [confidence],
                }
        for key in list(active):
            if key not in seen_keys:
                close(key)
    for key in list(active):
        close(key)
    detections.sort(key=lambda d: d.t_start)
    return detections


def detect_isolations(
    frames: list[Frame], iso_radius: float = 10.0, engage_dist: float = 5.0
) -> list[Detection]:
    """1v1 isolation: ball carrier tight-marked with empty space around.

    Per frame (only frames covered by a possession() spell, so the carrier is
    known): find the carrier's nearest opponent. If that defender is closer
    than engage_dist AND no other player (either team, excluding the two) is
    within iso_radius of the carrier, it's an isolation. Consecutive frames
    with the same (attacker, defender) pair merge into one Detection.
    """
    spells = possession(frames)
    active: dict | None = None
    detections: list[Detection] = []

    def close():
        nonlocal active
        confidences = active["confidences"]
        detections.append(
            Detection(
                type="isolation",
                players=[active["attacker"], active["defender"]],
                geometry=active["geometry"],
                confidence=sum(confidences) / len(confidences),
                t_start=active["t_start"],
                t_end=active["t_end"],
            )
        )
        active = None

    for i, frame in enumerate(frames):
        carrier_id = None
        for s in spells:
            if s.i_start <= i <= s.i_end:
                carrier_id, carrier_team = s.track_id, s.team
                break

        pair = None
        geometry = None
        confidence = None
        if carrier_id is not None:
            carrier = next(p for p in frame.players if p.track_id == carrier_id)
            defenders = frame.team_players(other(carrier_team))
            if defenders:
                defender = min(
                    defenders, key=lambda p: hypot(p.x - carrier.x, p.y - carrier.y)
                )
                d_dist = hypot(defender.x - carrier.x, defender.y - carrier.y)
                others = [
                    p
                    for p in frame.players
                    if p.track_id not in (carrier.track_id, defender.track_id)
                ]
                other_dists = [hypot(p.x - carrier.x, p.y - carrier.y) for p in others]
                nearest_other = min(other_dists) if other_dists else float("inf")
                if d_dist < engage_dist and nearest_other >= iso_radius:
                    pair = (carrier.track_id, defender.track_id)
                    geometry = {
                        "attacker": (carrier.x, carrier.y),
                        "defender": (defender.x, defender.y),
                    }
                    space = min(nearest_other, iso_radius * 3.0)
                    confidence = max(0.0, min(1.0, (space - iso_radius) / iso_radius))

        if active is not None and pair != active["pair"]:
            close()
        if pair is not None:
            if active is None:
                active = {
                    "pair": pair,
                    "attacker": pair[0],
                    "defender": pair[1],
                    "t_start": frame.t,
                    "t_end": frame.t,
                    "geometry": geometry,
                    "confidences": [confidence],
                }
            else:
                active["t_end"] = frame.t
                active["geometry"] = geometry
                active["confidences"].append(confidence)
    if active is not None:
        close()
    detections.sort(key=lambda d: d.t_start)
    return detections


if __name__ == "__main__":
    from bto.core import PlayerPos

    # Synthetic self-check: a 2v2 duel on the right wing (present the whole
    # time), a clean 1v1 isolation on the left while the ball starts there,
    # then the ball moves into a crowded midfield that must NOT fire as an
    # isolation even though a carrier is tightly marked there too.
    frames = []
    for k in range(30):
        t = k * 0.04  # 25 Hz
        ball_at_h3 = k < 15
        players = [
            # --- wing 2v2 (around x=90, y=10), stays tight all 30 frames ---
            PlayerPos("H1", HOME, 90.0, 8.0),
            PlayerPos("H2", HOME, 92.0, 12.0),
            PlayerPos("A1", AWAY, 91.0, 9.0),
            PlayerPos("A2", AWAY, 91.5, 11.0),
            # --- lone isolation duel far away (x=10, y=60) ---
            PlayerPos("H3", HOME, 10.0, 60.0),
            PlayerPos("A3", AWAY, 12.0, 60.0),  # 2m away: engaged, < engage_dist
            # --- crowded midfield (x=52, y=34) ---
            PlayerPos("H4", HOME, 50.0, 34.0),
            PlayerPos("A4", AWAY, 52.0, 34.0),
            PlayerPos("H5", HOME, 54.0, 35.0),  # within iso_radius of carrier
            PlayerPos("A5", AWAY, 48.0, 33.0),
        ]
        # frames 0-14: ball with H3, isolated on the far side of the pitch.
        # frames 15-29: ball moves into the crowded midfield (A4 carries).
        ball = (10.0, 60.0) if ball_at_h3 else (52.0, 34.0)
        frames.append(
            Frame(t=t, players=players, ball=ball, attacking={HOME: 1, AWAY: -1})
        )

    matchups = detect_matchups(frames, radius=8.0, n_max=3, min_duration_s=1.0)
    wing_2v2 = [d for d in matchups if d.type in ("2v2",) and "H1" in d.players]
    assert wing_2v2, f"expected a 2v2 on the wing, got {[d.type for d in matchups]}"
    d = wing_2v2[0]
    assert set(d.players) == {"H1", "H2", "A1", "A2"}
    assert d.t_end - d.t_start >= 1.0 - 1e-9
    assert "pairs" in d.geometry and "region" in d.geometry
    assert len(d.geometry["pairs"]) >= 2

    # H4/A4 are also a 1v1 proximity component; make sure the crowded
    # midfield group is dropped by the n_max/component test only if it forms
    # a bigger component (here H4-A4-H5-A5 are all mutually close so it's a
    # 2v2, not part of this isolation check) -- just confirm no crash and
    # that isolation logic (below) correctly excludes it.

    isolations = detect_isolations(frames, iso_radius=10.0, engage_dist=5.0)
    iso_players = [set(d.players) for d in isolations]
    assert {"H3", "A3"} in iso_players, f"expected H3/A3 isolation, got {iso_players}"
    for d in isolations:
        assert "A4" not in d.players and "H4" not in d.players, (
            "crowded midfield carrier must not fire as an isolation"
        )
    iso = next(d for d in isolations if set(d.players) == {"H3", "A3"})
    assert iso.confidence > 0.5
    assert iso.geometry["attacker"] == (10.0, 60.0) or iso.geometry["defender"] == (
        10.0,
        60.0,
    )

    print("OK", [d.type for d in matchups], [d.players for d in isolations])
