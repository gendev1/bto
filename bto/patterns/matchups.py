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

Precision gates (M3 precision pass):
  detect_matchups: confidence is scaled by ball proximity (a duel far from
      the live ball is not a duel -- audited FPs were 9-40m off the ball),
      detections whose lifetime-min ball distance exceeds ball_hard_gate_m
      or whose peak per-frame confidence never reaches min_confidence are
      dropped, and an active matchup is force-closed after max_active_s
      (with a reopen cooldown) so the rendered snapshot can never go stale
      for 8+ seconds.
  detect_isolations: the nearest defender must be a different track_id than
      the carrier AND more than 0.3m away (tracker id duplication paired
      players against themselves at distance 0), the condition must hold for
      min_duration_s before one merged Detection is emitted (kills
      single-frame flicker spam), and re-emissions of the same pair within
      cooldown_s of the previous emission are merged (gap < merge_gap_s) or
      dropped.
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
    frames: list[Frame],
    radius: float = 8.0,
    n_max: int = 3,
    min_duration_s: float = 1.0,
    ball_soft_m: float = 20.0,
    ball_hard_gate_m: float = 25.0,
    min_confidence: float = 0.1,
    max_active_s: float = 3.0,
    reopen_cooldown_s: float = 4.0,
) -> list[Detection]:
    """NvN local matchups: bipartite proximity connected components.

    A component with n home / m away players (max(n, m) <= n_max, sizes
    differing by at most 1) is a candidate matchup for that frame. Runs of
    consecutive frames sharing the exact same player set and type are merged
    into one Detection.

    Precision gates:
    - per-frame confidence = spacing confidence * ball_factor, where
      ball_factor = clip(1 - d_ball / ball_soft_m, 0.05, 1.0) and d_ball is
      the distance from the live ball to the component's nearest player
      (0.3 neutral when the ball is unseen). Two markers jogging together
      30m off the ball score ~0 and die at the min_confidence floor.
    - detections observed together with the ball but NEVER within
      ball_hard_gate_m of it over their whole life are dropped outright
      (a matchup whose ball is unseen for its entire life is only soft-gated
      via the neutral factor -- clean data has dead-ball gaps too).
    - matchups shorter than min_duration_s, or whose PEAK per-frame
      confidence never reaches min_confidence (i.e. at no moment did they
      look like a tight duel near the ball), are dropped. The reported
      confidence stays the mean, so a duel that drifted off the ball carries
      a low viewer-facing score even though it survived the gate.
    - an active matchup is force-closed once it has lived max_active_s and
      the same player set may not reopen for reopen_cooldown_s (bounds how
      stale the single rendered geometry snapshot can get).
    """
    spells = possession(frames)
    active: dict[frozenset, dict] = {}
    cooldown_until: dict[frozenset, float] = {}
    detections: list[Detection] = []

    def close(key):
        st = active.pop(key)
        confidences = st["confidences"]
        mean_conf = sum(confidences) / len(confidences)
        if (
            st["t_end"] - st["t_start"] >= min_duration_s
            and max(confidences) >= min_confidence
            and (st["min_d_ball"] is None or st["min_d_ball"] <= ball_hard_gate_m)
        ):
            detections.append(
                Detection(
                    type=st["type"],
                    players=sorted(key),
                    geometry=st["geometry"],
                    confidence=mean_conf,
                    t_start=st["t_start"],
                    t_end=st["t_end"],
                )
            )

    for i, frame in enumerate(frames):
        attacker = _attacking_team(spells, i)
        seen_keys = set()
        ball = frame.ball
        for key, type_, first, second in _candidates(frame, radius, n_max, attacker):
            if frame.t < cooldown_until.get(key, float("-inf")):
                continue  # recently force-closed: let it reopen fresh later
            if ball is not None:
                d_ball = min(
                    hypot(p.x - ball[0], p.y - ball[1]) for p in first + second
                )
                ball_factor = max(0.05, min(1.0, 1.0 - d_ball / ball_soft_m))
            else:
                d_ball = None  # unseen ball: no evidence either way
                ball_factor = 0.3  # neutral: don't reward or kill on a ball gap
            geometry = {
                "pairs": _pairs_geometry(first, second),
                "region": _region(first + second),
            }
            confidence = _matchup_confidence(first, second, radius) * ball_factor
            st = active.get(key)
            if st is not None and st["type"] == type_:
                if frame.t - st["t_start"] > max_active_s:
                    # stale-geometry cap: emit what we have, cool down.
                    close(key)
                    cooldown_until[key] = frame.t + reopen_cooldown_s
                    continue
                seen_keys.add(key)
                st["t_end"] = frame.t
                st["geometry"] = geometry
                st["confidences"].append(confidence)
                if d_ball is not None:
                    st["min_d_ball"] = (
                        d_ball
                        if st["min_d_ball"] is None
                        else min(st["min_d_ball"], d_ball)
                    )
            else:
                if st is not None:
                    close(key)
                seen_keys.add(key)
                active[key] = {
                    "type": type_,
                    "t_start": frame.t,
                    "t_end": frame.t,
                    "geometry": geometry,
                    "confidences": [confidence],
                    "min_d_ball": d_ball,
                }
        for key in list(active):
            if key not in seen_keys:
                close(key)
    for key in list(active):
        close(key)
    detections.sort(key=lambda d: d.t_start)
    return detections


def detect_isolations(
    frames: list[Frame],
    iso_radius: float = 10.0,
    engage_dist: float = 5.0,
    min_duration_s: float = 0.3,
    carrier_ball_max: float = 3.0,
    min_engage_dist: float = 0.3,
    cooldown_s: float = 8.0,
    merge_gap_s: float = 1.0,
) -> list[Detection]:
    """1v1 isolation: ball carrier tight-marked with empty space around.

    Per frame (only frames covered by a possession() spell, so the carrier is
    known): find the carrier's nearest opponent. If that defender is closer
    than engage_dist AND no other player (either team, excluding the two) is
    within iso_radius of the carrier, it's an isolation. Consecutive frames
    with the same (attacker, defender) pair merge into one Detection; pairs
    that don't sustain for min_duration_s are dropped (temporal hysteresis
    against single-frame flicker). The defender can never share the carrier's
    track_id and must stand at least min_engage_dist away (a 'defender' at
    0.0m is the carrier under a duplicated tracker id). Once a pair has been
    SEEN (emitted or not), a re-detection of the SAME unordered pair starting
    within cooldown_s of that sighting's end is merged into the previous
    emission when the gap is under merge_gap_s, else dropped -- one sustained
    duel, one chip, and a pair that keeps flickering in and out (tracker
    noise) can't re-chip every few seconds.
    """
    spells = possession(frames)
    active: dict | None = None
    detections: list[Detection] = []
    # pair -> [t_end of last sighting, detections-index of last emission|None]
    last_seen: dict[frozenset, list] = {}

    def close():
        nonlocal active
        confidences = active["confidences"]
        pair_key = frozenset(active["pair"])
        prev = last_seen.get(pair_key)
        gap = active["t_start"] - prev[0] if prev is not None else float("inf")
        prev_i = prev[1] if prev is not None else None
        if prev_i is not None and gap < merge_gap_s:
            # same duel resurfacing almost immediately: extend, don't re-chip
            old = detections[prev_i]
            merged_conf = old.confidence * max(old.t_end - old.t_start, 1e-9)
            merged_conf += (sum(confidences) / len(confidences)) * max(
                active["t_end"] - active["t_start"], 1e-9
            )
            span = max(old.t_end - old.t_start, 1e-9) + max(
                active["t_end"] - active["t_start"], 1e-9
            )
            detections[prev_i] = Detection(
                type="isolation",
                players=old.players,
                geometry=active["geometry"],
                confidence=merged_conf / span,
                t_start=old.t_start,
                t_end=active["t_end"],
            )
            last_seen[pair_key] = [active["t_end"], prev_i]
        elif prev is not None and gap < cooldown_s:
            # cooldown: this pair chipped (or flickered) moments ago. Drop it
            # and clear the emission index so a later segment can't merge
            # backward across the dropped gap into a long-dead chip.
            last_seen[pair_key] = [active["t_end"], None]
        elif active["t_end"] - active["t_start"] >= min_duration_s:
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
            last_seen[pair_key] = [active["t_end"], len(detections) - 1]
        else:
            # too short to chip, but remember the sighting: sub-threshold
            # flicker must still hold the cooldown for this pair
            last_seen[pair_key] = [active["t_end"], None]
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
        # the spell's original tid can be absent from a frame (tid swap
        # mid-spell, or occlusion during a bridged ball gap) -> skip frame
        carrier = None
        if carrier_id is not None:
            carrier = next(
                (p for p in frame.players if p.track_id == carrier_id), None
            )
        # spell tid can go stale across tracker id swaps: only trust frames
        # where the looked-up carrier is actually on the ball.
        if carrier is not None and (
            frame.ball is None
            or hypot(carrier.x - frame.ball[0], carrier.y - frame.ball[1])
            > carrier_ball_max
        ):
            carrier = None
        if carrier is not None:
            defenders = [
                p
                for p in frame.team_players(other(carrier_team))
                if p.track_id != carrier.track_id  # never pair a player with himself
            ]
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
                if min_engage_dist < d_dist < engage_dist and nearest_other >= iso_radius:
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

    # Synthetic self-check: a clean 1v1 isolation on the left while the ball
    # starts there, then the ball moves to a tight 2v2 duel on the right wing
    # (present the whole time, but it must only fire as a matchup WHILE the
    # ball is there -- the ball-proximity gate).
    frames = []
    for k in range(55):
        t = k * 0.04  # 25 Hz
        ball_at_h3 = k < 25
        players = [
            # --- wing 2v2 (around x=90, y=10), stays tight all 55 frames ---
            PlayerPos("H1", HOME, 90.0, 8.0),
            PlayerPos("H2", HOME, 92.0, 12.0),
            PlayerPos("A1", AWAY, 91.0, 9.0),
            PlayerPos("A2", AWAY, 91.5, 11.0),
            # --- lone isolation duel far away (x=10, y=60) ---
            PlayerPos("H3", HOME, 10.0, 60.0),
            PlayerPos("A3", AWAY, 12.0, 60.0),  # 2m away: engaged, < engage_dist
            # --- crowded midfield (x=52, y=34), never near the ball ---
            PlayerPos("H4", HOME, 50.0, 34.0),
            PlayerPos("A4", AWAY, 52.0, 34.0),
            PlayerPos("H5", HOME, 54.0, 35.0),  # within iso_radius of carrier
            PlayerPos("A5", AWAY, 48.0, 33.0),
        ]
        # frames 0-24: ball with H3, isolated on the far side of the pitch.
        # frames 25-54: ball moves to the wing 2v2.
        ball = (10.0, 60.0) if ball_at_h3 else (91.0, 10.0)
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

    # The crowded midfield H4-A4-H5-A5 is also a tight 2v2 component, but the
    # ball NEVER comes near it -- the ball hard gate must suppress it (this
    # was the audited FP mode: cross-team clusters 9-40m off the ball).
    assert not any("H4" in d.players for d in matchups), (
        f"off-ball midfield 2v2 must be suppressed, got {[(d.type, d.players) for d in matchups]}"
    )

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
