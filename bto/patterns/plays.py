"""Play grammar over roles + edges + possession (M6 relational layer).

Composes the primitive layers (possession spells, marking/press edges,
formation roles) into socker-plays-ref rows: overlap (upgraded with marker
lag), give-and-go, third-man run, pressing trap, and roles' rotations.

Every geometry carries a moving '*_path' ([(t, x, y), ...]) per the M6
render convention: the renderer interpolates at the current frame t so the
overlay follows the players instead of blinking.

detect_plays(frames, role_maps_home=None, role_maps_away=None, edges=None)
computes roles/edges itself when they are not injected; the integrator
injects shared computations. All logic is causal windowed geometry so the
live host can reuse it.
"""

from bisect import bisect_left
from math import hypot, inf

from bto.core import AWAY, HOME, PITCH_WIDTH, Detection, Frame, track_velocity
from bto.patterns.possession import possession

_COOLDOWN_S = 8.0  # same-type + same-players re-emit cooldown
_PATH_HZ = 5.0  # geometry path sampling rate
_MAX_SPEED = 12.0  # m/s; above elite sprint -> tracker tid teleport, reject


# ---------------------------------------------------------------- helpers
def _infer_dt(frames: list[Frame]) -> float:
    diffs = sorted(b.t - a.t for a, b in zip(frames, frames[1:]) if b.t > a.t)
    return diffs[len(diffs) // 2] if diffs else 0.04


def _pos(frame: Frame, track_id: str) -> tuple[float, float] | None:
    for p in frame.players:
        if p.track_id == track_id:
            return (p.x, p.y)
    return None


def _path(
    frames: list[Frame], track_id: str, i0: int, i1: int, dt: float
) -> list[tuple[float, float, float]]:
    """[(t, x, y)] for track_id over frames [i0, i1], sampled ~_PATH_HZ."""
    stride = max(1, round(1.0 / (_PATH_HZ * dt)))
    idxs = list(range(i0, i1 + 1, stride))
    if idxs and idxs[-1] != i1:
        idxs.append(i1)
    pts = []
    for k in idxs:
        p = _pos(frames[k], track_id)
        if p is not None:
            pts.append((frames[k].t, p[0], p[1]))
    return pts


def _index_at(ts: list[float], t: float) -> int:
    j = bisect_left(ts, t)
    if j <= 0:
        return 0
    if j >= len(ts):
        return len(ts) - 1
    return j if ts[j] - t < t - ts[j - 1] else j - 1


def _sep_at(samples: list[tuple], t: float) -> float:
    """Defender<->attacker distance at the edge sample nearest to t."""
    s = min(samples, key=lambda s: abs(s[0] - t))
    return hypot(s[1] - s[3], s[2] - s[4])


def _marker_growth(edges, t0: float, t1: float) -> float | None:
    """Marker-distance change over [t0, t1] on the strongest overlapping
    marking edge; None when no marking edge overlaps the window."""
    best = None
    for e in edges:
        if not e.samples or e.t_start > t1 or e.t_end < t0:
            continue
        if best is None or e.strength > best.strength:
            best = e
    if best is None:
        return None
    return _sep_at(best.samples, t1) - _sep_at(best.samples, t0)


# ------------------------------------------------------------ give-and-go
def _give_and_gos(frames, spells, dt) -> list[Detection]:
    """Spell A -> spell B (same team, <2.5 s gap) -> A again (<4 s after B
    starts) with A >= 6 m further up the attacking axis on the return."""
    out = []
    for a, b, a2 in zip(spells, spells[1:], spells[2:]):
        if not (a.team == b.team == a2.team):
            continue
        if b.track_id == a.track_id or a2.track_id != a.track_id:
            continue
        if b.t_start - a.t_end > 2.5 or a2.t_start - b.t_start > 4.0:
            continue
        p0 = _pos(frames[a.i_end], a.track_id)
        p1 = _pos(frames[a2.i_start], a.track_id)
        if p0 is None or p1 is None:
            continue
        d = frames[a.i_end].attacking.get(a.team, 1)
        fwd = d * (p1[0] - p0[0])
        if fwd < 6.0:
            continue
        span = frames[a2.i_start].t - frames[a.i_end].t
        if span <= 0 or hypot(p1[0] - p0[0], p1[1] - p0[1]) / span > _MAX_SPEED:
            continue  # tid teleport, not a run
        out.append(
            Detection(
                type="give_and_go",
                players=[a.track_id, b.track_id],
                geometry={
                    "a_path": _path(frames, a.track_id, a.i_end, a2.i_start, dt),
                    "wall": _pos(frames[b.i_start], b.track_id),
                },
                confidence=min(0.9, 0.55 + fwd / 40.0),
                t_start=a.t_end,
                t_end=a2.t_start,
            )
        )
    return out


# ------------------------------------------------------------ overlap run
def _overlap_window(frames, i, j, rid, cid, d, vstep, half):
    """Speed of a qualifying overlap window, else None: R behind C at i,
    ahead at j, ends outside (wider y), sustained > 4 m/s throughout."""
    r0, c0 = _pos(frames[i], rid), _pos(frames[i], cid)
    r1, c1 = _pos(frames[j], rid), _pos(frames[j], cid)
    if None in (r0, c0, r1, c1):
        return None
    if d * r0[0] > d * c0[0] + 0.5:  # must start behind (or level with) C
        return None
    if d * r1[0] < d * c1[0] + 0.5:  # must end ahead of C
        return None
    ry, cy = r1[1], c1[1]
    if (ry - half) * (cy - half) < 0 or abs(ry - half) < abs(cy - half) + 0.5:
        return None  # must pass OUTSIDE: same side, wider than C
    ks = list(range(i + vstep, j + 1, vstep))
    if len(ks) < 3:
        return None
    speeds = []
    for k in ks:
        v = track_velocity(frames, k, rid)
        if v is None:
            return None
        s = hypot(*v)
        if s < 4.0 or s > _MAX_SPEED:  # sustained sprint, no teleports
            return None
        speeds.append(s)
    return sum(speeds) / len(speeds)


def _overlap_runs(frames, spells, mark_by, dt) -> list[Detection]:
    out = []
    win_n = max(2, round(1.5 / dt))
    step = max(1, win_n // 6)
    vstep = max(1, round(0.25 / dt))
    half = PITCH_WIDTH / 2.0
    for sp in spells:
        lo, hi = sp.i_start, sp.i_end
        if hi - lo < win_n:
            continue
        cid, team = sp.track_id, sp.team
        d = frames[lo].attacking.get(team, 1)
        mates = {
            p.track_id
            for f in frames[lo : hi + 1]
            for p in f.team_players(team)
        } - {cid}
        for rid in sorted(mates):
            for i in range(lo, hi - win_n + 1, step):
                j = i + win_n
                speed = _overlap_window(frames, i, j, rid, cid, d, vstep, half)
                if speed is None:
                    continue
                t0, t1 = frames[i].t, frames[j].t
                growth = _marker_growth(mark_by.get(rid, ()), t0, t1)
                if growth is not None and growth <= 2.0:
                    continue  # marker kept pace -> run achieved nothing
                conf = 0.5 if growth is None else min(0.85, 0.55 + growth / 10.0)
                out.append(
                    Detection(
                        type="overlap_run",
                        players=[rid, cid],
                        geometry={
                            "run_path": _path(frames, rid, i, j, dt),
                            "carrier": _pos(frames[j], cid),
                        },
                        confidence=conf,
                        t_start=t0,
                        t_end=t1,
                    )
                )
                break  # first qualifying window per (spell, runner)
    return out


# --------------------------------------------------------- third-man run
def _third_man_runs(frames, spells, mark_by, dt) -> list[Detection]:
    """Pass A->B while a third teammate C sprints forward > 4 m/s and C's
    marking edge dissolves / marker distance grows > 3 m. Hard row:
    confidence is capped at 0.6."""
    out = []
    vstep = max(1, round(0.25 / dt))
    pad = round(0.5 / dt)
    for a, b in zip(spells, spells[1:]):
        if a.team != b.team or a.track_id == b.track_id:
            continue
        if b.t_start - a.t_end > 2.5:
            continue
        i0 = a.i_end
        i1 = min(len(frames) - 1, max(b.i_start + pad, i0 + 1))
        t0, t1 = frames[i0].t, frames[i1].t
        d = frames[i0].attacking.get(a.team, 1)
        cands = {
            p.track_id
            for f in frames[i0 : i1 + 1]
            for p in f.team_players(a.team)
        } - {a.track_id, b.track_id}
        for cid in sorted(cands):
            vs = [track_velocity(frames, k, cid) for k in range(i0, i1 + 1, vstep)]
            vs = [v for v in vs if v is not None]
            if len(vs) < 2 or any(hypot(*v) > _MAX_SPEED for v in vs):
                continue
            fast_fwd = sum(1 for v in vs if d * v[0] > 4.0)
            if fast_fwd < max(2, len(vs) // 2):
                continue  # not accelerating forward through the pass
            medges = [
                e
                for e in mark_by.get(cid, ())
                if e.samples and e.t_start <= t1 and e.t_end >= t0 - 1.0
            ]
            if not medges:
                continue  # nobody was marking C -> no marker beaten
            # per edge: separation growth from t0 to (end of window or edge).
            # Edges are spell-bounded upstream, so a bare dissolve at the
            # pass moment proves nothing -- the marker must actually be
            # getting dropped (growing separation) for the edge to count.
            growth = max(
                _sep_at(e.samples, min(t1, e.t_end)) - _sep_at(e.samples, t0)
                for e in medges
            )
            dissolved = any(e.t_end <= t1 for e in medges)
            if not ((dissolved and growth > 1.5) or growth > 3.0):
                continue
            conf = min(0.6, 0.4 + growth / 15.0)
            path = _path(frames, cid, i0, i1, dt)
            if len(path) < 2:
                continue
            out.append(
                Detection(
                    type="third_man_run",
                    players=[a.track_id, b.track_id, cid],
                    geometry={
                        "run_path": path,
                        "pass_from": _pos(frames[i0], a.track_id),
                        "pass_to": _pos(frames[i1], b.track_id),
                    },
                    confidence=conf,
                    t_start=t0,
                    t_end=t1,
                )
            )
    return out


# ---------------------------------------------------------- pressing trap
def _pressing_traps(frames, spells, presses, mark_by, dt, ts) -> list[Detection]:
    """>= 3 press edges born within 1.5 s on one carrier while every
    teammate within 12 m is marked -> trap until the spell ends."""
    out = []
    by_carrier: dict[str, list] = {}
    for e in presses:
        by_carrier.setdefault(e.b, []).append(e)
    done: set[tuple[str, int]] = set()
    for cid, es in sorted(by_carrier.items()):
        es.sort(key=lambda e: e.t_start)
        for k in range(len(es)):
            group = [e for e in es if 0 <= e.t_start - es[k].t_start <= 1.5]
            if len(group) < 3:
                continue
            t_trap = es[k].t_start
            sp = next(
                (
                    s
                    for s in spells
                    if s.track_id == cid and s.t_start - 0.5 <= t_trap <= s.t_end
                ),
                None,
            )
            if sp is None or sp.t_end - t_trap < 0.4:
                continue
            key = (cid, sp.i_start)
            if key in done:
                continue
            i_trap = max(sp.i_start, _index_at(ts, t_trap))
            cpos = _pos(frames[i_trap], cid)
            if cpos is None:
                continue
            t_now = frames[i_trap].t
            open_mate = False
            for p in frames[i_trap].team_players(sp.team):
                if p.track_id == cid:
                    continue
                if hypot(p.x - cpos[0], p.y - cpos[1]) > 12.0:
                    continue
                if not any(
                    e.t_start <= t_now <= e.t_end
                    for e in mark_by.get(p.track_id, ())
                ):
                    open_mate = True  # a nearby unmarked outlet: no trap
                    break
            if open_mate:
                continue
            done.add(key)
            pressers = [e.a for e in group]
            out.append(
                Detection(
                    type="pressing_trap",
                    players=[cid] + pressers,
                    geometry={
                        "carrier_path": _path(frames, cid, i_trap, sp.i_end, dt),
                        "pressers": pressers,
                    },
                    confidence=min(0.8, 0.5 + 0.1 * (len(group) - 3)),
                    t_start=t_trap,
                    t_end=sp.t_end,
                )
            )
    return out


# ---------------------------------------------------------- rate limiting
def _rate_limit(dets: list[Detection]) -> list[Detection]:
    """Drop a detection when the same type fired for the same players
    within the last _COOLDOWN_S seconds."""
    dets = sorted(dets, key=lambda d: (d.t_start, d.type))
    last: dict[tuple, float] = {}
    kept = []
    for det in dets:
        key = (det.type, tuple(sorted(det.players)))
        if det.t_start - last.get(key, -inf) < _COOLDOWN_S:
            continue
        last[key] = det.t_start
        kept.append(det)
    return kept


# ------------------------------------------------------------- public API
def detect_plays(
    frames: list[Frame],
    role_maps_home=None,
    role_maps_away=None,
    edges=None,
) -> list[Detection]:
    """Play-grammar detections over roles + edges + possession.

    role_maps_home/role_maps_away/edges are injectable so the integrator
    can share computations; when role_maps are None assign_roles is run
    (and its rotation detections re-emitted), when edges is None
    track_edges is run.
    """
    if len(frames) < 2:
        return []
    dt = _infer_dt(frames)
    ts = [f.t for f in frames]
    dets: list[Detection] = []

    if role_maps_home is None and role_maps_away is None:
        from bto.patterns.roles import assign_roles  # sibling module

        for team in (HOME, AWAY):
            _, _, rotations = assign_roles(frames, team)
            dets.extend(rotations)
    if edges is None:
        from bto.patterns.edges import track_edges  # sibling module

        edges = track_edges(frames)

    mark_by: dict[str, list] = {}
    for e in edges:
        if e.kind == "marking":
            mark_by.setdefault(e.b, []).append(e)
    presses = [e for e in edges if e.kind == "press"]

    spells = possession(frames)
    dets += _give_and_gos(frames, spells, dt)
    dets += _overlap_runs(frames, spells, mark_by, dt)
    dets += _third_man_runs(frames, spells, mark_by, dt)
    dets += _pressing_traps(frames, spells, presses, mark_by, dt, ts)
    return _rate_limit(dets)


# --------------------------------------------------------------------------
# Self-check: scripted positives fire exactly once with the right players;
# scripted negatives stay silent. Then a real-Metrica sanity count.
if __name__ == "__main__":
    from dataclasses import dataclass, field

    from bto.core import PlayerPos

    DT = 0.04  # 25 Hz
    ATT = {HOME: 1, AWAY: -1}

    @dataclass
    class _E:  # stand-in matching the frozen edges.Edge contract
        kind: str
        team: str
        a: str
        b: str
        i_start: int
        i_end: int
        t_start: float
        t_end: float
        samples: list = field(default_factory=list)
        strength: float = 0.8

    def fr(t, players, ball):
        return Frame(t=t, players=players, ball=ball, attacking=ATT)

    def run(frames, edges=()):
        # inject role_maps/edges: siblings' modules may not exist yet
        return detect_plays(
            frames, role_maps_home=[], role_maps_away=[], edges=list(edges)
        )

    # --- 1) clean give-and-go: A passes to B, sprints 10 m forward, gets it
    # back. 2) wall pass where A stays put -> NOT a give-and-go.
    def one_two(a_moves: bool):
        frames = []
        n = round(3.8 / DT)
        for k in range(n):
            t = k * DT
            if not a_moves or t <= 1.0:
                ax, ay = 30.0, 34.0
            else:  # sprint (30,34)->(40,38) over 1.6 s, then hold
                f = min(1.0, (t - 1.0) / 1.6)
                ax, ay = 30.0 + 10.0 * f, 34.0 + 4.0 * f
            recv = (ax, ay)
            if t <= 1.0:
                ball = (30.0, 34.0)
            elif t <= 1.3:  # A -> B flight, 20 m/s
                ball = (30.0 + 20.0 * (t - 1.0), 34.0)
            elif t <= 2.4:
                ball = (36.0, 34.0)
            elif t <= 2.75:  # B -> back to A (wherever A now is)
                f = (t - 2.4) / 0.35
                ball = (36.0 + (recv[0] - 36.0) * f, 34.0 + (recv[1] - 34.0) * f)
            else:
                ball = recv
            frames.append(
                fr(
                    t,
                    [
                        PlayerPos("A", HOME, ax, ay),
                        PlayerPos("B", HOME, 36.0, 34.0),
                        PlayerPos("D1", AWAY, 80.0, 50.0),
                    ],
                    ball,
                )
            )
        return frames

    dets = run(one_two(a_moves=True))
    gg = [d for d in dets if d.type == "give_and_go"]
    assert len(gg) == 1 and gg[0].players == ["A", "B"], (gg, dets)
    assert len(gg[0].geometry["a_path"]) >= 2 and gg[0].confidence <= 0.9
    dets = run(one_two(a_moves=False))
    assert not [d for d in dets if d.type == "give_and_go"], dets
    print("give_and_go OK: fires on the forward one-two, not on the wall pass")

    # --- 3) overlap with a lagging marker fires; marker keeping pace kills it
    def overlap(marker_speed: float):
        frames, samples = [], []
        n = round(3.0 / DT)
        for k in range(n):
            t = k * DT
            cx, cy = 40.0 + 0.3 * t, 55.0
            rx, ry = 35.0 + 6.0 * t, 62.0
            mx, my = 35.5 + marker_speed * t, 62.5
            frames.append(
                fr(
                    t,
                    [
                        PlayerPos("C", HOME, cx, cy),
                        PlayerPos("R", HOME, rx, ry),
                        PlayerPos("M", AWAY, mx, my),
                    ],
                    (cx, cy),
                )
            )
            if k % 5 == 0:
                samples.append((t, mx, my, rx, ry))
        edge = _E("marking", AWAY, "M", "R", 0, n - 1, 0.0, (n - 1) * DT, samples)
        return frames, [edge]

    dets = run(*overlap(marker_speed=3.0))  # marker at half R's speed: lags
    ov = [d for d in dets if d.type == "overlap_run"]
    assert len(ov) == 1 and ov[0].players == ["R", "C"], (ov, dets)
    assert ov[0].confidence > 0.5 and len(ov[0].geometry["run_path"]) >= 2
    dets = run(*overlap(marker_speed=6.0))  # marker keeps pace -> no overlap
    assert not [d for d in dets if d.type == "overlap_run"], dets
    print("overlap_run OK: fires with lagging marker, silent when tracked")

    # --- 4) third-man run: pass A->B while C sprints forward, dropping a
    # static marker whose edge dissolves mid-pass (separation growing)
    def third_man():
        frames, samples = [], []
        n = round(3.0 / DT)
        for k in range(n):
            t = k * DT
            cx, cy = 50.0 + 6.0 * t, 20.0
            if t <= 1.0:
                ball = (30.0, 34.0)
            elif t <= 1.6:  # 12 m pass at 20 m/s
                ball = (30.0 + 20.0 * (t - 1.0), 34.0)
            else:
                ball = (42.0, 34.0)
            frames.append(
                fr(
                    t,
                    [
                        PlayerPos("A", HOME, 30.0, 34.0),
                        PlayerPos("B", HOME, 42.0, 34.0),
                        PlayerPos("C", HOME, cx, cy),
                        PlayerPos("Dm", AWAY, 49.0, 20.5),  # left standing
                    ],
                    ball,
                )
            )
            if k % 5 == 0 and t <= 1.6:
                samples.append((t, 49.0, 20.5, cx, cy))
        edge = _E("marking", AWAY, "Dm", "C", 0, round(1.6 / DT), 0.0, 1.6, samples)
        return frames, [edge]

    dets = run(*third_man())
    tm = [d for d in dets if d.type == "third_man_run"]
    assert len(tm) == 1 and tm[0].players == ["A", "B", "C"], (tm, dets)
    assert tm[0].confidence <= 0.6 and len(tm[0].geometry["run_path"]) >= 2
    print("third_man_run OK: fires once, conf", tm[0].confidence)

    # --- 5) pressing trap: 3 press edges born inside 1.5 s on carrier X
    # whose only near teammate is marked; unmarked outlet -> no trap
    def trap(outlet_marked: bool):
        frames = []
        n = round(4.0 / DT)
        for k in range(n):
            t = k * DT
            frames.append(
                fr(
                    t,
                    [
                        PlayerPos("X", HOME, 30.0, 34.0),
                        PlayerPos("Y", HOME, 38.0, 34.0),  # 8 m outlet
                        PlayerPos("Z", HOME, 60.0, 34.0),  # 30 m, irrelevant
                        PlayerPos("P1", AWAY, 27.0, 31.0),
                        PlayerPos("P2", AWAY, 27.0, 37.0),
                        PlayerPos("P3", AWAY, 33.0, 30.0),
                        PlayerPos("DY", AWAY, 38.5, 34.5),
                    ],
                    (30.0, 34.0),
                )
            )
        end_t = (n - 1) * DT
        edges = [
            _E("press", AWAY, p, "X", round(t0 / DT), n - 1, t0, end_t,
               [(t0, 27.0, 31.0, 30.0, 34.0)])
            for p, t0 in (("P1", 0.5), ("P2", 1.0), ("P3", 1.6))
        ]
        if outlet_marked:
            edges.append(
                _E("marking", AWAY, "DY", "Y", 0, n - 1, 0.0, end_t,
                   [(0.0, 38.5, 34.5, 38.0, 34.0)])
            )
        return frames, edges

    dets = run(*trap(outlet_marked=True))
    tr = [d for d in dets if d.type == "pressing_trap"]
    assert len(tr) == 1 and tr[0].players[0] == "X", (tr, dets)
    assert set(tr[0].players[1:]) == {"P1", "P2", "P3"}
    assert len(tr[0].geometry["carrier_path"]) >= 2
    dets = run(*trap(outlet_marked=False))
    assert not [d for d in dets if d.type == "pressing_trap"], dets
    print("pressing_trap OK: fires when the outlet is marked, else silent")

    # --- real Metrica sanity: counts per type over the clean 5-min window
    import os

    home = "data/metrica/Sample_Game_1_RawTrackingData_Home_Team.csv"
    away = "data/metrica/Sample_Game_1_RawTrackingData_Away_Team.csv"
    if os.path.exists(home):
        from collections import Counter

        from bto.io.metrica import load_metrica

        frames = load_metrica(home, away, downsample=2)  # 12.5 Hz
        frames = [f for f in frames if 300.0 <= f.t <= 600.0]
        kw = {}
        try:
            from bto.patterns.edges import track_edges  # noqa: F401
        except ImportError:
            kw["edges"] = []  # sibling module not landed yet
        try:
            from bto.patterns.roles import assign_roles  # noqa: F401
        except ImportError:
            kw.setdefault("role_maps_home", [])
            kw.setdefault("role_maps_away", [])
        dets = detect_plays(frames, **kw)
        counts = Counter(d.type for d in dets)
        note = " (edges/roles injected empty: siblings pending)" if kw else ""
        print(f"metrica t=300-600s per-5min counts{note}: {dict(counts)}")
    print("plays.py self-check OK")
