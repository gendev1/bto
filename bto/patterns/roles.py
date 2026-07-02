"""Formation ROLE assignment + rotation detection (SPEC S6 relational layer;
socker-plays-ref Medium row 'Rotations (positional play)').

Roles are the answer to broadcast track-id churn: instead of anchoring
structure to fragile track_ids, each team is described by 10 persistent
formation SLOTS ('R1'..'R10', ordered defense->attack lines then bottom->top
y) that players are assigned to per frame. A slot survives a track dying and
being reborn under a new id.

Method (standard role-alignment loop, cf. Bialkowski et al.):
  1. Work in attacking-normalized coords: x is flipped so the team always
     attacks +x (x' = PITCH_LENGTH - x when Frame.attacking[team] == -1).
  2. Per frame, the rearmost team player is dropped as the GK; the rest are
     the outfield ensemble.
  3. Slots init from the mean positions of the 10 longest-lived tracks
     (padded by farthest-point sampling over the pooled ensemble + a coarse
     pitch grid when fewer than 10 tracks exist, e.g. broadcast churn).
  4. <=5 EM rounds: per frame, Hungarian-assign visible players to slots
     (scipy linear_sum_assignment, euclidean cost; rectangular when <10
     visible -- the visible subset is assigned), then re-estimate each slot
     as the mean of its assigned positions; stop when slots move < 0.5 m.
  5. role_ids ordered by gap-clustering the final slots into 2-4 lines on x
     (defense->attack), then bottom->top y inside each line.
  6. Per-frame assignments are smoothed causally (live-host reusable), two
     layers: the Hungarian cost is made STICKY (a player's current role is
     discounted by stick_m meters, so he only loses it when someone fits it
     clearly better), and on top a hysteresis rule: a player keeps his role
     unless a different assignment persists >= 1.0 s (kills per-frame
     flicker).

Rotations: two players whose SMOOTHED roles swap (a: ra->rb while b: rb->ra
within 2 s of each other) and HOLD the swapped roles >= 4 s, both visible
through the swap. Emitted once per swap as Detection type='rotation',
players=[a, b], geometry={'a_path': [(t, x, y), ...], 'b_path': [...],
'roles': [ra, rb]} with both paths in RAW pitch meters sampled ~5 Hz over
[swap - 1 s, swap + 4 s] -- per the moving-geometry render convention
('*_path' keys are interpolated at the current frame t so the marker moves
with the player). confidence = fraction of the swap window both are visible.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from bto.core import PITCH_LENGTH, Detection, Frame

N_ROLES = 10


def _outfield(frame: Frame, team: str) -> list[tuple[str, float, float]]:
    """(track_id, x, y) in attacking-normalized coords, rearmost (GK) dropped."""
    sign = frame.attacking.get(team, 1)
    pts = [
        (p.track_id, p.x if sign > 0 else PITCH_LENGTH - p.x, p.y)
        for p in frame.players
        if p.team == team
    ]
    if len(pts) <= 1:
        return []
    gk = min(range(len(pts)), key=lambda i: pts[i][1])
    return [p for i, p in enumerate(pts) if i != gk]


def _init_slots(per_frame: list[list[tuple[str, float, float]]]) -> np.ndarray:
    """Initial (10, 2) slot template: mean positions of the 10 longest-lived
    tracks, farthest-point padded from the pooled ensemble + pitch grid."""
    stats: dict[str, list[float]] = {}
    for pts in per_frame:
        for tid, x, y in pts:
            s = stats.setdefault(tid, [0.0, 0.0, 0.0])
            s[0] += x
            s[1] += y
            s[2] += 1.0
    keep = sorted(stats, key=lambda t: (-stats[t][2], t))[:N_ROLES]
    slots = [(stats[t][0] / stats[t][2], stats[t][1] / stats[t][2]) for t in keep]
    if len(slots) < N_ROLES:
        pool = [(x, y) for pts in per_frame[:: max(1, len(per_frame) // 200)] for _, x, y in pts]
        pool += [(x, y) for x in range(15, 105, 15) for y in range(10, 68, 12)]
        cand = np.array(pool)
        while len(slots) < N_ROLES:
            d = np.min(
                np.linalg.norm(cand[:, None, :] - np.array(slots)[None, :, :], axis=2), axis=1
            )
            slots.append(tuple(cand[int(np.argmax(d))]))
    return np.array(slots, dtype=float)


def _assign(
    pts: list[tuple[str, float, float]],
    slots: np.ndarray,
    cur: dict[str, int] | None = None,
    stick_m: float = 5.0,
) -> list[tuple[int, int]]:
    """Hungarian player->slot pairs [(pt_idx, slot_idx)]; rectangular OK.

    When cur (track_id -> current slot idx) is given, that slot's cost is
    discounted by stick_m meters for its holder (sticky assignment)."""
    P = np.array([[x, y] for _, x, y in pts])
    cost = np.linalg.norm(P[:, None, :] - slots[None, :, :], axis=2)
    if cur:
        for r, (tid, _, _) in enumerate(pts):
            c = cur.get(tid)
            if c is not None:
                cost[r, c] -= stick_m
    ri, ci = linear_sum_assignment(cost)
    return list(zip(ri.tolist(), ci.tolist()))


def _fit_slots(
    per_frame: list[list[tuple[str, float, float]]],
    slots: np.ndarray,
    rounds: int = 5,
    tol: float = 0.5,
) -> np.ndarray:
    for _ in range(rounds):
        sums = np.zeros((N_ROLES, 2))
        cnt = np.zeros(N_ROLES)
        for pts in per_frame:
            if not pts:
                continue
            for r, c in _assign(pts, slots):
                sums[c] += (pts[r][1], pts[r][2])
                cnt[c] += 1.0
        new = slots.copy()
        mask = cnt > 0
        new[mask] = sums[mask] / cnt[mask, None]
        moved = float(np.max(np.linalg.norm(new - slots, axis=1)))
        slots = new
        if moved < tol:
            break
    return slots


def _order_slots(slots: np.ndarray, min_line_gap: float = 5.0) -> tuple[list[int], list[int]]:
    """(slot indices in role order, line sizes defense->attack).

    Gap-clusters slots into 2-4 lines on normalized x (largest gaps split,
    as in formation.py), then sorts bottom->top y inside each line.
    """
    by_x = np.argsort(slots[:, 0], kind="stable")
    xs = slots[by_x, 0]
    gaps = np.diff(xs)
    cuts = np.where(gaps >= min_line_gap)[0]
    if len(cuts) == 0 and len(gaps):
        cuts = np.array([np.argmax(gaps)])
    elif len(cuts) > 3:
        cuts = cuts[np.argsort(gaps[cuts])[-3:]]
    bounds = [0] + sorted(int(c) + 1 for c in cuts) + [len(xs)]
    order: list[int] = []
    sizes: list[int] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = by_x[a:b]
        order.extend(int(i) for i in idx[np.argsort(slots[idx, 1], kind="stable")])
        sizes.append(b - a)
    return order, sizes


def _smooth_assign(
    per_frame: list[list[tuple[str, float, float]]],
    times: list[float],
    slots: np.ndarray,
    hyst_s: float,
) -> list[dict[str, str]]:
    """Causal per-frame assignment: sticky Hungarian (biased toward each
    player's current role) + hysteresis (keep the current role unless a
    different raw assignment persists >= hyst_s)."""
    cur: dict[str, int] = {}  # tid -> current slot idx
    pend: dict[str, tuple[int, float]] = {}
    out: list[dict[str, str]] = []
    for t, pts in zip(times, per_frame):
        m: dict[str, str] = {}
        for r, c in _assign(pts, slots, cur) if pts else []:
            tid = pts[r][0]
            k = cur.get(tid)
            if k is None:
                cur[tid] = c
            elif c == k:
                pend.pop(tid, None)
            else:
                p = pend.get(tid)
                if p is None or p[0] != c:
                    pend[tid] = (c, t)
                elif t - p[1] >= hyst_s:
                    cur[tid] = c
                    pend.pop(tid, None)
            m[tid] = f"R{cur[tid] + 1}"
        out.append(m)
    return out


def _raw_path(
    frames: list[Frame], tid: str, i0: int, i1: int, step: int
) -> list[tuple[float, float, float]]:
    """(t, x, y) raw pitch coords of tid over frames[i0:i1:step]."""
    path = []
    for f in frames[i0:i1:step]:
        p = next((p for p in f.players if p.track_id == tid), None)
        if p is not None:
            path.append((f.t, p.x, p.y))
    return path


def _rotations(
    frames: list[Frame],
    role_maps: list[dict[str, str]],
    hold_s: float,
    hyst_s: float,
    pair_s: float = 2.0,
    min_conf: float = 0.5,
) -> list[Detection]:
    times = [f.t for f in frames]
    # smoothed role-change events, in frame order
    events: list[tuple[int, str, str, str]] = []  # (i, tid, old, new)
    prev: dict[str, str] = {}
    for i, m in enumerate(role_maps):
        for tid, role in m.items():
            if tid in prev and prev[tid] != role:
                events.append((i, tid, prev[tid], role))
            prev[tid] = role
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.08
    step = max(1, round(1.0 / (dt * 5.0)))  # ~5 Hz path sampling

    out: list[Detection] = []
    used: set[int] = set()
    for ei, (i, a, ra, rb) in enumerate(events):
        if ei in used:
            continue
        for ej in range(ei + 1, len(events)):
            j, b, old, new = events[ej]
            if times[j] - times[i] > pair_s:
                break
            if ej in used or b == a or old != rb or new != ra:
                continue
            t_swap = times[j]  # later of the two flips
            t1 = t_swap + hold_s
            if times[-1] < t1:
                continue  # segment ends before the hold can be confirmed
            i1 = int(np.searchsorted(times, t1, side="right"))
            held = all(
                m.get(a, rb) == rb and m.get(b, ra) == ra
                for m in role_maps[j:i1]
            )
            if not held:
                continue
            t0 = max(times[0], times[i] - hyst_s)  # include the pending run-up
            i0 = int(np.searchsorted(times, t0))
            both = [
                sum(1 for p in f.players if p.track_id in (a, b)) == 2
                for f in frames[i0:i1]
            ]
            conf = sum(both) / max(len(both), 1)
            if conf < min_conf:
                continue
            out.append(
                Detection(
                    type="rotation",
                    players=[a, b],
                    geometry={
                        "a_path": _raw_path(frames, a, i0, i1, step),
                        "b_path": _raw_path(frames, b, i0, i1, step),
                        "roles": [ra, rb],
                    },
                    confidence=conf,
                    t_start=times[i0],
                    t_end=times[i1 - 1],
                )
            )
            used.add(ei)
            used.add(ej)
            break
    return out


def assign_roles(
    frames: list[Frame],
    team: str,
    rounds: int = 5,
    hyst_s: float = 1.0,
    hold_s: float = 4.0,
) -> tuple[list[dict[str, str]], dict[str, tuple[float, float]], list[Detection]]:
    """Persistent formation roles for one team; see module docstring.

    Returns (role_maps, slots, rotations):
      role_maps : per-frame dict track_id -> 'R1'..'R10', 1:1 with frames,
                  hysteresis-smoothed
      slots     : role_id -> (x, y) mean template, ATTACKING-normalized coords
      rotations : Detection list, type='rotation'
    """
    per_frame = [_outfield(f, team) for f in frames]
    if not any(per_frame):
        return [{} for _ in frames], {}, []
    slots = _fit_slots(per_frame, _init_slots(per_frame), rounds)
    order, _ = _order_slots(slots)
    slots = slots[order]
    slots_out = {f"R{k + 1}": (float(x), float(y)) for k, (x, y) in enumerate(slots)}

    role_maps = _smooth_assign(per_frame, [f.t for f in frames], slots, hyst_s)
    rotations = _rotations(frames, role_maps, hold_s, hyst_s)
    return role_maps, slots_out, rotations


if __name__ == "__main__":
    from bto.core import AWAY, HOME, PlayerPos

    FPS = 12.5

    # 4-4-2 + GK, HOME attacking +x. LB and LM are the swap pair.
    base = {
        "GK": (5.0, 34.0),
        "LB": (20.0, 10.0), "CB1": (20.0, 25.0), "CB2": (20.0, 43.0), "RB": (20.0, 58.0),
        "LM": (40.0, 10.0), "CM1": (40.0, 25.0), "CM2": (40.0, 43.0), "RM": (40.0, 58.0),
        "ST1": (60.0, 27.0), "ST2": (60.0, 41.0),
    }

    def mk(t: float, pos: dict, only: list[str] | None = None) -> Frame:
        players = [
            PlayerPos(tid, HOME, x, y)
            for tid, (x, y) in pos.items()
            if only is None or tid in only
        ]
        return Frame(t=t, players=players, ball=None, attacking={HOME: 1, AWAY: -1})

    def lerp(a, b, u):
        return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)

    # --- 1. sustained LB<->LM swap: base 15 s, 1 s transition, swapped to 30 s
    frames = []
    for k in range(int(30 * FPS)):
        t = k / FPS
        pos = dict(base)
        if 15.0 <= t < 16.0:
            u = t - 15.0
            pos["LB"] = lerp(base["LB"], base["LM"], u)
            pos["LM"] = lerp(base["LM"], base["LB"], u)
        elif t >= 16.0:
            pos["LB"], pos["LM"] = base["LM"], base["LB"]
        frames.append(mk(t, pos))

    role_maps, slots, rotations = assign_roles(frames, HOME)
    assert len(role_maps) == len(frames)
    assert set(slots) == {f"R{i}" for i in range(1, 11)}
    assert len(role_maps[0]) == 10 and len(set(role_maps[0].values())) == 10
    assert "GK" not in role_maps[0]
    r_lb, r_lm = role_maps[0]["LB"], role_maps[0]["LM"]
    assert r_lb == "R1", (r_lb, slots)  # deepest line, lowest y
    assert len(rotations) == 1, [(d.players, d.geometry["roles"]) for d in rotations]
    det = rotations[0]
    assert set(det.players) == {"LB", "LM"}
    assert set(det.geometry["roles"]) == {r_lb, r_lm}
    assert det.confidence == 1.0
    assert det.geometry["a_path"] and det.geometry["b_path"]
    assert all(len(p) == 3 for p in det.geometry["a_path"])
    # roles actually swapped and held in the smoothed maps
    assert role_maps[-1]["LB"] == r_lm and role_maps[-1]["LM"] == r_lb

    # --- 2. 1-frame flicker swap -> no rotation, no smoothed role change
    frames_f = []
    for k in range(int(20 * FPS)):
        t = k / FPS
        pos = dict(base)
        if k == int(10 * FPS):
            pos["LB"], pos["LM"] = base["LM"], base["LB"]
        frames_f.append(mk(t, pos))
    maps_f, _, rot_f = assign_roles(frames_f, HOME)
    assert rot_f == [], rot_f
    assert all(m["LB"] == maps_f[0]["LB"] and m["LM"] == maps_f[0]["LM"] for m in maps_f)

    # --- 3. only 7 visible players: subset roles, distinct, no crash
    seven = ["GK", "LB", "CB1", "CB2", "LM", "CM1", "ST1"]
    frames_7 = [mk(k / FPS, base, only=seven) for k in range(int(10 * FPS))]
    maps_7, slots_7, rot_7 = assign_roles(frames_7, HOME)
    assert len(slots_7) == 10
    assert all(len(m) == 6 for m in maps_7)  # 7 visible minus rearmost GK
    assert all(len(set(m.values())) == 6 for m in maps_7)
    assert rot_7 == []

    # --- 4. broadcast-style tid churn + dropout: no crash, subset assigned
    rng = np.random.default_rng(0)
    frames_c = []
    for k in range(int(20 * FPS)):
        t = k / FPS
        gen = int(t // 4)  # every track reborn under a new id every 4 s
        hide = set(rng.choice(list(base), size=3, replace=False))
        pos = {f"{tid}g{gen}": xy for tid, xy in base.items() if tid not in hide}
        frames_c.append(mk(t, pos))
    maps_c, slots_c, _ = assign_roles(frames_c, HOME)
    assert len(slots_c) == 10
    valid = set(slots_c)
    assert all(set(m.values()) <= valid for m in maps_c)
    assert all(len(m) >= 6 for m in maps_c), min(len(m) for m in maps_c)

    print("roles synthetic self-check OK")

    # --- REAL Metrica sanity, t=300-600 s at 12.5 Hz
    from pathlib import Path

    data = Path(__file__).resolve().parents[2] / "data" / "metrica"
    home_csv = data / "Sample_Game_1_RawTrackingData_Home_Team.csv"
    away_csv = data / "Sample_Game_1_RawTrackingData_Away_Team.csv"
    if home_csv.exists():
        from bto.io.metrica import load_metrica

        all_frames = load_metrica(str(home_csv), str(away_csv), downsample=2)
        win = [f for f in all_frames if 300.0 <= f.t <= 600.0]
        for team in (HOME, AWAY):
            maps, slots_t, rots = assign_roles(win, team)
            _, sizes = _order_slots(np.array([slots_t[f"R{i}"] for i in range(1, 11)]))
            label = "-".join(str(s) for s in sizes)
            # role stability: mean smoothed role changes per player per minute
            changes: dict[str, int] = {}
            prev: dict[str, str] = {}
            for m in maps:
                for tid, role in m.items():
                    if tid in prev and prev[tid] != role:
                        changes[tid] = changes.get(tid, 0) + 1
                    prev[tid] = role
            n_players = len(prev)
            minutes = (win[-1].t - win[0].t) / 60.0
            stab = sum(changes.values()) / max(n_players, 1) / minutes
            print(
                f"metrica {team}: formation={label} rotations={len(rots)} "
                f"role-changes/player/min={stab:.2f} players={n_players}"
            )
            assert 0 <= len(rots) <= 6, len(rots)
            assert stab < 2.0, stab
        print("roles metrica sanity OK")
