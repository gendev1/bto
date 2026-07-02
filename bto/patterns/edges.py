"""Relationship edges with lifecycles (M6 relational layer).

The "player 1 covers player 1a" tracker: instead of re-deriving pairings
per frame (dancing rectangles), each defender-attacker relationship is a
stateful EDGE that is born after sustained evidence, survives excursions
and tracker id churn, and dissolves only on sustained separation.

  'marking' -- defender a within r_mark (6 m) of opponent b, sustained
      >= 1.2 s with coupled motion (when b is moving the held distance is
      the evidence; when b is near-static the distance must be stable, so
      a fly-by past a statue never births an edge). One defender marks at
      most one attacker at a time (closest sustained candidate wins,
      re-evaluated when an edge dissolves); an attacker can be marked by
      up to two defenders (double-team). The edge SURVIVES excursions:
      it dissolves only after > 1.5 s continuously outside r_dissolve
      (9 m) or either player missing > 2 s.
  'press'  -- opponent a within r_press (5 m) of the current ball carrier
      (possession() spell) and CLOSING (negative d(dist)/dt from
      core.track_velocity) sustained >= 0.6 s; dissolves when the spell
      ends or the presser retreats beyond 7 m.

Birth is backdated to the start of the sustained window, so a 10 s shadow
job yields one ~10 s edge. Marking candidacy is one-directional: while a
possession spell is live only the OUT-of-possession side can be 'a'; in
possession gaps a goal-side fallback (a nearer its own goal line than b)
keeps the machine running without minting reciprocal a<->b duplicates.

Tid churn (the FIFA CV path): if an endpoint's track_id vanishes but a
same-team player stands within 2 m of its last position within 0.5 s, the
edge adopts the new tid -- same trick possession.py uses for spell holders.
Edge.a / Edge.b keep the ORIGINAL tids; samples follow the adopted player.

samples: [(t, xa, ya, xb, yb)] at ~5 Hz from actual positions -- the
renderer's moving-geometry convention interpolates a Detection carrying a
'track' key at the current frame t, so the drawn link moves with the
players. strength = mean(1 - dist/r_dissolve) over the edge's life.

Everything is causal: state at frame i only looks backward.
"""

from collections import Counter
from dataclasses import dataclass, field
from math import hypot
from statistics import pstdev

from bto.core import PITCH_LENGTH, Detection, Frame, other, track_velocity
from bto.patterns.possession import possession

R_MARK = 6.0  # m: marking birth radius
R_DISSOLVE = 9.0  # m: marking survives up to here; also the strength scale
MARK_SUSTAIN_S = 1.2  # s: evidence required before a marking edge is born
EXCURSION_S = 1.5  # s: continuously outside r_dissolve before dissolve
MISSING_S = 2.0  # s: either endpoint unseen this long dissolves the edge
ADOPT_DIST = 2.0  # m: tid-swap adoption radius around the last position
ADOPT_GAP_S = 0.5  # s: adoption must happen this soon after the vanish
CAND_GAP_S = 0.3  # s: candidacy survives blips this short (dropout/jitter)
R_PRESS = 5.0  # m: press birth radius around the ball carrier
PRESS_SUSTAIN_S = 0.6  # s: sustained closing required for a press edge
PRESS_RETREAT = 7.0  # m: presser beyond this dissolves the press edge
CARRIER_BALL_MAX = 3.0  # m: spell tid must be near the visible ball (stale-tid guard)
SAMPLE_DT = 0.18  # s: >= this between samples -> ~4-5 Hz track
B_MOVING = 2.0  # m/s: b faster than this counts as moving
STATIC_STD_MAX = 1.0  # m: max distance std for marking a near-static b
GHOST_DIST = 0.3  # m: closer than this is a duplicate/ghost box, never a pair


@dataclass
class Edge:
    kind: str  # 'marking' | 'press'
    team: str  # team of the DEFENDING side (a)
    a: str  # defender track_id (original, pre-adoption)
    b: str  # attacker/carrier track_id (original, pre-adoption)
    i_start: int
    i_end: int
    t_start: float
    t_end: float
    samples: list = field(default_factory=list)  # [(t, xa, ya, xb, yb), ...] ~5 Hz
    strength: float = 0.0  # 0..1, mean(1 - dist/r_dissolve) over life


def _adopt(frame: Frame, team: str, last_pos, avoid: set):
    """Nearest same-team player within ADOPT_DIST of last_pos, or None."""
    best, best_d = None, ADOPT_DIST
    for p in frame.players:
        if p.team != team or p.track_id in avoid:
            continue
        d = hypot(p.x - last_pos[0], p.y - last_pos[1])
        if d <= best_d:
            best, best_d = p, d
    return best


def _goal_side(frame: Frame, a, b) -> bool:
    """a is nearer its OWN goal line than b (fallback marker test)."""
    gx = 0.0 if frame.attacking.get(a.team, 1) == 1 else PITCH_LENGTH
    return abs(a.x - gx) < abs(b.x - gx)


class _Live:
    """Mutable state of one alive edge. entries: [(i, t, d, xa, ya, xb, yb)]."""

    def __init__(self, kind, team, a, b, entries, spell=None):
        self.kind, self.team, self.a, self.b, self.spell = kind, team, a, b, spell
        self.cur = {"a": a, "b": b}  # adopted tids
        self.teams = {"a": team, "b": other(team)}
        self.i_start, self.t_start = entries[0][0], entries[0][1]
        self.samples: list = []
        self.dists: list = []  # (frame_idx, dist) whenever both endpoints seen
        last = entries[-1]
        self.last_pos = {"a": (last[3], last[4]), "b": (last[5], last[6])}
        self.last_t = {"a": last[1], "b": last[1]}
        self.last_good_i, self.last_good_t = entries[0][0], entries[0][1]
        self.last_good_sample = None
        self.outside_since = None
        for i, t, d, xa, ya, xb, yb in entries:
            self.dists.append((i, d))
            self._record(i, t, d, xa, ya, xb, yb)

    def _record(self, i, t, d, xa, ya, xb, yb):
        if not self.samples or t - self.samples[-1][0] >= SAMPLE_DT - 1e-9:
            self.samples.append((t, xa, ya, xb, yb))
        self.last_good_i, self.last_good_t = i, t
        self.last_good_sample = (t, xa, ya, xb, yb)

    def step(self, i, frame, byid) -> bool:
        """Advance one frame; False means dissolve now."""
        t = frame.t
        pos = {}
        for ep in ("a", "b"):
            p = byid.get(self.cur[ep])
            if p is None and t - self.last_t[ep] <= ADOPT_GAP_S:
                p = _adopt(frame, self.teams[ep], self.last_pos[ep], set(self.cur.values()))
                if p is not None:
                    self.cur[ep] = p.track_id  # tid swap adopted
            if p is not None:
                self.last_pos[ep] = (p.x, p.y)
                self.last_t[ep] = t
            pos[ep] = p
        if pos["a"] is None or pos["b"] is None:
            return t - min(self.last_t.values()) <= MISSING_S
        pa, pb = pos["a"], pos["b"]
        d = hypot(pa.x - pb.x, pa.y - pb.y)
        self.dists.append((i, d))
        r_out = R_DISSOLVE if self.kind == "marking" else PRESS_RETREAT
        if d <= r_out:
            self.outside_since = None
            self._record(i, t, d, pa.x, pa.y, pb.x, pb.y)
            return True
        if self.kind == "press":
            return False  # presser retreated
        if self.outside_since is None:
            self.outside_since = t
        return t - self.outside_since <= EXCURSION_S

    def done(self) -> Edge | None:
        """Finalize: end at the last in-range frame, trim, score strength."""
        i_end, t_end = self.last_good_i, self.last_good_t
        if i_end <= self.i_start:
            return None
        ds = [d for k, d in self.dists if k <= i_end]
        strength = max(0.0, min(1.0, sum(1.0 - d / R_DISSOLVE for d in ds) / len(ds)))
        samples = [s for s in self.samples if s[0] <= t_end + 1e-9]
        if samples and self.last_good_sample and samples[-1][0] < self.last_good_sample[0] - 1e-9:
            samples.append(self.last_good_sample)
        return Edge(
            self.kind, self.team, self.a, self.b,
            self.i_start, i_end, self.t_start, t_end, samples, strength,
        )


def track_edges(frames: list[Frame]) -> list[Edge]:
    """Run the marking/press edge state machine over a frame segment."""
    if not frames:
        return []
    spells = possession(frames)
    spell_at = {}
    for s in spells:
        for i in range(s.i_start, s.i_end + 1):
            spell_at[i] = s

    out: list[Edge] = []
    live: list[_Live] = []
    mark_cand: dict = {}  # (a_tid, b_tid) -> {'entries': [...], 'speeds': [...]}
    press_cand: dict = {}  # (a_tid, spell.i_start) -> [entries]

    for i, frame in enumerate(frames):
        t = frame.t
        byid = {p.track_id: p for p in frame.players}

        # -- advance live edges (dissolves free defenders before births) --
        still = []
        for e in live:
            alive = False if (e.kind == "press" and i > e.spell.i_end) else e.step(i, frame, byid)
            if alive:
                still.append(e)
            else:
                fin = e.done()
                if fin is not None:
                    out.append(fin)
        live = still

        spell = spell_at.get(i)

        # -- marking candidacy: ordered pairs (a marks b) within r_mark --
        by_team: dict = {}
        for p in frame.players:
            by_team.setdefault(p.team, []).append(p)
        qual = {}  # (a_tid, b_tid) -> (d, entry, b_speed)
        speed_cache: dict = {}
        for a_team, a_list in by_team.items():
            if spell is not None and a_team == spell.team:
                continue  # only the out-of-possession side marks
            for a in a_list:
                for b in by_team.get(other(a_team), []):
                    d = hypot(a.x - b.x, a.y - b.y)
                    if not (GHOST_DIST <= d <= R_MARK):
                        continue
                    if spell is None and not _goal_side(frame, a, b):
                        continue  # no possession info: goal-side only
                    sp = speed_cache.get(b.track_id)
                    if sp is None:
                        v = track_velocity(frames, i, b.track_id)
                        sp = hypot(*v) if v is not None else 0.0
                        speed_cache[b.track_id] = sp
                    qual[(a.track_id, b.track_id)] = (
                        d, (i, t, d, a.x, a.y, b.x, b.y), sp,
                    )
        # accumulate QUALIFIED TIME per candidate; a blip (distance jitter
        # over r_mark, a dropout, a possession flip) <= CAND_GAP_S pauses the
        # clock instead of resetting 1.2s of evidence -- flicker-proofing.
        for key, (d, entry, sp) in qual.items():
            c = mark_cand.setdefault(key, {"entries": [], "speeds": [], "qual_t": 0.0})
            if c["entries"]:
                c["qual_t"] += min(t - c["entries"][-1][1], CAND_GAP_S)
            c["entries"].append(entry)
            c["speeds"].append(sp)
        for key in list(mark_cand):
            if key not in qual and t - mark_cand[key]["entries"][-1][1] > CAND_GAP_S:
                del mark_cand[key]  # not qualifying for too long: reset

        # -- marking births: closest sustained candidate per free defender --
        busy = {e.cur["a"] for e in live if e.kind == "marking"}
        marked = Counter(e.cur["b"] for e in live if e.kind == "marking")
        ready: dict = {}  # a_tid -> (d_now, key)
        for key, c in mark_cand.items():
            if key not in qual:
                continue
            entries = c["entries"]
            if c["qual_t"] < MARK_SUSTAIN_S:
                continue
            a_tid, b_tid = key
            if a_tid in busy or marked[b_tid] >= 2:
                continue
            # coupled motion over the trailing sustain window: a moving b is
            # coupled by the held distance itself (spec: cos-sim OR stays
            # within r_mark); a near-static b needs a STABLE distance.
            wd = [e[2] for e in entries if e[1] >= t - MARK_SUSTAIN_S]
            ws = [s for e, s in zip(entries, c["speeds"]) if e[1] >= t - MARK_SUSTAIN_S]
            if ws and sum(ws) / len(ws) <= B_MOVING and len(wd) > 1 and pstdev(wd) > STATIC_STD_MAX:
                continue
            d_now = qual[key][0]
            if a_tid not in ready or d_now < ready[a_tid][0]:
                ready[a_tid] = (d_now, key)
        for a_tid, (_, key) in ready.items():
            if marked[key[1]] >= 2:
                continue  # double-team cap raced by another birth this frame
            c = mark_cand.pop(key)
            live.append(_Live("marking", byid[a_tid].team, key[0], key[1], c["entries"]))
            marked[key[1]] += 1

        # -- press candidacy: closing on the carrier, sustained --
        if spell is None:
            press_cand.clear()
            continue
        carrier = byid.get(spell.track_id)
        pq = {}
        if (
            carrier is not None
            and frame.ball is not None
            and hypot(carrier.x - frame.ball[0], carrier.y - frame.ball[1]) <= CARRIER_BALL_MAX
        ):
            vb = track_velocity(frames, i, carrier.track_id)
            if vb is not None:
                for p in frame.team_players(other(spell.team)):
                    if p.track_id == carrier.track_id:
                        continue
                    d = hypot(p.x - carrier.x, p.y - carrier.y)
                    if not (GHOST_DIST <= d <= R_PRESS):
                        continue
                    va = track_velocity(frames, i, p.track_id)
                    if va is None:
                        continue
                    ddot = (
                        (p.x - carrier.x) * (va[0] - vb[0])
                        + (p.y - carrier.y) * (va[1] - vb[1])
                    ) / d
                    if ddot < -0.05:  # closing
                        pq[p.track_id] = (i, t, d, p.x, p.y, carrier.x, carrier.y)
        for tid, entry in pq.items():
            c = press_cand.setdefault((tid, spell.i_start), {"entries": [], "qual_t": 0.0})
            if c["entries"]:
                c["qual_t"] += min(t - c["entries"][-1][1], CAND_GAP_S)
            c["entries"].append(entry)
        for key in list(press_cand):
            if key[1] != spell.i_start:
                del press_cand[key]  # spell over: reset
            elif key[0] not in pq and t - press_cand[key]["entries"][-1][1] > CAND_GAP_S:
                del press_cand[key]  # stopped closing for too long: reset
        pressing = {e.cur["a"] for e in live if e.kind == "press" and e.spell is spell}
        for key in list(press_cand):
            if key[0] not in pq:
                continue  # only birth on a qualifying frame
            entries = press_cand[key]["entries"]
            if press_cand[key]["qual_t"] < PRESS_SUSTAIN_S:
                continue
            tid = key[0]
            del press_cand[key]
            if tid in pressing:
                continue
            live.append(_Live("press", byid[tid].team, tid, spell.track_id, entries, spell=spell))
            pressing.add(tid)

    for e in live:
        fin = e.done()
        if fin is not None:
            out.append(fin)
    out.sort(key=lambda e: (e.t_start, e.kind, e.a))
    return out


def edges_to_detections(edges: list[Edge]) -> list[Detection]:
    """Edge -> Detection with a moving 'track' the renderer interpolates."""
    return [
        Detection(
            type="marking" if e.kind == "marking" else "press_engage",
            players=[e.a, e.b],
            geometry={"track": e.samples, "kind": e.kind},
            confidence=e.strength,
            t_start=e.t_start,
            t_end=e.t_end,
        )
        for e in edges
    ]


# --------------------------------------------------------------------------
# Self-check: synthetic lifecycles, then real-data sanity rates.
if __name__ == "__main__":
    import os

    from bto.core import AWAY, HOME, PlayerPos

    DT = 0.08  # 12.5 Hz, same as the Metrica sanity window

    def fr(t, players, ball):
        return Frame(t=t, players=players, ball=ball, attacking={HOME: 1, AWAY: -1})

    # 1) defender shadows a carrier for 10s with a 1s excursion to 8m
    #    -> ONE marking edge spanning ~10s (8m < r_dissolve: no split).
    fs = []
    for k in range(125):
        t = k * DT
        ax = 20.0 + 3.0 * t  # attacker (carrier) advancing at 3 m/s
        off = 8.0 if 4.0 <= t < 5.0 else 2.0  # 1s excursion to 8m
        fs.append(fr(t, [
            PlayerPos("A", HOME, ax, 34.0),
            PlayerPos("D", AWAY, ax + off, 34.0),
        ], (ax, 34.0)))
    edges = track_edges(fs)
    mk = [e for e in edges if e.kind == "marking"]
    assert len(mk) == 1, mk
    e = mk[0]
    assert (e.a, e.b, e.team) == ("D", "A", AWAY), e
    assert e.t_end - e.t_start >= 9.0, (e.t_start, e.t_end)
    assert not [x for x in edges if x.kind == "press"], edges  # constant gap: no press
    assert 0.0 < e.strength <= 1.0
    span1 = e.t_end - e.t_start

    # 2) defender merely stands near a CROSSING attacker for ~0.8s -> no edge.
    fs = []
    for k in range(50):
        t = k * DT
        fs.append(fr(t, [
            PlayerPos("D", HOME, 30.0, 34.0),
            PlayerPos("X", AWAY, 35.0, 20.0 + 8.0 * t),  # fly-by, <=6m for ~0.83s
        ], None))
    assert track_edges(fs) == [], track_edges(fs)

    # 3) scripted press: two defenders close on a static carrier -> 2 press edges.
    fs = []
    for k in range(50):
        t = k * DT
        adv = min(3.0 * t, 5.5)  # from 8m down to 2.5m, then hold
        fs.append(fr(t, [
            PlayerPos("C", HOME, 50.0, 34.0),
            PlayerPos("P1", AWAY, 58.0 - adv, 34.0),
            PlayerPos("P2", AWAY, 50.0, 26.0 + adv),
        ], (50.0, 34.0)))
    pr = [e for e in track_edges(fs) if e.kind == "press"]
    assert len(pr) == 2 and {e.a for e in pr} == {"P1", "P2"}, pr
    assert all(e.b == "C" and e.team == AWAY and 0 < e.strength <= 1 for e in pr), pr

    # 4) tid swap mid-edge (D -> D2 at t=5) -> edge continues, samples continuous.
    fs = []
    for k in range(125):
        t = k * DT
        ax = 20.0 + 3.0 * t
        tid = "D" if k < 62 else "D2"
        fs.append(fr(t, [
            PlayerPos("A", HOME, ax, 34.0),
            PlayerPos(tid, AWAY, ax + 2.0, 34.0),
        ], (ax, 34.0)))
    mk = [e for e in track_edges(fs) if e.kind == "marking"]
    assert len(mk) == 1 and mk[0].a == "D", mk  # original tid kept, no split
    e = mk[0]
    assert e.t_end - e.t_start >= 9.0, (e.t_start, e.t_end)
    gaps = [b[0] - a[0] for a, b in zip(e.samples, e.samples[1:])]
    assert gaps and max(gaps) <= 0.4, max(gaps)  # samples continuous across swap

    dets = edges_to_detections(track_edges(fs))
    assert all(d.type in ("marking", "press_engage") for d in dets)
    assert all("track" in d.geometry and d.geometry["track"] for d in dets)
    print(f"edges.py self-check OK: shadow span {span1:.2f}s (one edge), "
          f"fly-by 0, press {len(pr)}, tid-swap continuous (max sample gap {max(gaps):.2f}s)")

    # ---- REAL Metrica sanity (t=300-600s at 12.5 Hz) ----
    home_csv = "data/metrica/Sample_Game_1_RawTrackingData_Home_Team.csv"
    away_csv = "data/metrica/Sample_Game_1_RawTrackingData_Away_Team.csv"
    if os.path.exists(home_csv):
        from bto.io.metrica import load_metrica
        from bto.patterns.pressing import detect_pressing

        window = [f for f in load_metrica(home_csv, away_csv, downsample=2)
                  if 300.0 <= f.t <= 600.0]
        mins = (window[-1].t - window[0].t) / 60.0
        edges = track_edges(window)
        mk = [e for e in edges if e.kind == "marking"]
        pr = [e for e in edges if e.kind == "press"]
        conc = [sum(1 for e in edges if e.t_start <= f.t <= e.t_end) for f in window]
        old = detect_pressing(window)
        mean_life = sum(e.t_end - e.t_start for e in edges) / max(1, len(edges))
        print(f"metrica {mins:.1f} min: marking {len(mk)} ({len(mk)/mins:.1f}/min), "
              f"press {len(pr)} ({len(pr)/mins:.1f}/min), "
              f"concurrent alive mean {sum(conc)/len(conc):.1f} max {max(conc)}, "
              f"mean edge life {mean_life:.1f}s, "
              f"mean strength {sum(e.strength for e in edges)/max(1,len(edges)):.2f}; "
              f"old detect_pressing {len(old)} ({len(old)/mins:.1f}/min)")

    # ---- FIFA CV path tolerance check (fragmented tids, partial visibility) ----
    base = "out/cwc2021_chelsea_palmeiras_20m"
    if os.path.exists(os.path.join(base, "perception.jsonl")):
        from bto.vision.bridge import build_frames

        segs = build_frames(
            os.path.join(base, "perception.jsonl"),
            os.path.join(base, "calib.jsonl"),
            os.path.join(base, "ball.jsonl"),
        )
        total_s = sum(s[-1].t - s[0].t for s in segs)
        all_edges = [e for s in segs for e in track_edges(s)]
        mk = [e for e in all_edges if e.kind == "marking"]
        pr = [e for e in all_edges if e.kind == "press"]
        life = sum(e.t_end - e.t_start for e in all_edges) / max(1, len(all_edges))
        print(f"cwc {total_s/60:.1f} min ({len(segs)} segments): "
              f"marking {len(mk)} ({len(mk)/(total_s/60):.1f}/min), "
              f"press {len(pr)} ({len(pr)/(total_s/60):.1f}/min), mean life {life:.1f}s")
