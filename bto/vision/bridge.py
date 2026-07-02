"""M2 -> M1 bridge (SPEC S3: the coordinate stream).

Joins perception.jsonl (pixel-space players/ball per frame, M2) with
calib.jsonl (per-frame px->meters homography H, M3 calibration) and emits
contiguous main-camera SEGMENTS of bto.core.Frame for the pattern engine.

Rules (frozen M3 interchange):
- Only rows with shot=='main' AND a non-null H produce a Frame.
- Segments split on: a t-gap > 2.0s between emitted frames, or any
  intervening non-main shot (replay/close-up breaks tracking + team
  continuity even when shorter than 2s). Frames inside a main run whose H
  is null/missing are holes: they emit nothing and only split via the
  2s-gap rule. Segments shorter than 5s are dropped.
- Players: foot [x,y] px -> meters via H (homogeneous divide). No clamping;
  points outside [-3,108]x[-3,71] are DROPPED (bad H tail / detector junk).
  Referees dropped. Goalkeepers folded into the team whose outfield
  centroid-x (over the segment) lies on the same half as the gk's mean x:
  a gk belongs to the team defending the goal he stands next to.
- attacking: recomputed per segment (tracking is broadcast-relative and
  segments are short; teams don't swap mid-half but each segment stands
  alone). The team whose gk -- or, lacking one, rearmost per-frame mean
  x -- is nearer x=0 attacks +1; the other team -1.
"""

import json

from bto.core import AWAY, HOME, Frame, PlayerPos

X_MIN, X_MAX = -3.0, 108.0
Y_MIN, Y_MAX = -3.0, 71.0
MAX_GAP_S = 2.0
MIN_SEGMENT_S = 5.0
HALF_X = 52.5


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _flat_h(h):
    """Accept 9 row-major floats (frozen format) or a nested 3x3."""
    if h is None:
        return None
    if len(h) == 3:
        h = [v for row in h for v in row]
    return h


def _project(h, x, y):
    w = h[6] * x + h[7] * y + h[8]
    if abs(w) < 1e-9:
        return None
    return ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)


def _in_bounds(p):
    return X_MIN <= p[0] <= X_MAX and Y_MIN <= p[1] <= Y_MAX


def _project_row(row, h, ball_by_idx):
    """One perception row -> (t, raw_players, ball_m). raw player: (tid, tag, x, y)
    with tag in {'home','away','gk'}; refs and out-of-bounds points dropped."""
    players = []
    for p in row.get("players") or []:
        team = p.get("team")
        if p.get("cls") == "referee" or team == "ref":
            continue
        if team not in ("home", "away", "gk"):
            continue
        foot = p.get("foot")
        if not foot:
            continue
        pt = _project(h, foot[0], foot[1])
        if pt is None or not _in_bounds(pt):
            continue
        players.append((p["tid"], team, pt[0], pt[1]))

    ball_px = row.get("ball")
    if ball_px is None and ball_by_idx is not None:
        ball_px = ball_by_idx.get(row["frame_idx"])
    ball_m = None
    if ball_px is not None:
        pt = _project(h, ball_px[0], ball_px[1])
        if pt is not None and _in_bounds(pt):
            ball_m = (pt[0], pt[1])
    return row["t"], players, ball_m


def _finalize(raw):
    """Raw segment rows -> list[Frame]: fold gks, compute attacking."""
    # Outfield centroid-x per team, gk mean-x per gk tid (over the segment).
    sums = {HOME: [0.0, 0], AWAY: [0.0, 0]}
    gk_sums = {}
    for _, players, _ in raw:
        for tid, tag, x, _y in players:
            if tag == "gk":
                s = gk_sums.setdefault(tid, [0.0, 0])
                s[0] += x
                s[1] += 1
            else:
                sums[tag][0] += x
                sums[tag][1] += 1
    cent = {t: (s[0] / s[1] if s[1] else HALF_X) for t, s in sums.items()}

    # gk -> team defending the goal he stands next to: the team whose
    # outfield centroid-x is on the gk's half (== nearest that goal end).
    gk_team = {}
    for tid, (sx, n) in gk_sums.items():
        gx = sx / n
        left = min((HOME, AWAY), key=lambda t: cent[t])
        gk_team[tid] = left if gx < HALF_X else (AWAY if left == HOME else HOME)

    # Attacking direction: reference back-x per team = folded gk mean x if
    # any, else mean of that team's per-frame rearmost (min) x. Nearer x=0
    # => defends the left goal => attacks +1.
    ref_x = {}
    for team in (HOME, AWAY):
        gxs = [gk_sums[tid][0] / gk_sums[tid][1] for tid in gk_team if gk_team[tid] == team]
        if gxs:
            ref_x[team] = sum(gxs) / len(gxs)
        else:
            rears = []
            for _, players, _ in raw:
                xs = [x for _tid, tag, x, _y in players if tag == team]
                if xs:
                    rears.append(min(xs))
            ref_x[team] = sum(rears) / len(rears) if rears else HALF_X
    home_plus = ref_x[HOME] < ref_x[AWAY] or ref_x[HOME] == ref_x[AWAY]
    attacking = {HOME: 1 if home_plus else -1, AWAY: -1 if home_plus else 1}

    frames = []
    for t, players, ball in raw:
        pps = [
            PlayerPos(f"T{tid}", gk_team.get(tid, tag) if tag == "gk" else tag, x, y)
            for tid, tag, x, y in players
            if tag != "gk" or tid in gk_team
        ]
        frames.append(Frame(t=t, players=pps, ball=ball, attacking=dict(attacking)))
    return frames


def build_frames(perception_jsonl, calib_jsonl, ball_jsonl=None):
    """Bridge M2 perception + M3 calibration into main-camera segments of
    bto.core.Frame. Returns list of segments (each a list[Frame])."""
    perception = _read_jsonl(perception_jsonl)
    calib = {r["frame_idx"]: _flat_h(r.get("H")) for r in _read_jsonl(calib_jsonl)}
    ball_by_idx = None
    if ball_jsonl is not None:
        ball_by_idx = {
            r["frame_idx"]: r["ball"] for r in _read_jsonl(ball_jsonl) if r.get("ball")
        }

    segments_raw, cur, broken = [], [], False
    for row in perception:
        if row.get("shot") != "main":
            broken = True  # non-main region: split even if shorter than 2s
            continue
        h = calib.get(row["frame_idx"])
        if h is None:
            continue  # hole inside a main run; splits only via the 2s rule
        if cur and (broken or row["t"] - cur[-1][0] > MAX_GAP_S):
            segments_raw.append(cur)
            cur = []
        broken = False
        cur.append(_project_row(row, h, ball_by_idx))
    if cur:
        segments_raw.append(cur)

    segments = []
    for raw in segments_raw:
        if raw[-1][0] - raw[0][0] < MIN_SEGMENT_S:
            continue
        segments.append(_finalize(raw))
    return segments


def frames_stats(segments):
    """Summary stats for reporting."""
    n_frames = sum(len(s) for s in segments)
    n_players = sum(len(f.players) for s in segments for f in s)
    n_ball = sum(1 for s in segments for f in s if f.ball is not None)
    return {
        "n_segments": len(segments),
        "n_frames": n_frames,
        "total_s": round(sum(s[-1].t - s[0].t for s in segments), 2),
        "mean_players_per_frame": round(n_players / n_frames, 2) if n_frames else 0.0,
        "ball_coverage": round(n_ball / n_frames, 3) if n_frames else 0.0,
    }


# --------------------------------------------------------------------------
# Self-check: synthetic perception + calib fixtures, pure CPU.
if __name__ == "__main__":
    import os
    import tempfile

    # Identity-ish H: top-down 1050x680 "video", 10 px per meter, y flipped
    # (pixel y down, pitch y up):  x_m = px/10, y_m = 68 - py/10.
    H = [0.1, 0.0, 0.0, 0.0, -0.1, 68.0, 0.0, 0.0, 1.0]

    def px(x_m, y_m):
        return [x_m * 10.0, (68.0 - y_m) * 10.0]

    def row(i, t, shot="main"):
        players = [
            {"tid": 1, "team": "home", "cls": "player", "conf": 0.9, "foot": px(30, 30)},
            {"tid": 2, "team": "home", "cls": "player", "conf": 0.9, "foot": px(40, 40)},
            {"tid": 3, "team": "away", "cls": "player", "conf": 0.9, "foot": px(60, 30)},
            {"tid": 4, "team": "away", "cls": "player", "conf": 0.9, "foot": px(70, 40)},
            {"tid": 5, "team": "gk", "cls": "goalkeeper", "conf": 0.9, "foot": px(4, 34)},   # home end
            {"tid": 6, "team": "gk", "cls": "goalkeeper", "conf": 0.9, "foot": px(101, 34)},  # away end
            {"tid": 7, "team": "ref", "cls": "referee", "conf": 0.9, "foot": px(52, 34)},
            {"tid": 8, "team": "away", "cls": "player", "conf": 0.9, "foot": px(-10, 34)},   # out of bounds
        ]
        return {"frame_idx": i, "t": round(t, 3), "shot": shot,
                "players": players, "ball": px(52.5, 34.0)}

    rows, calib, dt = [], [], 0.1
    i = 0
    for _ in range(71):  # segment A: 7.0s of main
        rows.append(row(i, i * dt)); calib.append({"frame_idx": i, "t": i * dt, "H": H,
                                                   "n_kp": 12, "rmse_m": 0.4, "src": "fit"})
        i += 1
    for _ in range(10):  # 1.0s replay -> shot break (splits even though < 2s)
        rows.append(row(i, i * dt, shot="replay")); i += 1
    for _ in range(61):  # segment B: 6.0s of main
        rows.append(row(i, i * dt)); calib.append({"frame_idx": i, "t": i * dt, "H": H,
                                                   "n_kp": 12, "rmse_m": 0.4, "src": "ema"})
        i += 1
    i += 30              # 3.0s skipped region (> 2s t-gap)
    for _ in range(30):  # 2.9s of main -> below 5s minimum, dropped
        rows.append(row(i, i * dt)); calib.append({"frame_idx": i, "t": i * dt, "H": H,
                                                   "n_kp": 12, "rmse_m": 0.4, "src": "held"})
        i += 1

    tmp = tempfile.mkdtemp()
    pj, cj = os.path.join(tmp, "p.jsonl"), os.path.join(tmp, "c.jsonl")
    with open(pj, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(cj, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in calib)

    segs = build_frames(pj, cj)
    assert len(segs) == 2, f"expected 2 segments, got {len(segs)}"
    assert abs((segs[0][-1].t - segs[0][0].t) - 7.0) < 1e-6
    assert abs((segs[1][-1].t - segs[1][0].t) - 6.0) < 1e-6

    f0 = segs[0][0]
    ids = {p.track_id for p in f0.players}
    assert "T7" not in ids, "referee must be dropped"
    assert "T8" not in ids, "out-of-bounds point must be dropped"
    assert ids == {"T1", "T2", "T3", "T4", "T5", "T6"}
    by_id = {p.track_id: p for p in f0.players}
    assert by_id["T5"].team == HOME, "left gk folds into home (defends x=0)"
    assert by_id["T6"].team == AWAY, "right gk folds into away"
    assert abs(by_id["T1"].x - 30.0) < 1e-6 and abs(by_id["T1"].y - 30.0) < 1e-6
    assert f0.ball is not None and abs(f0.ball[0] - 52.5) < 1e-6 and abs(f0.ball[1] - 34.0) < 1e-6
    for seg in segs:
        att = seg[0].attacking
        assert att[HOME] == 1 and att[AWAY] == -1, att
        assert att[HOME] == -att[AWAY]

    stats = frames_stats(segs)
    assert stats["n_segments"] == 2 and stats["ball_coverage"] == 1.0
    assert stats["mean_players_per_frame"] == 6.0
    print("synthetic self-check OK:", stats)

    real = os.path.join(os.path.dirname(__file__), "..", "..", "out", "bundesliga_smoke", "calib.jsonl")
    real = os.path.normpath(real)
    if os.path.exists(real):
        p = os.path.join(os.path.dirname(real), "perception.jsonl")
        b = os.path.join(os.path.dirname(real), "ball.jsonl")
        segs = build_frames(p, real, b if os.path.exists(b) else None)
        print("bundesliga_smoke:", frames_stats(segs))
    else:
        print("bundesliga_smoke/calib.jsonl not present yet; real-data run left to integrator")
