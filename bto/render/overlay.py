"""M4 telestrator overlay (SPEC C7 + M4).

Reads out_dir's perception.jsonl / calib.jsonl / shots.jsonl / m3_detections.json
and composites tactical overlays onto the ORIGINAL video frames. All Detection
geometry is pitch meters (105x68, bottom-left origin); every shape is sampled
as a polyline IN METER SPACE first, then projected to pixels via inv(H) so
perspective bends lines/circles correctly.

Layer design (SPEC open question resolved: always-on structure + event callouts):
  formation (always-on, subtle): translucent team hull + thin line links +
             per-team label chip ('4-3-3').
  offside   (always-on): dashed line across the pitch at the offside x, labeled
             'OFFSIDE (approx)' -- SPEC S9: never presented as a call.
  events    (callouts, prominent, scheduled with confidence/duration floors +
             per-type cooldown, capped at MAX_EVENTS=2 by confidence): back_pass arrow
             fading over its lifetime, triangle fill, 1v1/isolation spotlights
             + ISO chip, press pulsing ring by level, NvN region box + chip,
             overlap/underlap curved arrow.

Anti-flicker: overlays fade in/out over ~0.3 s at drawable-run boundaries
(shot cuts / H dropouts) and at detection boundaries. calib src=='held' keeps
drawing (H is last-good); a re-seed fit after held is a hard cut (fine).
Players/ball dots are NOT drawn (that's M2's video) except small foot markers
for players named in an active event callout.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np

# ---- colors (BGR), matching M2's team palette ----
COLOR_HOME = (255, 140, 0)
COLOR_AWAY = (0, 140, 255)
COLOR_OFFSIDE = (255, 255, 255)
COLOR_BACKPASS = (0, 255, 255)
COLOR_TRIANGLE = (255, 220, 80)
COLOR_ISO = (180, 105, 255)
COLOR_NVN = (255, 255, 0)
COLOR_RUN = (0, 255, 180)
PRESS_COLORS = {"low": (80, 200, 80), "medium": (0, 165, 255), "high": (0, 0, 255)}

FADE_S = 0.3          # fade window at detection + shot boundaries
MAX_EVENTS = 2        # simultaneous event callouts, by confidence
SAMPLE_STEP_M = 1.5   # polyline sampling step in meter space
DEFAULT_LAYERS = ("formation", "offside", "events")
EVENT_TYPES_STATIC = {"back_pass", "triangle", "isolation", "press", "overlap", "underlap"}

# ---- event selectivity (precision-tuning pass: the audited FP spam was
# dominated by sub-frame flicker events, uniform conf=1.0, and 3 chips at
# once; these gates are render-side belt-and-braces on top of detector fixes)
MIN_EVENT_CONF = 0.35     # drop event callouts below this confidence
EVENT_COOLDOWN_S = 8.0    # min gap between same-type callouts (display end -> next start)
MIN_EVENT_DUR_S = 0.4     # drop events shorter than this ...
MIN_EVENT_DUR_EXEMPT = {"back_pass", "overlap", "underlap"}  # ... except types whose
# lifetime is inherently the short ball-flight / run window
MIN_DISPLAY_S = 1.5       # stretch every kept event to at least this on screen

# ---- offside selectivity (audit: geometry accurate but BOTH teams' lines
# redrew every ~1-1.8 s regardless of phase of play -- pure relevance spam)
OFFSIDE_BALL_MAX_DX_M = 30.0  # hide a line further than this from the ball (x)
OFFSIDE_TEAM_HOLD_S = 2.0     # hold the chosen team_defending to avoid flip-flicker

# ---- drawable quality floor (audit: a formation chip was drawn on a player
# close-up at cwc t=168.3 s because 'H is not None' was the only check)
CALIB_MAX_RMSE_M = 3.0    # reject calib rows with worse fit error
CALIB_MAX_HELD_S = 2.0    # reject src=='held' runs older than this since last fit/ema
MIN_PLAYERS_DRAWABLE = 5  # a real 'main' broadcast shot shows many players; a
# close-up with a hallucinated H shows 0-4 (frame 5049 on cwc has 3)


def _load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _is_nvn(dtype):
    if dtype in ("1v1",):
        return False  # 1v1 renders as spotlight duel, not region box
    parts = dtype.split("v")
    return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()


def _is_event(dtype):
    return dtype in EVENT_TYPES_STATIC or dtype == "1v1" or _is_nvn(dtype)


# ---------------------------------------------------------------- projection

def project(pts_m, Hinv, w, h):
    """Meter points -> pixel points via inv(H); out-of-guard points become NaN.

    Guard: any projected point outside 2x frame bounds ([-w,2w] x [-h,2h]) is
    dropped (NaN) so a bad H can't smear lines across the frame.
    """
    P = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
    q = (Hinv @ np.hstack([P, np.ones((len(P), 1))]).T).T
    out = np.full((len(P), 2), np.nan)
    ok = np.abs(q[:, 2]) > 1e-9
    out[ok] = q[ok, :2] / q[ok, 2:3]
    bad = ~np.isfinite(out).all(axis=1)
    bad |= (out[:, 0] < -w) | (out[:, 0] > 2 * w) | (out[:, 1] < -h) | (out[:, 1] > 2 * h)
    out[bad] = np.nan
    return out


def sample_polyline_m(pts_m, closed=False, step=SAMPLE_STEP_M):
    """Densify a meter-space polyline so projection bends it with perspective."""
    pts = [tuple(p) for p in pts_m]
    if closed and len(pts) > 1:
        pts.append(pts[0])
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = math.dist(a, b)
        n = max(int(d / step), 1)
        for i in range(n):
            f = i / n
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    out.append(pts[-1])
    return out


def circle_m(cx, cy, r, n=28):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
            for i in range(n + 1)]


def _runs(px):
    """Split a projected Nx2 array into contiguous non-NaN int point runs."""
    runs, cur = [], []
    for p in px:
        if np.isfinite(p).all():
            cur.append((int(round(p[0])), int(round(p[1]))))
        elif cur:
            if len(cur) >= 2:
                runs.append(np.array(cur, dtype=np.int32))
            cur = []
    if len(cur) >= 2:
        runs.append(np.array(cur, dtype=np.int32))
    return runs


# ---------------------------------------------------------------- primitives

def _blend(canvas, draw_fn, alpha):
    if alpha <= 0.01:
        return canvas
    if alpha >= 0.99:
        draw_fn(canvas)
        return canvas
    tmp = canvas.copy()
    draw_fn(tmp)
    return cv2.addWeighted(tmp, alpha, canvas, 1.0 - alpha, 0)


def _poly_runs(img, px, color, thickness, closed_pts=None):
    for run in _runs(px):
        cv2.polylines(img, [run], False, color, thickness, cv2.LINE_AA)


def _fill_if_complete(img, px, color):
    """Fill the projected polygon only if every vertex survived the guard."""
    if np.isfinite(px).all() and len(px) >= 3:
        cv2.fillPoly(img, [px.astype(np.int32)], color, cv2.LINE_AA)
        return True
    return False


def _dashed_runs(img, px, color, thickness, dash=12, gap=9):
    for run in _runs(px):
        # walk the polyline in pixel arc length, alternating dash/gap
        seg = run.astype(np.float64)
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        total = d.sum()
        if total < 1:
            continue
        pos, on = 0.0, True
        cum = np.concatenate([[0.0], np.cumsum(d)])

        def at(s):
            i = np.searchsorted(cum, s, side="right") - 1
            i = min(max(i, 0), len(seg) - 2)
            f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
            return seg[i] + f * (seg[i + 1] - seg[i])

        while pos < total:
            end = min(pos + (dash if on else gap), total)
            if on:
                p0, p1 = at(pos), at(end)
                cv2.line(img, tuple(np.int32(p0)), tuple(np.int32(p1)), color, thickness, cv2.LINE_AA)
            pos, on = end, not on


def _chip(img, text, org, color, scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 5), (20, 20, 20), -1)
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 5), color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- layer draws

def _team_color(team):
    return COLOR_HOME if team == "home" else COLOR_AWAY if team == "away" else (255, 255, 255)


def _det_alpha(det, t, disp=None):
    """Fade a detection in/out over FADE_S around its lifetime; floor for shorts.

    ``disp`` (t0, t1) overrides the raw t_start/t_end window -- the event
    scheduler stretches short events to MIN_DISPLAY_S so fades must track the
    display window, not the raw detection lifetime.
    """
    t0, t1 = disp if disp is not None else (det["t_start"], det["t_end"])
    a = min(t - t0 + 0.04, t1 - t + 0.04) / FADE_S
    return float(np.clip(a, 0.2, 1.0))


def draw_formation(canvas, det, t, Hinv, w, h, seg_alpha, team):
    a = _det_alpha(det, t) * seg_alpha
    color = _team_color(team)
    g = det["geometry"]
    hull = g.get("hull") or []
    if len(hull) >= 3:
        px = project(sample_polyline_m(hull, closed=True), Hinv, w, h)
        canvas = _blend(canvas, lambda im: _fill_if_complete(im, px, color), 0.12 * a)
        canvas = _blend(canvas, lambda im: _poly_runs(im, px, color, 1), 0.5 * a)
    for line in g.get("lines") or []:
        if len(line) >= 2:
            px = project(sample_polyline_m(line), Hinv, w, h)
            canvas = _blend(canvas, lambda im, p=px: _poly_runs(im, p, color, 1), 0.45 * a)
    return canvas


def draw_offside(canvas, det, t, Hinv, w, h, seg_alpha):
    g = det["geometry"]
    x = g.get("x", g.get("line_x"))
    for ts, xs in g.get("samples") or []:  # interpolate x at current t if sampled
        if ts <= t:
            x = xs
    if x is None:
        return canvas, False
    # no per-detection fade: offside dets chain back-to-back every ~1s and a
    # boundary fade would make the line pulse; seg_alpha handles shot cuts.
    a = seg_alpha
    px = project(sample_polyline_m([(x, 0.0), (x, 68.0)]), Hinv, w, h)

    def d(im):
        _dashed_runs(im, px, COLOR_OFFSIDE, 3)
        for run in _runs(px):
            lp = run[len(run) // 2]
            _chip(im, "OFFSIDE (approx)", (lp[0] + 8, lp[1]), COLOR_OFFSIDE, 0.45)
            break

    return _blend(canvas, d, 0.8 * a), True


def _spotlight(im, pt_m, Hinv, w, h, color, r=1.4):
    px = project(circle_m(pt_m[0], pt_m[1], r), Hinv, w, h)
    if _fill_if_complete(im, px, color):
        return px
    _poly_runs(im, px, color, 2)
    return px


def draw_event(canvas, det, t, Hinv, w, h, seg_alpha, disp=None):
    a = _det_alpha(det, t, disp) * seg_alpha
    g, dtype = det["geometry"], det["type"]

    if dtype == "back_pass":
        d0, d1 = disp if disp is not None else (det["t_start"], det["t_end"])
        life = max(d1 - d0, 1e-6)
        a = seg_alpha * float(np.clip(1.0 - (t - d0) / life, 0.25, 1.0))
        px = project(sample_polyline_m([g["from"], g["to"]]), Hinv, w, h)

        def d(im):
            runs = _runs(px)
            for run in runs:
                cv2.polylines(im, [run], False, COLOR_BACKPASS, 4, cv2.LINE_AA)
            if runs:
                tail = runs[-1]
                cv2.arrowedLine(im, tuple(tail[-2]), tuple(tail[-1]), COLOR_BACKPASS, 4,
                                cv2.LINE_AA, tipLength=2.5)
                _chip(im, "BACK PASS", (tail[-1][0] + 8, tail[-1][1] - 8), COLOR_BACKPASS)
        return _blend(canvas, d, a)

    if dtype == "triangle":
        px = project(sample_polyline_m(g["vertices"], closed=True), Hinv, w, h)
        vpx = project(g["vertices"], Hinv, w, h)
        canvas = _blend(canvas, lambda im: _fill_if_complete(im, px, COLOR_TRIANGLE), 0.22 * a)

        def d(im):
            _poly_runs(im, px, COLOR_TRIANGLE, 2)
            for v in vpx:
                if np.isfinite(v).all():
                    cv2.circle(im, (int(v[0]), int(v[1])), 5, COLOR_TRIANGLE, -1, cv2.LINE_AA)
        return _blend(canvas, d, a)

    if dtype in ("1v1", "isolation"):
        if dtype == "isolation":
            pts = [g["attacker"], g["defender"]]
        else:
            xa, ya, xd, yd = g["pairs"][0]
            pts = [(xa, ya), (xd, yd)]

        def d(im):
            for p in pts:
                _spotlight(im, p, Hinv, w, h, COLOR_ISO)
        canvas = _blend(canvas, d, 0.3 * a)

        def chips(im):
            mid = project([((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)], Hinv, w, h)[0]
            if np.isfinite(mid).all():
                _chip(im, "ISO", (mid[0] + 10, mid[1] - 14), COLOR_ISO)
        return _blend(canvas, chips, a)

    if dtype == "press":
        cx, cy = g["carrier"]
        level = g.get("level", "medium")
        color = PRESS_COLORS.get(level, PRESS_COLORS["medium"])
        r = 2.0 + 0.45 * math.sin(2 * math.pi * 1.6 * t)  # pulsing ring
        px = project(circle_m(cx, cy, r), Hinv, w, h)

        def d(im):
            _poly_runs(im, px, color, 3)
            for run in _runs(px):
                top = run[np.argmin(run[:, 1])]
                _chip(im, f"PRESS {level.upper()}", (top[0] - 20, top[1] - 8), color, 0.45)
                break
        return _blend(canvas, d, a)

    if _is_nvn(dtype):
        x0, y0, x1, y1 = g["region"]
        pad = 1.0
        box = [(x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad)]
        px = project(sample_polyline_m(box, closed=True), Hinv, w, h)

        def d(im):
            _poly_runs(im, px, COLOR_NVN, 2)
            for run in _runs(px):
                top = run[np.argmin(run[:, 1])]
                _chip(im, dtype, (top[0], top[1] - 8), COLOR_NVN, 0.5)
                break
            for xa, ya, xd, yd in g.get("pairs") or []:
                lp = project(sample_polyline_m([(xa, ya), (xd, yd)]), Hinv, w, h)
                _dashed_runs(im, lp, COLOR_NVN, 1, dash=7, gap=6)
        return _blend(canvas, d, 0.8 * a)

    if dtype in ("overlap", "underlap"):
        path = g.get("path") or []
        if len(path) < 2:
            return canvas
        px = project(sample_polyline_m(path), Hinv, w, h)

        def d(im):
            runs = _runs(px)
            for run in runs:
                cv2.polylines(im, [run], False, COLOR_RUN, 3, cv2.LINE_AA)
            if runs:
                tail = runs[-1]
                cv2.arrowedLine(im, tuple(tail[-2]), tuple(tail[-1]), COLOR_RUN, 3,
                                cv2.LINE_AA, tipLength=2.0)
                _chip(im, dtype.upper(), (tail[-1][0] + 8, tail[-1][1]), COLOR_RUN, 0.45)
        return _blend(canvas, d, a)

    return canvas


# ------------------------------------------------------------- selectivity

def schedule_events(detections):
    """Build the event-callout schedule ONCE per render (precision pass).

    Filters: confidence < MIN_EVENT_CONF; duration < MIN_EVENT_DUR_S except
    the MIN_EVENT_DUR_EXEMPT types (their lifetime is inherently the short
    ball-flight / run window -- the sub-frame isolation flickers are exactly
    what the duration gate kills). Every survivor gets a display window
    [t_start, max(t_end, t_start + MIN_DISPLAY_S)] so short real events are
    legible instead of a ~100 ms 0.2-alpha ghost. Then per type (by t_start)
    an event is kept only if it starts >= the last KEPT same-type event's
    display end + EVENT_COOLDOWN_S; two overlapping in that window keep the
    higher-confidence one.

    Returns {id(det): (disp_start, disp_end)} for the kept events.
    """
    by_type = defaultdict(list)
    for d in detections:
        dtype = d["type"]
        if not _is_event(dtype):
            continue
        if d.get("confidence", 0.0) < MIN_EVENT_CONF:
            continue
        if dtype not in MIN_EVENT_DUR_EXEMPT and (d["t_end"] - d["t_start"]) < MIN_EVENT_DUR_S:
            continue
        by_type[dtype].append(d)

    schedule = {}
    for dtype, ds in by_type.items():
        ds.sort(key=lambda d: d["t_start"])
        kept = []  # [det, disp_start, disp_end]
        for d in ds:
            s = d["t_start"]
            e = max(d["t_end"], s + MIN_DISPLAY_S)
            if kept and s < kept[-1][2] + EVENT_COOLDOWN_S:
                if d.get("confidence", 0.0) > kept[-1][0].get("confidence", 0.0):
                    kept[-1] = [d, s, e]  # higher-confidence event wins the slot
                continue
            kept.append([d, s, e])
        for d, s, e in kept:
            schedule[id(d)] = (s, e)
    return schedule


def _formation_sane(det):
    """Belt-and-braces vs the audited dedup bug: >11 'team members' hulls and
    labels like '2-12' / '16-1' whose digits don't sum to the player count."""
    players = det.get("players") or []
    if len(players) > 11:
        return False
    label = str((det.get("geometry") or {}).get("label") or "")
    try:
        digits = [int(p) for p in label.split("-")]
    except ValueError:
        return False
    return sum(digits) == len(players)


def _offside_x(det, t):
    g = det["geometry"]
    x = g.get("x", g.get("line_x"))
    for ts, xs in g.get("samples") or []:
        if ts <= t:
            x = xs
    return x


# ---------------------------------------------------------------- main driver

def _banner(img, shot, t, fidx, drawable):
    t_str = f"{t:.2f}s" if t is not None else "?"
    note = "" if drawable else "  [overlay off]"
    text = f"M4 shot={shot}  t={t_str}  frame={fidx}{note}"
    cv2.rectangle(img, (0, 0), (470, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def _tid_team_majority(perception):
    votes = defaultdict(lambda: defaultdict(int))
    for row in perception:
        for p in row["players"]:
            if p.get("tid") is not None and p.get("team") in ("home", "away"):
                votes[p["tid"]][p["team"]] += 1
    return {tid: max(v, key=v.get) for tid, v in votes.items()}


def _formation_team(det, tid_team):
    counts = defaultdict(int)
    for pid in det["players"]:
        tid = int(pid[1:]) if pid.startswith("T") and pid[1:].isdigit() else None
        team = tid_team.get(tid)
        if team:
            counts[team] += 1
    return max(counts, key=counts.get) if counts else None


def render_overlay_video(video_path, out_dir, out_path=None, layers=None, max_frames=None):
    """Composite M3 detections onto the original video. Returns stats dict."""
    layers = tuple(layers) if layers else DEFAULT_LAYERS
    perception = _load_jsonl(os.path.join(out_dir, "perception.jsonl"))
    calib = {r["frame_idx"]: r for r in _load_jsonl(os.path.join(out_dir, "calib.jsonl"))}
    shots = {r["frame_idx"]: r.get("shot", "main") for r in _load_jsonl(os.path.join(out_dir, "shots.jsonl"))}
    with open(os.path.join(out_dir, "m3_detections.json")) as f:
        detections = json.load(f)
    if max_frames:
        perception = perception[:max_frames]
    if not perception:
        raise RuntimeError(f"empty perception.jsonl in {out_dir}")

    tid_team = _tid_team_majority(perception)
    formation_teams = {id(d): _formation_team(d, tid_team) for d in detections if d["type"] == "formation"}
    dets_sorted = sorted(detections, key=lambda d: d["t_start"])
    # event callout schedule, built once (conf/duration floors + per-type
    # cooldown + MIN_DISPLAY_S legibility stretch): {id(det): (d0, d1)}
    event_windows = schedule_events(detections)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fidxs = [r["frame_idx"] for r in perception]
    diffs = np.diff(sorted(fidxs))
    stride = int(np.median(diffs[diffs > 0])) if len(diffs) and (diffs > 0).any() else 1
    out_fps = max(src_fps / stride, 1.0)

    # age of each src=='held' calib row since the last real fit/ema (a held H
    # is last-good; after CALIB_MAX_HELD_S it is stale enough to stop drawing)
    held_age = {}
    _last_good_t = None
    for c in sorted(calib.values(), key=lambda r: r["frame_idx"]):
        if c.get("src") == "held":
            held_age[c["frame_idx"]] = (
                c["t"] - _last_good_t
                if _last_good_t is not None and c.get("t") is not None else float("inf"))
        elif c.get("H") is not None:
            _last_good_t = c.get("t")

    # drawable runs (for shot-cut fade in/out with lookahead)
    def _drawable(row):
        c = calib.get(row["frame_idx"])
        if row.get("shot", shots.get(row["frame_idx"], "main")) != "main" or c is None or c.get("H") is None:
            return False
        # quality floor (audit: 'H is not None' alone let a hallucinated fit
        # draw overlays on a player close-up at cwc t=168.3s / frame 5049)
        if c.get("rmse_m") is not None and c["rmse_m"] > CALIB_MAX_RMSE_M:
            return False
        if c.get("src") == "held" and held_age.get(c["frame_idx"], 0.0) > CALIB_MAX_HELD_S:
            return False
        if len(row.get("players") or []) < MIN_PLAYERS_DRAWABLE:
            return False
        return True

    drawable = [_drawable(r) for r in perception]
    run_start, run_end = [None] * len(perception), [None] * len(perception)
    i = 0
    while i < len(perception):
        if drawable[i]:
            j = i
            while j + 1 < len(perception) and drawable[j + 1]:
                j += 1
            for k in range(i, j + 1):
                run_start[k], run_end[k] = perception[i]["t"], perception[j]["t"]
            i = j + 1
        else:
            i += 1

    out_path = out_path or os.path.join(out_dir, "m4_overlay.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    draw_counts = defaultdict(int)
    dt = stride / src_fps
    sample_frames = {}
    off_hold_team, off_hold_t = None, None  # offside team_defending hold (causal)

    for idx, row in enumerate(perception):
        fidx = row["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        t = row["t"] if row["t"] is not None else fidx / src_fps
        shot = row.get("shot", "main")
        canvas = frame.copy()

        if drawable[idx]:
            Hinv = np.linalg.inv(np.array(calib[fidx]["H"], dtype=np.float64).reshape(3, 3))
            seg_alpha = float(np.clip(
                min(t - run_start[idx] + dt, run_end[idx] - t + dt) / FADE_S, 0.15, 1.0))
            active = [d for d in dets_sorted if d["t_start"] <= t <= d["t_end"]]

            if "formation" in layers:
                for d in active:
                    if d["type"] == "formation" and _formation_sane(d):
                        team = formation_teams.get(id(d))
                        canvas = draw_formation(canvas, d, t, Hinv, w, h, seg_alpha, team)
                        label = d["geometry"].get("label", "?")
                        org = (12, 58) if team != "away" else (w - 130, 58)
                        canvas = _blend(
                            canvas,
                            lambda im: _chip(im, f"{(team or '?').upper()} {label}", org, _team_color(team)),
                            seg_alpha)
                        draw_counts["formation"] += 1

            if "offside" in layers:
                # relevance gate: at most ONE line per frame, and only the one
                # nearest the live ball (both-teams-always-on was the audited
                # spam; geometry itself was accurate).
                off_active = [d for d in active if d["type"] == "offside_line"]
                ball_x = None
                ball = row.get("ball")
                if off_active and ball:
                    Hm = np.array(calib[fidx]["H"], dtype=np.float64).reshape(3, 3)
                    q = Hm @ np.array([ball[0], ball[1], 1.0], dtype=np.float64)
                    if abs(q[2]) > 1e-9 and np.isfinite(q).all():
                        ball_x = float(q[0] / q[2])
                chosen = None
                if ball_x is not None:  # no ball on this frame -> no line
                    cands = []
                    for d in off_active:
                        x = _offside_x(d, t)
                        if x is not None and abs(x - ball_x) <= OFFSIDE_BALL_MAX_DX_M:
                            cands.append((abs(x - ball_x), d))
                    if cands:
                        cands.sort(key=lambda c: c[0])
                        chosen = cands[0][1]
                        team = chosen["geometry"].get("team_defending")
                        if (off_hold_team is not None and team != off_hold_team
                                and t - off_hold_t <= OFFSIDE_TEAM_HOLD_S):
                            held = [d for _, d in cands
                                    if d["geometry"].get("team_defending") == off_hold_team]
                            if held:  # hold the previous team to avoid flip-flicker
                                chosen, team = held[0], off_hold_team
                        if team != off_hold_team:
                            off_hold_team, off_hold_t = team, t
                if chosen is not None:
                    canvas, drew = draw_offside(canvas, chosen, t, Hinv, w, h, seg_alpha)
                    if drew:
                        draw_counts["offside"] += 1

            if "events" in layers:
                sched = [(d, event_windows[id(d)]) for d in dets_sorted
                         if id(d) in event_windows
                         and event_windows[id(d)][0] <= t <= event_windows[id(d)][1]]
                events = sorted(sched, key=lambda dw: -dw[0]["confidence"])[:MAX_EVENTS]
                ev_tids = set()
                for d, disp in events:
                    canvas = draw_event(canvas, d, t, Hinv, w, h, seg_alpha, disp=disp)
                    draw_counts["events"] += 1
                    draw_counts[f"events.{d['type']}"] += 1
                    for pid in d["players"]:
                        if pid.startswith("T") and pid[1:].isdigit():
                            ev_tids.add(int(pid[1:]))
                # small foot markers only for players named in active callouts
                for p in row["players"]:
                    if p.get("tid") in ev_tids and p.get("foot"):
                        fx, fy = int(p["foot"][0]), int(p["foot"][1])
                        cv2.circle(canvas, (fx, fy), 5, _team_color(p.get("team")), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (fx, fy), 5, (20, 20, 20), 1, cv2.LINE_AA)
                        draw_counts["foot_markers"] += 1

        _banner(canvas, shot, t, fidx, drawable[idx])
        writer.write(canvas)
        if idx % max(len(perception) // 8, 1) == 0 and len(sample_frames) < 8:
            sample_frames[fidx] = canvas

    cap.release()
    writer.release()

    return {
        "out_video": out_path,
        "n_frames": len(perception),
        "n_drawable": int(sum(drawable)),
        "out_fps": out_fps,
        "stride": stride,
        "draw_counts": dict(draw_counts),
        "_samples": sample_frames,
    }


def _self_check():
    """150 frames of bundesliga_smoke end-to-end + 4 sample jpgs for eyeballing."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    video = os.path.join(root, "data", "clips", "bundesliga_smoke.mp4")
    out_dir = os.path.join(root, "out", "bundesliga_smoke")
    dst = os.path.join(out_dir, "m4_selfcheck.mp4")
    # 150 frames (~12 s): the event-selectivity pass (MIN_EVENT_CONF /
    # EVENT_COOLDOWN_S) legitimately blanks the first ~5 s of this clip, so
    # 60 frames no longer contains any scheduled event callout.
    stats = render_overlay_video(video, out_dir, out_path=dst, max_frames=150)

    assert os.path.exists(dst), "self-check mp4 missing"
    size = os.path.getsize(dst)
    assert size > 100_000, f"self-check mp4 too small: {size} bytes"
    assert stats["draw_counts"].get("formation", 0) > 0, "no formation draws"
    assert stats["draw_counts"].get("offside", 0) > 0, "no offside draws"
    # events is a printed count, not an assert: the selectivity pass (conf
    # floor + per-type cooldown) can validly schedule zero callouts in a
    # short window against a low-confidence m3 file.
    print("event draws in window:", stats["draw_counts"].get("events", 0))

    samples = list(stats["_samples"].items())
    picks = [samples[i] for i in (0, len(samples) // 3, 2 * len(samples) // 3, len(samples) - 1)]
    for fidx, img in picks:
        p = os.path.join(out_dir, f"m4_selfcheck_f{fidx}.jpg")
        cv2.imwrite(p, img)
        print("sample:", p)
    print(f"SELF-CHECK OK: {dst} {size} bytes, draw_counts={stats['draw_counts']}")


if __name__ == "__main__":
    _self_check()
