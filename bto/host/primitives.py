"""SPEC C10/S4.1: Detection -> frozen WS protocol PRIM geometry.

Ports the meter->pixel drawing semantics of ``bto.render.overlay`` (M4,
batch/offline) into the frozen WS protocol used by the live host
(SPEC S4.1 / S7 near-live): instead of alpha-compositing shapes onto a
video frame with cv2, every layer is expressed as a JSON-serialisable PRIM
dict the browser extension draws on a `<canvas>`.

    PRIM = {"gid": str, "kind": "polygon"|"polyline"|"circle"|"arrow"|
            "label"|"chip", "pts": [[x,y], ...], "color": [r,g,b],
            "alpha": f, "width": f, "text": str|None, "fill": bool,
            "dash": bool}

We deliberately do NOT re-implement the meter->pixel math: ``project``,
``sample_polyline_m``, ``circle_m`` and the BGR colour constants are
imported straight from ``bto.render.overlay``. ``project()`` already NaN's
out any point that lands outside a generous [-w,2w]x[-h,2h] guard band, so
a single ``np.isfinite(...).all()`` check on the projected points is the
"clip/NaN-guard, drop the whole prim" rule required here -- overlay.py
instead keeps partial polyline *runs*, which the batch cv2 renderer needs
for a video but a browser-side PRIM list does not (the client would rather
not draw a shape at all than draw a broken one).

Two entry points:

  detections_to_prims(detections, H, t, w, h) -> list[PRIM]
      The documented contract (SPEC S4.1). ``detections`` is a
      ``list[bto.core.Detection]`` (attribute access: .type/.players/
      .geometry/.confidence/.t_start/.t_end) covering (at least) every
      detection active at time ``t``; ``H`` is the per-frame pixel->pitch
      calibration homography (3x3, row-major, as stored in calib.jsonl --
      the same convention bto.render.overlay consumes: prims are projected
      with inv(H)); ``w``/``h`` are the pixel dimensions of the JPEG frame
      the client sent (PRIM pts are in that space).

  detections_to_prims(bundle) -> list[PRIM]
      Convenience form for bto.host.server's current stub-tolerant call
      site (`fn(detections)`, single positional arg -- see its
      `_call_detections_to_prims`). If the first argument is a dict and no
      H/t/w/h are supplied, it is treated as
      {"detections": [...], "H": ..., "t": ..., "w": ..., "h": ...} and
      unpacked. Missing H/w/h -> no geometry can be projected -> [].

Draw-order / cap rules mirror overlay.py exactly: formation + offside_line
are always-on layers (every active instance drawn); everything else is an
"event" (back_pass, triangle, isolation, 1v1, NvN, press, overlap,
underlap), filtered by the shared selectivity floors (MIN_EVENT_CONF,
MIN_EVENT_DUR_S with the back_pass/overlap/underlap exemption) and capped
at the imported MAX_EVENTS by confidence, drawn in that order. "block"
detections are not rendered (overlay.py doesn't draw them either).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from bto.render.overlay import (
    COLOR_BACKPASS,
    COLOR_HOME,
    COLOR_AWAY,
    COLOR_ISO,
    COLOR_NVN,
    COLOR_OFFSIDE,
    COLOR_RUN,
    COLOR_TRIANGLE,
    FADE_S,
    MAX_EVENTS,
    MIN_EVENT_CONF,
    MIN_EVENT_DUR_EXEMPT,
    MIN_EVENT_DUR_S,
    PRESS_COLORS,
    circle_m,
    project,
    sample_polyline_m,
)

EVENT_TYPES = {"back_pass", "triangle", "isolation", "press", "overlap", "underlap"}


def _is_nvn_or_1v1(dtype: str) -> bool:
    parts = dtype.split("v")
    return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()


def _is_event(dtype: str) -> bool:
    return dtype in EVENT_TYPES or _is_nvn_or_1v1(dtype)


def _get(det: Any, name: str, default=None):
    """Attribute access for bto.core.Detection, falling back to dict keys
    (e.g. a JSON round-trip of the same shape) so the self-check and any
    future JSON-fed caller both work."""
    if hasattr(det, name):
        return getattr(det, name)
    if isinstance(det, dict):
        return det.get(name, default)
    return default


def _rgb(bgr) -> list[int]:
    b, g, r = bgr
    return [int(r), int(g), int(b)]


def _bgr(rgb) -> tuple[int, int, int]:
    r, g, b = rgb
    return (int(b), int(g), int(r))


# How long past t_end a detection stays drawable. Matches the live
# pipeline's StreamPipeline(det_grace_s=...) default: bto.host.stream serves
# cached run_all detections up to this long past their t_end so the overlay
# does not strobe between pattern runs; a fresh live detection always has
# t ~= t_end, so fading against the raw t_end floors every prim at the
# minimum alpha (M5 integration fix).
DET_GRACE_S = 0.75


def _fade_alpha(det: Any, t: float) -> float:
    t_start, t_end = _get(det, "t_start"), _get(det, "t_end")
    a = min(t - t_start + 0.04, (t_end + DET_GRACE_S) - t + 0.04) / FADE_S
    return float(np.clip(a, 0.2, 1.0))


def _proj(pts_m: Sequence[tuple[float, float]], Hinv, w: int, h: int) -> list[list[float]] | None:
    """Project meter pts -> pixel pts; None (drop the prim) if ANY point is
    invalid (NaN-guarded by overlay.project's out-of-bounds check)."""
    if not pts_m:
        return None
    px = project(pts_m, Hinv, w, h)
    if not np.isfinite(px).all():
        return None
    return [[round(float(x), 1), round(float(y), 1)] for x, y in px]


def _prim(gid, kind, pts, color, alpha=1.0, width=1.5, text=None, fill=False, dash=False):
    return {
        "gid": gid,
        "kind": kind,
        "pts": pts,
        "color": color,
        "alpha": round(float(np.clip(alpha, 0.0, 1.0)), 3),
        "width": float(width),
        "text": text,
        "fill": bool(fill),
        "dash": bool(dash),
    }


def _players_tag(det: Any, n: int = 2) -> str:
    players = _get(det, "players") or []
    return "-".join(sorted(str(p) for p in players)[:n])


def _base_gid(det: Any) -> str:
    t_start = _get(det, "t_start")
    return f"{_get(det, 'type')}:{t_start:.2f}"


# --------------------------------------------------------------- per-type


def _prims_formation(det, t, Hinv, w, h, color) -> list[dict]:
    a = _fade_alpha(det, t)
    g = _get(det, "geometry") or {}
    base = _base_gid(det) + ":" + _players_tag(det)
    out = []

    hull = g.get("hull") or []
    if len(hull) >= 3:
        pts = _proj(sample_polyline_m(hull, closed=True), Hinv, w, h)
        if pts:
            out.append(_prim(base + ":hull", "polygon", pts, _rgb(color), alpha=0.12 * a, fill=True))

    for i, line in enumerate(g.get("lines") or []):
        if len(line) < 2:
            continue
        pts = _proj(sample_polyline_m(line), Hinv, w, h)
        if pts:
            out.append(_prim(f"{base}:line{i}", "polyline", pts, _rgb(color), alpha=0.5 * a, width=1.5))

    anchor = _proj([hull[0]], Hinv, w, h) if hull else None
    if anchor:
        label = g.get("label", "?")
        out.append(_prim(base + ":chip", "chip", anchor, _rgb(color), alpha=a, text=label))
    return out


def _prims_offside(det, t, Hinv, w, h) -> list[dict]:
    g = _get(det, "geometry") or {}
    x = g.get("x", g.get("line_x"))
    for ts, xs in g.get("samples") or []:
        if ts <= t:
            x = xs
    if x is None:
        return []
    gid = f"{_base_gid(det)}:{g.get('team_defending', '?')}"
    pts = _proj(sample_polyline_m([(x, 0.0), (x, 68.0)]), Hinv, w, h)
    if not pts:
        return []
    out = [_prim(gid + ":line", "polyline", pts, _rgb(COLOR_OFFSIDE), alpha=0.8, width=3, dash=True)]
    mid = pts[len(pts) // 2]
    out.append(_prim(gid + ":chip", "chip", [mid], _rgb(COLOR_OFFSIDE), alpha=0.8, text="OFFSIDE (approx)"))
    return out


def _prims_back_pass(det, t, Hinv, w, h) -> list[dict]:
    g = _get(det, "geometry") or {}
    t_start, t_end = _get(det, "t_start"), _get(det, "t_end")
    life = max(t_end - t_start, 1e-6)
    a = float(np.clip(1.0 - (t - t_start) / life, 0.25, 1.0))
    pts = _proj(sample_polyline_m([g["from"], g["to"]]), Hinv, w, h)
    if not pts:
        return []
    gid = _base_gid(det)
    out = [_prim(gid + ":arrow", "arrow", pts, _rgb(COLOR_BACKPASS), alpha=a, width=4)]
    out.append(_prim(gid + ":chip", "chip", [pts[-1]], _rgb(COLOR_BACKPASS), alpha=a, text="BACK PASS"))
    return out


def _prims_triangle(det, t, Hinv, w, h) -> list[dict]:
    a = _fade_alpha(det, t)
    g = _get(det, "geometry") or {}
    verts = g.get("vertices") or []
    if len(verts) < 3:
        return []
    gid = _base_gid(det)
    out = []
    poly = _proj(sample_polyline_m(verts, closed=True), Hinv, w, h)
    if poly:
        out.append(_prim(gid + ":poly", "polygon", poly, _rgb(COLOR_TRIANGLE), alpha=0.22 * a, fill=True))
    for i, v in enumerate(verts):
        c = _proj(circle_m(v[0], v[1], 0.6), Hinv, w, h)
        if c:
            out.append(_prim(f"{gid}:v{i}", "circle", c, _rgb(COLOR_TRIANGLE), alpha=a, fill=True))
    return out


def _prims_matchup(det, t, Hinv, w, h) -> list[dict]:
    """1v1 and NvN (2v2, 3v3, ...): both carry geometry={pairs, region}."""
    a = _fade_alpha(det, t)
    g = _get(det, "geometry") or {}
    dtype = _get(det, "type")
    gid = _base_gid(det) + ":" + _players_tag(det)
    out = []

    region = g.get("region")
    if region and len(region) == 4:
        x0, y0, x1, y1 = region
        pad = 1.0
        box = [(x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad)]
        pts = _proj(sample_polyline_m(box, closed=True), Hinv, w, h)
        if pts:
            out.append(_prim(gid + ":region", "polygon", pts, _rgb(COLOR_NVN), alpha=0.8 * a, width=2))

    for i, (xa, ya, xd, yd) in enumerate(g.get("pairs") or []):
        pts = _proj(sample_polyline_m([(xa, ya), (xd, yd)]), Hinv, w, h)
        if pts:
            out.append(_prim(f"{gid}:pair{i}", "polyline", pts, _rgb(COLOR_NVN), alpha=0.8 * a, width=1, dash=True))

    if region:
        cx, cy = (region[0] + region[2]) / 2.0, region[1] - 1.0
        anchor = _proj([(cx, cy)], Hinv, w, h)
        if anchor:
            out.append(_prim(gid + ":chip", "chip", anchor, _rgb(COLOR_NVN), alpha=a, text=dtype))
    return out


def _prims_isolation(det, t, Hinv, w, h) -> list[dict]:
    a = _fade_alpha(det, t)
    g = _get(det, "geometry") or {}
    attacker, defender = g.get("attacker"), g.get("defender")
    if attacker is None or defender is None:
        return []
    gid = _base_gid(det) + ":" + _players_tag(det)
    out = []
    for tag, pt in (("atk", attacker), ("def", defender)):
        c = _proj(circle_m(pt[0], pt[1], 1.4), Hinv, w, h)
        if c:
            out.append(_prim(f"{gid}:{tag}", "circle", c, _rgb(COLOR_ISO), alpha=0.3 * a, fill=True))
    mid = ((attacker[0] + defender[0]) / 2.0, (attacker[1] + defender[1]) / 2.0)
    anchor = _proj([mid], Hinv, w, h)
    if anchor:
        out.append(_prim(gid + ":chip", "chip", anchor, _rgb(COLOR_ISO), alpha=a, text="ISO"))
    return out


def _prims_press(det, t, Hinv, w, h) -> list[dict]:
    a = _fade_alpha(det, t)
    g = _get(det, "geometry") or {}
    carrier = g.get("carrier")
    if carrier is None:
        return []
    level = g.get("level", "medium")
    color = PRESS_COLORS.get(level, PRESS_COLORS["medium"])
    r = 2.0 + 0.45 * math.sin(2 * math.pi * 1.6 * t)
    pts = _proj(circle_m(carrier[0], carrier[1], r), Hinv, w, h)
    if not pts:
        return []
    gid = _base_gid(det)
    out = [_prim(gid + ":ring", "circle", pts, _rgb(color), alpha=a, width=3)]
    out.append(_prim(gid + ":chip", "chip", [pts[0]], _rgb(color), alpha=a, text=f"PRESS {level.upper()}"))
    return out


def _prims_run(det, t, Hinv, w, h) -> list[dict]:
    a = _fade_alpha(det, t)
    g = _get(det, "geometry") or {}
    path = g.get("path") or []
    if len(path) < 2:
        return []
    pts = _proj(sample_polyline_m(path), Hinv, w, h)
    if not pts:
        return []
    gid = _base_gid(det)
    dtype = _get(det, "type")
    out = [_prim(gid + ":arrow", "arrow", pts, _rgb(COLOR_RUN), alpha=a, width=3)]
    out.append(_prim(gid + ":chip", "chip", [pts[-1]], _rgb(COLOR_RUN), alpha=a, text=dtype.upper()))
    return out


_EVENT_BUILDERS = {
    "back_pass": _prims_back_pass,
    "triangle": _prims_triangle,
    "isolation": _prims_isolation,
    "press": _prims_press,
    "overlap": _prims_run,
    "underlap": _prims_run,
}


def _event_prims(det, t, Hinv, w, h) -> list[dict]:
    dtype = _get(det, "type")
    fn = _EVENT_BUILDERS.get(dtype)
    if fn is not None:
        return fn(det, t, Hinv, w, h)
    if _is_nvn_or_1v1(dtype):
        return _prims_matchup(det, t, Hinv, w, h)
    return []


# ------------------------------------------------------------------ public


def detections_to_prims(detections, H=None, t=None, w=None, h=None) -> list[dict]:
    """Detection stream (SPEC S6 output) -> frozen-protocol PRIM list.

    Primary contract: detections_to_prims(detections, H, t, w, h).
    Also tolerates detections_to_prims(bundle) where bundle is a dict
    {"detections": [...], "H": ..., "t": ..., "w": ..., "h": ...} -- the
    calling convention bto.host.server's stub-tolerant `fn(detections)`
    site currently uses.
    """
    if isinstance(detections, dict) and H is None:
        bundle = detections
        detections = bundle.get("detections") or []
        H = bundle.get("H", H)
        t = bundle.get("t", t)
        w = bundle.get("w", w)
        h = bundle.get("h", h)

    if H is None or w is None or h is None or t is None:
        return []
    detections = detections or []

    Hinv = np.linalg.inv(np.asarray(H, dtype=np.float64).reshape(3, 3))
    w, h = int(w), int(h)

    # Grace-extended activity window: the live pipeline intentionally serves
    # detections up to det_grace_s past t_end (cached between run_all calls);
    # re-filtering strictly on t_end here dropped ALL of them on the frames
    # between pattern runs and made the overlay strobe (M5 integration fix).
    active = [
        d for d in detections
        if _get(d, "t_start") <= t <= _get(d, "t_end") + DET_GRACE_S
    ]

    out: list[dict] = []

    # always-on layers, home/away colour assigned by encounter order within
    # this call (Detection carries no explicit team field -- see module
    # docstring on bto.core.Detection; two simultaneous formation dets in
    # the same window are HOME then AWAY by detector construction order).
    #
    # M5 integration fix: with the live pipeline's cached run_all output +
    # grace window, several GENERATIONS of the same always-on layer can be
    # active at once (an old window fading out next to the fresh one),
    # stacking 3-4 hulls / 4-6 offside lines per frame. Keep only the newest
    # generation: formation dets tied for max t_end (detector emits home
    # then away, so cap 2), offside lines deduped per defending team.
    formation_dets = [d for d in active if _get(d, "type") == "formation"]
    if formation_dets:
        t_new = max(_get(d, "t_end") for d in formation_dets)
        formation_dets = [d for d in formation_dets if _get(d, "t_end") >= t_new - 1e-9][:2]
    for i, d in enumerate(formation_dets):
        color = COLOR_HOME if i % 2 == 0 else COLOR_AWAY
        out.extend(_prims_formation(d, t, Hinv, w, h, color))

    newest_offside: dict = {}
    for d in active:
        if _get(d, "type") == "offside_line":
            key = (_get(d, "geometry") or {}).get("team_defending")
            prev = newest_offside.get(key)
            if prev is None or _get(d, "t_end") > _get(prev, "t_end"):
                newest_offside[key] = d
    for d in newest_offside.values():
        out.extend(_prims_offside(d, t, Hinv, w, h))

    # Precision-tuning selectivity (mirrors overlay.py's schedule pre-pass as
    # far as a stateless per-frame call allows): confidence floor + minimum
    # duration, with the same exemption for types whose lifetime is
    # inherently the short ball-flight / run window. No cross-call cooldown
    # state: this function is stateless per frame by design.
    def _keep_event(d):
        if _get(d, "confidence", 0.0) < MIN_EVENT_CONF:
            return False
        dtype = _get(d, "type")
        if dtype in MIN_EVENT_DUR_EXEMPT:
            return True
        return (_get(d, "t_end") - _get(d, "t_start")) >= MIN_EVENT_DUR_S

    events = sorted((d for d in active if _is_event(_get(d, "type")) and _keep_event(d)),
                    key=lambda d: -_get(d, "confidence", 0.0))
    for d in events[:MAX_EVENTS]:
        out.extend(_event_prims(d, t, Hinv, w, h))

    return out


# ------------------------------------------------------------------ debug renderer


_KIND_DEFAULT_COLOR = (255, 255, 255)


def prims_render_debug(frame_bgr: np.ndarray, prims: list[dict]) -> np.ndarray:
    """Tiny cv2 renderer of the protocol, for host-side debugging and the
    integrator's preview video. Arrows/dashes are approximated (no
    per-segment arc-length dash walk like overlay._dashed_runs)."""
    canvas = frame_bgr.copy()
    for p in prims or []:
        pts = p.get("pts") or []
        if not pts:
            continue
        color = _bgr(p.get("color") or _KIND_DEFAULT_COLOR)
        alpha = float(p.get("alpha", 1.0))
        width = max(int(round(p.get("width", 1.5))), 1)
        kind = p.get("kind")
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        layer = canvas.copy()
        if kind == "polygon":
            if p.get("fill"):
                cv2.fillPoly(layer, [arr.reshape(-1, 2)], color, cv2.LINE_AA)
            else:
                cv2.polylines(layer, [arr], True, color, width, cv2.LINE_AA)
        elif kind == "polyline":
            if p.get("dash"):
                ipts = arr.reshape(-1, 2)
                for i in range(0, len(ipts) - 1, 2):
                    j = min(i + 1, len(ipts) - 1)
                    cv2.line(layer, tuple(int(v) for v in ipts[i]), tuple(int(v) for v in ipts[j]),
                             color, width, cv2.LINE_AA)
            else:
                cv2.polylines(layer, [arr], False, color, width, cv2.LINE_AA)
        elif kind == "circle":
            if p.get("fill"):
                cv2.fillPoly(layer, [arr.reshape(-1, 2)], color, cv2.LINE_AA)
            else:
                cv2.polylines(layer, [arr], True, color, width, cv2.LINE_AA)
        elif kind == "arrow":
            cv2.polylines(layer, [arr], False, color, width, cv2.LINE_AA)
            if len(pts) >= 2:
                ipts = arr.reshape(-1, 2)
                cv2.arrowedLine(layer, tuple(int(v) for v in ipts[-2]), tuple(int(v) for v in ipts[-1]),
                                 color, width, cv2.LINE_AA, tipLength=2.0)
        elif kind in ("label", "chip"):
            x, y = int(pts[0][0]), int(pts[0][1])
            text = p.get("text") or ""
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(layer, (x - 4, y - th - 6), (x + tw + 4, y + 5), (20, 20, 20), -1)
            cv2.rectangle(layer, (x - 4, y - th - 6), (x + tw + 4, y + 5), color, 1, cv2.LINE_AA)
            cv2.putText(layer, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        if alpha >= 0.99:
            canvas = layer
        elif alpha > 0.01:
            canvas = cv2.addWeighted(layer, alpha, canvas, 1.0 - alpha, 0)
    return canvas


# ------------------------------------------------------------------ self-check


def _fake_detection(**kwargs):
    from bto.core import Detection

    return Detection(**kwargs)


def _build_fake_detections():
    """One Detection per PRIM-producing type, in plausible pitch-meter
    coordinates, active at t=10.0."""
    t0, t1, t = 9.0, 11.0, 10.0
    dets = [
        _fake_detection(
            type="formation", players=["T1", "T2", "T3", "T4"],
            geometry={"label": "4-3-3", "hull": [(20, 10), (30, 5), (45, 20), (35, 40), (18, 30)],
                      "lines": [[(20, 10), (18, 30)], [(30, 5), (35, 40)]]},
            confidence=0.9, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="formation", players=["T11", "T12", "T13", "T14"],
            geometry={"label": "4-4-2", "hull": [(60, 8), (85, 12), (90, 35), (70, 45), (58, 28)],
                      "lines": [[(60, 8), (58, 28)], [(85, 12), (90, 35)]]},
            confidence=0.85, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="offside_line", players=[],
            geometry={"x": 42.0, "team_defending": "home", "approximate": True, "samples": [(t0, 40.0), (t, 42.0)]},
            confidence=0.5, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="back_pass", players=["T5", "T6"],
            geometry={"from": (50.0, 34.0), "to": (35.0, 30.0)},
            confidence=0.7, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="triangle", players=["T7", "T8", "T9"],
            geometry={"vertices": [(40.0, 20.0), (55.0, 25.0), (48.0, 40.0)]},
            confidence=0.6, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="1v1", players=["T10", "T20"],
            geometry={"pairs": [(70.0, 30.0, 72.0, 31.0)], "region": (69.0, 29.0, 73.0, 32.0)},
            confidence=0.55, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="2v2", players=["T21", "T22", "T23", "T24"],
            geometry={"pairs": [(15.0, 50.0, 16.0, 51.0), (20.0, 55.0, 19.0, 54.0)],
                      "region": (15.0, 50.0, 20.0, 55.0)},
            confidence=0.5, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="isolation", players=["T25", "T26"],
            geometry={"attacker": (80.0, 50.0), "defender": (81.0, 51.0)},
            confidence=0.65, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="press", players=["T27", "T28", "T29"],
            geometry={"carrier": (52.0, 34.0), "pressers": [], "intensity": 3.5, "level": "high"},
            confidence=0.75, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="overlap", players=["T30", "T31"],
            geometry={"path": [(10.0, 10.0), (20.0, 12.0), (30.0, 8.0)], "carrier": (30.0, 8.0)},
            confidence=0.45, t_start=t0, t_end=t1,
        ),
        _fake_detection(
            type="underlap", players=["T32", "T33"],
            geometry={"path": [(60.0, 60.0), (65.0, 55.0), (70.0, 58.0)], "carrier": (70.0, 58.0)},
            confidence=0.4, t_start=t0, t_end=t1,
        ),
    ]
    return dets, t


def _self_check():
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    calib_path = os.path.join(root, "out", "bundesliga_smoke", "calib.jsonl")
    with open(calib_path) as f:
        row = json.loads(f.readline())
    H = row["H"]  # pixel->pitch homography, same convention as calib.jsonl / overlay.py

    video = os.path.join(root, "data", "clips", "bundesliga_smoke.mp4")
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read a frame from bundesliga_smoke.mp4"
    frame = cv2.resize(frame, (1280, 720))
    h, w = frame.shape[:2]

    dets, t = _build_fake_detections()

    # Each type in isolation (MAX_EVENTS=3 caps a combined call -- see below
    # -- so "every type yields >=1 prim" is checked one detection at a time).
    by_type: dict[str, list] = {}
    for d in dets:
        by_type.setdefault(d.type, []).append(d)
    for dtype, dlist in by_type.items():
        p1 = detections_to_prims(dlist, H, t, w, h)
        assert p1, f"no prim produced for type={dtype}"
        for p in p1:
            assert p["pts"], f"empty pts in prim {p['gid']} (type={dtype})"
            for x, y in p["pts"]:
                assert np.isfinite(x) and np.isfinite(y), f"non-finite pt in prim {p['gid']}"
            assert p["kind"] in ("polygon", "polyline", "circle", "arrow", "label", "chip")
            assert len(p["color"]) == 3

    # combined call: formation(x2) + offside always-on, events capped at 3 --
    # this is also what's rendered into the debug jpg below.
    prims = detections_to_prims(dets, H, t, w, h)
    seen_types = {p["gid"].split(":")[0] for p in prims}
    n_events = sum(1 for p in prims if p["gid"].split(":")[0] in EVENT_TYPES | {"1v1", "2v2"})
    assert {"formation", "offside_line"} <= seen_types
    assert n_events <= MAX_EVENTS * 3  # each event type emits multiple prims (shape+chip [+vertices])

    # bundle-call convenience form
    bundle_prims = detections_to_prims({"detections": dets, "H": H, "t": t, "w": w, "h": h})
    assert len(bundle_prims) == len(prims), "bundle-call form should match positional-call form"

    # empty/missing-context calls must not raise, return []
    assert detections_to_prims([]) == []
    assert detections_to_prims({"detections": dets}) == []

    out_dir = os.path.join(root, "out", "bundesliga_smoke")
    os.makedirs(out_dir, exist_ok=True)
    debug = prims_render_debug(frame, prims)
    out_path = os.path.join(out_dir, "primitives_selfcheck.jpg")
    cv2.imwrite(out_path, debug)

    print(f"SELF-CHECK OK: {len(prims)} prims across {len(seen_types)} types -> {out_path}")
    print("types:", sorted(seen_types))


if __name__ == "__main__":
    _self_check()
