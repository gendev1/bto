"""2D pitch renderer for M1 tactical overlays (SPEC S3/S6/S8-M1).

Two entry points:
  draw_pitch(ax)   -- static pitch markings, 105x68m, bottom-left origin
                      (matches bto.core's coordinate convention).
  render_clip(...) -- animates PlayerPos dots + ball (+ trail) and, per
                      frame, whatever Detections are active, then saves a
                      GIF via matplotlib's PillowWriter (no ffmpeg on this
                      machine).

Detection geometry contract this module consumes (owned by each detector
module; documented here since this is the consumer -- confirmed against
bto/patterns/{matchups,runs,pressing,formation,offside,passing}.py):
  formation:    hull=[(x,y),...] convex hull points, drawn as a translucent
                polygon; lines=[[(x,y),(x,y),...], ...] one polyline per
                team "line" (defense/mid/attack).
  block:        formation.py's detect_block reports width/depth/line_height/
                line_x/block(level) rather than absolute corners, so there's
                no rect to place on the pitch -- per SPEC ("ignore or thin
                rect") this module ignores it; rect=(min_x,min_y,max_x,max_y)
                is still drawn as a thin dashed rectangle if some other
                producer supplies it.
  1v1 / NvN:    either matchups.py's convention --
                pairs=[(xa,ya,xd,yd),...], region=(minx,miny,maxx,maxy) --
                or a plain pair -- attacker=(x,y), defender=(x,y) (used by
                detect_isolations's type="isolation"). Any type string
                matching r"^\\d+v\\d+$" (e.g. "2v2", "3v3") is treated as a
                matchup and dispatched to the pairs/region drawer.
  triangle:     vertices=[(x,y),(x,y),(x,y)] (passing.py's key; "triangle"
                also accepted), drawn filled+translucent.
  back_pass:    "from"=(x,y), "to"=(x,y) -- an arrow from -> to, shown for
                ~1.5s starting at t_start (independent of the Detection's
                own t_end, since a pass is a near-instantaneous event).
  overlap /
  underlap:     runs.py's convention -- path=[(x,y),...] sampled run path,
                carrier=(x,y). Drawn as a line along the path with an arrow
                head at the end.
  press:        pressing.py's convention -- carrier=(x,y), pressers=[(x,y),
                ...], level='low'|'medium'|'high'. Drawn as a ring around
                the carrier, colored by level.
  offside_line: x=float (pitch-meter line position; "line_x" also accepted)
                -- dashed vertical line + an explicit "approx offside" label
                (SPEC: never present this as an actual call).
"""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle

from bto.core import AWAY, Detection, Frame, HOME, PITCH_LENGTH, PITCH_WIDTH

HOME_COLOR = "#e63946"
AWAY_COLOR = "#457bff"
BALL_COLOR = "black"
LINE_COLOR = "white"
PITCH_COLOR = "#2e7d32"

_NVN_RE = re.compile(r"^\d+v\d+$")
_BACK_PASS_DISPLAY_S = 1.5


def draw_pitch(ax) -> None:
    """Draw standard pitch markings, 105x68m, origin bottom-left."""
    L, W = PITCH_LENGTH, PITCH_WIDTH
    lw = 1.2
    ax.set_facecolor(PITCH_COLOR)

    # outer boundary (touchlines + goal lines)
    ax.plot([0, 0, L, L, 0], [0, W, W, 0, 0], color=LINE_COLOR, lw=lw)
    # halfway line + center circle + center spot
    ax.plot([L / 2, L / 2], [0, W], color=LINE_COLOR, lw=lw)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, fill=False, color=LINE_COLOR, lw=lw))
    ax.plot([L / 2], [W / 2], marker="o", color=LINE_COLOR, ms=2)

    box_w, box_d = 40.32, 16.5
    six_w, six_d = 18.32, 5.5
    spot_d = 11.0
    goal_w, goal_d = 7.32, 2.0

    for x0, sign in ((0.0, 1), (L, -1)):
        bx = x0 + sign * box_d
        ax.plot(
            [x0, bx, bx, x0],
            [W / 2 - box_w / 2, W / 2 - box_w / 2, W / 2 + box_w / 2, W / 2 + box_w / 2],
            color=LINE_COLOR, lw=lw,
        )
        sx = x0 + sign * six_d
        ax.plot(
            [x0, sx, sx, x0],
            [W / 2 - six_w / 2, W / 2 - six_w / 2, W / 2 + six_w / 2, W / 2 + six_w / 2],
            color=LINE_COLOR, lw=lw,
        )
        spot_x = x0 + sign * spot_d
        ax.plot([spot_x], [W / 2], marker="o", color=LINE_COLOR, ms=2)
        angle = 0 if sign > 0 else 180
        ax.add_patch(
            Arc((spot_x, W / 2), 2 * 9.15, 2 * 9.15, angle=0,
                theta1=angle - 53, theta2=angle + 53, color=LINE_COLOR, lw=lw)
        )
        gx = x0 - sign * goal_d
        ax.plot(
            [x0, gx, gx, x0],
            [W / 2 - goal_w / 2, W / 2 - goal_w / 2, W / 2 + goal_w / 2, W / 2 + goal_w / 2],
            color=LINE_COLOR, lw=lw,
        )

    ax.set_xlim(-3, L + 3)
    ax.set_ylim(-3, W + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _team_color(team: str | None) -> str:
    return HOME_COLOR if team == HOME else AWAY_COLOR if team == AWAY else "yellow"


def _team_of(track_id: str, frame: Frame) -> str | None:
    for p in frame.players:
        if p.track_id == track_id:
            return p.team
    return None


def _draw_players(ax, frame: Frame) -> None:
    for p in frame.players:
        ax.plot(p.x, p.y, "o", color=_team_color(p.team), ms=9,
                 mec="white", mew=0.6, zorder=5)
        ax.text(p.x + 0.6, p.y + 0.6, p.track_id, fontsize=5, color="white", zorder=6)


def _draw_ball(ax, frames: list[Frame], i: int, trail_len: int) -> None:
    lo = max(0, i - trail_len)
    trail = [f.ball for f in frames[lo:i + 1] if f.ball is not None]
    if len(trail) > 1:
        xs, ys = zip(*trail)
        ax.plot(xs, ys, "-", color=BALL_COLOR, lw=1, alpha=0.4, zorder=4)
    if frames[i].ball is not None:
        bx, by = frames[i].ball
        ax.plot(bx, by, "o", color=BALL_COLOR, ms=4, zorder=6)


# ---- per-type overlay drawers -------------------------------------------

def _draw_formation(ax, d: Detection, frame: Frame) -> None:
    team = _team_of(d.players[0], frame) if d.players else None
    color = _team_color(team)
    hull = d.geometry.get("hull")
    if hull:
        ax.add_patch(Polygon(hull, closed=True, facecolor=color, alpha=0.15,
                              edgecolor=color, lw=0.8, zorder=2))
    for line in d.geometry.get("lines", []):
        if len(line) >= 2:
            xs, ys = zip(*line)
            ax.plot(xs, ys, "-", color=color, lw=1.2, alpha=0.7, zorder=3)


def _draw_block(ax, d: Detection, frame: Frame) -> None:
    rect = d.geometry.get("rect")
    if not rect:
        return
    x0, y0, x1, y1 = rect
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            edgecolor="cyan", lw=0.8, ls="--", alpha=0.6, zorder=2))


def _draw_pair_or_matchup(ax, d: Detection, frame: Frame) -> None:
    g = d.geometry
    if "pairs" in g:
        for xa, ya, xd, yd in g.get("pairs", []):
            ax.plot([xa, xd], [ya, yd], "-", color="orange", lw=1.0, alpha=0.7, zorder=3)
        region = g.get("region")
        if region:
            x0, y0, x1, y1 = region
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="orange", lw=0.8, ls=":", zorder=2))
    elif "attacker" in g and "defender" in g:
        ax_x, ay = g["attacker"]
        dx_, dy = g["defender"]
        ax.plot([ax_x, dx_], [ay, dy], "-", color="orange", lw=1.2, alpha=0.8, zorder=3)
        pad = 1.5
        x0, x1 = min(ax_x, dx_) - pad, max(ax_x, dx_) + pad
        y0, y1 = min(ay, dy) - pad, max(ay, dy) + pad
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor="orange", lw=0.8, ls=":", zorder=2))


def _draw_triangle(ax, d: Detection, frame: Frame) -> None:
    # passing.py's detect_triangles uses "vertices"; "triangle" also
    # accepted for any other producer.
    tri = d.geometry.get("vertices") or d.geometry.get("triangle")
    if tri and len(tri) == 3:
        ax.add_patch(Polygon(tri, closed=True, facecolor="magenta", alpha=0.2,
                              edgecolor="magenta", lw=1.0, zorder=2))


def _draw_back_pass(ax, d: Detection, frame: Frame) -> None:
    g = d.geometry
    fr, to = g.get("from"), g.get("to")
    if fr is None or to is None:
        return
    ax.add_patch(FancyArrowPatch(fr, to, arrowstyle="-|>", mutation_scale=12,
                                  color="red", lw=1.5, alpha=0.85, zorder=6))


def _draw_run(ax, d: Detection, frame: Frame) -> None:
    path = d.geometry.get("path")
    if not path or len(path) < 2:
        return
    xs, ys = zip(*path)
    color = "lime" if d.type == "overlap" else "deepskyblue"
    ax.plot(xs, ys, "-", color=color, lw=1.2, alpha=0.7, zorder=3)
    ax.add_patch(FancyArrowPatch(path[-2], path[-1], arrowstyle="-|>",
                                  mutation_scale=10, color=color, lw=1.2, alpha=0.9, zorder=6))


def _draw_press(ax, d: Detection, frame: Frame) -> None:
    g = d.geometry
    carrier = g.get("carrier")
    if carrier is None:
        return
    level = g.get("level", "low")
    color = {"low": "yellow", "medium": "orange", "high": "red"}.get(level, "yellow")
    ax.add_patch(Circle(carrier, 2.5, fill=False, edgecolor=color, lw=2.0, alpha=0.85, zorder=6))


def _draw_offside_line(ax, d: Detection, frame: Frame) -> None:
    g = d.geometry
    x = g.get("x", g.get("line_x"))
    if x is None:
        return
    ax.plot([x, x], [0, PITCH_WIDTH], color="yellow", lw=1.2, ls="--", alpha=0.8, zorder=3)
    ax.text(x + 0.5, PITCH_WIDTH - 3, "approx offside", fontsize=6, color="yellow", zorder=6)


_DRAW_BY_TYPE = {
    "formation": _draw_formation,
    "block": _draw_block,
    "isolation": _draw_pair_or_matchup,
    "triangle": _draw_triangle,
    "back_pass": _draw_back_pass,
    "overlap": _draw_run,
    "underlap": _draw_run,
    "press": _draw_press,
    "offside_line": _draw_offside_line,
}


def _resolve_drawer(type_: str):
    if type_ in _DRAW_BY_TYPE:
        return _DRAW_BY_TYPE[type_]
    if _NVN_RE.match(type_):
        return _draw_pair_or_matchup
    return None


def _is_active(d: Detection, t: float) -> bool:
    if d.type == "back_pass":
        return d.t_start <= t <= d.t_start + _BACK_PASS_DISPLAY_S
    return d.t_start <= t <= d.t_end


def render_clip(
    frames: list[Frame],
    detections: list[Detection],
    out_path: str,
    fps: float = 12.5,
    dpi: int = 90,
    figsize: tuple[float, float] = (9.0, 6.0),
    trail_len: int = 10,
) -> str:
    """Animate frames + active-Detection overlays and save as a GIF.

    Detections active at frame.t (t_start <= t <= t_end, except back_pass
    which is shown for ~1.5s after t_start regardless of its own t_end) are
    drawn via the type->draw dispatch above; unknown types are skipped.
    """
    if not frames:
        raise ValueError("render_clip needs at least one frame")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def draw(i):
        ax.clear()
        draw_pitch(ax)
        frame = frames[i]
        for d in detections:
            if _is_active(d, frame.t):
                drawer = _resolve_drawer(d.type)
                if drawer is not None:
                    drawer(ax, d, frame)
        _draw_ball(ax, frames, i, trail_len)
        _draw_players(ax, frame)
        ax.set_title(f"t={frame.t:6.2f}s", fontsize=8)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000.0 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import os
    import tempfile

    from bto.core import PlayerPos

    # 50 synthetic frames at 12.5Hz (4s clip); two players + a moving ball,
    # plus one fake Detection of each geometry kind so every draw function
    # in _DRAW_BY_TYPE (and the NvN regex path) gets exercised at least once.
    dt = 0.08
    n = 50
    frames = []
    for k in range(n):
        t = k * dt
        players = [
            PlayerPos("H1", HOME, 40.0 + 0.2 * k, 30.0),
            PlayerPos("H2", HOME, 45.0, 40.0),
            PlayerPos("A1", AWAY, 55.0, 30.0),
            PlayerPos("A2", AWAY, 60.0, 20.0),
        ]
        ball = (40.0 + 0.2 * k, 30.0)
        frames.append(Frame(t=t, players=players, ball=ball, attacking={HOME: 1, AWAY: -1}))

    detections = [
        Detection("formation", ["H1", "H2"], {
            "hull": [(30, 10), (30, 50), (50, 50), (50, 10)],
            "lines": [[(30, 10), (50, 10)], [(30, 50), (50, 50)]],
        }, 0.8, 0.0, 4.0),
        Detection("block", ["A1", "A2"], {"rect": (55, 10, 70, 50)}, 0.7, 0.0, 4.0),
        Detection("isolation", ["H1", "A1"], {
            "attacker": (40.0, 30.0), "defender": (43.0, 31.0),
        }, 0.9, 0.0, 4.0),
        Detection("2v2", ["H1", "H2", "A1", "A2"], {
            "pairs": [(40.0, 30.0, 55.0, 30.0), (45.0, 40.0, 60.0, 20.0)],
            "region": (40.0, 20.0, 60.0, 40.0),
        }, 0.6, 0.0, 4.0),
        Detection("triangle", ["H1", "H2", "A1"], {
            "triangle": [(40, 30), (45, 40), (55, 30)],
        }, 0.5, 0.0, 4.0),
        Detection("back_pass", ["H1", "H2"], {
            "from": (40.0, 30.0), "to": (45.0, 40.0),
        }, 0.9, 0.5, 0.5),
        Detection("overlap", ["H1", "H2"], {
            "path": [(30, 5), (35, 6), (40, 8)], "carrier": (40.0, 30.0),
        }, 0.7, 0.0, 4.0),
        Detection("underlap", ["H1", "H2"], {
            "path": [(30, 25), (35, 27), (40, 29)], "carrier": (40.0, 30.0),
        }, 0.7, 0.0, 4.0),
        Detection("press", ["A1", "H1", "H2"], {
            "carrier": (55.0, 30.0), "pressers": [(40.0, 30.0), (45.0, 40.0)],
            "intensity": 6.0, "level": "high",
        }, 0.8, 0.0, 4.0),
        Detection("offside_line", ["A2"], {"x": 58.0}, 0.4, 0.0, 4.0),
    ]

    # sanity: every non-NvN type used above has a drawer, and "2v2" resolves
    # via the regex fallback.
    for d in detections:
        assert _resolve_drawer(d.type) is not None, d.type
    assert _resolve_drawer("3v3") is _draw_pair_or_matchup
    assert _resolve_drawer("nonsense") is None

    # back_pass should only be active near its t_start, not for the whole
    # detection window's nominal t_end (which we set equal to t_start above).
    assert _is_active(detections[5], 0.5)
    assert _is_active(detections[5], 0.5 + _BACK_PASS_DISPLAY_S - 0.01)
    assert not _is_active(detections[5], 0.5 + _BACK_PASS_DISPLAY_S + 0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "self_check.gif")
        render_clip(frames, detections, out_path, fps=12.5, dpi=70)
        assert os.path.exists(out_path), "render_clip did not write a file"
        size = os.path.getsize(out_path)
        assert size > 5000, f"GIF looks too small to be real ({size} bytes)"
        print(f"OK: wrote {out_path} ({size} bytes)")
