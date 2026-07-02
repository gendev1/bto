"""SPEC S6 pattern detectors. Every detector: list[Frame] -> list[Detection].

M6 relational rework: the coordinate stream now also feeds a relational
layer -- formation ROLES per team (bto.patterns.roles.assign_roles) and
marking/press EDGES with lifecycles (bto.patterns.edges.track_edges) -- that
detect_matchups, detect_pressing, and detect_plays consume instead of
re-deriving per-frame proximity from scratch. run_all computes roles and
edges ONCE per call and injects them into every detector that accepts them,
so a live host running run_all every tick doesn't pay for possession() /
track_edges() / assign_roles() N times over.

ALL_DETECTORS stays the plain frames-only list (detectors that take no extra
kwargs) for callers that want a flat, uniform detector list; run_all's own
default pipeline (detectors=None) is the full M6 set described above and is
NOT just "run every entry in ALL_DETECTORS" -- it also runs detect_plays and
emits the marking/press layer itself as Detections via edges_to_detections,
neither of which fits the frames-only shape.
"""

from bto.core import AWAY, Detection, Frame, HOME
from bto.patterns.edges import edges_to_detections, track_edges
from bto.patterns.formation import detect_block, detect_formation
from bto.patterns.matchups import detect_isolations, detect_matchups
from bto.patterns.offside import offside_line
from bto.patterns.passing import detect_back_passes, detect_triangles
from bto.patterns.plays import detect_plays
from bto.patterns.pressing import detect_pressing
from bto.patterns.roles import assign_roles
from bto.patterns.runs import detect_runs

ALL_DETECTORS = [
    detect_formation,
    detect_block,
    detect_matchups,
    detect_isolations,
    detect_triangles,
    detect_back_passes,
    detect_runs,
    detect_pressing,
    offside_line,
]


def run_all(frames: list[Frame], detectors=None) -> list[Detection]:
    """Run every detector (or the given ones); concat sorted by t_start.

    detectors=None (the default) runs the full M6 pipeline: formation,
    block, offside, the possession-based passing detectors (unchanged),
    matchups + pressing (edge-injected reworks), detect_plays (play
    grammar + roles' rotations), and edges_to_detections (the raw
    marking/press relational layer itself, so the renderer can draw the
    "who's covering whom" links even where no higher-level pattern fired).
    roles and edges are each computed exactly once and shared across every
    detector below that accepts them.

    detectors=<explicit list> runs exactly those frames-only callables
    instead (e.g. ALL_DETECTORS, or a subset) -- no relational injection,
    for callers that want the plain per-frame detector set.
    """
    if detectors is not None:
        out: list[Detection] = []
        for det in detectors:
            out.extend(det(frames))
        out.sort(key=lambda d: d.t_start)
        return out

    edges = track_edges(frames)
    role_maps: dict[str, list[dict]] = {}
    rotations: list[Detection] = []
    for team in (HOME, AWAY):
        maps, _slots, rots = assign_roles(frames, team)
        role_maps[team] = maps
        rotations.extend(rots)

    out: list[Detection] = []
    out.extend(detect_formation(frames))
    out.extend(detect_block(frames))
    out.extend(offside_line(frames))
    out.extend(detect_triangles(frames))
    out.extend(detect_back_passes(frames))
    out.extend(detect_matchups(frames, edges=edges))
    out.extend(detect_isolations(frames))
    out.extend(detect_pressing(frames, edges=edges))
    out.extend(detect_runs(frames))
    out.extend(
        detect_plays(
            frames,
            role_maps_home=role_maps[HOME],
            role_maps_away=role_maps[AWAY],
            edges=edges,
        )
    )
    out.extend(rotations)
    out.extend(edges_to_detections(edges))
    out.sort(key=lambda d: d.t_start)
    return out
