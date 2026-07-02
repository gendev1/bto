"""SPEC S6 pattern detectors. Every detector: list[Frame] -> list[Detection]."""

from bto.core import Detection, Frame
from bto.patterns.formation import detect_block, detect_formation
from bto.patterns.matchups import detect_isolations, detect_matchups
from bto.patterns.offside import offside_line
from bto.patterns.passing import detect_back_passes, detect_triangles
from bto.patterns.pressing import detect_pressing
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
    """Run every detector (or the given ones) with default params; concat sorted by t_start."""
    out: list[Detection] = []
    for det in detectors if detectors is not None else ALL_DETECTORS:
        out.extend(det(frames))
    out.sort(key=lambda d: d.t_start)
    return out
