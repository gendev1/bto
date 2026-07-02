"""M1 demo (SPEC S8-M1): run every available pattern-engine detector on a
Metrica clip and render an animated 2D pitch view.

Loads Sample Game 1 tracking data, takes a ~45s window starting a few
minutes in, downsamples to 12.5 Hz, runs whichever SPEC S6 detectors import
cleanly (skipping + printing a warning for any sibling module that isn't
implemented yet), and renders the merged Detection list to out/demo_m1.gif.

Run: uv run python scripts/demo_m1.py
"""

import os
import time
from collections import Counter

from bto.core import Detection
from bto.io.metrica import load_metrica
from bto.render.pitch import render_clip

HOME_CSV = "data/metrica/Sample_Game_1_RawTrackingData_Home_Team.csv"
AWAY_CSV = "data/metrica/Sample_Game_1_RawTrackingData_Away_Team.csv"
OUT_PATH = "out/demo_m1.gif"

WINDOW_START_S = 3 * 60.0  # a few minutes in
WINDOW_LEN_S = 45.0

# (module, function) candidates for every SPEC S6 detector. Modules that
# don't exist yet (sibling agents build them in parallel) are skipped with a
# printed warning rather than failing the demo.
DETECTOR_CANDIDATES = [
    ("bto.patterns.matchups", "detect_matchups"),
    ("bto.patterns.matchups", "detect_isolations"),
    ("bto.patterns.runs", "detect_runs"),
    ("bto.patterns.pressing", "detect_pressing"),
    ("bto.patterns.formation", "detect_formation"),
    ("bto.patterns.formation", "detect_block"),
    ("bto.patterns.passing", "detect_triangles"),
    ("bto.patterns.passing", "detect_back_passes"),
    ("bto.patterns.offside", "offside_line"),
]


def _load_detectors():
    fns = []
    for mod_name, fn_name in DETECTOR_CANDIDATES:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
        except (ImportError, AttributeError) as e:
            print(f"[demo_m1] skipping {mod_name}.{fn_name}: {e}")
            continue
        fns.append((f"{mod_name}.{fn_name}", fn))
    return fns


def main() -> None:
    print("[demo_m1] loading Metrica Sample Game 1 tracking data...")
    frames = load_metrica(HOME_CSV, AWAY_CSV, downsample=2)  # 25Hz -> 12.5Hz
    start_i = next((i for i, f in enumerate(frames) if f.t >= WINDOW_START_S), 0)
    end_t = frames[start_i].t + WINDOW_LEN_S
    end_i = next(
        (i for i, f in enumerate(frames[start_i:], start_i) if f.t > end_t), len(frames)
    )
    clip = frames[start_i:end_i]
    print(f"[demo_m1] clip: {len(clip)} frames, t={clip[0].t:.1f}s..{clip[-1].t:.1f}s")

    detections: list[Detection] = []
    for name, fn in _load_detectors():
        t0 = time.time()
        try:
            dets = fn(clip)
        except Exception as e:  # a detector bug shouldn't kill the demo
            print(f"[demo_m1] {name} raised {e!r}, skipping its output")
            continue
        print(f"[demo_m1] {name}: {len(dets)} detections in {time.time() - t0:.2f}s")
        detections.extend(dets)

    detections.sort(key=lambda d: d.t_start)
    by_type = Counter(d.type for d in detections)
    print("[demo_m1] detections by type:")
    for dtype, n in sorted(by_type.items()):
        print(f"[demo_m1]   {dtype}: {n}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"[demo_m1] rendering {len(clip)} frames / {len(detections)} detections -> {OUT_PATH}")
    render_clip(clip, detections, OUT_PATH, fps=12.5, dpi=90)
    print("[demo_m1] done:", OUT_PATH)


if __name__ == "__main__":
    main()
