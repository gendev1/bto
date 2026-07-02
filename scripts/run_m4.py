"""M4 CLI: composite tactical overlays onto the original clip (SPEC M4/C7).

    uv run python scripts/run_m4.py <clip_stem> [--layers formation,offside,events]

Reads out/<stem>/{perception.jsonl,calib.jsonl,shots.jsonl,m3_detections.json}
and writes out/<stem>/m4_overlay.mp4 at processed-frame fps, drawing on the
ORIGINAL video frames (frame_idx-exact, same mapping as annotate_m2).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bto.render.overlay import DEFAULT_LAYERS, render_overlay_video  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_video(stem):
    cands = [
        os.path.join(ROOT, "data", "clips", f"{stem}.mp4"),
        os.path.join(ROOT, "out", stem, f"{stem}.mp4"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no video for stem {stem!r}; tried {cands}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="clip stem, e.g. bundesliga_smoke")
    ap.add_argument("--layers", default=",".join(DEFAULT_LAYERS),
                    help="csv of layers: formation,offside,events")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "out", args.stem)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"missing out dir: {out_dir}")
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]

    stats = render_overlay_video(find_video(args.stem), out_dir, layers=layers)

    print(f"out_video={stats['out_video']}")
    print(f"n_frames={stats['n_frames']}  drawable={stats['n_drawable']}  "
          f"out_fps={stats['out_fps']:.2f}  stride={stats['stride']}")
    print("per-layer draw counts:")
    for k in sorted(stats["draw_counts"]):
        print(f"  {k}: {stats['draw_counts'][k]}")


if __name__ == "__main__":
    main()
