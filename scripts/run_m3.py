"""M3 runner: out/<stem>/{perception,calib}.jsonl -> segments -> pattern
engine -> out/<stem>/m3_detections.json + m3_pitch.gif (longest segment).

Usage: uv run python scripts/run_m3.py <clip_stem>
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from bto.patterns import run_all
from bto.render.pitch import render_clip
from bto.vision.bridge import build_frames, frames_stats

GIF_MAX_S = 45.0


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: uv run python scripts/run_m3.py <clip_stem>")
    stem = sys.argv[1]
    d = Path(__file__).resolve().parent.parent / "out" / stem
    perception, calib, ball = d / "perception.jsonl", d / "calib.jsonl", d / "ball.jsonl"
    for p in (perception, calib):
        if not p.exists():
            sys.exit(f"missing {p}")

    segments = build_frames(perception, calib, ball if ball.exists() else None)
    print(f"{stem}: {frames_stats(segments)}")
    if not segments:
        sys.exit("no segments >= 5s; nothing to run")

    records, counts, per_segment = [], {}, []
    for i, seg in enumerate(segments):
        dets = run_all(seg)
        per_segment.append(dets)
        for det in dets:
            rec = asdict(det)
            rec["segment"] = i
            records.append(rec)
            counts[det.type] = counts.get(det.type, 0) + 1

    out_json = d / "m3_detections.json"
    with open(out_json, "w") as f:
        json.dump(records, f, default=float)
    print(f"wrote {out_json} ({len(records)} detections)")
    for typ in sorted(counts):
        print(f"  {typ}: {counts[typ]}")

    # Render the longest segment, capped at ~45s of frames.
    longest = max(range(len(segments)), key=lambda i: segments[i][-1].t - segments[i][0].t)
    frames = segments[longest]
    frames = [f for f in frames if f.t - frames[0].t <= GIF_MAX_S]
    dts = [b.t - a.t for a, b in zip(frames, frames[1:])]
    fps = 1.0 / (sorted(dts)[len(dts) // 2]) if dts else 12.5
    gif = d / "m3_pitch.gif"
    render_clip(frames, per_segment[longest], str(gif), fps=fps)
    print(f"rendered segment {longest} ({frames[-1].t - frames[0].t:.1f}s, "
          f"{len(frames)} frames @ {fps:.1f} fps) -> {gif}")


if __name__ == "__main__":
    main()
