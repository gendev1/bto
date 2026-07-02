"""M2 entrypoint: full perception pipeline on one broadcast video.

Runs, sequentially (8GB unified memory -- one YOLO model at a time):
    1. detect+track   (bto.vision.detect,  subprocess: player model)
    2. shot classify  (bto.vision.shots,   in-process: pixel heuristic)
    3. team assign    (bto.vision.teams,   in-process: histograms + kmeans)
    4. ball track     (bto.vision.ball,    subprocess: ball model)
    5. annotate       (scripts.annotate_m2, in-process: merge + render)

All stages use the SAME stride so every interchange stream covers the same
frame_idx set (annotate merges by frame_idx union; aligned strides avoid
flicker rows that have players but no ball or vice versa).

Usage:
    uv run python scripts/run_perception.py <video> [--stride N] [--max-frames N] [--out DIR]

Outputs under out/<clip_stem>/ (or --out DIR):
    detections.jsonl, shots.jsonl, teams.json, ball.jsonl,
    perception.jsonl, m2_annotated.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run_subprocess(args: list[str], stage: str) -> None:
    cmd = [sys.executable] + args
    print(f"[{stage}] $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"stage '{stage}' failed with exit code {res.returncode}")


def run_pipeline(video: str, stride: int = 2, max_frames: int | None = None,
                 out_dir: str | None = None) -> dict:
    stem = os.path.splitext(os.path.basename(video))[0]
    out_dir = out_dir or os.path.join("out", stem)
    os.makedirs(out_dir, exist_ok=True)

    timings = {}
    mf = ["--max-frames", str(max_frames)] if max_frames is not None else []

    # 1. detect + track (player YOLO model, own subprocess)
    t0 = time.time()
    _run_subprocess(["-m", "bto.vision.detect", video, "--stride", str(stride),
                     "--out", out_dir] + mf, "detect")
    timings["detect_s"] = time.time() - t0

    # 2. shot classifier (pixel-only)
    from bto.vision.shots import classify_shots
    t0 = time.time()
    records = classify_shots(video, out_path=out_dir, stride=stride, max_frames=max_frames)
    timings["shots_s"] = time.time() - t0
    n_main = sum(1 for r in records if r["shot"] == "main")
    print(f"[shots] {len(records)} frames: main={n_main} other={len(records) - n_main} "
          f"({timings['shots_s']:.1f}s)")

    # 3. team assignment (histograms + kmeans, no model)
    from bto.vision.teams import assign_teams
    t0 = time.time()
    teams = assign_teams(video, os.path.join(out_dir, "detections.jsonl"),
                         out_path=os.path.join(out_dir, "teams.json"))
    timings["teams_s"] = time.time() - t0
    print(f"[teams] {len(teams)} tracks assigned ({timings['teams_s']:.1f}s)")

    # 4. ball tracking (ball YOLO model, own subprocess)
    t0 = time.time()
    _run_subprocess(["-m", "bto.vision.ball", video, "--stride", str(stride),
                     "--out", out_dir] + mf, "ball")
    timings["ball_s"] = time.time() - t0

    # 5. annotate (merge + render)
    from annotate_m2 import run as annotate_run, print_stats
    t0 = time.time()
    stats = annotate_run(video, out_dir)
    timings["annotate_s"] = time.time() - t0
    print_stats(stats)

    stats["timings"] = timings
    print("[timings] " + "  ".join(f"{k}={v:.1f}s" for k, v in timings.items()))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 perception pipeline (detect -> shots -> teams -> ball -> annotate)")
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="output dir (default: out/<clip_stem>)")
    args = ap.parse_args()
    run_pipeline(args.video, stride=args.stride, max_frames=args.max_frames, out_dir=args.out)


if __name__ == "__main__":
    main()
