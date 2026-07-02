"""M2 entrypoint: full perception pipeline on one broadcast video.

Runs, sequentially (8GB unified memory -- one YOLO model at a time):
    1. shot classify  (bto.vision.shots,   in-process: pixel heuristic)
    2. detect+track   (bto.vision.detect,  subprocess: player model)
                       -- fed shots.jsonl from stage 1 (M3 track-hygiene:
                       skips 'other' frames, resets ByteTrack at each main
                       segment start, offsets tids to stay unique)
    3. team assign    (bto.vision.teams,   in-process: histograms + kmeans)
    4. ball track     (bto.vision.ball,    subprocess: ball model)
    5. annotate       (scripts.annotate_m2, in-process: merge + render)

shots runs first and is cheap/pixel-only (no model), so moving it ahead of
detect costs nothing and lets detect skip GPU work on 'other' frames and
avoid falsely linking tracks across shot cuts.

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


def compute_segment_churn(out_dir: str, stride: int, fps: float) -> dict:
    """Segment-aware new-tids/min: SPEC C3's honest churn metric.

    Reads detections.jsonl + shots.jsonl from out_dir. Builds contiguous
    'main' segments (consecutive main frame_idx separated by exactly
    `stride`, matching how bto.vision.detect resets ByteTrack). Counts a
    "new tid" event only for tids appearing for the first time within a
    segment, and only after that segment's first frame (excluded, since
    every tid there is trivially "new" right after a reset/shot cut).
    """
    import json as _json

    det_path = os.path.join(out_dir, "detections.jsonl")
    shots_path = os.path.join(out_dir, "shots.jsonl")
    if not (os.path.exists(det_path) and os.path.exists(shots_path)):
        return {}

    with open(shots_path) as f:
        shot_of = {}
        for line in f:
            rec = _json.loads(line)
            shot_of[rec["frame_idx"]] = rec["shot"]

    with open(det_path) as f:
        det_lines = [_json.loads(line) for line in f]
    det_lines.sort(key=lambda r: r["frame_idx"])

    segments: list[list[dict]] = []
    prev_idx = None
    for rec in det_lines:
        if shot_of.get(rec["frame_idx"]) != "main":
            prev_idx = None
            continue
        if prev_idx is None or rec["frame_idx"] - prev_idx != stride:
            segments.append([])
        segments[-1].append(rec)
        prev_idx = rec["frame_idx"]

    n_players_seen = set()
    new_tid_events = 0
    counted_seconds = 0.0
    for seg in segments:
        seen: set[int] = set()
        for i, rec in enumerate(seg):
            tids_here = {b["tid"] for b in rec["boxes"] if b.get("tid") is not None}
            n_players_seen |= tids_here
            if i == 0:
                seen |= tids_here
                continue
            new_here = tids_here - seen
            new_tid_events += len(new_here)
            seen |= tids_here
            counted_seconds += stride / fps

    churn_per_min = new_tid_events / (counted_seconds / 60.0) if counted_seconds > 0 else float("nan")
    return {
        "n_segments": len(segments),
        "n_unique_tids": len(n_players_seen),
        "new_tid_events": new_tid_events,
        "counted_seconds": counted_seconds,
        "new_tids_per_min": churn_per_min,
    }


def run_pipeline(video: str, stride: int = 2, max_frames: int | None = None,
                 out_dir: str | None = None) -> dict:
    stem = os.path.splitext(os.path.basename(video))[0]
    out_dir = out_dir or os.path.join("out", stem)
    os.makedirs(out_dir, exist_ok=True)

    timings = {}
    mf = ["--max-frames", str(max_frames)] if max_frames is not None else []

    # 1. shot classifier (pixel-only, cheap, no model -- runs first so detect
    #    can consume shots.jsonl for track hygiene)
    from bto.vision.shots import classify_shots
    t0 = time.time()
    records = classify_shots(video, out_path=out_dir, stride=stride, max_frames=max_frames)
    timings["shots_s"] = time.time() - t0
    n_main = sum(1 for r in records if r["shot"] == "main")
    print(f"[shots] {len(records)} frames: main={n_main} other={len(records) - n_main} "
          f"({timings['shots_s']:.1f}s)")

    # 2. detect + track (player YOLO model, own subprocess), fed shots.jsonl
    shots_path = os.path.join(out_dir, "shots.jsonl")
    t0 = time.time()
    _run_subprocess(["-m", "bto.vision.detect", video, "--stride", str(stride),
                     "--out", out_dir, "--shots", shots_path] + mf, "detect")
    timings["detect_s"] = time.time() - t0

    # 3. team assignment (histograms + kmeans, no model)
    from bto.vision.teams import assign_teams
    t0 = time.time()
    teams = assign_teams(video, os.path.join(out_dir, "detections.jsonl"),
                         out_path=os.path.join(out_dir, "teams.json"),
                         frames_out_path=os.path.join(out_dir, "teams_frames.jsonl"))
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
    ap = argparse.ArgumentParser(description="M2 perception pipeline (shots -> detect -> teams -> ball -> annotate)")
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="output dir (default: out/<clip_stem>)")
    ap.add_argument("--churn", action="store_true",
                     help="print segment-aware new-tids/min report after the pipeline finishes")
    args = ap.parse_args()
    out_dir = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.video))[0])
    run_pipeline(args.video, stride=args.stride, max_frames=args.max_frames, out_dir=args.out)

    if args.churn:
        import cv2
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        report = compute_segment_churn(out_dir, args.stride, fps)
        if report:
            print(
                f"[churn] {report['n_segments']} main segments, "
                f"{report['n_unique_tids']} unique tids, "
                f"{report['new_tid_events']} new-tid events over "
                f"{report['counted_seconds']:.1f}s counted "
                f"(excl. segment-first frames) "
                f"= {report['new_tids_per_min']:.2f} new-tids/min "
                f"(SPEC C3 target: <1 switch/player/min)"
            )
        else:
            print("[churn] no detections.jsonl/shots.jsonl found to report on")


if __name__ == "__main__":
    main()
