"""SPEC C2+C3: player/GK/referee detection + ByteTrack tracking on a broadcast clip.

Writes detections.jsonl (frozen interchange format, see project brief):
    {"frame_idx": int, "t": float, "boxes": [
        {"tid": int|null, "cls": "player|goalkeeper|referee", "conf": float, "xyxy": [x1,y1,x2,y2]}
    ]}

Ball (class 0) detections are dropped here — see bto/vision/ball.py (C9) for the ball path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import cv2
import torch

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
DEFAULT_MODEL = "models/football-player-detection.pt"


def _ensure_lap_shim() -> None:
    """ultralytics' ByteTrack hard-imports `lap` (github.com/gatagat/lap) for linear
    assignment. It's not installed and this venv has no pip to auto-fetch it. Rather
    than add a dependency, register a tiny scipy-backed shim under sys.modules['lap']
    exposing just the `lapjv` entry point ByteTrack calls (scipy is already a dep).
    """
    try:
        import lap  # noqa: F401
        return
    except ImportError:
        pass

    from scipy.optimize import linear_sum_assignment
    import numpy as np

    def lapjv(cost_matrix, extend_cost=True, cost_limit=None):
        cost_matrix = np.asarray(cost_matrix)
        n, m = cost_matrix.shape
        x = np.full(n, -1, dtype=int)
        y = np.full(m, -1, dtype=int)
        if n == 0 or m == 0:
            return 0.0, x, y
        limit = np.inf if cost_limit is None else cost_limit
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = 0.0
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= limit:
                x[r] = c
                y[c] = r
                total += cost_matrix[r, c]
        return total, x, y

    shim = types.ModuleType("lap")
    shim.lapjv = lapjv
    shim.__version__ = "0.5.12-scipy-shim"
    sys.modules["lap"] = shim


_ensure_lap_shim()


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_shot_labels(shots_jsonl: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    with open(shots_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            labels[int(rec["frame_idx"])] = rec["shot"]
    return labels


def run_detection(
    video_path: str,
    model_path: str = DEFAULT_MODEL,
    out_path: str | None = None,
    stride: int = 2,
    imgsz: int = 1280,
    device: str = "auto",
    conf: float = 0.3,
    max_frames: int | None = None,
    shots_jsonl: str | None = None,
    start_frame: int = 0,
) -> str:
    """Run detection+tracking over `video_path`, writing detections.jsonl.

    frame_idx is the index into the ORIGINAL video (stride-aware: jumps by `stride`).
    t = frame_idx / fps.

    If `shots_jsonl` is given (path to a shots.jsonl written by bto.vision.shots
    over the SAME frame_idx/stride), frames labeled 'other' are skipped entirely
    (the tracker never sees them; an empty-boxes record is still emitted so
    frame_idx coverage stays explicit). Each run of consecutive 'main' frames
    (a "main segment", i.e. no 'other'/missing frame in between at this stride)
    starts a FRESH ByteTrack state (persist=False on the segment's first frame),
    so tracks are never falsely linked across a shot cut. Raw tracker ids are
    offset per-segment by a running max-seen-tid so tids stay unique across the
    whole video despite the resets.

    When shots_jsonl is None, behavior is unchanged from before (single
    persistent ByteTrack run across the whole clip).

    `start_frame` seeks the video before processing begins (frame_idx in the
    output still refers to the ORIGINAL video's absolute frame index); used by
    the self-check to probe a specific shot-boundary window without paying for
    GPU inference on every main-segment frame from 0 up to the window.

    Returns the output path.
    """
    from ultralytics import YOLO

    dev = _pick_device(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if out_path is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join("out", stem)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "detections.jsonl")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    model = YOLO(model_path)

    shot_labels = _load_shot_labels(shots_jsonl) if shots_jsonl else None

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    n_processed = 0
    t0 = time.time()

    # Track-hygiene state (only used when shot_labels is set):
    tid_offset = 0          # added to every raw tracker id emitted so far
    prev_main_frame_idx = None  # frame_idx of the previous MAIN frame processed
    max_raw_tid_seen = 0    # max raw (pre-offset) tid seen in the CURRENT segment

    with open(out_path, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                if shot_labels is not None:
                    label = shot_labels.get(frame_idx, "other")
                    if label != "main":
                        rec = {"frame_idx": frame_idx, "t": frame_idx / fps, "boxes": []}
                        f.write(json.dumps(rec) + "\n")
                        n_processed += 1
                        if max_frames is not None and n_processed >= max_frames:
                            break
                        frame_idx += 1
                        continue

                    # main frame: is this the start of a new segment?
                    is_segment_start = (
                        prev_main_frame_idx is None
                        or frame_idx - prev_main_frame_idx != stride
                    )
                    if is_segment_start:
                        tid_offset += max_raw_tid_seen
                        max_raw_tid_seen = 0
                    persist = not is_segment_start
                    prev_main_frame_idx = frame_idx
                else:
                    persist = True

                result = model.track(
                    frame,
                    persist=persist,
                    tracker="bytetrack.yaml",
                    imgsz=imgsz,
                    conf=conf,
                    device=dev,
                    verbose=False,
                )[0]

                boxes_out = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    tids = boxes.id
                    tids = tids.cpu().numpy().astype(int) if tids is not None else None

                    for i in range(len(boxes)):
                        c = int(cls[i])
                        if c == 0:  # drop ball; handled by the ball specialist module
                            continue
                        if tids is not None:
                            raw_tid = int(tids[i])
                            if shot_labels is not None:
                                max_raw_tid_seen = max(max_raw_tid_seen, raw_tid)
                                tid = raw_tid + tid_offset
                            else:
                                tid = raw_tid
                        else:
                            tid = None
                        boxes_out.append({
                            "tid": tid,
                            "cls": CLASS_NAMES.get(c, str(c)),
                            "conf": float(confs[i]),
                            "xyxy": [float(v) for v in xyxy[i]],
                        })

                rec = {
                    "frame_idx": frame_idx,
                    "t": frame_idx / fps,
                    "boxes": boxes_out,
                }
                f.write(json.dumps(rec) + "\n")

                n_processed += 1
                if n_processed % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"[detect] processed={n_processed} frame_idx={frame_idx} "
                          f"fps={n_processed / elapsed:.2f}")

                if max_frames is not None and n_processed >= max_frames:
                    break

            frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"[detect] done: {n_processed} frames processed in {elapsed:.1f}s "
          f"({n_processed / max(elapsed, 1e-9):.2f} fps) -> {out_path}")
    return out_path


def _self_check() -> None:
    """Track-hygiene self-check (M3): probe a 40-frame window of the cwc clip
    that spans two known 'other' shot-cut gaps (per the existing shots.jsonl),
    at the same stride=3 shots.jsonl was computed with. Only the frames
    labeled 'main' inside the window actually hit the GPU tracker (26 of 40
    here), well under the 40-GPU-frame budget.
    """
    video = "data/clips/cwc2021_chelsea_palmeiras_20m.mp4"
    if not os.path.exists(video):
        video = "data/clips/cwc2021_chelsea_palmeiras_20m.webm"
    shots_path = "out/cwc2021_chelsea_palmeiras_20m/shots.jsonl"
    out_path = "out/cwc2021_chelsea_palmeiras_20m/_selfcheck_detections.jsonl"

    stride = 3
    start_frame = 1500
    n_window = 40

    t0 = time.time()
    run_detection(
        video, out_path=out_path, stride=stride, max_frames=n_window,
        device="auto", shots_jsonl=shots_path, start_frame=start_frame,
    )
    elapsed = time.time() - t0

    with open(out_path) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == n_window, f"expected {n_window} frames, got {len(lines)}"

    shot_labels = _load_shot_labels(shots_path)

    # (a) 'other' frames in the window emit empty boxes.
    other_lines = [rec for rec in lines if shot_labels.get(rec["frame_idx"]) != "main"]
    assert other_lines, "window has no 'other' frames -- pick a different window"
    for rec in other_lines:
        assert rec["boxes"] == [], f"'other' frame {rec['frame_idx']} has non-empty boxes"
    print(f"[self-check] {len(other_lines)} 'other' frames all emitted empty boxes")

    # (b)/(c) tids: unique overall but persist within a segment, differ across boundaries.
    # Recover segments the same way run_detection did: consecutive main frame_idx
    # separated by exactly `stride` are the same segment.
    main_lines = [rec for rec in lines if shot_labels.get(rec["frame_idx"]) == "main"]
    segments: list[list[dict]] = []
    prev_idx = None
    for rec in main_lines:
        if prev_idx is None or rec["frame_idx"] - prev_idx != stride:
            segments.append([])
        segments[-1].append(rec)
        prev_idx = rec["frame_idx"]
    assert len(segments) >= 2, f"expected >=2 main segments in window, got {len(segments)}"
    print(f"[self-check] window has {len(segments)} main segments "
          f"(sizes {[len(s) for s in segments]})")

    all_tids: dict[int, int] = {}  # tid -> which segment index first saw it
    for seg_i, seg in enumerate(segments):
        seg_tids = set()
        for rec in seg:
            for b in rec["boxes"]:
                if b["tid"] is not None:
                    seg_tids.add(b["tid"])
                    if b["tid"] in all_tids and all_tids[b["tid"]] != seg_i:
                        raise AssertionError(
                            f"tid {b['tid']} reused across segments {all_tids[b['tid']]} and {seg_i}"
                        )
                    all_tids[b["tid"]] = seg_i
        assert seg_tids, f"segment {seg_i} has no tracked ids at all"

    print(f"[self-check] {len(all_tids)} unique tids total across {len(segments)} segments, "
          f"none reused across a shot-cut boundary")

    # within-segment persistence: at least one tid should survive >= half a segment.
    for seg_i, seg in enumerate(segments):
        if len(seg) < 3:
            continue
        tid_counts: dict[int, int] = {}
        for rec in seg:
            for b in rec["boxes"]:
                if b["tid"] is not None:
                    tid_counts[b["tid"]] = tid_counts.get(b["tid"], 0) + 1
        best = max(tid_counts.values()) if tid_counts else 0
        assert best >= len(seg) // 2, (
            f"segment {seg_i}: no tid persisted across >= half its {len(seg)} frames "
            f"(best={best})"
        )
    print("[self-check] within-segment tid persistence OK")

    # segment-aware churn: new-tids per minute, counting only within contiguous
    # segments and excluding each segment's first frame (the honest SPEC C3 metric).
    fps = 30.0
    seen_before: set[int] = set()
    new_tid_events = 0
    counted_seconds = 0.0
    for seg in segments:
        seen_before |= set()  # tids are already globally unique per segment; nothing carried in
        for i, rec in enumerate(seg):
            tids_here = {b["tid"] for b in rec["boxes"] if b["tid"] is not None}
            if i == 0:
                seen_before |= tids_here
                continue
            new_here = tids_here - seen_before
            new_tid_events += len(new_here)
            seen_before |= tids_here
            counted_seconds += stride / fps
    churn_per_min = new_tid_events / (counted_seconds / 60.0) if counted_seconds > 0 else float("nan")
    print(f"[self-check] segment-aware churn: {new_tid_events} new tids over "
          f"{counted_seconds:.1f}s counted (excl. segment-first frames) "
          f"= {churn_per_min:.2f} new-tids/min")

    print(f"[self-check] OK: {len(lines)} frames, {elapsed:.1f}s total, "
          f"{elapsed / len(lines):.3f}s/frame")


def main() -> None:
    ap = argparse.ArgumentParser(description="Player/GK/referee detection + tracking (SPEC C2+C3)")
    ap.add_argument("video")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory (writes detections.jsonl inside it)")
    ap.add_argument("--shots", default=None,
                     help="path to shots.jsonl (same video/stride) for segment-reset track hygiene")
    args = ap.parse_args()

    out_path = None
    if args.out is not None:
        out_path = os.path.join(args.out, "detections.jsonl")

    run_detection(
        args.video,
        model_path=args.model,
        out_path=out_path,
        stride=args.stride,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        max_frames=args.max_frames,
        shots_jsonl=args.shots,
    )


if __name__ == "__main__":
    main()
