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


def run_detection(
    video_path: str,
    model_path: str = DEFAULT_MODEL,
    out_path: str | None = None,
    stride: int = 2,
    imgsz: int = 1280,
    device: str = "auto",
    conf: float = 0.3,
    max_frames: int | None = None,
) -> str:
    """Run detection+tracking over `video_path`, writing detections.jsonl.

    frame_idx is the index into the ORIGINAL video (stride-aware: jumps by `stride`).
    t = frame_idx / fps.

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

    frame_idx = 0
    n_processed = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                result = model.track(
                    frame,
                    persist=True,
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
                        tid = int(tids[i]) if tids is not None else None
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
    video = "data/clips/bundesliga_smoke.mp4"
    out_path = "out/bundesliga_smoke/detections.jsonl"
    t0 = time.time()
    run_detection(video, out_path=out_path, stride=2, max_frames=40, device="auto")
    elapsed = time.time() - t0

    with open(out_path) as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 40, f"expected 40 frames, got {len(lines)}"

    n_player_boxes = [
        sum(1 for b in rec["boxes"] if b["cls"] in ("player", "goalkeeper")) for rec in lines
    ]
    n_player_boxes.sort()
    median = n_player_boxes[len(n_player_boxes) // 2]
    print(f"[self-check] median player+gk boxes/frame = {median}")
    assert median >= 15, f"median player boxes/frame {median} < 15"

    tid_counts: dict[int, int] = {}
    for rec in lines:
        for b in rec["boxes"]:
            if b["tid"] is not None:
                tid_counts[b["tid"]] = tid_counts.get(b["tid"], 0) + 1
    persistent = [tid for tid, cnt in tid_counts.items() if cnt >= 10]
    print(f"[self-check] tids seen >=10 times: {len(persistent)} of {len(tid_counts)} total tids")
    assert len(persistent) >= 1, "no track id persisted across >= 10 frames"

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
    )


if __name__ == "__main__":
    main()
