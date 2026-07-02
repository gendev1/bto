"""Ball tracking (SPEC C9): dedicated small-object path.

Two detection modes, driven by a hand-rolled constant-velocity Kalman
filter (state = [x, y, vx, vy], pixel units, dt = 1 processed frame):

  CROP mode -- ball position known (KF initialized, not "lost"): crop a
    640x640 window centered on the Kalman prediction, run the ball model
    on the crop at imgsz=640 (ball appears bigger -> cheaper + easier),
    map the detection back to full-frame pixel coords.

  FULL mode -- lost (no accepted detection for > 25 frames) or at start:
    run the ball model on the full frame at imgsz=1280, but only every
    5th processed frame (cheap idle scan) until reacquired.

A detection is accepted if conf > 0.35 and, once the KF is initialized,
its center falls within a gating distance (150px + 3*sigma, sigma from
the predicted position covariance) of the KF prediction. On accept: KF
update, emit src="det". On miss: KF predict-only, emit src="kf" with the
predicted position -- unless there have been > 12 consecutive misses,
in which case we stop hallucinating and emit ball=null.

Output: out/<clip_stem>/ball.jsonl, one line per PROCESSED frame:
  {"frame_idx": int, "t": float, "ball": [x, y] | null, "src": "det"|"kf"}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hand-rolled constant-velocity Kalman filter (~30 lines)
# ---------------------------------------------------------------------------


class KalmanBall:
    """Constant-velocity KF over pixel state [x, y, vx, vy]."""

    def __init__(self, dt: float = 1.0, q_pos: float = 1.0, q_vel: float = 25.0, r_pos: float = 25.0):
        self.dt = dt
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1e4
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)
        self.R = np.eye(2) * r_pos
        self.initialized = False

    def init(self, x: float, y: float) -> None:
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([25.0, 25.0, 400.0, 400.0]).astype(float)
        self.initialized = True

    def predict(self) -> tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, z: tuple[float, float]) -> None:
        y = np.array(z) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def gate(self) -> float:
        """Gating distance: 150px + 3*sigma of predicted position uncertainty."""
        sigma = float(np.sqrt(max(self.P[0, 0], 0.0) + max(self.P[1, 1], 0.0)))
        return 150.0 + 3.0 * sigma


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

CONF_THRESH = 0.35
LOST_AFTER = 25  # processed frames since last accepted detection -> FULL mode
NULL_AFTER = 12  # consecutive misses -> stop emitting KF-predicted coords
FULL_SCAN_EVERY = 5  # while lost, only scan full frame every Nth processed frame
CROP_SIZE = 640


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _best_box(result, w: int, h: int, x_off: int = 0, y_off: int = 0):
    """Return (cx, cy, conf) of the highest-confidence box above CONF_THRESH, or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    keep = confs > CONF_THRESH
    if not keep.any():
        return None
    xyxy = boxes.xyxy.cpu().numpy()[keep]
    confs = confs[keep]
    best = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[best]
    cx = (x1 + x2) / 2.0 + x_off
    cy = (y1 + y2) / 2.0 + y_off
    return float(cx), float(cy), float(confs[best])


def _crop_window(px: float, py: float, w: int, h: int, size: int = CROP_SIZE):
    half = size // 2
    x1 = int(round(px)) - half
    y1 = int(round(py)) - half
    x1 = max(0, min(x1, max(0, w - size)))
    y1 = max(0, min(y1, max(0, h - size)))
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def track_ball(
    video_path: str,
    out_path: str | None = None,
    model_path: str = "models/football-ball-detection.pt",
    stride: int = 1,
    imgsz: int = 1280,
    device: str = "auto",
    max_frames: int | None = None,
) -> str:
    from ultralytics import YOLO

    video_path = str(video_path)
    resolved_device = _resolve_device(device)
    model = YOLO(model_path)

    if out_path is None:
        stem = Path(video_path).stem
        out_dir = Path("out") / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "ball.jsonl")
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    kf = KalmanBall()
    misses_since_accept = 0
    lost_scan_counter = 0  # counts processed frames while lost, for the every-5th schedule

    stats = {
        "n_processed": 0,
        "n_crop": 0,
        "n_full": 0,
        "n_det": 0,
        "n_kf": 0,
        "n_null": 0,
        "t_crop": 0.0,
        "t_full": 0.0,
        "transitions": [],  # (frame_idx, "crop"|"full")
    }
    last_mode = None

    frame_idx = 0
    processed_count = 0
    records = []

    while True:
        if frame_idx % stride == 0:
            ret, frame = cap.read()
        else:
            ret = cap.grab()
            frame = None
        if not ret:
            break

        if frame is not None:
            h, w = frame.shape[:2]
            lost = (not kf.initialized) or (misses_since_accept > LOST_AFTER)

            pred = kf.predict() if kf.initialized else None

            run_detector = (not lost) or (lost_scan_counter % FULL_SCAN_EVERY == 0)
            accepted = None
            mode = None

            if lost:
                lost_scan_counter += 1
            else:
                lost_scan_counter = 0

            if run_detector:
                if not lost:
                    mode = "crop"
                    x1, y1, x2, y2 = _crop_window(pred[0], pred[1], w, h)
                    crop = frame[y1:y2, x1:x2]
                    t0 = time.time()
                    results = model(crop, imgsz=640, device=resolved_device, verbose=False)
                    stats["t_crop"] += time.time() - t0
                    stats["n_crop"] += 1
                    cand = _best_box(results[0], w, h, x_off=x1, y_off=y1)
                else:
                    mode = "full"
                    t0 = time.time()
                    results = model(frame, imgsz=imgsz, device=resolved_device, verbose=False)
                    stats["t_full"] += time.time() - t0
                    stats["n_full"] += 1
                    cand = _best_box(results[0], w, h)

                if cand is not None:
                    cx, cy, conf = cand
                    if kf.initialized:
                        gate = kf.gate()
                        d = float(np.hypot(cx - pred[0], cy - pred[1]))
                        if d <= gate:
                            accepted = (cx, cy)
                    else:
                        accepted = (cx, cy)

                if mode is not None and mode != last_mode:
                    stats["transitions"].append((frame_idx, mode))
                    last_mode = mode

            t = frame_idx / fps
            if accepted is not None:
                if not kf.initialized:
                    kf.init(accepted[0], accepted[1])
                else:
                    kf.update(accepted)
                misses_since_accept = 0
                stats["n_det"] += 1
                records.append({"frame_idx": frame_idx, "t": t, "ball": [accepted[0], accepted[1]], "src": "det"})
            else:
                misses_since_accept += 1
                if kf.initialized and misses_since_accept <= NULL_AFTER:
                    stats["n_kf"] += 1
                    records.append({"frame_idx": frame_idx, "t": t, "ball": [pred[0], pred[1]], "src": "kf"})
                else:
                    stats["n_null"] += 1
                    records.append({"frame_idx": frame_idx, "t": t, "ball": None, "src": "kf"})

            stats["n_processed"] += 1
            processed_count += 1
            if max_frames is not None and processed_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    track_ball.last_stats = stats  # stash for self-check / CLI reporting
    return out_path


def _cli():
    ap = argparse.ArgumentParser(description="Ball tracking (SPEC C9)")
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory (writes ball.jsonl inside)")
    ap.add_argument("--model", default="models/football-ball-detection.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_path = None
    if args.out is not None:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "ball.jsonl")

    path = track_ball(
        args.video,
        out_path=out_path,
        model_path=args.model,
        stride=args.stride,
        imgsz=args.imgsz,
        device=args.device,
        max_frames=args.max_frames,
    )
    stats = track_ball.last_stats
    n = stats["n_processed"]
    found = stats["n_det"] + stats["n_kf"]
    print(f"wrote {path} ({n} frames, ball found in {found}/{n} = {found / max(n,1):.0%})")
    print(f"  det={stats['n_det']} kf={stats['n_kf']} null={stats['n_null']}")
    print(f"  crop calls={stats['n_crop']} full calls={stats['n_full']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _cli()
    else:
        # Self-check: 40 frames of the smoke clip.
        clip = "data/clips/bundesliga_smoke.mp4"
        t0 = time.time()
        path = track_ball(clip, out_path="out/bundesliga_smoke/ball.jsonl", stride=1, max_frames=40)
        elapsed = time.time() - t0
        stats = track_ball.last_stats
        n = stats["n_processed"]
        found = stats["n_det"] + stats["n_kf"]
        frac_found = found / max(n, 1)
        frac_crop = stats["n_crop"] / max(stats["n_crop"] + stats["n_full"], 1)

        print(f"self-check: {n} frames processed in {elapsed:.2f}s ({elapsed / max(n,1):.3f}s/frame avg)")
        print(f"  ball found (det+kf) in {found}/{n} = {frac_found:.0%} of frames")
        print(f"  det={stats['n_det']} kf={stats['n_kf']} null={stats['n_null']}")
        print(f"  crop-mode calls={stats['n_crop']} full-mode calls={stats['n_full']} (crop fraction={frac_crop:.0%})")
        if stats["n_crop"] > 0:
            print(f"  avg crop-call time={stats['t_crop']/stats['n_crop']:.3f}s")
        if stats["n_full"] > 0:
            print(f"  avg full-call time={stats['t_full']/stats['n_full']:.3f}s")
        print("  mode transitions:", stats["transitions"])

        assert frac_found >= 0.40, f"ball found in only {frac_found:.0%} of frames, expected >= 40%"
        print(f"OK: wrote {path}")
