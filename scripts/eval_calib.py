"""SPEC S7 calibration verifier (M3).

Usage:
    uv run python scripts/eval_calib.py <video> <calib.jsonl> [--n 8] [--out DIR]

Reads a calib.jsonl (frozen M3 interchange, one line per processed
MAIN-shot frame: {"frame_idx","t","H"|null,"n_kp","rmse_m","src"}) and
reports:

  1. valid-H fraction, broken down by src (fit/ema/held) vs null.
  2. rmse_m distribution over 'fit' frames (mean/median/p90) vs the SPEC S7
     target of < 1.5 m RMS.
  3. temporal stability: for consecutive frames that both have a valid H,
     the meter-space displacement of the projected video-frame-center
     point -- catches jitter that per-frame RMSE and even the EMA smoother
     can miss (two "good" homographies that disagree with each other).
  4. a VISUAL check: for --n frames spread across the file (only frames
     with a valid H), draw the canonical pitch wireframe (touchlines,
     halfway line, both penalty + goal boxes, center circle) projected
     through H^-1 onto the original video frame, saved as jpgs. This is
     the actual deliverable of this module -- a low numeric RMSE can
     still come from a homography fit to the wrong (but self-consistent)
     keypoints, so a human (the calling agent) must eyeball every jpg and
     grade it good / drift / bad; see the printed VISUAL CHECK section and
     record verdicts back into calib_report.json via --record-verdicts.

Writes out/<clip_stem>/calib_report.json and out/<clip_stem>/calib_viz/*.jpg.

Self-check (no calib.jsonl available): pass --selfcheck-gen to generate a
small calib.jsonl directly from the pitch keypoint model (stride 5, <=40
GPU frames, frozen cm->m conversion, plain cv2.findHomography, no
smoothing -- every entry has src="fit") before running the eval.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "models/football-pitch-detection.pt"


def _load_pitch_config():
    spec = importlib.util.spec_from_file_location(
        "roboflow_soccer_pitch_config", ROOT / "docs" / "roboflow_soccer_pitch_config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SoccerPitchConfiguration()


def cm_to_m(vx: float, vy: float) -> tuple[float, float]:
    """Frozen conversion: canonical vertex (vx,vy) cm -> our 105x68m pitch."""
    x_m = vx / 12000.0 * 105.0
    y_m = 68.0 - vy / 7000.0 * 68.0
    return (x_m, y_m)


def build_wireframe_m(cfg) -> list[list[tuple[float, float]]]:
    """Polylines (meter space) for touchlines/halfway/penalty+goal boxes
    (from cfg.edges, each a 2-point segment) plus the center circle
    (sampled as a 36-gon, since edges only encodes straight tangent lines
    through the circle's box, not the arc itself)."""
    verts_m = [cm_to_m(vx, vy) for vx, vy in cfg.vertices]
    lines = [[verts_m[a - 1], verts_m[b - 1]] for a, b in cfg.edges]

    cx_cm, cy_cm = cfg.length / 2.0, cfg.width / 2.0
    r_cm = cfg.centre_circle_radius
    circle = []
    for i in range(37):
        theta = 2 * np.pi * i / 36
        circle.append(cm_to_m(cx_cm + r_cm * np.cos(theta), cy_cm + r_cm * np.sin(theta)))
    lines.append(circle)
    return lines


# ---------------------------------------------------------------------------
# calib.jsonl I/O + stats
# ---------------------------------------------------------------------------


def load_entries(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _pctile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def compute_stats(entries: list[dict], video_path: str) -> dict:
    total = len(entries)
    valid = [e for e in entries if e.get("H") is not None]
    n_valid = len(valid)
    src_counts: dict[str, int] = {}
    for e in entries:
        key = e.get("src") if e.get("H") is not None else "null"
        src_counts[key] = src_counts.get(key, 0) + 1

    fit_rmse = [e["rmse_m"] for e in entries if e.get("src") == "fit" and e.get("rmse_m") is not None]
    rmse_stats = {
        "n": len(fit_rmse),
        "mean": statistics.mean(fit_rmse) if fit_rmse else None,
        "median": statistics.median(fit_rmse) if fit_rmse else None,
        "p90": _pctile(fit_rmse, 0.90) if fit_rmse else None,
        "target_m": 1.5,
        "frac_under_target": (sum(1 for r in fit_rmse if r < 1.5) / len(fit_rmse)) if fit_rmse else None,
    }

    # temporal stability: displacement (meters) of the projected video-frame-center
    # point between consecutive valid-H entries.
    cap = cv2.VideoCapture(video_path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0
    cap.release()
    center_px = np.array([w / 2.0, h / 2.0, 1.0])

    disps = []
    prev_center_m = None
    for e in entries:
        if e.get("H") is None:
            prev_center_m = None
            continue
        Hm = np.array(e["H"], dtype=float).reshape(3, 3)
        p = Hm @ center_px
        p = p[:2] / p[2]
        if prev_center_m is not None:
            disps.append(float(np.linalg.norm(p - prev_center_m)))
        prev_center_m = p

    stability = {
        "video_wh": [w, h],
        "n_pairs": len(disps),
        "mean_disp_m": statistics.mean(disps) if disps else None,
        "median_disp_m": statistics.median(disps) if disps else None,
        "max_disp_m": max(disps) if disps else None,
    }

    return {
        "total_frames": total,
        "n_valid_H": n_valid,
        "valid_H_fraction": n_valid / total if total else None,
        "src_breakdown": src_counts,
        "rmse_fit": rmse_stats,
        "temporal_stability": stability,
    }


# ---------------------------------------------------------------------------
# Visual check
# ---------------------------------------------------------------------------


def sample_indices(n_available: int, n: int) -> list[int]:
    if n_available == 0:
        return []
    n = min(n, n_available)
    if n == 1:
        return [0]
    return sorted(set(round(i * (n_available - 1) / (n - 1)) for i in range(n)))


def draw_wireframe(frame_bgr: np.ndarray, H_flat: list[float], wireframe_m: list[list[tuple[float, float]]]) -> np.ndarray:
    Hm = np.array(H_flat, dtype=float).reshape(3, 3)
    Hinv = np.linalg.inv(Hm)
    img = frame_bgr.copy()
    for line_m in wireframe_m:
        pts_m = np.array([[x, y, 1.0] for x, y in line_m]).T  # 3xN
        pts_px = Hinv @ pts_m
        pts_px = (pts_px[:2] / pts_px[2]).T  # Nx2
        pts_px = pts_px.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_px], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return img


def visual_check(video_path: str, entries: list[dict], n: int, viz_dir: Path, wireframe_m) -> list[dict]:
    viz_dir.mkdir(parents=True, exist_ok=True)
    valid = [e for e in entries if e.get("H") is not None]
    idxs = sample_indices(len(valid), n)
    picked = [valid[i] for i in idxs]

    cap = cv2.VideoCapture(video_path)
    saved = []
    for e in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["frame_idx"])
        ok, frame = cap.read()
        if not ok:
            continue
        img = draw_wireframe(frame, e["H"], wireframe_m)
        out_path = viz_dir / f"frame_{e['frame_idx']:06d}.jpg"
        cv2.imwrite(str(out_path), img)
        saved.append({"frame_idx": e["frame_idx"], "t": e.get("t"), "src": e.get("src"), "rmse_m": e.get("rmse_m"), "path": str(out_path)})
    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Self-check calib.jsonl generator (no smoothing, src="fit" always)
# ---------------------------------------------------------------------------


def generate_calib_selfcheck(video_path: str, out_path: str, stride: int = 5, max_frames: int = 30, imgsz: int = 640) -> None:
    import torch
    from ultralytics import YOLO

    cfg = _load_pitch_config()
    dst_all_m = np.array([cm_to_m(vx, vy) for vx, vy in cfg.vertices], dtype=np.float64)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(DEFAULT_MODEL)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    n_written = 0
    frame_idx = 0
    with open(out_path, "w") as f:
        while n_written < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            res = model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
            H = None
            n_kp = 0
            rmse = None
            if res.keypoints is not None and len(res.keypoints) > 0:
                kp = res.keypoints.data[0].cpu().numpy()  # (32, 3): x,y,conf
                mask = kp[:, 2] > 0.5
                n_kp = int(mask.sum())
                if n_kp >= 4:
                    src_px = kp[mask, :2].astype(np.float64)
                    dst_m = dst_all_m[mask]
                    Hmat, inliers = cv2.findHomography(src_px, dst_m, cv2.RANSAC, ransacReprojThreshold=1.5)
                    if Hmat is not None:
                        proj = cv2.perspectiveTransform(src_px.reshape(-1, 1, 2), Hmat).reshape(-1, 2)
                        err = np.linalg.norm(proj - dst_m, axis=1)
                        rmse = float(np.sqrt(np.mean(err**2)))
                        H = Hmat.flatten().tolist()
            entry = {
                "frame_idx": frame_idx,
                "t": frame_idx / fps,
                "H": H,
                "n_kp": n_kp,
                "rmse_m": rmse,
                "src": "fit",
            }
            f.write(json.dumps(entry) + "\n")
            n_written += 1
            frame_idx += stride
    cap.release()
    print(f"[selfcheck] wrote {n_written} entries to {out_path}")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("calib_jsonl")
    ap.add_argument("--n", type=int, default=8, help="number of sample frames for the visual check")
    ap.add_argument("--out", default=None, help="output dir (default out/<clip_stem>)")
    ap.add_argument("--selfcheck-gen", action="store_true", help="generate calib.jsonl at calib_jsonl path if missing (stride-5 pose-model pass, <=40 frames, no smoothing)")
    ap.add_argument("--selfcheck-max-frames", type=int, default=30)
    args = ap.parse_args()

    stem = Path(args.video).stem
    out_dir = Path(args.out) if args.out else ROOT / "out" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.selfcheck_gen and not os.path.exists(args.calib_jsonl):
        generate_calib_selfcheck(args.video, args.calib_jsonl, max_frames=args.selfcheck_max_frames)

    entries = load_entries(args.calib_jsonl)
    if not entries:
        print(f"no entries in {args.calib_jsonl}", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(entries, args.video)

    cfg = _load_pitch_config()
    wireframe_m = build_wireframe_m(cfg)
    viz_dir = out_dir / "calib_viz"
    saved = visual_check(args.video, entries, args.n, viz_dir, wireframe_m)

    report = {
        "video": args.video,
        "calib_jsonl": args.calib_jsonl,
        **stats,
        "visual_check_frames": saved,
        "visual_verdicts": None,  # filled in by the human/agent eyeball pass, see calib_report.json after grading
    }
    report_path = out_dir / "calib_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"== calib eval: {args.video} / {args.calib_jsonl} ==")
    print(f"total frames: {stats['total_frames']}, valid H: {stats['n_valid_H']} ({stats['valid_H_fraction']:.1%})")
    print(f"src breakdown: {stats['src_breakdown']}")
    r = stats["rmse_fit"]
    if r["n"]:
        print(f"rmse_m (fit, n={r['n']}): mean={r['mean']:.3f} median={r['median']:.3f} p90={r['p90']:.3f} "
              f"(target <1.5m: {r['frac_under_target']:.1%} under)")
    else:
        print("rmse_m (fit): no fit frames with rmse_m")
    ts = stats["temporal_stability"]
    if ts["n_pairs"]:
        print(f"temporal stability: mean_disp={ts['mean_disp_m']:.3f}m median={ts['median_disp_m']:.3f}m max={ts['max_disp_m']:.3f}m over {ts['n_pairs']} consecutive valid pairs")
    else:
        print("temporal stability: no consecutive valid-H pairs")
    print(f"\nVISUAL CHECK: {len(saved)} frames saved to {viz_dir}")
    for s in saved:
        print(f"  frame {s['frame_idx']:>6}  t={s['t']:.2f}s  src={s['src']}  rmse_m={s['rmse_m']}  -> {s['path']}")
    print(f"\nreport written to {report_path}")


if __name__ == "__main__":
    main()
