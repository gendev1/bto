"""Shot classifier (SPEC C1): main wide tactical camera vs everything else
(replay / close-up / graphic / crowd). Heuristic, pixel-only, no ML.

Signal (computed per sampled frame on the raw BGR pixels):
  green_frac  : fraction of green (pitch-colored) pixels in the lower 2/3
                of the frame, HSV H in [35, 85], S > 60, V > 40.
  width_span  : (rightmost - leftmost) column containing any green pixel,
                as a fraction of frame width. Main camera pitch fills most
                of the frame width; close-ups/graphics/crowd shots don't.
  horizon_std : for each column with green, the row of the topmost green
                pixel forms a "horizon" profile across the frame. On the
                main camera this profile is a roughly straight, coherent
                line (the pitch/stand boundary), so its std (as a fraction
                of frame height) is low. Close-ups, crowd shots, and
                graphics have no coherent horizon line, so std is high (or
                undefined -> treated as high).

A frame is 'main' iff green_frac >= 0.55 and width_span >= 0.75 and
horizon_std <= 0.15. Thresholds were calibrated by hand-labeling 20 frames
each from bundesliga_smoke.mp4 (all main) and cwc2021 (mixed, incl. real
replays/close-ups/crowd/graphics) -- see scratchpad calibration notes in
the M2 task. 39/39 hand labels agreed with these thresholds.

Raw per-frame classifications are then smoothed with hysteresis: a shot
change only takes effect once >= HYSTERESIS consecutive samples agree,
since broadcast cuts are hard cuts (no gradual transition to detect, but
we don't want single flickery samples to spuriously toggle the label).
"""

import argparse
import json
import os

import cv2
import numpy as np

GREEN_FRAC_MIN = 0.55
WIDTH_SPAN_MIN = 0.75
HORIZON_STD_MAX = 0.15
HYSTERESIS = 3

_H_LO, _H_HI = 35, 85
_S_MIN = 60
_V_MIN = 40


def _green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([_H_LO, _S_MIN, _V_MIN])
    upper = np.array([_H_HI, 255, 255])
    return cv2.inRange(hsv, lower, upper)


def frame_features(frame):
    """Return (green_frac, width_span, horizon_std) for one BGR frame."""
    h, w = frame.shape[:2]
    mask = _green_mask(frame)

    y0 = h // 3
    lower_mask = mask[y0:, :]
    green_frac = float(lower_mask.mean() / 255.0)

    col_any = lower_mask.any(axis=0)
    cols = np.where(col_any)[0]
    width_span = float((cols[-1] - cols[0] + 1) / w) if len(cols) else 0.0

    col_any_full = mask.any(axis=0)
    cols_full = np.where(col_any_full)[0]
    if len(cols_full) < w * 0.3:
        horizon_std = float("inf")
    else:
        tops = np.argmax(mask[:, cols_full] > 0, axis=0).astype(np.float32)
        horizon_std = float(np.std(tops) / h)

    return green_frac, width_span, horizon_std


def classify_frame(frame):
    """Return 'main' or 'other' for one BGR frame, no temporal smoothing."""
    green_frac, width_span, horizon_std = frame_features(frame)
    is_main = (
        green_frac >= GREEN_FRAC_MIN
        and width_span >= WIDTH_SPAN_MIN
        and horizon_std <= HORIZON_STD_MAX
    )
    return "main" if is_main else "other"


def _smooth(labels, hysteresis=HYSTERESIS):
    """Hysteresis smoothing: only flip the running label once the new label
    has appeared for >= hysteresis consecutive samples."""
    if not labels:
        return []
    out = [labels[0]]
    current = labels[0]
    run_label = labels[0]
    run_len = 0
    for lab in labels:
        if lab == run_label:
            run_len += 1
        else:
            run_label = lab
            run_len = 1
        if run_label != current and run_len >= hysteresis:
            current = run_label
        out.append(current)
    return out[1:]


def classify_shots(video_path, out_path=None, stride=5, max_frames=None):
    """Classify every stride-th frame of video_path as 'main' or 'other'.

    Writes shots.jsonl (one line per processed frame_idx) if out_path is
    given (a directory; file written as out_path/shots.jsonl), and always
    returns the list of {"frame_idx", "t", "shot"} dicts.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = []
    raw_labels = []
    idx = 0
    processed = 0
    while True:
        if max_frames is not None and processed >= max_frames:
            break
        if idx >= n_total > 0:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        raw_labels.append(classify_frame(frame))
        frame_idxs.append(idx)
        processed += 1
        idx += stride
    cap.release()

    smoothed = _smooth(raw_labels)

    records = [
        {"frame_idx": fi, "t": fi / fps, "shot": lab}
        for fi, lab in zip(frame_idxs, smoothed)
    ]

    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        dest = os.path.join(out_path, "shots.jsonl")
        with open(dest, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    return records


def _clip_stem(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join("out", _clip_stem(args.video))
    records = classify_shots(
        args.video, out_path=out_dir, stride=args.stride, max_frames=args.max_frames
    )
    n_main = sum(1 for r in records if r["shot"] == "main")
    n_other = len(records) - n_main
    n = max(len(records), 1)
    print(f"{args.video}: {len(records)} frames processed -> {out_dir}/shots.jsonl")
    print(f"  main={n_main} ({n_main / n:.1%})  other={n_other} ({n_other / n:.1%})")


if __name__ == "__main__":
    import time

    if len(__import__("sys").argv) > 1:
        main()
    else:
        # Self-check: <= 40 frames on the smoke clip.
        video = "data/clips/bundesliga_smoke.mp4"
        t0 = time.time()
        records = classify_shots(video, out_path=None, stride=5, max_frames=40)
        dt = time.time() - t0
        n = len(records)
        n_main = sum(1 for r in records if r["shot"] == "main")
        print(f"self-check: {n} frames in {dt:.2f}s ({dt / max(n, 1) * 1000:.1f} ms/frame)")
        print(f"  main={n_main}/{n}")
        assert n > 0
        assert all(r["shot"] in ("main", "other") for r in records)
        # smoke clip should be almost entirely 'main'
        assert n_main / n >= 0.9, "expected smoke clip to be mostly main-camera"
        print("OK")
