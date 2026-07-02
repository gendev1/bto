"""Team assignment via HSV torso-histogram k-means (SPEC C4).

Reads a detections.jsonl (frozen interchange format, see SPEC.md S4/S3) plus
the source video, and assigns every track id to one of home/away/gk/ref,
writing teams.json:

    {"<tid>": "home"|"away"|"gk"|"ref", ...}

Method: for each cls='player' track, sample up to N_SAMPLES frames spread
over its life, crop the torso (rows 15%-45% of the box height, middle 60% of
the width), mask out pitch-green pixels (HSV hue 35-85), and build an
L1-normalized 2D hue-saturation histogram of what's left. Average the
histograms per track, then hand-rolled k-means(k=2, kmeans++ init, fixed
seed) clusters tracks into two kits; cluster 0 -> 'home', cluster 1 ->
'away' (arbitrary but stable given the fixed seed). cls='goalkeeper' tracks
are always 'gk'; cls='referee' tracks are always 'ref'. Tracks with too few
or too small torso crops are still assigned to the nearest centroid, and are
listed (with a low-confidence note) in the sidecar teams_confidence.json.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

HUE_BINS = 16
SAT_BINS = 8
GREEN_H_LO, GREEN_H_HI = 35, 85
N_SAMPLES = 10
MIN_CROP_PX = 6  # min crop width/height in px to trust a sample
MIN_VALID_SAMPLES = 2  # below this a track is flagged low-confidence
KMEANS_SEED = 0
KMEANS_ITERS = 20

TEAM_COLORS_BGR = {
    "home": (0, 0, 220),
    "away": (220, 90, 0),
    "gk": (0, 210, 210),
    "ref": (20, 20, 20),
    "?": (255, 255, 255),
}


# --------------------------------------------------------------------------
# detections.jsonl -> per-track history
# --------------------------------------------------------------------------

def _load_detections(path):
    """frame_idx -> parsed record dict."""
    frames = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frames[rec["frame_idx"]] = rec
    return frames


def _tracks_from_frames(frames):
    """tid -> {'cls': majority class, 'frames': [(frame_idx, xyxy), ...] sorted}."""
    tid_boxes = defaultdict(list)
    tid_cls_counts = defaultdict(lambda: defaultdict(int))
    for frame_idx, rec in frames.items():
        for box in rec["boxes"]:
            tid = box.get("tid")
            if tid is None:
                continue
            tid_boxes[tid].append((frame_idx, box["xyxy"]))
            tid_cls_counts[tid][box["cls"]] += 1
    tracks = {}
    for tid, boxes in tid_boxes.items():
        cls = max(tid_cls_counts[tid].items(), key=lambda kv: kv[1])[0]
        boxes.sort(key=lambda b: b[0])
        tracks[tid] = {"cls": cls, "frames": boxes}
    return tracks


def _sample_frame_indices(frame_list, n=N_SAMPLES):
    if len(frame_list) <= n:
        return list(frame_list)
    idxs = sorted(set(np.linspace(0, len(frame_list) - 1, n).round().astype(int).tolist()))
    return [frame_list[i] for i in idxs]


# --------------------------------------------------------------------------
# torso crop -> hue/sat histogram
# --------------------------------------------------------------------------

def _torso_crop(frame, xyxy):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    tx1, tx2 = x1 + 0.20 * w, x1 + 0.80 * w
    ty1, ty2 = y1 + 0.15 * h, y1 + 0.45 * h
    ix1, iy1 = int(round(tx1)), int(round(ty1))
    ix2, iy2 = int(round(tx2)), int(round(ty2))
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(frame.shape[1], ix2), min(frame.shape[0], iy2)
    if ix2 - ix1 < MIN_CROP_PX or iy2 - iy1 < MIN_CROP_PX:
        return None
    return frame[iy1:iy2, ix1:ix2]


def _hue_sat_hist(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    green = (h >= GREEN_H_LO) & (h <= GREEN_H_HI)
    keep = ~green
    if keep.sum() < 5:
        return None
    hist, _, _ = np.histogram2d(
        h[keep].ravel().astype(np.float64),
        s[keep].ravel().astype(np.float64),
        bins=[HUE_BINS, SAT_BINS],
        range=[[0, 180], [0, 256]],
    )
    total = hist.sum()
    if total <= 0:
        return None
    return (hist / total).ravel()


def _collect_track_features(video_path, tracks):
    """Returns (tid -> mean histogram vector for tracks with >=1 sample, set of low-confidence tids)."""
    needed_frames = defaultdict(list)  # frame_idx -> [(tid, xyxy)]
    sample_counts = {}
    for tid, t in tracks.items():
        if t["cls"] != "player":
            continue
        sampled = _sample_frame_indices(t["frames"])
        sample_counts[tid] = len(sampled)
        for frame_idx, xyxy in sampled:
            needed_frames[frame_idx].append((tid, xyxy))

    hists = defaultdict(list)
    if needed_frames:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        remaining = set(needed_frames.keys())
        frame_idx = 0
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in needed_frames:
                for tid, xyxy in needed_frames[frame_idx]:
                    crop = _torso_crop(frame, xyxy)
                    if crop is not None:
                        hist = _hue_sat_hist(crop)
                        if hist is not None:
                            hists[tid].append(hist)
                remaining.discard(frame_idx)
            frame_idx += 1
        cap.release()

    feats = {}
    low_conf = set()
    for tid, n_sampled in sample_counts.items():
        vecs = hists.get(tid, [])
        if len(vecs) < MIN_VALID_SAMPLES:
            low_conf.add(tid)
        if vecs:
            feats[tid] = np.mean(vecs, axis=0)
    return feats, low_conf


# --------------------------------------------------------------------------
# hand-rolled k-means (kmeans++ init)
# --------------------------------------------------------------------------

def _kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0), axis=0)
        s = d2.sum()
        probs = d2 / s if s > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=probs)])
    return np.stack(centers)


def _kmeans(X, k=2, iters=KMEANS_ITERS, seed=KMEANS_SEED):
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(X, k, rng)
    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(iters):
        dists = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        labels = np.argmin(dists, axis=0)
        centers = np.stack(
            [X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)]
        )
    return labels, centers


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def assign_teams(video_path, detections_jsonl, out_path=None):
    """Assigns every track id to home/away/gk/ref, writes teams.json, returns the dict."""
    frames = _load_detections(detections_jsonl)
    tracks = _tracks_from_frames(frames)
    feats, low_conf = _collect_track_features(video_path, tracks)

    global_mean = np.mean(list(feats.values()), axis=0) if feats else np.zeros(HUE_BINS * SAT_BINS)
    player_tids = [tid for tid, t in tracks.items() if t["cls"] == "player"]
    for tid in player_tids:
        if tid not in feats:
            feats[tid] = global_mean
            low_conf.add(tid)

    teams = {}
    notes = {}

    if player_tids:
        X = np.stack([feats[tid] for tid in player_tids])
        if len(player_tids) >= 2:
            labels, _ = _kmeans(X, k=2)
        else:
            labels = np.zeros(len(player_tids), dtype=int)
        for tid, lab in zip(player_tids, labels):
            teams[str(tid)] = "home" if lab == 0 else "away"
            if tid in low_conf:
                notes[str(tid)] = "low_confidence: insufficient/small torso crops, assigned to nearest centroid"

    for tid, t in tracks.items():
        if t["cls"] == "goalkeeper":
            teams[str(tid)] = "gk"
        elif t["cls"] == "referee":
            teams[str(tid)] = "ref"

    if out_path is None:
        out_path = "teams.json"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(teams, f, indent=2, sort_keys=True)

    if notes:
        meta_path = os.path.join(out_dir or ".", "teams_confidence.json")
        with open(meta_path, "w") as f:
            json.dump(notes, f, indent=2, sort_keys=True)

    return teams


def sanity_report(video_path, detections_jsonl, teams_json, out_dir=None, n=6):
    """Saves n annotated sample frames (boxes tinted by team assignment) as jpgs; returns their paths."""
    frames = _load_detections(detections_jsonl)
    with open(teams_json) as f:
        teams = json.load(f)

    frame_idxs = sorted(frames.keys())
    if len(frame_idxs) <= n:
        pick = frame_idxs
    else:
        idxs = sorted(set(np.linspace(0, len(frame_idxs) - 1, n).round().astype(int).tolist()))
        pick = [frame_idxs[i] for i in idxs]
    pick_set = set(pick)

    out_dir = out_dir or "sanity"
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_idx = 0
    while pick_set:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in pick_set:
            rec = frames[frame_idx]
            for box in rec["boxes"]:
                tid = box.get("tid")
                team = teams.get(str(tid), "?") if tid is not None else "?"
                color = TEAM_COLORS_BGR.get(team, (255, 255, 255))
                x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{tid}:{team}"
                cv2.putText(frame, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)
            pick_set.discard(frame_idx)
        frame_idx += 1
    cap.release()
    return saved


# --------------------------------------------------------------------------
# self-check helper: generate a tiny detections.jsonl if none exists yet
# (bto.vision.detect may not be importable while this module is built)
# --------------------------------------------------------------------------

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _generate_smoke_detections(video_path, out_path, n_frames=30,
                                model_path="models/football-player-detection.pt"):
    """Self-check-only detections.jsonl generator: plain per-frame YOLO predict
    (no ultralytics .track(), whose bytetrack matcher needs the optional
    unvendored `lap` package) + a hand-rolled greedy IoU tracker, same-class
    only. Good enough to exercise this module's own logic; not a tracker
    deliverable (that's bto.vision.detect / bto.vision.track)."""
    from ultralytics import YOLO

    names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = YOLO(model_path)
    next_tid = 1
    active = []  # list of {"tid": int, "cls": str, "xyxy": [..]}

    with open(out_path, "w") as f:
        count = 0
        for result in model.predict(
            source=video_path, imgsz=1280, stream=True, verbose=False, device="mps",
        ):
            if count >= n_frames:
                break
            dets = []
            if result.boxes is not None:
                for b in result.boxes:
                    cls_id = int(b.cls[0])
                    dets.append({
                        "cls": names.get(cls_id, "player"),
                        "conf": float(b.conf[0]),
                        "xyxy": [float(v) for v in b.xyxy[0].tolist()],
                    })

            unmatched = list(range(len(dets)))
            new_active = []
            for tr in active:
                best_j, best_iou = None, 0.3
                for j in unmatched:
                    if dets[j]["cls"] != tr["cls"]:
                        continue
                    iou = _iou(tr["xyxy"], dets[j]["xyxy"])
                    if iou > best_iou:
                        best_j, best_iou = j, iou
                if best_j is not None:
                    dets[best_j]["tid"] = tr["tid"]
                    unmatched.remove(best_j)
                    new_active.append({"tid": tr["tid"], "cls": dets[best_j]["cls"], "xyxy": dets[best_j]["xyxy"]})
            for j in unmatched:
                dets[j]["tid"] = next_tid
                new_active.append({"tid": next_tid, "cls": dets[j]["cls"], "xyxy": dets[j]["xyxy"]})
                next_tid += 1
            active = new_active

            f.write(json.dumps({"frame_idx": count, "t": count / fps, "boxes": dets}) + "\n")
            count += 1
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Assign tracks to home/away/gk/ref (SPEC C4)")
    ap.add_argument("video")
    ap.add_argument("detections", help="detections.jsonl (frozen format); auto-generated (30 frames) if missing")
    ap.add_argument("--out", default=None, help="output dir (default: out/<clip_stem>)")
    args = ap.parse_args()

    clip_stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out or os.path.join("out", clip_stem)
    os.makedirs(out_dir, exist_ok=True)

    detections_path = args.detections
    if not os.path.exists(detections_path):
        print(f"[teams] {detections_path} not found; generating a 30-frame smoke detections.jsonl for self-check")
        _generate_smoke_detections(args.video, detections_path, n_frames=30)

    teams_path = os.path.join(out_dir, "teams.json")
    t0 = time.time()
    teams = assign_teams(args.video, detections_path, out_path=teams_path)
    dt = time.time() - t0
    n_tracks = len(teams)
    print(f"[teams] wrote {teams_path}: {n_tracks} tracks in {dt:.2f}s ({dt / max(n_tracks, 1):.3f}s/track)")

    sanity_dir = os.path.join(out_dir, "sanity")
    saved = sanity_report(args.video, detections_path, teams_path, out_dir=sanity_dir, n=6)
    print(f"[teams] wrote {len(saved)} sanity frames to {sanity_dir}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
