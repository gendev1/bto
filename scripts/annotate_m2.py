"""M2 annotate: merge perception contract files (detections/shots/teams/ball)
into perception.jsonl and an annotated video.

CLI:
    uv run python scripts/annotate_m2.py <video> <out_dir_with_jsonls>

Reads (any missing file degrades gracefully):
    <out_dir>/detections.jsonl
    <out_dir>/shots.jsonl
    <out_dir>/teams.json
    <out_dir>/ball.jsonl

Writes:
    <out_dir>/perception.jsonl
    <out_dir>/m2_annotated.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

# ---- colors (BGR) ----
COLOR_HOME = (255, 140, 0)   # blue-ish
COLOR_AWAY = (0, 140, 255)   # orange
COLOR_GK = (0, 255, 255)     # yellow
COLOR_REF = (200, 200, 200)  # light grey
COLOR_UNKNOWN = (255, 255, 255)
COLOR_BALL = (0, 0, 255)     # red
BALL_TRAIL_LEN = 10


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def foot_point(xyxy):
    x1, y1, x2, y2 = xyxy
    return [(x1 + x2) / 2.0, y2]


def merge_perception(detections, shots, teams, ball_rows, frame_teams=None):
    """Merge the four contract streams into a list of per-frame perception dicts.

    Returns list sorted by frame_idx, and the raw list of processed frame_idxs
    used to drive the output (union of everything we have, detections preferred).

    frame_teams: optional {frame_idx: {"<tid>": "home"|"away"}} per-frame team
    labels (from teams_frames.jsonl); preferred over the per-track teams dict
    because ByteTrack identity swaps make long tracks kit-impure.
    """
    frame_teams = frame_teams or {}
    shot_by_frame = {r["frame_idx"]: r.get("shot", "main") for r in shots}
    ball_by_frame = {r["frame_idx"]: r for r in ball_rows}

    det_by_frame = defaultdict(list)
    for r in detections:
        det_by_frame[r["frame_idx"]].append(r)

    frame_idxs = set(det_by_frame.keys()) | set(ball_by_frame.keys()) | set(shot_by_frame.keys())
    frame_idxs = sorted(frame_idxs)

    merged = []
    for fidx in frame_idxs:
        # t: prefer detections row's t, else ball row's t, else None -> filled later
        t = None
        players = []
        if fidx in det_by_frame:
            rows = det_by_frame[fidx]
            t = rows[0].get("t")
            for r in rows:
                for b in r.get("boxes", []):
                    if b.get("cls") == "ball":
                        continue
                    tid = b.get("tid")
                    team = None
                    if tid is not None:
                        team = frame_teams.get(fidx, {}).get(str(tid)) or teams.get(str(tid))
                    players.append({
                        "tid": tid,
                        "team": team,
                        "cls": b.get("cls"),
                        "conf": b.get("conf"),
                        "box": b.get("xyxy"),
                        "foot": foot_point(b.get("xyxy")),
                    })
        ball_xy = None
        if fidx in ball_by_frame:
            brow = ball_by_frame[fidx]
            ball_xy = brow.get("ball")
            if t is None:
                t = brow.get("t")
        shot = shot_by_frame.get(fidx, "main")
        merged.append({
            "frame_idx": fidx,
            "t": t,
            "shot": shot,
            "players": players,
            "ball": ball_xy,
        })
    return merged


def write_perception_jsonl(merged, out_path):
    with open(out_path, "w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")


def team_color(player):
    team = player.get("team")
    if team == "home":
        return COLOR_HOME
    if team == "away":
        return COLOR_AWAY
    if team == "gk":
        return COLOR_GK
    if team == "ref":
        return COLOR_REF
    # fallback on cls if no team assigned yet
    cls = player.get("cls")
    if cls == "goalkeeper":
        return COLOR_GK
    if cls == "referee":
        return COLOR_REF
    return COLOR_UNKNOWN


def draw_frame(frame, row, ball_trail):
    canvas = frame.copy()

    for p in row["players"]:
        color = team_color(p)
        fx, fy = p["foot"]
        fx, fy = int(fx), int(fy)
        cv2.ellipse(canvas, (fx, fy), (18, 7), 0, -30, 220, color, 2, cv2.LINE_AA)
        label = str(p["tid"]) if p["tid"] is not None else "?"
        cv2.putText(canvas, label, (fx - 8, fy - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    if ball_trail:
        pts = [(int(x), int(y)) for (x, y) in ball_trail if x is not None]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, COLOR_BALL, 2, cv2.LINE_AA)
        if row["ball"] is not None:
            bx, by = row["ball"]
            cv2.circle(canvas, (int(bx), int(by)), 6, COLOR_BALL, -1, cv2.LINE_AA)

    shot = row.get("shot", "main")
    t = row.get("t")
    t_str = f"{t:.2f}s" if t is not None else "?"
    banner = f"shot={shot}  t={t_str}  frame={row['frame_idx']}"
    cv2.rectangle(canvas, (0, 0), (420, 30), (0, 0, 0), -1)
    cv2.putText(canvas, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    if shot != "main":
        out = cv2.addWeighted(canvas, 0.35, frame, 0.65, 0)
        # keep banner fully visible on top of dimmed overlay
        cv2.rectangle(out, (0, 0), (420, 30), (0, 0, 0), -1)
        cv2.putText(out, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out
    return canvas


def compute_stats(merged, fps):
    n_frames = len(merged)
    tid_first = {}
    tid_last = {}
    tid_frame_count = defaultdict(int)
    team_tids = defaultdict(set)
    ball_covered = 0

    frame_idxs_sorted = [r["frame_idx"] for r in merged]
    first_frame_idx = frame_idxs_sorted[0] if frame_idxs_sorted else None

    for row in merged:
        for p in row["players"]:
            tid = p["tid"]
            if tid is None:
                continue
            t = row["t"] if row["t"] is not None else row["frame_idx"] / fps
            if tid not in tid_first:
                tid_first[tid] = (row["frame_idx"], t)
            tid_last[tid] = (row["frame_idx"], t)
            tid_frame_count[tid] += 1
            team_tids[p.get("team") or "unknown"].add(tid)
        if row["ball"] is not None:
            ball_covered += 1

    unique_tids = list(tid_first.keys())
    n_unique_tids = len(unique_tids)

    if unique_tids:
        lengths = [tid_last[t][1] - tid_first[t][1] for t in unique_tids]
        mean_track_len_s = sum(lengths) / len(lengths)
    else:
        mean_track_len_s = 0.0

    # churn: tids whose first appearance frame_idx > first processed frame_idx
    new_tids_after_start = [t for t in unique_tids if tid_first[t][0] > first_frame_idx] if first_frame_idx is not None else []
    if merged:
        t0 = merged[0]["t"] if merged[0]["t"] is not None else merged[0]["frame_idx"] / fps
        t1 = merged[-1]["t"] if merged[-1]["t"] is not None else merged[-1]["frame_idx"] / fps
        duration_min = max((t1 - t0) / 60.0, 1e-9)
    else:
        duration_min = 1e-9
    churn_per_min = len(new_tids_after_start) / duration_min

    ball_coverage_pct = (ball_covered / n_frames * 100.0) if n_frames else 0.0

    return {
        "n_frames": n_frames,
        "n_unique_tids": n_unique_tids,
        "mean_track_length_s": mean_track_len_s,
        "tid_churn_per_min": churn_per_min,
        "team_split_counts": {k: len(v) for k, v in team_tids.items()},
        "ball_coverage_pct": ball_coverage_pct,
    }


def estimate_stride(frame_idxs):
    if len(frame_idxs) < 2:
        return 1
    diffs = np.diff(sorted(frame_idxs))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1
    return int(np.median(diffs))


def run(video_path, out_dir):
    detections = load_jsonl(os.path.join(out_dir, "detections.jsonl"))
    shots = load_jsonl(os.path.join(out_dir, "shots.jsonl"))
    teams = load_json(os.path.join(out_dir, "teams.json"))
    ball_rows = load_jsonl(os.path.join(out_dir, "ball.jsonl"))
    frame_teams = {r["frame_idx"]: r.get("teams", {})
                   for r in load_jsonl(os.path.join(out_dir, "teams_frames.jsonl"))}

    merged = merge_perception(detections, shots, teams, ball_rows, frame_teams=frame_teams)

    perception_path = os.path.join(out_dir, "perception.jsonl")
    write_perception_jsonl(merged, perception_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idxs = [r["frame_idx"] for r in merged]
    stride = estimate_stride(frame_idxs)
    out_fps = max(src_fps / stride, 1.0)

    mp4_path = os.path.join(out_dir, "m2_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, out_fps, (w, h))

    ball_trail = []
    for row in merged:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_idx"])
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if row["ball"] is not None:
            ball_trail.append(tuple(row["ball"]))
        else:
            ball_trail.append((None, None))
        ball_trail = ball_trail[-BALL_TRAIL_LEN:]

        annotated = draw_frame(frame, row, ball_trail)
        writer.write(annotated)

    cap.release()
    writer.release()

    stats = compute_stats(merged, src_fps)
    stats["out_video"] = mp4_path
    stats["perception_jsonl"] = perception_path
    return stats


def print_stats(stats):
    print(f"n_frames={stats['n_frames']}")
    print(f"n_unique_tids={stats['n_unique_tids']}")
    print(f"mean_track_length_s={stats['mean_track_length_s']:.2f}")
    print(f"tid_churn_per_min={stats['tid_churn_per_min']:.2f}")
    print(f"team_split_counts={stats['team_split_counts']}")
    print(f"ball_coverage_pct={stats['ball_coverage_pct']:.1f}%")
    print(f"out_video={stats['out_video']}")
    print(f"perception_jsonl={stats['perception_jsonl']}")


def _self_check():
    import shutil
    import tempfile

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(repo_root, "data", "clips", "bundesliga_smoke.mp4")

    tmp_dir = tempfile.mkdtemp(prefix="m2_annotate_selfcheck_")
    try:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "cannot open smoke clip"
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        n = 10
        stride = 2
        detections = []
        shots = []
        ball_rows = []
        teams = {"1": "home", "2": "away"}
        for i in range(n):
            fidx = i * stride
            t = fidx / fps
            boxes = [
                {"tid": 1, "cls": "player", "conf": 0.9, "xyxy": [100 + i, 200, 140 + i, 280]},
                {"tid": 2, "cls": "player", "conf": 0.85, "xyxy": [400 - i, 300, 440 - i, 380]},
            ]
            detections.append({"frame_idx": fidx, "t": t, "boxes": boxes})
            shots.append({"frame_idx": fidx, "t": t, "shot": "main" if i < 8 else "other"})
            bx = 300 + i * 5
            by = 250 + (i % 3) * 4
            ball_rows.append({"frame_idx": fidx, "t": t, "ball": [bx, by], "src": "det"})

        with open(os.path.join(tmp_dir, "detections.jsonl"), "w") as f:
            for r in detections:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(tmp_dir, "shots.jsonl"), "w") as f:
            for r in shots:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(tmp_dir, "ball.jsonl"), "w") as f:
            for r in ball_rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(tmp_dir, "teams.json"), "w") as f:
            json.dump(teams, f)

        stats = run(video_path, tmp_dir)
        print_stats(stats)

        mp4_path = os.path.join(tmp_dir, "m2_annotated.mp4")
        assert os.path.exists(mp4_path), "annotated mp4 missing"
        size = os.path.getsize(mp4_path)
        assert size > 50_000, f"annotated mp4 too small: {size} bytes"
        assert os.path.exists(os.path.join(tmp_dir, "perception.jsonl"))

        with open(os.path.join(tmp_dir, "perception.jsonl")) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == n
        assert lines[0]["players"][0]["foot"] == [(100 + 140) / 2.0, 280]
        assert stats["n_unique_tids"] == 2

        print("SELF-CHECK OK:", mp4_path, size, "bytes")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check or not args.video:
        _self_check()
        return

    stats = run(args.video, args.out_dir)
    print_stats(stats)


if __name__ == "__main__":
    main()
