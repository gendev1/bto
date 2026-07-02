# Spec: Broadcast Tactical Overlay ("TV Telestrator")

**Version:** 0.1 · **Status:** Draft · **Owner:** you

## 1. Summary

A system that takes a live or recorded single-camera TV broadcast of a soccer match and renders real-time tactical overlays on top of the video: formation shape, 3v3 / 1v1 matchups, passing triangles, back passes, pressing structure, and an approximate offside line. No stadium multi-camera rig, no official telemetry — pixels only (single-camera **broadcast tracking**).

## 2. Goals / Non-Goals

**Goals**

- G1: Extract per-frame player positions `(track_id, team, x, y, t)` in pitch coordinates from broadcast video.
- G2: Detect tactical patterns from that coordinate stream (see §6).
- G3: Render overlays back onto the video frame at ≥ 15 FPS on a consumer GPU, end-to-end latency ≤ 2 s.
- G4: Survive real broadcast conditions: camera pans/zooms, replays, close-ups, graphics, crowd shots.

**Non-Goals (v1)**

- Broadcast-grade offside (needs limb-level 3D pose from calibrated multi-camera rigs — out of scope; ours is approximate, torso-center based).
- xG-grade ball physics (ball is tracked in v1 for possession/pass logic, not shot modeling).
- Player identity (names/numbers via jersey OCR) — v2.
- Fullscreen overlay, mobile, non-Chrome browsers — v1 targets Chrome, non-fullscreen video players only.

**Decisions (locked)**

- **D1:** Ball tracking is **in v1** (drives possession, back-pass, and pass-network detection).
- **D2:** Output surface is a **Chrome extension** that injects a transparent canvas positioned over the page's `<video>` element and draws overlays per frame.
- **D3:** Target broadcast style: **FIFA tournament coverage** (World Cup / Club World Cup wide main-camera grammar) — calibrate and fine-tune for this style first.

## 3. Architecture

```
video frames ──► Shot classifier ──► (main-camera frames only)
                      │
                      ▼
              Player detection (YOLO fine-tuned)
                      │
                      ▼
              Multi-object tracking (ByteTrack) ──► stable track IDs
                      │
                      ▼
              Team assignment (jersey-crop embeddings + k-means:
              team A / team B / GK / referee)
                      │
                      ▼
              Pitch calibration (keypoint model → homography H per frame)
                      │
                      ▼
              Coordinate stream: (track_id, team, x_pitch, y_pitch, t)
                      │                      ▲
                      ▼                      │ same interface —
              Pattern engine (rules, §6)     │ can be swapped for
                      │                      │ recorded tracking data
                      ▼                      │ (SkillCorner/Metrica)
              Overlay renderer (draws in pixel space via H⁻¹,
              composited on the frame)
```

**Key design rule:** the pattern engine consumes only the coordinate stream. It never sees pixels. This lets you develop and unit-test all tactical logic on recorded open tracking data before the vision pipeline is finished.

## 4. Components

| #   | Component            | Approach                                                                                                                                                                                                                           | Model needed?                                      |
| --- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| C1  | Shot classifier      | Lightweight CNN or histogram heuristic: main wide camera vs replay/close-up/graphic. Pause overlay on non-main shots.                                                                                                              | Small ML model (or heuristic first)                |
| C2  | Player detection     | YOLOv8/v11 fine-tuned on broadcast soccer frames (players, GK, referee, ball classes).                                                                                                                                             | Yes — fine-tune pretrained                         |
| C3  | Tracking             | ByteTrack / BoT-SORT over detections. Re-ID embedding model to survive occlusions and re-entry into frame.                                                                                                                         | Yes — pretrained re-ID, tune thresholds            |
| C4  | Team assignment      | Crop torso per track → visual embedding (e.g., SigLIP) → k-means into 2 teams + GK + ref. Cluster once per shot, cache per track.                                                                                                  | Pretrained embedding, classical clustering         |
| C5  | Pitch calibration    | Keypoint model detects pitch landmarks (line intersections, penalty box corners, circle) → homography to canonical 105×68 m field per frame, temporally smoothed.                                                                  | Yes — fine-tune pretrained (SoccerNet calibration) |
| C6  | Pattern engine       | Pure geometry/rules on coordinates (§6). No ML in v1.                                                                                                                                                                              | No                                                 |
| C7  | Overlay renderer     | Draw shapes in pitch space, project to pixel space with H⁻¹, alpha-composite on frame. OpenCV or WebGL canvas.                                                                                                                     | No                                                 |
| C8  | Ingest               | Chrome extension content script grabs frames from the page `<video>` via canvas `drawImage` (or `chrome.tabCapture` fallback), downscales, and streams JPEG/WebP frames over WebSocket to a local inference host.                  | No                                                 |
| C9  | Ball tracking        | Dedicated small-object path: high-res crop inference around last known ball position + Kalman filter; falls back to full-frame ball class detections. Feeds possession inference (nearest player within control radius over time). | Yes — same fine-tuned detector, ball class         |
| C10 | Local inference host | Native Python app (FastAPI + WebSocket) running C1–C6 on GPU; returns per-frame overlay geometry as JSON. Extension stays thin: capture + draw only.                                                                               | —                                                  |

### 4.1 Output surface: Chrome extension

- **Content script** locates the `<video>` element, injects an absolutely-positioned transparent `<canvas>` sized/synced to the video's bounding box (ResizeObserver + scroll/zoom handling). Non-fullscreen only in v1.
- **Frame path:** `drawImage(video)` → offscreen canvas → downscale to ~720p → WebSocket → local host → JSON geometry back → draw on overlay canvas. Overlay geometry is timestamped; renderer interpolates between results so drawing stays smooth even if inference runs at 10–15 Hz.
- **In-browser inference is a non-goal for v1** (WebGPU/ONNX-web can't hit the FPS budget for this whole stack yet); the extension is capture + render only.
- **DRM constraint (material):** streams using EME/Widevine (most official FIFA rights-holder streams) black out or taint canvas capture — `drawImage`/`getImageData` on protected video fails. Mitigations: `chrome.tabCapture` (works for software-decoded L3 streams, not hardware L1), or develop against non-DRM sources (FIFA+ free matches, YouTube full-match replays) and treat DRM capture as a known product risk, not an engineering task.

## 5. Data (analysis + model prep)

| Purpose                                     | Dataset                                                                                                                  | What it gives you                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Fine-tune detection + tracking (C2, C3)     | **SoccerNet Tracking / GSR**                                                                                             | Broadcast clips with player boxes, track IDs, team labels                                        |
| Fine-tune pitch calibration (C5)            | **SoccerNet Camera Calibration**                                                                                         | Broadcast frames annotated with pitch lines/keypoints                                            |
| Quick-start detection without annotating    | **Roboflow Universe soccer datasets + Roboflow `sports` repo**                                                           | Ready labeled players/ball datasets, reference pipeline code                                     |
| Develop pattern engine (C6) before CV works | **SkillCorner Open Data** (9 matches broadcast tracking), **Metrica Sports sample games** (full continuous tracking)     | Real `(player, team, x, y, t)` streams to write and test 3v3/1v1/back-pass/offside rules against |
| Ground truth to validate your patterns      | **StatsBomb Open Data** (events + 360 freeze frames)                                                                     | Labeled real events to sanity-check your detector output                                         |
| End-to-end demo/eval clips (FIFA style)     | FIFA+ free full-match replays, YouTube official full matches (non-DRM)                                                   | Realistic FIFA-grammar test input the extension can legally and technically capture              |
| Ball detection fine-tuning (C9)             | SoccerNet ball annotations + Roboflow ball datasets; add ~300 hand-labeled FIFA-style frames (ball is tiny at wide zoom) | Small-object detection quality on the target camera style                                        |

**Annotation budget if fine-tuning underperforms:** ~500–1,500 frames labeled in CVAT/Roboflow from your target broadcast style usually closes the gap.

## 6. Pattern engine — v1 detectors (rules on coordinates)

- **Formation shape:** cluster outfield players into lines by x-coordinate (defense/mid/attack); render convex hull + line links. Report e.g. 4-3-3 → 4-5-1 shifts via rolling window.
- **1v1 isolation:** attacker with ball-side proximity whose nearest defender is < 5 m and no other attacker/defender within r meters.
- **NvN (2v2, 3v3) local matchups:** bipartite proximity graph between teams in a local region; connected components of size N-N.
- **Passing triangles:** three same-team players with pairwise distances in [5, 25] m and open lanes (no opponent within d of each segment).
- **Back pass:** possession transfer (from C9 ball tracking: ball leaves player A's control radius, travels, enters player B's) where B's x is behind A's x relative to attacking direction.
- **Pressing intensity:** count defenders within 10 m of ball carrier + closing speed.
- **Approximate offside line:** second-rearmost defender's x; render as a line with an explicit "approximate" label.

Each detector outputs `{type, players[], geometry, confidence, t_start, t_end}` consumed by the renderer.

## 7. Performance targets (v1)

- ≥ 15 FPS end-to-end at 720p on a single consumer GPU (RTX 3060-class); 25+ FPS stretch.
- Homography reprojection error < 1.5 m RMS on visible pitch area.
- Track ID switches: < 1 per player per minute on main-camera segments.
- Overlay latency behind live frame ≤ 2 s (buffered "near-live" is acceptable v1).

## 8. Milestones

1. **M1 — Pattern engine on recorded tracking data.** Load Metrica/SkillCorner, implement §6 detectors, render animated 2D pitch view. _(No CV yet; validates the interesting logic.)_
2. **M2 — Perception pipeline on a broadcast clip.** Detection + tracking + team clustering on a 2–3 min clip.
3. **M3 — Calibration.** Homography per frame, coordinates in meters; feed M1 engine with M2 output.
4. **M4 — Overlay compositing.** Draw patterns back onto the video; shot classifier to blank on replays.
5. **M5 — Near-live.** Screen-capture ingest, latency/FPS optimization (TensorRT/ONNX, frame skipping + interpolation).

## 9. Risks

- **Homography drift on extreme zoom** → temporal smoothing, fall back to last-good H, hide overlay below confidence threshold.
- **Occlusion clumps (corners, walls)** → accept ID switches, mark pattern confidence low in crowded regions.
- **Offside expectations** → label it approximate everywhere; do not present as a call.
- **Legal:** rebroadcasting overlaid TV feeds publicly implicates broadcast rights; keep prototype personal-use, publish demos only on footage you have rights to.
- **DRM capture (highest product risk):** if the target stream is Widevine-protected, the extension cannot read frames. Develop on non-DRM FIFA-style sources; decide later whether the product story is "works on capturable streams" or requires a desktop capture app instead of an extension.
- **Ball at wide zoom:** the ball is often < 10 px in FIFA main-camera shots; expect misses during long balls — Kalman prediction bridges gaps, and possession logic tolerates short dropouts.

## 10. Open questions

- Extension ↔ host transport: plain WebSocket vs Chrome Native Messaging (WebSocket is simpler; Native Messaging avoids port conflicts).
- Overlay UX: always-on formation layer + event-triggered callouts (1v1, back pass), or user-toggled layers?
- Minimum GPU target for the local host (defines downscale + frame-skip strategy) — pick after M2 profiling.
