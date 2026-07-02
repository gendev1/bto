# BTO — Broadcast Tactical Overlay

TV telestrator per [SPEC.md](SPEC.md): single-camera soccer broadcast → player/ball tracking →
pitch homography → tactical pattern detection → overlays, live via a Chrome extension.
All five SPEC milestones (M1–M5) are implemented; measured numbers in [docs/m3-report.md](docs/m3-report.md).

## Setup

```bash
uv sync                                   # Python 3.12 env
# model weights + test clips (gitignored): see docs/research-m2m3.md for download commands
uv run pytest -q                          # 19 tests
```

## Run it

```bash
# Live (M5): start the host, then open the demo page and press play
uv run python -m bto.host.server         # ws://127.0.0.1:8517
open http://127.0.0.1:8517/              # no-install demo (needs extension/demo/clip.mp4)
# Real site: chrome://extensions → Developer mode → Load unpacked → extension/
# (non-DRM video only; the badge shows "DRM protected" otherwise)

# Offline pipeline on any clip (M2–M4):
uv run python scripts/run_perception.py data/clips/<clip>.mp4 --stride 3   # detect/track/teams/ball
uv run python -m bto.vision.calib data/clips/<clip>.mp4 --shots out/<stem>/shots.jsonl --stride 3
uv run python scripts/run_m3.py <stem>   # → meters, tactical detections, 2D pitch GIF
uv run python scripts/run_m4.py <stem>   # → m4_overlay.mp4 (overlays on the broadcast)

# Pattern engine on recorded tracking data (M1, no CV needed):
uv run python scripts/demo_m1.py         # Metrica sample game → out/demo_m1.gif
```

## Layout

- `bto/core.py` — canonical coordinate stream (105×68 m); the pattern engine sees only this, never pixels
- `bto/patterns/` — rule-based detectors (§6): formation, matchups/ISO, triangles, back pass, pressing, offside (approx), runs, block
- `bto/io/` — Metrica + SkillCorner tracking-data loaders
- `bto/vision/` — detect+track (YOLOv8x/ByteTrack), shot classifier, team clustering, ball Kalman, calibration (32-keypoint → homography), px→meters bridge
- `bto/render/` — 2D pitch animation + broadcast overlay compositor (H⁻¹)
- `bto/host/` + `extension/` — near-live: FastAPI WebSocket host + MV3 Chrome extension

## Known limits (M1 MacBook, 8 GB)

- Live inference ~2.6 fps at imgsz 960 (interpolated draw stays smooth; e2e latency ~0.4 s). SPEC's 15 fps target assumes an RTX-3060-class GPU — path: TensorRT/ONNX export or a distilled yolov8s (docs/research-m2m3.md).
- At live sampling rates possession starves, so ball-dependent callouts (back pass, press) fire offline (M4) but rarely live; live layers today are formation + offside.
- Offside line is torso-center approximate by design and labeled as such everywhere.
