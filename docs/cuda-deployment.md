# CUDA deployment plan (Triton / TensorRT)

Status: **design doc, untested** — the dev machine is an M1 Mac (MPS, ~2.6 proc fps live).
This documents how to hit SPEC §7 (≥15 FPS end-to-end on an RTX-3060-class GPU) when the host
runs on NVIDIA hardware. Nothing here changes the extension, the WS protocol, or the pattern
engine — the coordinate-stream design (SPEC §3) isolates all of that from the inference backend.

## Which "Triton"?

| Name | What it is | Fit for us |
|---|---|---|
| **NVIDIA Triton Inference Server** | Model-serving daemon (gRPC/HTTP) running TensorRT/ONNX/PyTorch backends, batching, multi-model | ✅ Optional — worth it for multi-stream/scale-out |
| **OpenAI Triton** | Python DSL for writing custom GPU kernels | ❌ Not needed — our GPU work is standard convs; TensorRT already generates near-optimal kernels. Only relevant if we someday write a custom fused op (we have none; the pattern engine is <1 ms on CPU) |

## Recommended path (least code, most speedup): in-process TensorRT

Ultralytics loads TensorRT engines transparently — the only changes are the model path and device.

```bash
# On the CUDA box (engine files are GPU+TensorRT-version specific — always build on the target):
pip install tensorrt  # or use NGC PyTorch container
yolo export model=models/football-player-detection.pt format=engine half=True imgsz=960 device=0
yolo export model=models/football-pitch-detection.pt  format=engine half=True imgsz=640 device=0
```

Then run the existing host with `--device cuda` and point `StreamPipeline` at the `.engine`
files (the `player_model` / `pitch_model` ctor params already exist). That's the whole migration.

Expected throughput (estimates from published YOLOv8 TensorRT benchmarks — verify on target):

| Stage | M1 MPS today | RTX 3060 TRT FP16 (est.) |
|---|---|---|
| detect v8x @ 960 | ~400 ms | ~25–40 ms |
| pitch pose v8x @ 640 (every 4th frame) | ~245 ms/fit | ~10–15 ms/fit |
| everything else (shot, teams, bridge, patterns) | ~8 ms | ~8 ms (CPU) |
| **end-to-end proc fps** | **~2.6** | **~20–30 → SPEC 15 FPS target met** |

Knobs if the target is missed: `half=True` (FP16, ~2× vs FP32), INT8 calibration (~1.5× more,
needs a calibration image set — use frames from `data/clips/`), drop detect to imgsz 640 (~2×,
loses far-touchline recall), or swap to a distilled yolov8s (docs/research-m2m3.md) for ~5–8×.

## Scale-out path: NVIDIA Triton Inference Server

Worth adding only when serving **multiple concurrent streams** (the in-process path saturates one
GPU with one stream comfortably). Layout:

```
model_repository/
├── player_detect/
│   ├── 1/model.plan          # the TensorRT engine from the export above
│   └── config.pbtxt          # platform: "tensorrt_plan", max_batch_size: 4, dynamic_batching {}
└── pitch_pose/
    ├── 1/model.plan
    └── config.pbtxt
```

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 \
  -v $PWD/model_repository:/models nvcr.io/nvidia/tritonserver:<yy.mm>-py3 \
  tritonserver --model-repository=/models
```

Host-side change: `StreamPipeline` grows a thin backend seam — instead of `model(frame)`, call
Triton via `tritonclient.grpc` (letterbox preprocessing and NMS/keypoint postprocessing move
client-side; ultralytics' `nms=True` export option can bake NMS into the engine — verify against
the ultralytics version on the target). Keep ByteTrack, teams, calib EMA, bridge, and patterns
exactly where they are (CPU, per-connection state — they don't belong in Triton).
`dynamic_batching` lets several extension clients share one GPU; one `StreamPipeline` instance
per WS connection, all sharing the Triton backend.

## Video decode

One 720p stream: CPU decode (current cv2 path) is fine and not the bottleneck. Many streams:
NVDEC via PyAV/`torchcodec` or DALI — only revisit if host CPU becomes the limiter.

## What we can do on the M1 today (prep, no CUDA needed)

1. `yolo export format=onnx` both models — ONNX is portable; engines get built from it (or from
   `.pt`) on the target box. Check the exports into `models/` alongside the `.pt` files.
2. Keep `--device` plumbed everywhere (already done: detect/calib/stream all take `device`).
3. The live-rate detector gates (precision pass) were tuned at ~2.6 fps sampling; at 20+ fps the
   possession/persistence windows tighten naturally — re-check `min_frames`-style params expressed
   in *frames* vs *seconds* when the frame rate jumps (most are seconds-based already; audit the
   few frame-count ones: `possession(min_frames=3, max_gap=12)`).

## Checklist for the first CUDA deploy

- [ ] Build engines on target (`half=True`), smoke `yolo predict` on a clip frame
- [ ] `uv run python -m bto.host.server --device cuda` with `.engine` paths
- [ ] Re-run the M5 live-pacing test (`docs/m3-report.md` M5 section has the method): expect
      proc fps ≥ 15, e2e latency ≪ 2 s
- [ ] Re-run `scripts/eval_calib.py` — engine FP16 keypoints can shift slightly; rmse target < 1.5 m
- [ ] Revisit frame-count-based detector params per note 3 above
