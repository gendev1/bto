#!/usr/bin/env bash
# One-shot BTO setup on an NVIDIA PC via WSL2 (Ubuntu). Run INSIDE WSL2 from the repo root:
#   bash scripts/setup_cuda_wsl2.sh
# Prereqs (Windows side): recent NVIDIA driver (that's all — do NOT install a Linux driver in WSL).
# Chrome runs on Windows; WSL2 localhost forwarding makes ws://127.0.0.1:8517 reach this server.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 GPU visible in WSL2?"
nvidia-smi >/dev/null || { echo "nvidia-smi failed: update the Windows NVIDIA driver + 'wsl --update'"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== 2/5 Python env (uv)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
# CUDA extras: TensorRT + ONNX for engine export (pip wheels work in WSL2)
uv pip install tensorrt onnx onnxslim gdown

echo "== 3/5 model weights (models/ is gitignored — downloaded fresh)"
mkdir -p models data/clips
[ -f models/football-player-detection.pt ] || uv run gdown -O models/football-player-detection.pt "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
[ -f models/football-pitch-detection.pt ]  || uv run gdown -O models/football-pitch-detection.pt  "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
[ -f models/football-ball-detection.pt ]   || uv run gdown -O models/football-ball-detection.pt   "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
[ -f data/clips/bundesliga_smoke.mp4 ]     || uv run gdown -O data/clips/bundesliga_smoke.mp4     "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"

echo "== 4/5 TensorRT engines (FP16; built for THIS GPU — takes several minutes each)"
# 12GB budget: detect at full 1280 for max recall (M1 ran 960 as a speed compromise)
[ -f models/football-player-detection.engine ] || uv run yolo export model=models/football-player-detection.pt format=engine half=True imgsz=1280 device=0
[ -f models/football-pitch-detection.engine ]  || uv run yolo export model=models/football-pitch-detection.pt  format=engine half=True imgsz=640  device=0

echo "== 5/5 smoke test: 40 frames through the live pipeline on CUDA"
uv run python - <<'EOF'
import time, cv2
from bto.host.stream import StreamPipeline
p = StreamPipeline(device="cuda", imgsz=1280, calib_every=2,
                   player_model="models/football-player-detection.engine",
                   pitch_model="models/football-pitch-detection.engine")
cap = cv2.VideoCapture("data/clips/bundesliga_smoke.mp4")
t0 = time.time(); n = 0
for i in range(40):
    ok, fr = cap.read()
    if not ok: break
    p.process(fr, i / 25.0); n += 1
dt = time.time() - t0
print(f"processed {n} frames in {dt:.1f}s = {n/dt:.1f} proc fps (SPEC target >= 15)")
EOF

cat <<'EON'

Done. Run the live host (max profile for RTX 3060 12GB):

  uv run python -m bto.host.server --device cuda --imgsz 1280 --calib-every 2 \
      --player-model models/football-player-detection.engine \
      --pitch-model  models/football-pitch-detection.engine

Then on Windows: open http://127.0.0.1:8517/ in Chrome (demo page), or load the
extension/ folder unpacked via chrome://extensions for a real video page.
If localhost doesn't reach WSL2: add "networkingMode=mirrored" under [wsl2] in
%UserProfile%\.wslconfig and run 'wsl --shutdown' once (Win11), or use the WSL IP.
EON
