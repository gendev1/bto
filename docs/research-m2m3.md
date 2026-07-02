# M2/M3 model choices (researched + verified 2026-07-02)

Dev machine: Apple M1, 8 GB, MPS only. Rule: run model passes **sequentially** over a clip
(detect → track → calibrate → teams), writing intermediate JSON — never hold all models resident.

| Component | Primary | Fallback | M1 8GB verdict |
|---|---|---|---|
| Detection C2/C9 | Roboflow sports YOLOv8x `.pt` (classes: 0=ball, 1=goalkeeper, 2=player, 3=referee; imgsz=1280) | self-trained yolov8s on same CC-BY-4.0 dataset; HF `uisikdag/yolo-v8-football-players-detection`, `Adit-jain/soccana` | v8x@1280 ~0.5–1.5 FPS MPS → offline OK; v8s@640 15–25 FPS |
| Calibration C5 | Roboflow yolov8x-pose, 32 pitch keypoints @640 + homography fit (conf > 0.5), `SoccerPitchConfiguration` 105×68 vertex map | PnLCalib `SV_FT_WC14_{kp,lines}` (GPL-2.0, 2×265 MB, WC14-finetuned = FIFA grammar) — offline gold standard | pose@640 ~2–4 FPS; PnLCalib ~1 FPS → calibrate every N frames + smooth |
| Tracking C3 | ultralytics ByteTrack (`tracker="bytetrack.yaml"`, no weights) | BoT-SORT `with_reid: True, model: auto` if ID switches > 1/player/min | negligible |
| Teams C4 | HSV torso histogram (upper 40% of box, mask green H≈35–85, hue-sat hist) + KMeans(2), GK/ref = outliers | `google/siglip-base-patch16-224` mean-pooled + UMAP(3) + KMeans(2) (roboflow approach, ~375 MB, off hot path) | histogram ~free; SigLIP fine off hot path |

## Weight downloads (no API key; gdown)

```bash
uv tool run gdown -O models/football-player-detection.pt "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
uv tool run gdown -O models/football-ball-detection.pt   "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
uv tool run gdown -O models/football-pitch-detection.pt  "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
# PnLCalib (fallback): https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_FT_WC14_kp + SV_FT_WC14_lines
```

License: sports code MIT, weights AGPL-3.0-derived (fine for personal research), dataset CC BY 4.0.

## Test footage (non-DRM, yt-dlp-verified 720p)

- Smoke test: Roboflow Bundesliga wide-camera 30s clips, e.g. `gdown "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"` → `data/clips/`
- FIFA grammar (locked D3 target): Chelsea v Palmeiras CWC 2021 final `https://youtu.be/pEwwC3y0yJ4`;
  France v Croatia WC18 final `https://youtu.be/SvV6aUki6LU`; Belgium v Japan WC18 `https://youtu.be/GrkiZjoyugA`.
  Grab a section only: `yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" --download-sections "*00:20:00-00:23:00" <url>`
- SoccerNet (fine-tune data, deferred): NDA form at soccer-net.org → password → `pip install SoccerNet`.

Full research transcript: see session notes; primary sources github.com/roboflow/sports, github.com/mguti97/PnLCalib.
