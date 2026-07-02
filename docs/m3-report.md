# M3 Report — Pitch Calibration + Feeding M1 with M2 Perception

Date: 2026-07-02. Pipeline: shots -> (shot-gated) detect/track -> teams -> ball -> merge (`perception.jsonl`) -> calib (`calib.jsonl`) -> bridge -> M1 pattern engine -> 2D pitch render.

## Per-clip results

| metric | bundesliga_smoke (30s, 1080p25, stride 2) | cwc2021_chelsea_palmeiras_20m (180.9s, 720p30, stride 3) |
|---|---|---|
| main frames processed (calib) | 375 (clip is all wide camera) | 1141 of 1809 records (668 `other` skipped) |
| valid-H fraction | **100%** (375/375) | **99.2%** (1132/1141; 9 null after hold expiry) |
| calib `src` breakdown | fit=2, ema=372, held=1 | fit=51, ema=890, held=200 (17.5% held) |
| rmse_m median / p90 / max | **0.229 / 0.406 / 0.625** | **0.348 / 0.598 / 0.807** |
| n_kp median | 9 | 9 |
| calib speed (MPS, yolov8x-pose@640) | 6.5 fps | 6.1 fps |
| temporal stability (proj. frame-center disp/frame) | median 0.092 m, max 1.21 m | median 0.111 m, max 49.9 m (segment re-seed jumps, see issues) |
| tracker: unique tids / mean track len | 36 / — (M2, unchanged) | **117 / 19.2 s** (was 527 / 3.6 s) |
| segment-aware churn (SPEC C3) | — | **23.3 new-tids/min** (was ~148/min before shot-gated tracking) |
| GPU detect frames | — | 1141 vs 1809 = **36.9% fewer** (shots gate detect) |
| ball coverage (bridge, main segments) | 92.8% | 73.9% (52.2% of all frames incl. replays) |
| bridge segments | 1 (29.9 s) | 4 (51.0 / 14.6 / 31.5 / 10.6 s = 107.7 s) |
| mean players/frame (bridge) | 20.2 | 14.6 |
| M3 detections (total) | 140 | 357 |

SPEC S7 target (<1.5 m RMS reprojection): met on both clips with large margin (p90 <= 0.6 m).

### Detection counts by type

- **bundesliga_smoke** (30 s): offside_line 54, block 30, formation 12, 1v1 11, isolation 7, back_pass 7, 1v2 7, triangle 4, press 3, 2v1 2, 2v2 2, 2v3 1.
  In-band vs the M1 Metrica sanity run (formations/blocks/offside dominate, a few triangles/presses per minute); back_pass is a little hot (7 in 30 s) — possession churn noise.
- **cwc** (107.7 s of main segments): offside_line 197, block 112, formation 48, **everything ball-dependent = 0** (no 1v1/NvN/triangle/press/back_pass). `possession()` latched only 3 spells in 108 s (vs 11 in 30 s on bundesliga): it requires the SAME tid nearest the ball within 2 m for 3 consecutive frames, which tid swaps + 720p ball jitter break, even though nearest-player-to-ball distance stats are identical across the clips (median ~1.3 m, 64% < 2 m).

## Eyeball verdicts (calib_viz wireframes, graded by hand)

- **bundesliga**: 4/4 sampled frames GOOD — halfway line, center circle, boxes and touchlines hug the broadcast markings; only sub-meter divergence near the bottom frame edge. GIF (`out/bundesliga_smoke/m3_pitch.gif`) shows two coherent teams, formation hulls, both offside lines, ball, matchup/press overlays.
- **cwc**: 8 frames sampled across the clip — 5 GOOD, 3 FAIR (frames 495 / 3771 / 4422: center circle drawn 1–3 m off during camera pans — EMA lag; lines/boxes still track). **No garbage H anywhere**: on tight/low-keypoint views the hold/ema fallbacks engage (200 held frames, 9 nulls) instead of emitting nonsense. GIF shows two coherent teams + ball on the 2D pitch.

## Fix applied during integration: team assignment collapse on cwc

The perception rerun produced a **107 home / 1 away** track split. Root cause (verified by eyeballing crops across track lifetimes): ByteTrack **identity swaps within main segments** make long tracks kit-impure — e.g. tid 1 (51 s) spends roughly half its life on a blue-kit player and half on a white-kit player. The track-MEAN hue/sat histogram then lands midway between kits, the cluster structure vanishes, and the single fixed-seed k-means init degenerated onto an outlier.

Fix (in `bto/vision/teams.py` + `scripts/annotate_m2.py` + `scripts/run_perception.py`):
1. k-means now does 10 kmeans++ restarts, best inertia wins (fixes degenerate init on any feature set);
2. clustering runs on individual SAMPLE histograms (94–96% sample purity vs a saturation proxy), tracks take the majority vote;
3. new optional per-frame output `teams_frames.jsonl` (every player box classified independently per frame); the perception merge prefers the per-frame label over the per-track one.

Result: per-frame team split home 104 / away 105 / ref 9; annotated video and 2D pitch show clean blue/white separation. Bundesliga path is unaffected (falls back to per-track labels when `teams_frames.jsonl` is absent).

## Top issues for M4

1. **Tracker identity swaps within main segments** (cwc): the single biggest quality limiter. Breaks per-track team labels (worked around with per-frame labels) and starves every possession-based detector. Candidates: kit-color gate in the association cost, or make `possession()` tid-agnostic (nearest player of either team with hysteresis).
2. **Ball tracking at 720p FIFA grammar**: 52% coverage overall / 74% on main segments, plus jitter -> only 3 possession spells in 108 s. Improve the small-object crop path + Kalman bridging; consider possession by team-proximity rather than same-tid persistence.
3. **Calib EMA lag on pans**: center circle drifts 1–3 m mid-pan (lines still fine). A faster EMA when keypoint count is high, or velocity-aware smoothing, would tighten it.
4. **Segment re-seed jumps**: held->fresh-fit transitions after shot cuts produce up to ~50 m projected-center jumps between consecutive calib entries. The M4 overlay must blank for a few frames after each cut (it already gets `src` to key off).
5. **back_pass over-firing** on bundesliga (7 in 30 s) — same possession-churn noise, low priority.

## Artifacts (kept)

- `out/<stem>/calib.jsonl`, `calib_report.json`, `calib_viz/*.jpg` (both clips)
- `out/<stem>/m3_detections.json`, `m3_pitch.gif` (both clips)
- `out/cwc2021_chelsea_palmeiras_20m/{teams.json,teams_frames.jsonl,perception.jsonl,m2_annotated.mp4}` regenerated with the team fix
