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

---

# M4 — Overlay compositing + possession/calib fixes

Date: 2026-07-02. New module: `bto/render/overlay.py` + `scripts/run_m4.py` (SPEC C7: all geometry in pitch meters, projected to px via inv(H) per frame, alpha-composited on the original video; shot classifier blanks replays). Acceptance artifact: `out/<stem>/m4_overlay.mp4` (both clips).

## What changed

1. **Possession robustness** (`bto/patterns/possession.py`, M3 top issue #1/#2): a nearest-in-radius candidate now CONTINUES a spell if it is the same tid OR the same team within `handoff_dist=2.0` m of the holder's last known position (original spell tid kept, position tracked per frame); an opponent or a >2 m teammate still splits. Kills the "same tid for 3 consecutive frames" starvation caused by ByteTrack id swaps. Tests 17 -> 19 (tid-swap does not split; 6 m teammate pass splits). `tests/__init__.py` added: ultralytics 8.4.84 ships a top-level `tests` package that shadowed the local namespace tests dir and broke pytest collection.
2. **Calib pan lag + reseed jumps** (`bto/vision/calib.py`, M3 top issues #3/#4): (a) adaptive EMA alpha — ramps 0.5 -> 0.9 when the raw-fit anchor residual is > 0.5 m in a consistent direction for 2 consecutive frames (a pan, not noise); (b) reseed interpolation — a fresh fit landing > 5 m from the last held anchors only moves 40% of the way on the reseed frame, the adaptive alpha pulls in the rest over the next 1–2 frames.
3. **Run-path teleport gate** (`bto/patterns/runs.py`, found during M4 eyeball): with possession spells restored on cwc, overlap/underlap fired 28 times but EVERY path had physically impossible raw steps (18.7–56.7 m in ~0.2 s = tid teleports), drawn as full-pitch zigzag storms in the overlay. Added `_MAX_SPEED=12` m/s raw-step gate in `_check_window`; all 28 rejected (honest 0 — real overlaps are rare in 108 s of churny tracking). Bundesliga unaffected (had 0).
4. **Guard fix** (`bto/patterns/matchups.py` `detect_isolations`): unguarded `next()` raised StopIteration when a spell's original tid is absent from a covered frame (tid swap mid-spell); now `next((...), None)` + skip, matching every other Spell consumer.

## cwc numbers (regenerated calib, stride 3, shots-gated, 1141 main frames @ 6.4 fps MPS)

| metric | old (M3) | new (M4) |
|---|---|---|
| valid-H fraction | 99.2% (1132/1141) | 99.2% (1132/1141), src fit=60 ema=870 held=202 null=9 |
| rmse_m fit median / p90 | 0.348 / 0.598 | 0.465 / 0.647 (100% under the 1.5 m SPEC target) |
| temporal stability median / mean / max | 0.111 / 0.461 / **49.92 m** | 0.108 / 0.367 / **21.35 m** (worst reseed jump −57%; all top-10 deltas sit at the frame-5049 reseed, now interpolated) |
| possession spells | 3 in 107.7 s | **33** (18.4/min); bundesliga 11 -> 11 |
| detections total | 357 | **502** |
| ball-dependent | **0** | back_pass 18, isolation 37, press 89 (+ offside 198, block 112, formation 48) |
| tracker churn | 23.3 new-tids/min | unchanged (tracker untouched) |

Bundesliga regression: 140 -> 136 detections (back_pass 7->6, triangle 4->3, isolation 7->5 — nearby-teammate spells that used to split now merge); all 12 types still fire. pytest 19 passed.

Bands vs expectation: back_pass 18/107.7 s (~10/min) is still hot (M3 issue #5, possession-churn noise now on more spells) and triangle is still 0 (clear-lane + 1 s same-trio starved by teammate tid churn) — both are tracker-identity problems, not detector bugs; deferred to M5.

Calib wireframe eyeball (8 frames): 5 GOOD / 3 FAIR — the same three mid-pan frames as M3 (495/3771/4422, center circle 1–3 m off, lines/boxes still track). H at the sampled frames moved <= 0.07 m vs the old calib, so verdicts are unchanged: the reseed fix is large and real, the pan-lag fix is modest (median center-disp −17–23% in the pan windows).

## Overlay eyeball verdicts (m4_overlay.mp4, 8 sampled frames per clip)

- **bundesliga** (375 frames @ 12.5 fps, drawable 375/375, 38.9 MB): 8/8 GOOD. f0 fade-in from banner-only; f106 hulls hug both blocks, 2v1/1v2 boxes + dashed pair links land on real players, PRESS ring on the carrier; f214 BACK PASS arrow tip touches the receiving white player, ISO spotlights hug the duel; f428/f534/f642 triangles/1v2/ISO on real players, offside dashed lines perspective-correct on both halves; f748 end-of-clip fade-out. Note: the black box top-right is baked into the source (scoreboard blackout), and formation label chips occasionally show data-side noise (e.g. "AWAY 8-3").
- **cwc** (1809 frames @ 10 fps, drawable 1132/1809 = the valid-H main frames, 101 MB; final draw counts formation=2162, offside=2115, events=360 [back_pass 171, press 151, isolation 38]): first render exposed the overlap/underlap zigzag storm (fix #3 above); after the gate, re-rendered clean (re-eyeballed the same frames). f300 BACK PASS arrow + hull GOOD; f900/f1500/f3900 hulls + both offside lines GOOD; f1656 (last main frame before the replay wipe, mislabeled main by the shot classifier) draws banner-only — the drawable-run fade already suppressed overlays, no flicker at the cut; f1662/f5100 (shot=other) show "[overlay off]" banner and NOTHING drawn on the replay wipe/close-ups; f2718 (first main after a replay) correctly blank-then-fade-in. Formation chip shows data-side label noise ("HOME 15-1") — renderer is faithful to m3_detections.

## Top issues for M5 (near-live)

1. **Tracker identity swaps** remain the single root cause: triangle starvation, back_pass over-firing, and every overlap/underlap being a teleport all trace to tid churn inside main segments. Kit-color gate in the ByteTrack association cost is the highest-leverage fix.
2. **Throughput**: calib 6.4 fps + detect ~6 fps on MPS is ~2.5x too slow for near-live at stride 3. Calib only needs a FIT every N frames — interpolate H (anchor-space lerp) between fits; batch the detector; consider 480p detect with 720p ball crop.
3. **Interpolation/latency**: the pipeline is batch (jsonl interchange). Near-live needs a streaming loop with bounded lag — the EMA/hold logic already works causally; the bridge and pattern engine need incremental variants (possession/formation are already windowed).
4. **Ball tracking**: 74% coverage on main segments, jitter breaks spells — Kalman bridging + crop-path upgrade.
5. **Pan lag residual**: center circle still 1–3 m off mid-pan (FAIR frames); velocity-aware prediction (extrapolate anchor motion) instead of pure EMA.

---

# M5 — Near-live: local inference host + Chrome extension (SPEC C8/C10, S4.1, S7)

Date: 2026-07-02. New package `bto/host/` (three sibling-built modules integrated here) + `extension/` (MV3) + `extension/demo/` (no-install demo page). Acceptance artifact: `out/cwc2021_chelsea_palmeiras_20m/m5_preview.mp4` (what the extension user sees at reply keyframes, pre-interpolation).

## Architecture

- **`bto/host/stream.py` — `StreamPipeline`**: incremental port of the M2–M4 batch pipeline. Per frame: shot gate (CPU, ~3 ms; `other` frames return immediately, no GPU) -> detect+track (player model, imgsz 960, ball = class 0 + Kalman) -> calibration (full keypoint fit every `calib_every=4` main frames, EMA/gate/hold/reseed state machine ported from `calib.calibrate()`, held H in between) -> live team k-means (80-crop warmup) -> causal px->Frame bridge -> `bto.patterns.run_all` over a rolling 15 s buffer every processed main frame. Shot cuts arm a full segment reset (fresh tracker ids, calib, ball KF, buffer) — same semantics as the offline shot-gated M2 path.
- **`bto/host/primitives.py`**: `Detection` -> frozen-protocol PRIM dicts (polygon/polyline/circle/arrow/label/chip in *sent-frame pixel space*), reusing `overlay.py`'s `project()`/`sample_polyline_m()`/colors; `prims_render_debug()` is the cv2 renderer used for the preview video.
- **`bto/host/server.py`**: FastAPI + WS on `ws://127.0.0.1:8517/stream` (frozen protocol: 12-byte `<Id` header + JPEG in, JSON geometry out), `/health`, static mount of `extension/demo/` at `/`. Drops stale frames if a client doesn't throttle; one active stream per process; models lazy-load on first stream connect.
- **`extension/`**: MV3 content script finds the largest `<video>`, streams ~960px JPEG q=0.7 frames self-clocked (next send only after previous reply), draws replies on an overlaid canvas with per-gid interpolation/extrapolation between the last two replies (client stays smooth at 60 Hz rAF while the host replies at ~2.6 Hz), 2 s staleness blank, DRM detection, fullscreen hide. `extension/demo/index.html` loads the same three JS files against `/clip.mp4` with zero install.

## Integration fixes (found by streaming 90 s at real-time pacing, t=35–125 s of cwc, incl. the 55.3–90.5 s replay)

1. **Overlay strobed at 1/3 cadence, hulls at alpha 0.024** (`primitives.py`): `detections_to_prims` re-filtered strictly on `t_start <= t <= t_end`, discarding the pipeline's deliberately grace-extended cached detections, and `_fade_alpha` faded against raw `t_end` — a *live* detection always has `t ~= t_end`, so everything sat at the 0.2 alpha floor. Fixed: activity window and fade now honor `DET_GRACE_S=0.75` (matches `StreamPipeline.det_grace_s`).
2. **Stacked "generations" of always-on layers** (`primitives.py`): live cache + grace kept old and new windows of the same layer active at once -> up to 4 offside lines + 4 hulls per frame. Fixed: formation keeps only dets tied for max `t_end` (cap 2), offside deduped per defending team.
3. **Stale geometry between pattern runs** (`stream.py`): `patterns_every` 3 -> 1. `run_all` on a full 15 s buffer costs ~0.4 ms/frame — running it every processed main frame keeps geometry fresh instead of up to ~1.2 s stale.
4. **Periodic null-H flashes** (`stream.py`): fit cadence 5 frames x ~0.4 s live gap ≈ `HOLD_MAX_S` (2.0 s), so a single failed keypoint fit landed exactly at hold expiry -> one-frame overlay blackout every ~2 s on hard stretches. Fixed: `calib_every` 5 -> 4 and a failed fit retries the NEXT frame instead of waiting another cycle. Nulls on main frames: 9/75 -> 3/55 (remaining ones are the cold-start frame and a 1.2 s main sliver at a shot cut; all recover on the next frame).
5. **`GET /` was 404**: demo page was `player.html` but Starlette's `StaticFiles(html=True)` serves `index.html`; added `extension/demo/index.html` (identical copy). Also cut a 60 s demo clip to `extension/demo/clip.mp4` (gitignored) since the page's `<video src="/clip.mp4">` expects it.

`uv run pytest -q`: 19 passed throughout.

## Measured (90 s real-time stream, M1 8GB MPS, imgsz 960, self-clocked client, send-latest-drop-stale)

| metric | measured | S7-adjusted target (M1 8GB) | verdict |
|---|---|---|---|
| host proc fps on main-camera segments | **2.3–2.8 (median run 2.6)** | >= 3 | **just under** — detect is 80% of budget |
| `proc_ms` main frames (excl. cold start) | median **362**, p90 513, max 1243 | — | detect ~300–500, calib ~60–100 amortized (fit ~245 every 4th), shot 4, teams 2, bridge+patterns <1 |
| `proc_ms` `other` frames | median **3** (replies at full video rate) | — | shot gate pays for itself |
| end-to-end latency (send -> reply) | median **369 ms**, p90 541 ms | <= 2 s | **PASS** (only the cold-start frame violates: 2.7–8.1 s lazy model load + MPS warmup) |
| overlay staleness (video playhead - reply t) | median **393 ms**, p90 580 ms | <= 2 s | **PASS**; client interpolation covers the 2.6 Hz keyframe gap |
| replay behavior | 1777 `shot=other` replies, **0** with geometry; tracker fully reset on resume (fresh tids, n_tracks 7->18 within 3 frames at t=117) | blank on replays | **PASS** |
| frames dropped host-side | 0 (client throttles per protocol) | — | drop path exists, exercised in server self-check |
| 15 FPS full-inference (SPEC S7 original) | not attempted | RTX-3060 target | **not reachable on M1 8GB with v8x models — by design** |

Honest gap: **no event callouts (press/back-pass/isolation) fired in the live window** — at ~2.6 fps sampling the possession spells that gate every ball-dependent detector rarely survive, and class-0 ball recall at imgsz 960 is thin. Live overlay today = formation hulls + chips + offside lines (+ blocks computed but not rendered, same as M4). The offline 20-min run still fires all types; this is a live-cadence starvation, not a detector regression.

Preview eyeball (6 frames from the actual WS replies): hulls/lines land on the pitch with correct perspective on the FIFA+ letterboxed source, exactly 2 offside lines + 2 hulls max after the dedupe fix, replay frames blank, post-replay resume re-seeds calib correctly on a goal-area camera. Known cosmetic noise carried over from M4: formation chip labels (e.g. "7-1") reflect tracked-subset counts.

## What remains for the real target (RTX-3060, SPEC S7 15 FPS)

- **Export both models to ONNX -> TensorRT** (`yolo export format=engine half=True`): v8x player detect at 960 runs ~8–12 ms on a 3060 vs ~300 ms MPS here; pitch-pose at 640 ~4 ms. That alone clears 15 fps with the current single-frame loop.
- If headroom is still short: v8m/v8l player weights (fine-tune from the same Roboflow dataset), detect at 720–800, calib fit every 8–10 frames (hold is cheap and HOLD_MAX_S allows 2 s), batch=2 pipelining of decode/infer.
- Raise proc fps first, then re-check event detectors: at >= 8 fps sampling possession spells survive and press/back-pass/iso should fire live (they already do offline at 10 fps stride-3).
- Ball: re-enable the dedicated ball model (fits in >8 GB VRAM budgets) or the high-res crop path from C9 for real possession quality.

## Try it live

1. `uv run python -m bto.host.server` (from the repo root; add `--imgsz 960 --device auto` to taste). First stream connect lazy-loads models (~3–8 s).
2. **No-install demo**: open `http://127.0.0.1:8517/` — the bundled 60 s cwc clip plays with the live overlay (badge: connecting -> live; replays show "overlay off").
3. **Real site**: `chrome://extensions` -> enable *Developer mode* -> *Load unpacked* -> select the `extension/` directory. Open any page with a non-DRM `<video>` (FIFA+ free replays, YouTube full matches); the content script finds the largest video and overlays automatically. DRM/EME streams (Widevine) black out capture — the badge reads "DRM protected" (known product risk per SPEC S4.1).
