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
