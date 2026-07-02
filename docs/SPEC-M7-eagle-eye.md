# SPEC M7 — "Eagle Eye": Progressive Pitch Discovery + Line-Anchored Calibration

**Version:** 0.1 · **Status:** Draft for review (no code exists yet) · **Depends on:** M1–M6 (all merged)
**Origin:** two user observations from watching the telecast output:
1. *"Figure out the size of the playground vs the view size of the telecast, then a sliding-window /
   progressive-discovery algo similar to how a self-driving car works."*
2. *"Line mechanics change when nearing the goal posts — we need an algo like an OCR scanner that
   figures out the boundary of the field."*

Both are real, named techniques. (1) is a **SLAM-style persistent world model** built from partial
views: the camera is a moving sensor with a known frustum; the pitch is a static map; the players
are dynamic objects to be tracked *through* occlusion-by-framing. (2) is **line-feature
calibration** (points-*and-lines* homography refinement, cf. PnLCalib) driven by classical
line-scanning rather than a learned keypoint model. Together they produce the **eagle-eye view**: a
full-pitch, all-22-players radar reconstructed live from a single broadcast camera.

---

## 0. Table of contents

1. Goals / non-goals
2. Coordinate systems and existing contracts (what M7 builds on)
3. Component A — View frustum & coverage ("how much of the playground do we see?")
4. Component B — World model / progressive discovery (the eagle-eye state)
5. Component C — Line-anchored calibration refinement (the boundary scanner)
6. Rendering: the radar layer
7. Validation protocol (with numeric acceptance targets)
8. Performance budgets
9. Failure modes and mitigations
10. File-by-file integration plan
11. Sub-milestones and sequencing
12. Open questions

---

## 1. Goals / non-goals

**Goals**

- G1: Per frame, compute the **visible pitch region** (frustum polygon in pitch meters) and a
  **coverage fraction**, exposed as telemetry and consumable by every downstream component.
- G2: Maintain a **persistent world state of all 22 players + ball** across the whole match
  timeline, where unseen players carry explicit *predicted* positions with quantified, decaying
  confidence — never silently absent, never silently hallucinated as observed.
- G3: Render the world state as a **radar** (full-pitch 2D view, seen vs predicted visually
  distinct) — standalone, as a picture-in-picture layer on the broadcast overlay, and as
  primitives over the live WebSocket protocol.
- G4: **Refine the homography with line features** so calibration survives goal-area/tight views
  (few of the 32 keypoints visible) and gains an absolute **y-anchor** (touchlines/goal lines),
  complementing stripe-lock's x-anchor.
- G5: Everything causal (live-safe), gated (degrades to current behavior when signals are absent),
  and measured (before/after numbers for every claim).

**Non-goals (M7)**

- Jersey OCR / appearance re-ID (explicitly deferred by user decision; roles are the identity
  proxy).
- Feeding *predicted* (unseen) positions into event detectors (back-pass, press, …) — prediction
  is for the radar and for association priors only, v1. See §4.8 for the exact consumption policy.
- Full camera-pose SLAM (rotation/zoom estimation beyond H). H per frame remains the only camera
  model; we do not estimate focal length or camera position explicitly.
- Multi-ball / ball-in-flight 3D. Ball stays 2D ground-plane as today.

---

## 2. Coordinate systems and existing contracts

Everything below reuses (and must not break) these frozen conventions:

- **Pitch frame:** meters, x ∈ [0, 105], y ∈ [0, 68], origin bottom-left, `bto/core.py`.
- **H:** per-frame 3×3 homography, **pixels → meters** (row-major 9-float in `calib.jsonl`);
  `H_inv = np.linalg.inv(H)` maps meters → pixels. Emitted by `bto/vision/calib.py` (offline) and
  `_IncrementalCalib` in `bto/host/stream.py` (live) with `src ∈ {fit, ema, held}` + stripe-lock
  telemetry (`stripe_dx`, `stripe_strength`).
- **Frames:** `core.Frame(t, players[PlayerPos], ball, attacking)`; broadcast frames legitimately
  contain 0–22 players (visibility-limited).
- **Roles (M6):** `bto/patterns/roles.py::assign_roles(frames, team) -> (role_maps, slots,
  rotations)`; slots are attacking-normalized mean positions 'R1'..'R10'; assignments are causal
  and hysteresis-smoothed. GK excluded from slots (rearmost heuristic).
- **Re-tracker (M6):** `bto/vision/bridge.py::_retrack` — downstream tids are already causal
  Hungarian-consistent; median per-frame step 0.1–0.25 m.
- **Detection contract:** `Detection(type, players, geometry, confidence, t_start, t_end)`;
  moving geometry via `geometry['track']` / `geometry['*_path']` sampled ~5 Hz, renderer
  interpolates (M6 convention).
- **Live pipeline:** `StreamPipeline.process(frame_bgr, t)`; relational layer amortized
  (`relational_every=5`); patterns budget ≈ 13.5 ms/frame amortized on M1.

---

## 3. Component A — View frustum & coverage

### 3.1 What it is

The instantaneous answer to *"which part of the 105×68 playground is on screen right now?"* — a
polygon `V(t)` in pitch meters, plus scalars derived from it. This is the "view size of the
telecast vs size of the playground" measurement the user asked for, and it is the enabling
primitive for the world model (visibility tests), the radar (draw the camera wedge), coverage
telemetry, and smarter detector confidence.

### 3.2 Math

Naively, project the four frame corners through H. **This is wrong** for broadcast cameras: the
top of the frame is usually *above the horizon line of the ground plane* — the homography maps
those pixels to points behind the camera (the projected point crosses infinity; the w component of
`H · [u, v, 1]ᵀ` changes sign). Stripe-lock already met this problem (it anchors on lower-half
corners only). The correct construction:

1. Sample the frame border densely: the two vertical edges and the bottom edge at ~24 points, plus
   the top edge at ~16 points. For each pixel point `p = (u, v, 1)`:
   `q = H·p`, `w = q[2]`.
2. **Horizon guard:** keep only points with `w > w_min` where `w_min = ε·median(w of bottom-edge
   points)` (ε ≈ 0.05). Points with small or negative w are beyond the horizon.
3. For each *vertical* frame edge, if its top sample was culled, binary-search the edge (in pixel
   space, ≤8 iterations) for the highest v whose w passes the guard — this finds where the pitch
   horizon crosses the frame edge, giving a clean polygon top.
4. Project surviving points to meters, then **clip the polygon against the pitch rectangle**
   [−1, 106] × [−1, 69] (Sutherland–Hodgman, ~30 lines, no deps; 1 m margin tolerates calibration
   error at the boundary).
5. Result `V(t)`: a convex-ish polygon (ordered, ≤ ~40 vertices). Degenerate results (area < 50 m²
   or < 3 vertices) ⇒ `V(t) = None` (treat as "unknown visibility", not "nothing visible").

Scalars:
- `coverage(t) = area(V) / (105·68)` — typical wide main camera: 0.25–0.45; goal-area tight view:
  0.08–0.18. (Verify these bands empirically in M7.1 and bake them into shot-classifier hints.)
- `view_center(t) = centroid(V)`, `view_span_x`, `view_span_y` — the camera's pan/zoom proxy;
  d(view_center)/dt is the pan velocity that stripe-lock currently infers indirectly.

### 3.3 API (frozen for M7 implementation)

```python
# bto/vision/frustum.py  (new, pure numpy, no model)
def view_polygon(H: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray | None:
    """(N,2) polygon in pitch meters, or None when degenerate. Pure function of H."""

def coverage(poly: np.ndarray | None) -> float:  # 0.0 when None
def point_visible(poly: np.ndarray | None, x: float, y: float, margin: float = 0.0) -> bool:
    """Point-in-polygon with a shrink margin (negative margin = require strictly inside).
    None poly -> False (conservative: unknown visibility is not visibility)."""
```

Emitted telemetry: `view_poly` (rounded to 0.1 m, ≤ 40 pts) and `coverage` appended to every
`calib.jsonl` row and to `StreamPipeline.process()` extras. Cost budget: < 0.3 ms/frame.

### 3.4 Subtleties

- **Margin for player visibility tests (§4):** a player at the frustum edge is often half-cropped
  and missed by the detector even though his feet are inside V. World-model visibility uses
  `margin = -2.0` m (must be ≥2 m inside the polygon to count as "should have been seen").
- **Held H:** frustum from a held H is stale by up to 2 s of panning. `view_poly` carries the same
  `src` flag as its H; the world model treats `src='held'` frusta as **soft** (misses inside the
  polygon do not trigger lost-track logic, §4.5).
- Broadcast letterboxing/graphics bands: frame edges are sampled inside a 3% inset to avoid
  scoreboard/ticker rows contaminating the border sampling.

---

## 4. Component B — World model / progressive discovery

### 4.1 Concept

A fixed-size state of **22 tracked entities** (2 teams × {10 role slots + GK}) **+ ball**, updated
every processed frame, never created/destroyed — only transitioning between observation regimes:

```
SEEN      — a visible player is currently associated to this entity (age_unseen = 0)
COASTING  — recently seen; position advanced by decayed velocity          (age_unseen ≤ T_coast)
ANCHORED  — long unseen; position pulled toward the entity's formation-slot prior
LOST      — anchor itself unreliable (e.g., role template unstable); drawn greyed-out, wide σ
```

This is the "progressive discovery" the user described: each camera sweep *discovers* part of the
pitch; entities inside the frustum get measurements; entities outside it are dead-reckoned exactly
the way a self-driving stack coasts an occluded pedestrian — with a motion model, an uncertainty
that grows with time, and a prior that keeps them plausible.

### 4.2 State per entity

```python
@dataclass
class Entity:
    key: str                 # 'H:R1'..'H:R10', 'H:GK', 'A:R1'.., 'A:GK'
    team: str                # core.HOME / core.AWAY
    x: float; y: float       # current estimate, pitch meters
    vx: float; vy: float     # smoothed velocity (m/s)
    sigma: float             # isotropic 1-σ position uncertainty, meters
    age_unseen: float        # seconds since last association
    track_id: str | None     # current bridge tid when SEEN, else last known
    regime: str              # SEEN | COASTING | ANCHORED | LOST
```

`WorldFrame = (t, entities: list[Entity-snapshot], ball: BallEntity, view_poly, coverage)` — an
immutable per-frame snapshot for renderers/consumers.

### 4.3 Update cycle (per processed main frame, causal)

```
1. PREDICT  every entity:   x += vx·dt ; y += vy·dt
                            vx *= exp(-dt/τ_v) (velocity decay, τ_v = 2.5 s)
                            sigma = min(σ_max, sigma + σ_rate(regime)·dt)
2. ASSOCIATE observations (Frame.players, already re-tracked + team-labeled + role-mapped):
     a. tid continuity: same tid as an entity's current track_id -> direct match (cost 0)
     b. else Hungarian over (entity, observation) pairs within the same team:
        cost = ||pos_pred − obs|| / sigma  +  λ_role·[role_map(obs) != entity.role]
        gate: cost > 4.0 -> forbidden.  λ_role = 1.5 (role disagreement is evidence,
        not veto — role_maps themselves have hysteresis lag).
     c. GK entities associate only to gk-classified observations (bridge already folds GK team).
3. CORRECT matched entities: α-β filter update (α = 0.6 on position, β = 0.3 on velocity —
   full Kalman unnecessary: measurement noise is ~homoscedastic post-retrack; revisit only if
   the Metrica eval (§7.1) shows filter lag).  sigma -> σ_obs (1.0 m). regime = SEEN.
4. REGIME TRANSITIONS for unmatched entities:
     visible-but-missed (point_visible(V, x, y, margin=-2) and src != 'held'):
         miss_streak += 1;  3 consecutive genuine misses -> accelerate σ growth ×3
         (the entity is NOT where we think — e.g., unseen substitution or bad coast)
     outside frustum or held H: normal coasting.
     age_unseen > T_coast (= 3.0 s): regime = ANCHORED — blend toward slot prior:
         anchor = team_centroid_offset + slot_template[role] · team_spread   (see 4.4)
         x += k_a·dt·(anchor_x − x), k_a = 1/6 s⁻¹  (≈63% of the gap closed in 6 s)
     slot template unavailable/unstable (fresh segment, <8 visible for >20 s): regime = LOST.
5. BALL: reuse the existing Kalman ball (stream) / perception ball; world model only adds
   visibility context (ball outside frustum -> mark predicted, never anchored — balls have no
   formation slot).
6. EMIT WorldFrame snapshot.
```

### 4.4 The formation-slot prior (why M6 makes this possible)

`roles.assign_roles` yields, per team, slot template `S = {R1..R10 -> (x̂, ŷ)}` in
attacking-normalized coordinates plus per-frame assignments. The anchor for an unseen entity is
**not** the raw template position — teams translate and stretch as a block:

```
anchor(role) = centroid_seen + A · (S[role] − centroid_slots_of_seen)
```

where `centroid_seen` is the mean of currently-SEEN teammates, `centroid_slots_of_seen` the mean of
*their* slots, and `A = diag(spread_x, spread_y)` the ratio of observed spread to template spread
(clamped to [0.6, 1.6]). Interpretation: *"the left-back sits where the left-back slot sits,
relative to where the rest of his team currently is."* With ≥5 teammates seen this is a strong
prior; below 5, freeze `A = I` and use the last good centroid offset (and σ_rate doubles).

GK anchor: own goal line x ± 8 m box center, y = 34 blended toward last seen y — GKs barely move
in pitch terms; T_coast for GK = 10 s.

### 4.5 Substitutions, red cards, and identity capture

- A substitution appears as: one entity goes visible-but-missed repeatedly (player left the
  field), a *new* tid appears near the touchline. The role layer will reassign the slot to the
  new tid within its hysteresis; the entity **key survives** (it is the *role*, not the man) —
  exactly the right semantics for the radar, and honest about our no-OCR identity model.
- Red card: an entity that stays visible-but-missed with no replacement for > 90 s enters LOST and
  is drawn hollow-grey; no special-case logic in v1 (log it as telemetry `lost_entities`).

### 4.6 Segment boundaries (shot cuts, replays)

On non-main shots the world model **keeps predicting** (that is its whole point — the game
continues during a replay). On segment resume: sigma has grown; the association gate (cost 4.0)
naturally allows the larger search radius. A hard reset happens only when: (a) halftime detected
(t jumps or attacking flips), or (b) > 60 s without any main frame. Reset = re-seed entities from
the first 2 s of observations with the role bootstrap.

### 4.7 API (frozen)

```python
# bto/world.py (new)
class WorldModel:
    def __init__(self, tau_v=2.5, t_coast=3.0, k_anchor=1/6, sigma_obs=1.0,
                 sigma_max=12.0): ...
    def update(self, frame: core.Frame, view_poly: np.ndarray | None,
               role_maps: dict[str, dict[str, str]] | None,
               slots: dict[str, dict[str, tuple[float, float]]] | None,
               calib_src: str = 'fit') -> WorldFrame: ...
    def reset(self) -> None: ...

def replay_world(frames: list[core.Frame], calib_rows: list[dict]) -> list[WorldFrame]:
    """Offline convenience: run the model over a bridged segment list + calib.jsonl rows."""
```

Live wiring: `StreamPipeline` owns one `WorldModel`; `process()` gains `world` key in its result
(list of entity dicts) — primitives layer decides what to send (§6).

### 4.8 Consumption policy (anti-hallucination rule)

**Hard rule:** event detectors (press, back-pass, matchups, plays) consume **observed** frames
only, as today. World-model output feeds exactly three consumers:
1. the radar renderer (predicted entities drawn visually distinct — hollow, σ-scaled halo),
2. the *association prior* inside the world model itself and (M7.2+, optional) the bridge
   re-tracker gate,
3. **formation/block detectors only** (they are means over many players; blending ANCHORED
   entities with confidence-weighted contribution `w = exp(−age_unseen/4)` measurably stabilizes
   the formation string when 7–9 players are visible — validate in §7.1, adopt only if the
   Metrica masked eval shows label accuracy improves).

Rationale: a predicted left-back must never trigger a "1v1 isolation" against a real winger. Seen
= evidence; predicted = decoration + prior. This line is not crossable without a new spec.

---

## 5. Component C — Line-anchored calibration refinement ("the boundary scanner")

### 5.1 Problem statement (the user's observation, made precise)

Near the goal the main camera tightens: the 32-keypoint pose model finds 3–6 keypoints (penalty
box corners, penalty spot), frequently below the 4-point homography minimum or with degenerate
geometry (near-collinear). Current behavior: EMA/hold (up to 2 s), then null-H — overlay blanks.
Meanwhile the frame is *full* of high-contrast calibration evidence: the goal line, the 16.5 m box
lines, the 5.5 m box lines, the touchline, all as bright white-on-green **line segments**. Lines
constrain H differently from points: a matched line kills one degree of freedom but does not need
a distinguishable *point* — exactly what tight views offer.

### 5.2 Canonical line map

From `docs/roboflow_soccer_pitch_config.py` geometry, in pitch meters (both halves):

| id | segment (endpoints, m) | class |
|---|---|---|
| L_goal_0 / L_goal_105 | (0,13.84)–(0,54.16) …full goal lines (0,0)–(0,68), (105,0)–(105,68) | vertical |
| L_touch_0 / L_touch_68 | (0,0)–(105,0), (0,68)–(105,68) | horizontal |
| L_half | (52.5,0)–(52.5,68) | vertical |
| L_box16_front_{0,105} | x = 16.5 / 88.5, y ∈ [13.84, 54.16] | vertical |
| L_box16_side_{0,105}×{lo,hi} | y = 13.84 / 54.16, x ∈ [0,16.5] or [88.5,105] | horizontal |
| L_box5_front_{0,105} | x = 5.5 / 99.5, y ∈ [24.84, 43.16] | vertical |
| L_box5_side_{0,105}×{lo,hi} | y = 24.84 / 43.16, x ∈ [0,5.5] or [99.5,105] | horizontal |

(≈17 segments; the center circle and arcs are **excluded** from line matching v1 — conics need
different residuals; keypoints already cover the circle.)

### 5.3 Pixel-space extraction (the "scanner")

Classical CV, no model, target < 8 ms/frame at 720p:

1. **White-line mask:** within the green chroma mask (reuse stripe-lock's: G≥B+8, R≤G+12,
   dilated 5 px), top-hat: `white = (L > percentile95(L_local)) & brightness > 140`, where L is
   grayscale and the local percentile is a 31×31 box approximation. Morphological thin (1 erode).
   This is deliberately the OCR-scanner idea: sweep for bright structure *inside* the field
   surface only — stands, kits, and graphics are outside the green mask.
2. **Segment fitting:** `cv2.HoughLinesP(white, rho=1, theta=π/360, threshold=40,
   minLineLength=frame_w/12, maxLineGap=12)` → ≤ ~40 raw segments; merge collinear neighbors
   (angle < 2°, gap < 20 px); keep the longest 12.
3. **Sample points:** each merged segment contributes points every ~15 px (endpoints + interior),
   giving 100–400 line points per frame with segment provenance.

### 5.4 Association and refinement

Given current H (from keypoint fit, or held H in tight views — that is the payoff case):

1. Project each pixel segment's endpoints to meters. Compute its pitch-space angle; classify
   vertical (|dir_x| < sin 25°) vs horizontal. Match to the nearest canonical segment of the same
   class by perpendicular distance of the segment midpoint, **gate 1.8 m**, and require projected
   segment length > 4 m. Ambiguity guard: if the two nearest candidates are within 30% distance
   of each other (e.g., 5.5 m vs 16.5 m box lines under a bad H), drop the segment — wrong-line
   association is the failure mode that poisons everything (§9).
2. **Residual:** for pixel point p_i matched to canonical line (nᵢ, cᵢ) (unit normal, offset in
   meters): `r_i = nᵢᵀ · π(H·p_i) − cᵢ` where π is the perspective divide. Robust weight: Huber,
   δ = 0.5 m.
3. **Refine:** Gauss–Newton on the 8 dof of H (fix H₃₃ = 1), 3–5 iterations. The Jacobian
   ∂π(H·p)/∂H is the standard 2×9 homography Jacobian row pair contracted with nᵢ — ~25 lines of
   numpy. Add a **prior term** anchoring to the input H: `μ·||π(H·aⱼ) − π(H₀·aⱼ)||²` over the 4
   stripe-lock anchor points with μ ramping from 0 (≥120 line points, spanning both classes) to
   strong (< 40 points or single-class): line evidence bends H only as far as its coverage
   justifies. Single-class point sets (only horizontal lines visible) can only correct
   y-translation+shear — the prior pins the rest.
4. **Accept** iff: post-refine line RMS < 0.35 m AND keypoint reprojection rmse (when keypoints
   exist) does not worsen by > 0.05 m AND corner-gate displacement vs input H < 3 m. Else keep
   input H. Telemetry per frame: `line_pts`, `line_rms`, `line_dy` (the y-translation component —
   the number stripe-lock cannot produce).

### 5.5 Where it runs

- **Offline `calibrate()`:** after every keypoint fit (improves fits) and on every held frame
  (replaces pure hold — the big win: goal-area sequences currently ride a 2 s hold into null-H).
- **Live `_IncrementalCalib`:** held frames first (same as stripe-lock placement), fit frames if
  the ms budget allows (§8).
- Order per frame: keypoint fit → line refine → stripe-lock (x) — line refine feeds stripe-lock a
  better H; stripe-lock's deadband prevents double-correcting x.
- New `src` values: `'fit+lines'`, `'held+lines'` (calib.jsonl consumers treat any `*+lines` as
  its base class).

### 5.6 Explicitly out of scope for C (v1)

Center-circle conic fitting; goal *posts* (vertical structures need the full camera model, not
H); shadows-on-lines robustness beyond the Huber loss; learned line segmentation (if classical
extraction proves too brittle on evening matches, the fallback is the PnLCalib `SV_lines` HRNet
already downloaded in `models/` — swap §5.3 only, keep §5.4 unchanged).

---

## 6. Rendering: the radar layer

- **Standalone:** `bto/render/pitch.py::render_radar(world_frames, out_path)` — full-pitch
  animation; SEEN = solid team-color dot + tid label; COASTING = same dot, hollow, thin σ ring;
  ANCHORED = hollow + dashed σ ring; LOST = grey hollow. Camera frustum V(t) drawn as a
  translucent wedge (this single element *explains* the whole system to a viewer: you can watch
  discovery happen as the wedge sweeps).
- **Broadcast PiP:** new overlay layer `'radar'` in `bto/render/overlay.py` — 22% width mini-pitch,
  bottom-right (configurable corner), alpha 0.85, drawn every frame including replays (**the radar
  is exempt from the shot gate** — that is its point; label it "RADAR (est.)" during non-main
  shots since the whole state is then prediction).
- **Live:** `bto/host/primitives.py` gains radar prims: one `radar_frame` prim per reply
  `{kind:'radar', entities:[{k, team, x, y, regime, sigma}], view_poly, coverage}` in **pitch
  meters** (extension draws its own mini-pitch; sending meters keeps the protocol
  resolution-independent). `extension/draw.js`: mini-pitch canvas corner, ~60 lines; interpolates
  entity positions between replies exactly like other gids.

---

## 7. Validation protocol (acceptance gates — every number gets measured, none hand-waved)

### 7.1 World model — the masked-frustum replay (the crown jewel eval)

Metrica has **full-pitch ground truth**, so partial visibility can be *simulated* and predictions
scored against truth:

1. Take Metrica t=300–900 s @ 12.5 Hz (7500 frames, both teams).
2. Synthesize a broadcast frustum: trapezoid of width ~42 m at the near touchline / ~65 m at the
   far one (matched to measured cwc coverage ≈ 0.33), centered on the ball's smoothed x plus
   realistic pan lag (2nd-order follow with ωₙ = 0.8 rad/s) — replicating how real cameras chase
   play. Optionally: replay the *actual* cwc frustum trace scaled onto the Metrica timeline.
3. Hide all players outside the frustum; feed the visible remainder through the full stack
   (roles → world model).
4. Score predicted (hidden) entities against their true positions.

**Acceptance targets** (median absolute error over all hidden entity-frames, by age-unseen):

| age unseen | target MAE | stretch |
|---|---|---|
| ≤ 1 s (COASTING) | < 1.5 m | 1.0 m |
| 1–3 s | < 3.0 m | 2.0 m |
| 3–8 s (ANCHORED) | < 5.5 m | 4.0 m |
| > 8 s | < 7.0 m and σ honest (68% of errors < reported σ) | — |

Also: identity integrity — role-entity association survives ≥ 95% of frustum re-entries without a
swap (measured against Metrica's true player ids per slot); formation-label accuracy with §4.8(3)
blending ON vs OFF (adopt blending only if label accuracy improves ≥ 5 points on masked windows).
Baseline to beat: "teleport naive" (entity frozen at last seen position) and "pure slot prior" —
the model must beat both at every age bucket or it is not earning its complexity.

### 7.2 Line refinement

- **Synthetic:** render the canonical wireframe onto a synthetic green frame through a known
  H_true, perturb (x±0.8 m, y±0.8 m, slight rotation), refine → recover to < 0.15 m in both axes;
  single-class case recovers y only (assert x untouched by the prior).
- **Real, cwc:** the three FAIR pan frames (495/3771/4422 — **known y-offset cases stripe-lock
  could not touch**) plus every `held` run: wireframe eyeball verdicts must improve (FAIR→GOOD on
  ≥ 2 of 3), `line_dy` telemetry must correlate with the visually observed offset; rmse median
  must not regress; null-H frame count in goal-area sequences must drop (measure on the two
  longest held runs).
- **Bundesliga:** regression guard — everything unchanged within noise (it has no long holds).

### 7.3 Live

40-frame `StreamPipeline` self-check extended: world model + frustum present in extras, total
added cost within budget (§8); radar prims render in the demo page (manual check on the Mac, then
3060).

## 8. Performance budgets (added cost per processed main frame)

| component | M1 8GB budget | expected | 3060 budget |
|---|---|---|---|
| frustum (A) | 0.3 ms | ~0.1 ms | 0.3 ms |
| world model update (B), 23 entities | 1.5 ms | ~0.5 ms (α-β + one 10×10 Hungarian) | 1.5 ms |
| line extraction+refine (C) | 8 ms held-frames / skip on fit if over | 4–7 ms | 8 ms every frame |
| radar prims/render | 1 ms | ~0.4 ms | 1 ms |
| **total** | **≤ 11 ms** (inside current 13.5 ms relational amortization headroom) | | |

## 9. Failure modes and mitigations

| failure | symptom | mitigation |
|---|---|---|
| Wrong-line association (5.5 m vs 16.5 m box) | H snaps ~11 m in x, catastrophic | ambiguity guard (§5.4.1), corner gate < 3 m, accept-test vs keypoints |
| Frustum from held H during fast pan | world model marks visible players missed | `src='held'` ⇒ soft frustum (§3.4, §4.3.4) |
| Role template drift during masked spells | anchors pull entities to wrong slots | anchor uses centroid-relative form (§4.4); < 5 seen ⇒ freeze A, widen σ |
| Substitution mid-replay | entity teleports on resume | gate at 4.0σ allows recapture; miss-streak accelerates σ (§4.3.4) |
| Night matches / grey lines | line mask empty | `line_pts < 40` ⇒ component silently off (same philosophy as stripe gate) |
| Radar overconfidence (looks authoritative) | user trusts predicted dots | hollow + σ ring + "est." labeling is **required** rendering, not optional |

## 10. File-by-file integration plan

| file | change |
|---|---|
| `bto/vision/frustum.py` | **new** — §3 API |
| `bto/world.py` | **new** — §4 API |
| `bto/vision/lines.py` | **new** — §5.2–5.4 (extraction, map, association, GN refine) |
| `bto/vision/calib.py` | wire LineRefine into calibrate() (fit + held paths); `src='*+lines'`; telemetry columns |
| `bto/host/stream.py` | frustum + world model + line-refine on held; extras keys `world`, `view_poly`, `coverage`, `line_*` |
| `bto/render/pitch.py` | `render_radar` |
| `bto/render/overlay.py` | `'radar'` PiP layer (shot-gate exempt) |
| `bto/host/primitives.py` | radar prim kind |
| `extension/draw.js` | mini-pitch radar widget |
| `scripts/eval_world.py` | **new** — §7.1 masked-frustum harness + report |
| `scripts/eval_calib.py` | line telemetry columns; goal-area held-run analysis |
| `tests/` | frustum polygon cases (horizon crossing!), world-model regimes (synthetic), line refine synthetic recovery, association ambiguity guard |

## 11. Sub-milestones (sequenced so each ships something visible)

1. **M7.1 — Frustum + radar-of-the-seen** (A + render, no prediction): wedge + solid dots on the
   radar, coverage telemetry. Small, immediately demo-able, unblocks B.
2. **M7.2 — World model** (B + §7.1 eval): the eagle eye proper. Gate on the MAE table.
3. **M7.3 — Line-anchored calibration** (C + §7.2): goal-area robustness + y-anchor. Independent
   of B; can run parallel to M7.2 if two tracks are available.
4. **M7.4 — Live integration**: stream + primitives + extension radar; 3060 measurement pass.

## 12. Open questions (decide before or during build — flagged, not blocking spec approval)

1. Radar PiP default-on or user-toggled layer? (Lean: default-on in demo page, toggled in
   extension popup.)
2. Should the bridge re-tracker consume world-model predictions as association priors (§4.8.2)?
   Powerful but couples two subsystems — propose measuring M7.2 first, then A/B the coupling.
3. Ball prediction during replays: coast (current) vs freeze-with-banner? Coasting a ball 30 s is
   fiction; proposal: coast ≤ 2 s, then freeze + mark stale on radar.
4. Halftime/side-swap detection is currently heuristic (attacking flip) — is that reliable enough
   for the world-model hard reset, or does M7 need an explicit period detector?
5. Do we surface coverage/frustum to the shot classifier (a tight-view class between 'main' and
   'other')? Would let detectors keep running with reduced confidence instead of binary gating.
