# SPEC M8 — Fluid Presentation: Narrate at Human Speed, Detect at Machine Speed

**Version:** 0.1 · **Status:** Draft for review (no code) · **Depends on:** M1–M6 (merged); M7 (spec'd, independent — M8 touches only the presentation path and can build before/parallel to M7)
**Origin:** user observation: *"a game being played is much faster than the samples … when I overlay
I see positions come and then instantly go away, I can't make sense of what is really going on —
but when I see the gif and samples it feels like the thing is working absolutely great."*

## 1. Diagnosis (why the samples lie and live confuses)

The detectors are correct at both speeds. What differs is the **viewer's time budget**:

- A detection like `back_pass` spans its *physical* duration: ~0.8–1.5 s. In a sample GIF the
  viewer replays it, pauses it, and was told what to look for. Live, an unexpected label that
  appears for <1.5 s while play continues is below the threshold of comprehension — by the time
  the eye saccades to the chip, it is gone ("positions come and then instantly go away").
- Live geometry updates arrive at host cadence (2.6 Hz on the M1; 15–25 Hz on the 3060). The
  extension lerps between keyframes, but a detection *born* in one keyframe and *dead* by the next
  literally flashes for one interpolation interval.
- Multiple simultaneous callouts split attention. At real speed a human parses roughly **one
  labeled fact per ~3 seconds**; we currently allow 3 concurrent events + 2 always-on layers.
- Broadcast telestration norms (the calibration point for "feels professional"): a highlighted
  fact stays on screen 2–4 s minimum, enters/exits with animation (never pops), moves with eased
  motion (never jumps), and frequently refers to a moment *just past* — the audience accepts and
  expects ~1–3 s of narrative lag. Our pipeline already runs 0.4–2 s behind live (SPEC §7 budget),
  so we have the latency budget to narrate the recent past honestly.

**Design thesis:** split the system into a fast **detection clock** and a slow **narration
clock**. Detectors keep firing at machine speed with exact timestamps; a new *presenter* decides
what the human sees, when, for how long, and with what motion. Nothing in bto/patterns changes.

## 2. Principles (the contract every renderer must obey)

P1. **States never blink.** Continuous layers (formation hull, offside line, marking edges, radar)
    change only by smooth motion or slow cross-fade. A state that would flicker must instead
    degrade to "hidden until stable for ≥ 1 s" (hysteresis on *visibility*, not just on value).
P2. **Events are stories, not states.** A discrete event (back pass, overlap, trap, give-and-go)
    is presented as a fixed-shape lifecycle — enter → hold → linger → exit — whose duration is set
    by *presentation* needs (≥ 2.5 s), independent of the physical event duration.
P3. **Attention budget: one story at a time.** At most **1 active event callout** on screen
    (+ always-on state layers). Concurrent detections queue; stale ones drop.
P4. **Nothing teleports.** Every drawn element moves under a critically-damped spring toward its
    target; hard repositioning is allowed only across shot cuts (where the whole overlay resets).
P5. **The recent past is legal material.** The presenter may begin telling a story up to
    `T_stale = 4 s` after the event's `t_end`; it draws the *trail* of what happened (the path
    geometry we already record) rather than pretending it is happening now.
P6. **Same presenter everywhere.** Offline renders (m4 videos) and the live extension must share
    the identical choreography semantics, or samples will keep looking different from live.

## 3. Component A — The Presenter (event choreography state machine)

New module consumed by both `bto/render/overlay.py` and (mirrored in JS) `extension/draw.js`.

### 3.1 Event lifecycle

```
QUEUED   arrival; scored, waiting for the stage to be free
ENTER    0.35 s   chip + geometry fade/scale in (ease-out cubic)
HOLD     max(2.5 s, physical duration)   fully visible; geometry tracks live players (P4 springs)
LINGER   1.2 s    event is physically over: freeze label, draw the completed trail, alpha 1→0.35
EXIT     0.4 s    fade to zero; stage freed
KILLED   (any time) shot cut or superseding higher-priority story: fast 0.2 s exit
```

Minimum on-screen life = 0.35 + 2.5 + 1.2 + 0.4 ≈ **4.5 s** per story — the number that fixes
"come and instantly go away".

### 3.2 Queue and scoring

- Score = `confidence × W_type × freshness`, `freshness = exp(−(now − t_end)/2 s)`.
  `W_type`: pressing_trap 1.3, give_and_go 1.2, overlap_run/third_man 1.1, back_pass 1.0,
  isolation 0.9, NvN 0.7, rotation 0.6 (tunable table, one place).
- Stage free → pop best-scoring queued story with `now − t_end < T_stale`; others age out.
- A story already in HOLD is superseded only by a score ≥ 1.5× its own (rare; fast-exit then).
- Global pacing: ≥ 1.5 s gap of "no event on stage" between stories (breathing room), and a
  per-type echo suppression: same type + same players within 10 s → merged into the first story.

### 3.3 API sketch

```python
# bto/render/presenter.py (new; pure logic, no drawing — testable headless)
class Presenter:
    def __init__(self, max_stage=1, t_stale=4.0, min_hold=2.5, pace_gap=1.5): ...
    def offer(self, det: core.Detection, now: float) -> None      # detectors -> queue
    def tick(self, now: float) -> list[Story]                     # advance lifecycles
@dataclass
class Story:
    det: core.Detection
    phase: str            # enter|hold|linger|exit
    phase_frac: float     # 0..1 within phase (renderer maps to alpha/scale)
    stage_alpha: float    # composite alpha the renderer must multiply in
```

The offline renderer calls `tick` per output frame; the extension mirrors the same constants in JS
(`presenter.js`, shared constant block generated from one JSON so the two never drift — build step
writes `extension/presenter_constants.json` and `bto/render/presenter_constants.py` from a single
source in `bto/render/presenter.py`).

## 4. Component B — Motion choreography (kill the jumps)

### 4.1 Springs

Every drawn point (chip anchor, line endpoint, hull vertex, radar dot) tracks its target through a
critically damped spring: `x'' = ω²(target − x) − 2ω·x'` with per-class ω:

| element | ω (rad/s) | rationale |
|---|---|---|
| event geometry endpoints (arrows, spotlights) | 8 | snappy, tracks sprinting players |
| chips/labels | 4 + **deadband 12 px** | text must be readable; ignores jitter |
| formation hull vertices / offside line | 2.5 | states drift, never dart |
| radar dots (M7) | 6 | |

At live host cadence (keyframes ≥ 0.4 s apart on M1) the spring runs client-side in `draw.js` at
rAF rate between keyframes (replacing the current linear lerp, ~20 lines); offline it runs per
output frame. Springs reset (snap) on shot cuts only.

### 4.2 Trails (how a finished moment stays comprehensible)

In LINGER, event geometry switches from "live endpoints" to the **completed path**: back pass →
full arrow from origin to receiver frozen; overlap → the whole run path with a motion-direction
chevron; trap → the three press-edge segments at their maximal convergence. We already record all
of this in `geometry['track']/'*_path'` (M6 convention) — the presenter just chooses *which slice
of time* to draw instead of always "now".

## 5. Component C — State-layer stabilization

- **Formation hull/label:** label changes gated by 8 s hysteresis (a 4-3-3 that flickers to
  4-4-2 for 3 s stays 4-3-3 visually; the *detection stream* keeps the truth). Hull vertices on
  ω=2.5 springs; when visible-player count < 7 the hull fades out entirely (P1: hide rather than
  jitter).
- **Offside line:** drawn only when its x-estimate has moved < 1.5 m over the last 0.6 s
  (stability gate); otherwise it fades. Prevents the line "hunting" during scrambles.
- **Marking edges (M6):** already lifecycle-based; add ENTER/EXIT fades (0.3 s) and cap concurrent
  drawn edges at 4 by strength with **sticky selection** (an edge keeps its slot until it dies or
  is beaten by 1.3×) — prevents the top-4 set itself from churning, which currently reads as
  blinking even though each edge is stable.

## 6. Component D — Cadence and processing-side changes (small but real)

- The live host currently *filters detections to those active at t*. Change: host sends each
  detection **once, complete, on birth** (and an amend on death with final t_end/geometry) over
  the existing protocol — a `story` message kind — rather than re-sending active geometry every
  reply. The extension's presenter owns display timing (P5 needs t_end + full path; per-reply
  active-filtering throws that away). Reply size shrinks; semantics get richer. Offline renderer
  reads m3_detections.json directly (already complete) — no change needed there beyond using the
  Presenter.
- Detector windows themselves: **no retuning for "fast games" in M8.** The user's instinct that
  fast play needs different processing is addressed at the presentation layer first because the
  precision pass showed detectors are firing correctly; if comprehension problems persist after
  M8 on the 3060 (full frame rate), revisit window constants with measured data. Explicit
  non-goal here to avoid two variables changing at once.

## 7. Validation

Perceptual quality is the target, so validation combines hard proxies with an A/B for the user:

- **Proxies (scripted, in `scripts/eval_presentation.py`):** rendered on both clips —
  (a) min/median on-screen lifetime of every visible callout ≥ 4.0 s / ≥ 4.5 s;
  (b) zero geometry discontinuities > 40 px/frame outside shot cuts (measure by differencing
  drawn-element positions frame to frame);
  (c) events on stage simultaneously: max 1; stage-occupancy ≤ 60% of main-camera time (room to
  breathe); (d) state layers: zero visibility flips shorter than 1 s.
- **A/B:** regenerate the same 50 s FIFA cut with old vs new presentation, deliver both; the
  user's "can I follow what it's telling me at full speed" verdict is the acceptance gate.
- **Live:** demo page on the Mac (2.6 Hz worst case — springs must make even that watchable),
  then 3060 (does the choreography hold at 20 Hz keyframes).

## 8. Integration plan

| file | change |
|---|---|
| `bto/render/presenter.py` | **new** — lifecycle/queue/springs constants (single source of truth) |
| `bto/render/overlay.py` | route all event drawing through Presenter; state-layer gates (§5) |
| `bto/host/server.py` + `primitives.py` | story-on-birth protocol message (§6); keep legacy per-reply geometry behind a flag for one release |
| `extension/draw.js` + `presenter.js` (new) | JS mirror: queue, lifecycle, springs at rAF; constants from generated JSON |
| `scripts/eval_presentation.py` | **new** — §7 proxies |
| `tests/test_presenter.py` | lifecycle math, queue priority, echo suppression, spring convergence — all headless |

## 9. Open questions

1. Stage size on big screens: is 1 concurrent story too strict for a tactics-nerd mode? Proposal:
   `max_stage` user-configurable (extension popup), default 1.
2. Audio/haptic cue on story ENTER (extension can ping)? Deferred; visual-only v1.
3. Should LINGER pause *longer* when the shot goes 'other' right after an event (replay of the
   same moment on TV)? Nice touch; needs shot-classifier tie-in; flag for M8.1.
4. Micro-replay PiP (buffer 8 s, re-render the moment slowed in a corner box) — the full
   broadcast-telestrator move. Big; spec separately as M9 if M8 lands well.
