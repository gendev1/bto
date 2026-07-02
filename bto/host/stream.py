"""Incremental live pipeline for the local inference host (SPEC C10 core).

One StreamPipeline instance owns BOTH resident models (player-detect +
pitch-pose, ~550 MB total -- fits the 8 GB budget; the dedicated ball model
stays offline-only, live ball = class 0 of the player detector) and turns a
single decoded BGR frame + video timestamp into everything the WS server
needs to build its reply: shot label, current homography, pattern-engine
detections active at t, and track count.

Per process(frame_bgr, t) call:

1. Shot gate (bto.vision.shots.classify_frame, pure CPU). 'other' frames
   return immediately (no GPU) and arm a segment reset so the NEXT main
   frame starts a fresh ByteTrack state (persist=False + tid offsetting,
   same semantics as bto.vision.detect.run_detection's shot-gated mode),
   fresh calib EMA, fresh ball Kalman, and an empty pattern buffer.
2. Detect+track via ultralytics .track(persist=True within a segment).
   imgsz defaults to 960 (not the offline 1280): ~1.7x faster on MPS at the
   cost of recall on the smallest far-touchline players (the offline M2/M3
   runs at 1280 saw ~20 players/frame on 1080p; expect a couple fewer here
   -- pattern confidences already scale with visible-player count).
   Ball = best class-0 box this frame (conf > BALL_CONF) feeding a
   constant-velocity Kalman (reused from bto.vision.ball.KalmanBall) that
   bridges up to BALL_MAX_MISSES processed-frame gaps.
3. Calibration: a full keypoint fit every `calib_every` main frames (or
   whenever there is no live H), else HOLD the last smoothed H. The fit +
   EMA/gate/hold/reseed state machine is an incremental port of the loop
   body of bto.vision.calib.calibrate(), reusing calib.py's fit internals
   (_frame_keypoints/_fit_homography/anchor-space EMA helpers) without
   touching calib.py itself. calib_src: 'fit' (fresh seed) | 'ema'
   (blended fit) | 'held' (held H, incl. the frames between fits) | 'null'.
4. Teams: warmup collects torso hue/sat histograms (teams.py helpers) from
   every player box until >= `warmup_crops`, then fits the 2-means kit
   model once; warmup tids get majority-vote labels, every NEW tid after
   that is classified nearest-centroid from its first valid crop and cached.
   Until warmup completes, unlabeled players are DROPPED from Frames (the
   pattern engine needs teams). GK/ref come from the detector class.
5. Bridge px -> bto.core.Frame: foot-point projection through H, bounds
   filter, referee drop, GK folding and attacking direction -- a causal
   (running-sums) port of bto.vision.bridge's per-segment logic -- appended
   to a rolling buffer trimmed to `buffer_s` seconds.
6. bto.patterns.run_all over the buffer every `patterns_every` processed
   main frames (cached in between); the reply carries the cached detections
   active at the current t (t_start <= t <= t_end + det_grace_s).

Self-check (<= 40 GPU frames): `uv run python -m bto.host.stream`.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from bto.core import AWAY, HOME, Detection, Frame, PlayerPos
from bto.patterns import run_all
from bto.vision import calib as _calib
from bto.vision import teams as _teams
from bto.vision.ball import KalmanBall
from bto.vision.bridge import HALF_X, X_MAX, X_MIN, Y_MAX, Y_MIN
from bto.vision.detect import (
    CLASS_NAMES,
    DEFAULT_MODEL as PLAYER_MODEL,
    _ensure_lap_shim,
    _pick_device,
)
from bto.vision.shots import classify_frame

PITCH_MODEL = _calib.DEFAULT_MODEL

_ensure_lap_shim()  # ByteTrack needs `lap`; detect.py's scipy shim covers it

BALL_CONF = 0.3        # min class-0 confidence to accept a ball measurement
BALL_MAX_MISSES = 8    # processed-frame gaps the Kalman is allowed to bridge


# ---------------------------------------------------------------------------
# Incremental calibration: port of calib.calibrate()'s per-frame state machine
# ---------------------------------------------------------------------------


class _IncrementalCalib:
    """Anchor-space EMA + sanity gate + hold + reseed interpolation, exactly
    the state machine inside bto.vision.calib.calibrate()'s loop, exposed as
    fit(frame, t) / hold(t) so the live pipeline can run a real keypoint fit
    only every few frames. All math is delegated to calib.py helpers."""

    def __init__(self, model, imgsz: int = 640, device: str = "cpu", kp_conf: float = 0.5):
        self.model = model
        self.imgsz = imgsz
        self.device = device
        self.kp_conf = kp_conf
        self.reset()

    def reset(self) -> None:
        self.ema_anchor = None      # (4,2) smoothed anchor projections [m]
        self.ema_h = None           # H through the smoothed anchors
        self.rej_anchor = None      # anchors of the last gate-rejected fit
        self.last_good_t = None
        self.last_held_anchor = None
        self.prev_resid = None
        self.prev_resid_mag = 0.0
        self.pan_streak = 0
        self.last_n_kp = 0
        self.last_rmse = None

    def hold(self, t: float):
        """No fit this frame: hold the last good H up to HOLD_MAX_S, else null."""
        if self.ema_h is not None and self.last_good_t is not None \
                and t - self.last_good_t <= _calib.HOLD_MAX_S:
            return self.ema_h, "held"
        self.ema_anchor = self.ema_h = None
        self.last_good_t = None
        # last_held_anchor deliberately kept: it seeds the reseed interpolation
        return None, "null"

    def fit(self, frame_bgr: np.ndarray, t: float):
        """Run the pitch-pose model + full fit path. Returns (H|None, src)."""
        frame_h, frame_w = frame_bgr.shape[:2]
        a_px = _calib._anchor_px(frame_w, frame_h)

        result = self.model(frame_bgr, imgsz=self.imgsz, device=self.device,
                            verbose=False)[0]
        kp_px, kp_m = _calib._frame_keypoints(result, self.kp_conf)
        self.last_n_kp = len(kp_px)

        H_fit, rmse = _calib._fit_homography(kp_px, kp_m)
        fit_anchor = _calib.project(H_fit, a_px) if H_fit is not None else None
        if fit_anchor is not None and not np.all(np.isfinite(fit_anchor)):
            H_fit, fit_anchor, rmse = None, None, None

        # sanity gate; two consecutive mutually-consistent rejects re-seed
        if fit_anchor is not None and self.ema_anchor is not None \
                and not _calib._anchors_close(fit_anchor, self.ema_anchor):
            if self.rej_anchor is not None \
                    and _calib._anchors_close(fit_anchor, self.rej_anchor):
                self.ema_anchor = None  # re-seed from this fit
            else:
                self.rej_anchor = fit_anchor
                H_fit, fit_anchor, rmse = None, None, None

        if H_fit is None:
            self.last_rmse = None
            return self.hold(t)

        if self.ema_anchor is None:
            # reseed: interpolate part-way when it disagrees with the last hold
            if self.last_held_anchor is not None:
                resid0, jump = _calib._residual(fit_anchor, self.last_held_anchor)
            else:
                resid0, jump = None, 0.0
            if resid0 is not None and jump > _calib.RESEED_INTERP_M:
                a = _calib.RESEED_INTERP_ALPHA
                self.ema_anchor = a * fit_anchor + (1.0 - a) * self.last_held_anchor
                self.prev_resid, self.prev_resid_mag, self.pan_streak = resid0, jump, 1
            else:
                self.ema_anchor = fit_anchor
                self.prev_resid, self.prev_resid_mag, self.pan_streak = None, 0.0, 0
            src = "fit"
        else:
            # anchor-space EMA with pan-adaptive alpha
            resid, mag = _calib._residual(fit_anchor, self.ema_anchor)
            consistent = (
                self.prev_resid is not None
                and mag > _calib.PAN_DISP_M and self.prev_resid_mag > _calib.PAN_DISP_M
                and _calib._cos_sim(resid, self.prev_resid) > _calib.PAN_COS_MIN
            )
            self.pan_streak = self.pan_streak + 1 if consistent \
                else int(mag > _calib.PAN_DISP_M)
            alpha = _calib.PAN_ALPHA if self.pan_streak >= 2 else _calib.EMA_ALPHA
            self.ema_anchor = alpha * fit_anchor + (1.0 - alpha) * self.ema_anchor
            self.prev_resid, self.prev_resid_mag = resid, mag
            src = "ema"

        self.ema_h = _calib._h_from_anchors(a_px, self.ema_anchor)
        self.last_good_t = t
        self.rej_anchor = None
        self.last_held_anchor = self.ema_anchor
        self.last_rmse = rmse
        return self.ema_h, src


# ---------------------------------------------------------------------------
# Incremental team assignment (SPEC C4, live variant)
# ---------------------------------------------------------------------------


class _LiveTeams:
    """Warmup: collect per-box torso hue/sat histograms until >= warmup_crops,
    fit 2-means once (teams.py helpers, kmeans++ restarts); warmup tids take
    the majority vote of their samples, later NEW tids are classified nearest
    centroid from their first valid crop and cached. Cluster 0 -> 'home'
    (arbitrary but stable, same convention as teams.py)."""

    def __init__(self, warmup_crops: int = 80):
        self.warmup_crops = warmup_crops
        self.centers: np.ndarray | None = None
        self.cache: dict[int, str] = {}   # tid -> HOME|AWAY
        self._warm: list[tuple[int, np.ndarray]] = []
        self.n_collected = 0

    @property
    def ready(self) -> bool:
        return self.centers is not None

    def _hist(self, frame_bgr, xyxy):
        crop = _teams._torso_crop(frame_bgr, xyxy)
        if crop is None:
            return None
        return _teams._hue_sat_hist(crop)

    def observe(self, frame_bgr: np.ndarray, players: list[tuple[int, list]]) -> None:
        """players: [(tid, xyxy)] for cls=='player' boxes with a tid."""
        if not self.ready:
            for tid, xyxy in players:
                h = self._hist(frame_bgr, xyxy)
                if h is not None:
                    self._warm.append((tid, h))
            self.n_collected = len(self._warm)
            if self.n_collected >= self.warmup_crops:
                X = np.stack([h for _, h in self._warm])
                labels, self.centers = _teams._kmeans(X, k=2)
                votes: dict[int, list[int]] = defaultdict(lambda: [0, 0])
                for (tid, _), lab in zip(self._warm, labels):
                    votes[tid][int(lab)] += 1
                for tid, v in votes.items():
                    self.cache[tid] = HOME if v[0] >= v[1] else AWAY
                self._warm = []
            return
        for tid, xyxy in players:
            if tid in self.cache:
                continue
            h = self._hist(frame_bgr, xyxy)
            if h is None:
                continue  # retry on a later frame
            d = ((self.centers - h) ** 2).sum(axis=1)
            self.cache[tid] = HOME if int(np.argmin(d)) == 0 else AWAY

    def team_of(self, tid: int) -> str | None:
        return self.cache.get(tid)


# ---------------------------------------------------------------------------
# Incremental px -> Frame bridge (causal port of bto.vision.bridge)
# ---------------------------------------------------------------------------


class _LiveBridge:
    """Per-segment running sums replace bridge.py's whole-segment stats:
    GK folds into the team whose running outfield centroid-x sits on the
    gk's half; attacking = team whose back reference (folded gk mean x, else
    mean per-frame rearmost x) is nearer x=0 attacks +1."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._team_x = {HOME: [0.0, 0], AWAY: [0.0, 0]}
        self._gk_x: dict[int, list] = {}
        self._rear_x = {HOME: [0.0, 0], AWAY: [0.0, 0]}

    @staticmethod
    def _project(H: np.ndarray, x: float, y: float):
        p = H @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    @staticmethod
    def _in_bounds(pt) -> bool:
        return X_MIN <= pt[0] <= X_MAX and Y_MIN <= pt[1] <= Y_MAX

    def frame(self, t: float, H: np.ndarray,
              entries: list[tuple[int, str, float, float]],
              ball_px) -> Frame:
        """entries: (tid, tag, foot_px_x, foot_px_y), tag in {home, away, gk}."""
        placed: list[tuple[int, str, float, float]] = []
        for tid, tag, fx, fy in entries:
            pt = self._project(H, fx, fy)
            if pt is None or not self._in_bounds(pt):
                continue
            placed.append((tid, tag, pt[0], pt[1]))
            if tag == "gk":
                s = self._gk_x.setdefault(tid, [0.0, 0])
                s[0] += pt[0]
                s[1] += 1
            else:
                self._team_x[tag][0] += pt[0]
                self._team_x[tag][1] += 1
        # per-frame rearmost x per team (attacking fallback when no gk seen)
        for team in (HOME, AWAY):
            xs = [x for _, tag, x, _ in placed if tag == team]
            if xs:
                self._rear_x[team][0] += min(xs)
                self._rear_x[team][1] += 1

        have_both = all(self._team_x[tm][1] > 0 for tm in (HOME, AWAY))
        cent = {tm: (s[0] / s[1] if s[1] else HALF_X) for tm, s in self._team_x.items()}
        left = min((HOME, AWAY), key=lambda tm: cent[tm])

        gk_team: dict[int, str] = {}
        if have_both:
            for tid, (sx, n) in self._gk_x.items():
                gx = sx / n
                gk_team[tid] = left if gx < HALF_X else (AWAY if left == HOME else HOME)

        ref_x = {}
        for team in (HOME, AWAY):
            gxs = [self._gk_x[tid][0] / self._gk_x[tid][1]
                   for tid in gk_team if gk_team[tid] == team]
            if gxs:
                ref_x[team] = sum(gxs) / len(gxs)
            elif self._rear_x[team][1]:
                ref_x[team] = self._rear_x[team][0] / self._rear_x[team][1]
            else:
                ref_x[team] = HALF_X
        home_plus = ref_x[HOME] <= ref_x[AWAY]
        attacking = {HOME: 1 if home_plus else -1, AWAY: -1 if home_plus else 1}

        players = [
            PlayerPos(f"T{tid}", gk_team[tid] if tag == "gk" else tag, x, y)
            for tid, tag, x, y in placed
            if tag != "gk" or tid in gk_team
        ]

        ball_m = None
        if ball_px is not None:
            pt = self._project(H, ball_px[0], ball_px[1])
            if pt is not None and self._in_bounds(pt):
                ball_m = pt

        return Frame(t=t, players=players, ball=ball_m, attacking=attacking)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class StreamPipeline:
    """Loads both models once; process(frame_bgr, t) -> dict with keys
    shot, H (np 3x3 or None), calib_src, frames_appended, detections
    (list[bto.core.Detection] active at t), n_tracks (+ timings extras)."""

    def __init__(
        self,
        device: str = "auto",
        imgsz: int = 960,
        calib_every: int = 4,   # M5 tune: 5 x ~0.4s live frame gap ~= HOLD_MAX_S
                                # (2.0s), so one failed fit == instant hold expiry
                                # (null-H flash); 4 keeps a failed fit inside hold
        patterns_every: int = 1,  # M5 tune: run_all is ~1ms on a full buffer --
                                  # running it every processed main frame keeps
                                  # geometry fresh (was 3: overlay data went up
                                  # to ~1.2s stale between runs at ~2.6 proc fps)
        buffer_s: float = 15.0,
        warmup_crops: int = 80,
        conf: float = 0.3,
        calib_imgsz: int = 640,
        kp_conf: float = 0.5,
        det_grace_s: float = 0.75,
        player_model: str = PLAYER_MODEL,
        pitch_model: str = PITCH_MODEL,
    ):
        from ultralytics import YOLO

        self.device = _pick_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.calib_every = calib_every
        self.patterns_every = patterns_every
        self.buffer_s = buffer_s
        self.det_grace_s = det_grace_s

        self.player_model = YOLO(player_model)
        self.calib = _IncrementalCalib(YOLO(pitch_model), imgsz=calib_imgsz,
                                       device=self.device, kp_conf=kp_conf)
        self.teams = _LiveTeams(warmup_crops=warmup_crops)
        self.bridge = _LiveBridge()
        self.models_loaded = True

        # segment / tracker state (detect.py tid-offset semantics)
        self._pending_reset = True
        self._tid_offset = 0
        self._max_raw_tid = 0

        # ball Kalman
        self._kf = KalmanBall()
        self._ball_misses = 0

        # buffers / caches
        self._buffer: list[Frame] = []
        self._frames_appended = 0
        self._since_fit = 0
        self._since_patterns = 0
        self._cached_dets: list[Detection] = []

        # cumulative per-stage timings [ms] for reporting
        self.stage_ms: dict[str, float] = defaultdict(float)
        self.n_main = 0
        self.n_calls = 0

    # -- segment handling ---------------------------------------------------

    def _start_segment(self) -> None:
        self._tid_offset += self._max_raw_tid
        self._max_raw_tid = 0
        self.calib.reset()
        self.bridge.reset()
        self._kf = KalmanBall()
        self._ball_misses = 0
        self._buffer.clear()
        self._cached_dets = []
        self._since_fit = 0
        self._since_patterns = 0

    # -- per-stage helpers ---------------------------------------------------

    def _detect(self, frame_bgr: np.ndarray, persist: bool):
        """Returns (boxes, ball_meas): boxes = [(tid|None, cls_name, xyxy)],
        ball_meas = (cx, cy) of the best class-0 detection or None."""
        result = self.player_model.track(
            frame_bgr,
            persist=persist,
            tracker="bytetrack.yaml",
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]
        boxes_out: list[tuple[int | None, str, list[float]]] = []
        ball_best = None
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            tids = boxes.id
            tids = tids.cpu().numpy().astype(int) if tids is not None else None
            for i in range(len(cls)):
                c = int(cls[i])
                if c == 0:  # ball: best-confidence class-0 box this frame
                    if confs[i] > BALL_CONF and (ball_best is None or confs[i] > ball_best[2]):
                        x1, y1, x2, y2 = xyxy[i]
                        ball_best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, float(confs[i]))
                    continue
                tid = None
                if tids is not None:
                    raw = int(tids[i])
                    self._max_raw_tid = max(self._max_raw_tid, raw)
                    tid = raw + self._tid_offset
                boxes_out.append((tid, CLASS_NAMES.get(c, str(c)), [float(v) for v in xyxy[i]]))
        ball_meas = (ball_best[0], ball_best[1]) if ball_best is not None else None
        return boxes_out, ball_meas

    def _ball(self, meas):
        """Kalman-smoothed ball px (bridges gaps up to BALL_MAX_MISSES)."""
        if meas is not None:
            if not self._kf.initialized:
                self._kf.init(meas[0], meas[1])
            else:
                self._kf.predict()
                self._kf.update(meas)
            self._ball_misses = 0
            return float(self._kf.x[0]), float(self._kf.x[1])
        if self._kf.initialized:
            px, py = self._kf.predict()
            self._ball_misses += 1
            if self._ball_misses <= BALL_MAX_MISSES:
                return px, py
        return None

    # -- main entry point ----------------------------------------------------

    def process(self, frame_bgr: np.ndarray, t: float) -> dict:
        self.n_calls += 1
        t_call = time.perf_counter()

        # (1) shot gate -- 'other' costs no GPU and arms a segment reset
        t0 = time.perf_counter()
        shot = classify_frame(frame_bgr)
        self.stage_ms["shot"] += (time.perf_counter() - t0) * 1e3
        if shot != "main":
            self._pending_reset = True
            return {
                "shot": shot,
                "H": None,
                "calib_src": "null",
                "frames_appended": self._frames_appended,
                "detections": [],
                "n_tracks": 0,
                "timings": {"total_ms": (time.perf_counter() - t_call) * 1e3},
            }

        persist = not self._pending_reset
        if self._pending_reset:
            self._start_segment()
            self._pending_reset = False
        self.n_main += 1

        # (2) detect + track (+ ball Kalman)
        t0 = time.perf_counter()
        boxes, ball_meas = self._detect(frame_bgr, persist)
        ball_px = self._ball(ball_meas)
        self.stage_ms["detect"] += (time.perf_counter() - t0) * 1e3

        # (3) calibration: full fit every calib_every main frames or whenever
        # there is no live H; hold in between.
        t0 = time.perf_counter()
        if self._since_fit >= self.calib_every - 1 or self.calib.ema_h is None:
            H, calib_src = self.calib.fit(frame_bgr, t)
            # M5 fix: a failed fit falls back to hold ('held'/'null'); retry
            # NEXT frame instead of waiting another calib_every frames, or the
            # hold expires (HOLD_MAX_S) before the next attempt at live cadence.
            self._since_fit = 0 if calib_src in ("fit", "ema") else self.calib_every - 1
        else:
            H, calib_src = self.calib.hold(t)
            self._since_fit += 1
        self.stage_ms["calib"] += (time.perf_counter() - t0) * 1e3

        # (4) teams (warmup then per-new-track nearest centroid)
        t0 = time.perf_counter()
        self.teams.observe(
            frame_bgr,
            [(tid, xyxy) for tid, cls, xyxy in boxes if cls == "player" and tid is not None],
        )
        self.stage_ms["teams"] += (time.perf_counter() - t0) * 1e3

        # (5) bridge -> rolling Frame buffer (only when H exists; null-H
        # frames are holes, matching bridge.py)
        t0 = time.perf_counter()
        if H is not None:
            entries = []
            for tid, cls, xyxy in boxes:
                if tid is None or cls == "referee":
                    continue
                foot = ((xyxy[0] + xyxy[2]) / 2.0, xyxy[3])
                if cls == "goalkeeper":
                    entries.append((tid, "gk", foot[0], foot[1]))
                else:
                    team = self.teams.team_of(tid)
                    if team is None:
                        continue  # pre-warmup: unlabeled players are dropped
                    entries.append((tid, team, foot[0], foot[1]))
            self._buffer.append(self.bridge.frame(t, H, entries, ball_px))
            self._frames_appended += 1
            while self._buffer and t - self._buffer[0].t > self.buffer_s:
                self._buffer.pop(0)
        self.stage_ms["bridge"] += (time.perf_counter() - t0) * 1e3

        # (6) pattern engine every patterns_every processed main frames
        t0 = time.perf_counter()
        self._since_patterns += 1
        if self._buffer and self._since_patterns >= self.patterns_every:
            self._cached_dets = run_all(self._buffer)
            self._since_patterns = 0
        active = [
            d for d in self._cached_dets
            if d.t_start <= t + 1e-9 and d.t_end >= t - self.det_grace_s
        ]
        self.stage_ms["patterns"] += (time.perf_counter() - t0) * 1e3

        n_tracks = len({tid for tid, _, _ in boxes if tid is not None})
        total_ms = (time.perf_counter() - t_call) * 1e3
        self.stage_ms["total"] += total_ms

        return {
            "shot": shot,
            "H": H,
            "calib_src": calib_src,
            "frames_appended": self._frames_appended,
            "detections": active,
            "n_tracks": n_tracks,
            "timings": {
                "total_ms": total_ms,
                "n_kp": self.calib.last_n_kp,
                "rmse_m": self.calib.last_rmse,
                "teams_ready": self.teams.ready,
                "ball_px": ball_px,
                "buffer_len": len(self._buffer),
            },
        }


# ---------------------------------------------------------------------------
# Self-check: 40 frames straight from the cwc clip (a main-camera stretch)
# ---------------------------------------------------------------------------


def _self_check() -> None:
    import cv2

    video = "data/clips/cwc2021_chelsea_palmeiras_20m.mp4"
    start_frame = 4200  # inside the long main run [4125, 5070] per shots.jsonl
    n_frames = 40       # GPU budget: 40 detect calls + ~9 pitch fits

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"cannot open {video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    t_load = time.perf_counter()
    pipe = StreamPipeline(device="auto")
    print(f"[self-check] models loaded in {time.perf_counter() - t_load:.1f}s "
          f"(device={pipe.device}, imgsz={pipe.imgsz})")

    results = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        ok, frame = cap.read()
        assert ok, f"video ended at frame {start_frame + i}"
        t = (start_frame + i) / fps
        results.append(pipe.process(frame, t))
    elapsed = time.perf_counter() - t0
    cap.release()

    # shot: the stretch is main camera
    n_main = sum(1 for r in results if r["shot"] == "main")
    print(f"[self-check] shot: {n_main}/{n_frames} main")
    assert n_main >= n_frames * 0.9, "expected a main-camera stretch"

    # H arrives by frame 5 and holds between fits
    first_h = next((i for i, r in enumerate(results) if r["H"] is not None), None)
    assert first_h is not None and first_h < 5, f"H first arrived at frame {first_h}"
    held = [i for i, r in enumerate(results) if r["calib_src"] == "held"]
    assert held, "expected held frames between fits (calib_every=5)"
    for i in held:
        # a held frame's H must be EXACTLY the previous frame's H (no refit)
        assert results[i - 1]["H"] is not None
        assert np.array_equal(results[i]["H"], results[i - 1]["H"]), \
            f"held frame {i} changed H"
    srcs = {s: sum(1 for r in results if r["calib_src"] == s)
            for s in ("fit", "ema", "held", "null")}
    print(f"[self-check] H first at frame {first_h}; calib_src breakdown: {srcs} "
          f"(held frames verified identical to their fit)")

    # teams warmup completes or is progressing
    if pipe.teams.ready:
        n_home = sum(1 for v in pipe.teams.cache.values() if v == HOME)
        n_away = len(pipe.teams.cache) - n_home
        print(f"[self-check] teams warmup COMPLETE: {len(pipe.teams.cache)} tids "
              f"cached (home={n_home} away={n_away})")
        assert n_home > 0 and n_away > 0, "kit clustering degenerated to one team"
    else:
        print(f"[self-check] teams warmup progressing: "
              f"{pipe.teams.n_collected}/{pipe.teams.warmup_crops} crops")
        assert pipe.teams.n_collected > 0, "no torso crops collected at all"

    # >= 1 formation/offside detection by the end
    types_seen = {d.type for r in results for d in r["detections"]}
    print(f"[self-check] detection types seen: {sorted(types_seen)}")
    assert types_seen & {"formation", "offside_line"}, \
        f"expected formation/offside by the end, saw {types_seen}"
    last = results[-1]
    print(f"[self-check] last frame: n_tracks={last['n_tracks']} "
          f"frames_appended={last['frames_appended']} "
          f"active_dets={len(last['detections'])} "
          f"ball_px={last['timings']['ball_px']}")

    # per-stage ms breakdown + proc fps
    n = pipe.n_main
    print("[self-check] per-main-frame stage breakdown:")
    for stage in ("shot", "detect", "calib", "teams", "bridge", "patterns", "total"):
        print(f"  {stage:9s} {pipe.stage_ms[stage] / max(n, 1):8.1f} ms")
    print(f"[self-check] {n_frames} frames in {elapsed:.1f}s "
          f"= {n_frames / elapsed:.2f} proc fps")
    print("[self-check] OK")


if __name__ == "__main__":
    _self_check()
