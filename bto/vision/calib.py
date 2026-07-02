"""Pitch calibration (SPEC C5): pose model detects 32 pitch landmarks per frame,
fit a px -> meters homography, temporally smoothed (SPEC S9 drift mitigations).

Writes calib.jsonl (frozen interchange format, one line per processed MAIN frame):
    {"frame_idx": int, "t": float, "H": [9 floats row-major, px->meters]|null,
     "n_kp": int, "rmse_m": float|null, "src": "fit"|"ema"|"held"}

H maps homogeneous pixel coords in the ORIGINAL video resolution to the bto.core
105x68 meter frame (origin bottom-left, y up).

src semantics:
    "fit"  - this frame produced an accepted keypoint fit and it seeded a fresh
             EMA state (first good frame of a shot segment / after a hold expiry).
    "ema"  - this frame produced an accepted keypoint fit, blended into the
             running EMA state (the normal steady case).
    "held" - no accepted fit this frame; H is the last good (EMA) H held for up
             to HOLD_MAX_S seconds, or null once the hold expires.

Drift mitigations (SPEC S9), all three, plus an M4 pan-lag tune (d):
    (a) EMA smoothing of H across consecutive frames, reset at shot
        boundaries. Smoothing happens in ACTION space, not matrix space: we EMA
        the meter-projections of 4 fixed pixel anchor points and refit H exactly
        through them (cv2.getPerspectiveTransform), because elementwise mixing
        of H[2][2]=1-normalized matrices is unusable on this data -- with the
        7-9 visible keypoints clustered near the midline the perspective row of
        H is ill-determined and flips sign between consecutive fits (measured on
        bundesliga_smoke frames 40/42: parents agree to <= 6.6 m at the anchors
        but their elementwise 50/50 blend lands 9-20 m away from BOTH parents).
        Anchor-space EMA is exact linear interpolation of the mapping.
        Alpha is ADAPTIVE (docs/m3-report.md "Calib EMA lag on pans"): a fixed
        alpha=0.5 tracks noise well but lags a fast camera pan by 1-3 m
        (measured on cwc2021 wireframe eyeball grading: center circle drawn
        1-3 m off mid-pan). Each frame with an accepted fit we look at the
        residual between the raw fit's anchors and the current EMA anchor: if
        that residual is large (per-anchor mean > PAN_DISP_M) AND points the
        same direction (cosine > PAN_COS_MIN) as the PREVIOUS frame's residual
        -- i.e. the EMA is being pulled the same way two frames running, a
        pan, not single-frame noise -- alpha ramps to PAN_ALPHA (~0.9) so the
        EMA catches up within ~2 frames; otherwise alpha stays at EMA_ALPHA
        (~0.5) for noise smoothing.
    (b) hold last good H up to 2 s when a frame yields no valid fit, else null;
    (c) sanity gate: each new fit must not move any anchor projection by > 8 m
        relative to the current smoothed state (else treated as no-fit -> hold
        path). Two CONSECUTIVE gated-off fits that agree with each other
        (<= 8 m) outvote the held state and re-seed the EMA, so a stale hold
        cannot deadlock the gate. Anchors are the corners of the LOWER HALF of
        the frame rather than the true frame corners: the top corners of a
        broadcast frame sit near or above the pitch plane's vanishing line,
        where a px->m homography is numerically explosive.
    (d) re-seed interpolation: a hold-expiry or gate-outvote reseed used to
        hard-jump the EMA straight onto the fresh fit's anchors, producing up
        to ~50 m single-frame jumps in the temporal-stability metric whenever
        the last held H and the fresh fit disagree (measured on cwc2021:
        held->fit transitions after shot cuts). Now, if the fresh fit
        disagrees with the LAST HELD anchor position by > RESEED_INTERP_M, the
        reseed frame (still emitted as src="fit", per the interchange
        contract) only moves RESEED_INTERP_ALPHA of the way there, primed with
        an elevated pan streak so the next 1-2 "ema" frames (via the adaptive
        alpha in (a)) finish closing the gap instead of a hard cut.
    (e) stripe-lock (StripeLock class below, M5.x): the pitch's mowing
        stripes are used as a temporal x-drift anchor between keypoint fits.
        Adds nullable "stripe_dx" / "stripe_strength" fields to each
        calib.jsonl row; disable with calibrate(..., stripe_lock=False).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch

DEFAULT_MODEL = "models/football-pitch-detection.pt"

EMA_ALPHA = 0.5          # base/noise alpha: small or inconsistent residuals
PAN_ALPHA = 0.9          # elevated alpha once a pan is detected (see (a) above)
PAN_DISP_M = 0.5         # per-anchor mean residual [m] above which a step is "large"
PAN_COS_MIN = 0.3        # min cosine similarity between consecutive residuals to call them "consistent"
HOLD_MAX_S = 2.0
GATE_TOL_M = 8.0
RESEED_INTERP_M = 5.0    # anchor disagreement [m] above which a reseed interpolates instead of jumping
RESEED_INTERP_ALPHA = 0.4  # fraction of the gap the reseed frame itself closes
RANSAC_THRESH_M = 1.5  # findHomography threshold applies in TARGET units = meters


# ---------------------------------------------------------------------------
# canonical keypoint coordinates
# ---------------------------------------------------------------------------
# The pose model's 32 keypoints follow the vertex order of
# docs/roboflow_soccer_pitch_config.py (SoccerPitchConfiguration.vertices):
# nominal 12000x7000 cm pitch, origin top-left, y DOWN. Constants inlined here
# (docs/ is not a package).

def _canonical_keypoints_m() -> np.ndarray:
    L, W = 12000.0, 7000.0          # nominal pitch length/width [cm]
    pb_w, pb_l = 4100.0, 2015.0     # penalty box
    gb_w, gb_l = 1832.0, 550.0      # goal box
    cc_r, ps_d = 915.0, 1100.0      # centre circle radius, penalty spot distance

    v = [
        (0, 0), (0, (W - pb_w) / 2), (0, (W - gb_w) / 2),                    # 1-3
        (0, (W + gb_w) / 2), (0, (W + pb_w) / 2), (0, W),                    # 4-6
        (gb_l, (W - gb_w) / 2), (gb_l, (W + gb_w) / 2), (ps_d, W / 2),       # 7-9
        (pb_l, (W - pb_w) / 2), (pb_l, (W - gb_w) / 2),                      # 10-11
        (pb_l, (W + gb_w) / 2), (pb_l, (W + pb_w) / 2),                      # 12-13
        (L / 2, 0), (L / 2, W / 2 - cc_r), (L / 2, W / 2 + cc_r), (L / 2, W),  # 14-17
        (L - pb_l, (W - pb_w) / 2), (L - pb_l, (W - gb_w) / 2),              # 18-19
        (L - pb_l, (W + gb_w) / 2), (L - pb_l, (W + pb_w) / 2),              # 20-21
        (L - ps_d, W / 2),                                                   # 22
        (L - gb_l, (W - gb_w) / 2), (L - gb_l, (W + gb_w) / 2),              # 23-24
        (L, 0), (L, (W - pb_w) / 2), (L, (W - gb_w) / 2),                    # 25-27
        (L, (W + gb_w) / 2), (L, (W + pb_w) / 2), (L, W),                    # 28-30
        (L / 2 - cc_r, W / 2), (L / 2 + cc_r, W / 2),                        # 31-32
    ]
    # frozen conversion: (vx, vy) cm -> 105x68 m frame, origin bottom-left, y UP.
    return np.array(
        [[vx / L * 105.0, 68.0 - vy / W * 68.0] for vx, vy in v], dtype=np.float64
    )


KP_METERS = _canonical_keypoints_m()  # (32, 2)


# ---------------------------------------------------------------------------
# homography math
# ---------------------------------------------------------------------------

def _norm_h(H: np.ndarray) -> np.ndarray:
    return H / H[2, 2]


def project(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Project (n,2) pixel points through H -> (n,2) meter points."""
    p = np.hstack([pts_px, np.ones((len(pts_px), 1))]) @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


def _fit_homography(kp_px: np.ndarray, kp_m: np.ndarray):
    """RANSAC-fit px->m homography. Returns (H normalized, rmse_m) or (None, None)."""
    if len(kp_px) < 4:
        return None, None
    H, mask = cv2.findHomography(kp_px, kp_m, cv2.RANSAC, RANSAC_THRESH_M)
    if H is None or abs(H[2, 2]) < 1e-12:
        return None, None
    inl = mask.ravel().astype(bool)
    if inl.sum() < 4:
        return None, None
    err = project(H, kp_px[inl]) - kp_m[inl]
    rmse = float(math.sqrt(float((err ** 2).sum(axis=1).mean())))
    return _norm_h(H), rmse


def _anchor_px(w: int, h: int) -> np.ndarray:
    # corners of the lower half of the frame (pitch-plane region; see module doc)
    return np.array([[0, h / 2], [w, h / 2], [w, h], [0, h]], dtype=np.float64)


def _anchors_close(a: np.ndarray, b: np.ndarray) -> bool:
    d = np.sqrt(((a - b) ** 2).sum(axis=1))
    return bool(np.all(np.isfinite(d)) and float(d.max()) <= GATE_TOL_M)


def _residual(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """(mean direction vector [m], per-anchor mean magnitude [m]) of a - b."""
    d = a - b
    mag = float(np.sqrt((d ** 2).sum(axis=1)).mean())
    return d.mean(axis=0), mag


def _cos_sim(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _h_from_anchors(anchor_px: np.ndarray, anchor_m: np.ndarray) -> np.ndarray:
    H = cv2.getPerspectiveTransform(
        anchor_px.astype(np.float32), anchor_m.astype(np.float32)
    )
    return _norm_h(H.astype(np.float64))


def _frame_keypoints(result, kp_conf: float):
    """Extract (kp_px, kp_m) correspondences from one ultralytics pose result,
    keeping keypoints with conf > kp_conf. If the model emitted several pitch
    instances, use the one with the highest box confidence."""
    kps = result.keypoints
    if kps is None or kps.xy is None or len(kps.xy) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    i = 0
    boxes = result.boxes
    if boxes is not None and len(boxes) > 1:
        i = int(boxes.conf.argmax())
    xy = kps.xy[i].cpu().numpy().astype(np.float64)
    conf = kps.conf[i].cpu().numpy() if kps.conf is not None else np.ones(len(xy))
    keep = (conf > kp_conf) & ((xy[:, 0] > 0) | (xy[:, 1] > 0))
    return xy[keep], KP_METERS[keep]


# ---------------------------------------------------------------------------
# StripeLock: mowing-stripe temporal x-drift anchor (drift mitigation (e))
# ---------------------------------------------------------------------------
# Mowing stripes (dark/light green bands parallel to the goal line, i.e.
# constant-x bands of the 105x68 frame) are FIXED in the world. If the current
# H maps a stripe edge to a sliding pitch-x during a camera pan, that slide IS
# calibration drift -- measurable and correctable. Stripes cannot give
# absolute position (widths are not standardized): the reference profile is
# (re)anchored on frames with an accepted RAW keypoint fit (absolute truth)
# and only holds x steady between/against the smoothed outputs.
#
# Per-frame use (causal, live-safe):
#   * anchored frames (an accepted raw fit exists, src 'fit'/'ema'):
#     update_reference(frame, H_fit_raw).
#   * HELD frames only (no accepted fit this frame): refine(frame, H_out)
#     cross-correlates the current profile against the reference over
#     dx in +-STRIPE_MAX_SHIFT_M; _stripe_refine_anchors() then absorbs the
#     damped, capped correction H' = T(dx_applied) @ H into the anchor-space
#     EMA state so the next frame does not fight it.
#     Refining 'ema' frames too was tried and measured (cwc2021, per-frame
#     raw fits as truth): the EMA already tracks fits to 0.06 m median in x,
#     while the stripe dx measurement carries 0.3-0.5 m noise/bias episodes
#     -- 23 of 31 fired corrections moved x AWAY from the fit. Stripes hold
#     x steady BETWEEN fits; they never overrule a frame that has one.
#
# Telemetry ("stripe_dx"/"stripe_strength", both nullable): stripe_dx is the
# MEASURED drift [m] (the applied correction is STRIPE_DAMP * clip(dx,
# +-STRIPE_DX_CAP_M) and only fires when |dx| >= STRIPE_DEADBAND_M, so
# fit-noise-level jitter is not fed back); stripe_strength is the stripe
# ALTERNATION score: max anti-correlation of the detrended profile with
# itself over half-period lags 2-8 m (periods ~4-16 m), scaled down when the
# detrended amplitude is tiny. Alternating bands anti-correlate strongly at
# half period; noise and smooth (shadow/vignette) variation do not. Below
# STRIPE_STRENGTH_MIN the whole feature is a silent no-op (plain pitches,
# heavy shadow, motion blur).
#
# Robustness (measured on cwc2021 FIFA+ 720p / bundesliga 1080p): the raw
# profile is polluted by non-pitch pixels that pass the in-frame bounds check
# (broadcast overlay panels sample 0/255 spikes) and by players/kits,
# drowning a +-4 gray-level stripe signal. Hence (1) a per-sample grass
# chroma mask (G dominant over B and not far below R), (2) MEDIAN aggregation
# across the y rows instead of mean, and (3) winsorizing the detrended
# profile at 3x its median |value| (white pitch-line spikes and player
# residue otherwise dominate the L2 correlations: on cwc frame 495 the
# half-period dip is -0.16 raw vs -0.49 winsorized). Surviving line spikes
# are harmless -- they are world-fixed, i.e. legitimate x-alignment signal.

STRIPE_STEP_M = 0.25          # x sampling step of the pitch-space profile
STRIPE_Y_ROWS = tuple(float(y) for y in range(14, 55, 4))  # central band
STRIPE_MIN_ROWS = 3           # min valid y samples per x cell
STRIPE_BASELINE_M = 8.0       # detrend box-filter width
STRIPE_HALFP_M = (2.0, 8.0)   # half-period lag range for the alternation dip
STRIPE_WINSOR_K = 3.0         # winsorize detrended profile at K * median|d|
STRIPE_AMP_MIN = 1.0          # full-credit rms amplitude [gray levels]
STRIPE_STRENGTH_MIN = 0.30    # min strength to trust the stripes at all
STRIPE_MIN_OVERLAP_M = 15.0   # min x-overlap for detrend/strength/correlation
STRIPE_MAX_SHIFT_M = 2.0      # correlation search window
STRIPE_PROM_MARGIN = 0.10     # peak must beat rivals > 1 m away by this
STRIPE_DEADBAND_M = 0.35      # |dx| below this: measure but do not correct
STRIPE_DX_CAP_M = 1.5         # max |dx| fed into the correction
STRIPE_DAMP = 0.7             # damping on the applied correction
STRIPE_REF_BLEND = 0.5        # blend of a new anchored profile into the ref

_STRIPE_N_X = int(round(105.0 / STRIPE_STEP_M)) + 1          # 421 cells
_STRIPE_MIN_CELLS = int(round(STRIPE_MIN_OVERLAP_M / STRIPE_STEP_M))  # 60


class StripeLock:
    """1-D pitch-x luminance profile matcher. All methods are per-frame and
    causal; every failure path degrades to a silent no-op."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.ref: np.ndarray | None = None  # (_STRIPE_N_X,) detrended, NaN=unknown
        self.prev_dx: float | None = None   # previous refine() measurement

    # -- profile extraction ---------------------------------------------------

    def _profile(self, frame_bgr: np.ndarray, H) -> tuple[np.ndarray | None, float | None]:
        """Detrended stripe profile p(x) over the visible pitch (NaN where not
        visible/sampled) + stripe_strength. (None, None) if not measurable."""
        h, w = frame_bgr.shape[:2]
        Hm = np.asarray(H, dtype=np.float64).reshape(3, 3)
        try:
            Hinv = np.linalg.inv(Hm)
        except np.linalg.LinAlgError:
            return None, None
        # in-front-of-camera sign: weight of a surely-on-pitch pixel (bottom center)
        bc_w = float((Hm @ np.array([w / 2.0, h - 1.0, 1.0]))[2])
        if abs(bc_w) < 1e-12:
            return None, None
        sgn = 1.0 if bc_w > 0 else -1.0

        xs = np.arange(_STRIPE_N_X) * STRIPE_STEP_M
        ys = np.asarray(STRIPE_Y_ROWS)
        X, Y = np.meshgrid(xs, ys)                      # (R, N)
        pts = np.stack([X, Y, np.ones_like(X)])         # (3, R, N)
        q = np.tensordot(Hinv, pts, axes=1)             # (3, R, N)
        den = q[2]
        ok = sgn * den > 1e-9
        safe = np.where(ok, den, 1.0)
        u = q[0] / safe
        v = q[1] / safe
        ok &= (u >= 0.0) & (u <= w - 1.001) & (v >= 0.0) & (v <= h - 1.001)
        if not ok.any():
            return None, None

        u = np.where(ok, u, 0.0)
        v = np.where(ok, v, 0.0)
        u0 = np.floor(u).astype(np.intp)
        v0 = np.floor(v).astype(np.intp)
        fu, fv = u - u0, v - v0
        w00, w10 = (1 - fu) * (1 - fv), fu * (1 - fv)
        w01, w11 = (1 - fu) * fv, fu * fv

        def _bilinear(ch: np.ndarray) -> np.ndarray:
            # gather-then-convert: only the sampled pixels leave uint8
            return (w00 * ch[v0, u0] + w10 * ch[v0, u0 + 1]
                    + w01 * ch[v0 + 1, u0] + w11 * ch[v0 + 1, u0 + 1])

        bv = _bilinear(frame_bgr[:, :, 0])
        gv = _bilinear(frame_bgr[:, :, 1])
        rv = _bilinear(frame_bgr[:, :, 2])
        # grass chroma mask: overlay graphics / white kits (G ~ B) and warm
        # colors (R >> G) pass the bounds check but are not pitch surface
        ok &= (gv >= bv + 8.0) & (rv <= gv + 12.0)

        cnt = ok.sum(axis=0)
        vals = np.where(ok, gv, np.nan)
        p = np.full(_STRIPE_N_X, np.nan)
        cols = cnt >= STRIPE_MIN_ROWS
        if not cols.any():
            return None, None
        p[cols] = np.nanmedian(vals[:, cols], axis=0)    # robust to players

        m = np.isfinite(p)
        if int(m.sum()) < _STRIPE_MIN_CELLS:
            return None, None
        # detrend: subtract a heavily-smoothed (normalized box) baseline
        win = int(round(STRIPE_BASELINE_M / STRIPE_STEP_M)) | 1
        k = np.ones(win)
        base_n = np.convolve(np.where(m, p, 0.0), k, "same")
        base_c = np.convolve(m.astype(np.float64), k, "same")
        base = base_n / np.maximum(base_c, 1.0)
        d = np.where(m & (base_c >= win * 0.5), p - base, np.nan)
        if int(np.isfinite(d).sum()) < _STRIPE_MIN_CELLS:
            return None, None
        # winsorize: line spikes / player residue otherwise dominate the
        # L2 correlations (see module comment)
        lim = STRIPE_WINSOR_K * float(np.nanmedian(np.abs(d)))
        if lim > 1e-9:
            d = np.clip(d, -lim, lim)
        return d, self._strength(d)

    @staticmethod
    def _strength(d: np.ndarray) -> float | None:
        """Stripe alternation score: max ANTI-correlation of the detrended
        profile with itself over half-period lags (STRIPE_HALFP_M), scaled by
        amplitude credit (rms of the detrended profile vs STRIPE_AMP_MIN)."""
        valid = np.isfinite(d)
        vals = d[valid]
        if len(vals) < _STRIPE_MIN_CELLS:
            return None
        amp = float(np.sqrt((vals ** 2).mean()))
        lo = int(round(STRIPE_HALFP_M[0] / STRIPE_STEP_M))
        hi = int(round(STRIPE_HALFP_M[1] / STRIPE_STEP_M))
        dip = 0.0
        for lag in range(lo, hi + 1):
            a, b = d[:-lag], d[lag:]
            m2 = np.isfinite(a) & np.isfinite(b)
            if int(m2.sum()) < 40:
                continue
            aa, bb = a[m2] - a[m2].mean(), b[m2] - b[m2].mean()
            denom = float(np.sqrt((aa ** 2).sum() * (bb ** 2).sum()))
            if denom < 1e-9:
                continue
            dip = max(dip, float(-(aa * bb).sum() / denom))
        return dip * min(1.0, amp / STRIPE_AMP_MIN)

    # -- reference anchoring / refinement --------------------------------------

    def update_reference(self, frame_bgr: np.ndarray, H_fit_raw) -> float | None:
        """Anchor/refresh the reference profile from a frame whose RAW keypoint
        fit passed the gate (absolute truth). Returns stripe_strength."""
        d, strength = self._profile(frame_bgr, H_fit_raw)
        if d is None or strength is None or strength < STRIPE_STRENGTH_MIN:
            return strength
        if self.ref is None:
            self.ref = d
        else:
            newm, refm = np.isfinite(d), np.isfinite(self.ref)
            both = newm & refm
            out = np.where(refm, self.ref, np.nan)
            out[newm] = d[newm]
            out[both] = ((1.0 - STRIPE_REF_BLEND) * self.ref[both]
                         + STRIPE_REF_BLEND * d[both])
            self.ref = out
        return strength

    def refine(self, frame_bgr: np.ndarray, H) -> tuple[float | None, float | None]:
        """Measure the x-drift of H against the reference. Returns
        (dx_measured_m | None, stripe_strength | None): applying T(dx) @ H
        (T = [[1,0,dx],[0,1,0],[0,0,1]]) moves the mapped stripe x's back onto
        the reference. None dx = no confident measurement (no-op)."""
        d, strength = self._profile(frame_bgr, H)
        if (d is None or strength is None or strength < STRIPE_STRENGTH_MIN
                or self.ref is None):
            return None, strength

        smax = int(round(STRIPE_MAX_SHIFT_M / STRIPE_STEP_M))
        n = _STRIPE_N_X
        scores = np.full(2 * smax + 1, np.nan)
        for i, sft in enumerate(range(-smax, smax + 1)):
            # drift e satisfies cur(x) = ref(x - e): score(sft) matches
            # ref(x) against cur(x + sft), peaking at sft = e.
            if sft >= 0:
                a, b = self.ref[:n - sft], d[sft:]
            else:
                a, b = self.ref[-sft:], d[:n + sft]
            m2 = np.isfinite(a) & np.isfinite(b)
            if int(m2.sum()) < _STRIPE_MIN_CELLS:
                continue
            aa, bb = a[m2] - a[m2].mean(), b[m2] - b[m2].mean()
            denom = float(np.sqrt((aa ** 2).sum() * (bb ** 2).sum()))
            if denom < 1e-9:
                continue
            scores[i] = float((aa * bb).sum() / denom)

        if not np.isfinite(scores).any():
            return None, strength
        bi = int(np.nanargmax(scores))
        if bi == 0 or bi == len(scores) - 1:      # peak at the search border
            return None, strength
        best = scores[bi]
        # prominence: any rival > 1 m away within the margin -> ambiguous
        for j in range(len(scores)):
            if abs(j - bi) > 4 and np.isfinite(scores[j]) \
                    and scores[j] > best - STRIPE_PROM_MARGIN:
                return None, strength
        # sub-cell parabolic refinement
        c0, c1, c2 = scores[bi - 1], scores[bi], scores[bi + 1]
        delta = 0.0
        if np.isfinite(c0) and np.isfinite(c2):
            denom = c0 - 2.0 * c1 + c2
            if abs(denom) > 1e-9:
                delta = float(np.clip(0.5 * (c0 - c2) / denom, -0.5, 0.5))
        e = (bi - smax + delta) * STRIPE_STEP_M
        dx = -e
        if abs(dx) > STRIPE_MAX_SHIFT_M:
            return None, strength
        return float(dx), strength


def _stripe_refine_anchors(stripes: StripeLock, frame_bgr: np.ndarray,
                           ema_h: np.ndarray, ema_anchor: np.ndarray,
                           a_px: np.ndarray):
    """Measure the stripe x-drift of the smoothed output H and absorb the
    damped, capped correction into the anchor-space EMA state (equivalent to
    H' = T(dx_applied) @ H, since the 4 anchors define H exactly). Returns
    (ema_h, ema_anchor, dx_measured | None, dx_applied | None,
    stripe_strength | None); dx_applied is None when no correction fired.

    A correction only fires when TWO CONSECUTIVE measurements agree in sign
    and both exceed the deadband (same philosophy as the pan detector in
    (a)): single-frame dx measurement noise (sigma ~0.3-0.5 m from reference
    phase jitter + sub-cell peak localization, measured on cwc2021 against
    per-frame raw fits) must not be fed back, while real drift -- a held H
    during camera motion, a stale hold after a cut -- keeps measuring the
    same sign every frame and passes."""
    dx, strength = stripes.refine(frame_bgr, ema_h)
    prev = stripes.prev_dx
    stripes.prev_dx = dx
    if (dx is None or abs(dx) < STRIPE_DEADBAND_M
            or prev is None or abs(prev) < STRIPE_DEADBAND_M or prev * dx <= 0):
        return ema_h, ema_anchor, dx, None, strength
    dx_app = STRIPE_DAMP * float(np.clip(dx, -STRIPE_DX_CAP_M, STRIPE_DX_CAP_M))
    ema_anchor = ema_anchor + np.array([dx_app, 0.0])
    ema_h = _h_from_anchors(a_px, ema_anchor)
    # a correction moves the output onto the reference: restart the streak so
    # the NEXT independent drift measurement must re-qualify
    stripes.prev_dx = None
    return ema_h, ema_anchor, dx, dx_app, strength


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_shot_labels(shots_jsonl: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    with open(shots_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels[int(rec["frame_idx"])] = rec["shot"]
    return labels


def calibrate(
    video_path: str,
    shots_jsonl: str | None = None,
    out_path: str | None = None,
    stride: int = 2,
    imgsz: int = 640,
    device: str = "auto",
    kp_conf: float = 0.5,
    model_path: str = DEFAULT_MODEL,
    max_frames: int | None = None,
    stripe_lock: bool = True,
    _kp_cache: str | None = None,
) -> str:
    """Run pitch calibration over `video_path`, writing calib.jsonl.

    Only frames labeled 'main' in shots_jsonl (same video / same stride) are
    processed and written; all frames are treated as main when it is None.
    frame_idx indexes the ORIGINAL video, t = frame_idx / fps. `max_frames`
    caps the number of PROCESSED (main) frames. Returns the output path.

    stripe_lock: enable drift mitigation (e) (StripeLock; silent no-op on
    pitches without visible mowing stripes). _kp_cache: internal A/B-testing
    hook -- path to a keypoint-cache jsonl; if the file exists the pose model
    is not loaded and per-frame keypoints are read from it, otherwise the
    model runs normally and the cache is written alongside.
    """
    from ultralytics import YOLO

    dev = _pick_device(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if out_path is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join("out", stem)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "calib.jsonl")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    kp_cache_data: dict | None = None
    kp_cache_f = None
    model = None
    if _kp_cache is not None and os.path.exists(_kp_cache):
        kp_cache_data = {}
        with open(_kp_cache) as cf:
            for line in cf:
                if line.strip():
                    r = json.loads(line)
                    kp_cache_data[int(r["frame_idx"])] = (
                        np.array(r["kp_px"], dtype=np.float64).reshape(-1, 2),
                        np.array(r["kp_m"], dtype=np.float64).reshape(-1, 2),
                    )
    else:
        model = YOLO(model_path)
        if _kp_cache is not None:
            kp_cache_f = open(_kp_cache, "w")
    shot_labels = _load_shot_labels(shots_jsonl) if shots_jsonl else None
    stripes = StripeLock() if stripe_lock else None
    stripe_ms_total, stripe_n, stripe_n_corr = 0.0, 0, 0

    ema_anchor: np.ndarray | None = None  # (4,2) smoothed anchor projections [m]
    ema_h: np.ndarray | None = None       # H through the smoothed anchors (last good H)
    rej_anchor: np.ndarray | None = None  # anchors of the last gate-rejected fit
    last_good_t: float | None = None
    # (d) last known-good anchor position, kept across hold/expiry/reseed so a
    # fresh fit can be compared against where the EMA actually last was (not
    # just whether ema_anchor happens to be non-None right now).
    last_held_anchor: np.ndarray | None = None
    # (a) pan-detection state: previous frame's fit-vs-ema residual + streak
    # of consecutive large-and-consistent residuals.
    prev_resid: np.ndarray | None = None
    prev_resid_mag: float = 0.0
    pan_streak = 0
    prev_main_idx: int | None = None
    frame_idx = 0
    n_written = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                if shot_labels is not None and shot_labels.get(frame_idx) != "main":
                    frame_idx += 1
                    continue

                frame_h, frame_w = frame.shape[:2]
                a_px = _anchor_px(frame_w, frame_h)
                t = frame_idx / fps

                # shot boundary at this stride -> reset all temporal state
                if prev_main_idx is not None and frame_idx - prev_main_idx != stride:
                    ema_anchor = ema_h = rej_anchor = None
                    last_good_t = None
                    last_held_anchor = None
                    prev_resid, prev_resid_mag, pan_streak = None, 0.0, 0
                    if stripes is not None:
                        stripes.reset()
                prev_main_idx = frame_idx

                t_frame = time.time()
                if kp_cache_data is not None:
                    kp_px, kp_m = kp_cache_data.get(
                        frame_idx, (np.empty((0, 2)), np.empty((0, 2))))
                else:
                    result = model(frame, imgsz=imgsz, device=dev, verbose=False)[0]
                    kp_px, kp_m = _frame_keypoints(result, kp_conf)
                    if kp_cache_f is not None:
                        kp_cache_f.write(json.dumps({
                            "frame_idx": frame_idx,
                            "kp_px": kp_px.tolist(),
                            "kp_m": kp_m.tolist(),
                        }) + "\n")
                n_kp = len(kp_px)

                H_fit, rmse = _fit_homography(kp_px, kp_m)
                fit_anchor = project(H_fit, a_px) if H_fit is not None else None
                if fit_anchor is not None and not np.all(np.isfinite(fit_anchor)):
                    H_fit, fit_anchor, rmse = None, None, None

                # (c) sanity gate against the current smoothed state (skipped
                # right after a reset). Two consecutive rejected-but-mutually-
                # consistent fits outvote a stale hold and re-seed the EMA.
                if fit_anchor is not None and ema_anchor is not None \
                        and not _anchors_close(fit_anchor, ema_anchor):
                    if rej_anchor is not None and _anchors_close(fit_anchor, rej_anchor):
                        ema_anchor = None  # re-seed from this fit
                    else:
                        rej_anchor = fit_anchor
                        H_fit, fit_anchor, rmse = None, None, None

                if H_fit is not None:
                    if ema_anchor is None:
                        # (d) reseed: jump straight in if there's no prior
                        # anchor to compare to, or the disagreement is small;
                        # otherwise interpolate part-way and prime the pan
                        # streak so the following ema frames close the rest.
                        if last_held_anchor is not None:
                            resid0, jump_mag = _residual(fit_anchor, last_held_anchor)
                        else:
                            resid0, jump_mag = None, 0.0
                        if resid0 is not None and jump_mag > RESEED_INTERP_M:
                            ema_anchor = (RESEED_INTERP_ALPHA * fit_anchor
                                          + (1.0 - RESEED_INTERP_ALPHA) * last_held_anchor)
                            prev_resid, prev_resid_mag, pan_streak = resid0, jump_mag, 1
                        else:
                            ema_anchor = fit_anchor
                            prev_resid, prev_resid_mag, pan_streak = None, 0.0, 0
                        ema_h, src = _h_from_anchors(a_px, ema_anchor), "fit"
                    else:  # (a) EMA in anchor (action) space, see module doc
                        resid, mag = _residual(fit_anchor, ema_anchor)
                        consistent = (prev_resid is not None
                                      and mag > PAN_DISP_M and prev_resid_mag > PAN_DISP_M
                                      and _cos_sim(resid, prev_resid) > PAN_COS_MIN)
                        pan_streak = pan_streak + 1 if consistent else int(mag > PAN_DISP_M)
                        alpha = PAN_ALPHA if pan_streak >= 2 else EMA_ALPHA
                        ema_anchor = alpha * fit_anchor + (1.0 - alpha) * ema_anchor
                        ema_h = _h_from_anchors(a_px, ema_anchor)
                        prev_resid, prev_resid_mag = resid, mag
                        src = "ema"
                    last_good_t = t
                    rej_anchor = None
                    last_held_anchor = ema_anchor
                    H_out, rmse_out = ema_h, rmse
                else:
                    # (b) hold last good H for up to HOLD_MAX_S, else null
                    src, rmse_out = "held", None
                    if ema_h is not None and last_good_t is not None \
                            and t - last_good_t <= HOLD_MAX_S:
                        H_out = ema_h
                    else:
                        H_out = ema_anchor = ema_h = None
                        last_good_t = None
                        # last_held_anchor deliberately NOT cleared here: it
                        # is the reseed's comparison point in (d) above.

                # (e) stripe-lock: refine smoothed/held output frames against
                # the reference, then (re)anchor the reference on frames with
                # an accepted RAW fit. Absorbed into the EMA anchors so the
                # next frame does not fight the correction.
                stripe_dx = stripe_strength = None
                if stripes is not None and H_out is not None:
                    t_stripe = time.time()
                    if src == "held" and ema_anchor is not None:
                        ema_h, ema_anchor, stripe_dx, dx_app, stripe_strength = \
                            _stripe_refine_anchors(stripes, frame, ema_h,
                                                   ema_anchor, a_px)
                        if dx_app is not None:
                            last_held_anchor = ema_anchor
                            H_out = ema_h
                            stripe_n_corr += 1
                    if H_fit is not None:
                        stripes.prev_dx = None  # a fit breaks the drift streak
                        s = stripes.update_reference(frame, H_fit)
                        if stripe_strength is None:
                            stripe_strength = s
                    stripe_ms_total += (time.time() - t_stripe) * 1e3
                    stripe_n += 1

                rec = {
                    "frame_idx": frame_idx,
                    "t": t,
                    "H": [float(v) for v in H_out.ravel()] if H_out is not None else None,
                    "n_kp": int(n_kp),
                    "rmse_m": rmse_out,
                    "src": src,
                    "stripe_dx": stripe_dx,
                    "stripe_strength": stripe_strength,
                }
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                # per-frame timing on short (self-check-sized) runs, else every 100
                if (max_frames is not None and max_frames <= 40) or n_written % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"[calib] frame_idx={frame_idx} n_kp={n_kp} src={src} "
                          f"frame_ms={(time.time() - t_frame) * 1e3:.0f} "
                          f"avg_fps={n_written / elapsed:.2f}")
                if max_frames is not None and n_written >= max_frames:
                    break
            frame_idx += 1

    cap.release()
    if kp_cache_f is not None:
        kp_cache_f.close()
    elapsed = time.time() - t0
    print(f"[calib] done: {n_written} main frames in {elapsed:.1f}s "
          f"({n_written / max(elapsed, 1e-9):.2f} fps) -> {out_path}")
    if stripes is not None and stripe_n:
        print(f"[calib] stripe-lock: {stripe_ms_total / stripe_n:.2f} ms/frame "
              f"avg over {stripe_n} frames, {stripe_n_corr} corrections applied")
    return out_path


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """40 main frames of bundesliga_smoke.mp4 (stride 2, matching its
    shots.jsonl): >= 80% of frames must get an H from that frame's own accepted
    fit (src 'fit' or 'ema' -- with EMA smoothing on, only the segment-seeding
    frame is literally 'fit'; all later fitted frames are 'ema'), median
    rmse_m < 1.5 m (SPEC S7), and a real player foot point from perception.jsonl
    must project into [-5,110]x[-5,73]."""
    video = "data/clips/bundesliga_smoke.mp4"
    shots = "out/bundesliga_smoke/shots.jsonl"
    perception = "out/bundesliga_smoke/perception.jsonl"
    out_path = "out/bundesliga_smoke/_selfcheck_calib.jsonl"
    n_frames = 40

    t_start = time.time()
    calibrate(video, shots_jsonl=shots, out_path=out_path, stride=2,
              max_frames=n_frames)

    with open(out_path) as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) == n_frames, f"expected {n_frames} records, got {len(recs)}"

    fitted = [r for r in recs if r["src"] in ("fit", "ema") and r["H"] is not None]
    frac = len(fitted) / len(recs)
    srcs = {s: sum(1 for r in recs if r["src"] == s) for s in ("fit", "ema", "held")}
    print(f"[self-check] src breakdown: {srcs}  fitted-frac={frac:.2f}")
    assert frac >= 0.80, f"only {frac:.0%} of frames got a fitted H (need >= 80%)"

    rmses = sorted(r["rmse_m"] for r in fitted)
    med_rmse = rmses[len(rmses) // 2]
    n_kps = sorted(r["n_kp"] for r in recs)
    med_nkp = n_kps[len(n_kps) // 2]
    print(f"[self-check] median rmse_m={med_rmse:.3f} (max {rmses[-1]:.3f}), "
          f"median n_kp={med_nkp}")
    assert med_rmse < 1.5, f"median rmse {med_rmse:.3f} m >= 1.5 m (SPEC S7)"

    # project one real player foot point through the matching frame's H
    h_by_idx = {r["frame_idx"]: r["H"] for r in recs if r["H"] is not None}
    foot = xy_m = None
    with open(perception) as f:
        for line in f:
            p = json.loads(line)
            if p["frame_idx"] in h_by_idx and p["players"]:
                foot = p["players"][0]["foot"]
                H = np.array(h_by_idx[p["frame_idx"]]).reshape(3, 3)
                xy_m = project(H, np.array([foot], dtype=np.float64))[0]
                break
    assert xy_m is not None, "no perception frame overlapped a calibrated frame"
    x, y = float(xy_m[0]), float(xy_m[1])
    print(f"[self-check] foot px={foot} -> pitch ({x:.1f}, {y:.1f}) m")
    assert -5.0 <= x <= 110.0 and -5.0 <= y <= 73.0, \
        f"projected foot ({x:.1f}, {y:.1f}) outside [-5,110]x[-5,73]"

    elapsed = time.time() - t_start
    print(f"[self-check] total {elapsed:.1f}s for {n_frames} frames "
          f"= {elapsed / n_frames * 1e3:.0f} ms/frame (incl. model load)")
    print("[self-check] OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pitch calibration -> homography (SPEC C5)")
    ap.add_argument("video", nargs="?", default=None)
    ap.add_argument("--shots", default=None, help="shots.jsonl (same video/stride); skip 'other' frames")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--kp-conf", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory (writes calib.jsonl inside it)")
    ap.add_argument("--no-stripe-lock", action="store_true",
                    help="disable the mowing-stripe x-drift anchor (mitigation (e))")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        _self_check()
        return
    if args.video is None:
        ap.error("video is required unless --self-check is given")

    out_path = os.path.join(args.out, "calib.jsonl") if args.out else None
    calibrate(
        args.video,
        shots_jsonl=args.shots,
        out_path=out_path,
        stride=args.stride,
        imgsz=args.imgsz,
        device=args.device,
        kp_conf=args.kp_conf,
        model_path=args.model,
        max_frames=args.max_frames,
        stripe_lock=not args.no_stripe_lock,
    )


if __name__ == "__main__":
    main()
