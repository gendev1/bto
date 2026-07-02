"""StripeLock (bto.vision.calib drift mitigation (e)) synthetic unit tests.

Builds a synthetic top-view striped pitch image with a KNOWN px->m homography
H_true, perturbs its x-translation by a known amount, and proves that
StripeLock.refine recovers the opposite shift (sign + magnitude), i.e. that
H' = T(dx) @ H_pert moves the mapped stripe x's back onto the reference.
A plain (unstriped) pitch must gate off into a silent no-op.
"""

import numpy as np
import pytest

from bto.vision.calib import (
    STRIPE_STRENGTH_MIN,
    StripeLock,
    _stripe_refine_anchors,
    _anchor_px,
    project,
)

# top view: pixel (u, v) -> meters (u * 0.1, 68 - v * 0.1); 1050x680 image
# covers the full 105x68 pitch. An affine map is a valid homography and
# StripeLock must not care about perspective.
H_TRUE = np.array([
    [0.1, 0.0, 0.0],
    [0.0, -0.1, 68.0],
    [0.0, 0.0, 1.0],
])
W, H = 1050, 680


def _t(dx: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _synth_frame(striped: bool) -> np.ndarray:
    """Top-view pitch: mowing stripes = 3 m dark/light bands (6 m period),
    constant-x in world space, or a plain pitch of the same mean green."""
    rng = np.random.default_rng(0)
    x_m = np.arange(W) * 0.1  # world x of each pixel column
    if striped:
        band = (np.floor(x_m / 3.0) % 2).astype(np.float64)  # 0/1 every 3 m
        col = 110.0 + 25.0 * band
    else:
        col = np.full(W, 122.5)
    g = np.tile(col, (H, 1)) + rng.normal(0.0, 2.0, size=(H, W))
    g = np.clip(g, 0, 255).astype(np.uint8)
    frame = np.stack([np.full_like(g, 40), g, np.full_like(g, 40)], axis=-1)
    return frame


def test_stripelock_recovers_known_x_drift():
    frame = _synth_frame(striped=True)
    sl = StripeLock()

    strength = sl.update_reference(frame, H_TRUE)
    assert strength is not None and strength >= STRIPE_STRENGTH_MIN, \
        f"synthetic stripes not detected (strength={strength})"
    assert sl.ref is not None

    # perturb the x-translation by +0.8 m: every pixel now projects 0.8 m too
    # far in +x. refine must recover dx ~= -0.8 (sign + magnitude).
    H_pert = _t(+0.8) @ H_TRUE
    dx, strength2 = sl.refine(frame, H_pert)
    assert dx is not None, "refine returned no measurement on clean stripes"
    assert strength2 is not None and strength2 >= STRIPE_STRENGTH_MIN
    assert abs(dx - (-0.8)) <= 0.15, f"expected dx ~= -0.8, got {dx:+.3f}"

    # sign proof: applying T(dx) @ H_pert must land pixels back on H_true
    H_corr = _t(dx) @ H_pert
    px = np.array([[300.0, 400.0], [700.0, 250.0]])
    err = np.abs(project(H_corr, px) - project(H_TRUE, px)).max()
    assert err <= 0.15, f"T(dx) @ H_pert is {err:.3f} m from H_true"


def test_stripelock_correction_absorbed_into_anchors():
    """_stripe_refine_anchors requires two consecutive same-sign measurements
    (noise guard), then applies the damped correction as a pure x-shift of
    both H and the EMA anchor state (so the EMA cannot fight it)."""
    frame = _synth_frame(striped=True)
    sl = StripeLock()
    sl.update_reference(frame, H_TRUE)

    a_px = _anchor_px(W, H)
    H_pert = _t(+0.8) @ H_TRUE
    anchor = project(H_pert, a_px)

    # 1st measurement: qualifies the streak, must NOT apply yet
    ema_h, ema_anchor, dx1, app1, _ = _stripe_refine_anchors(
        sl, frame, H_pert, anchor, a_px)
    assert dx1 is not None and app1 is None
    assert ema_h is H_pert and ema_anchor is anchor, "first measurement corrected"

    # 2nd consecutive same-sign measurement: fires
    ema_h, ema_anchor, dx, app, strength = _stripe_refine_anchors(
        sl, frame, H_pert, anchor, a_px)
    assert dx is not None and abs(dx - (-0.8)) <= 0.15
    assert app is not None and abs(app - 0.7 * dx) <= 1e-9
    shift = ema_anchor - anchor
    assert np.allclose(shift[:, 1], 0.0, atol=1e-9), "correction moved y"
    assert np.allclose(shift[:, 0], app, atol=1e-9), \
        "anchors did not absorb exactly the applied dx"
    # the returned H goes through the corrected anchors
    assert np.abs(project(ema_h, a_px) - ema_anchor).max() < 1e-3
    # the streak restarts after a correction
    assert sl.prev_dx is None


def test_stripelock_plain_pitch_is_a_noop():
    frame = _synth_frame(striped=False)
    sl = StripeLock()

    strength = sl.update_reference(frame, H_TRUE)
    assert strength is None or strength < STRIPE_STRENGTH_MIN, \
        f"plain pitch passed the stripe gate (strength={strength})"
    assert sl.ref is None, "plain pitch must not seed a reference"

    dx, _ = sl.refine(frame, _t(0.5) @ H_TRUE)
    assert dx is None, "refine must be a no-op without stripes"
