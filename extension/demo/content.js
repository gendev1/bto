/**
 * BTO Telestrator — content.js
 *
 * Classic script exposing `window.BTOContent`. Orchestrates capture.js +
 * draw.js: finds the largest visible <video> on the page, injects a
 * transparent overlay canvas + status badge kept in sync with the video's
 * bounding rect, and wires the WebSocket capture/render pipeline to it.
 *
 * Works unmodified both as an MV3 content script (manifest.json loads
 * capture.js, draw.js, content.js in that order on every page) and as a
 * plain <script src> include on extension/demo/player.html.
 *
 * SPEC S4.1: non-fullscreen only — the overlay hides while
 * document.fullscreenElement is set.
 *
 * NOTE: extension/demo/content.js is a byte-identical copy of this file
 * (see capture.js's header comment for why). Keep the two in sync.
 */
(function () {
  'use strict';

  var WS_URL = 'ws://127.0.0.1:8517/stream';
  var RESCAN_INTERVAL_MS = 1000;
  var SYNC_INTERVAL_MS = 250;
  var BADGE_POLL_MS = 400;
  var MIN_VIDEO_AREA = 64 * 64;

  var STATUS_TEXT = {
    connecting: 'connecting…',
    live: 'live',
    'overlay-off': 'overlay off',
    drm: 'DRM protected',
    'host-down': 'host down',
  };

  var state = {
    video: null,
    canvas: null,
    badge: null,
    capture: null,
    draw: null,
    captureStatus: 'connecting',
    resizeObserver: null,
  };

  function isVisible(el) {
    if (!el || !el.isConnected) return false;
    var rect = el.getBoundingClientRect();
    if (rect.width * rect.height < MIN_VIDEO_AREA) return false;
    var style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    if (parseFloat(style.opacity || '1') === 0) return false;
    return true;
  }

  function findLargestVideo() {
    var vids = Array.prototype.filter.call(document.querySelectorAll('video'), isVisible);
    if (!vids.length) return null;
    vids.sort(function (a, b) {
      var ra = a.getBoundingClientRect();
      var rb = b.getBoundingClientRect();
      return rb.width * rb.height - ra.width * ra.height;
    });
    return vids[0];
  }

  function ensureUi() {
    if (state.canvas && state.canvas.isConnected) return;
    var canvas = document.createElement('canvas');
    canvas.id = 'bto-overlay-canvas';
    canvas.className = 'bto-overlay-canvas';
    var badge = document.createElement('div');
    badge.id = 'bto-status-badge';
    badge.className = 'bto-status-badge bto-status-connecting';
    badge.textContent = STATUS_TEXT.connecting;
    document.documentElement.appendChild(canvas);
    document.documentElement.appendChild(badge);
    state.canvas = canvas;
    state.badge = badge;
  }

  function removeUi() {
    if (state.canvas) state.canvas.remove();
    if (state.badge) state.badge.remove();
    state.canvas = null;
    state.badge = null;
  }

  function setBadge(effectiveStatus) {
    if (!state.badge) return;
    var text = STATUS_TEXT[effectiveStatus] || effectiveStatus;
    if (state.badge.textContent !== text) state.badge.textContent = text;
    var cls = 'bto-status-badge bto-status-' + effectiveStatus;
    if (state.badge.className !== cls) state.badge.className = cls;
  }

  function updateBadge() {
    if (!state.badge) return;
    var effective = state.captureStatus;
    if (effective === 'live' && state.draw && state.draw.isStale()) {
      effective = 'overlay-off';
    }
    setBadge(effective);
  }

  function syncCanvasRect() {
    if (!state.video || !state.canvas) return;
    var rect = state.video.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    var canvas = state.canvas;

    canvas.style.left = rect.left + 'px';
    canvas.style.top = rect.top + 'px';
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    var bw = Math.max(1, Math.round(rect.width * dpr));
    var bh = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }

    if (state.badge) {
      state.badge.style.left = rect.left + 8 + 'px';
      state.badge.style.top = rect.top + 8 + 'px';
    }

    var inFullscreen = !!document.fullscreenElement;
    canvas.style.display = inFullscreen ? 'none' : 'block';
    if (state.badge) state.badge.style.display = inFullscreen ? 'none' : 'block';
  }

  function teardownStreams() {
    if (state.capture) {
      state.capture.stop();
      state.capture = null;
    }
    if (state.draw) {
      state.draw.stop();
      state.draw = null;
    }
    if (state.resizeObserver) {
      state.resizeObserver.disconnect();
      state.resizeObserver = null;
    }
    state.captureStatus = 'connecting';
  }

  function attachToVideo(video) {
    if (state.video === video) return;
    teardownStreams();
    state.video = video;
    ensureUi();
    syncCanvasRect();

    state.resizeObserver = new ResizeObserver(function () {
      syncCanvasRect();
    });
    state.resizeObserver.observe(video);

    var draw = window.BTODraw.create(state.canvas);
    draw.setVideo(video);
    draw.start();
    state.draw = draw;

    var capture = window.BTOCapture.create({
      video: video,
      wsUrl: WS_URL,
      onStatus: function (s) {
        state.captureStatus = s;
        updateBadge();
      },
      onMessage: function (msg, dims) {
        draw.pushGeometry(msg, dims);
        updateBadge();
      },
      onDrm: function () {
        draw.stop();
        draw.clear();
        updateBadge();
      },
    });
    capture.start();
    state.capture = capture;

    updateBadge();
  }

  function detach() {
    teardownStreams();
    removeUi();
    state.video = null;
  }

  function rescan() {
    var v = findLargestVideo();
    if (v && v !== state.video) {
      attachToVideo(v);
    } else if (!v && state.video) {
      detach();
    }
  }

  function init() {
    rescan();
    var mo = new MutationObserver(function () {
      rescan();
    });
    mo.observe(document.documentElement, { childList: true, subtree: true });

    setInterval(rescan, RESCAN_INTERVAL_MS);
    setInterval(syncCanvasRect, SYNC_INTERVAL_MS);
    setInterval(updateBadge, BADGE_POLL_MS);

    window.addEventListener('scroll', syncCanvasRect, { passive: true, capture: true });
    window.addEventListener('resize', syncCanvasRect, { passive: true });
    document.addEventListener('fullscreenchange', syncCanvasRect);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.BTOContent = { rescan: rescan, _state: state };
})();
