/**
 * BTO Telestrator — draw.js
 *
 * Classic script exposing `window.BTODraw`. Renders the geometry received
 * over the WS protocol onto an overlay <canvas>, smoothing across the gap
 * between host replies (host runs at ~10-15 Hz; render loop runs at
 * display refresh rate).
 *
 * Interpolation model: keep the last TWO geometry replies (prev, new).
 * Every rAF tick, for gids present in both, extrapolate forward from the
 * newest known position using the velocity implied by (new - prev):
 *
 *   f = clamp(clamp(now - t_new, 0, 200ms) / (t_new - t_prev), 0, 1)
 *   pt(now) = pt_new + f * (pt_new - pt_prev)
 *
 * `now` is read from the driving <video>'s currentTime so extrapolation
 * naturally halts when the video is paused/seeking. gids that vanish
 * between replies fade out over 300ms instead of popping. If no reply has
 * arrived in the last 2s (wall clock), the canvas is cleared.
 *
 * NOTE: extension/demo/draw.js is a byte-identical copy of this file (see
 * capture.js's header comment for why). Keep the two in sync.
 */
(function () {
  'use strict';

  var FADE_MS = 300;
  var STALE_MS = 2000;
  var MAX_EXTRAPOLATE_S = 0.2; // 200ms

  function indexByGid(geometry) {
    var m = new Map();
    (geometry || []).forEach(function (p) {
      if (p && p.gid) m.set(p.gid, p);
    });
    return m;
  }

  function roundRectPath(ctx, x, y, w, h, r) {
    var rr = Math.min(r, w / 2, h / 2);
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function lerpPts(ptsPrev, ptsNew, f) {
    if (!ptsPrev || ptsPrev.length !== ptsNew.length || f === 0) return ptsNew;
    var out = new Array(ptsNew.length);
    for (var i = 0; i < ptsNew.length; i++) {
      var a = ptsPrev[i];
      var b = ptsNew[i];
      out[i] = [b[0] + f * (b[0] - a[0]), b[1] + f * (b[1] - a[1])];
    }
    return out;
  }

  /** @param {HTMLCanvasElement} canvas */
  function create(canvas) {
    var ctx = canvas.getContext('2d');
    var video = null;
    var sentW = 0;
    var sentH = 0;
    var prevMsg = null; // {t, byGid}
    var newMsg = null; // {t, byGid}
    var lastMsgWallMs = 0;
    var vanished = new Map(); // gid -> {prim, sinceMs}
    var running = false;
    var rafHandle = null;

    function setVideo(v) {
      video = v;
    }

    function pushGeometry(msg, frameDims) {
      if (frameDims && frameDims.w && frameDims.h) {
        sentW = frameDims.w;
        sentH = frameDims.h;
      }
      lastMsgWallMs = performance.now();
      var nextByGid = indexByGid(msg.geometry);

      if (newMsg) {
        newMsg.byGid.forEach(function (prim, gid) {
          if (!nextByGid.has(gid)) {
            vanished.set(gid, { prim: prim, sinceMs: performance.now() });
          }
        });
        nextByGid.forEach(function (_prim, gid) {
          vanished.delete(gid);
        });
        prevMsg = newMsg;
      }
      newMsg = { t: msg.t, byGid: nextByGid };
    }

    function clear() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function scalePt(pt) {
      var sx = sentW > 0 ? canvas.width / sentW : 1;
      var sy = sentH > 0 ? canvas.height / sentH : 1;
      return [pt[0] * sx, pt[1] * sy];
    }

    function drawPrim(prim, alphaMul) {
      if (!prim || !prim.kind) return;
      var color = prim.color || [255, 255, 255];
      var baseAlpha = typeof prim.alpha === 'number' ? prim.alpha : 1;
      var alpha = baseAlpha * alphaMul;
      if (alpha <= 0.002) return;
      var pts = (prim.pts || []).map(scalePt);
      var style = 'rgba(' + color[0] + ',' + color[1] + ',' + color[2] + ',' + alpha + ')';

      ctx.save();
      ctx.strokeStyle = style;
      ctx.fillStyle = style;
      ctx.lineWidth = typeof prim.width === 'number' && prim.width > 0 ? prim.width : 2;
      ctx.setLineDash(prim.dash ? [8, 6] : []);
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';

      switch (prim.kind) {
        case 'polygon':
          if (pts.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
            ctx.closePath();
            if (prim.fill) ctx.fill();
            ctx.stroke();
          }
          break;

        case 'polyline':
          if (pts.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (var j = 1; j < pts.length; j++) ctx.lineTo(pts[j][0], pts[j][1]);
            ctx.stroke();
          }
          break;

        case 'circle':
          if (pts.length >= 1) {
            var r = typeof prim.width === 'number' && prim.width > 0 ? prim.width : 10;
            if (pts.length >= 2) {
              var dx = pts[1][0] - pts[0][0];
              var dy = pts[1][1] - pts[0][1];
              r = Math.sqrt(dx * dx + dy * dy) || r;
            }
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(pts[0][0], pts[0][1], Math.max(1, r), 0, Math.PI * 2);
            if (prim.fill) ctx.fill();
            ctx.stroke();
          }
          break;

        case 'arrow':
          if (pts.length >= 2) {
            var x0 = pts[0][0];
            var y0 = pts[0][1];
            var x1 = pts[pts.length - 1][0];
            var y1 = pts[pts.length - 1][1];
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            for (var k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0], pts[k][1]);
            ctx.stroke();
            var ang = Math.atan2(y1 - y0, x1 - x0);
            var headLen = Math.max(8, ctx.lineWidth * 3);
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x1 - headLen * Math.cos(ang - Math.PI / 6), y1 - headLen * Math.sin(ang - Math.PI / 6));
            ctx.moveTo(x1, y1);
            ctx.lineTo(x1 - headLen * Math.cos(ang + Math.PI / 6), y1 - headLen * Math.sin(ang + Math.PI / 6));
            ctx.stroke();
          }
          break;

        case 'label':
        case 'chip':
          if (pts.length >= 1) {
            var x = pts[0][0];
            var y = pts[0][1];
            var text = prim.text || '';
            ctx.setLineDash([]);
            ctx.font = '600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
            ctx.textBaseline = 'alphabetic';
            var metrics = ctx.measureText(text);
            var padX = 6;
            var padY = 4;
            var boxW = metrics.width + padX * 2;
            var boxH = 16 + padY;
            var rx = prim.kind === 'chip' ? boxH / 2 : 3;
            ctx.beginPath();
            roundRectPath(ctx, x, y, boxW, boxH, rx);
            if (prim.fill !== false) {
              var prevAlpha = ctx.globalAlpha;
              ctx.globalAlpha = alpha * 0.85;
              ctx.fill();
              ctx.globalAlpha = prevAlpha;
            }
            ctx.stroke();
            ctx.fillStyle = 'rgba(255,255,255,' + alpha + ')';
            ctx.fillText(text, x + padX, y + boxH - padY - 3);
          }
          break;

        default:
          break;
      }
      ctx.restore();
    }

    function computeNow() {
      if (video && !video.paused && !video.seeking && isFinite(video.currentTime)) {
        return video.currentTime;
      }
      return newMsg ? newMsg.t : 0;
    }

    function isStale() {
      return !newMsg || performance.now() - lastMsgWallMs > STALE_MS;
    }

    function frame() {
      if (!running) return;
      rafHandle = requestAnimationFrame(frame);

      if (isStale()) {
        clear();
        return;
      }

      clear();

      var now = computeNow();
      var tNew = newMsg.t;
      var tPrev = prevMsg ? prevMsg.t : null;
      var f = 0;
      if (tPrev !== null && tNew > tPrev) {
        var dtReply = tNew - tPrev;
        var dtNow = Math.min(Math.max(now - tNew, 0), MAX_EXTRAPOLATE_S);
        f = dtNow / dtReply;
        f = Math.min(Math.max(f, 0), 1);
      }

      newMsg.byGid.forEach(function (prim, gid) {
        var prevPrim = prevMsg ? prevMsg.byGid.get(gid) : null;
        var pts = prevPrim ? lerpPts(prevPrim.pts, prim.pts, f) : prim.pts;
        var toDraw = prim;
        if (pts !== prim.pts) {
          toDraw = Object.assign({}, prim, { pts: pts });
        }
        drawPrim(toDraw, 1);
      });

      var nowMs = performance.now();
      vanished.forEach(function (v, gid) {
        var age = nowMs - v.sinceMs;
        if (age > FADE_MS) {
          vanished.delete(gid);
          return;
        }
        drawPrim(v.prim, 1 - age / FADE_MS);
      });
    }

    function start() {
      if (running) return;
      running = true;
      rafHandle = requestAnimationFrame(frame);
    }

    function stop() {
      running = false;
      if (rafHandle !== null) cancelAnimationFrame(rafHandle);
      rafHandle = null;
      clear();
    }

    return {
      setVideo: setVideo,
      pushGeometry: pushGeometry,
      start: start,
      stop: stop,
      clear: clear,
      isStale: isStale,
    };
  }

  window.BTODraw = { create: create };
})();
