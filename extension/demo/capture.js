/**
 * BTO Telestrator — capture.js
 *
 * Classic script (no ES modules) so it can be loaded both as a Chrome
 * extension content script and directly via <script src> on the plain
 * demo page (extension/demo/player.html). Exposes a single global,
 * `window.BTOCapture`.
 *
 * Responsibilities (SPEC C8 / S4.1, frozen WS protocol):
 *  - Grab frames from a <video> element via an offscreen canvas.
 *  - Downscale the longest side to 960px, encode JPEG q=0.7.
 *  - Self-clock to the host: only send a new frame once the socket is
 *    OPEN *and* the previous reply has arrived (no unbounded queueing).
 *  - Frame the outgoing binary message per the frozen header layout:
 *    12 bytes = uint32 LE seq (offset 0) + float64 LE t_video_sec (offset 4).
 *  - Detect DRM/protected video (drawImage/toBlob throws, or the sampled
 *    frame is all-black) and stop sending, surfacing a 'drm' status.
 *
 * NOTE: extension/demo/capture.js is a byte-identical copy of this file
 * (the local inference host's StaticFiles mount serves extension/demo/ at
 * "/" and does not follow symlinks outside that directory, so the demo
 * page needs its own copy on disk). Keep the two in sync.
 */
(function () {
  'use strict';

  var HEADER_BYTES = 12; // uint32 LE seq (4) + float64 LE t_video_sec (8)
  var SEND_LONG_SIDE = 960;
  var JPEG_QUALITY = 0.7;
  var BLACK_CHECK_EVERY = 50; // frames
  var BLACK_CHECK_SIZE = 16;
  var BLACK_SUM_THRESHOLD = 4; // mean per-channel value below this => "all black"
  var RECONNECT_DELAY_MS = 1000;
  var MAX_DIMS_TRACKED = 64;

  /**
   * @param {Object} opts
   * @param {HTMLVideoElement} opts.video
   * @param {string} opts.wsUrl
   * @param {(status: 'connecting'|'live'|'drm'|'host-down') => void} [opts.onStatus]
   * @param {(msg: Object, frameDims: {w:number,h:number}|null) => void} [opts.onMessage]
   * @param {() => void} [opts.onDrm]
   */
  function create(opts) {
    var video = opts.video;
    var wsUrl = opts.wsUrl;
    var onStatus = opts.onStatus || function () {};
    var onMessage = opts.onMessage || function () {};
    var onDrm = opts.onDrm || function () {};

    var ws = null;
    var seq = 0;
    var awaitingReply = false;
    var stopped = true;
    var drm = false;
    var frameCounter = 0;
    var rvfcHandle = null;
    var rafHandle = null;
    var reconnectTimer = null;
    var frameDimsBySeq = new Map(); // seq -> {w,h} of the frame actually sent

    var offCanvas = document.createElement('canvas');
    var offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
    var blackCanvas = document.createElement('canvas');
    blackCanvas.width = BLACK_CHECK_SIZE;
    blackCanvas.height = BLACK_CHECK_SIZE;
    var blackCtx = blackCanvas.getContext('2d', { willReadFrequently: true });

    function connect() {
      if (stopped || drm) return;
      onStatus('connecting');
      try {
        ws = new WebSocket(wsUrl);
      } catch (e) {
        ws = null;
        onStatus('host-down');
        scheduleReconnect();
        return;
      }
      ws.binaryType = 'arraybuffer';

      ws.addEventListener('open', function () {
        awaitingReply = false;
        onStatus('live');
      });

      ws.addEventListener('message', function (ev) {
        awaitingReply = false;
        var msg;
        try {
          msg = JSON.parse(ev.data);
        } catch (e) {
          return;
        }
        var dims = null;
        if (typeof msg.seq === 'number') {
          dims = frameDimsBySeq.get(msg.seq) || null;
          // Prune everything up to and including this seq — the host
          // processes latest-only, so older in-flight dims are moot.
          frameDimsBySeq.forEach(function (_v, k) {
            if (k <= msg.seq) frameDimsBySeq.delete(k);
          });
        }
        onMessage(msg, dims);
      });

      ws.addEventListener('close', function () {
        awaitingReply = false;
        ws = null;
        if (!stopped && !drm) {
          onStatus('host-down');
          scheduleReconnect();
        }
      });

      ws.addEventListener('error', function () {
        // 'close' always follows 'error' for WebSocket; nothing to do here.
      });
    }

    function scheduleReconnect() {
      if (reconnectTimer || stopped || drm) return;
      reconnectTimer = setTimeout(function () {
        reconnectTimer = null;
        connect();
      }, RECONNECT_DELAY_MS);
    }

    function checkAllBlack() {
      try {
        blackCtx.drawImage(video, 0, 0, BLACK_CHECK_SIZE, BLACK_CHECK_SIZE);
        var data = blackCtx.getImageData(0, 0, BLACK_CHECK_SIZE, BLACK_CHECK_SIZE).data;
        var sum = 0;
        for (var i = 0; i < data.length; i += 4) {
          sum += data[i] + data[i + 1] + data[i + 2];
        }
        var avg = sum / (BLACK_CHECK_SIZE * BLACK_CHECK_SIZE * 3);
        return avg < BLACK_SUM_THRESHOLD;
      } catch (e) {
        // SecurityError (tainted canvas) => treat as DRM signal.
        return 'error';
      }
    }

    function triggerDrm() {
      if (drm) return;
      drm = true;
      stopLoops();
      closeSocket();
      onStatus('drm');
      onDrm();
    }

    function sendFrame(tVideo) {
      if (stopped || drm) return;
      if (!ws || ws.readyState !== WebSocket.OPEN || awaitingReply) return;
      var vw = video.videoWidth;
      var vh = video.videoHeight;
      if (!vw || !vh) return;

      frameCounter++;
      if (frameCounter % BLACK_CHECK_EVERY === 0) {
        var blackResult = checkAllBlack();
        if (blackResult === 'error' || blackResult === true) {
          triggerDrm();
          return;
        }
      }

      var scale = Math.min(1, SEND_LONG_SIDE / Math.max(vw, vh));
      var dw = Math.max(1, Math.round(vw * scale));
      var dh = Math.max(1, Math.round(vh * scale));
      if (offCanvas.width !== dw || offCanvas.height !== dh) {
        offCanvas.width = dw;
        offCanvas.height = dh;
      }
      try {
        offCtx.drawImage(video, 0, 0, dw, dh);
      } catch (e) {
        triggerDrm();
        return;
      }

      var curSeq = seq++;
      var t = typeof tVideo === 'number' ? tVideo : video.currentTime;
      awaitingReply = true;
      frameDimsBySeq.set(curSeq, { w: dw, h: dh });
      if (frameDimsBySeq.size > MAX_DIMS_TRACKED) {
        var oldestKey = frameDimsBySeq.keys().next().value;
        frameDimsBySeq.delete(oldestKey);
      }

      var finish = function (blob) {
        if (!blob) {
          awaitingReply = false;
          return;
        }
        blob.arrayBuffer().then(function (buf) {
          if (stopped || drm || !ws || ws.readyState !== WebSocket.OPEN) {
            awaitingReply = false;
            return;
          }
          var header = new ArrayBuffer(HEADER_BYTES);
          var dv = new DataView(header);
          dv.setUint32(0, curSeq >>> 0, true); // LE uint32 seq
          dv.setFloat64(4, t, true); // LE float64 t_video_sec
          var out = new Uint8Array(HEADER_BYTES + buf.byteLength);
          out.set(new Uint8Array(header), 0);
          out.set(new Uint8Array(buf), HEADER_BYTES);
          try {
            ws.send(out.buffer);
          } catch (e) {
            awaitingReply = false;
          }
        }, function () {
          awaitingReply = false;
        });
      };

      try {
        offCanvas.toBlob(finish, 'image/jpeg', JPEG_QUALITY);
      } catch (e) {
        awaitingReply = false;
        triggerDrm();
      }
    }

    function loopRvfc(_now, metadata) {
      if (stopped) return;
      var t = metadata && typeof metadata.mediaTime === 'number' ? metadata.mediaTime : video.currentTime;
      sendFrame(t);
      rvfcHandle = video.requestVideoFrameCallback(loopRvfc);
    }

    function loopRaf() {
      if (stopped) return;
      sendFrame(video.currentTime);
      rafHandle = requestAnimationFrame(loopRaf);
    }

    function startLoops() {
      if (typeof video.requestVideoFrameCallback === 'function') {
        rvfcHandle = video.requestVideoFrameCallback(loopRvfc);
      } else {
        rafHandle = requestAnimationFrame(loopRaf);
      }
    }

    function stopLoops() {
      if (rvfcHandle !== null && typeof video.cancelVideoFrameCallback === 'function') {
        try {
          video.cancelVideoFrameCallback(rvfcHandle);
        } catch (e) {
          /* ignore */
        }
      }
      rvfcHandle = null;
      if (rafHandle !== null) cancelAnimationFrame(rafHandle);
      rafHandle = null;
    }

    function closeSocket() {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws) {
        try {
          ws.close();
        } catch (e) {
          /* ignore */
        }
      }
      ws = null;
    }

    function start() {
      if (!stopped) return;
      stopped = false;
      drm = false;
      connect();
      startLoops();
    }

    function stop() {
      stopped = true;
      stopLoops();
      closeSocket();
    }

    return {
      start: start,
      stop: stop,
      isDrm: function () {
        return drm;
      },
    };
  }

  window.BTOCapture = { create: create, HEADER_BYTES: HEADER_BYTES };
})();
