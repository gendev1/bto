"""SPEC C10: local inference host — FastAPI + WebSocket transport.

Implements the FROZEN WS PROTOCOL on ws://127.0.0.1:8517/stream:

client -> host: ONE binary message per frame:
    12-byte header (uint32 LE seq, float64 LE t_video_sec) + JPEG bytes.
    The client throttles (sends the next frame only after the previous
    reply), but this host ALSO drops stale frames if more than one is
    queued when the processor is ready (process latest only; tracked via
    stats.dropped).

host -> client: JSON text:
    {"seq": int, "t": float, "proc_ms": float, "shot": "main"|"other",
     "drm": false,
     "geometry": [PRIM, ...],
     "stats": {"fps_in": f, "fps_proc": f, "n_tracks": int,
               "calib_src": "fit|ema|held|null", "dropped": int}}

PRIM (pixel space of the sent frame):
    {"gid": str, "kind": "polygon"|"polyline"|"circle"|"arrow"|"label"|"chip",
     "pts": [[x,y], ...], "color": [r,g,b], "alpha": f, "width": f,
     "text": str|None, "fill": bool, "dash": bool}

/health -> {"ok": true, "models_loaded": bool}
/         -> StaticFiles mount serving extension/demo/ (built by the
             extension sibling module; this server only mounts the dir).

Perception contract this module expects from the sibling `bto.host.stream`
/ `bto.host.primitives` modules (not built by this file):

    from bto.host.stream import StreamPipeline
    from bto.host.primitives import detections_to_prims

    pipeline = StreamPipeline(imgsz=..., device=...)
    result = pipeline.process(frame_bgr_ndarray, t_video_sec)
        # result is a dict-like with (at least) keys:
        #   "shot": "main"|"other"
        #   "n_tracks": int
        #   "calib_src": "fit"|"ema"|"held"|"null"
        #   "H": 3x3 (or 9-float) homography or None
        #   "detections": list[bto.core.Detection], fed to detections_to_prims
        # (result may also directly carry "geometry": [PRIM,...], in which
        # case detections_to_prims is skipped)
    prims = detections_to_prims({"detections": result["detections"], "H": result["H"],
                                  "t": t_video_sec, "w": frame_w, "h": frame_h})

If those sibling modules are not importable, this file falls back to an
in-process stub pipeline (see `_StubPipeline` / `_stub_detections_to_prims`
below) so the server + WS protocol can still be started, exercised and
self-checked end-to-end. Swap-in is automatic: once the real modules land
on disk, this file imports them with no code changes needed here.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import struct
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("bto.host.server")

HEADER_FMT = "<Id"  # uint32 LE seq, float64 LE t_video_sec
HEADER_LEN = struct.calcsize(HEADER_FMT)  # 12
EXTENSION_DEMO_DIR = Path(__file__).resolve().parents[2] / "extension" / "demo"

# ---------------------------------------------------------------------------
# Real pipeline (sibling modules) with stub fallback.
# ---------------------------------------------------------------------------

try:
    from bto.host.stream import StreamPipeline as _RealStreamPipeline  # type: ignore
except Exception:  # pragma: no cover - exercised when sibling not present yet
    _RealStreamPipeline = None

try:
    from bto.host.primitives import detections_to_prims as _real_detections_to_prims  # type: ignore
except Exception:  # pragma: no cover
    _real_detections_to_prims = None

STREAM_MODULES_AVAILABLE = _RealStreamPipeline is not None and _real_detections_to_prims is not None


class _StubPipeline:
    """Fallback used only when bto.host.stream is not yet importable.

    Mimics the expected StreamPipeline interface closely enough to exercise
    the WS protocol end-to-end (frame decode, reply shape, stats) without
    real detection/calibration. Never used if the real module is present.
    """

    def __init__(self, imgsz: int = 960, device: str = "auto") -> None:
        self.imgsz = imgsz
        self.device = device
        self.models_loaded = True  # stub has nothing to lazily load

    def process(self, frame: np.ndarray, t: float) -> dict:
        h, w = frame.shape[:2]
        return {
            "shot": "main",
            "n_tracks": 0,
            "calib_src": "null",
            "H": None,
            "detections": [],
        }


def _stub_detections_to_prims(bundle: Any) -> list[dict]:
    return []


def _build_pipeline(imgsz: int, device: str, **pipeline_kw):
    if STREAM_MODULES_AVAILABLE:
        return (
            _RealStreamPipeline(imgsz=imgsz, device=device, **pipeline_kw),
            _real_detections_to_prims,
        )
    log.warning(
        "bto.host.stream / bto.host.primitives not importable yet; "
        "running with a stub pipeline (no real detections)."
    )
    return _StubPipeline(imgsz=imgsz, device=device), _stub_detections_to_prims


def _call_process(pipeline, frame: np.ndarray, t: float) -> dict:
    """Call pipeline.process(frame, t) tolerating a (frame,) signature too."""
    try:
        sig = inspect.signature(pipeline.process)
        n_params = len(
            [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        )
    except (TypeError, ValueError):
        n_params = 2
    if n_params >= 2:
        result = pipeline.process(frame, t)
    else:
        result = pipeline.process(frame)
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    # dataclass / object with attributes
    return {
        "shot": getattr(result, "shot", "other"),
        "n_tracks": getattr(result, "n_tracks", 0),
        "calib_src": getattr(result, "calib_src", "null"),
        "detections": getattr(result, "detections", result),
        "geometry": getattr(result, "geometry", None),
    }


def _call_detections_to_prims(fn, result: dict, t: float, w: int, h: int) -> list[dict]:
    """Bundle-call convention: fn({"detections","H","t","w","h"}). This
    matches bto.host.primitives.detections_to_prims's documented
    stub-tolerant contract (also accepts the plain fn(detections, H, t, w, h)
    signature, but the bundle form lets this server stay agnostic of the
    exact positional order)."""
    if "geometry" in result and result["geometry"] is not None:
        return result["geometry"]
    bundle = {
        "detections": result.get("detections"),
        "H": result.get("H"),
        "t": t,
        "w": w,
        "h": h,
    }
    try:
        prims = fn(bundle)
    except Exception:
        log.exception("detections_to_prims failed; returning empty geometry")
        return []
    return prims or []


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------


class HostState:
    """One pipeline per process; at most one active /stream connection.

    8GB VRAM budget only holds player-detect + pitch-pose models resident,
    so concurrent streams are rejected rather than queued/shared.
    """

    def __init__(self, imgsz: int, device: str, pipeline_kw: dict | None = None) -> None:
        self.imgsz = imgsz
        self.device = device
        self.pipeline_kw = pipeline_kw or {}
        self.pipeline = None
        self._detections_to_prims = None
        self._stream_lock = asyncio.Lock()
        self.streaming = False

    def ensure_pipeline(self):
        if self.pipeline is None:
            self.pipeline, self._detections_to_prims = _build_pipeline(
                self.imgsz, self.device, **self.pipeline_kw
            )
        return self.pipeline, self._detections_to_prims

    @property
    def models_loaded(self) -> bool:
        return self.pipeline is not None


def create_app(imgsz: int = 960, device: str = "auto", pipeline_kw: dict | None = None) -> FastAPI:
    state = HostState(imgsz=imgsz, device=device, pipeline_kw=pipeline_kw)
    app = FastAPI()
    app.state.bto_host = state

    @app.get("/health")
    async def health():
        return {"ok": True, "models_loaded": state.models_loaded}

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()

        async with state._stream_lock:
            if state.streaming:
                await ws.send_text(json.dumps({"error": "stream already active"}))
                await ws.close()
                return
            state.streaming = True

        try:
            pipeline, to_prims = state.ensure_pipeline()

            queue: asyncio.Queue = asyncio.Queue(maxsize=1)
            dropped = {"n": 0}
            recv_times: list[float] = []

            async def reader():
                try:
                    while True:
                        data = await ws.receive_bytes()
                        if len(data) < HEADER_LEN:
                            continue
                        seq, t_video = struct.unpack(HEADER_FMT, data[:HEADER_LEN])
                        jpeg_bytes = data[HEADER_LEN:]
                        item = (seq, t_video, jpeg_bytes, time.monotonic())
                        if queue.full():
                            try:
                                queue.get_nowait()
                                dropped["n"] += 1
                            except asyncio.QueueEmpty:
                                pass
                        await queue.put(item)
                finally:
                    # Client went away (or reader died): wake the processor loop
                    # with a sentinel or it blocks on queue.get() forever and
                    # state.streaming leaks True, rejecting every reconnect.
                    while queue.full():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    queue.put_nowait(None)

            reader_task = asyncio.create_task(reader())
            last_proc_end = None
            try:
                while True:
                    item = await queue.get()
                    if item is None:  # reader died: client disconnected
                        break
                    seq, t_video, jpeg_bytes, recv_t = item
                    recv_times.append(recv_t)
                    if len(recv_times) > 30:
                        recv_times.pop(0)
                    fps_in = 0.0
                    if len(recv_times) >= 2:
                        span = recv_times[-1] - recv_times[0]
                        if span > 0:
                            fps_in = (len(recv_times) - 1) / span

                    t0 = time.monotonic()
                    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    shot = "other"
                    n_tracks = 0
                    calib_src = "null"
                    geometry: list[dict] = []
                    if frame is not None:
                        fh, fw = frame.shape[:2]
                        result = _call_process(pipeline, frame, t_video)
                        shot = result.get("shot", "other")
                        n_tracks = result.get("n_tracks", 0)
                        calib_src = result.get("calib_src", "null")
                        geometry = _call_detections_to_prims(to_prims, result, t_video, fw, fh)

                    proc_ms = (time.monotonic() - t0) * 1000.0

                    fps_proc = 0.0
                    now = time.monotonic()
                    if last_proc_end is not None:
                        dt = now - last_proc_end
                        if dt > 0:
                            fps_proc = 1.0 / dt
                    last_proc_end = now

                    reply = {
                        "seq": seq,
                        "t": t_video,
                        "proc_ms": proc_ms,
                        "shot": shot,
                        "drm": False,
                        "geometry": geometry,
                        "stats": {
                            "fps_in": round(fps_in, 2),
                            "fps_proc": round(fps_proc, 2),
                            "n_tracks": n_tracks,
                            "calib_src": calib_src,
                            "dropped": dropped["n"],
                        },
                    }
                    await ws.send_text(json.dumps(reply))
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            state.streaming = False

    if EXTENSION_DEMO_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(EXTENSION_DEMO_DIR), html=True), name="demo")
    else:
        log.warning("extension/demo dir not found at %s; static mount skipped", EXTENSION_DEMO_DIR)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bto.host.server")
    parser.add_argument("--port", type=int, default=8517)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--player-model", type=str, default=None,
                        help=".pt or TensorRT .engine path (CUDA deploy)")
    parser.add_argument("--pitch-model", type=str, default=None)
    parser.add_argument("--calib-every", type=int, default=None,
                        help="fit homography every Nth main frame (2 on CUDA)")
    args = parser.parse_args()

    import uvicorn

    pipeline_kw = {
        k: v
        for k, v in {
            "player_model": args.player_model,
            "pitch_model": args.pitch_model,
            "calib_every": args.calib_every,
        }.items()
        if v is not None
    }
    app = create_app(imgsz=args.imgsz, device=args.device, pipeline_kw=pipeline_kw)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
