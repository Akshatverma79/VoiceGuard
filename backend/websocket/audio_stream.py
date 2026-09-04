"""
VoiceGuard — backend/websocket/audio_stream.py
WebSocket handler for real-time audio streaming and AASIST inference.

Flow per connection
───────────────────
1.  Accept WebSocket connection, send "connected" status.
2.  Wait for JSON config message: { "type": "config", "sample_rate": <int> }
3.  Create a per-connection AudioBuffer at the reported native sample rate.
4.  Loop: receive messages
    a.  Binary frame  → push raw bytes into AudioBuffer
    b.  JSON "stop"   → break loop cleanly
    c.  Other JSON    → log and ignore
5.  When AudioBuffer.should_process() is True:
    a.  Consume samples from buffer
    b.  Convert list[float] → torch.Tensor (1, N)
    c.  Resample from native_sr → 16 000 Hz using torchaudio
    d.  Normalize (peak)
    e.  Run VAD — if silent, send "waiting_for_speech" status and continue
    f.  Chunk into AASIST windows (AudioChunker)
    g.  For each chunk: run predict_waveform() in thread executor
    h.  Aggregate chunk scores
    i.  Update RiskEngine
    j.  Send detection result JSON
6.  On disconnect (WebSocketDisconnect): clean up buffer, log.

AASIST inference runs in asyncio.run_in_executor(None, ...) so the
event loop and other WebSocket connections are not blocked.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from fastapi import WebSocket, WebSocketDisconnect

# ── Path setup ─────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent.parent.resolve()
_ROOT_DIR = _BACKEND_DIR.parent.resolve()
_ML_DIR = _ROOT_DIR / "ml"
for _p in [str(_ML_DIR), str(_ROOT_DIR), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audio.vad import VoiceActivityDetector
from audio.chunker import AudioChunker
from services.model_manager import ModelManager
from services.risk_engine import RiskEngine, RiskLevel
from websocket.audio_buffer import AudioBuffer

# ── Verbose performance logging (dev only) ─────────────────────────────────
VERBOSE_PERF: bool = True   # set False to suppress perf messages in production

# ── VAD & chunker (shared, stateless — safe to share across connections) ───
_vad = VoiceActivityDetector()
_chunker = AudioChunker(overlap_s=0.0)   # no overlap for streaming (lower latency)

# ── Helpers ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _send(ws: WebSocket, payload: Dict[str, Any]) -> None:
    """Send JSON safely; silently ignore if connection is already closed."""
    try:
        await ws.send_json(payload)
    except Exception:
        pass


def _samples_to_tensor(samples: List[float], native_sr: int) -> torch.Tensor:
    """
    Convert a list of float32 PCM samples at native_sr → 16 kHz mono tensor.
    Returns shape (1, N) float32.
    """
    import torchaudio

    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)  # (1, N)

    # Resample to 16 kHz if needed
    if native_sr != 16_000:
        resampler = torchaudio.transforms.Resample(orig_freq=native_sr, new_freq=16_000)
        waveform = resampler(waveform)

    # Peak normalize (same as AudioProcessor.normalize)
    peak = waveform.abs().max().item()
    if peak > 1e-6:
        waveform = waveform / peak

    return waveform.float()


def _aggregate_scores(scores: List[float]) -> float:
    """Mean aggregation of per-chunk spoof probabilities."""
    return sum(scores) / len(scores) if scores else 0.0


# ── Main WebSocket handler ──────────────────────────────────────────────────

async def handle_audio_stream(websocket: WebSocket) -> None:
    """
    Handle one WebSocket connection for real-time audio streaming.
    Called by the FastAPI route in main.py.
    """
    await websocket.accept()
    await _send(websocket, {"type": "status", "status": "connected"})

    # ── Wait for config ────────────────────────────────────────────────────
    native_sr: int = 44_100   # default; overridden by client config message
    initial_bytes: Optional[bytes] = None
    try:
        raw = await asyncio.wait_for(websocket.receive(), timeout=10.0)
        if raw.get("type") == "websocket.receive":
            if raw.get("text"):
                try:
                    msg = json.loads(raw["text"])
                    if msg.get("type") == "config":
                        native_sr = int(msg.get("sample_rate", 44_100))
                except Exception:
                    await _send(websocket, {"type": "error", "message": "Invalid JSON in config message"})
            elif raw.get("bytes"):
                initial_bytes = raw["bytes"]
    except asyncio.TimeoutError:
        pass   # use default sample rate

    buffer = AudioBuffer(native_sample_rate=native_sr)
    if initial_bytes:
        buffer.push_bytes(initial_bytes)
    risk = RiskEngine()

    await _send(websocket, {
        "type": "status",
        "status": "listening",
        "native_sample_rate": native_sr,
    })

    # ── Main receive loop ──────────────────────────────────────────────────
    try:
        while True:
            # ── Receive one WebSocket message ──────────────────────────────
            try:
                raw = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await _send(websocket, {"type": "status", "status": "timeout"})
                break

            msg_type = raw.get("type", "")

            if msg_type == "websocket.disconnect":
                break

            # Binary frame → audio samples
            if raw.get("bytes") is not None:
                buffer.push_bytes(raw["bytes"])

            # JSON control message
            elif raw.get("text"):
                try:
                    ctrl = json.loads(raw["text"])
                    if ctrl.get("type") == "stop":
                        break
                except Exception:
                    await _send(websocket, {"type": "error", "message": "Invalid JSON message"})

            # ── Check if buffer is ready for processing ────────────────────
            if not buffer.should_process():
                continue

            t_total_start = time.perf_counter()

            # ── Step 1: Consume buffer ────────────────────────────────────
            samples = buffer.consume()
            buf_seconds = len(samples) / native_sr

            # ── Step 2: Convert to tensor at 16 kHz ──────────────────────
            t_pre_start = time.perf_counter()
            try:
                waveform = _samples_to_tensor(samples, native_sr)
            except Exception as exc:
                await _send(websocket, {"type": "error", "message": f"Preprocessing failed: {exc}"})
                continue
            t_pre_ms = round((time.perf_counter() - t_pre_start) * 1000, 1)

            # ── Step 3: VAD ───────────────────────────────────────────────
            vad_result = _vad.detect(waveform)
            if not vad_result.has_speech:
                await _send(websocket, {
                    "type": "status",
                    "status": "waiting_for_speech",
                    "speech_ratio": vad_result.speech_ratio,
                })
                continue

            # ── Step 4: Chunk ─────────────────────────────────────────────
            chunks = _chunker.chunk(waveform)

            # ── Step 5: AASIST inference (in executor — non-blocking) ─────
            await _send(websocket, {"type": "status", "status": "processing"})

            loop = asyncio.get_event_loop()
            detector = ModelManager.get_detector()

            chunk_scores: List[float] = []
            chunk_times: List[float] = []

            for chunk in chunks:
                chunk_ref = chunk   # capture for lambda
                try:
                    result = await loop.run_in_executor(
                        None, detector.predict_waveform, chunk_ref
                    )
                    chunk_scores.append(result["spoof_probability"])
                    chunk_times.append(result["inference_time_s"])
                except Exception as exc:
                    await _send(websocket, {
                        "type": "error",
                        "message": f"Inference error on chunk: {exc}",
                    })

            if not chunk_scores:
                continue

            # ── Step 6: Aggregate + risk ──────────────────────────────────
            agg_score = _aggregate_scores(chunk_scores)
            risk_result = risk.update(agg_score)

            total_inference_ms = round(sum(chunk_times) * 1000, 1)
            total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
            prediction = "spoof" if agg_score > 0.5 else "real"

            # ── Step 7: Send detection result ─────────────────────────────
            await _send(websocket, {
                "type": "detection",
                "spoof_probability": round(agg_score, 4),
                "prediction": prediction,
                "risk_score": risk_result.risk_score,
                "risk_level": risk_result.risk_level.value,
                "chunks_analyzed": len(chunk_scores),
                "history": risk_result.history,
                "timestamp": _now_iso(),
            })

            # ── Step 8: Optional perf message ────────────────────────────
            if VERBOSE_PERF:
                await _send(websocket, {
                    "type": "perf",
                    "buffer_s": round(buf_seconds, 2),
                    "preprocess_ms": t_pre_ms,
                    "inference_ms": total_inference_ms,
                    "total_ms": total_ms,
                    "chunks": len(chunk_scores),
                })

    except WebSocketDisconnect:
        pass   # client disconnected cleanly
    except Exception as exc:
        await _send(websocket, {"type": "error", "message": str(exc)})
    finally:
        buffer.clear()
