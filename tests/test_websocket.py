"""
VoiceGuard — tests/test_websocket.py
Phase 3 test suite: AudioBuffer, RiskEngine, ModelManager, and WebSocket endpoint.
"""

import math
import struct
import sys
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).parent.parent.resolve()
_BACKEND = _ROOT / "backend"
_ML = _ROOT / "ml"
for p in [str(_ROOT), str(_BACKEND), str(_ML)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.websocket.audio_buffer import AudioBuffer
from backend.services.risk_engine import (
    RiskEngine,
    RiskLevel,
    RiskResult,
    EWMA_ALPHA,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
    MIN_CHUNKS_FOR_HIGH,
)
from backend.services.model_manager import ModelManager


# ==============================================================================
# AudioBuffer Tests
# ==============================================================================

class TestAudioBuffer:
    def test_initial_state(self):
        buf = AudioBuffer(native_sample_rate=16000, process_seconds=4.0, overlap_seconds=1.0)
        assert buf.native_sample_rate == 16000
        assert buf.buffered_samples == 0
        assert buf.buffered_seconds == 0.0
        assert not buf.should_process()

    def test_push_bytes(self):
        buf = AudioBuffer(native_sample_rate=16000, process_seconds=1.0, overlap_seconds=0.25)
        # 8000 samples = 0.5s
        samples = np.linspace(-1.0, 1.0, 8000, dtype=np.float32)
        raw_bytes = samples.tobytes()
        buf.push_bytes(raw_bytes)

        assert buf.buffered_samples == 8000
        assert pytest.approx(buf.buffered_seconds, 0.01) == 0.5
        assert not buf.should_process()

    def test_ready_and_consume(self):
        buf = AudioBuffer(native_sample_rate=16000, process_seconds=1.0, overlap_seconds=0.25)
        # 16000 samples = 1.0s
        samples = np.ones(16000, dtype=np.float32) * 0.5
        buf.push_bytes(samples.tobytes())

        assert buf.should_process()
        extracted = buf.consume()
        assert len(extracted) == 16000
        np.testing.assert_allclose(extracted, 0.5, atol=1e-5)

        # After consume, overlap is kept (0.25s = 4000 samples)
        assert buf.buffered_samples == 4000
        assert pytest.approx(buf.buffered_seconds, 0.01) == 0.25
        assert not buf.should_process()

    def test_buffer_overflow_cap(self):
        # max_seconds = 2.0s -> 32000 samples
        buf = AudioBuffer(native_sample_rate=16000, max_seconds=2.0)
        large_samples = np.zeros(48000, dtype=np.float32)
        buf.push_bytes(large_samples.tobytes())

        # Buffer should be trimmed to max_samples
        assert buf.buffered_samples <= 32000

    def test_clear(self):
        buf = AudioBuffer(native_sample_rate=16000)
        buf.push_bytes(np.ones(1000, dtype=np.float32).tobytes())
        assert buf.buffered_samples > 0
        buf.clear()
        assert buf.buffered_samples == 0
        assert buf.buffered_seconds == 0.0


# ==============================================================================
# RiskEngine Tests
# ==============================================================================

class TestRiskEngine:
    def test_initial_state(self):
        engine = RiskEngine()
        assert engine.current_score == 0.0
        assert engine.chunks_seen == 0
        assert len(engine.history) == 0

    def test_first_score_sets_baseline(self):
        engine = RiskEngine()
        score = 0.20
        res = engine.update(score)

        assert isinstance(res, RiskResult)
        assert res.risk_score == pytest.approx(0.20, 0.001)
        # 1 chunk < MIN_CHUNKS_FOR_HIGH (2), so classify gives UNKNOWN
        assert res.risk_level == RiskLevel.UNKNOWN
        assert res.chunks_seen == 1
        assert engine.current_score == pytest.approx(0.20, 0.001)

    def test_ewma_decay(self):
        engine = RiskEngine(alpha=0.4, min_chunks_for_high=1)
        # Chunk 1: 0.8
        engine.update(0.8)
        # Chunk 2: 0.2
        # Expected: 0.4 * 0.2 + (1 - 0.4) * 0.8 = 0.08 + 0.48 = 0.56
        res2 = engine.update(0.2)
        assert pytest.approx(res2.risk_score, 0.001) == 0.56
        assert res2.chunks_seen == 2

    def test_min_chunks_for_high_risk(self):
        engine = RiskEngine(min_chunks_for_high=2)
        # Chunk 1: high score, but chunks_seen < 2 -> UNKNOWN
        res1 = engine.update(0.95)
        assert res1.risk_level == RiskLevel.UNKNOWN

        # Chunk 2: high score, chunks_seen == 2 -> HIGH
        res2 = engine.update(0.90)
        assert res2.risk_level == RiskLevel.HIGH

    def test_risk_levels(self):
        engine = RiskEngine(min_chunks_for_high=1)

        # Low (< 0.35)
        res_low = engine.update(0.10)
        assert res_low.risk_level == RiskLevel.LOW

        # Reset & Medium (0.35 <= x < 0.65)
        engine.reset()
        res_med = engine.update(0.50)
        assert res_med.risk_level == RiskLevel.MEDIUM

    def test_history_cap(self):
        engine = RiskEngine(window_max=5)
        for i in range(10):
            engine.update(0.5)
        assert len(engine.history) == 5

    def test_reset(self):
        engine = RiskEngine()
        engine.update(0.8)
        assert engine.chunks_seen == 1
        engine.reset()
        assert engine.chunks_seen == 0
        assert engine.current_score == 0.0
        assert len(engine.history) == 0


# ==============================================================================
# ModelManager Tests
# ==============================================================================

class TestModelManager:
    def test_model_manager_is_ready_check(self):
        # Verify is_ready returns bool
        ready = ModelManager.is_ready()
        assert isinstance(ready, bool)


# ==============================================================================
# WebSocket Endpoint Tests
# ==============================================================================

class TestWebSocketEndpoint:
    def test_websocket_connection_and_config(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app) as client:
            with client.websocket_connect("/ws/audio") as websocket:
                # 1. First message from server should be status: connected
                data = websocket.receive_json()
                assert data["type"] == "status"
                assert data["status"] == "connected"

                # 2. Send config
                websocket.send_json({"type": "config", "sample_rate": 16000})

                # 3. Send small audio frame (float32, 128 samples)
                samples = np.zeros(128, dtype=np.float32)
                websocket.send_bytes(samples.tobytes())

                # 4. Close gracefully
                websocket.close()

    def test_websocket_invalid_json(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app) as client:
            with client.websocket_connect("/ws/audio") as websocket:
                data = websocket.receive_json()
                assert data["type"] == "status"

                # Send invalid text message
                websocket.send_text("INVALID_NOT_JSON")
                err = websocket.receive_json()
                assert err["type"] == "error"

    def test_websocket_streaming_increments_chunks_analyzed(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.audio.vad import VADResult

        fake_detector = MagicMock()
        fake_detector.predict_waveform.return_value = {
            "spoof_probability": 0.25,
            "inference_time_s": 0.01,
        }
        fake_vad_result = VADResult(
            has_speech=True,
            speech_ratio=1.0,
            n_frames=200,
            n_speech_frames=200,
            peak_rms=0.5,
            mean_rms=0.5,
        )

        with patch("backend.websocket.audio_stream.ModelManager.get_detector", return_value=fake_detector), \
             patch("backend.websocket.audio_stream._vad.detect", return_value=fake_vad_result):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/audio") as ws:
                    status = ws.receive_json()
                    assert status["type"] == "status"

                    ws.send_json({"type": "config", "sample_rate": 16000})
                    listening = ws.receive_json()
                    assert listening["type"] == "status"
                    assert listening["status"] == "listening"

                    # Send 5.0s of audio (80,000 float32 samples at 16 kHz)
                    audio_window = (np.sin(np.linspace(0, 440 * 2 * np.pi * 5.0, 80000)) * 0.5).astype(np.float32)
                    ws.send_bytes(audio_window.tobytes())

                    # First window: status: processing -> detection -> optional perf
                    proc1 = ws.receive_json()
                    assert proc1["type"] == "status"
                    assert proc1["status"] == "processing"

                    det1 = ws.receive_json()
                    assert det1["type"] == "detection"
                    assert det1["chunks_analyzed"] == 1
                    assert det1["spoof_probability"] == 0.25

                    perf1 = ws.receive_json()
                    assert perf1["type"] == "perf"

                    # Send second 5.0s of audio
                    ws.send_bytes(audio_window.tobytes())

                    proc2 = ws.receive_json()
                    assert proc2["type"] == "status"
                    assert proc2["status"] == "processing"

                    det2 = ws.receive_json()
                    assert det2["type"] == "detection"
                    assert det2["chunks_analyzed"] == 2
                    assert det2["spoof_probability"] == 0.25

                    perf2 = ws.receive_json()
                    assert perf2["type"] == "perf"

                    ws.close()

    def test_websocket_two_chunk_moving_average(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.audio.vad import VADResult

        fake_detector = MagicMock()
        # Return alternating spoof scores: 0.8 on chunk 1, 0.2 on chunk 2, 0.8 on chunk 3
        scores = iter([
            {"spoof_probability": 0.80, "inference_time_s": 0.01},
            {"spoof_probability": 0.20, "inference_time_s": 0.01},
            {"spoof_probability": 0.80, "inference_time_s": 0.01},
        ])
        fake_detector.predict_waveform.side_effect = lambda chunk: next(scores)
        fake_vad_result = VADResult(
            has_speech=True,
            speech_ratio=1.0,
            n_frames=200,
            n_speech_frames=200,
            peak_rms=0.5,
            mean_rms=0.5,
        )

        with patch("backend.websocket.audio_stream.ModelManager.get_detector", return_value=fake_detector), \
             patch("backend.websocket.audio_stream._vad.detect", return_value=fake_vad_result):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/audio") as ws:
                    ws.receive_json()  # status: connected
                    ws.send_json({"type": "config", "sample_rate": 16000})
                    ws.receive_json()  # status: listening

                    audio_window = (np.sin(np.linspace(0, 440 * 2 * np.pi * 5.0, 80000)) * 0.5).astype(np.float32)

                    # Chunk 1: score 0.80 -> avg is 0.80 -> prediction "spoof"
                    ws.send_bytes(audio_window.tobytes())
                    assert ws.receive_json()["status"] == "processing"
                    d1 = ws.receive_json()
                    assert d1["spoof_probability"] == 0.80
                    assert d1["prediction"] == "spoof"
                    assert d1["chunks_analyzed"] == 1
                    ws.receive_json()  # perf

                    # Chunk 2: score 0.20 -> avg(0.80, 0.20) = 0.50 -> prediction "real" (<= 0.5)
                    ws.send_bytes(audio_window.tobytes())
                    assert ws.receive_json()["status"] == "processing"
                    d2 = ws.receive_json()
                    assert d2["spoof_probability"] == 0.50
                    assert d2["prediction"] == "real"
                    assert d2["chunks_analyzed"] == 2
                    ws.receive_json()  # perf

                    # Chunk 3: score 0.80 -> avg(0.20, 0.80) = 0.50 -> prediction "real"
                    ws.send_bytes(audio_window.tobytes())
                    assert ws.receive_json()["status"] == "processing"
                    d3 = ws.receive_json()
                    assert d3["spoof_probability"] == 0.50
                    assert d3["prediction"] == "real"
                    assert d3["chunks_analyzed"] == 3
                    ws.receive_json()  # perf

                    ws.close()

    def test_websocket_waiting_for_speech_emits_overall_average(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.audio.vad import VADResult

        fake_detector = MagicMock()
        # Chunk 1: 0.80, Chunk 2: 0.40 -> overall average is 0.60
        scores = iter([
            {"spoof_probability": 0.80, "inference_time_s": 0.01},
            {"spoof_probability": 0.40, "inference_time_s": 0.01},
        ])
        fake_detector.predict_waveform.side_effect = lambda chunk: next(scores)
        fake_vad_speech = VADResult(
            has_speech=True,
            speech_ratio=1.0,
            n_frames=200,
            n_speech_frames=200,
            peak_rms=0.5,
            mean_rms=0.5,
        )
        fake_vad_silence = VADResult(
            has_speech=False,
            speech_ratio=0.0,
            n_frames=200,
            n_speech_frames=0,
            peak_rms=0.0,
            mean_rms=0.0,
        )
        vad_results = iter([fake_vad_speech, fake_vad_speech, fake_vad_silence])

        with patch("backend.websocket.audio_stream.ModelManager.get_detector", return_value=fake_detector), \
             patch("backend.websocket.audio_stream._vad.detect", side_effect=lambda w: next(vad_results)):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/audio") as ws:
                    ws.receive_json()  # connected
                    ws.send_json({"type": "config", "sample_rate": 16000})
                    ws.receive_json()  # listening

                    audio_window = (np.sin(np.linspace(0, 440 * 2 * np.pi * 5.0, 80000)) * 0.5).astype(np.float32)

                    # Chunk 1 (speech - 80,000 samples)
                    ws.send_bytes(audio_window.tobytes())
                    ws.receive_json()  # processing
                    ws.receive_json()  # detection
                    ws.receive_json()  # perf

                    # Chunk 2 (speech - 64,000 samples completes 80,000 with 16,000 overlap)
                    ws.send_bytes(audio_window[:64000].tobytes())
                    ws.receive_json()  # processing
                    ws.receive_json()  # detection
                    ws.receive_json()  # perf

                    # Now send silence (audio stops)
                    silence = np.zeros(64000, dtype=np.float32)
                    ws.send_bytes(silence.tobytes())

                    # Server must emit detection with average of all chunks (0.80 + 0.40) / 2 = 0.60
                    summary_det = ws.receive_json()
                    assert summary_det["type"] == "detection"
                    assert summary_det["spoof_probability"] == 0.60
                    assert summary_det["prediction"] == "spoof"
                    assert summary_det["chunks_analyzed"] == 2

                    # Followed by status: waiting_for_speech
                    wait_status = ws.receive_json()
                    assert wait_status["type"] == "status"
                    assert wait_status["status"] == "waiting_for_speech"

                    ws.close()

    def test_websocket_noise_gate_ignores_ambient_noise(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app) as client:
            with client.websocket_connect("/ws/audio") as ws:
                status = ws.receive_json()
                assert status["type"] == "status"

                ws.send_json({"type": "config", "sample_rate": 16000})
                listening = ws.receive_json()
                assert listening["status"] == "listening"

                # Send 5.0s of very faint ambient noise (peak amplitude 0.005 < MIN_SPEECH_PEAK 0.05)
                faint_noise = (np.random.randn(80000) * 0.002).astype(np.float32)
                ws.send_bytes(faint_noise.tobytes())

                # Server should NOT trigger detection; must send waiting_for_speech
                msg = ws.receive_json()
                assert msg["type"] == "status"
                assert msg["status"] == "waiting_for_speech"

                ws.close()




