"""
VoiceGuard — tests/test_websocket.py
Phase 3 test suite: AudioBuffer, RiskEngine, ModelManager, and WebSocket endpoint.
"""

import math
import struct
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

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
