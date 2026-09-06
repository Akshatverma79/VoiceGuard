"""
VoiceGuard — tests/test_upload_analysis.py
Automated tests for the local manual audio upload endpoint: POST /api/analyze.

Tests:
  - Valid WAV file upload & analysis
  - Valid MP3 file upload & analysis
  - Valid M4A file upload & analysis
  - Long audio file chunking and aggregation
  - Empty file upload rejection (HTTP 400)
  - Unsupported format rejection (HTTP 400)
  - Silent audio rejection (HTTP 400)
  - Too-short audio rejection (HTTP 400)
  - Temporary file cleanup (no disk leak)
"""

import io
import sys
import wave
import struct
from pathlib import Path
from typing import Optional

import pytest
import numpy as np
import torch
from starlette.testclient import TestClient

# ── Path setup ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.resolve()
_BACKEND = _ROOT / "backend"
_ML = _ROOT / "ml"
for p in [str(_BACKEND), str(_ML), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import av
from main import app
from services.model_manager import ModelManager
import api.analyze as analyze_module


# ── Helpers to generate synthetic audio files in-memory ──────────────────

def _make_wav_bytes(
    duration_s: float = 2.0,
    sample_rate: int = 16_000,
    frequency_hz: float = 440.0,
    silent: bool = False,
    amplitude: float = 0.5,
) -> bytes:
    """Generate in-memory WAV bytes with a sine wave or silence."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        if silent or n_samples == 0:
            w.writeframes(b"\x00\x00" * n_samples)
        else:
            frames = []
            for i in range(n_samples):
                val = int(32767.0 * amplitude * np.sin(2.0 * np.pi * frequency_hz * i / sample_rate))
                frames.append(struct.pack("<h", max(-32768, min(32767, val))))
            w.writeframes(b"".join(frames))
    return buf.getvalue()


def _make_mp3_bytes(
    duration_s: float = 2.0,
    sample_rate: int = 16_000,
    frequency_hz: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Generate in-memory MP3 bytes using PyAV."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * frequency_hz * t) * 32767 * amplitude).astype(np.int16)

    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp3")
    stream = out.add_stream("mp3", rate=sample_rate)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.rate = sample_rate
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    return buf.getvalue()


def _make_m4a_bytes(
    duration_s: float = 2.0,
    sample_rate: int = 16_000,
    frequency_hz: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Generate in-memory M4A (AAC) bytes using PyAV."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * frequency_hz * t) * 32767 * amplitude).astype(np.int16)

    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="ipod")  # ipod format container = m4a
    stream = out.add_stream("aac", rate=sample_rate)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.rate = sample_rate
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    return buf.getvalue()


# ── Mock detector for fast, predictable testing ────────────────────────────

class MockUploadDetector:
    """Deterministic mock detector to test upload pipeline without heavy GPU inference."""
    def predict_waveform(self, waveform: torch.Tensor):
        return {
            "prediction": "spoof",
            "spoof_probability": 0.82,
            "real_probability": 0.18,
            "inference_time_s": 0.05,
            "device": "cpu",
        }


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure ModelManager is initialized (or mocked) during tests."""
    ModelManager._detector = MockUploadDetector()
    ModelManager._initialized = True
    analyze_module._analyzer = None
    yield
    analyze_module._analyzer = None


@pytest.fixture
def client():
    return TestClient(app)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestManualAudioUploadAPI:
    """Tests for POST /api/analyze."""

    def test_analyze_valid_wav(self, client):
        wav_bytes = _make_wav_bytes(duration_s=2.0)
        response = client.post(
            "/api/analyze",
            files={"file": ("speech.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["prediction"] in {"real", "spoof"}
        assert 0.0 <= data["spoof_probability"] <= 1.0
        assert data["risk_level"] in {"low", "medium", "high"}
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["chunks_analyzed"] >= 1
        assert data["audio_duration_s"] >= 1.5
        assert "recommendation" in data
        assert len(data["recommendation"]) > 0
        assert "processing_time_ms" in data

    def test_analyze_valid_mp3(self, client):
        mp3_bytes = _make_mp3_bytes(duration_s=2.0)
        response = client.post(
            "/api/analyze",
            files={"file": ("speech.mp3", io.BytesIO(mp3_bytes), "audio/mpeg")},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["prediction"] in {"real", "spoof"}
        assert 0.0 <= data["spoof_probability"] <= 1.0
        assert data["risk_level"] in {"low", "medium", "high"}
        assert data["chunks_analyzed"] >= 1

    def test_analyze_valid_m4a(self, client):
        m4a_bytes = _make_m4a_bytes(duration_s=2.0)
        response = client.post(
            "/api/analyze",
            files={"file": ("speech.m4a", io.BytesIO(m4a_bytes), "audio/mp4")},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["prediction"] in {"real", "spoof"}
        assert 0.0 <= data["spoof_probability"] <= 1.0
        assert data["risk_level"] in {"low", "medium", "high"}
        assert data["chunks_analyzed"] >= 1

    def test_analyze_long_audio_chunks(self, client):
        """Audio > 8s should result in multiple chunks and aggregate scores."""
        long_wav = _make_wav_bytes(duration_s=9.0)
        response = client.post(
            "/api/analyze",
            files={"file": ("long_audio.wav", io.BytesIO(long_wav), "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_analyzed"] >= 2
        assert len(data["chunk_scores"]) >= 2
        assert data["audio_duration_s"] >= 8.5

    def test_empty_file_rejected(self, client):
        response = client.post(
            "/api/analyze",
            files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower() or "could not be processed" in data["detail"].lower()

    def test_unsupported_format_rejected(self, client):
        response = client.post(
            "/api/analyze",
            files={"file": ("document.pdf", io.BytesIO(b"fake pdf"), "application/pdf")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "unsupported" in data["detail"].lower()

    def test_too_short_audio_rejected(self, client):
        """Audio clip under 0.5s should be rejected with friendly message."""
        short_wav = _make_wav_bytes(duration_s=0.2)
        response = client.post(
            "/api/analyze",
            files={"file": ("short.wav", io.BytesIO(short_wav), "audio/wav")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "too short" in data["detail"].lower()

    def test_silent_audio_rejected(self, client):
        """Silent audio should be rejected by VAD or silence detector."""
        silent_wav = _make_wav_bytes(duration_s=2.0, silent=True)
        response = client.post(
            "/api/analyze",
            files={"file": ("silent.wav", io.BytesIO(silent_wav), "audio/wav")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "no sufficient speech" in data["detail"].lower() or "silent" in data["detail"].lower()

    def test_temp_file_deleted_after_analysis(self, client, monkeypatch):
        """Verify uploaded audio file is deleted immediately after analysis."""
        import tempfile
        created_paths = []
        original_temp = tempfile.NamedTemporaryFile

        def tracked_temp(*args, **kwargs):
            tmp = original_temp(*args, **kwargs)
            created_paths.append(tmp.name)
            return tmp

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracked_temp)

        wav_bytes = _make_wav_bytes(duration_s=2.0)
        response = client.post(
            "/api/analyze",
            files={"file": ("cleanup_test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert response.status_code == 200
        assert len(created_paths) == 1
        assert not Path(created_paths[0]).exists(), f"Temporary file {created_paths[0]} was not deleted!"

