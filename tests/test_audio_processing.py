"""
VoiceGuard — tests/test_audio_processing.py
Automated unit tests for the Phase 2 audio processing pipeline.

These tests use in-memory synthetic audio (torch tensors / WAV bytes)
and do NOT require:
  - AASIST.pth checkpoint
  - Any files in data/real/ or data/fake/
  - Internet access

Run with:
    pip install pytest
    pytest tests/test_audio_processing.py -v

Test coverage:
  - AudioProcessor  : mono, stereo, resampling, silence, invalid path
  - Normalization   : peak ≤ 1.0, silent audio raises SilentAudioError
  - VoiceActivityDetector : silent, speech-like, edge cases
  - AudioChunker    : output shapes, chunk count with/without overlap
"""

import io
import struct
import sys
import wave
from pathlib import Path
from typing import Optional

import pytest
import torch

# ── Path setup ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.resolve()
_BACKEND = _ROOT / "backend"
_ML = _ROOT / "ml"
for p in [str(_BACKEND), str(_ML)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from audio.processor import (
    AudioProcessor,
    AudioLoadError,
    SilentAudioError,
    UnsupportedFormatError,
    AASIST_SAMPLE_RATE,
)
from audio.vad import VoiceActivityDetector, VADResult
from audio.chunker import AudioChunker, TARGET_SAMPLES, CHUNK_DURATION_SECONDS


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_wav_bytes(
    n_samples: int,
    sample_rate: int = 16_000,
    n_channels: int = 1,
    silent: bool = False,
    amplitude: float = 0.5,
) -> bytes:
    """Create an in-memory WAV file as bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        if silent or n_samples == 0:
            data = b"\x00" * (n_samples * n_channels * 2)
        else:
            # Simple sine-like data: ramp pattern
            samples = []
            for i in range(n_samples * n_channels):
                val = int(amplitude * 32767 * (((i * 7) % 1000) / 500 - 1))
                val = max(-32768, min(32767, val))
                samples.append(struct.pack("<h", val))
            data = b"".join(samples)
        wf.writeframes(data)
    return buf.getvalue()


def _write_tmp_wav(tmp_path: Path, **kwargs) -> Path:
    """Write a temporary WAV file and return its path."""
    wav_bytes = _make_wav_bytes(**kwargs)
    p = tmp_path / "test.wav"
    p.write_bytes(wav_bytes)
    return p


# ════════════════════════════════════════════════════════════════════════════
# AudioProcessor tests
# ════════════════════════════════════════════════════════════════════════════

class TestAudioProcessor:
    def setup_method(self):
        self.proc = AudioProcessor()

    # ── File validation ───────────────────────────────────────────────────

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AudioLoadError):
            self.proc.process(str(tmp_path / "nonexistent.wav"))

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.wav"
        p.write_bytes(b"")
        with pytest.raises(AudioLoadError):
            self.proc.process(str(p))

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "audio.mp3"
        p.write_bytes(b"fake mp3 data")
        with pytest.raises(UnsupportedFormatError):
            self.proc.process(str(p))

    def test_corrupted_wav_raises(self, tmp_path):
        p = tmp_path / "bad.wav"
        p.write_bytes(b"NOT A WAV FILE - just garbage bytes 0xDEADBEEF")
        with pytest.raises(AudioLoadError):
            self.proc.process(str(p))

    # ── Mono audio ────────────────────────────────────────────────────────

    def test_mono_wav_processes_correctly(self, tmp_path):
        p = _write_tmp_wav(tmp_path, n_samples=16_000, n_channels=1)
        audio = self.proc.process(str(p))
        assert audio.waveform.shape[0] == 1, "Should be mono (1 channel)"
        assert audio.sample_rate == AASIST_SAMPLE_RATE
        assert audio.original_channels == 1
        assert audio.duration_s > 0

    # ── Stereo → mono ──────────────────────────────────────────────────────

    def test_stereo_converted_to_mono(self, tmp_path):
        p = _write_tmp_wav(tmp_path, n_samples=16_000, n_channels=2)
        audio = self.proc.process(str(p))
        assert audio.waveform.shape[0] == 1, "Stereo must be collapsed to mono"
        assert audio.original_channels == 2

    # ── Resampling ────────────────────────────────────────────────────────

    def test_different_sample_rate_resampled(self, tmp_path):
        # 8 kHz input should be resampled to 16 kHz
        p = _write_tmp_wav(tmp_path, n_samples=8_000, sample_rate=8_000)
        audio = self.proc.process(str(p))
        assert audio.sample_rate == AASIST_SAMPLE_RATE
        assert audio.original_sr == 8_000
        # After resampling 8000 samples from 8 kHz → 16 kHz, expect ~16 000
        assert audio.waveform.shape[1] > 0

    def test_44100_hz_resampled(self, tmp_path):
        p = _write_tmp_wav(tmp_path, n_samples=44_100, sample_rate=44_100)
        audio = self.proc.process(str(p))
        assert audio.sample_rate == AASIST_SAMPLE_RATE
        assert audio.original_sr == 44_100

    # ── Normalization ─────────────────────────────────────────────────────

    def test_normalized_peak_lte_one(self, tmp_path):
        p = _write_tmp_wav(tmp_path, n_samples=16_000, amplitude=0.9)
        audio = self.proc.process(str(p))
        peak = audio.waveform.abs().max().item()
        assert peak <= 1.0 + 1e-6, f"Peak {peak} exceeds 1.0"

    def test_silent_audio_raises_silent_audio_error(self, tmp_path):
        p = _write_tmp_wav(tmp_path, n_samples=16_000, silent=True)
        with pytest.raises(SilentAudioError):
            self.proc.process(str(p))

    # ── Very short audio ──────────────────────────────────────────────────

    def test_very_short_audio_processes(self, tmp_path):
        # 0.1 second at 16 kHz = 1600 samples — should process fine
        p = _write_tmp_wav(tmp_path, n_samples=1_600)
        audio = self.proc.process(str(p))
        assert audio.waveform.shape[0] == 1

    # ── Output dtype ──────────────────────────────────────────────────────

    def test_output_is_float32(self, tmp_path):
        p = _write_tmp_wav(tmp_path, n_samples=16_000)
        audio = self.proc.process(str(p))
        assert audio.waveform.dtype == torch.float32


# ════════════════════════════════════════════════════════════════════════════
# VoiceActivityDetector tests
# ════════════════════════════════════════════════════════════════════════════

class TestVoiceActivityDetector:
    def setup_method(self):
        self.vad = VoiceActivityDetector()

    # ── Silent tensors ────────────────────────────────────────────────────

    def test_zero_tensor_no_speech(self):
        waveform = torch.zeros(1, 16_000)
        result = self.vad.detect(waveform)
        assert result.has_speech is False
        assert result.speech_ratio == 0.0

    def test_empty_tensor_no_speech(self):
        waveform = torch.zeros(1, 0)
        result = self.vad.detect(waveform)
        assert result.has_speech is False

    # ── Speech-like tensors ───────────────────────────────────────────────

    def test_random_noise_has_speech(self):
        # Random noise has high energy → should trigger speech detection
        torch.manual_seed(42)
        waveform = torch.randn(1, 16_000) * 0.5
        result = self.vad.detect(waveform)
        assert result.has_speech is True
        assert result.speech_ratio > 0.0

    def test_speech_ratio_in_range(self):
        torch.manual_seed(0)
        waveform = torch.randn(1, 32_000) * 0.3
        result = self.vad.detect(waveform)
        assert 0.0 <= result.speech_ratio <= 1.0

    def test_n_frames_positive(self):
        waveform = torch.randn(1, 16_000)
        result = self.vad.detect(waveform)
        assert result.n_frames > 0

    # ── Partial silence ───────────────────────────────────────────────────

    def test_half_silent_half_speech(self):
        silence = torch.zeros(1, 8_000)
        speech = torch.randn(1, 8_000) * 0.5
        waveform = torch.cat([silence, speech], dim=1)
        result = self.vad.detect(waveform)
        # With 50% speech, should be detected (min_speech_ratio = 5%)
        assert result.has_speech is True
        assert result.speech_ratio < 0.6  # not fully speech

    # ── VADResult fields ──────────────────────────────────────────────────

    def test_vad_result_has_expected_fields(self):
        waveform = torch.randn(1, 16_000) * 0.3
        result = self.vad.detect(waveform)
        assert hasattr(result, "has_speech")
        assert hasattr(result, "speech_ratio")
        assert hasattr(result, "n_frames")
        assert hasattr(result, "n_speech_frames")
        assert hasattr(result, "peak_rms")
        assert hasattr(result, "mean_rms")

    # ── Edge: 1D tensor ───────────────────────────────────────────────────

    def test_1d_tensor_accepted(self):
        waveform = torch.randn(16_000) * 0.3
        result = self.vad.detect(waveform)
        assert isinstance(result, VADResult)

    # ── Frame RMS output ──────────────────────────────────────────────────

    def test_return_frame_rms(self):
        vad = VoiceActivityDetector(return_frame_rms=True)
        waveform = torch.randn(1, 16_000) * 0.3
        result = vad.detect(waveform)
        assert len(result.frame_rms) > 0


# ════════════════════════════════════════════════════════════════════════════
# AudioChunker tests
# ════════════════════════════════════════════════════════════════════════════

class TestAudioChunker:
    def setup_method(self):
        self.chunker = AudioChunker()

    # ── Output shape ──────────────────────────────────────────────────────

    def test_each_chunk_has_target_shape(self):
        waveform = torch.randn(1, 64_600)
        chunks = self.chunker.chunk(waveform)
        for chunk in chunks:
            assert chunk.shape == (1, TARGET_SAMPLES), (
                f"Expected (1, {TARGET_SAMPLES}), got {chunk.shape}"
            )

    def test_long_audio_multiple_chunks(self):
        # 30 seconds at 16 kHz
        waveform = torch.randn(1, 480_000)
        chunks = self.chunker.chunk(waveform)
        assert len(chunks) > 1, "30-second audio should produce multiple chunks"

    def test_short_audio_one_chunk(self):
        # 1 second — shorter than one chunk window
        waveform = torch.randn(1, 16_000)
        chunks = self.chunker.chunk(waveform)
        assert len(chunks) >= 1

    def test_exact_one_chunk_length(self):
        # Exactly one chunk = exactly TARGET_SAMPLES
        waveform = torch.randn(1, TARGET_SAMPLES)
        chunks = self.chunker.chunk(waveform)
        assert len(chunks) >= 1
        assert chunks[0].shape == (1, TARGET_SAMPLES)

    # ── Chunk count ───────────────────────────────────────────────────────

    def test_no_overlap_chunk_count(self):
        chunker = AudioChunker(overlap_s=0.0)
        # 8 seconds / 4.04 s per chunk ≈ 2 chunks
        waveform = torch.randn(1, 8 * 16_000)
        chunks = chunker.chunk(waveform)
        assert len(chunks) >= 2

    def test_overlap_increases_chunk_count(self):
        no_overlap = AudioChunker(overlap_s=0.0)
        with_overlap = AudioChunker(overlap_s=1.0)
        waveform = torch.randn(1, 16 * 16_000)  # 16 seconds
        n_no_overlap = len(no_overlap.chunk(waveform))
        n_with_overlap = len(with_overlap.chunk(waveform))
        assert n_with_overlap >= n_no_overlap, (
            "Overlap should produce at least as many chunks as no overlap"
        )

    # ── Output dtype ──────────────────────────────────────────────────────

    def test_chunks_are_float32(self):
        waveform = torch.randn(1, 64_600)
        chunks = self.chunker.chunk(waveform)
        for chunk in chunks:
            assert chunk.dtype == torch.float32

    # ── Wrong shape raises ────────────────────────────────────────────────

    def test_wrong_shape_raises(self):
        waveform = torch.randn(2, 64_600)  # 2 channels — should be 1
        with pytest.raises(ValueError):
            self.chunker.chunk(waveform)

    def test_zero_samples_raises(self):
        waveform = torch.zeros(1, 0)
        with pytest.raises(ValueError):
            self.chunker.chunk(waveform)

    # ── expected_chunk_count ──────────────────────────────────────────────

    def test_expected_chunk_count_matches_actual(self):
        waveform = torch.randn(1, 20 * 16_000)
        chunks = self.chunker.chunk(waveform)
        expected = self.chunker.expected_chunk_count(waveform.shape[1])
        assert len(chunks) == expected

    # ── Invalid config ────────────────────────────────────────────────────

    def test_overlap_gte_duration_raises(self):
        with pytest.raises(ValueError):
            AudioChunker(chunk_duration_s=2.0, overlap_s=2.0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            AudioChunker(overlap_s=-1.0)


# ════════════════════════════════════════════════════════════════════════════
# Constants sanity checks
# ════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_aasist_sample_rate(self):
        assert AASIST_SAMPLE_RATE == 16_000

    def test_target_samples(self):
        assert TARGET_SAMPLES == 64_600

    def test_chunk_duration_derived_from_aasist(self):
        # CHUNK_DURATION_SECONDS must match TARGET_SAMPLES / AASIST_SAMPLE_RATE
        expected = TARGET_SAMPLES / AASIST_SAMPLE_RATE
        assert abs(CHUNK_DURATION_SECONDS - expected) < 0.01
