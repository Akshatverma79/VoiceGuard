"""
VoiceGuard — backend/audio/processor.py
Audio loading, conversion, resampling, and normalization.

AASIST requirements (read from ml/preprocessing.py):
  - Sample rate : 16,000 Hz
  - Channels    : mono (1 channel)
  - dtype       : float32

This module is responsible for all steps BEFORE chunking and VAD.
It returns a ProcessedAudio dataclass — a clean, validated, normalized
mono waveform at 16 kHz, with original metadata preserved for diagnostics.

Handles:
  - WAV files (mono and stereo)
  - Different sample rates (resampled to 16 kHz)
  - Empty / zero-length audio  → SilentAudioError
  - Completely silent audio    → SilentAudioError
  - Unsupported formats        → UnsupportedFormatError
  - Corrupted / unreadable     → AudioLoadError
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

# ── AASIST constants (must match ml/preprocessing.py) ─────────────────────
AASIST_SAMPLE_RATE: int = 16_000

# ── Supported file extensions ──────────────────────────────────────────────
_SUPPORTED_EXTENSIONS = {".wav", ".wave"}

# ── Silence threshold ──────────────────────────────────────────────────────
# Peak amplitude below this value → audio is considered silent.
# 1e-6 corresponds to ~-120 dBFS — well below any real microphone noise floor.
_SILENCE_THRESHOLD: float = 1e-6


# ── Custom exceptions ──────────────────────────────────────────────────────

class AudioLoadError(Exception):
    """Raised when an audio file cannot be read (corrupted, missing, etc.)."""


class SilentAudioError(ValueError):
    """
    Raised when audio is completely silent (peak amplitude < threshold).
    Running AASIST on silent audio produces meaningless predictions.
    """


class UnsupportedFormatError(ValueError):
    """Raised for unsupported audio file formats."""


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class ProcessedAudio:
    """
    Clean, validated, normalized mono waveform ready for VAD and chunking.

    Attributes:
        waveform         : float32 Tensor of shape (1, N) at 16 kHz
        sample_rate      : always AASIST_SAMPLE_RATE (16 000)
        duration_s       : duration in seconds (after resampling)
        original_sr      : sample rate of the source file
        original_channels: channel count of the source file
        file_path        : absolute path of the source file (str)
    """
    waveform: Tensor
    sample_rate: int
    duration_s: float
    original_sr: int
    original_channels: int
    file_path: str


# ── AudioProcessor ─────────────────────────────────────────────────────────

class AudioProcessor:
    """
    Loads and prepares a WAV file for the VoiceGuard pipeline.

    Usage::

        processor = AudioProcessor()
        audio = processor.process("path/to/audio.wav")
        # audio.waveform → Tensor (1, N), float32, 16 kHz, peak-normalized

    The processor is stateless; one instance can process many files.
    """

    def __init__(
        self,
        target_sr: int = AASIST_SAMPLE_RATE,
        silence_threshold: float = _SILENCE_THRESHOLD,
    ) -> None:
        self._target_sr = target_sr
        self._silence_threshold = silence_threshold
        self._import_torchaudio()  # fail early if not installed

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, audio_path: str) -> ProcessedAudio:
        """
        Full pipeline: validate → load → mono → resample → normalize.

        Args:
            audio_path: Path to a WAV audio file.

        Returns:
            ProcessedAudio with a (1, N) float32 waveform at 16 kHz.

        Raises:
            AudioLoadError      : file missing, unreadable, or empty.
            UnsupportedFormatError: non-WAV extension.
            SilentAudioError    : audio is completely silent.
        """
        path = Path(audio_path).resolve()
        self._validate_path(path)

        waveform, orig_sr, orig_channels = self._load(path)
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, orig_sr)
        waveform = self._normalize(waveform, path)

        n_samples = waveform.shape[1]
        duration_s = round(n_samples / self._target_sr, 4)

        return ProcessedAudio(
            waveform=waveform.float(),
            sample_rate=self._target_sr,
            duration_s=duration_s,
            original_sr=orig_sr,
            original_channels=orig_channels,
            file_path=str(path),
        )

    # ── Internal steps ──────────────────────────────────────────────────────

    @staticmethod
    def _import_torchaudio() -> None:
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "torchaudio is not installed. Run: pip install torchaudio"
            )

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.exists():
            raise AudioLoadError(f"Audio file not found: {path}")
        if path.stat().st_size == 0:
            raise AudioLoadError(f"Audio file is empty: {path}")
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported format '{path.suffix}'. "
                f"Only WAV files are supported in Phase 2. "
                f"Supported extensions: {sorted(_SUPPORTED_EXTENSIONS)}"
            )

    @staticmethod
    def _load(path: Path) -> tuple[Tensor, int, int]:
        """Returns (waveform (C, N), sample_rate, num_channels)."""
        import torchaudio

        try:
            waveform, sr = torchaudio.load(str(path))
        except Exception as exc:
            raise AudioLoadError(
                f"Failed to load '{path}'. "
                f"The file may be corrupted or in an unsupported encoding.\n"
                f"Details: {exc}"
            ) from exc

        if waveform.numel() == 0 or waveform.shape[1] == 0:
            raise AudioLoadError(f"Audio file contains no samples: {path}")

        return waveform, sr, waveform.shape[0]

    @staticmethod
    def _to_mono(waveform: Tensor) -> Tensor:
        """Average channels → mono. Shape: (C, N) → (1, N)."""
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0, keepdim=True)
        return waveform[:1, :]

    def _resample(self, waveform: Tensor, orig_sr: int) -> Tensor:
        """Resample to target sample rate if needed."""
        if orig_sr == self._target_sr:
            return waveform
        import torchaudio
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=self._target_sr
        )
        resampled = resampler(waveform)
        if resampled.shape[1] == 0:
            raise AudioLoadError(
                f"After resampling from {orig_sr} Hz to {self._target_sr} Hz, "
                f"audio has 0 samples. The clip may be too short."
            )
        return resampled

    def _normalize(self, waveform: Tensor, path: Path) -> Tensor:
        """
        Peak normalization: divide by the global maximum absolute value.

        - Ensures values stay in [-1, 1] without clipping.
        - If the audio is completely silent (peak < threshold), raises
          SilentAudioError rather than sending noise into the model.
        """
        peak = waveform.abs().max().item()
        if peak < self._silence_threshold:
            raise SilentAudioError(
                f"Audio '{path.name}' is completely silent "
                f"(peak amplitude = {peak:.2e} < threshold {self._silence_threshold:.2e}). "
                f"Running AASIST on silent audio produces meaningless predictions. "
                f"Please provide an audio file with actual speech content."
            )
        return waveform / peak
