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
_SUPPORTED_EXTENSIONS = {".wav", ".wave", ".mp3", ".m4a"}

# ── Size and duration limits ───────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
MIN_DURATION_SECONDS: float = 0.5            # 0.5 seconds minimum

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


class TooShortAudioError(ValueError):
    """Raised when audio clip duration is below the minimum required for analysis."""


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
    Loads and prepares an audio file (WAV, MP3, M4A) for the VoiceGuard pipeline.

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
        min_duration_s: float = 0.0,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        self._target_sr = target_sr
        self._silence_threshold = silence_threshold
        self._min_duration_s = min_duration_s
        self._max_file_size_bytes = max_file_size_bytes
        self._import_torchaudio()  # fail early if not installed

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, audio_path: str) -> ProcessedAudio:
        """
        Full pipeline: validate → load → mono → resample → normalize.

        Args:
            audio_path: Path to an audio file (WAV, MP3, M4A).

        Returns:
            ProcessedAudio with a (1, N) float32 waveform at 16 kHz.

        Raises:
            AudioLoadError        : file missing, unreadable, empty, or oversized.
            UnsupportedFormatError: unsupported audio format.
            TooShortAudioError    : audio is shorter than minimum duration.
            SilentAudioError      : audio is completely silent.
        """
        path = Path(audio_path).resolve()
        self._validate_path(path)

        waveform, orig_sr, orig_channels = self._load(path)
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, orig_sr)
        waveform = self._normalize(waveform, path)

        n_samples = waveform.shape[1]
        duration_s = round(n_samples / self._target_sr, 4)

        if self._min_duration_s > 0 and duration_s < self._min_duration_s:
            raise TooShortAudioError(
                f"Audio is too short for reliable analysis ({duration_s:.2f}s). "
                f"Minimum duration is {self._min_duration_s}s."
            )

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

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise AudioLoadError(f"Audio file not found: {path}")
        file_size = path.stat().st_size
        if file_size == 0:
            raise AudioLoadError("The uploaded audio file is empty.")
        if file_size > self._max_file_size_bytes:
            raise AudioLoadError(
                f"Audio file exceeds maximum size limit of {self._max_file_size_bytes // (1024 * 1024)} MB."
            )
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported audio format '{path.suffix}'. "
                f"Supported formats: {', '.join(sorted(ext.upper().lstrip('.') for ext in _SUPPORTED_EXTENSIONS))}."
            )

    @staticmethod
    def _load(path: Path) -> tuple[Tensor, int, int]:
        """Returns (waveform (C, N), sample_rate, num_channels)."""
        import torchaudio
        import numpy as np

        suffix = path.suffix.lower()

        # Try standard torchaudio / soundfile for WAV & MP3
        if suffix in {".wav", ".wave", ".mp3"}:
            try:
                waveform, sr = torchaudio.load(str(path))
                if waveform.numel() > 0 and waveform.shape[1] > 0:
                    return waveform, sr, waveform.shape[0]
            except Exception:
                try:
                    import soundfile as sf
                    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                    waveform = torch.from_numpy(data.T)
                    if waveform.numel() > 0 and waveform.shape[1] > 0:
                        return waveform, sr, waveform.shape[0]
                except Exception:
                    pass  # Fall through to PyAV

        # PyAV decoder (handles M4A, AAC, MP3, WAV, etc.)
        try:
            import av

            container = av.open(str(path))
            if not container.streams.audio:
                container.close()
                raise AudioLoadError(f"No audio stream found in '{path.name}'.")

            stream = container.streams.audio[0]
            orig_sr = stream.sample_rate
            orig_channels = stream.channels or 1

            frames = []
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray())
            container.close()

            if not frames:
                raise AudioLoadError(f"Audio file contains no audio frames: {path.name}")

            arr = np.concatenate(frames, axis=-1)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]

            # Normalize integer formats to [-1.0, 1.0] float32
            if np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                waveform = torch.from_numpy(arr).float() / max(abs(info.min), abs(info.max))
            else:
                waveform = torch.from_numpy(arr).float()

            if waveform.numel() == 0 or waveform.shape[1] == 0:
                raise AudioLoadError(f"Audio file contains no samples: {path}")

            return waveform, orig_sr, orig_channels

        except AudioLoadError:
            raise
        except Exception as exc:
            raise AudioLoadError(
                f"Failed to load '{path.name}'. The file may be corrupted or in an unsupported encoding.\n"
                f"Details: {exc}"
            ) from exc

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
