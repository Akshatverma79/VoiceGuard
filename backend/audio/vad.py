"""
VoiceGuard — backend/audio/vad.py
Energy-based Voice Activity Detection (VAD).

Method: Short-Time Energy (STE)
───────────────────────────────
The waveform is split into short frames (default: 20 ms at 16 kHz = 320 samples).
For each frame, the Root Mean Square (RMS) energy is computed.
A frame is classified as "speech" if its RMS exceeds a configurable threshold.

Why energy-based?
  - Language-agnostic (works on any spoken language)
  - No external cloud services or large ML models
  - Sufficient for Phase 2 — the goal is to detect whether ANY meaningful
    audio is present, not to transcribe or diarize speech
  - Fast: O(N) in the number of samples

Limitations (documented):
  - Music or loud noise can also pass the energy threshold
  - The threshold is tunable but not adaptive per recording
  - No speaker-turn detection
  - Real-time / streaming VAD will be implemented in Phase 3

Usage::

    vad = VoiceActivityDetector()
    result = vad.detect(waveform)  # waveform: (1, N) float32 Tensor
    if not result.has_speech:
        raise SomeError("No speech detected")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor


# ── VAD constants ──────────────────────────────────────────────────────────

# Frame length for Short-Time Energy calculation (20 ms at 16 kHz)
FRAME_LENGTH_MS: int = 20
FRAME_LENGTH_SAMPLES: int = 320  # 20 ms × 16 000 Hz

# RMS energy threshold below which a frame is considered silence.
# This is the fraction of the waveform's peak RMS that must be exceeded.
# Default 0.01 = a frame must have at least 1 % of peak frame energy to
# count as speech. Conservative — errs on the side of detecting speech.
DEFAULT_ENERGY_THRESHOLD: float = 0.01

# Minimum fraction of frames that must be classified as speech for the
# clip to be considered to contain meaningful speech content.
DEFAULT_MIN_SPEECH_RATIO: float = 0.05  # 5 % of frames


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class VADResult:
    """
    Result of voice activity detection.

    Attributes:
        has_speech      : True if meaningful speech was detected.
        speech_ratio    : fraction of frames classified as speech [0.0, 1.0].
        n_frames        : total number of frames analyzed.
        n_speech_frames : number of frames classified as speech.
        peak_rms        : highest RMS energy seen across all frames.
        mean_rms        : mean RMS energy across speech frames (0 if none).
        frame_rms       : per-frame RMS values (empty if not requested).
    """
    has_speech: bool
    speech_ratio: float
    n_frames: int
    n_speech_frames: int
    peak_rms: float
    mean_rms: float
    frame_rms: List[float] = field(default_factory=list)


# ── VoiceActivityDetector ──────────────────────────────────────────────────

class VoiceActivityDetector:
    """
    Lightweight energy-based VAD.

    Splits the waveform into fixed-length frames and computes RMS energy
    per frame. Frames exceeding a threshold fraction of the peak RMS are
    classified as speech.

    Args:
        frame_length_samples: Samples per analysis frame. Default: 320 (20 ms).
        energy_threshold    : Relative RMS threshold [0, 1]. Default: 0.01.
        min_speech_ratio    : Minimum speech frame fraction. Default: 0.05.
        return_frame_rms    : If True, VADResult includes per-frame RMS list.
    """

    def __init__(
        self,
        frame_length_samples: int = FRAME_LENGTH_SAMPLES,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        min_speech_ratio: float = DEFAULT_MIN_SPEECH_RATIO,
        return_frame_rms: bool = False,
    ) -> None:
        if frame_length_samples <= 0:
            raise ValueError("frame_length_samples must be > 0")
        if not 0.0 <= energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in [0.0, 1.0]")
        if not 0.0 <= min_speech_ratio <= 1.0:
            raise ValueError("min_speech_ratio must be in [0.0, 1.0]")

        self._frame_len = frame_length_samples
        self._energy_threshold = energy_threshold
        self._min_speech_ratio = min_speech_ratio
        self._return_frame_rms = return_frame_rms

    def detect(self, waveform: Tensor) -> VADResult:
        """
        Detect voice activity in a waveform.

        Args:
            waveform: float32 Tensor of shape (1, N) or (N,).

        Returns:
            VADResult with speech/silence classification.
        """
        # Flatten to 1D
        samples = waveform.squeeze().float()
        if samples.dim() != 1:
            raise ValueError(
                f"Expected 1D or (1, N) waveform, got shape {waveform.shape}"
            )

        n_total = samples.shape[0]
        if n_total == 0:
            return VADResult(
                has_speech=False,
                speech_ratio=0.0,
                n_frames=0,
                n_speech_frames=0,
                peak_rms=0.0,
                mean_rms=0.0,
            )

        # ── Compute per-frame RMS ──────────────────────────────────────────
        # Pad so total length is divisible by frame_len
        pad = (self._frame_len - n_total % self._frame_len) % self._frame_len
        if pad > 0:
            samples = torch.cat([samples, torch.zeros(pad)])

        n_frames = samples.shape[0] // self._frame_len
        frames = samples.view(n_frames, self._frame_len)  # (F, frame_len)

        # RMS per frame: sqrt(mean(x²))
        rms_per_frame = frames.pow(2).mean(dim=1).sqrt()  # (F,)

        peak_rms = float(rms_per_frame.max().item())

        # ── Classify frames ────────────────────────────────────────────────
        if peak_rms == 0.0:
            # Completely silent
            return VADResult(
                has_speech=False,
                speech_ratio=0.0,
                n_frames=n_frames,
                n_speech_frames=0,
                peak_rms=0.0,
                mean_rms=0.0,
                frame_rms=rms_per_frame.tolist() if self._return_frame_rms else [],
            )

        # Threshold is relative to peak RMS
        abs_threshold = peak_rms * self._energy_threshold
        speech_mask = rms_per_frame > abs_threshold  # (F,) bool

        n_speech_frames = int(speech_mask.sum().item())
        speech_ratio = n_speech_frames / n_frames

        speech_rms_values = rms_per_frame[speech_mask]
        mean_rms = float(speech_rms_values.mean().item()) if n_speech_frames > 0 else 0.0

        has_speech = speech_ratio >= self._min_speech_ratio

        return VADResult(
            has_speech=has_speech,
            speech_ratio=round(speech_ratio, 4),
            n_frames=n_frames,
            n_speech_frames=n_speech_frames,
            peak_rms=round(peak_rms, 6),
            mean_rms=round(mean_rms, 6),
            frame_rms=rms_per_frame.tolist() if self._return_frame_rms else [],
        )
