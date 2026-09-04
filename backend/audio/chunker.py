"""
VoiceGuard — backend/audio/chunker.py
Split a waveform into AASIST-sized segments with optional overlap.

Chunk size design rationale
───────────────────────────
AASIST expects exactly 64 600 samples at 16 000 Hz:
    64 600 / 16 000 = 4.0375 seconds ≈ 4.04 seconds

This is not an arbitrary value — it matches the ASVspoof LA evaluation
protocol window used during AASIST training.  All chunks produced by this
module are padded or trimmed to exactly TARGET_SAMPLES (64 600) before
being fed to the model.

Overlap design rationale
────────────────────────
A 1.0-second overlap (25 % of chunk duration) ensures that speech
segments spanning chunk boundaries are captured in at least one chunk.
Overlap is configurable; set CHUNK_OVERLAP_SECONDS = 0 for non-overlapping.

Note: Overlap increases the number of AASIST inference calls linearly.
In Phase 3 (real-time streaming), this will be revisited.

Usage::

    chunker = AudioChunker()
    chunks = chunker.chunk(waveform)   # waveform: (1, N) Tensor at 16 kHz
    # chunks: List[Tensor], each (1, 64600) float32
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

# ── AASIST constants (must match ml/preprocessing.py) ─────────────────────
AASIST_SAMPLE_RATE: int = 16_000
TARGET_SAMPLES: int = 64_600        # exactly one AASIST inference window

# ── Default chunking parameters ────────────────────────────────────────────
# Expressed in seconds for readability; converted to samples at runtime.
CHUNK_DURATION_SECONDS: float = TARGET_SAMPLES / AASIST_SAMPLE_RATE  # ≈ 4.04 s
CHUNK_OVERLAP_SECONDS: float = 1.0                                    # 1 s overlap


class AudioChunker:
    """
    Splits a waveform into fixed-size chunks for AASIST inference.

    Each chunk is exactly TARGET_SAMPLES (64 600) samples long:
      - Chunks shorter than TARGET_SAMPLES are padded by repeating the chunk.
      - The final chunk of a recording may be shorter before padding.
      - Chunks are returned as float32 Tensors of shape (1, 64600).

    Args:
        chunk_duration_s : Duration of each chunk in seconds. Default: 4.04 s.
        overlap_s        : Overlap between consecutive chunks. Default: 1.0 s.
        target_samples   : Exact sample count each chunk must have. Default: 64 600.
        sample_rate      : Sample rate of the input waveform. Default: 16 000 Hz.
    """

    def __init__(
        self,
        chunk_duration_s: float = CHUNK_DURATION_SECONDS,
        overlap_s: float = CHUNK_OVERLAP_SECONDS,
        target_samples: int = TARGET_SAMPLES,
        sample_rate: int = AASIST_SAMPLE_RATE,
    ) -> None:
        if chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be > 0")
        if overlap_s < 0:
            raise ValueError("overlap_s must be >= 0")
        if overlap_s >= chunk_duration_s:
            raise ValueError("overlap_s must be < chunk_duration_s")
        if target_samples <= 0:
            raise ValueError("target_samples must be > 0")

        self._chunk_samples = int(chunk_duration_s * sample_rate)
        self._overlap_samples = int(overlap_s * sample_rate)
        self._hop_samples = self._chunk_samples - self._overlap_samples
        self._target_samples = target_samples
        self._sample_rate = sample_rate

    # ── Public API ──────────────────────────────────────────────────────────

    def chunk(self, waveform: Tensor) -> List[Tensor]:
        """
        Slice waveform into overlapping chunks, each padded to TARGET_SAMPLES.

        Args:
            waveform: float32 Tensor of shape (1, N).

        Returns:
            List of Tensors, each (1, TARGET_SAMPLES). Always ≥ 1 chunk.

        Raises:
            ValueError: if waveform has wrong shape or zero samples.
        """
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"Expected waveform of shape (1, N), got {waveform.shape}"
            )

        n_samples = waveform.shape[1]
        if n_samples == 0:
            raise ValueError("Cannot chunk a waveform with 0 samples.")

        chunks: List[Tensor] = []
        start = 0

        while start < n_samples:
            end = start + self._chunk_samples
            segment = waveform[:, start:end]  # (1, chunk_samples or less)
            chunks.append(self._pad_to_target(segment))
            start += self._hop_samples

        return chunks

    def expected_chunk_count(self, n_samples: int) -> int:
        """
        Return the number of chunks that will be produced for a given number
        of samples. Useful for pre-allocation and progress reporting.
        """
        if n_samples <= 0:
            return 0
        count = 0
        start = 0
        while start < n_samples:
            count += 1
            start += self._hop_samples
        return count

    # ── Internal helpers ────────────────────────────────────────────────────

    def _pad_to_target(self, segment: Tensor) -> Tensor:
        """
        Pad (by repeating) or trim a segment to exactly TARGET_SAMPLES.
        Shape: (1, any) → (1, TARGET_SAMPLES).
        """
        n = segment.shape[1]

        if n >= self._target_samples:
            return segment[:, : self._target_samples].float()

        # Repeat the segment until we have enough samples, then trim
        repeats = (self._target_samples // n) + 1
        padded = segment.repeat(1, repeats)
        return padded[:, : self._target_samples].float()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples

    @property
    def hop_samples(self) -> int:
        return self._hop_samples

    @property
    def target_samples(self) -> int:
        return self._target_samples
