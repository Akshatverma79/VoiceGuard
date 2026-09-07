"""
VoiceGuard — backend/websocket/audio_buffer.py
Per-connection audio sample buffer for real-time streaming.

Design
──────
The browser's AudioWorklet fires every 128 samples (≈ 2.9 ms at 44 100 Hz).
Sending each 128-sample block directly to AASIST would be absurd — the model
needs 64 600 samples (≈ 4 s) per inference call.

This buffer:
  1. Accumulates raw float32 PCM samples at the browser's native sample rate.
  2. Triggers processing when BUFFER_PROCESS_SECONDS of audio is accumulated.
  3. After processing, retains the last OVERLAP_SECONDS of samples (sliding window)
     to avoid missing speech that spans buffer boundaries.
  4. Caps total buffer size at MAX_BUFFER_SECONDS to prevent unbounded memory growth.
     Oldest samples are discarded when the cap is reached.

The buffer stores samples as a Python list of floats (simplest, no numpy needed).
For Phase 3 throughput (4 s trigger, browser at 44 kHz = ~176 000 samples) this
is well within acceptable memory bounds per connection.
"""

from __future__ import annotations

import struct
from typing import List, Optional

# ── Buffer configuration constants ─────────────────────────────────────────
# How many seconds to accumulate before triggering processing
BUFFER_PROCESS_SECONDS: float = 5.0

# How many seconds of overlap to keep after processing (sliding window)
OVERLAP_SECONDS: float = 1.0

# Hard cap on buffer length to prevent memory leaks
MAX_BUFFER_SECONDS: float = 30.0


class AudioBuffer:
    """
    Per-WebSocket-connection raw PCM sample accumulator.

    One instance is created per accepted connection and discarded on disconnect.

    Args:
        native_sample_rate: The sample rate reported by the browser (e.g. 44100).
        process_seconds   : Trigger processing after this many seconds of audio.
        overlap_seconds   : Retain this many seconds after processing.
        max_seconds       : Hard memory cap; oldest samples discarded if exceeded.
    """

    def __init__(
        self,
        native_sample_rate: int = 44_100,
        process_seconds: float = BUFFER_PROCESS_SECONDS,
        overlap_seconds: float = OVERLAP_SECONDS,
        max_seconds: float = MAX_BUFFER_SECONDS,
    ) -> None:
        if native_sample_rate <= 0:
            raise ValueError("native_sample_rate must be > 0")
        if overlap_seconds >= process_seconds:
            raise ValueError("overlap_seconds must be < process_seconds")

        self._rate = native_sample_rate
        self._process_samples = int(process_seconds * native_sample_rate)
        self._overlap_samples = int(overlap_seconds * native_sample_rate)
        self._max_samples = int(max_seconds * native_sample_rate)

        self._samples: List[float] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def push_bytes(self, data: bytes) -> None:
        """
        Append raw bytes from a WebSocket binary frame.
        Interprets data as a packed array of IEEE 754 float32 values (little-endian).
        This matches the Float32Array binary format sent by AudioWorklet.
        """
        n_floats = len(data) // 4
        if n_floats == 0:
            return
        floats = struct.unpack_from(f"<{n_floats}f", data, 0)
        self._samples.extend(floats)

        # Hard cap: drop oldest samples if buffer exceeds MAX_BUFFER_SECONDS
        if len(self._samples) > self._max_samples:
            excess = len(self._samples) - self._max_samples
            self._samples = self._samples[excess:]

    def should_process(self) -> bool:
        """Return True when enough audio has been accumulated to run AASIST."""
        return len(self._samples) >= self._process_samples

    def consume(self) -> List[float]:
        """
        Extract samples for processing, keeping the overlap tail.

        Returns:
            The accumulated samples (process_samples long).
            After this call, the buffer retains only the overlap tail from
            the processed window.
        """
        count = self._process_samples if len(self._samples) >= self._process_samples else len(self._samples)
        result = self._samples[:count]
        # Keep overlap tail for the next window
        if self._overlap_samples > 0 and count >= self._overlap_samples:
            self._samples = self._samples[count - self._overlap_samples:]
        else:
            self._samples = self._samples[count:]
        return result

    def clear(self) -> None:
        """Discard all buffered samples (e.g., on disconnect)."""
        self._samples = []

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def buffered_samples(self) -> int:
        return len(self._samples)

    @property
    def buffered_seconds(self) -> float:
        return len(self._samples) / self._rate if self._rate > 0 else 0.0

    @property
    def native_sample_rate(self) -> int:
        return self._rate
