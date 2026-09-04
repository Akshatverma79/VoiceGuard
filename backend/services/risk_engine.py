"""
VoiceGuard — backend/services/risk_engine.py
Rolling risk score engine for real-time live detection.

Formula: Exponential Weighted Moving Average (EWMA)
───────────────────────────────────────────────────
risk_score(t) = EWMA_ALPHA × spoof_prob(t) + (1 − EWMA_ALPHA) × risk_score(t−1)

Rationale:
    - EWMA reacts faster than a simple moving average to genuine changes
      while still smoothing out single-chunk noise.
    - α = 0.4 means the current chunk has 40% weight; the accumulated
      history has 60% weight. This avoids the REAL/FAKE/REAL flip-flopping
      that would occur if each chunk's raw probability were displayed directly.
    - All thresholds are defined here as module-level constants — never
      scattered through handler code.

⚠️  Calibration note:
    These thresholds are starting values only. They require calibration
    against a real-world distribution of genuine and spoofed speech.
    Do NOT treat the current values as validated accuracy claims.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Deque

# ── Configuration constants ────────────────────────────────────────────────
# Decay factor for EWMA. Higher = more reactive, lower = smoother.
EWMA_ALPHA: float = 0.4

# Risk level thresholds (spoof probability in [0, 1])
LOW_THRESHOLD: float = 0.35     # below → LOW
HIGH_THRESHOLD: float = 0.65    # above → HIGH  (between → MEDIUM)

# Minimum valid speech chunks before emitting a HIGH verdict.
# Prevents a single loud noise from triggering a false HIGH alert.
MIN_CHUNKS_FOR_HIGH: int = 2

# Maximum number of score entries to keep in history (for UI display)
RISK_WINDOW_MAX: int = 20


# ── Types ──────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"   # not enough data yet


@dataclass
class RiskResult:
    """Output of a single RiskEngine.update() call."""
    risk_score: float          # EWMA-smoothed spoof probability [0, 1]
    risk_level: RiskLevel      # LOW / MEDIUM / HIGH / UNKNOWN
    chunks_seen: int           # total valid chunks processed so far
    history: List[float]       # recent raw chunk scores (for UI sparkline)


# ── RiskEngine ─────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Per-session rolling risk tracker.

    One instance per WebSocket connection. Not thread-safe across connections
    (each connection owns its own RiskEngine instance).

    Usage::

        engine = RiskEngine()
        result = engine.update(spoof_probability=0.82)
        print(result.risk_level)  # RiskLevel.HIGH
    """

    def __init__(
        self,
        alpha: float = EWMA_ALPHA,
        low_threshold: float = LOW_THRESHOLD,
        high_threshold: float = HIGH_THRESHOLD,
        min_chunks_for_high: int = MIN_CHUNKS_FOR_HIGH,
        window_max: int = RISK_WINDOW_MAX,
    ) -> None:
        self._alpha = alpha
        self._low = low_threshold
        self._high = high_threshold
        self._min_high = min_chunks_for_high

        self._ewma: float = 0.0
        self._chunks_seen: int = 0
        self._history: Deque[float] = deque(maxlen=window_max)

    def update(self, spoof_probability: float) -> RiskResult:
        """
        Ingest a new per-chunk spoof probability and return the updated risk.

        Args:
            spoof_probability: float in [0.0, 1.0] from AASIST softmax output.

        Returns:
            RiskResult with updated score, level, and history.
        """
        p = max(0.0, min(1.0, spoof_probability))  # clamp
        self._history.append(round(p, 4))

        if self._chunks_seen == 0:
            # Seed EWMA with first observation
            self._ewma = p
        else:
            self._ewma = self._alpha * p + (1.0 - self._alpha) * self._ewma

        self._chunks_seen += 1

        return RiskResult(
            risk_score=round(self._ewma, 4),
            risk_level=self._classify(),
            chunks_seen=self._chunks_seen,
            history=list(self._history),
        )

    def reset(self) -> None:
        """Reset engine state (e.g., after a session restart)."""
        self._ewma = 0.0
        self._chunks_seen = 0
        self._history.clear()

    def _classify(self) -> RiskLevel:
        if self._chunks_seen < self._min_high:
            return RiskLevel.UNKNOWN
        if self._ewma >= self._high:
            return RiskLevel.HIGH
        if self._ewma >= self._low:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # ── Read-only properties ───────────────────────────────────────────────

    @property
    def current_score(self) -> float:
        return round(self._ewma, 4)

    @property
    def chunks_seen(self) -> int:
        return self._chunks_seen

    @property
    def history(self) -> List[float]:
        return list(self._history)
