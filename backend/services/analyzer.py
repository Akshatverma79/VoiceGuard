"""
VoiceGuard — backend/services/analyzer.py
Full audio analysis pipeline: processor → VAD → chunker → AASIST × N → aggregate.

This is the single entry point for the backend API and evaluation scripts.
It chains all Phase 2 components and returns a structured result.

Aggregation strategy: arithmetic mean of per-chunk spoof probabilities.

Rationale:
    - Mean is stable: one anomalous chunk in a long recording does not
      dominate the result the way max-pooling would.
    - Mean is interpretable: the result is the average "spoof likelihood"
      across all analyzed windows.
    - The strategy is explicit and configurable via AggregationStrategy.
    - A simple 0.5 threshold separates real from spoof at the file level,
      consistent with standard binary classification practice.

Note: The threshold (0.5) and strategy can be changed without touching
the ML layer. Phase 3 may introduce per-class calibration.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

# ── Path setup — locate ml/ directory ─────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent.parent.resolve()
_ROOT_DIR = _BACKEND_DIR.parent.resolve()
_ML_DIR = _ROOT_DIR / "ml"

for _p in [str(_ML_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local imports ──────────────────────────────────────────────────────────
from audio.processor import AudioProcessor, SilentAudioError, AudioLoadError, UnsupportedFormatError
from audio.vad import VoiceActivityDetector
from audio.chunker import AudioChunker


# ── Aggregation strategy ───────────────────────────────────────────────────

class AggregationStrategy(str, Enum):
    """
    How per-chunk spoof probabilities are combined into a file-level score.

    MEAN : Arithmetic mean — balanced, interpretable, default.
    MAX  : Maximum — most sensitive to any single spoof chunk.
    """
    MEAN = "mean"
    MAX = "max"


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Full pipeline result for one audio file.

    Attributes:
        prediction              : "real" or "spoof"
        spoof_probability       : aggregated file-level spoof score [0, 1]
        chunks_analyzed         : number of AASIST inference calls made
        audio_duration_s        : duration of the source audio (seconds)
        total_inference_time_s  : wall-clock time for all AASIST calls
        avg_chunk_inference_time_s : mean per-chunk inference time
        vad_speech_ratio        : fraction of frames classified as speech
        aggregation_strategy    : which strategy was used
        chunk_scores            : list of per-chunk spoof probabilities
        device                  : torch device used ("cpu" or "cuda")
        error                   : error message if pipeline failed (else None)
    """
    prediction: str
    spoof_probability: float
    chunks_analyzed: int
    audio_duration_s: float
    total_inference_time_s: float
    avg_chunk_inference_time_s: float
    vad_speech_ratio: float
    aggregation_strategy: str
    chunk_scores: List[float]
    device: str
    error: Optional[str] = None

    def to_api_response(self) -> Dict[str, Any]:
        """Minimal API response matching the Phase 2 spec."""
        return {
            "prediction": self.prediction,
            "spoof_probability": self.spoof_probability,
            "chunks_analyzed": self.chunks_analyzed,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full result dict for evaluation scripts and logging."""
        return {
            "prediction": self.prediction,
            "spoof_probability": self.spoof_probability,
            "chunks_analyzed": self.chunks_analyzed,
            "audio_duration_s": self.audio_duration_s,
            "total_inference_time_s": self.total_inference_time_s,
            "avg_chunk_inference_time_s": self.avg_chunk_inference_time_s,
            "vad_speech_ratio": self.vad_speech_ratio,
            "aggregation_strategy": self.aggregation_strategy,
            "chunk_scores": self.chunk_scores,
            "device": self.device,
            "error": self.error,
        }


# ── AudioAnalyzer ──────────────────────────────────────────────────────────

class AudioAnalyzer:
    """
    Orchestrates the full VoiceGuard Phase 2 pipeline:

        WAV file
          → AudioProcessor   (load, mono, resample, normalize)
          → VoiceActivityDetector (silence check)
          → AudioChunker     (split into 4.04 s segments)
          → AASISTDetector × N (per-chunk inference)
          → aggregate        (mean / max of chunk scores)
          → AnalysisResult

    Args:
        strategy : Aggregation strategy. Default: MEAN.

    The AASIST model is loaded lazily on the first call to analyze().
    Subsequent calls reuse the same loaded model.
    """

    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.MEAN) -> None:
        self._strategy = strategy
        self._processor = AudioProcessor()
        self._vad = VoiceActivityDetector()
        self._chunker = AudioChunker()
        self._detector: Optional[Any] = None  # lazy-loaded

    def _get_detector(self) -> Any:
        if self._detector is None:
            from detector import AASISTDetector
            self._detector = AASISTDetector()
        return self._detector

    def analyze(self, audio_path: str) -> AnalysisResult:
        """
        Run the full pipeline on a WAV file.

        Args:
            audio_path: Path to a WAV file.

        Returns:
            AnalysisResult — always returned, error field set on failure.
        """
        path_str = str(Path(audio_path).resolve())

        # ── Step 1: Load and preprocess ───────────────────────────────────
        try:
            audio = self._processor.process(path_str)
        except SilentAudioError as exc:
            return self._error_result(str(exc))
        except (AudioLoadError, UnsupportedFormatError) as exc:
            return self._error_result(str(exc))
        except Exception as exc:
            return self._error_result(f"Unexpected error during audio processing: {exc}")

        # ── Step 2: VAD — silence detection ───────────────────────────────
        vad_result = self._vad.detect(audio.waveform)
        if not vad_result.has_speech:
            return self._error_result(
                f"No speech detected in audio (speech_ratio={vad_result.speech_ratio:.2%}). "
                f"The recording appears to be silent or nearly silent. "
                f"AASIST inference skipped."
            )

        # ── Step 3: Chunk ─────────────────────────────────────────────────
        chunks = self._chunker.chunk(audio.waveform)

        # ── Step 4: AASIST inference per chunk ────────────────────────────
        detector = self._get_detector()
        chunk_scores: List[float] = []
        inference_times: List[float] = []

        for chunk in chunks:
            try:
                result = detector.predict_waveform(chunk)
                chunk_scores.append(result["spoof_probability"])
                inference_times.append(result["inference_time_s"])
            except RuntimeError as exc:
                # Log chunk failure but continue with remaining chunks
                print(f"[VoiceGuard] Warning: chunk inference failed — {exc}")

        if not chunk_scores:
            return self._error_result(
                "All chunk inferences failed. The model may not be loaded correctly."
            )

        # ── Step 5: Aggregate ─────────────────────────────────────────────
        agg_score = self._aggregate(chunk_scores)
        prediction = "spoof" if agg_score > 0.5 else "real"

        total_time = round(sum(inference_times), 4)
        avg_time = round(total_time / len(inference_times), 4)
        device_str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        return AnalysisResult(
            prediction=prediction,
            spoof_probability=round(agg_score, 4),
            chunks_analyzed=len(chunk_scores),
            audio_duration_s=audio.duration_s,
            total_inference_time_s=total_time,
            avg_chunk_inference_time_s=avg_time,
            vad_speech_ratio=vad_result.speech_ratio,
            aggregation_strategy=self._strategy.value,
            chunk_scores=[round(s, 4) for s in chunk_scores],
            device=device_str,
        )

    def _aggregate(self, scores: List[float]) -> float:
        """Apply the configured aggregation strategy to chunk scores."""
        if self._strategy == AggregationStrategy.MEAN:
            return sum(scores) / len(scores)
        elif self._strategy == AggregationStrategy.MAX:
            return max(scores)
        # Default fallback
        return sum(scores) / len(scores)

    @staticmethod
    def _error_result(message: str) -> AnalysisResult:
        return AnalysisResult(
            prediction="error",
            spoof_probability=0.0,
            chunks_analyzed=0,
            audio_duration_s=0.0,
            total_inference_time_s=0.0,
            avg_chunk_inference_time_s=0.0,
            vad_speech_ratio=0.0,
            aggregation_strategy="none",
            chunk_scores=[],
            device="unknown",
            error=message,
        )
