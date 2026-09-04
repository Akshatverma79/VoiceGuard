"""
VoiceGuard — backend/audio/__init__.py
Audio processing module for Phase 2.

Exports the three core components:
  - AudioProcessor  : load, mono-convert, resample, normalize
  - VoiceActivityDetector : energy-based speech/silence detection
  - AudioChunker    : split waveform into AASIST-sized segments
"""

from .processor import AudioProcessor, ProcessedAudio, AudioLoadError, SilentAudioError, UnsupportedFormatError
from .vad import VoiceActivityDetector, VADResult
from .chunker import AudioChunker, CHUNK_DURATION_SECONDS, CHUNK_OVERLAP_SECONDS

__all__ = [
    "AudioProcessor",
    "ProcessedAudio",
    "AudioLoadError",
    "SilentAudioError",
    "UnsupportedFormatError",
    "VoiceActivityDetector",
    "VADResult",
    "AudioChunker",
    "CHUNK_DURATION_SECONDS",
    "CHUNK_OVERLAP_SECONDS",
]
