"""
VoiceGuard — ml/preprocessing.py
Audio preprocessing for AASIST inference.

AASIST expects:
  - Sample rate: 16,000 Hz
  - Channels: mono (single channel)
  - Length: exactly 64,600 samples (≈ 4.04 seconds)
  - Normalized waveform tensor

This module handles loading and converting any WAV file to that format.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# ── AASIST fixed input length (matches ASVspoof LA eval protocol) ──────────
SAMPLE_RATE: int = 16_000
TARGET_SAMPLES: int = 64_600  # ~4.04 seconds at 16 kHz


def load_wav(audio_path: str, target_samples: Optional[int] = None) -> Tensor:
    """
    Load a WAV file and preprocess it for model inference.

    Pipeline:
      1. Validate file exists and has .wav extension.
      2. Read audio via torchaudio (soundfile fallback).
      3. Convert stereo to mono by averaging channels.
      4. Resample to 16 kHz.
      5. Optionally pad or trim to target_samples (if specified).
      6. Peak-normalize to prevent clipping.

    Args:
        audio_path: Absolute or relative path to a WAV file.
        target_samples: Optional target sample length. If None, full audio is kept.

    Returns:
        Tensor of shape (1, N) — ready for model inference.
    """
    path = Path(audio_path).resolve()

    # ── Validate file ─────────────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {path}\n"
            "Please provide a valid WAV file path."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if path.suffix.lower() not in (".wav", ".wave"):
        raise ValueError(
            f"Unsupported audio format: '{path.suffix}'. "
            "Only WAV files are supported in Phase 1."
        )

    # ── Load audio ────────────────────────────────────────────────────────
    try:
        import torchaudio  # imported here so the module is importable without torchaudio
    except ImportError:
        raise RuntimeError(
            "torchaudio is not installed. Run: pip install torchaudio"
        )

    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)
        except Exception as exc:
            raise ValueError(
                f"Failed to load audio file '{path}'. "
                f"The file may be corrupted or in an unsupported encoding.\n"
                f"Details: {exc}"
            ) from exc

    # waveform: (channels, samples)
    if waveform.numel() == 0:
        raise ValueError(f"Audio file contains no audio data: {path}")

    # ── Stereo → Mono ─────────────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, samples)
    # ensure shape is (1, samples)
    waveform = waveform[:1, :]

    # ── Resample to 16 kHz ────────────────────────────────────────────────
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # ── Optional pad or trim ──────────────────────────────────────────────
    n_samples = waveform.shape[1]

    if n_samples == 0:
        raise ValueError(
            f"After resampling, audio has 0 samples. "
            f"The original clip may be too short or corrupted: {path}"
        )

    if target_samples is not None:
        if n_samples < target_samples:
            # Repeat the clip until we reach target_samples
            repeats = (target_samples // n_samples) + 1
            waveform = waveform.repeat(1, repeats)
        # Trim to exact length
        waveform = waveform[:, :target_samples]

    # ── Normalize ─────────────────────────────────────────────────────────
    # AASIST operates on raw, unnormalized waveforms; the SincConv layer
    # is designed to work with values in the range expected from 16-bit PCM.
    # We do a light normalization to guard against clipping.
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    # Final shape: (1, 64600) — float32
    return waveform.float()


def get_audio_info(audio_path: str) -> dict:
    """
    Return basic metadata about a WAV file without full preprocessing.
    Useful for diagnostic / error messages.
    """
    try:
        import torchaudio

        info = torchaudio.info(str(audio_path))
        return {
            "sample_rate": info.sample_rate,
            "num_channels": info.num_channels,
            "num_frames": info.num_frames,
            "duration_s": round(info.num_frames / info.sample_rate, 2),
            "encoding": str(info.encoding),
        }
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return {
                "sample_rate": info.samplerate,
                "num_channels": info.channels,
                "num_frames": info.frames,
                "duration_s": round(info.duration, 2),
                "encoding": info.subtype,
            }
        except Exception as exc:
            return {"error": str(exc)}
