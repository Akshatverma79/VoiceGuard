"""
VoiceGuard — ml/detector.py
AASIST Anti-Spoofing Detector

Loads the pretrained AASIST model and runs inference on a WAV file.
Returns a structured prediction result with:
  - prediction: "real" or "spoof"
  - spoof_probability: float in [0, 1]
  - inference_time_s: float

The AASIST checkpoint (AASIST.pth, ~17 MB) is automatically downloaded
from the official clovaai/aasist GitHub repository on first use.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────
_ML_DIR = Path(__file__).parent.resolve()
_PRETRAINED_DIR = _ML_DIR / "aasist" / "pretrained"
_DEFAULT_CHECKPOINT = _PRETRAINED_DIR / "AASIST.pth"

# Official checkpoint from clovaai/aasist GitHub repo
_CHECKPOINT_URL = (
    "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"
)

# ── AASIST model hyper-parameters (from config/AASIST.conf) ───────────────
_AASIST_ARGS: Dict[str, Any] = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
    "temperature": 2.0,
    "nb_classes": 2,
}


def _download_checkpoint(dest: Path) -> None:
    """
    Download the AASIST pretrained checkpoint from GitHub.
    Raises RuntimeError if download fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[VoiceGuard] Downloading AASIST checkpoint (~17 MB)…")
    print(f"  Source : {_CHECKPOINT_URL}")
    print(f"  Target : {dest}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100.0)
            sys.stdout.write(f"\r  Progress: {pct:.1f}%   ")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(_CHECKPOINT_URL, str(dest), reporthook=_progress)
        print("\n[VoiceGuard] Checkpoint downloaded successfully.")
    except Exception as exc:
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"\n[VoiceGuard] ❌ Failed to auto-download the AASIST checkpoint.\n"
            f"Error: {exc}\n\n"
            f"Manual setup:\n"
            f"  1. Visit: https://github.com/clovaai/aasist/blob/main/models/weights/AASIST.pth\n"
            f"  2. Download the file (click 'Download raw file')\n"
            f"  3. Place it at: {dest}\n"
        ) from exc


class AASISTDetector:
    """
    Pretrained AASIST detector for voice anti-spoofing.

    Usage:
        detector = AASISTDetector()
        result = detector.predict("path/to/audio.wav")
        # result = {"prediction": "spoof", "spoof_probability": 0.87, "inference_time_s": 1.42}
    """

    def __init__(self, checkpoint_path: str | None = None) -> None:
        """
        Load AASIST model from checkpoint.

        Args:
            checkpoint_path: Path to AASIST.pth. If None, uses the default
                             location and auto-downloads if missing.

        Raises:
            RuntimeError: If the checkpoint cannot be loaded.
        """
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Load and return the AASIST model in eval mode."""
        # Auto-download checkpoint if missing
        if not self._checkpoint_path.exists():
            print(
                f"[VoiceGuard] Checkpoint not found at: {self._checkpoint_path}"
            )
            try:
                _download_checkpoint(self._checkpoint_path)
            except RuntimeError:
                raise

        # Import model architecture
        sys.path.insert(0, str(_ML_DIR))
        try:
            from aasist.model import Model as AASISTModel
        except ImportError as exc:
            raise RuntimeError(
                f"[VoiceGuard] Could not import AASIST model architecture.\n"
                f"Details: {exc}"
            ) from exc

        # Instantiate model
        try:
            model = AASISTModel(d_args=_AASIST_ARGS)
        except Exception as exc:
            raise RuntimeError(
                f"[VoiceGuard] Failed to instantiate AASIST model.\n"
                f"Details: {exc}"
            ) from exc

        # Load weights
        try:
            state_dict = torch.load(
                str(self._checkpoint_path),
                map_location=self._device,
                weights_only=True,
            )
        except TypeError:
            # Older PyTorch versions don't have weights_only
            state_dict = torch.load(
                str(self._checkpoint_path),
                map_location=self._device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[VoiceGuard] Failed to load checkpoint from: {self._checkpoint_path}\n"
                f"The file may be corrupted or not a valid AASIST checkpoint.\n"
                f"Details: {exc}"
            ) from exc

        # Handle state dict wrapped in outer dict (some checkpoints do this)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Strip 'module.' prefix (DataParallel artifacts)
        cleaned: Dict[str, Any] = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v

        try:
            model.load_state_dict(cleaned, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"[VoiceGuard] Checkpoint weights do not match the AASIST architecture.\n"
                f"Details: {exc}"
            ) from exc

        model = model.to(self._device)
        model.eval()

        device_label = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
        print(f"[VoiceGuard] AASIST model loaded — running on {device_label}")
        return model

    def predict(self, wav_path: str) -> Dict[str, Any]:
        """
        Run AASIST inference on a WAV file.

        Args:
            wav_path: Path to a WAV audio file.

        Returns:
            dict with keys:
              - prediction (str): "real" or "spoof"
              - spoof_probability (float): probability in [0.0, 1.0]
              - inference_time_s (float): wall-clock time for inference

        Raises:
            FileNotFoundError: If wav_path does not exist.
            ValueError: If audio loading or preprocessing fails.
            RuntimeError: If model inference fails.
        """
        # Import here to avoid circular import issues
        from preprocessing import load_wav

        # Preprocess audio
        try:
            waveform = load_wav(wav_path)  # (1, 64600)
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"[VoiceGuard] Unexpected error during audio preprocessing.\n"
                f"Details: {exc}"
            ) from exc

        # Add batch dimension: (1, 1, 64600)
        waveform = waveform.unsqueeze(0).to(self._device)

        # Run inference
        t_start = time.perf_counter()
        try:
            with torch.no_grad():
                logits = self._model(waveform)  # (1, 2)
        except Exception as exc:
            raise RuntimeError(
                f"[VoiceGuard] Model inference failed.\n"
                f"Details: {exc}"
            ) from exc
        t_end = time.perf_counter()

        inference_time = round(t_end - t_start, 4)

        # Convert logits to probabilities
        # AASIST output: [bonafide_score, spoof_score]
        # Index 0 → bonafide (real), Index 1 → spoof
        probs = F.softmax(logits, dim=1).squeeze(0)  # (2,)

        spoof_prob = float(probs[1].item())
        real_prob = float(probs[0].item())
        prediction = "spoof" if spoof_prob > real_prob else "real"

        return {
            "prediction": prediction,
            "spoof_probability": round(spoof_prob, 4),
            "real_probability": round(real_prob, 4),
            "inference_time_s": inference_time,
            "device": str(self._device),
        }

    def predict_waveform(self, waveform: "torch.Tensor") -> Dict[str, Any]:
        """
        Run AASIST inference on a pre-processed waveform Tensor.

        This is the Phase 2 entry point used by the chunking pipeline.
        It accepts a tensor directly, avoiding any disk I/O.

        Args:
            waveform: float32 Tensor of shape (1, 64600).
                      Must already be at 16 kHz, mono, and normalized.
                      Produced by AudioChunker.chunk().

        Returns:
            dict with keys:
              - prediction (str): "real" or "spoof"
              - spoof_probability (float): probability in [0.0, 1.0]
              - real_probability (float): probability in [0.0, 1.0]
              - inference_time_s (float): wall-clock time for this chunk

        Raises:
            ValueError: If waveform has incorrect shape.
            RuntimeError: If model inference fails.
        """
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"[VoiceGuard] predict_waveform expects shape (1, N), "
                f"got {tuple(waveform.shape)}"
            )

        # Add batch dimension: (1, 1, N)
        x = waveform.unsqueeze(0).to(self._device)

        t_start = time.perf_counter()
        try:
            with torch.no_grad():
                logits = self._model(x)  # (1, 2)
        except Exception as exc:
            raise RuntimeError(
                f"[VoiceGuard] Model inference failed on waveform tensor.\n"
                f"Details: {exc}"
            ) from exc
        t_end = time.perf_counter()

        inference_time = round(t_end - t_start, 4)

        probs = F.softmax(logits, dim=1).squeeze(0)  # (2,)
        spoof_prob = float(probs[1].item())
        real_prob = float(probs[0].item())
        prediction = "spoof" if spoof_prob > real_prob else "real"

        return {
            "prediction": prediction,
            "spoof_probability": round(spoof_prob, 4),
            "real_probability": round(real_prob, 4),
            "inference_time_s": inference_time,
            "device": str(self._device),
        }

