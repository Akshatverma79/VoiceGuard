"""
VoiceGuard — ml/detector.py
Wav2Vec2 Deepfake Voice Detector

Replaces previous model with garystafford/wav2vec2-deepfake-voice-detector:
  - Architecture: Hugging Face Wav2Vec2ForSequenceClassification
  - Pretrained weights: garystafford/wav2vec2-deepfake-voice-detector (Hugging Face Hub)
  - Input: 16 kHz mono waveform
  - Output: {"prediction": "real"|"spoof", "spoof_probability": float, "real_probability": float, ...}

Preserves exact drop-in compatibility with the VoiceGuard pipeline (ModelManager, AudioAnalyzer, WebSocket).
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────
_ML_DIR = Path(__file__).parent.resolve()
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

MODEL_ID: str = "garystafford/wav2vec2-deepfake-voice-detector"


class AASISTDetector:
    """
    Drop-in detector replacement using garystafford/wav2vec2-deepfake-voice-detector.
    Preserves exact method signatures and return dictionary format.
    """

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        from transformers import AutoFeatureExtractor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self._model = self._load_model()

        # Resolve classification label indices from config
        id2label = getattr(self._model.config, "id2label", {0: "real", 1: "fake"})
        self._real_idx = 0
        self._fake_idx = 1
        for idx, label in id2label.items():
            l_str = str(label).lower()
            if "fake" in l_str or "spoof" in l_str:
                self._fake_idx = int(idx)
            elif "real" in l_str or "bonafide" in l_str:
                self._real_idx = int(idx)

    def _load_model(self) -> Any:
        """Load and return the model in eval mode."""
        from transformers import AutoModelForAudioClassification

        print(f"[VoiceGuard] Loading {MODEL_ID} from Hugging Face Hub…")
        model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        model = model.to(self._device)
        model.eval()

        device_label = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
        print(f"[VoiceGuard] wav2vec2-deepfake-voice-detector model loaded — running on {device_label}")
        return model

    def predict(self, wav_path: str) -> Dict[str, Any]:
        """
        Run deepfake inference on a WAV file.

        Args:
            wav_path: Path to a WAV audio file.

        Returns:
            dict with keys:
              - prediction (str): "real" or "spoof"
              - spoof_probability (float): probability in [0.0, 1.0]
              - real_probability (float): probability in [0.0, 1.0]
              - inference_time_s (float): wall-clock time for inference
              - device (str): device used
        """
        from preprocessing import load_wav

        waveform = load_wav(wav_path)
        return self.predict_waveform(waveform)

    def predict_waveform(self, waveform: torch.Tensor) -> Dict[str, Any]:
        """
        Run inference on a pre-processed waveform Tensor.

        Args:
            waveform: float32 Tensor of shape (1, N) or (N,).

        Returns:
            dict with prediction, spoof_probability, real_probability, inference_time_s.
        """
        if waveform.dim() == 1:
            x = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            x = waveform
        elif waveform.dim() == 3 and waveform.shape[1] == 1:
            x = waveform.squeeze(1)
        else:
            raise ValueError(
                f"[VoiceGuard] predict_waveform expects shape (1, N) or (N,), "
                f"got {tuple(waveform.shape)}"
            )

        # Reshape to flat 1D numpy array for the feature extractor
        wav_np = x.detach().cpu().reshape(-1).numpy()
        inputs = self._feature_extractor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        t_start = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
        except Exception as exc:
            raise RuntimeError(
                f"[VoiceGuard] Model inference failed on waveform tensor.\n"
                f"Details: {exc}"
            ) from exc
        t_end = time.perf_counter()

        inference_time = round(t_end - t_start, 4)

        probs = F.softmax(logits, dim=-1).squeeze(0)
        spoof_prob = float(probs[self._fake_idx].item())
        real_prob = float(probs[self._real_idx].item())
        prediction = "spoof" if spoof_prob > real_prob else "real"

        return {
            "prediction": prediction,
            "spoof_probability": round(spoof_prob, 4),
            "real_probability": round(real_prob, 4),
            "inference_time_s": inference_time,
            "device": str(self._device),
        }


# Aliases for backward compatibility
AntiDeepfakeDetector = AASISTDetector
Wav2Vec2AntiDeepfakeModel = AASISTDetector

