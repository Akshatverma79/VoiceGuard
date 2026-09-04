"""
VoiceGuard — ml/detector.py
Wav2Vec2-Small-AntiDeepfake Anti-Spoofing Detector

Replaces AASIST with nii-yamagishilab/wav2vec-small-anti-deepfake:
  - Architecture: Hugging Face Wav2Vec2Model backbone + AdaptiveAvgPool1d + Linear classifier
  - Pretrained weights: nii-yamagishilab/wav2vec-small-anti-deepfake (Hugging Face Hub)
  - Input: 16 kHz mono waveform
  - Output: {"prediction": "real"|"spoof", "spoof_probability": float, "real_probability": float, ...}

Preserves exact drop-in compatibility with the VoiceGuard pipeline (ModelManager, AudioAnalyzer, WebSocket).
"""

import re
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

MODEL_ID: str = "nii-yamagishilab/wav2vec-small-anti-deepfake"


class Wav2Vec2AntiDeepfakeModel(nn.Module):
    """
    Wav2Vec2-Small-AntiDeepfake network architecture:
      - Feature Extractor & Transformer: Wav2Vec2Model (facebook/wav2vec2-base post-trained)
      - Pooling: AdaptiveAvgPool1d(1) across time
      - Classifier: Linear(768 -> 2)
        Index 0: Fake (Spoof)
        Index 1: Real (Bonafide)
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        super().__init__()
        from transformers import AutoConfig, Wav2Vec2Model

        self.config = AutoConfig.from_pretrained(model_id)
        self.wav2vec2 = Wav2Vec2Model(self.config)
        self.adap_pool1d = nn.AdaptiveAvgPool1d(1)
        self.proj_fc = nn.Linear(768, 2)
        self._load_safetensors_weights(model_id)

    def _load_safetensors_weights(self, model_id: str) -> None:
        """Download and load weights from Hugging Face safetensors."""
        import safetensors.torch as st
        from huggingface_hub import hf_hub_download

        print(f"[VoiceGuard] Loading weights for {model_id} from Hugging Face Hub…")
        weights_path = hf_hub_download(model_id, "model.safetensors")
        raw_sd = st.load_file(weights_path)

        expected_keys = set(self.wav2vec2.state_dict().keys())
        hf_sd: Dict[str, torch.Tensor] = {}

        for k, v in raw_sd.items():
            if k == "proj_fc.weight":
                self.proj_fc.weight.data.copy_(v)
                continue
            if k == "proj_fc.bias":
                self.proj_fc.bias.data.copy_(v)
                continue

            # Strip custom namespace prefix
            if k.startswith("m_ssl.model."):
                k = k[len("m_ssl.model."):]

            new_k = k
            new_k = re.sub(r"feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv.", new_k)
            new_k = re.sub(r"feature_extractor\.conv_layers\.0\.2\.", r"feature_extractor.conv_layers.0.layer_norm.", new_k)
            new_k = new_k.replace("post_extract_proj.", "feature_projection.projection.")
            if new_k.startswith("layer_norm."):
                new_k = new_k.replace("layer_norm.", "feature_projection.layer_norm.")
            new_k = new_k.replace("encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.parametrizations.weight.original0")
            new_k = new_k.replace("encoder.pos_conv.0.weight_v", "encoder.pos_conv_embed.conv.parametrizations.weight.original1")
            new_k = new_k.replace("encoder.pos_conv.0.", "encoder.pos_conv_embed.conv.")
            new_k = re.sub(r"encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention.", new_k)
            new_k = re.sub(r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"encoder.layers.\1.layer_norm.", new_k)
            new_k = re.sub(r"encoder\.layers\.(\d+)\.fc1\.", r"encoder.layers.\1.feed_forward.intermediate_dense.", new_k)
            new_k = re.sub(r"encoder\.layers\.(\d+)\.fc2\.", r"encoder.layers.\1.feed_forward.output_dense.", new_k)
            if new_k == "mask_emb":
                new_k = "masked_spec_embed"

            if new_k in expected_keys:
                hf_sd[new_k] = v

        self.wav2vec2.load_state_dict(hf_sd, strict=True)
        print(f"[VoiceGuard] Successfully loaded Wav2Vec2-Small-AntiDeepfake weights.")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            wav: (B, T) float32 tensor at 16 kHz
        Returns:
            logits: (B, 2) where index 0 = fake/spoof, index 1 = real/bonafide
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)

        # Normalize waveform per instance (layer norm across time)
        wav = F.layer_norm(wav, (wav.shape[-1],))

        outputs = self.wav2vec2(wav)
        emb = outputs.last_hidden_state           # [B, T, 768]
        emb = emb.transpose(1, 2)                 # [B, 768, T]
        pooled = self.adap_pool1d(emb).squeeze(-1)  # [B, 768]
        logits = self.proj_fc(pooled)             # [B, 2]
        return logits


class AASISTDetector:
    """
    Drop-in detector replacement using Wav2Vec2-Small-AntiDeepfake.
    Preserves exact method signatures and return dictionary format.
    """

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model()

    def _load_model(self) -> Wav2Vec2AntiDeepfakeModel:
        """Load and return the model in eval mode."""
        model = Wav2Vec2AntiDeepfakeModel()
        model = model.to(self._device)
        model.eval()

        device_label = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
        print(f"[VoiceGuard] Wav2Vec2-Small-AntiDeepfake model loaded — running on {device_label}")
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

        x = x.to(self._device)

        t_start = time.perf_counter()
        try:
            with torch.no_grad():
                logits = self._model(x)
        except Exception as exc:
            raise RuntimeError(
                f"[VoiceGuard] Model inference failed on waveform tensor.\n"
                f"Details: {exc}"
            ) from exc
        t_end = time.perf_counter()

        inference_time = round(t_end - t_start, 4)

        # Output logits: [bonafide_score, spoof_score]
        # Index 0 -> real (bonafide), Index 1 -> spoof (fake)
        probs = F.softmax(logits, dim=1).squeeze(0)

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


# Alias for explicit naming while keeping backward compatibility
AntiDeepfakeDetector = AASISTDetector
