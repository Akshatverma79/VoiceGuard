"""
VoiceGuard — backend/services/model_manager.py
Global AASIST model singleton — loaded ONCE at backend startup.

Design rationale
────────────────
AASIST takes 2–4 seconds to load (checkpoint deserialization + GPU/CPU init).
Loading it per-request or per-WebSocket connection would make the API unusable.
This module holds a single shared AASISTDetector instance that is initialized
during FastAPI's lifespan startup event and reused for every inference call.

Thread safety
─────────────
AASISTDetector.predict_waveform() is stateless after __init__ (all state is in
the PyTorch model weights, which are read-only during inference). Concurrent
calls from multiple WebSocket connections are safe when run through
asyncio.run_in_executor(), because each call holds its own local tensors.
The model itself is not modified during inference.

Usage (in main.py lifespan)::

    from services.model_manager import ModelManager
    await ModelManager.initialize()          # called once at startup
    detector = ModelManager.get_detector()   # called per request/WS
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ── Path setup ─────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent.parent.resolve()
_ROOT_DIR = _BACKEND_DIR.parent.resolve()
_ML_DIR = _ROOT_DIR / "ml"

for _p in [str(_ML_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class ModelManager:
    """
    Singleton manager for the AASISTDetector.

    Call ModelManager.initialize() once (in FastAPI lifespan).
    Call ModelManager.get_detector() anywhere — raises if not initialized.
    """

    _detector: Optional[object] = None  # AASISTDetector, typed as object to avoid circular import
    _initialized: bool = False

    @classmethod
    async def initialize(cls) -> None:
        """
        Load the AASIST model asynchronously (runs blocking load in executor).
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if cls._initialized:
            return

        import asyncio
        loop = asyncio.get_event_loop()

        def _load() -> object:
            from detector import AASISTDetector
            return AASISTDetector()

        print("[VoiceGuard] Loading model at startup...")
        try:
            cls._detector = await loop.run_in_executor(None, _load)
            cls._initialized = True
            print("[VoiceGuard] Model ready.")
        except Exception as exc:
            print(f"[VoiceGuard] [FAIL] Failed to load model: {exc}")
            raise RuntimeError(f"Model failed to load: {exc}") from exc

    @classmethod
    def get_detector(cls) -> object:
        """
        Return the shared AASISTDetector.
        Raises RuntimeError if ModelManager.initialize() has not been called.
        """
        if not cls._initialized or cls._detector is None:
            raise RuntimeError(
                "AASIST model is not loaded. "
                "Call ModelManager.initialize() during application startup."
            )
        return cls._detector

    @classmethod
    def is_ready(cls) -> bool:
        """Return True if the model has been loaded successfully."""
        return cls._initialized and cls._detector is not None
