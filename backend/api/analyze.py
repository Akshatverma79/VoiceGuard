"""
VoiceGuard Backend — backend/api/analyze.py
POST /api/analyze — Phase 2 + Phase 3 audio analysis endpoint.

Phase 3 changes:
  - Uses ModelManager singleton (AASIST loaded once at startup)
  - Adds processing_time_ms to response
  - Cleaner error handling

Accepts a WAV file upload and runs the full pipeline:
  processor → VAD → chunker → AASIST × N → aggregation

Response (200 OK):
    {
        "prediction": "spoof",
        "spoof_probability": 0.87,
        "chunks_analyzed": 4,
        "processing_time_ms": 732
    }

Error responses:
    400 — bad input (silent audio, unsupported format, etc.)
    503 — model not yet loaded
    500 — internal model/pipeline failure
"""

import os
import sys
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Ensure backend root is on path
_BACKEND_DIR = Path(__file__).parent.parent.resolve()
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from services.model_manager import ModelManager
from services.analyzer import AudioAnalyzer, AggregationStrategy

router = APIRouter(prefix="/api", tags=["Analysis"])

# Module-level analyzer singleton (shares the same detector from ModelManager)
_analyzer: AudioAnalyzer | None = None


def _get_analyzer() -> AudioAnalyzer:
    global _analyzer
    if _analyzer is None:
        if not ModelManager.is_ready():
            raise HTTPException(
                status_code=503,
                detail="AASIST model is still loading. Please retry in a few seconds.",
            )
        # Build analyzer with pre-loaded detector
        _analyzer = AudioAnalyzer(
            strategy=AggregationStrategy.MEAN,
            detector=ModelManager.get_detector(),
        )
    return _analyzer


@router.post(
    "/analyze",
    summary="Analyze audio for voice spoofing",
    response_description="Prediction, spoof probability, and timing",
)
async def analyze_audio(
    file: UploadFile = File(..., description="WAV audio file to analyze"),
) -> JSONResponse:
    """
    Upload a WAV audio file and receive a spoofing prediction.

    Returns:
    - **prediction**: `"real"` or `"spoof"`
    - **spoof_probability**: aggregated probability in [0.0, 1.0]
    - **chunks_analyzed**: number of chunks sent to AASIST
    - **processing_time_ms**: total wall-clock processing time
    """
    t_start = time.perf_counter()

    # ── Validate extension ─────────────────────────────────────────────────
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".wav", ".wave"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only WAV files are accepted.",
        )

    # ── Read upload ────────────────────────────────────────────────────────
    try:
        audio_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Write to temp file (torchaudio requires a path) ───────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        analyzer = _get_analyzer()
        result = analyzer.analyze(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Handle pipeline errors ─────────────────────────────────────────────
    if result.error:
        raise HTTPException(status_code=400, detail=result.error)

    processing_ms = round((time.perf_counter() - t_start) * 1000)

    return JSONResponse(content={
        **result.to_api_response(),
        "processing_time_ms": processing_ms,
    })
