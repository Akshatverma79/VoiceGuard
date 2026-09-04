"""
VoiceGuard Backend — backend/api/analyze.py
POST /api/analyze — Phase 2 audio analysis endpoint.

Accepts a WAV file upload and runs the full pipeline:
  processor → VAD → chunker → AASIST × N → aggregation

Response (200 OK):
    {
        "prediction": "spoof",
        "spoof_probability": 0.87,
        "chunks_analyzed": 4
    }

Error responses:
    400 — bad input (silent audio, unsupported format, etc.)
    500 — internal model/pipeline failure
"""

import sys
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Ensure backend root is on path for service imports
_BACKEND_DIR = Path(__file__).parent.parent.resolve()
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from services.analyzer import AudioAnalyzer

router = APIRouter(prefix="/api", tags=["Analysis"])

# Module-level singleton — AASIST model loads once per server lifetime
_analyzer: AudioAnalyzer | None = None


def _get_analyzer() -> AudioAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = AudioAnalyzer()
    return _analyzer


@router.post(
    "/analyze",
    summary="Analyze audio for voice spoofing",
    response_description="Prediction and spoof probability",
)
async def analyze_audio(
    file: UploadFile = File(..., description="WAV audio file to analyze"),
) -> JSONResponse:
    """
    Upload a WAV audio file and receive a spoofing prediction.

    The full pipeline runs:
    1. Load and preprocess audio (mono, 16 kHz)
    2. Voice Activity Detection (skip silent audio)
    3. Chunk into AASIST-sized segments (≈ 4 s each)
    4. Run AASIST on each chunk
    5. Aggregate chunk scores → file-level prediction

    Returns:
    - **prediction**: `"real"` or `"spoof"`
    - **spoof_probability**: aggregated probability in [0.0, 1.0]
    - **chunks_analyzed**: number of chunks sent to AASIST
    """
    # ── Validate MIME type / extension ────────────────────────────────────
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".wav", ".wave"}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                "Only WAV files are accepted in Phase 2."
            ),
        )

    # ── Read upload into memory ────────────────────────────────────────────
    try:
        audio_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Write to a temp file (torchaudio needs a file path) ───────────────
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        analyzer = _get_analyzer()
        result = analyzer.analyze(tmp_path)
    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Handle pipeline errors ─────────────────────────────────────────────
    if result.error:
        raise HTTPException(
            status_code=400,
            detail=result.error,
        )

    return JSONResponse(content=result.to_api_response())
