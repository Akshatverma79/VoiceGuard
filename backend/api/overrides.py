"""
VoiceGuard — backend/api/overrides.py
CRUD REST API for the SQLite audio-hash override cache.

Endpoints:
    GET    /api/overrides          — list all override entries
    POST   /api/overrides          — create / update an override
    DELETE /api/overrides/{hash}   — remove an override by SHA-256 hash
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure backend root is on path
_BACKEND_DIR = Path(__file__).parent.parent.resolve()
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from db.database import upsert_override, list_overrides, delete_override

router = APIRouter(prefix="/api/overrides", tags=["Demo Overrides"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class OverrideCreateRequest(BaseModel):
    sha256_hash: str = Field(..., description="SHA-256 hex digest of the audio file bytes")
    filename: str = Field("", description="Original filename for display")
    spoof_probability: float = Field(0.97, ge=0.0, le=1.0)
    prediction: str = Field("spoof", pattern="^(spoof|real)$")
    risk_level: str = Field("high", pattern="^(low|medium|high)$")
    risk_score: float = Field(0.97, ge=0.0, le=1.0)
    recommendation: str = Field(
        "🚨 AI-generated voice detected (Demo Override). Do NOT proceed — verify through a trusted channel.",
    )
    note: str = Field("", description="Optional free-form label, e.g. 'ElevenLabs clone'")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", summary="List all demo overrides")
async def get_overrides() -> JSONResponse:
    """Return all stored audio-hash overrides (newest first)."""
    return JSONResponse(content=list_overrides())


@router.post("/", summary="Add or update a demo override")
async def create_override(req: OverrideCreateRequest) -> JSONResponse:
    """
    Create or replace the override for a given SHA-256 hash.
    When this hash is uploaded to /api/analyze, the stored scores are
    returned immediately without running AI inference.
    """
    upsert_override(
        sha256_hash=req.sha256_hash,
        filename=req.filename,
        spoof_probability=req.spoof_probability,
        prediction=req.prediction,
        risk_level=req.risk_level,
        risk_score=req.risk_score,
        recommendation=req.recommendation,
        note=req.note,
    )
    return JSONResponse(
        content={
            "status": "ok",
            "sha256_hash": req.sha256_hash,
            "message": f"Override saved for '{req.filename or req.sha256_hash[:12]}…'",
        }
    )


@router.delete("/{sha256_hash}", summary="Remove a demo override")
async def remove_override(sha256_hash: str) -> JSONResponse:
    """Delete the override for the given SHA-256 hash."""
    deleted = delete_override(sha256_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail="Override not found.")
    return JSONResponse(content={"status": "ok", "sha256_hash": sha256_hash})
