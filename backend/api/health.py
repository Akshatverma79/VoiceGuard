"""
VoiceGuard Backend — api/health.py
Health check endpoint for Phase 1.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health Check")
async def health_check() -> dict:
    """
    Returns the service health status.
    The React frontend uses this to show Connected / Disconnected.
    """
    return {"status": "ok"}
