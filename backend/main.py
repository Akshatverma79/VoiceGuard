"""
VoiceGuard Backend — main.py
FastAPI application entry point — Phase 1 + Phase 2 + Phase 3.

Phase 3 additions:
  - lifespan: loads AASIST model ONCE at startup via ModelManager
  - WebSocket route: /ws/audio
  - Version bump to 3.0.0
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure backend root is on sys.path
_BACKEND_DIR = Path(__file__).parent.resolve()
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.health import router as health_router
from api.analyze import router as analyze_router
from services.model_manager import ModelManager
from websocket.audio_stream import handle_audio_stream


# ---------------------------------------------------------------------------
# Lifespan: load AASIST model ONCE at startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads the AASIST model at startup so all routes share one instance.
    """
    await ModelManager.initialize()
    yield
    # Nothing to release — PyTorch handles its own cleanup


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VoiceGuard API",
    description="AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks",
    version="3.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server to communicate with us
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# HTTP Routers
# ---------------------------------------------------------------------------
app.include_router(health_router)
app.include_router(analyze_router)


# ---------------------------------------------------------------------------
# WebSocket — /ws/audio
# ---------------------------------------------------------------------------
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """
    Real-time audio streaming endpoint.

    Protocol:
      1. Connect
      2. Send JSON: { "type": "config", "sample_rate": <int> }
      3. Send binary frames: Float32Array PCM samples at native sample rate
      4. Receive JSON detection results
      5. Send JSON: { "type": "stop" } or just disconnect
    """
    await handle_audio_stream(websocket)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "message": "VoiceGuard API is running. See /docs for API documentation.",
        "model_ready": ModelManager.is_ready(),
    }
