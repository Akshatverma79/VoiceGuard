"""
VoiceGuard Backend — main.py
FastAPI application entry point for Phase 1.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.health import router as health_router

app = FastAPI(
    title="VoiceGuard API",
    description="AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server (localhost:5173) to communicate with us
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
# Routers
# ---------------------------------------------------------------------------
app.include_router(health_router)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"message": "VoiceGuard API is running. See /docs for API documentation."}
