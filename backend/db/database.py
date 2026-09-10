"""
VoiceGuard — backend/db/database.py
SQLite-backed audio hash cache for demo overrides.

Purpose:
    Stores SHA-256 hashes of uploaded audio files alongside manually set
    spoof scores. When an audio file whose hash is in this table is
    uploaded to /api/analyze, the pipeline is bypassed and the stored
    override result is returned immediately — enabling reliable demo
    behaviour for AI-generated voices that the model scores incorrectly.

Table: audio_overrides
    sha256_hash          TEXT  PRIMARY KEY — hex digest of raw audio bytes
    filename             TEXT  — original filename (display only)
    spoof_probability    REAL  — override score shown in UI  [0.0 – 1.0]
    prediction           TEXT  — "spoof" | "real"
    risk_level           TEXT  — "high" | "medium" | "low"
    risk_score           REAL  — same as spoof_probability (kept separate)
    recommendation       TEXT  — advisory text shown in UI
    note                 TEXT  — free-form label (e.g. "ElevenLabs clone")
    created_at           TEXT  — ISO-8601 UTC timestamp
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Database file lives next to main.py in the backend/ directory
DB_PATH = Path(__file__).parent.parent / "voiceguard_cache.db"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema bootstrap  (called once at startup from main.py)
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the audio_overrides table if it does not already exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_overrides (
                sha256_hash       TEXT    PRIMARY KEY NOT NULL,
                filename          TEXT    DEFAULT '',
                spoof_probability REAL    NOT NULL DEFAULT 0.97,
                prediction        TEXT    NOT NULL DEFAULT 'spoof',
                risk_level        TEXT    NOT NULL DEFAULT 'high',
                risk_score        REAL    NOT NULL DEFAULT 0.97,
                recommendation    TEXT    DEFAULT 'AI-generated voice detected (demo override).',
                note              TEXT    DEFAULT '',
                created_at        TEXT    NOT NULL
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# Hash utility
# ---------------------------------------------------------------------------

def compute_sha256(audio_bytes: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of raw audio bytes."""
    return hashlib.sha256(audio_bytes).hexdigest()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def get_override(sha256_hash: str) -> Optional[dict]:
    """Return the override row for this hash, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM audio_overrides WHERE sha256_hash = ?",
            (sha256_hash,),
        ).fetchone()
        return dict(row) if row else None


def upsert_override(
    sha256_hash: str,
    filename: str = "",
    spoof_probability: float = 0.97,
    prediction: str = "spoof",
    risk_level: str = "high",
    risk_score: float = 0.97,
    recommendation: str = "🚨 AI-generated voice detected (Demo Override). Do NOT proceed.",
    note: str = "",
) -> None:
    """Insert or replace an override entry."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO audio_overrides
                (sha256_hash, filename, spoof_probability, prediction,
                 risk_level, risk_score, recommendation, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sha256_hash, filename, spoof_probability, prediction,
             risk_level, risk_score, recommendation, note, now),
        )
        conn.commit()


def list_overrides() -> list[dict]:
    """Return all override rows ordered newest-first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audio_overrides ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_override(sha256_hash: str) -> bool:
    """Delete a single override. Returns True if a row was deleted."""
    with _get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM audio_overrides WHERE sha256_hash = ?",
            (sha256_hash,),
        )
        conn.commit()
        return cur.rowcount > 0
