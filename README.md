# VoiceGuard

> **SIH26104** — AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

VoiceGuard uses the **AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) model to detect whether audio is genuine or a deepfake/voice-clone.

---

## Project Structure

```
VoiceGuard/
├── frontend/          # React + Vite + TypeScript + Tailwind CSS
│   ├── public/
│   │   └── audio-processor.js # AudioWorklet processor (PCM capture) ← Phase 3
│   ├── src/
│   │   ├── hooks/
│   │   │   └── useAudioStream.ts # Real-time streaming hook ← Phase 3
│   │   ├── pages/
│   │   │   └── Live.tsx          # Real-time detection UI ← Phase 3
│   │   ├── types/
│   │   │   └── detection.ts      # WebSocket message types ← Phase 3
│   │   ├── App.tsx               # Navigation tabs (Home / Live)
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── backend/           # FastAPI + Python
│   ├── api/
│   │   ├── health.py        # GET /health
│   │   └── analyze.py       # POST /api/analyze (Phase 2, uses ModelManager singleton)
│   ├── audio/               # Audio processing module ← Phase 2
│   │   ├── processor.py     # Load, mono, resample, normalize
│   │   ├── vad.py           # Energy-based VAD
│   │   └── chunker.py       # AASIST-sized chunking
│   ├── services/
│   │   ├── analyzer.py      # File pipeline orchestrator
│   │   ├── model_manager.py # AASIST model singleton (loaded once) ← Phase 3
│   │   └── risk_engine.py   # Rolling EWMA risk engine ← Phase 3
│   ├── websocket/           # WebSocket streaming ← Phase 3
│   │   ├── audio_buffer.py  # Per-connection sample accumulator
│   │   └── audio_stream.py  # /ws/audio endpoint handler
│   ├── main.py              # FastAPI lifespan & routes
│   └── requirements.txt
├── ml/                # ML layer (AASIST)
│   ├── aasist/
│   │   ├── model.py          # Official AASIST architecture
│   │   └── pretrained/       # AASIST.pth (auto-downloaded)
│   ├── detector.py           # AASISTDetector class
│   ├── preprocessing.py      # Audio loading & preprocessing
│   └── requirements.txt
├── data/
│   ├── real/          # Place real WAV files here for local testing
│   └── fake/          # Place spoof WAV files here for local testing
├── tests/
│   ├── test_detector.py         # Phase 1 single-file test
│   ├── test_audio_processing.py # Phase 2 unit tests
│   ├── evaluate_audio.py        # Phase 2 batch evaluation script
│   └── test_websocket.py       # Phase 3 unit & endpoint tests ← Phase 3
├── docs/
├── .gitignore
└── README.md
```

---

## ⚠️ ASVspoof Dataset — NOT Required

**Phase 1, Phase 2, and Phase 3 do NOT require the ASVspoof dataset.**
You only need WAV audio files of your own or a microphone to test AASIST inference.

---

## Python Setup

Requires **Python 3.9+**.

### ML dependencies (AASIST)

```bash
cd VoiceGuard
pip install -r ml/requirements.txt
```

> **GPU optional** — AASIST will automatically use CUDA if available, otherwise CPU.

### Backend dependencies (FastAPI)

```bash
pip install -r backend/requirements.txt
```

---

## Frontend Setup

Requires **Node.js 18+**.

```bash
cd VoiceGuard/frontend
npm install
npm run dev
```

Open: http://localhost:5173

---

## AASIST Setup

The pretrained AASIST checkpoint (`AASIST.pth`, ~17 MB) is **automatically downloaded** from the official [clovaai/aasist](https://github.com/clovaai/aasist) repository the first time you run inference.

It is saved to:
```
ml/aasist/pretrained/AASIST.pth
```

### Manual download (if auto-download fails)

1. Visit: https://github.com/clovaai/aasist/blob/main/models/weights/AASIST.pth
2. Download `AASIST.pth`
3. Place it in `ml/aasist/pretrained/AASIST.pth`

---

## Phase 3: Real-Time Audio Streaming Architecture

```
Browser Microphone
       │
AudioWorklet (public/audio-processor.js)
  - Captures 128-sample Float32 PCM blocks
       │
WebSocket Binary Frames (/ws/audio)
       │
Backend AudioBuffer (backend/websocket/audio_buffer.py)
  - Accumulates ~4.0 s of raw audio at native rate
  - Sliding window (keeps 1.0 s overlap)
       │
Audio Normalization & VAD Check (Energy-based)
       │ (if speech detected)
AASIST Inference (ml/detector.py via ModelManager singleton)
  - Executed in thread executor (non-blocking)
       │
Rolling Risk Engine (backend/services/risk_engine.py)
  - Exponential Weighted Moving Average (EWMA, α = 0.4)
  - Low (<0.35), Medium (0.35–0.65), High (≥0.65)
  - Stability guard: requires ≥2 speech chunks before emitting HIGH risk
       │
WebSocket JSON Messages → Live UI (/live)
  - Live risk badge, spoof probability, EWMA risk score, recent score history
```

### WebSocket API Protocol (`/ws/audio`)

1. **Connection handshake**:
   - Server sends: `{"type": "status", "status": "connected"}`
2. **Client configuration** (optional, recommended):
   - Client sends JSON: `{"type": "config", "sample_rate": 44100}`
   - Server replies: `{"type": "status", "status": "listening", "native_sample_rate": 44100}`
3. **Streaming audio**:
   - Client sends binary frames of raw float32 PCM samples (128 samples per chunk)
4. **Detection results**:
   - Server emits:
     ```json
     {
       "type": "detection",
       "spoof_probability": 0.02,
       "prediction": "real",
       "risk_score": 0.03,
       "risk_level": "low",
       "chunks_analyzed": 1,
       "timestamp": "2026-09-04T12:00:00.000Z"
     }
     ```

---

## Testing & Verification

### Unit Tests (All Phase 2 & Phase 3 components)

```bash
# Run all 51 automated unit tests:
pytest tests/test_audio_processing.py tests/test_websocket.py -v
```

Tests cover:
- Audio loading, stereo-to-mono, resampling, peak normalization, silence handling
- Energy-based VAD (frames, silence, noise, mixed speech)
- Audio chunker (shapes, 64600 samples, overlap)
- AudioBuffer (push_bytes, sample accumulation, sliding window, memory cap)
- RollingRiskEngine (EWMA decay, minimum chunks guard, risk levels, history capping)
- ModelManager singleton
- WebSocket connection, config exchange, error handling

### Single-file AASIST Test (Phase 1)

```bash
python tests/test_detector.py data/real/sample.wav
```

### Batch Evaluation Script (Phase 2)

```bash
python tests/evaluate_audio.py
```

---

## Phase Completion Status

- [x] **Phase 1**: Core backend + frontend communication, pretrained AASIST integration, single-file testing.
- [x] **Phase 2**: Full audio processing pipeline (mono, resampling, normalization, energy VAD, chunking, aggregation), upload API (`POST /api/analyze`), batch evaluation script.
- [x] **Phase 3**: Real-time WebSocket streaming (`/ws/audio`), AudioWorklet microphone capture, server-side buffering with sliding window, ModelManager singleton, EWMA rolling risk engine, live detection UI (`/live`).
