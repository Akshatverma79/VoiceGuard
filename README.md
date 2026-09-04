# VoiceGuard

> **SIH26104** — AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

VoiceGuard uses the **AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) model to detect whether audio is genuine or a deepfake/voice-clone.

---

## Project Structure

```
VoiceGuard/
├── frontend/          # React + Vite + TypeScript + Tailwind CSS
├── backend/           # FastAPI + Python
│   ├── api/
│   │   ├── health.py        # GET /health
│   │   └── analyze.py       # POST /api/analyze  ← Phase 2
│   ├── audio/               # Audio processing module ← Phase 2
│   │   ├── processor.py     # Load, mono, resample, normalize
│   │   ├── vad.py           # Energy-based VAD
│   │   └── chunker.py       # AASIST-sized chunking
│   ├── services/
│   │   └── analyzer.py      # Full pipeline orchestrator ← Phase 2
│   ├── main.py
│   └── requirements.txt
├── ml/                # ML layer (AASIST)
│   ├── aasist/
│   │   ├── model.py          # AASIST architecture
│   │   └── pretrained/       # AASIST.pth (auto-downloaded)
│   ├── detector.py           # AASISTDetector class
│   ├── preprocessing.py      # Audio loading & preprocessing
│   └── requirements.txt
├── data/
│   ├── real/          # Place real WAV files here for local testing
│   └── fake/          # Place spoof WAV files here for local testing
├── tests/
│   ├── test_detector.py         # Phase 1 single-file test
│   ├── test_audio_processing.py # Phase 2 unit tests (no model needed)
│   └── evaluate_audio.py        # Phase 2 batch evaluation script
├── docs/
├── .gitignore
└── README.md
```

---

## ⚠️ ASVspoof Dataset — NOT Required

**Neither Phase 1 nor Phase 2 require the ASVspoof dataset.**
You only need WAV audio files of your own to test AASIST inference.

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

It will be saved to:
```
ml/aasist/pretrained/AASIST.pth
```

### Manual download (if auto-download fails)

1. Visit: https://github.com/clovaai/aasist/blob/main/models/weights/AASIST.pth
2. Click **"Download raw file"**
3. Place the file at: `ml/aasist/pretrained/AASIST.pth`

---

## Running the Backend (FastAPI)

```bash
cd VoiceGuard/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify it's running:
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

Interactive API docs: http://localhost:8000/docs

### POST /api/analyze (Phase 2)

Upload a WAV file for full pipeline analysis:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@data/real/sample.wav"
```

Response:
```json
{
  "prediction": "real",
  "spoof_probability": 0.0312,
  "chunks_analyzed": 3
}
```

---

## Audio Processing Pipeline (Phase 2)

The Phase 2 pipeline runs in this order:

```
WAV file
  ↓
AudioProcessor
  → Validate (extension, size, corruption)
  → Stereo → Mono (channel averaging)
  → Resample to 16,000 Hz
  → Peak normalize (no clipping)
  ↓
VoiceActivityDetector (energy-based)
  → Split into 20 ms frames
  → Compute RMS energy per frame
  → Skip inference if audio is silent
  ↓
AudioChunker
  → Split into ≈ 4.04 s segments (64,600 samples each)
  → 1.0 s overlap between chunks
  → Pad short final chunk by repeating
  ↓
AASISTDetector × N  (one call per chunk)
  → Real model inference, no fake values
  ↓
Aggregation (arithmetic mean of spoof probabilities)
  ↓
Prediction + Spoof Probability
```

**Why 4.04 s chunks?**  
AASIST was trained on exactly 64,600-sample windows at 16,000 Hz
(`64600 / 16000 = 4.0375 s`). Every chunk must match this exactly.

**Why arithmetic mean aggregation?**  
The mean balances sensitivity across all chunks. A single suspicious
window raises the score without completely dominating the result, unlike
max-pooling. The strategy is configurable (`AggregationStrategy.MEAN` or
`AggregationStrategy.MAX`).

---

## Testing

### Unit Tests (no model or audio files required)

```bash
# From VoiceGuard root:
pip install pytest
pip install -r ml/requirements.txt
pytest tests/test_audio_processing.py -v
```

Tests cover:
- Mono WAV loading
- Stereo → mono conversion
- Resampling (8 kHz, 44.1 kHz → 16 kHz)
- Silent audio handling
- Very short audio padding
- Invalid/corrupted file handling
- Normalization (peak ≤ 1.0)
- VAD on silent / speech-like / mixed waveforms
- Chunker output shapes and chunk counts
- Overlap increases chunk count

### Single-file Test (Phase 1, requires model + WAV)

```bash
python tests/test_detector.py data/real/sample.wav
```

### Batch Evaluation (Phase 2, requires model + WAV files)

Place WAV files in `data/real/` and/or `data/fake/`, then:

```bash
# From VoiceGuard root:
python tests/evaluate_audio.py
```

This will:
1. Load the AASIST model (auto-downloads checkpoint if needed)
2. Run the full pipeline on every WAV file found
3. Print per-file: prediction, spoof probability, chunks, duration, inference time
4. If ≥ 1 real + ≥ 1 fake file: compute accuracy, precision, recall, F1, confusion matrix

**⚠️ Important limitations:**
- Metrics computed from your sample set do NOT prove model accuracy
- AASIST was trained on ASVspoof LA; performance on other distributions is not guaranteed
- A small sample set is not statistically significant

---

## Phase 1 Checklist

- [x] FastAPI backend with `GET /health`
- [x] React frontend with backend status indicator
- [x] Frontend ↔ Backend communication (CORS configured)
- [x] AASIST model integrated with pretrained checkpoint
- [x] Real inference (no fake/random values)
- [x] Audio preprocessing (mono, 16kHz, 64600 samples)
- [x] Test script with clear output

## Phase 2 Checklist

- [x] Audio loading and validation
- [x] Stereo → mono conversion
- [x] Resampling to 16,000 Hz
- [x] Peak normalization (no clipping, silence handled)
- [x] Energy-based VAD (20 ms frames, language-agnostic)
- [x] Audio chunking (4.04 s / 64,600 samples, 1 s overlap, configurable)
- [x] AASIST processes each chunk via `predict_waveform()`
- [x] Chunk scores aggregated (mean, configurable strategy)
- [x] `POST /api/analyze` endpoint
- [x] Batch evaluation script
- [x] Unit tests (no model or data files required)
- [x] Inference time measured per chunk and total
- [x] README updated

## Phase 3 (Not implemented yet)

- WebSocket streaming
- Live microphone input
- Real-time VAD optimization
- Supabase integration
- Authentication
- Dashboard & analytics
- Replay detection
- Render deployment

---

## References

- AASIST paper: https://arxiv.org/abs/2110.01200
- Official AASIST repo: https://github.com/clovaai/aasist
- FastAPI docs: https://fastapi.tiangolo.com
- Vite docs: https://vitejs.dev
