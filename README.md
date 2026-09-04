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
│   │   └── health.py
│   ├── services/
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
│   └── test_detector.py
├── docs/
├── .gitignore
└── README.md
```

---

## ⚠️ ASVspoof Dataset — NOT Required

**Phase 1 does NOT require the ASVspoof dataset.**
You only need a single WAV audio file to test AASIST inference.

---

## Python Setup

Requires **Python 3.9+**.

### ML dependencies (AASIST)

```bash
cd VoiceGuard
pip install torch torchaudio numpy
# Or use the requirements file:
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
```

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

---

## Running the Frontend (React)

```bash
cd VoiceGuard/frontend
npm run dev
```

Open: http://localhost:5173

The landing page will show **Connected** when the FastAPI backend is running.

---

## Testing AASIST

### Step 1 — Add a test audio file

Place any WAV file in `data/real/` or `data/fake/`:
```
data/real/sample.wav   (a real voice recording)
data/fake/spoof.wav    (a TTS/voice-cloned recording)
```

### Step 2 — Run the test script

```bash
# From the VoiceGuard root directory:
python tests/test_detector.py data/real/sample.wav
```

Example output:
```
=======================================================
  VoiceGuard — AASIST Anti-Spoofing Test
=======================================================

[1/3] Loading AASIST model…
[VoiceGuard] Downloading AASIST checkpoint (~17 MB)…
[VoiceGuard] AASIST model loaded — running on CPU
    Model loaded in 3.21s

[2/3] Running inference on: /path/to/data/real/sample.wav

[3/3] Result:
-------------------------------------------------------
  Audio            : sample.wav
  Prediction       : REAL
  Spoof Probability: 0.0312  (3.12%)
  Real  Probability: 0.9688  (96.88%)
  Inference Time   : 1.42 seconds
  Device           : cpu
-------------------------------------------------------

  ✅ VERDICT: REAL — No spoofing detected.
=======================================================
```

---

## Phase 1 Checklist

- [x] FastAPI backend with `GET /health`
- [x] React frontend with backend status indicator
- [x] Frontend ↔ Backend communication (CORS configured)
- [x] AASIST model integrated with pretrained checkpoint
- [x] Real inference (no fake/random values)
- [x] Audio preprocessing (mono, 16kHz, 64600 samples)
- [x] Test script with clear output

## Phase 2 (Not implemented yet)

- WebSocket streaming
- Microphone input
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
