# AI Interviewer — System Architecture

## Overview

A zero-cost, real-time voice AI interviewer that conducts adaptive technical vivas, analyzes speech behavior, and generates comprehensive evaluation reports — all powered by generous free-tier cloud APIs and local processing.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AI INTERVIEWER SYSTEM                        │
│                                                                       │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │   PyQt GUI   │────▶│          Session Orchestrator             │  │
│  │  (Thread-    │◀────│        (asyncio event loop in            │  │
│  │   safe       │     │         QThread worker)                  │  │
│  │   Signals)   │     └───────────────┬──────────────────────────┘  │
│  └──────────────┘                     │                              │
│                          ┌────────────┼────────────┐                 │
│                          │            │            │                 │
│              ┌───────────▼──┐  ┌──────▼──────┐  ┌─▼──────────────┐ │
│              │   MODULE 1   │  │  MODULE 2   │  │   MODULE 3      │ │
│              │   Resume     │  │  Voice      │  │   Behavioral    │ │
│              │   Intelligence│  │  Pipeline   │  │   Analyzer      │ │
│              │  (Local NLP) │  │(Cloud APIs) │  │(Audio Thread)   │ │
│              └──────────────┘  └─────────────┘  └────────────────┘ │
│                                       │                              │
│                       ┌───────────────┼───────────────┐             │
│                       │               │               │             │
│                  ┌────▼────┐   ┌──────▼────┐   ┌─────▼────┐       │
│                  │Deepgram │   │  Groq API  │   │ElevenLabs│       │
│                  │Nova-3   │   │Llama 3.3   │   │  TTS /   │       │
│                  │  STT    │   │  70B LLM   │   │Deepgram  │       │
│                  │WebSocket│   │  REST API  │   │  Aura    │       │
│                  └─────────┘   └────────────┘   └──────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## Free Tier Budget

| Service       | Free Allowance                        | Monthly Cost |
|---------------|---------------------------------------|--------------|
| Groq API      | 14,400 req/day, 500K tokens/day (free)| $0           |
| Deepgram      | $200 developer credit                 | $0           |
| ElevenLabs    | 10,000 chars/month                    | $0           |
| HuggingFace   | Inference API (rate-limited)          | $0           |
| spaCy / local | Runs locally                          | $0           |

## Module Breakdown

### Module 1 — Resume Intelligence Engine
- **Extraction**: PyMuPDF (fitz) for PDF, python-docx for DOCX
- **NER**: spaCy `en_core_web_sm` → skills, roles, orgs, tech stack
- **Semantic Matching**: `sentence-transformers/all-MiniLM-L6-v2` (local, ~80MB)
  - Cosine similarity against job role skill vectors
  - Produces per-skill readiness scores (0–1)
- **Output**: Structured candidate profile JSON fed to the LLM system prompt

### Module 2 — Voice Pipeline (Cloud WebSockets)
- **Microphone Capture**: PyAudio (16kHz, 16-bit, mono)
- **STT**: Deepgram Nova-3 via persistent WebSocket
  - Interim results for live transcript display
  - Final results trigger LLM inference
- **LLM**: Groq (Llama 3.3 70B) — streaming REST
  - Dynamic system prompt built from resume profile
  - Tracks interview state machine (intro → technical → behavioral → wrap-up)
  - Adaptive follow-up questions based on detected skill gaps
- **TTS**: ElevenLabs streaming or Deepgram Aura (fallback)
  - Audio chunks played via PyAudio output stream as they arrive
- **Latency Target**: < 800ms end-to-end (STT final → audio playback start)

### Module 3 — Behavioral Analysis
- **Raw Audio Thread**: Captures 30ms frames independently
- **MFCC Extraction**: librosa (13 coefficients + delta + delta-delta)
- **Disfluency Detection**: Lightweight sklearn classifier
  - Features: MFCC variance, zero-crossing rate, energy ratio, pause duration
  - Labels: fluent / stutter / filler / long-pause / repetition
  - Pre-trained on SEP-28k dataset features (model bundled)
- **Metrics Produced**:
  - Words per minute (WPM)
  - Stutter frequency (events/minute)
  - Filler word rate
  - Mean response latency
  - Confidence index (0–100) — composite score
  - Emotional valence (via pitch + energy trajectory)

### Module 4 — PyQt GUI
- **Stack**: PyQt6 + pyqtgraph + matplotlib
- **Aesthetic**: Dark glassmorphic terminal-meets-dashboard, monospace + geometric sans
- **Screens**:
  1. **Landing** — Resume drop zone + job role selector
  2. **Interview** — Live waveform, transcript feed, question display, mic status
  3. **Report** — Radar chart (skills), bar chart (behavioral), timeline, PDF export

## Threading Model

```
Main Thread (Qt Event Loop)
    │
    ├── QThread: AsyncWorker
    │       └── asyncio event loop
    │               ├── deepgram_ws_handler()     # STT WebSocket
    │               ├── tts_stream_handler()       # TTS audio chunks
    │               └── llm_query_handler()        # Groq REST
    │
    ├── QThread: BehavioralAnalyzer
    │       └── PyAudio callback thread
    │               └── MFCC + classifier pipeline
    │
    └── Qt Signals/Slots (thread-safe IPC)
            ├── transcript_updated(str)
            ├── ai_response_ready(str)
            ├── audio_chunk_ready(bytes)
            ├── behavioral_update(dict)
            └── interview_complete(dict)
```

## API Keys Required (all free)
1. `GROQ_API_KEY` — https://console.groq.com
2. `DEEPGRAM_API_KEY` — https://console.deepgram.com ($200 credit on signup)
3. `ELEVENLABS_API_KEY` — https://elevenlabs.io/app/sign-up

## Installation

```bash
pip install PyQt6 pyqtgraph matplotlib pyaudio websockets httpx \
            pymupdf python-docx spacy sentence-transformers \
            librosa scikit-learn numpy sounddevice groq \
            elevenlabs python-dotenv reportlab

python -m spacy download en_core_web_sm
```
