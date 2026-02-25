# 🎙️ AI Interviewer — Slingshot

An AI-powered mock interview platform that conducts adaptive voice interviews, analyzes your resume, and generates detailed performance reports with personalized feedback.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-green?logo=qt)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

### 🧠 Intelligent Interviewing
- **Adaptive Questions** — AI tailors questions based on your resume, role, and previous answers
- **Multi-Phase Structure** — Intro → Technical → Behavioral → Situational → Wrap-up
- **Real-time Voice Conversation** — Speak naturally, AI listens and responds with voice
- **Configurable Duration** — Set 5, 10, 15, or 20-minute interview sessions

### 📄 Resume Analysis
- **PDF & DOCX Support** — Upload any resume format with PyMuPDF + pdfplumber fallback
- **NLP-Powered Parsing** — Extracts name, skills, education, work experience using spaCy NER
- **Semantic Skill Matching** — Cosine similarity scoring against job role blueprints
- **Progress Indicator** — Real-time percentage progress bar during analysis

### 📊 Performance Report
- **Score Dashboard** — Overall readiness, communication, fluency, confidence, and WPM metrics
- **Skill Readiness Chart** — Visual bar chart of skill match scores
- **Behavioral Timeline** — Confidence and speaking speed over time
- **🤖 AI Feedback** — Personalized "What You Did Well", "Areas for Improvement", and "Topics to Revise"
- **PDF Export** — Download your full report

### 🎤 Voice Pipeline
- **STT** — Groq Whisper Large V3 Turbo with local energy-based VAD
- **LLM** — Groq Llama 3.3 70B for fast, intelligent responses
- **TTS** — ElevenLabs (primary) / Deepgram Aura (fallback) with low-latency streaming
- **Live Indicators** — Mic icon shows listening, processing, and AI speaking states

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+** installed
- **Microphone** connected
- **API Keys** (all have free tiers):

| Service | Purpose | Get Key |
|---------|---------|---------|
| Groq | LLM + STT | [console.groq.com](https://console.groq.com) |
| ElevenLabs | TTS Voice | [elevenlabs.io](https://elevenlabs.io) |
| Deepgram | TTS Fallback | [deepgram.com](https://deepgram.com) |

### Setup

```powershell
# 1. Clone the repo
git clone https://github.com/trigyy/slingshot.git
cd slingshot

# 2. Run the setup script (creates venv, installs deps, downloads models)
.\setup.ps1

# 3. Add your API keys to .env
# Edit .env and paste your keys:
# GROQ_API_KEY=gsk_...
# ELEVENLABS_API_KEY=sk_...
# DEEPGRAM_API_KEY=...

# 4. Launch!
.\run.ps1
```

Or manually:
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m spacy download en_core_web_sm
.\.venv\Scripts\python main.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    PyQt6 GUI                        │
│  Landing Screen → Interview Screen → Report Screen  │
└─────────────┬───────────────────────────┬───────────┘
              │                           │
    ┌─────────▼─────────┐     ┌──────────▼──────────┐
    │  Resume Engine     │     │  Voice Pipeline      │
    │  • PDF/DOCX Extract│     │  • Groq Whisper STT  │
    │  • spaCy NER       │     │  • Groq LLM          │
    │  • Skill Matching  │     │  • ElevenLabs TTS    │
    └────────────────────┘     └──────────────────────┘
                                        │
                               ┌────────▼────────┐
                               │ Behavioral       │
                               │ Analyzer         │
                               │ • Confidence     │
                               │ • WPM / Fluency  │
                               │ • Stutter Detect  │
                               └──────────────────┘
```

### Key Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — loads env, launches PyQt6 app |
| `main_window.py` | All UI screens (Landing, Interview, Report) |
| `voice_pipeline.py` | STT → LLM → TTS orchestration with async pipeline |
| `resume_engine.py` | Resume parsing, NER, semantic skill matching |
| `behavioral_analyzer.py` | Real-time confidence, WPM, stutter detection |
| `audio_manager.py` | PyAudio input/output with low-latency playback |

---

## 🎯 Supported Job Roles

- Software Engineer
- ML Engineer
- Data Scientist
- Frontend Engineer
- Backend Engineer
- DevOps Engineer
- Full Stack Engineer

---

## 📦 Building Desktop Executable

```powershell
pip install pyinstaller
pyinstaller --name "AI Interviewer" --onedir --windowed --noconfirm --add-data ".env;." main.py
```

Output will be in `dist/AI Interviewer/`. Share the entire folder.

---

## 🛠️ Tech Stack

- **UI**: PyQt6 with Material Design 3 styling
- **LLM**: Groq (Llama 3.3 70B Versatile)
- **STT**: Groq Whisper Large V3 Turbo
- **TTS**: ElevenLabs / Deepgram Aura
- **NLP**: spaCy (en_core_web_sm), sentence-transformers
- **Audio**: PyAudio, NumPy
- **Charts**: Matplotlib

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for the AMD Slingshot Hackathon
</p>
