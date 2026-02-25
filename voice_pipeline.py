"""
Module 2: Core AI & Voice Pipeline
====================================
Orchestrates real-time voice interviewing via:
  - Deepgram Nova-3 (WebSocket STT)
  - Groq Llama 3.3 70B (streaming LLM)
  - ElevenLabs / Deepgram Aura (streaming TTS)

Threading model: This module runs inside a QThread worker that spins
its own asyncio event loop. All communication back to the PyQt main
thread is via Qt Signals (see SessionWorker at bottom).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, AsyncGenerator, Callable

import httpx
import numpy as np
import websockets

logger = logging.getLogger(__name__)

# ─── Interview State Machine ──────────────────────────────────────────────────

class InterviewPhase(Enum):
    INTRO = auto()
    WARM_UP = auto()
    TECHNICAL_CORE = auto()
    BEHAVIORAL = auto()
    SKILL_PROBE = auto()        # Adaptive deep-dive on weak skills
    WRAP_UP = auto()
    COMPLETE = auto()

PHASE_SEQUENCE = [
    InterviewPhase.INTRO,
    InterviewPhase.WARM_UP,
    InterviewPhase.TECHNICAL_CORE,
    InterviewPhase.BEHAVIORAL,
    InterviewPhase.SKILL_PROBE,
    InterviewPhase.WRAP_UP,
]

PHASE_QUESTION_BUDGETS = {
    InterviewPhase.INTRO: 1,
    InterviewPhase.WARM_UP: 2,
    InterviewPhase.TECHNICAL_CORE: 4,
    InterviewPhase.BEHAVIORAL: 3,
    InterviewPhase.SKILL_PROBE: 3,
    InterviewPhase.WRAP_UP: 1,
}


# ─── System Prompt Builder ────────────────────────────────────────────────────

def build_system_prompt(candidate_profile_summary: str, job_role: str) -> str:
    return f"""You are a highly experienced Senior Technical Interviewer at a top-tier technology company. Your role is to conduct a structured yet highly adaptive viva interview.

CANDIDATE PROFILE:
{candidate_profile_summary}

TARGET ROLE: {job_role}

INTERVIEW GUIDELINES:
1. Ask ONE focused question at a time. Never ask multiple questions in the same turn.
2. Listen carefully to the candidate's answer before proceeding. 
3. THIS IS CRITICAL: Do NOT just read down a generic list of questions. You MUST generate your next question based specifically on what the candidate just said.
4. If the candidate mentions a specific technology, concept, or project superficially, your very next question MUST be a follow-up probing deeper into that specific topic.
5. If they demonstrate strong knowledge, acknowledge briefly and advance. If weak, probe but don't embarrass.
6. Maintain a professional yet warm and encouraging tone throughout.
7. Track what the candidate has answered and don't repeat topics.
8. Keep each question concise — 1-3 sentences maximum.
9. At the WRAP_UP phase, summarize 1-2 key strengths and 1-2 areas for growth.
10. DO NOT reveal that you have read their resume unless they ask directly.

RESPONSE FORMAT:
- Respond ONLY with what you would say aloud.
- No stage directions, no markdown, no internal thoughts.
- Keep responses under 80 words unless doing a final summary.
- For the INTRO phase, greet the candidate warmly and ask them to briefly introduce themselves.
"""

def build_phase_directive(phase: InterviewPhase, weak_skills: list[str]) -> str:
    directives = {
        InterviewPhase.INTRO: "Greet the candidate warmly. Ask them to introduce themselves and their background.",
        InterviewPhase.WARM_UP: "Ask a gentle warm-up question about their most recent project or role. Ask a direct follow up based on what they just said before moving on.",
        InterviewPhase.TECHNICAL_CORE: f"Engage in a technical discussion regarding {', '.join(weak_skills[:2]) if weak_skills else 'role fundamentals'}. Your goal is to drill down into the specific terms and answers they provide in the ongoing conversation.",
        InterviewPhase.BEHAVIORAL: "Ask a behavioral STAR-method question about teamwork, conflict, or challenge. Pay close attention to their answer and immediately ask them to clarify a specific point they made.",
        InterviewPhase.SKILL_PROBE: f"Probe specifically on identified weak areas: {', '.join(weak_skills[:3]) if weak_skills else 'general problem-solving'}. Be highly adaptive based on their responses.",
        InterviewPhase.WRAP_UP: "Thank the candidate. Provide 1-2 specific strengths and 1-2 growth areas. Ask if they have questions.",
    }
    return directives.get(phase, "Continue the interview naturally, generating follow-ups based strictly on the candidate's previous statements.")


# ─── Groq LLM Client ─────────────────────────────────────────────────────────

class GroqLLMClient:
    """
    Calls Groq's OpenAI-compatible API with streaming.
    Rate limits: 14,400 req/day, 500K tokens/day on free tier.
    Automatically falls back to Gemini Flash if Groq returns 429.
    """

    GROQ_BASE = "https://api.groq.com/openai/v1"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
    GEMINI_MODEL = "gemini-1.5-flash-latest"

    def __init__(self):
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self._groq_rate_limited = False
        self._rate_limit_reset = 0.0

    async def stream_response(
        self,
        system_prompt: str,
        conversation_history: list[dict],
        phase_directive: str,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Groq (or Gemini fallback)."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[-12:])  # Keep last 12 turns for context window
        messages.append({"role": "user", "content": f"[PHASE DIRECTIVE: {phase_directive}]"})

        if self._groq_rate_limited and time.time() < self._rate_limit_reset:
            logger.warning("Groq rate limited, using Gemini fallback")
            async for token in self._stream_gemini(messages):
                yield token
            return

        try:
            async for token in self._stream_groq(messages):
                yield token
        except RateLimitError:
            self._groq_rate_limited = True
            self._rate_limit_reset = time.time() + 60
            logger.warning("Groq 429: switching to Gemini for 60s")
            async for token in self._stream_gemini(messages):
                yield token

    async def _stream_groq(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", f"{self.GROQ_BASE}/chat/completions",
                                     headers=headers, json=payload) as response:
                if response.status_code == 429:
                    raise RateLimitError("Groq rate limit hit")
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError):
                            continue

    async def _stream_gemini(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Gemini 1.5 Flash free tier fallback (1M tokens/month free)."""
        if not self.gemini_key:
            yield "[API key not configured. Please set GEMINI_API_KEY.]"
            return

        # Convert OpenAI message format to Gemini format
        gemini_contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # Handle as first user message
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        system_text = next((m["content"] for m in messages if m["role"] == "system"), "")
        payload = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": gemini_contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 200},
        }

        url = f"{self.GEMINI_BASE}/{self.GEMINI_MODEL}:streamGenerateContent?key={self.gemini_key}&alt=sse"
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                            yield text
                        except (json.JSONDecodeError, KeyError):
                            continue


class RateLimitError(Exception):
    pass


# ─── Deepgram STT WebSocket Client ───────────────────────────────────────────

class DeepgramSTTClient:
    """
    Connects to Deepgram Nova-3 via WebSocket for real-time transcription.
    Uses $200 developer credit — sufficient for hundreds of hours of audio.
    Provides both interim (live) and final transcription results.
    """

    WS_URL = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&encoding=linear16"
        "&sample_rate=16000"
        "&channels=1"
        "&interim_results=true"
        "&endpointing=150"       # 150ms silence = utterance end (was 300ms)
        "&utterance_end_ms=1000" # Deepgram minimum; keep at 1000ms
        "&smart_format=true"
        "&punctuate=true"
        "&disfluencies=false"    # We handle disfluency detection locally
        "&keepalive=true"
    )

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    async def connect(self):
        headers = {"Authorization": f"Token {self.api_key}"}
        self._ws = await websockets.connect(
            self.WS_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
            max_size=2**20
        )
        self._connected = True
        logger.info("Deepgram STT WebSocket connected")

    async def send_audio(self, audio_bytes: bytes):
        """Push raw PCM bytes to Deepgram. Call from audio capture callback."""
        if self._connected and self._ws:
            try:
                await self._ws.send(audio_bytes)
            except websockets.ConnectionClosed:
                self._connected = False

    async def receive_transcripts(
        self,
        on_interim: Callable[[str], None],
        on_final: Callable[[str], None],
        stop_event: asyncio.Event,
    ):
        """
        Continuously read transcript events from Deepgram.
        Calls on_interim for live partial text, on_final for complete utterances.
        """
        async for message in self._ws:
            if stop_event.is_set():
                break
            try:
                event = json.loads(message)
                msg_type = event.get("type", "")

                if msg_type == "Results":
                    channel = event.get("channel", {})
                    alternatives = channel.get("alternatives", [{}])
                    transcript = alternatives[0].get("transcript", "").strip()
                    is_final = event.get("is_final", False)
                    speech_final = event.get("speech_final", False)

                    if not transcript:
                        continue

                    if is_final and speech_final:
                        on_final(transcript)
                    elif not is_final:
                        on_interim(transcript)

                elif msg_type == "Error":
                    logger.error(f"Deepgram error: {event.get('message')}")

            except json.JSONDecodeError:
                continue

    async def close(self):
        if self._ws and self._connected:
            await self._ws.send(json.dumps({"type": "CloseStream"}))
            await self._ws.close()
            self._connected = False


# ─── ElevenLabs TTS Client ────────────────────────────────────────────────────

class ElevenLabsTTSClient:
    """
    Streams TTS audio from ElevenLabs.
    Free tier: 10,000 chars/month.
    Falls back to Deepgram Aura (also generous free tier) if ElevenLabs is exhausted.
    """

    ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"

    # Thoughtful default: "Adam" — professional, neutral male voice
    DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"

    def __init__(self, elevenlabs_key: str, deepgram_key: str, voice_id: str = None):
        self.el_key = elevenlabs_key
        self.dg_key = deepgram_key
        self.voice_id = voice_id or self.DEFAULT_VOICE_ID
        self._use_deepgram_fallback = False
        # Persistent HTTP client — avoids TCP handshake latency on every sentence
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Yield audio chunks (MP3) as they stream from the API."""
        if self._use_deepgram_fallback:
            async for chunk in self._deepgram_tts(text):
                yield chunk
            return

        try:
            async for chunk in self._elevenlabs_tts(text):
                yield chunk
        except (httpx.HTTPStatusError, Exception) as e:
            logger.warning(f"ElevenLabs failed ({e}), switching to Deepgram Aura")
            self._use_deepgram_fallback = True
            async for chunk in self._deepgram_tts(text):
                yield chunk

    async def _elevenlabs_tts(self, text: str) -> AsyncGenerator[bytes, None]:
        url = self.ELEVENLABS_URL.format(voice_id=self.voice_id)
        headers = {
            "xi-api-key": self.el_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",   # Lowest latency model
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.75,
                "style": 0.3,
                "use_speaker_boost": True,
            },
            "output_format": "pcm_16000",    # Raw PCM for direct playback
        }

        client = await self._get_client()
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=1024):  # 1024 = 32ms chunks
                if chunk:
                    yield chunk

    async def _deepgram_tts(self, text: str) -> AsyncGenerator[bytes, None]:
        """Deepgram Aura TTS — covered by the $200 dev credit."""
        headers = {
            "Authorization": f"Token {self.dg_key}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}
        url = f"{self.DEEPGRAM_TTS_URL}?model=aura-asteria-en&encoding=linear16&sample_rate=16000"

        client = await self._get_client()
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=1024):
                if chunk:
                    yield chunk


# ─── Interview Session State ──────────────────────────────────────────────────

@dataclass
class InterviewSession:
    candidate_name: str
    job_role: str
    system_prompt: str
    weak_skills: list[str]
    conversation_history: list[dict] = field(default_factory=list)
    current_phase: InterviewPhase = InterviewPhase.INTRO
    phase_question_count: int = 0
    total_questions: int = 0
    start_time: float = field(default_factory=time.time)
    response_times: list[float] = field(default_factory=list)
    last_question_time: float = 0.0

    def add_ai_message(self, text: str):
        self.conversation_history.append({"role": "assistant", "content": text})
        self.last_question_time = time.time()

    def add_candidate_message(self, text: str):
        response_time = time.time() - self.last_question_time if self.last_question_time > 0 else 0
        self.response_times.append(response_time)
        self.conversation_history.append({"role": "user", "content": text})

    def should_advance_phase(self) -> bool:
        budget = PHASE_QUESTION_BUDGETS.get(self.current_phase, 1)
        return self.phase_question_count >= budget

    def advance_phase(self):
        idx = PHASE_SEQUENCE.index(self.current_phase)
        if idx + 1 < len(PHASE_SEQUENCE):
            self.current_phase = PHASE_SEQUENCE[idx + 1]
            self.phase_question_count = 0
            logger.info(f"Interview advanced to phase: {self.current_phase.name}")
        else:
            self.current_phase = InterviewPhase.COMPLETE


# ─── Groq Whisper STT Client (replaces Deepgram) ─────────────────────────────

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Convert raw 16-bit PCM bytes into an in-memory WAV file."""
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class GroqWhisperSTTClient:
    """
    High-accuracy STT using Groq Whisper Large V3 Turbo.
    Implements a local energy-based VAD to detect speech boundaries,
    then fires off batch Whisper API calls with sub-second latency.
    """
    SPEECH_RMS_THRESHOLD = 180   # RMS above this = speech (tune 100-400)
    SILENCE_CHUNKS_NEEDED = 22   # ~22 * 32ms = ~700ms of silence ends utterance
    MIN_SPEECH_CHUNKS = 10       # ~320ms minimum to avoid spurious triggers

    def __init__(self, groq_api_key: str):
        self._api_key = groq_api_key
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    def push_frame_threadsafe(self, pcm_bytes: bytes, loop: asyncio.AbstractEventLoop):
        """Called from PyAudio OS thread — routes frame into the asyncio queue safely."""
        try:
            loop.call_soon_threadsafe(self._audio_queue.put_nowait, pcm_bytes)
        except (RuntimeError, asyncio.QueueFull):
            pass

    async def run(
        self,
        on_interim,
        on_final,
        stop_event: asyncio.Event,
    ):
        """Main VAD + transcription coroutine. Run as a task inside the orchestrator."""
        from groq import AsyncGroq
        client = AsyncGroq(api_key=self._api_key)

        speech_buffer = bytearray()
        silence_chunks = 0
        speech_chunks = 0
        is_speaking = False

        while not stop_event.is_set():
            try:
                frame = await asyncio.wait_for(self._audio_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                # If speaking and a long timeout occurs, treat as end of utterance
                if is_speaking and silence_chunks >= self.SILENCE_CHUNKS_NEEDED:
                    await self._transcribe_and_fire(
                        client, bytes(speech_buffer), on_final
                    )
                    speech_buffer.clear()
                    speech_chunks = 0
                    silence_chunks = 0
                    is_speaking = False
                continue

            # Energy-based Voice Activity Detection
            samples = np.frombuffer(frame, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

            if rms >= self.SPEECH_RMS_THRESHOLD:
                # Active speech
                if not is_speaking:
                    is_speaking = True
                    on_interim("Speaking...")
                silence_chunks = 0
                speech_chunks += 1
                speech_buffer.extend(frame)
            else:
                # Silence
                if is_speaking:
                    silence_chunks += 1
                    speech_buffer.extend(frame)  # include trailing silence

                    if silence_chunks >= self.SILENCE_CHUNKS_NEEDED:
                        # End of utterance — transcribe if enough speech was captured
                        if speech_chunks >= self.MIN_SPEECH_CHUNKS:
                            await self._transcribe_and_fire(
                                client, bytes(speech_buffer), on_final
                            )
                        speech_buffer.clear()
                        speech_chunks = 0
                        silence_chunks = 0
                        is_speaking = False

    async def _transcribe_and_fire(self, client, pcm_bytes: bytes, on_final):
        """Send buffered audio to Groq Whisper API and fire on_final callback."""
        try:
            wav_bytes = _pcm_to_wav(pcm_bytes)
            result = await client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model="whisper-large-v3-turbo",
                language="en",
                response_format="text",
            )
            text = str(result).strip() if result else ""
            if text:
                logger.info(f"Whisper transcript: {text!r}")
                on_final(text)
        except Exception as e:
            logger.error(f"Groq Whisper error: {e}")


# ─── Main Orchestrator ────────────────────────────────────────────────────────

class VoicePipelineOrchestrator:
    """
    Master coordinator for the real-time voice interview.
    Manages the state machine and connects all cloud API clients.
    
    Must be run inside an asyncio event loop (use SessionWorker QThread).
    """

    def __init__(self, session: InterviewSession, callbacks: dict):
        """
        callbacks = {
            'on_interim_transcript': fn(str),
            'on_final_transcript': fn(str),
            'on_ai_text_chunk': fn(str),
            'on_audio_chunk': fn(bytes),
            'on_phase_change': fn(str),
            'on_session_complete': fn(dict),
            'on_error': fn(str),
        }
        """
        self.session = session
        self.cbs = callbacks

        # Use Groq Whisper for STT (higher accuracy, faster than Deepgram nova)
        self.stt = GroqWhisperSTTClient(os.environ.get("GROQ_API_KEY", ""))
        self.llm = GroqLLMClient()
        self.tts = ElevenLabsTTSClient(
            os.environ.get("ELEVENLABS_API_KEY", ""),
            os.environ.get("DEEPGRAM_API_KEY", ""),
        )

        self._stop_event = asyncio.Event()
        self._audio_input_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._tts_text_queue: asyncio.Queue = asyncio.Queue()
        self._processing_lock = asyncio.Lock()
        self._is_ai_speaking = False
        self._full_ai_response = ""
        self._bytes_sent_to_audio = 0
        self._audio_start_time = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self):
        """Connect to all APIs and begin the interview."""
        # No WebSocket to connect for Groq Whisper — it's REST-based
        await asyncio.gather(
            self.stt.run(
                on_interim=self._on_interim_transcript,
                on_final=self._on_final_transcript,
                stop_event=self._stop_event,
            ),
            self._tts_worker_loop(),
            self._kick_off_intro(),
        )

    def push_audio_frame(self, pcm_bytes: bytes, loop: asyncio.AbstractEventLoop = None):
        """Called by push_audio from the PyAudio OS thread.
        Routes audio to the Groq Whisper VAD queue (thread-safe via call_soon_threadsafe)."""
        if not self._is_ai_speaking and loop:
            self.stt.push_frame_threadsafe(pcm_bytes, loop)

    def stop(self):
        self._stop_event.set()

    # ── Internal Loops ────────────────────────────────────────────────────────
    
    async def _tts_worker_loop(self):
        """Asynchronously fetch TTS audio sequentially so LLM is not blocked."""
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self._tts_text_queue.get(), timeout=0.1)
                tts_text, pending_ui_text = item
                await self._stream_tts(tts_text, pending_ui_text)
                self._tts_text_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")
                self._tts_text_queue.task_done()

    async def _kick_off_intro(self):
        """Fire the first question to start the interview."""
        await asyncio.sleep(0.5)  # Brief startup pause
        await self._get_and_speak_ai_response()
    def _on_interim_transcript(self, text: str):
        """Callback for live partial transcripts — update UI immediately."""
        self.cbs.get("on_interim_transcript", lambda x: None)(text)

    def _on_final_transcript(self, text: str):
        """Callback for completed utterances — trigger LLM response."""
        if self._is_ai_speaking or not text.strip():
            return

        self.session.add_candidate_message(text)
        self.cbs.get("on_final_transcript", lambda x: None)(text)

        # Schedule LLM response (non-blocking)
        asyncio.create_task(self._handle_candidate_response(text))

    async def _handle_candidate_response(self, candidate_text: str):
        async with self._processing_lock:
            # Check if interview should end
            if self.session.current_phase == InterviewPhase.WRAP_UP:
                if self.session.phase_question_count >= PHASE_QUESTION_BUDGETS[InterviewPhase.WRAP_UP]:
                    await self._finalize_session()
                    return

            # Advance phase if budget exhausted
            if self.session.should_advance_phase():
                self.session.advance_phase()
                self.cbs.get("on_phase_change", lambda x: None)(
                    self.session.current_phase.name
                )

            await self._get_and_speak_ai_response()
            self.session.phase_question_count += 1
            self.session.total_questions += 1

    async def _get_and_speak_ai_response(self):
        """Stream LLM response and pipe it to TTS in real-time."""
        phase_directive = build_phase_directive(
            self.session.current_phase,
            self.session.weak_skills
        )

        self._is_ai_speaking = True
        self._full_ai_response = ""
        self._bytes_sent_to_audio = 0
        self._audio_start_time = time.time()
        
        tts_sentence_buffer = ""
        # Buffer of text not yet sent to the UI (will emit when audio arrives)
        pending_ui_text = ""

        # Stream LLM tokens
        async for token in self.llm.stream_response(
            self.session.system_prompt,
            self.session.conversation_history,
            phase_directive,
        ):
            self._full_ai_response += token
            tts_sentence_buffer += token
            pending_ui_text += token
            # Don't emit text immediately — we'll flush it when audio arrives

            # Send complete sentences to TTS for minimal latency and natural pacing
            if any(tts_sentence_buffer.endswith(p) for p in [".", "!", "?", ":", "\n"]):
                sentence = tts_sentence_buffer.strip()
                if sentence:
                    # Pass the pending UI text along so it emits right when audio starts
                    self._tts_text_queue.put_nowait((sentence + " ", pending_ui_text))
                    pending_ui_text = ""
                tts_sentence_buffer = ""

        # Flush remaining text
        if tts_sentence_buffer.strip():
            self._tts_text_queue.put_nowait((tts_sentence_buffer.strip() + " ", pending_ui_text))
            pending_ui_text = ""

        # Wait until all queued sentences are fully fetched from API (with timeout)
        try:
            await asyncio.wait_for(self._tts_text_queue.join(), timeout=30.0)
        except asyncio.TimeoutError:
            # Drain any unfinished TTS tasks so join() doesn't block later
            while not self._tts_text_queue.empty():
                try:
                    self._tts_text_queue.get_nowait()
                    self._tts_text_queue.task_done()
                except asyncio.QueueEmpty:
                    break
        
        # Audio bytes are queued in audio_manager continuously and play at 32000 bytes/sec
        if self._bytes_sent_to_audio > 0:
            elapsed_time = time.time() - self._audio_start_time
            sleep_time = (self._bytes_sent_to_audio / 32000.0) - elapsed_time
            # Add 0.6s physical acoustic tail clearing delay
            if sleep_time > 0:
                await asyncio.sleep(sleep_time + 0.6)

        self.session.add_ai_message(self._full_ai_response)
        self._is_ai_speaking = False

    async def _stream_tts(self, text: str, pending_ui_text: str = ""):
        """Stream TTS audio chunks to the UI callback.
        Emits pending UI text only when the first audio chunk arrives (sync voice+text).
        """
        try:
            first_chunk = True
            async for chunk in self.tts.synthesize_stream(text):
                if first_chunk:
                    # Only now reveal the text in the UI (voice and text are in sync)
                    if pending_ui_text:
                        self.cbs.get("on_ai_text_chunk", lambda x: None)(pending_ui_text)
                    first_chunk = False
                    self._audio_start_time = time.time()
                self._bytes_sent_to_audio += len(chunk)
                self.cbs.get("on_audio_chunk", lambda x: None)(chunk)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            self.cbs.get("on_error", lambda x: None)(f"TTS Error: {str(e)}")

    async def _finalize_session(self):
        """Build the final session report and signal completion."""
        self._stop_event.set()
        self._is_ai_speaking = False

        # Drain TTS queue so nothing blocks
        while not self._tts_text_queue.empty():
            try:
                self._tts_text_queue.get_nowait()
                self._tts_text_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Close the persistent HTTP client on the TTS side
        try:
            await self.tts.close()
        except Exception:
            pass

        report_data = {
            "candidate_name": self.session.candidate_name,
            "job_role": self.session.job_role,
            "total_questions": self.session.total_questions,
            "duration_seconds": int(time.time() - self.session.start_time),
            "conversation_history": self.session.conversation_history,
            "avg_response_time": (
                sum(self.session.response_times) / len(self.session.response_times)
                if self.session.response_times else 0
            ),
            "phase_progression": [p.name for p in PHASE_SEQUENCE],
        }
        self.cbs.get("on_session_complete", lambda x: None)(report_data)


# ─── Qt Worker Thread ─────────────────────────────────────────────────────────

try:
    from PyQt6.QtCore import QThread, pyqtSignal, QObject

    class SessionWorker(QObject):
        """
        Runs the asyncio event loop in a QThread.
        Bridges async callbacks to Qt signals for thread-safe GUI updates.
        """

        # Qt Signals for thread-safe GUI communication
        interim_transcript = pyqtSignal(str)
        final_transcript = pyqtSignal(str)
        ai_text_chunk = pyqtSignal(str)
        audio_chunk_ready = pyqtSignal(bytes)
        phase_changed = pyqtSignal(str)
        session_complete = pyqtSignal(dict)
        error_occurred = pyqtSignal(str)

        def __init__(self, session: InterviewSession):
            super().__init__()
            self.session = session
            self._orchestrator: Optional[VoicePipelineOrchestrator] = None
            self._loop: Optional[asyncio.AbstractEventLoop] = None

        def run(self):
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            callbacks = {
                "on_interim_transcript": lambda t: self.interim_transcript.emit(t),
                "on_final_transcript": lambda t: self.final_transcript.emit(t),
                "on_ai_text_chunk": lambda t: self.ai_text_chunk.emit(t),
                "on_audio_chunk": lambda b: self.audio_chunk_ready.emit(b),
                "on_phase_change": lambda p: self.phase_changed.emit(p),
                "on_session_complete": lambda r: self.session_complete.emit(r),
                "on_error": lambda e: self.error_occurred.emit(e),
            }

            self._orchestrator = VoicePipelineOrchestrator(self.session, callbacks)

            try:
                self._loop.run_until_complete(self._orchestrator.start())
            except Exception as e:
                self.error_occurred.emit(str(e))
            finally:
                self._loop.close()

        def push_audio(self, pcm_bytes: bytes):
            if self._orchestrator and self._loop and self._loop.is_running():
                # CRITICAL: call_soon_threadsafe properly bridges the PyAudio OS thread
                # into the asyncio event loop. Without this, asyncio.Queue.put_nowait()
                # is called across thread boundaries causing lag/drops.
                self._orchestrator.push_audio_frame(pcm_bytes, self._loop)

        def stop(self):
            if self._orchestrator:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._orchestrator.stop)
                    
        def end_session(self):
            """Trigger early finalization — guarantees session_complete fires.
            Uses non-blocking polling to avoid freezing the Qt main thread."""
            if not self._orchestrator:
                return

            if self._loop and self._loop.is_running():
                try:
                    self._end_future = asyncio.run_coroutine_threadsafe(
                        self._orchestrator._finalize_session(),
                        self._loop
                    )
                    # Poll for completion without blocking the Qt event loop
                    from PyQt6.QtCore import QTimer
                    self._end_poll_count = 0
                    self._end_timer = QTimer()
                    def _check_done():
                        self._end_poll_count += 1
                        if self._end_future.done():
                            self._end_timer.stop()
                            # session_complete was already emitted by _finalize_session callback
                            return
                        if self._end_poll_count > 50:  # 50 * 200ms = 10s timeout
                            self._end_timer.stop()
                            logger.warning("Async finalization timed out, using fallback")
                            self._emit_fallback_report()
                    self._end_timer.timeout.connect(_check_done)
                    self._end_timer.start(200)
                    return
                except Exception as e:
                    logger.warning(f"Async finalization failed ({e}), using fallback")

            # Fallback: build report directly from session state
            self._emit_fallback_report()

        def _emit_fallback_report(self):
            """Build and emit a report directly when the async path fails."""
            self._orchestrator._stop_event.set()
            session = self._orchestrator.session
            report_data = {
                "candidate_name": session.candidate_name,
                "job_role": session.job_role,
                "total_questions": session.total_questions,
                "duration_seconds": int(time.time() - session.start_time),
                "conversation_history": session.conversation_history,
                "avg_response_time": (
                    sum(session.response_times) / len(session.response_times)
                    if session.response_times else 0
                ),
                "phase_progression": [p.name for p in PHASE_SEQUENCE],
            }
            self.session_complete.emit(report_data)

except ImportError:
    logger.warning("PyQt6 not available — SessionWorker not defined")
