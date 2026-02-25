"""
Audio Manager
==============
Handles PyAudio microphone capture and speaker playback.
Routes audio frames simultaneously to:
  - VoicePipeline (for STT via Deepgram WebSocket)
  - BehavioralAnalyzer (for local MFCC analysis)
  - Playback stream (for TTS audio chunks from ElevenLabs/Deepgram)
"""

from __future__ import annotations

import io
import logging
import threading
import queue
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHANNELS = 1
FORMAT_INT16 = 8   # pyaudio.paInt16


class AudioManager:
    """
    Manages all audio I/O for the interviewer session.
    Thread-safe: input callbacks run on PyAudio's callback thread.
    """

    def __init__(self):
        self._pa = None
        self._input_stream = None
        self._output_stream = None
        self._is_running = False
        self._playback_lock = threading.Lock()

        # Callbacks registered by other modules
        self._on_frame_callbacks: list[Callable[[bytes], None]] = []

        # Playback buffer
        self._playback_queue_obj = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_playback = threading.Event()

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self):
        """Initialize PyAudio and open streams."""
        import pyaudio
        self._pa = pyaudio.PyAudio()
        self._open_input_stream()
        self._open_output_stream()
        self._is_running = True
        
        self._stop_playback.clear()
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()
        
        logger.info("AudioManager initialized")

    def _open_input_stream(self):
        import pyaudio
        self._input_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._input_callback,
        )
        self._input_stream.start_stream()

    def _open_output_stream(self):
        import pyaudio
        self._output_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,            # Match incoming PCM output rate
            output=True,
            frames_per_buffer=1024,
        )

    def _input_callback(self, in_data, frame_count, time_info, status):
        """PyAudio input callback — runs on audio thread."""
        import pyaudio
        if self._is_running and in_data:
            for cb in self._on_frame_callbacks:
                try:
                    cb(in_data)
                except Exception as e:
                    logger.error(f"Audio frame callback error: {e}")
        return (None, pyaudio.paContinue)

    # ── Registration API ──────────────────────────────────────────────────────

    def register_frame_callback(self, callback: Callable[[bytes], None]):
        """Register a callback to receive raw PCM frames (16kHz, 16-bit, mono)."""
        self._on_frame_callbacks.append(callback)

    # ── Playback API ──────────────────────────────────────────────────────────

    def play_audio_chunk(self, pcm_bytes: bytes):
        """
        Accepts raw PCM chunks from TTS stream and queues them for playback.
        Assumes 16kHz 16-bit mono PCM.
        """
        if pcm_bytes:
            self._playback_queue_obj.put(pcm_bytes)

    def _playback_worker(self):
        """Background thread that consumes PCM chunks and freely writes to PyAudio.
        Pre-buffers only on first utterance to avoid audio glitches at cold start.
        Subsequent utterances are played immediately as chunks arrive.
        """
        buffer = bytearray()
        WRITE_SLICE = 512     # 512 bytes = ~16ms — small enough for smooth flow
        playing = False       # Once True, never wait for pre-buffer again

        while not self._stop_playback.is_set():
            try:
                pcm_bytes = self._playback_queue_obj.get(timeout=0.02)
                buffer.extend(pcm_bytes)
            except queue.Empty:
                pass

            # Only pre-buffer on very first startup (~32ms); never block again after
            if not playing:
                if len(buffer) < 1024:
                    continue
                playing = True

            # Write in fixed slices — thread never blocks more than ~16ms at once
            while len(buffer) >= WRITE_SLICE and self._output_stream:
                chunk = bytes(buffer[:WRITE_SLICE])
                del buffer[:WRITE_SLICE]
                try:
                    self._output_stream.write(chunk)
                except Exception as e:
                    logger.warning(f"PCM playback error: {e}")
                    break

    def flush_playback(self):
        """No-op for raw PCM streaming."""
        pass

    def _check_ffmpeg(self) -> bool:
        import subprocess
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def get_input_device_list(self) -> list[dict]:
        """Return list of available input devices."""
        if not self._pa:
            return []
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({"index": i, "name": info["name"]})
        return devices

    def shutdown(self):
        self._is_running = False
        self._stop_playback.set()
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("AudioManager shut down")
