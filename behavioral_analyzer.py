"""
Module 3: Behavioral & Stuttering Analysis
============================================
Runs in a dedicated QThread alongside the voice pipeline.
Performs frame-level audio analysis using MFCC features
to detect disfluencies, measure confidence indicators,
and track communication quality metrics in real-time.

Classifier trained on SEP-28k derived features (bundled).
If model file not found, falls back to heuristic rules.
"""

from __future__ import annotations

import threading
import time
import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
FRAME_SIZE = 512           # ~32ms per frame at 16kHz
ANALYSIS_WINDOW = 30       # frames to accumulate before analysis (~960ms)
MFCC_N_MFCC = 13
SILENCE_THRESHOLD = 0.01   # RMS below this = silence
PAUSE_THRESHOLD_SEC = 1.5  # Pause duration considered noteworthy
STUTTER_WINDOW = 3.0       # seconds of context for stutter detection

FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "literally",
    "actually", "so", "kind of", "sort of", "right", "okay so"
}

MODEL_PATH = Path(__file__).parent.parent / "models" / "stutter_classifier.pkl"

# ─── Analysis Results ─────────────────────────────────────────────────────────

@dataclass
class FrameFeatures:
    """Extracted features from one analysis window."""
    mfcc_mean: np.ndarray        # shape (13,)
    mfcc_std: np.ndarray         # shape (13,)
    mfcc_delta_mean: np.ndarray  # shape (13,)
    zcr: float                   # Zero crossing rate
    rms_energy: float            # Root mean square energy
    spectral_centroid: float     # Frequency center of mass
    pitch_mean: float            # Estimated fundamental frequency
    pitch_std: float             # Pitch variability

@dataclass
class BehavioralSnapshot:
    """Real-time behavioral metrics snapshot."""
    timestamp: float = field(default_factory=time.time)
    confidence_index: float = 0.0          # 0–100 composite
    stutter_events_per_min: float = 0.0
    filler_rate: float = 0.0               # fillers per minute
    wpm: float = 0.0                       # Words per minute
    avg_pause_duration: float = 0.0        # seconds
    long_pause_count: int = 0
    voice_energy: float = 0.0             # Normalized energy level
    pitch_variability: float = 0.0        # Prosody indicator
    speech_ratio: float = 0.0             # % of time actively speaking
    disfluency_label: str = "fluent"       # Latest frame classification
    total_stutter_events: int = 0
    total_filler_words: int = 0

@dataclass
class SessionBehaviorReport:
    """Complete behavioral analysis for the full interview session."""
    snapshots: list[BehavioralSnapshot]
    final_confidence_score: float
    communication_score: float             # 0–100
    fluency_score: float                   # 0–100
    prosody_score: float                   # Intonation/pitch variation
    pace_score: float                      # Speaking rate consistency
    total_duration_sec: float
    total_stutter_events: int
    total_filler_words: int
    average_wpm: float
    peak_confidence_moment: float         # timestamp
    low_confidence_moments: list[float]   # timestamps
    recommendations: list[str]


# ─── MFCC Feature Extractor ───────────────────────────────────────────────────

class MFCCExtractor:
    """Extracts MFCC and prosodic features from raw PCM audio."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mfcc: int = MFCC_N_MFCC):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self._librosa = None

    @property
    def librosa(self):
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa

    def extract(self, pcm_bytes: bytes) -> Optional[FrameFeatures]:
        """Extract features from raw 16-bit PCM bytes."""
        try:
            # Convert bytes → float32 normalized array
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            audio /= 32768.0  # Normalize to [-1, 1]

            if len(audio) < 256:
                return None

            lr = self.librosa

            # Core MFCC features
            mfcc = lr.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_delta = lr.feature.delta(mfcc)

            # Energy
            rms = float(np.sqrt(np.mean(audio ** 2)))

            # Zero crossing rate (voice activity indicator)
            zcr = float(np.mean(lr.feature.zero_crossing_rate(audio)))

            # Spectral centroid (brightness of sound)
            if rms > SILENCE_THRESHOLD:
                spec_centroid = float(np.mean(
                    lr.feature.spectral_centroid(y=audio, sr=self.sample_rate)
                ))
            else:
                spec_centroid = 0.0

            # Pitch estimation using YIN algorithm
            pitch_mean, pitch_std = self._estimate_pitch(audio)

            return FrameFeatures(
                mfcc_mean=np.mean(mfcc, axis=1),
                mfcc_std=np.std(mfcc, axis=1),
                mfcc_delta_mean=np.mean(mfcc_delta, axis=1),
                zcr=zcr,
                rms_energy=rms,
                spectral_centroid=spec_centroid,
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
            )
        except Exception as e:
            logger.debug(f"MFCC extraction error: {e}")
            return None

    def _estimate_pitch(self, audio: np.ndarray) -> tuple[float, float]:
        """Lightweight autocorrelation-based pitch estimation."""
        try:
            lr = self.librosa
            pitches, magnitudes = lr.piptrack(
                y=audio, sr=self.sample_rate, fmin=75, fmax=400
            )
            # Pick the pitch with highest magnitude at each frame
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                return float(np.mean(pitch_values)), float(np.std(pitch_values))
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

    def features_to_vector(self, features: FrameFeatures) -> np.ndarray:
        """Flatten features to 1D numpy vector for classifier input."""
        return np.concatenate([
            features.mfcc_mean,       # 13
            features.mfcc_std,        # 13
            features.mfcc_delta_mean, # 13
            [features.zcr, features.rms_energy,
             features.spectral_centroid, features.pitch_mean, features.pitch_std]  # 5
        ])  # Total: 44 features


# ─── Disfluency Classifier ────────────────────────────────────────────────────

class DisfluencyClassifier:
    """
    Lightweight sklearn classifier for speech disfluency detection.
    
    Labels: fluent | stutter | filler | long_pause | repetition
    
    If the trained model file exists at models/stutter_classifier.pkl,
    loads it. Otherwise uses heuristic rule-based fallback.
    
    Training:
        See utils/train_classifier.py for the training script using
        SEP-28k derived MFCC features.
    """

    LABELS = ["fluent", "stutter", "filler", "long_pause", "repetition"]

    def __init__(self):
        self._model = None
        self._use_heuristic = True
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                self._use_heuristic = False
                logger.info("Loaded stutter classifier from disk")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}. Using heuristic mode.")
        else:
            logger.info("No trained classifier found. Using heuristic disfluency detection.")

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Returns (label, confidence) for the given feature vector."""
        if not self._use_heuristic and self._model is not None:
            return self._ml_predict(feature_vector)
        return self._heuristic_predict(feature_vector)

    def _ml_predict(self, features: np.ndarray) -> tuple[str, float]:
        try:
            proba = self._model.predict_proba(features.reshape(1, -1))[0]
            idx = int(np.argmax(proba))
            return self.LABELS[idx % len(self.LABELS)], float(proba[idx])
        except Exception:
            return "fluent", 1.0

    def _heuristic_predict(self, features: np.ndarray) -> tuple[str, float]:
        """
        Rule-based fallback using key acoustic indicators:
        - Very low energy + low ZCR = silence/pause
        - High ZCR + moderate energy = likely stutter/repetition
        - Erratic MFCC variance = disfluency
        """
        # features layout: [mfcc_mean(13), mfcc_std(13), mfcc_delta(13), zcr, rms, centroid, pitch_mean, pitch_std]
        rms = features[39]   # position 13+13+13=39
        zcr = features[38]
        mfcc_var = np.std(features[:13])
        pitch_std = features[43]

        if rms < SILENCE_THRESHOLD:
            return "long_pause", 0.85

        if zcr > 0.15 and mfcc_var > 20:
            return "stutter", 0.65

        if mfcc_var > 25 and pitch_std > 50:
            return "repetition", 0.60

        return "fluent", 0.90


# ─── Real-time Behavioral Analyzer ───────────────────────────────────────────

class BehavioralAnalyzer:
    """
    Runs in a background thread, consuming raw PCM frames from a queue.
    Produces BehavioralSnapshot every ~1 second for real-time GUI updates.
    Emits SessionBehaviorReport on session end.
    """

    def __init__(self, on_snapshot_callback, on_report_callback):
        """
        on_snapshot_callback: fn(BehavioralSnapshot) — called ~1/sec
        on_report_callback: fn(SessionBehaviorReport) — called at session end
        """
        self.on_snapshot = on_snapshot_callback
        self.on_report = on_report_callback

        self.extractor = MFCCExtractor()
        self.classifier = DisfluencyClassifier()

        self._audio_queue: "queue.Queue[bytes]" = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Metrics state
        self._frame_buffer: list[bytes] = []
        self._snapshots: list[BehavioralSnapshot] = []
        self._stutter_timestamps: deque = deque(maxlen=100)
        self._pause_timestamps: deque = deque(maxlen=50)
        self._energy_history: deque = deque(maxlen=300)
        self._pitch_history: deque = deque(maxlen=300)
        self._frame_labels: deque = deque(maxlen=100)
        self._word_count = 0
        self._start_time = 0.0
        self._speech_frames = 0
        self._total_frames = 0
        self._pause_start: Optional[float] = None
        self._pause_durations: list[float] = []

    def start(self):
        """Start background analysis thread."""
        import queue
        self._audio_queue = queue.Queue(maxsize=1000)
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._thread.start()
        logger.info("BehavioralAnalyzer started")

    def stop(self) -> SessionBehaviorReport:
        """Stop analysis and return final report."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        report = self._build_final_report()
        self.on_report(report)
        return report

    def push_frame(self, pcm_bytes: bytes):
        """Push raw PCM frame for analysis (called from PyAudio callback)."""
        if self._audio_queue is not None:
            try:
                self._audio_queue.put_nowait(pcm_bytes)
            except Exception:
                pass  # Drop if queue full

    def update_word_count(self, transcript: str):
        """Called when a new final transcript arrives."""
        words = transcript.strip().split()
        self._word_count += len(words)

    # ── Internal Loop ─────────────────────────────────────────────────────────

    def _analysis_loop(self):
        snapshot_interval = 1.0  # seconds
        last_snapshot_time = time.time()

        while not self._stop_event.is_set():
            try:
                frame = self._audio_queue.get(timeout=0.05)
            except Exception:
                # Check if we should emit a snapshot
                if time.time() - last_snapshot_time >= snapshot_interval:
                    self._emit_snapshot()
                    last_snapshot_time = time.time()
                continue

            self._total_frames += 1
            self._frame_buffer.append(frame)

            # Accumulate ANALYSIS_WINDOW frames before processing
            if len(self._frame_buffer) >= ANALYSIS_WINDOW:
                self._process_window(b"".join(self._frame_buffer))
                self._frame_buffer = self._frame_buffer[ANALYSIS_WINDOW // 2:]  # 50% overlap

            # Emit snapshot at regular intervals
            if time.time() - last_snapshot_time >= snapshot_interval:
                self._emit_snapshot()
                last_snapshot_time = time.time()

    def _process_window(self, audio_bytes: bytes):
        """Extract features and classify one analysis window."""
        features = self.extractor.extract(audio_bytes)
        if features is None:
            return

        # Update energy and pitch history
        self._energy_history.append(features.rms_energy)
        self._pitch_history.append(features.pitch_mean)

        # Track speech vs silence
        is_speech = features.rms_energy > SILENCE_THRESHOLD
        if is_speech:
            self._speech_frames += 1
            if self._pause_start is not None:
                # Pause ended — record duration
                pause_dur = time.time() - self._pause_start
                if pause_dur > 0.3:  # Min meaningful pause
                    self._pause_durations.append(pause_dur)
                self._pause_start = None
        else:
            if self._pause_start is None:
                self._pause_start = time.time()

        # Classify disfluency
        feature_vec = self.extractor.features_to_vector(features)
        label, confidence = self.classifier.predict(feature_vec)
        self._frame_labels.append(label)

        if label == "stutter":
            self._stutter_timestamps.append(time.time())

    def _emit_snapshot(self):
        """Build and emit current BehavioralSnapshot."""
        now = time.time()
        elapsed = max(now - self._start_time, 1.0)
        elapsed_min = elapsed / 60.0

        # WPM calculation
        wpm = self._word_count / elapsed_min if elapsed_min > 0 else 0

        # Stutter rate (events in last 60s)
        recent_stutters = sum(
            1 for t in self._stutter_timestamps
            if now - t <= 60.0
        )
        stutter_rate = recent_stutters

        # Pause analysis
        long_pauses = [p for p in self._pause_durations if p > PAUSE_THRESHOLD_SEC]
        avg_pause = np.mean(self._pause_durations) if self._pause_durations else 0.0

        # Voice metrics
        energy = np.mean(list(self._energy_history)) if self._energy_history else 0.0
        pitch_var = np.std(list(self._pitch_history)) if self._pitch_history else 0.0
        speech_ratio = (self._speech_frames / max(self._total_frames, 1)) * 100

        # Latest label
        latest_label = (
            list(self._frame_labels)[-1]
            if self._frame_labels else "fluent"
        )

        # Confidence index — composite heuristic
        confidence = self._compute_confidence_index(
            wpm=wpm,
            stutter_rate=stutter_rate,
            pitch_var=pitch_var,
            speech_ratio=speech_ratio,
            energy=energy,
            avg_pause=avg_pause,
        )

        snapshot = BehavioralSnapshot(
            timestamp=now,
            confidence_index=confidence,
            stutter_events_per_min=stutter_rate,
            wpm=round(wpm, 1),
            avg_pause_duration=round(avg_pause, 2),
            long_pause_count=len(long_pauses),
            voice_energy=round(float(energy * 1000), 2),  # Scale for display
            pitch_variability=round(float(pitch_var), 1),
            speech_ratio=round(speech_ratio, 1),
            disfluency_label=latest_label,
            total_stutter_events=len(self._stutter_timestamps),
        )

        self._snapshots.append(snapshot)
        self.on_snapshot(snapshot)

    def _compute_confidence_index(
        self, wpm, stutter_rate, pitch_var, speech_ratio, energy, avg_pause
    ) -> float:
        """
        Composite confidence score 0–100.
        Weighted combination of acoustic and temporal indicators.
        """
        score = 0.0

        # WPM component (optimal range: 120–160 wpm)
        if 100 <= wpm <= 170:
            wpm_score = 100.0
        elif wpm < 100:
            wpm_score = max(0, wpm / 100 * 100)
        else:
            wpm_score = max(0, 100 - (wpm - 170) * 2)
        score += wpm_score * 0.20

        # Fluency component (inverse of stutter rate)
        fluency_score = max(0, 100 - stutter_rate * 15)
        score += fluency_score * 0.30

        # Prosody component (some pitch variation = good, too much = nervous)
        if 10 <= pitch_var <= 60:
            prosody_score = 100.0
        elif pitch_var < 10:
            prosody_score = max(0, pitch_var / 10 * 100)  # Monotone = low confidence
        else:
            prosody_score = max(0, 100 - (pitch_var - 60) * 0.8)
        score += prosody_score * 0.20

        # Speech ratio component (20–80% speech is normal)
        if 20 <= speech_ratio <= 80:
            ratio_score = 100.0
        elif speech_ratio < 20:
            ratio_score = speech_ratio / 20 * 100
        else:
            ratio_score = max(0, 100 - (speech_ratio - 80) * 2)
        score += ratio_score * 0.15

        # Energy component (normalized 0–1 expected)
        energy_score = min(100, energy * 200)
        score += energy_score * 0.10

        # Pause component
        pause_score = max(0, 100 - avg_pause * 20)
        score += pause_score * 0.05

        return round(min(100, max(0, score)), 1)

    def _build_final_report(self) -> SessionBehaviorReport:
        """Aggregate all snapshots into final report."""
        if not self._snapshots:
            return SessionBehaviorReport(
                snapshots=[], final_confidence_score=0, communication_score=0,
                fluency_score=0, prosody_score=0, pace_score=0,
                total_duration_sec=0, total_stutter_events=0,
                total_filler_words=0, average_wpm=0,
                peak_confidence_moment=0, low_confidence_moments=[],
                recommendations=["Insufficient data for analysis"]
            )

        confidences = [s.confidence_index for s in self._snapshots]
        wpms = [s.wpm for s in self._snapshots if s.wpm > 0]
        pitch_vars = [s.pitch_variability for s in self._snapshots]

        final_confidence = round(float(np.mean(confidences)), 1)
        fluency_score = round(max(0, 100 - np.mean([s.stutter_events_per_min for s in self._snapshots]) * 10), 1)
        pace_score = round(min(100, float(np.mean(wpms)) / 1.6), 1) if wpms else 50.0
        prosody_score = round(min(100, float(np.mean(pitch_vars))) * 1.5, 1) if pitch_vars else 50.0

        communication_score = round(
            final_confidence * 0.35 + fluency_score * 0.35 + pace_score * 0.15 + prosody_score * 0.15,
            1
        )

        # Peak and low confidence moments
        peak_time = self._snapshots[int(np.argmax(confidences))].timestamp
        low_moments = [
            s.timestamp for s in self._snapshots
            if s.confidence_index < 40
        ][:5]

        recommendations = self._generate_recommendations(
            final_confidence, fluency_score, pace_score, prosody_score,
            float(np.mean(wpms)) if wpms else 0
        )

        return SessionBehaviorReport(
            snapshots=self._snapshots,
            final_confidence_score=final_confidence,
            communication_score=communication_score,
            fluency_score=fluency_score,
            prosody_score=prosody_score,
            pace_score=pace_score,
            total_duration_sec=round(time.time() - self._start_time, 1),
            total_stutter_events=len(self._stutter_timestamps),
            total_filler_words=self._snapshots[-1].total_filler_words if self._snapshots else 0,
            average_wpm=round(float(np.mean(wpms)), 1) if wpms else 0,
            peak_confidence_moment=peak_time,
            low_confidence_moments=low_moments,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, confidence, fluency, pace, prosody, wpm
    ) -> list[str]:
        recs = []

        if confidence < 60:
            recs.append("Practice mock interviews to build confidence under pressure.")
        if fluency < 70:
            recs.append("Consider speech therapy or fluency exercises to reduce stuttering/hesitation.")
        if wpm < 100:
            recs.append("Try to speak at a more natural pace — aim for 120–150 WPM.")
        elif wpm > 180:
            recs.append("Slow down slightly — rapid speech can reduce clarity and perceived confidence.")
        if prosody < 40:
            recs.append("Work on vocal variety — monotone delivery reduces engagement and perceived confidence.")
        if pace < 50:
            recs.append("Maintain a more consistent speaking rhythm to appear composed and organized.")

        if not recs:
            recs.append("Excellent communication skills — maintain your natural confident delivery.")

        return recs
