"""
Local STT engine using Faster-Whisper + Silero VAD.

Replaces the Google Web Speech API with local inference:
  - Faster-Whisper for transcription (base model, ~100ms/phrase on CPU)
  - Silero VAD for voice activity detection (replaces webrtcvad energy threshold)
  - Reduced pause thresholds for snappier turn-taking now that network latency is gone

Architecture remains producer/consumer:
  Listener Thread → audio_queue → Recognition Worker Thread
  (captures audio)                  (runs Whisper locally)
                                            │
                                    on_text_callback(text)
"""
from __future__ import annotations

import io
import logging
import os
import queue
import struct
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np
import torch

from src.speech.base import BaseSpeechEngine

logger = logging.getLogger("WhisperSpeechEngine")
logger.setLevel(logging.INFO)


class WhisperSpeechEngine(BaseSpeechEngine):
    """
    Local STT engine using Faster-Whisper and Silero VAD.
    
    Key changes from speech_engine.py (Google API version):
      - Faster-Whisper inference replaces recognize_google()
      - Silero VAD replaces webrtcvad energy-threshold gating
      - Reduced pause_threshold (0.9s → 0.6s) for snappier turn-taking
      - Reduced non_speaking_duration (0.5s → 0.3s) for tighter phrase boundaries
    """

    _QUEUE_MAXSIZE = 5

    def __init__(
        self,
        on_text_callback: Callable[[str], None],
        on_state_callback: Callable[[str], None] = None,
        model_size: str = "base",         # base = 72M params, ~100ms/phrase CPU
        device: str = "cpu",
        compute_type: str = "int8",       # int8 = quantized for CPU efficiency
    ):
        super().__init__(on_text_callback, on_state_callback)

        # ── Faster-Whisper initialization ─────────────────────────────────────
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Faster-Whisper model '{model_size}' on {device}...")
            t0 = time.time()
            self.whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                num_workers=1,
            )
            logger.info(f"Whisper model loaded in {time.time() - t0:.2f}s")
        except ImportError as e:
            logger.error(f"faster-whisper package not installed: {e}")
            logger.error("Install with: pip install faster-whisper")
            raise RuntimeError("faster-whisper not available") from e
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            logger.error("This may be due to:")
            logger.error("  - No internet connection (first-time model download)")
            logger.error("  - Insufficient disk space for model cache")
            logger.error("  - Corrupted model cache (try deleting ~/.cache/huggingface)")
            logger.error(f"To use Google API instead, set STT_BACKEND='google' in src/config/settings.py")
            raise RuntimeError(f"Whisper model initialization failed: {e}") from e

        # ── Silero VAD initialization ─────────────────────────────────────────
        try:
            import socket
            # Set timeout for torch.hub network operations
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(60)  # 60s timeout for model download
            
            logger.info("Loading Silero VAD model...")
            # Load Silero VAD v5 from torch.hub
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,  # Use PyTorch version for flexibility
                trust_repo=True,
            )
            (self.get_speech_timestamps, _, self.read_audio, _, _) = self.vad_utils
            logger.info("Silero VAD loaded successfully.")
            
            # Restore original timeout
            socket.setdefaulttimeout(original_timeout)
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            logger.error("This may be due to network timeout or torch.hub access issues.")
            logger.error("VAD is required for Whisper STT. Consider using STT_BACKEND='google' instead.")
            raise RuntimeError(f"Silero VAD initialization failed: {e}") from e

        # ── Microphone setup (using pyaudio directly for audio capture) ───────
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
            self.mic_available = True
            self.last_error: Optional[str] = None
            logger.info("Microphone initialized.")
        except Exception as e:
            logger.warning(f"Microphone init failed: {e}")
            self.mic_available = False
            self.last_error = str(e)
            self.pyaudio = None

        # ── Audio parameters ──────────────────────────────────────────────────
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1         # Mono
        self.chunk_size = 512     # PyAudio chunk size
        self.format = None        # set in _listener_loop
        if self.pyaudio:
            import pyaudio
            self.format = pyaudio.paInt16  # 16-bit PCM

        # ── Recognition parameters (tuned for local inference) ────────────────
        self.pause_threshold = 0.6        # Was 0.9s with Google API
                                          # Now 0.6s — network latency is gone
        self.non_speaking_duration = 0.3  # Was 0.5s; tighter phrase boundaries
        self.phrase_time_limit = 8.0      # Max phrase duration (unchanged)
        self.timeout = 3.0                # Wait up to 3s for speech to begin

        # ── State ─────────────────────────────────────────────────────────────
        self.is_running = False
        self.is_paused = False
        self.is_manual_listening = False

        # Duplicate guard
        self.last_recognized_text = ""
        self.last_recognized_time = 0.0
        self.duplicate_cooldown = 3.0  # Prevent re-fire within 3s

        # ── Queue & synchronization ───────────────────────────────────────────
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._active_event = threading.Event()
        self._shutdown_event = threading.Event()

        # UI compatibility
        self.stop_listening_func: Optional[bool] = None

        # ── Background threads ────────────────────────────────────────────────
        if self.mic_available:
            self._listener_thread = threading.Thread(
                target=self._listener_loop,
                name="Whisper-Listener",
                daemon=True,
            )
            self._listener_thread.start()

            self._worker_thread = threading.Thread(
                target=self._recognition_worker,
                name="Whisper-Worker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("Listener and worker threads started.")

    # ──────────────────────────────────────────────────────────────────────────
    # Listener thread (PRODUCER)
    # ──────────────────────────────────────────────────────────────────────────
    def _listener_loop(self):
        """
        Captures audio from microphone using PyAudio.
        Accumulates audio chunks, uses Silero VAD to detect speech,
        and enqueues complete phrases when silence is detected after speech.
        """
        import pyaudio

        try:
            stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            logger.info("Whisper-Listener: Audio stream opened.")

            phrase_buffer = []  # Accumulate chunks for current phrase
            silence_chunks = 0
            speech_started = False
            chunks_per_pause_check = int(self.sample_rate * 0.1 / self.chunk_size)  # Check every 100ms

            while not self._shutdown_event.is_set():
                if not self._active_event.wait(timeout=0.1):
                    # Pipeline stopped — clear buffers
                    phrase_buffer = []
                    silence_chunks = 0
                    speech_started = False
                    continue

                if self.is_paused:
                    time.sleep(0.05)
                    continue

                try:
                    # Read one chunk from mic
                    chunk_bytes = stream.read(self.chunk_size, exception_on_overflow=False)
                    phrase_buffer.append(chunk_bytes)

                    # Run VAD check every ~100ms to avoid excessive CPU
                    if len(phrase_buffer) % chunks_per_pause_check == 0:
                        # Concatenate recent chunks for VAD analysis
                        recent_audio = b"".join(phrase_buffer[-chunks_per_pause_check:])
                        has_speech = self._detect_speech(recent_audio)

                        if has_speech:
                            speech_started = True
                            silence_chunks = 0
                        elif speech_started:
                            # We had speech, now silence
                            silence_chunks += chunks_per_pause_check

                            # Check if silence duration exceeds pause_threshold
                            silence_duration = silence_chunks * self.chunk_size / self.sample_rate
                            if silence_duration >= self.pause_threshold:
                                # End of phrase detected
                                phrase_bytes = b"".join(phrase_buffer)
                                phrase_duration = len(phrase_buffer) * self.chunk_size / self.sample_rate

                                # Only enqueue if phrase is substantial (> 0.3s)
                                if phrase_duration >= self.non_speaking_duration:
                                    try:
                                        self.audio_queue.put_nowait(phrase_bytes)
                                        logger.debug(
                                            f"Whisper-Listener: Phrase enqueued "
                                            f"({phrase_duration:.2f}s, {len(phrase_bytes)} bytes)"
                                        )
                                    except queue.Full:
                                        try:
                                            self.audio_queue.get_nowait()
                                        except queue.Empty:
                                            pass
                                        self.audio_queue.put_nowait(phrase_bytes)
                                        logger.warning("Whisper-Listener: Queue full — oldest dropped.")

                                # Reset for next phrase
                                phrase_buffer = []
                                silence_chunks = 0
                                speech_started = False

                        # Prevent unbounded buffer growth (max 10s of audio)
                        max_chunks = int(self.phrase_time_limit * self.sample_rate / self.chunk_size)
                        if len(phrase_buffer) > max_chunks:
                            # Force end of phrase if it's been accumulating too long
                            phrase_bytes = b"".join(phrase_buffer)
                            try:
                                self.audio_queue.put_nowait(phrase_bytes)
                                logger.warning(
                                    f"Whisper-Listener: Forced phrase end (exceeded {self.phrase_time_limit}s)"
                                )
                            except queue.Full:
                                pass
                            phrase_buffer = []
                            silence_chunks = 0
                            speech_started = False

                except Exception as e:
                    if not self._shutdown_event.is_set():
                        logger.exception(f"Whisper-Listener: Error reading audio: {e}")
                    time.sleep(0.5)

            stream.stop_stream()
            stream.close()
            logger.info("Whisper-Listener: Audio stream closed.")

        except Exception as e:
            logger.exception(f"Whisper-Listener: Listener loop crashed: {e}")

    def _detect_speech(self, audio_bytes: bytes) -> bool:
        """
        Use Silero VAD to detect if audio contains speech.
        Returns True if speech probability > 0.5.
        """
        try:
            # Convert bytes to torch tensor (16kHz mono int16)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float)

            # Silero VAD expects 16kHz audio
            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

            return speech_prob > 0.5  # Threshold for speech detection

        except Exception as e:
            logger.debug(f"VAD check failed: {e} — assuming speech present.")
            return True  # On error, pass through

    # ──────────────────────────────────────────────────────────────────────────
    # Recognition worker (CONSUMER)
    # ──────────────────────────────────────────────────────────────────────────
    def _recognition_worker(self):
        """
        Pulls audio from queue and runs Faster-Whisper inference.
        Fires on_text_callback on successful recognition.
        """
        logger.info("Whisper-Worker: Recognition worker started.")
        while not self._shutdown_event.is_set():
            try:
                try:
                    audio_bytes = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if self.is_paused or not self.is_running:
                    logger.debug("Whisper-Worker: Audio discarded (paused/stopped).")
                    continue

                self.update_state("Processing")
                t0 = time.time()

                try:
                    # Convert bytes to numpy array for Whisper
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0

                    # Run Faster-Whisper transcription
                    segments, info = self.whisper_model.transcribe(
                        audio_float,
                        language="en",           # Force English for ISL + English mix
                        beam_size=1,             # Greedy decoding for speed
                        vad_filter=False,        # We already did VAD upstream
                        without_timestamps=True,
                    )

                    # Collect all segments into one text
                    text_parts = [seg.text.strip() for seg in segments]
                    text = " ".join(text_parts).strip()

                    t1 = time.time()
                    if text:
                        logger.info(f"Whisper-Worker: transcribed '{text}' in {t1 - t0:.3f}s")
                    else:
                        logger.debug(f"Whisper-Worker: No speech detected (empty transcript)")
                        if self.is_running and not self.is_paused:
                            self.update_state("Listening")
                        continue

                except Exception as e:
                    logger.exception(f"Whisper-Worker: Transcription error: {e}")
                    self.last_error = str(e)
                    if self.is_running and not self.is_paused:
                        self.update_state("Listening")
                    continue

                # ── Duplicate guard ───────────────────────────────────────────
                if not text:
                    if self.is_running and not self.is_paused:
                        self.update_state("Listening")
                    continue

                now = time.time()
                if (
                    text == self.last_recognized_text
                    and (now - self.last_recognized_time) < self.duplicate_cooldown
                ):
                    logger.info(f"Whisper-Worker: Duplicate ignored: '{text}'")
                    if self.is_running and not self.is_paused:
                        self.update_state("Listening")
                    continue

                self.last_recognized_text = text
                self.last_recognized_time = now

                # ── Fire callback ─────────────────────────────────────────────
                logger.info(f"Whisper-Worker: Firing callback with: '{text}'")
                try:
                    self.on_text_callback(text)
                    logger.info("Whisper-Worker: Callback executed successfully.")
                except Exception:
                    logger.exception("Whisper-Worker: on_text_callback raised an exception.")

                if self.is_running and not self.is_paused:
                    self.update_state("Listening")

            except Exception as e:
                logger.exception(f"Whisper-Worker: Unhandled error: {e}")
                time.sleep(0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def start(self):
        """Activate the STT pipeline. Safe to call multiple times."""
        if not self.mic_available:
            logger.error("WhisperSpeechEngine.start: Microphone unavailable.")
            self.update_state("Error")
            return

        if self.is_running:
            return

        self.is_running = True
        self.is_paused = False
        self._active_event.set()
        self.stop_listening_func = True
        self.update_state("Listening")
        logger.info("WhisperSpeechEngine: Pipeline activated.")

    def stop(self):
        """Deactivate the STT pipeline. Threads stay alive for reuse."""
        if not self.is_running:
            self.update_state("Mic Off")
            return

        logger.info("WhisperSpeechEngine: Deactivating pipeline...")
        self.is_running = False
        self._active_event.clear()
        self.stop_listening_func = None
        self.update_state("Mic Off")
        logger.info("WhisperSpeechEngine: Pipeline deactivated.")

    def pause(self):
        """Pause recognition (e.g., while TTS is speaking)."""
        self.is_paused = True
        self.update_state("Mic Off")
        logger.info("WhisperSpeechEngine: Paused.")

    def resume(self):
        """Resume recognition after pause."""
        self.is_paused = False
        self.last_recognized_text = ""
        if self.is_running:
            self.update_state("Listening")
        logger.info("WhisperSpeechEngine: Resumed.")

    def manual_listen(self, timeout: float = 6.0, phrase_time_limit: float = 10.0):
        """
        Single-shot blocking listen — kept for UI compatibility.
        In this queue architecture, it's a no-op since the listener loop
        handles the mic continuously.
        """
        logger.info("WhisperSpeechEngine.manual_listen: No-op (listener loop handles mic).")
