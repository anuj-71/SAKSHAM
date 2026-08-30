"""
Groq Hybrid Real-Time STT Engine for SAKSHAM
Primary: Groq Whisper Large V3 (Cloud LPU — ultra-fast, live streaming preview)
Fallback: Google Web Speech API (free, internet-based)

Provides real-time word-by-word live streaming preview directly into the chat input bar.
"""

import io
import math
import os
import queue
import struct
import threading
import time
import wave
import logging
from typing import Callable, Optional

import pyaudio
import speech_recognition as sr
from groq import Groq

logger = logging.getLogger("GroqSpeechEngine")
logger.setLevel(logging.INFO)


class GroqHybridSpeechEngine:
    """
    Hybrid STT with real-time rolling interim transcription preview.
    Uses PyAudio for continuous 16kHz audio capture, dynamic VAD gating,
    and asynchronous Groq LPU transcription for rolling word-by-word previews.
    """

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024       # 64ms per chunk at 16kHz
    CHANNELS = 1
    FORMAT = pyaudio.paInt16

    def __init__(self, api_key: str, callback: Optional[Callable[[str], None]] = None,
                 preview_callback: Optional[Callable[[str], None]] = None):
        self.api_key = api_key
        self.callback = callback                  # Called on final confirmed sentence
        self.preview_callback = preview_callback  # Called on live interim streaming text

        self.running = False
        self.backend = "groq"
        self.last_error = None
        self.state = "Idle"

        # Audio I/O
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self.mic_available = False

        try:
            default_input = self._pa.get_default_input_device_info()
            self.mic_available = True
            logger.info(f"Microphone detected: {default_input.get('name', 'Default')}")
        except Exception as e:
            logger.error(f"No microphone detected: {e}")
            self.last_error = str(e)

        # Groq client
        self._groq_client = None
        if self.api_key and self.api_key != "your_groq_api_key_here":
            try:
                self._groq_client = Groq(api_key=self.api_key)
                logger.info("Groq Cloud Whisper client ready.")
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
                self.backend = "google"

        # Google Fallback recognizer
        self._google_recognizer = sr.Recognizer()

        # Threading & Gating
        self._audio_thread = None
        self._interim_lock = threading.Lock()
        self._is_transcribing_interim = False
        self._last_interim_time = 0.0

        # Energy / Voice Activity Detection thresholds
        self.energy_threshold = 400.0
        self.silence_timeout = 1.1    # Seconds of silence to commit final sentence
        self.min_speech_duration = 0.4 # Minimum audio duration to send to Groq

    def update_state(self, state: str):
        self.state = state
        logger.debug(f"STT state: {state}")

    def start(self):
        """Starts microphone audio capture."""
        if not self.mic_available:
            self.update_state("No Microphone")
            return

        if self.running:
            return

        self.running = True
        try:
            self._stream = self._pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
        except Exception as e:
            logger.error(f"Failed to open PyAudio input stream: {e}")
            self.mic_available = False
            self.last_error = str(e)
            return

        self._audio_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._audio_thread.start()
        self.update_state(f"Listening [{self.backend.upper()}]")
        logger.info("Microphone stream active.")

    def pause(self):
        """Pauses STT listening temporarily (e.g. while TTS is speaking)."""
        self._is_paused = True
        logger.debug("STT paused.")

    def resume(self):
        """Resumes STT listening after pause."""
        self._is_paused = False
        logger.debug("STT resumed.")

    def stop(self):
        """Stops microphone audio capture."""
        self.running = False
        self._is_paused = False
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self.preview_callback:
            try:
                self.preview_callback("")
            except Exception:
                pass

        self.update_state("Stopped")
        logger.info("Microphone stream stopped.")

    def _calculate_rms(self, chunk: bytes) -> float:
        """Calculates Root-Mean-Square (RMS) audio energy."""
        count = len(chunk) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"{count}h", chunk)
        sum_squares = sum(s * s for s in shorts)
        return math.sqrt(sum_squares / count)

    def _audio_to_wav_bytes(self, frames: list[bytes]) -> bytes:
        """Encodes raw PCM audio frames into in-memory WAV format."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    def _transcribe_groq(self, wav_bytes: bytes) -> str:
        """Transcribes in-memory WAV audio via Groq Whisper Large V3."""
        if not self._groq_client:
            return ""
        try:
            transcription = self._groq_client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes, "audio/wav"),
                model="whisper-large-v3",
                language="en",
                response_format="text",
                temperature=0.0
            )
            return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "quota" in err or "limit" in err:
                logger.warning("Groq quota limit reached — switching to Google fallback.")
                self.backend = "google"
            else:
                logger.error(f"Groq transcription error: {e}")
            return ""

    def _transcribe_google(self, wav_bytes: bytes) -> str:
        """Fallback: Transcribes audio using Google Web Speech API."""
        try:
            audio_data = sr.AudioData(wav_bytes, self.SAMPLE_RATE, 2)
            return self._google_recognizer.recognize_google(audio_data, language="en-IN")
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            logger.error(f"Google STT fallback error: {e}")
            return ""

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        """Routes transcription to Groq or Google fallback."""
        if self.backend == "groq" and self._groq_client:
            text = self._transcribe_groq(wav_bytes)
            if text:
                return text
        return self._transcribe_google(wav_bytes)

    def _async_interim_transcribe(self, frames_snapshot: list[bytes]):
        """Runs rolling partial transcription asynchronously without blocking mic capture."""
        with self._interim_lock:
            if self._is_transcribing_interim:
                return
            self._is_transcribing_interim = True

        def _worker():
            try:
                wav_bytes = self._audio_to_wav_bytes(frames_snapshot)
                partial = self._transcribe_audio(wav_bytes)
                if partial and self.running:
                    # Clean up common hallucinations on short clips
                    cleaned = partial.strip()
                    if cleaned and not cleaned.lower().startswith(("you", "thank you", "subtitles")):
                        if self.preview_callback:
                            self.preview_callback(cleaned)
            except Exception as e:
                logger.debug(f"Interim error: {e}")
            finally:
                with self._interim_lock:
                    self._is_transcribing_interim = False

        threading.Thread(target=_worker, daemon=True).start()

    def _capture_loop(self):
        """Continuous audio capture loop with rolling VAD and streaming interim previews."""
        speech_frames = []
        preroll_buffer = []
        preroll_chunks_max = 5  # ~300ms pre-roll to capture word onsets
        is_speaking = False
        last_speech_time = 0.0

        # Ambient noise calibration for first ~0.4s
        calibration_energies = []
        for _ in range(6):
            if not self.running or self._stream is None:
                return
            try:
                chunk = self._stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                calibration_energies.append(self._calculate_rms(chunk))
            except Exception:
                break

        if calibration_energies:
            ambient = sum(calibration_energies) / len(calibration_energies)
            self.energy_threshold = max(300.0, ambient * 1.6)
            logger.info(f"Calibrated noise floor: {ambient:.1f}, Speech threshold: {self.energy_threshold:.1f}")

        while self.running and self._stream is not None:
            try:
                if getattr(self, "_is_paused", False):
                    time.sleep(0.05)
                    continue

                chunk = self._stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                energy = self._calculate_rms(chunk)
                now = time.time()

                if energy > self.energy_threshold:
                    # Voice detected
                    last_speech_time = now
                    if not is_speaking:
                        is_speaking = True
                        speech_frames = list(preroll_buffer)
                        if self.preview_callback:
                            self.preview_callback("...")

                    speech_frames.append(chunk)

                    # Trigger rolling interim preview every 0.6s of active speech
                    speech_duration = (len(speech_frames) * self.CHUNK_SIZE) / self.SAMPLE_RATE
                    if speech_duration >= self.min_speech_duration and (now - self._last_interim_time) >= 0.6:
                        self._last_interim_time = now
                        self._async_interim_transcribe(list(speech_frames))

                else:
                    # Silence / background
                    if is_speaking:
                        speech_frames.append(chunk)
                        silence_duration = now - last_speech_time

                        if silence_duration >= self.silence_timeout:
                            # Sentence finished — commit final transcript
                            is_speaking = False
                            final_frames = list(speech_frames)
                            speech_frames.clear()

                            speech_duration = (len(final_frames) * self.CHUNK_SIZE) / self.SAMPLE_RATE
                            if speech_duration >= self.min_speech_duration:
                                wav_bytes = self._audio_to_wav_bytes(final_frames)
                                final_text = self._transcribe_audio(wav_bytes)
                                if final_text and final_text.strip():
                                    cleaned = final_text.strip()
                                    logger.info(f"Final Transcript: {cleaned}")
                                    if self.callback:
                                        self.callback(cleaned)

                            if self.preview_callback:
                                self.preview_callback("")

                    else:
                        # Keep short rolling pre-roll buffer
                        preroll_buffer.append(chunk)
                        if len(preroll_buffer) > preroll_chunks_max:
                            preroll_buffer.pop(0)

            except Exception as e:
                logger.error(f"Error in audio capture loop: {e}")
                time.sleep(0.05)

    def get_status(self) -> dict:
        return {
            "state": self.state,
            "backend": self.backend,
            "mic_available": self.mic_available,
            "groq_ready": self._groq_client is not None,
            "last_error": self.last_error,
        }
