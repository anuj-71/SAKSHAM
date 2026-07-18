import queue
import threading
import speech_recognition as sr
import logging
import time
import struct
from typing import Callable, Optional

from src.speech.base import BaseSpeechEngine

logger = logging.getLogger("SpeechEngine")
logger.setLevel(logging.INFO)


class SpeechEngine(BaseSpeechEngine):
    """
    Production-quality STT engine using a Queue-based producer/consumer pipeline.

    Architecture
    ────────────
    Persistent Listener Thread  →  audio_queue  →  Recognition Worker Thread
          (captures audio)                                (calls Google API)
                                                              │
                                                     on_text_callback(text)

    Rules:
    - ONE listener thread, ONE recognition worker. Neither is ever recreated.
    - Mic context (`with self.mic as source`) is opened ONCE and held for the
      lifetime of the process.
    - pause() / resume() flip a flag; the listener discards audio while paused.
    - stop() / start() gate audio flow; they do NOT kill threads.
    """

    # Maximum number of unprocessed audio chunks queued before dropping oldest
    _QUEUE_MAXSIZE = 5

    def __init__(
        self,
        on_text_callback: Callable[[str], None],
        on_state_callback: Callable[[str], None] = None,
    ):
        super().__init__(on_text_callback, on_state_callback)

        # ── Recognizer configuration ──────────────────────────────────────────
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 400          # Higher = less sensitive to background noise
        self.recognizer.dynamic_energy_threshold = True  # Auto-adjusts to environment
        self.recognizer.pause_threshold = 0.9           # Seconds of silence before end-of-phrase (was 0.5)
        self.recognizer.non_speaking_duration = 0.5     # Min silence to mark phrase end (was 0.3)

        # ── Microphone ────────────────────────────────────────────────────────
        self.mic = None
        self.mic_available = False
        self.last_error: Optional[str] = None

        try:
            self.mic = sr.Microphone()
            with self.mic as source:
                logger.info("SpeechEngine: Calibrating for ambient noise (1.5s)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)  # Longer = more accurate baseline
                logger.info(
                    f"SpeechEngine: Calibration done. "
                    f"energy_threshold={self.recognizer.energy_threshold:.1f}"
                )
            self.mic_available = True
            logger.info("SpeechEngine: Microphone initialized successfully.")
        except Exception as e:
            logger.warning(f"SpeechEngine: Microphone init failed: {e}")
            self.last_error = str(e)

        # ── State ─────────────────────────────────────────────────────────────
        self.is_running = False
        self.is_paused = False
        self.is_manual_listening = False  # kept for interface compatibility

        # Duplicate-recognition guard
        self.last_recognized_text = ""
        self.last_recognized_time = 0.0
        self.duplicate_cooldown = 3.0  # Prevent re-firing same phrase within 3s (was 1.5)

        # ── VAD (Voice Activity Detection) ────────────────────────────────────
        # Filters out noise-only clips before they hit the Google API
        self._vad = None
        try:
            try:
                import webrtcvad_wheels as webrtcvad  # Windows pre-built wheels
            except ImportError:
                import webrtcvad  # Linux/Mac native build
            self._vad = webrtcvad.Vad(2)  # Aggressiveness 0–3 (2 = balanced)
            logger.info("SpeechEngine: VAD enabled (webrtcvad aggressiveness=2).")
        except ImportError:
            logger.warning("SpeechEngine: webrtcvad not installed — VAD disabled. "
                           "Install with: pip install webrtcvad-wheels")

        # ── Queue & synchronization ───────────────────────────────────────────
        self.audio_queue: queue.Queue[sr.AudioData] = queue.Queue(
            maxsize=self._QUEUE_MAXSIZE
        )
        # _active_event: SET   → pipeline is running (start() called)
        #                CLEAR → pipeline is paused/stopped
        self._active_event = threading.Event()
        self._shutdown_event = threading.Event()

        # stop_listening_func kept for UI compatibility (was used as a dummy flag)
        self.stop_listening_func: Optional[bool] = None

        # ── Background threads (started once, run forever) ────────────────────
        if self.mic_available:
            self._listener_thread = threading.Thread(
                target=self._listener_loop,
                name="STT-Listener",
                daemon=True,
            )
            self._listener_thread.start()

            self._worker_thread = threading.Thread(
                target=self._recognition_worker,
                name="STT-Worker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("SpeechEngine: Listener and worker threads started.")

    # ──────────────────────────────────────────────────────────────────────────
    # Listener thread  (PRODUCER)
    # ──────────────────────────────────────────────────────────────────────────
    def _listener_loop(self):
        """
        Holds the microphone context for the entire process lifetime.
        Captures audio chunks and enqueues them for recognition.
        Blocks (without spinning) when the engine is not active.
        """
        try:
            with self.mic as source:
                logger.info("STT-Listener: Microphone context opened.")
                while not self._shutdown_event.is_set():
                    # Block here while stopped/paused — no CPU spin
                    if not self._active_event.wait(timeout=0.1):
                        continue

                    if self.is_paused:
                        time.sleep(0.05)
                        continue

                    try:
                        t0 = time.time()
                        audio = self.recognizer.listen(
                            source,
                            timeout=3.0,         # Wait up to 3s for speech to begin
                            phrase_time_limit=8.0, # Capture up to 8s of speech
                        )
                        duration = time.time() - t0
                        logger.debug(f"STT-Listener: Audio captured ({duration:.1f}s)")
                    except sr.WaitTimeoutError:
                        # No speech within timeout — loop and wait again
                        continue
                    except Exception as e:
                        if not self._shutdown_event.is_set():
                            logger.exception(f"STT-Listener: Error: {e}")
                        time.sleep(0.5)
                        continue

                    # Discard if paused or stopped while we were recording
                    if self.is_paused or not self.is_running:
                        continue

                    # ── VAD gate: discard if no speech detected ─────────────────
                    if not self._contains_speech(audio):
                        logger.debug("STT-Listener: VAD rejected clip (no speech detected).")
                        continue

                    # Enqueue; drop oldest chunk if queue is full (overflow guard)
                    try:
                        self.audio_queue.put_nowait(audio)
                        logger.debug(
                            f"STT-Listener: Audio enqueued (queue size ~{self.audio_queue.qsize()})"
                        )
                    except queue.Full:
                        try:
                            self.audio_queue.get_nowait()  # drop oldest
                        except queue.Empty:
                            pass
                        self.audio_queue.put_nowait(audio)
                        logger.warning("STT-Listener: Queue full — oldest chunk dropped.")

        except Exception as e:
            logger.exception(f"STT-Listener: Listener loop crashed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Recognition worker  (CONSUMER)
    # ──────────────────────────────────────────────────────────────────────────
    def _recognition_worker(self):
        """
        Pulls AudioData from the queue and calls recognize_google().
        Fires on_text_callback on successful, non-duplicate recognition.
        """
        logger.info("STT-Worker: Recognition worker started.")
        while not self._shutdown_event.is_set():
            try:
                # Block until audio is available (1-second timeout to re-check shutdown)
                try:
                    audio = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if self.is_paused or not self.is_running:
                    logger.debug("STT-Worker: Audio discarded (paused/stopped).")
                    continue

                self.update_state("Processing")
                t0 = time.time()

                try:
                    text = self.recognizer.recognize_google(audio)
                    t1 = time.time()
                    logger.info(
                        f"STT-Worker: recognize_google → '{text}' in {t1 - t0:.3f}s"
                    )
                except sr.UnknownValueError:
                    logger.debug("STT-Worker: Unintelligible audio — skipped.")
                    if self.is_running and not self.is_paused:
                        self.update_state("Listening")
                    continue
                except sr.RequestError as e:
                    logger.exception(f"STT-Worker: API error: {e}")
                    self.last_error = str(e)
                    if self.is_running and not self.is_paused:
                        self.update_state("Listening")
                    continue
                except Exception as e:
                    logger.exception(f"STT-Worker: Unexpected error: {e}")
                    self.last_error = str(e)
                    if self.is_running and not self.is_paused:
                        self.update_state("Listening")
                    continue

                # ── Duplicate guard ───────────────────────────────────────────
                text = text.strip()
                if not text:
                    if self.is_running and not self.is_paused:
                        self.update_state("Listening")
                    continue

                now = time.time()
                if (
                    text == self.last_recognized_text
                    and (now - self.last_recognized_time) < self.duplicate_cooldown
                ):
                    logger.info(f"STT-Worker: Duplicate ignored: '{text}'")
                    if self.is_running and not self.is_paused:
                        self.update_state("Listening")
                    continue

                self.last_recognized_text = text
                self.last_recognized_time = now

                # ── Fire callback ─────────────────────────────────────────────
                logger.info(f"STT-Worker: Firing callback with: '{text}'")
                try:
                    self.on_text_callback(text)
                    logger.info("STT-Worker: Callback executed successfully.")
                except Exception:
                    logger.exception("STT-Worker: on_text_callback raised an exception.")

                if self.is_running and not self.is_paused:
                    self.update_state("Listening")

            except Exception as e:
                logger.exception(f"[Worker] Unhandled error in worker loop: {e}")
                time.sleep(0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # VAD Helper
    # ──────────────────────────────────────────────────────────────────────────
    def _contains_speech(self, audio: sr.AudioData) -> bool:
        """
        Returns True if the AudioData clip contains detectable human speech.
        Uses webrtcvad on 20ms PCM frames at 16kHz mono.
        Falls back to True (always pass) if VAD is unavailable.
        """
        if self._vad is None:
            return True  # No VAD available — let everything through

        try:
            # webrtcvad requires 16kHz mono 16-bit PCM
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            frame_ms = 20
            frame_bytes = int(16000 * frame_ms / 1000) * 2  # 640 bytes per 20ms frame

            speech_frames = 0
            total_frames = 0
            offset = 0
            while offset + frame_bytes <= len(raw):
                frame = raw[offset: offset + frame_bytes]
                try:
                    if self._vad.is_speech(frame, 16000):
                        speech_frames += 1
                except Exception:
                    pass
                total_frames += 1
                offset += frame_bytes

            if total_frames == 0:
                return False

            speech_ratio = speech_frames / total_frames
            return speech_ratio >= 0.15  # At least 15% of frames must contain speech
        except Exception as e:
            logger.debug(f"VAD check failed: {e} — passing clip through.")
            return True  # On error, don't block the clip

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def start(self):
        """Activate the STT pipeline. Safe to call multiple times."""
        if not self.mic_available:
            logger.error("SpeechEngine.start: Microphone unavailable.")
            self.update_state("Error")
            return

        if self.is_running:
            return  # already active

        self.is_running = True
        self.is_paused = False
        self._active_event.set()
        self.stop_listening_func = True  # UI compatibility flag
        self.update_state("Listening")
        logger.info("SpeechEngine: Pipeline activated.")

    def stop(self):
        """Deactivate the STT pipeline. Threads stay alive for reuse."""
        if not self.is_running:
            self.update_state("Mic Off")
            return

        logger.info("SpeechEngine: Deactivating pipeline...")
        self.is_running = False
        self._active_event.clear()
        self.stop_listening_func = None
        self.update_state("Mic Off")
        logger.info("SpeechEngine: Pipeline deactivated.")

    def pause(self):
        """Pause recognition (e.g., while TTS is speaking). Thread stays alive."""
        self.is_paused = True
        self.update_state("Mic Off")
        logger.info("SpeechEngine: Paused.")

    def resume(self):
        """Resume recognition after pause."""
        self.is_paused = False
        # Reset guard so the same phrase can be re-recognized intentionally
        self.last_recognized_text = ""
        if self.is_running:
            self.update_state("Listening")
        logger.info("SpeechEngine: Resumed.")

    def manual_listen(self, timeout: float = 6.0, phrase_time_limit: float = 10.0):
        """
        Single-shot blocking listen — kept for UI compatibility.
        Called from a worker thread (NOT the UI thread).
        NOTE: Because the persistent listener already holds the mic context,
        manual_listen is intentionally a no-op here. Audio will arrive
        automatically via the listener loop. Callers should just await
        the on_text_callback.
        """
        logger.info(
            "SpeechEngine.manual_listen: No-op in Queue architecture "
            "(listener loop handles mic continuously)."
        )
