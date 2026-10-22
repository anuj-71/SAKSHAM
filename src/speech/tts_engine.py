import threading
import queue
import logging
import asyncio
import tempfile
import os
import time

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import pygame
except ImportError:
    pygame = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseTTSEngine:
    """Base class for Text-to-Speech engines."""
    def start(self):
        pass

    def stop(self):
        pass

    def speak(self, text: str):
        pass


class PyTTSx3Engine(BaseTTSEngine):
    """
    Text-to-Speech Engine using edge-tts for synthesis and pygame for playback.
    Preserves the original public API (start, stop, speak) and callbacks
    `on_speech_start` and `on_speech_end`. Uses a background thread and a queue
    to ensure sequential playback of utterances.
    """
    def __init__(self, on_speech_start=None, on_speech_end=None, voice: str = "en-US-JennyNeural"):
        self.speech_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.voice = voice
        self.cache_dir = "tts_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.phrase_cache = {}

    def start(self):
        if self.is_running:
            return
        # Initialize pygame mixer for playback
        if pygame:
            try:
                pygame.mixer.init()
            except Exception:
                logging.exception("Failed to initialize pygame mixer")

        self._preload_cache()

        self.is_running = True
        self.thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.thread.start()
        logging.info("PyTTSx3Engine started.")

    def stop(self):
        self.is_running = False
        # Push a dummy item to unblock the queue
        self.speech_queue.put(None)
        if self.thread:
            self.thread.join(timeout=2.0)
        try:
            if pygame:
                pygame.mixer.quit()
        except Exception:
            pass
        logging.info("PyTTSx3Engine stopped.")

    def speak(self, text: str):
        """Queues text to be spoken."""
        if text and text.strip():
            logging.info(f"TTS QUEUED: '{text}'")
            self.speech_queue.put(text.strip())

    def _preload_cache(self):
        phrases = [
            "Hello", "Goodbye", "I need help", "I need water", 
            "Thank you", "Please stop", "Please", "Yes", "No",
            "I want food", "I need a drink of water", "I want to go home",
            "I am hungry", "I feel sick", "I am sleepy", "Where"
        ]
        for phrase in phrases:
            safe_name = "".join(c for c in phrase.lower() if c.isalnum() or c == " ").replace(" ", "_")
            filename = os.path.join(self.cache_dir, f"{safe_name}.mp3")
            if not os.path.exists(filename):
                logging.info(f"Pre-synthesizing cache for: '{phrase}'")
                try:
                    self._synthesize_to_file(phrase, filename)
                except Exception as e:
                    logging.error(f"Failed to cache phrase '{phrase}': {e}")
                    continue
            key = phrase.lower().strip().rstrip(".,!?;:")
            self.phrase_cache[key] = filename
        logging.info(f"TTS Phrase Cache loaded with {len(self.phrase_cache)} phrases.")

    def _synthesize_to_file(self, text: str, filename: str):
        if edge_tts is None:
            raise RuntimeError("edge-tts is not installed")

        async def _save():
            communicate = edge_tts.Communicate(text, voice=self.voice)
            await communicate.save(filename)

        # Run the async save in a fresh event loop
        try:
            asyncio.run(_save())
        except Exception:
            logging.exception("TTS ERROR during synthesis")
            raise

    def _play_file(self, filename: str):
        if pygame is None:
            raise RuntimeError("pygame is not installed")
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            # Unload the music to release the file handle
            try:
                pygame.mixer.music.unload()
            except AttributeError:
                pass # Fallback for older pygame versions where unload doesn't exist
        except Exception:
            logging.exception("TTS ERROR during playback")
            raise

    def _tts_loop(self):
        while self.is_running:
            try:
                # Block until there is text to speak
                text = self.speech_queue.get(timeout=0.5)
                if text is None:
                    continue

                logging.info(f"TTS START: '{text}'")
                if self.on_speech_start:
                    try:
                        self.on_speech_start()
                    except Exception:
                        logging.exception("on_speech_start handler raised")

                lookup_key = text.lower().strip().rstrip(".,!?;:")
                cached_file = self.phrase_cache.get(lookup_key)
                if cached_file and os.path.exists(cached_file):
                    logging.info(f"TTS CACHE HIT: '{text}'")
                    try:
                        self._play_file(cached_file)
                    except Exception:
                        logging.error(f"TTS ERROR: playback failed for cached '{text}'")
                    logging.info(f"TTS FINISHED: '{text}'")
                else:
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
                    os.close(tmp_fd)
                    try:
                        # Synthesize
                        try:
                            self._synthesize_to_file(text, tmp_path)
                        except Exception:
                            logging.error(f"TTS ERROR: synthesis failed for '{text}'")
                            continue

                        # Play
                        try:
                            self._play_file(tmp_path)
                        except Exception:
                            logging.error(f"TTS ERROR: playback failed for '{text}'")
                            continue

                        logging.info(f"TTS FINISHED: '{text}'")
                    finally:
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            logging.exception("Failed to delete temporary TTS file")

                # Ensure the end callback runs
                if self.on_speech_end:
                    try:
                        self.on_speech_end()
                    except Exception:
                        logging.exception("on_speech_end handler raised")

            except queue.Empty:
                continue
            except Exception:
                logging.exception("Unexpected exception in TTS loop")
                # keep loop alive
                continue
