import os
import sys
import warnings
import ctypes

# Enable high-DPI awareness on Windows to prevent OS bitmap blurring
try:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# ── Suppress noisy third-party startup banners ─────────────────────────────
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"   # Hides pygame banner & community message
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

import cv2
import time
import logging

import src.config.settings as config
from src.camera.camera_manager import CameraManager
from src.sign_language.engine import SignLanguageEngine
from src.config import settings
from src.speech.tts_engine import PyTTSx3Engine
from src.ui.renderer import UIRenderer
from src.session_manager import ConversationSession
from src.ui.live_transcript_widget import LiveTranscriptWidget

# ─── Logging ──────────────────────────────────────────────────────────────────
# Console: clean, short format  |  File: full timestamped log
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("  %(message)s"))

_file_handler = logging.FileHandler("saksham.log", mode="w", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.DEBUG, handlers=[_console_handler, _file_handler])

# Silence overly verbose third-party loggers in the console
for _noisy in ("absl", "onnxruntime", "mediapipe", "tensorflow", "STT-Listener", "STT-Worker"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Speech engine selection based on settings with graceful fallback
if settings.STT_BACKEND == "groq":
    try:
        from src.speech.speech_engine_groq import GroqHybridSpeechEngine as SpeechEngine
        logger.info("STT backend: Groq hybrid (Groq → Google fallback)")
    except Exception as e:
        logger.error(f"Groq engine import failed: {e}. Falling back to Google.")
        from src.speech.speech_engine import SpeechEngine
        settings.STT_BACKEND = "google"
elif settings.STT_BACKEND == "whisper":
    try:
        from src.speech.speech_engine_whisper import WhisperSpeechEngine as SpeechEngine
        logger.info(f"Using Whisper STT (model={settings.WHISPER_MODEL_SIZE}, device={settings.WHISPER_DEVICE})")
    except Exception as e:
        logger.error(f"Whisper engine initialization failed: {e}")
        logger.warning("Falling back to Google Web Speech API")
        from src.speech.speech_engine import SpeechEngine
        settings.STT_BACKEND = "google"  # Update runtime setting
else:
    from src.speech.speech_engine import SpeechEngine
    logger.info("Using Google Web Speech API STT")

# Global state for OpenCV callbacks
scroll_offset = 0
global_session = None
global_ui = None
global_live_widget = None
global_toggle_mic_fn = None
global_export_fn = None
global_toggle_fullscreen_fn = None
global_request_quit = False

def mouse_callback(event, x, y, flags, param):
    global scroll_offset, global_session, global_ui, global_live_widget
    global global_toggle_mic_fn, global_export_fn, global_toggle_fullscreen_fn, global_request_quit

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scroll_offset += 1  # Scroll Up
        else:
            scroll_offset -= 1  # Scroll Down
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Check Top Bar buttons
        if global_ui:
            top_action = global_ui.handle_click(x, y)
            if top_action == "mic":
                if global_toggle_mic_fn:
                    global_toggle_mic_fn()
                return
            elif top_action == "clear":
                if global_session:
                    global_session.clear()
                return
            elif top_action == "fullscreen" or top_action == "maximize":
                if global_toggle_fullscreen_fn:
                    global_toggle_fullscreen_fn()
                return
            elif top_action == "export":
                if global_export_fn:
                    global_export_fn()
                return
            elif top_action == "close":
                global_request_quit = True
                return

        # Check Bottom Bar widgets
        if global_live_widget and global_session:
            zone = global_live_widget.handle_click(x, y)
            if zone == 'mic':
                if global_toggle_mic_fn:
                    global_toggle_mic_fn()
                global_session.is_typing_focused = False
            elif zone == 'export':
                if global_export_fn:
                    global_export_fn()
                global_session.is_typing_focused = False
            elif zone == 'send':
                if global_session.typing_buffer.strip():
                    global_session.add_message("Hearing Person", global_session.typing_buffer.strip(), source="Typed")
                    global_session.typing_buffer = ""
                global_session.is_typing_focused = False
            elif zone == 'text':
                global_session.is_typing_focused = True
            else:
                global_session.is_typing_focused = False

def main():
    global scroll_offset
    
    logging.info("SAKSHAM — Starting up...")


    session = ConversationSession()
    global global_session
    global_session = session

    # ── Module initialisation ──────────────────────────────────────────────
    camera = CameraManager(
        camera_index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT
    )
    if not camera.start():
        logging.critical("Failed to start the camera. Exiting.")
        sys.exit(1)

    # Detect native monitor resolution for 1:1 pixel-perfect fullscreen rendering
    ui_w, ui_h = config.FRAME_WIDTH, config.FRAME_HEIGHT
    try:
        if sys.platform == "win32":
            scr_w = ctypes.windll.user32.GetSystemMetrics(0)
            scr_h = ctypes.windll.user32.GetSystemMetrics(1)
            if scr_w >= 1280 and scr_h >= 720:
                ui_w, ui_h = scr_w, scr_h
    except Exception:
        pass

    ui = UIRenderer(width=ui_w, height=ui_h)
    live_widget = LiveTranscriptWidget(bar_height=ui.bottom_bar_h)
    
    # ── Text-to-Speech Engine ──
    def on_tts_start():
        logging.info("TTS started speaking, pausing STT mic.")
        if session.mic_enabled:
            speech_engine.pause()
        
    def on_tts_end():
        logging.info("TTS finished speaking, resuming STT mic.")
        if session.mic_enabled:
            speech_engine.resume()
        
    tts_engine = PyTTSx3Engine(on_speech_start=on_tts_start, on_speech_end=on_tts_end)
    tts_engine.start()

    # ── Sign Language Engine ──
    # Maps model output labels → natural spoken sentences (fragments that join well)
    # These keys MUST match the exact class names the ONNX model outputs.
    SIGN_TO_PHRASE = {
        "hello":    "Hello!",
        "bye":      "Goodbye!",
        "yes":      "Yes.",
        "no":       "No.",
        "please":   "Please.",
        "thankyou": "Thank you!",
        "food":     "I want food.",
        "drink":    "I need a drink of water.",
        "home":     "I want to go home.",
        "hungry":   "I am hungry.",
        "sick":     "I feel sick.",
        "sleep":    "I am sleepy.",
        "where":    "Where?",
    }

    class PhraseBuilder:
        """
        Buffers recognised signs and speaks them as one natural sentence
        after PHRASE_PAUSE seconds of silence between signs.
        Prevents the robotic word-by-word TTS behaviour.
        """
        PHRASE_PAUSE = 2.0   # seconds of silence before flushing

        def __init__(self, sign_map: dict, speak_fn):
            self._sign_map = sign_map
            self._speak = speak_fn
            self._buffer: list[str] = []
            self._last_sign_time: float = 0.0
            self._lock = __import__("threading").Lock()
            self._flush_thread = __import__("threading").Thread(
                target=self._flush_loop, daemon=True
            )
            self._flush_thread.start()

        def add(self, raw_label: str):
            phrase = self._sign_map.get(raw_label, raw_label.replace("_", " ").capitalize())
            with self._lock:
                self._buffer.append(phrase)
                self._last_sign_time = time.time()

        def _flush_loop(self):
            """Background thread: flush buffer when PHRASE_PAUSE elapses."""
            while True:
                time.sleep(0.2)
                with self._lock:
                    if not self._buffer:
                        continue
                    elapsed = time.time() - self._last_sign_time
                    if elapsed >= self.PHRASE_PAUSE:
                        sentence = " ".join(self._buffer)
                        self._buffer.clear()
                # Speak outside the lock to avoid blocking add()
                if elapsed >= self.PHRASE_PAUSE and sentence:
                    self._speak(sentence)

    phrase_builder = PhraseBuilder(sign_map=SIGN_TO_PHRASE, speak_fn=tts_engine.speak)

    _last_sign_confirmed_time = 0.0
    SIGN_COOLDOWN_SECONDS = 3.0

    def on_sign_recognized(raw_sign_label: str):
        nonlocal _last_sign_confirmed_time
        if session.dataset_mode or not raw_sign_label or raw_sign_label == "null":
            return
        now = time.time()
        if (now - _last_sign_confirmed_time) < SIGN_COOLDOWN_SECONDS:
            return
        _last_sign_confirmed_time = now

        # 1. Show in conversation immediately so UI feels responsive
        session.add_message("Deaf User", raw_sign_label, source="Sign")
        # 2. Buffer into phrase builder — TTS fires after a natural pause
        phrase_builder.add(raw_sign_label)

    sl_engine = SignLanguageEngine(on_sign_recognized=on_sign_recognized)


    def refresh_dataset_insights():
        session.set_dataset_insights(
            sl_engine.dataset_manager.build_collection_report(session.dataset_labels)
        )

    refresh_dataset_insights()
    
    # ── Speech-to-Text Engine ──
    def on_speech_text(text: str):
        if text.strip():
            recognized = text.strip()
            session.typing_buffer = recognized
            session.add_message("Hearing Person", recognized, source="Speech")
            session.typing_buffer = ""

    def on_speech_state(state: str):
        session.mic_state = state
        if state == "Listening":
            if not session.draft_message:
                session.set_draft("...")
        elif state == "Processing":
            session.set_draft("Transcribing...")
        else:
            session.set_draft("")

    if settings.STT_BACKEND == "groq":
        speech_engine = SpeechEngine(
            api_key=settings.GROQ_API_KEY,
            callback=lambda text: (on_speech_text(text), live_widget.set_final(text)),
            preview_callback=lambda text: (session.set_draft(text), live_widget.set_preview(text))
        )
        original_update_state = speech_engine.update_state
        def _patched_update_state(state):
            original_update_state(state)
            on_speech_state(state)
        speech_engine.update_state = _patched_update_state
    elif settings.STT_BACKEND == "whisper":
        speech_engine = SpeechEngine(
            on_text_callback=on_speech_text,
            on_state_callback=on_speech_state,
            model_size=settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
    else:
        speech_engine = SpeechEngine(
            on_text_callback=on_speech_text,
            on_state_callback=on_speech_state,
        )

    # Microphone starts in OFF state - only starts when user clicks mic icon or presses [M]
    session.mic_enabled = False
    session.mic_state = "Mic Off"


    def toggle_mic():
        if session.mic_enabled:
            speech_engine.stop()
            session.mic_enabled = False
            session.mic_state = "Mic Off"
            session.set_draft("")
            live_widget.clear()
            session.set_toast("Microphone Off")
            return

        speech_engine.start()
        is_running = getattr(speech_engine, "is_running", getattr(speech_engine, "running", False))
        if is_running:
            session.mic_enabled = True
            session.set_toast("Microphone On")
        else:
            session.mic_enabled = False
            session.mic_state = "Error" if speech_engine.last_error else "Mic Off"
            if speech_engine.last_error:
                session.last_error = speech_engine.last_error
                session.set_toast(f"Mic: {speech_engine.last_error}")

    def start_dataset_recording():
        if sl_engine.has_review_clip():
            session.set_toast("Review or reject the pending clip first")
            return
        sl_engine.start_recording(session.current_dataset_label, session.current_signer_id)
        session.set_dataset_status("Recording")
        session.set_dataset_review_summary(None)
        session.set_dataset_live_feedback(sl_engine.dataset_manager.get_live_recording_feedback())
        session.set_toast(f"Recording {session.current_dataset_label} for {session.current_signer_id}")

    def stop_dataset_recording():
        review = sl_engine.stop_recording()
        session.set_dataset_review_summary(review)
        session.set_dataset_live_feedback(None)
        session.set_dataset_status("Review" if review else "Idle")
        if review:
            if not review["passes_quality_checks"]:
                session.set_toast("Clip blocked: fix sample quality before saving", duration=4.0)
            elif review["quality_warnings"]:
                session.set_toast("Clip has quality warnings; review before saving", duration=4.0)
            else:
                session.set_toast(f"Clip ready for review: {review['frame_count']} frames")

    def accept_dataset_clip():
        saved = sl_engine.accept_recording()
        if not saved:
            return
        if saved.get("status") == "blocked":
            blockers = saved.get("quality_blockers", [])
            blocker_text = blockers[0] if blockers else "Clip does not meet dataset quality thresholds."
            session.set_dataset_review_summary(sl_engine.get_review_summary())
            session.set_dataset_status("Review")
            session.set_toast(blocker_text, duration=4.0)
            return
        session.set_dataset_review_summary(None)
        session.set_dataset_live_feedback(None)
        session.set_dataset_status("Idle")
        refresh_dataset_insights()
        session.set_toast(f"Saved clip: {saved['label']} ({saved['frame_count']} frames)")

    def reject_dataset_clip(reason: str = "Rejected"):
        rejected = sl_engine.reject_recording(reason)
        if not rejected:
            return
        session.set_dataset_review_summary(None)
        session.set_dataset_live_feedback(None)
        session.set_dataset_status("Idle")
        session.set_toast(f"Clip discarded: {rejected['label']}")

    def toggle_dataset_mode():
        if session.dataset_mode:
            if sl_engine.dataset_manager.is_recording:
                stop_dataset_recording()
                reject_dataset_clip("Cancelled when exiting dataset mode")
            elif sl_engine.has_review_clip():
                reject_dataset_clip("Cancelled when exiting dataset mode")
            session.set_dataset_mode(False)
            return

        session.set_dataset_mode(True)
        session.set_dataset_status("Idle")
        session.set_dataset_review_summary(None)
        refresh_dataset_insights()

    def export_dataset_summary():
        exported_files = sl_engine.export_dataset("both")
        if not exported_files:
            session.set_dataset_export_result("", False)
            session.set_toast("No accepted dataset clips available to export", duration=4.0)
            return

        export_base = exported_files[0].rsplit(".", 1)[0]
        session.set_dataset_export_result(export_base, True)
        refresh_dataset_insights()
        session.set_toast(f"Dataset exported: {len(exported_files)} files", duration=4.0)

    def trigger_export():
        if session.dataset_mode:
            export_dataset_summary()
        else:
            session.export()

    # Window initialization - Launch Fullscreen
    window_name = "SAKSHAM V2 - AI Communication Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    def toggle_fullscreen_state():
        nonlocal is_fullscreen
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            session.set_toast("Fullscreen: ON")
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            session.set_toast("Fullscreen: OFF")

    # Bind global references for mouse interaction
    global global_ui, global_live_widget, global_toggle_mic_fn, global_export_fn, global_toggle_fullscreen_fn
    global_ui = ui
    global_live_widget = live_widget
    global_toggle_mic_fn = toggle_mic
    global_export_fn = trigger_export
    global_toggle_fullscreen_fn = toggle_fullscreen_state

    cv2.setMouseCallback(window_name, mouse_callback)

    main_fps = 0.0
    ema_alpha = 0.1
    elapsed_time = 1.0 / config.TARGET_FPS

    logging.info("Main loop active. Press 'q' / ESC to quit, 'f' to toggle Fullscreen, 'd' to toggle DEV MODE.")

    try:
        while True:
            if global_request_quit:
                logging.info("Quit requested via UI window control.")
                break

            t_start = time.perf_counter()

            if session.request_mic_toggle:
                session.request_mic_toggle = False
                toggle_mic()

            # ── 1. Grab frame ──────────────────────────────────────────
            success, raw_frame = camera.get_frame()

            if not success or raw_frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(raw_frame, 1)

            # ── 2. Sign Language Tracking & Recognition ──
            hand_detected, hand_data = sl_engine.process_frame(frame)
            session.set_live_hand_visibility(hand_data if hand_detected else None)

            # Forward inference state to session (for UI display)
            if sl_engine.inference is not None:
                inf_state = sl_engine.inference.get_state()
                raw_state = sl_engine.inference.get_raw_state()
                session.live_prediction = inf_state["prediction"]
                session.live_confidence = inf_state["confidence"]
                session.live_top3 = inf_state["top3"]
                session.buffer_progress = inf_state["buffer_progress"]
                session.buffer_capacity = inf_state["buffer_capacity"]

                session.raw_prediction = raw_state.get("raw_prediction")
                session.raw_confidence = raw_state.get("raw_confidence", 0.0)
                session.raw_top3 = raw_state.get("raw_top3", [])
                session.vote_window_size = raw_state.get("vote_window_size", 0)
                session.vote_window_capacity = raw_state.get("vote_window_capacity", 7)
            else:
                session.live_prediction = None
                session.live_confidence = 0.0
                session.live_top3 = []
                session.buffer_progress = 0
                session.buffer_capacity = 64
                session.raw_prediction = None
                session.raw_confidence = 0.0
                session.raw_top3 = []
                session.vote_window_size = 0
                session.vote_window_capacity = 7

            if session.dataset_mode and sl_engine.dataset_manager.is_recording:
                session.set_dataset_live_feedback(sl_engine.dataset_manager.get_live_recording_feedback())
            elif not sl_engine.has_review_clip():
                session.set_dataset_live_feedback(None)

            # ── 3. Render UI ─────────────────────────────────────────
            annotated_frame = ui.render(
                camera_frame=frame,
                session=session,
                fps=main_fps,
                scroll_offset=scroll_offset,
                hand_data=hand_data if hand_detected else None
            )

            # ── 4. Display ───────────────────────────────────────────
            annotated_frame = live_widget.draw(annotated_frame, session=session, mic_enabled=session.mic_enabled)
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key != 255: # A key was pressed
                if session.is_typing_focused:
                    if key == 13: # Enter
                        if session.typing_buffer.strip():
                            session.add_message("Hearing Person", session.typing_buffer.strip(), source="Typed")
                        session.typing_buffer = ""
                    elif key == 27: # Esc
                        session.is_typing_focused = False
                        session.typing_buffer = ""
                    elif key == 8: # Backspace
                        session.typing_buffer = session.typing_buffer[:-1]
                    elif 32 <= key <= 126: # Printable ASCII
                        session.typing_buffer += chr(key)
                else:
                    if key in (ord('q'), ord('Q'), 27):
                        logging.info("Quit key pressed.")
                        break
                    elif key in (ord('f'), ord('F')):
                        toggle_fullscreen_state()
                    elif key in (ord('m'), ord('M')):
                        toggle_mic()
                    elif key == ord(' '):
                        # Push-to-talk via SPACE
                        try:
                            if hasattr(speech_engine, '_capturing') and not getattr(speech_engine, '_capturing'):
                                if hasattr(speech_engine, 'push_start'):
                                    speech_engine.push_start()
                                    live_widget.set_preview("🎤 Listening...")
                                else:
                                    toggle_mic()
                            else:
                                if hasattr(speech_engine, 'push_stop'):
                                    speech_engine.push_stop()
                                else:
                                    toggle_mic()
                        except Exception:
                            pass
                    elif key in (ord('k'), ord('K')):
                        toggle_dataset_mode()
                    elif key in (ord('d'), ord('D')):
                        config.DEV_MODE = not config.DEV_MODE
                        logging.info(f"Developer Mode: {'ON' if config.DEV_MODE else 'OFF'}")
                    elif key in (ord('c'), ord('C')):
                        session.clear()
                    elif key in (ord('e'), ord('E')):
                        if session.dataset_mode:
                            export_dataset_summary()
                        else:
                            session.export()
                    elif key in (ord('o'), ord('O')):
                        # Open exports folder in Explorer (Windows)
                        import subprocess
                        from pathlib import Path
                        project_root = Path(__file__).resolve().parent.parent
                        try:
                            subprocess.Popen('explorer exports', shell=True, cwd=str(project_root))
                            logging.info('Opened exports folder.')
                        except Exception as e:
                            logging.error(f'Failed to open exports folder: {e}')
                    elif session.dataset_mode and key in (ord('j'), ord('J')):
                        if not sl_engine.dataset_manager.is_recording and not sl_engine.has_review_clip():
                            session.cycle_dataset_label(-1)
                    elif session.dataset_mode and key in (ord('l'), ord('L')):
                        if not sl_engine.dataset_manager.is_recording and not sl_engine.has_review_clip():
                            session.cycle_dataset_label(1)
                    elif session.dataset_mode and key in (ord('u'), ord('U')):
                        if not sl_engine.dataset_manager.is_recording and not sl_engine.has_review_clip():
                            session.cycle_dataset_signer(-1)
                    elif session.dataset_mode and key in (ord('i'), ord('I')):
                        if not sl_engine.dataset_manager.is_recording and not sl_engine.has_review_clip():
                            session.cycle_dataset_signer(1)
                    elif session.dataset_mode and key in (ord('r'), ord('R')):
                        if sl_engine.dataset_manager.is_recording:
                            stop_dataset_recording()
                        elif sl_engine.has_review_clip():
                            reject_dataset_clip("Re-record requested")
                            start_dataset_recording()
                        else:
                            start_dataset_recording()
                    elif session.dataset_mode and key in (ord('a'), ord('A')):
                        accept_dataset_clip()
                    elif session.dataset_mode and key in (ord('x'), ord('X')):
                        reject_dataset_clip()

            # Keyboard scrolling fallback
            if key == 0:  # Up Arrow
                scroll_offset += 1
            elif key == 1:  # Down Arrow
                scroll_offset -= 1

            # Keep scroll offset from going below 0
            if scroll_offset < 0:
                scroll_offset = 0

            # ── 5. Frame-rate cap ────────────────────────────────────
            elapsed_time = time.perf_counter() - t_start
            sleep_needed = (1.0 / config.TARGET_FPS) - elapsed_time
            if sleep_needed > 0:
                time.sleep(sleep_needed)

            t_end = time.perf_counter()
            loop_time = t_end - t_start
            if loop_time > 0:
                main_fps = ema_alpha * (1.0 / loop_time) + (1 - ema_alpha) * main_fps
            elapsed_time = loop_time if loop_time > 0 else (1.0 / config.TARGET_FPS)

    except Exception as e:
        logging.exception(f"Unhandled error: {e}")

    finally:
        camera.stop()
        speech_engine.stop()
        tts_engine.stop()
        cv2.destroyAllWindows()
        logging.info("SAKSHAM V2.0 shutdown cleanly.")
        logging.info("==========================================")

if __name__ == "__main__":
    main()
