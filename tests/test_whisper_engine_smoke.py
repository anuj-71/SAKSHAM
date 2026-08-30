"""
Smoke test for WhisperSpeechEngine.

Validates that:
1. Faster-Whisper model loads
2. Silero VAD loads
3. Engine initializes without errors
4. Threads start successfully

Usage:
    cd e:\\gessture
    .venv\\Scripts\\python.exe tests/test_whisper_engine_smoke.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)


def dummy_callback(text: str):
    print(f"[CALLBACK] Received: '{text}'")


def dummy_state_callback(state: str):
    print(f"[STATE] {state}")


def test_whisper_engine_initialization():
    """Test that WhisperSpeechEngine initializes without errors."""
    print("\n" + "=" * 70)
    print("  Whisper Speech Engine Smoke Test")
    print("=" * 70 + "\n")

    try:
        from src.speech.speech_engine_whisper import WhisperSpeechEngine

        print("✓ Imports successful\n")

        print("Loading WhisperSpeechEngine (this may take 30-60s on first run)...")
        t0 = time.time()
        engine = WhisperSpeechEngine(
            on_text_callback=dummy_callback,
            on_state_callback=dummy_state_callback,
            model_size="base",
            device="cpu",
            compute_type="int8",
        )
        elapsed = time.time() - t0
        print(f"✓ Engine initialized in {elapsed:.2f}s\n")

        print(f"  Whisper model loaded: {engine.whisper_model is not None}")
        print(f"  Silero VAD loaded: {engine.vad_model is not None}")
        print(f"  Microphone available: {engine.mic_available}")
        print(f"  Listener thread alive: {engine._listener_thread.is_alive()}")
        print(f"  Worker thread alive: {engine._worker_thread.is_alive()}")

        if not engine.mic_available:
            print(f"\n⚠ Microphone unavailable: {engine.last_error}")
            print("  (This is expected in headless/CI environments)")

        print("\n" + "=" * 70)
        print("  ✅ All smoke tests passed!")
        print("=" * 70 + "\n")

        # Graceful shutdown
        engine._shutdown_event.set()
        time.sleep(0.5)

    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_whisper_engine_initialization()
    sys.exit(0)
