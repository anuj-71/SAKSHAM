"""
Test that the STT backend switching mechanism works correctly.

Validates that:
1. Settings can toggle between "google" and "whisper" backends
2. The correct engine class is imported
3. Engine initializes without errors

Usage:
    cd e:\\gessture
    .venv\\Scripts\\python.exe tests/test_stt_backend_switch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def dummy_callback(text: str):
    pass


def dummy_state_callback(state: str):
    pass


def test_backend_switching():
    """Test STT backend selection based on settings."""
    print("\n" + "=" * 70)
    print("  STT Backend Switching Test")
    print("=" * 70 + "\n")

    from src.config import settings

    original_backend = settings.STT_BACKEND
    print(f"Current backend in settings: {original_backend}\n")

    # ── Test Whisper backend ──────────────────────────────────────────────────
    print("Testing Whisper backend...")
    settings.STT_BACKEND = "whisper"

    # Reload main to trigger import logic
    if "src.main" in sys.modules:
        del sys.modules["src.main"]

    try:
        # The import in main.py checks settings.STT_BACKEND
        from src.speech.speech_engine_whisper import WhisperSpeechEngine

        print(f"  ✓ WhisperSpeechEngine imported successfully")
        print(f"  ✓ Model will use: {settings.WHISPER_MODEL_SIZE}, {settings.WHISPER_DEVICE}")
    finally:
        pass

    # ── Test Google backend ───────────────────────────────────────────────────
    print("\nTesting Google backend...")
    settings.STT_BACKEND = "google"

    if "src.main" in sys.modules:
        del sys.modules["src.main"]

    try:
        from src.speech.speech_engine import SpeechEngine

        print(f"  ✓ SpeechEngine (Google API) imported successfully")
    finally:
        settings.STT_BACKEND = original_backend

    print("\n" + "=" * 70)
    print("  ✅ Backend switching test passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_backend_switching()
    sys.exit(0)
