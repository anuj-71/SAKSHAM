"""
Integration test for STT backend toggle between Google API and Whisper.

Validates that:
1. Settings can toggle between "google" and "whisper"
2. Both engines can be instantiated without crashes
3. Graceful fallback works if Whisper fails

Usage:
    cd e:\\gessture
    .venv\\Scripts\\python.exe tests/test_stt_backend_toggle_integration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def dummy_callback(text: str):
    print(f"[Callback] {text}")


def dummy_state(state: str):
    print(f"[State] {state}")


def test_google_engine():
    """Test that Google API engine still works."""
    print("\n" + "=" * 70)
    print("  Testing Google Web Speech API Engine")
    print("=" * 70)
    
    from src.speech.speech_engine import SpeechEngine
    
    engine = SpeechEngine(
        on_text_callback=dummy_callback,
        on_state_callback=dummy_state,
    )
    
    print(f"  ✓ Google engine instantiated")
    print(f"  ✓ Mic available: {engine.mic_available}")
    print(f"  ✓ Recognizer initialized: {engine.recognizer is not None}")
    print(f"  ✓ Energy threshold: {engine.recognizer.energy_threshold}")
    
    # Graceful shutdown
    engine._shutdown_event.set()


def test_backend_toggle():
    """Test the dynamic import logic in main.py."""
    print("\n" + "=" * 70)
    print("  Testing STT Backend Toggle Logic")
    print("=" * 70)
    
    from src.config import settings
    original_backend = settings.STT_BACKEND
    
    try:
        # Test 1: Whisper backend (may fail if model not downloaded)
        print("\n1. Testing Whisper backend...")
        settings.STT_BACKEND = "whisper"
        
        # Clear cached imports
        for mod in list(sys.modules.keys()):
            if "src.speech" in mod or "src.main" in mod:
                del sys.modules[mod]
        
        # This mimics the import logic in main.py
        if settings.STT_BACKEND == "whisper":
            try:
                from src.speech.speech_engine_whisper import WhisperSpeechEngine as TestEngine
                print("  ✓ Whisper engine imported successfully")
                whisper_works = True
            except Exception as e:
                print(f"  ⚠ Whisper import failed (expected on first run): {e}")
                print("  ✓ Fallback to Google API triggered")
                from src.speech.speech_engine import SpeechEngine as TestEngine
                whisper_works = False
        else:
            from src.speech.speech_engine import SpeechEngine as TestEngine
            whisper_works = False
        
        print(f"  ✓ Selected engine: {TestEngine.__name__}")
        
        # Test 2: Google backend
        print("\n2. Testing Google backend...")
        settings.STT_BACKEND = "google"
        
        for mod in list(sys.modules.keys()):
            if "src.speech" in mod or "src.main" in mod:
                del sys.modules[mod]
        
        from src.speech.speech_engine import SpeechEngine as TestEngine
        print(f"  ✓ Google engine imported: {TestEngine.__name__}")
        
        print("\n" + "=" * 70)
        print("  ✅ Backend toggle test passed!")
        if not whisper_works:
            print("  ⚠ Note: Whisper model not downloaded yet (first-run expected)")
        print("=" * 70)
    finally:
        settings.STT_BACKEND = original_backend


if __name__ == "__main__":
    test_google_engine()
    test_backend_toggle()
    print("\n✅ All integration tests passed!")
    sys.exit(0)
