"""
Import-only validation for STT migration.

Tests that all modules import correctly without actually
instantiating engines (avoids model download).

Usage:
    cd e:\\gessture
    .venv\\Scripts\\python.exe tests/test_stt_imports_only.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    """Validate all STT-related imports work."""
    print("\n" + "=" * 70)
    print("  STT Migration — Import Validation")
    print("=" * 70 + "\n")

    errors = []

    # ── Core dependencies ─────────────────────────────────────────────────────
    print("Checking core dependencies...")
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError as e:
        errors.append(f"numpy: {e}")

    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import pyaudio
        print("  ✓ pyaudio")
    except ImportError as e:
        errors.append(f"pyaudio: {e}")

    # ── Faster-Whisper (check package exists without importing classes) ──────
    print("\nChecking Faster-Whisper...")
    try:
        import faster_whisper
        print("  ✓ faster_whisper package installed")
        print("    (Skipping WhisperModel import to avoid model download)")
    except ImportError as e:
        errors.append(f"faster_whisper: {e}")

    # ── Silero VAD (torch.hub access) ────────────────────────────────────────
    print("\nChecking Silero VAD...")
    try:
        # Don't load the model, just check torch.hub is accessible
        import torch.hub
        print("  ✓ torch.hub (accessible for Silero VAD)")
    except ImportError as e:
        errors.append(f"torch.hub: {e}")

    # ── Project modules ───────────────────────────────────────────────────────
    print("\nChecking project modules...")
    try:
        from src.config import settings
        print(f"  ✓ src.config.settings")
        print(f"    - STT_BACKEND = {settings.STT_BACKEND}")
        print(f"    - WHISPER_MODEL_SIZE = {settings.WHISPER_MODEL_SIZE}")
    except ImportError as e:
        errors.append(f"src.config.settings: {e}")

    try:
        from src.speech.base import BaseSpeechEngine
        print("  ✓ src.speech.base.BaseSpeechEngine")
    except ImportError as e:
        errors.append(f"src.speech.base: {e}")

    try:
        from src.speech.speech_engine import SpeechEngine
        print("  ✓ src.speech.speech_engine.SpeechEngine (Google API)")
    except ImportError as e:
        errors.append(f"src.speech.speech_engine: {e}")

    try:
        # Import the module without instantiating the engine (avoids model download)
        import src.speech.speech_engine_whisper
        print("  ✓ src.speech.speech_engine_whisper module loaded")
        print("    (Skipping engine instantiation to avoid model download)")
    except ImportError as e:
        errors.append(f"src.speech.speech_engine_whisper: {e}")

    # ── Main app conditional import logic ─────────────────────────────────────
    print("\nTesting main.py import logic (without instantiation)...")
    try:
        from src.config import settings

        if settings.STT_BACKEND == "whisper":
            # Import module only, don't access class to avoid model download
            import src.speech.speech_engine_whisper
            engine_name = "WhisperSpeechEngine (module loaded)"
        else:
            from src.speech.speech_engine import SpeechEngine as SelectedEngine
            engine_name = "SpeechEngine (Google API)"

        print(f"  ✓ Conditional import logic validated: {engine_name}")
    except Exception as e:
        errors.append(f"Conditional import logic: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if errors:
        print("  ❌ IMPORT VALIDATION FAILED")
        print("=" * 70)
        for err in errors:
            print(f"  - {err}")
        assert not errors, f"Import errors encountered: {errors}"
    else:
        print("  ✅ ALL IMPORTS VALID")
        print("=" * 70)
        print("\nNext step: Run smoke test to validate engine initialization:")
        print("  .venv\\Scripts\\python.exe tests/test_whisper_engine_smoke.py")


if __name__ == "__main__":
    test_imports()
    sys.exit(0)
