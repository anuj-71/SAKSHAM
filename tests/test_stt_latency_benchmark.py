"""
STT latency benchmark: Google Web Speech API vs Faster-Whisper.

Synthesizes 10 short phrases with Edge-TTS, transcribes with both engines,
measures wall-clock latency and Word Error Rate (WER).

Usage:
    cd e:\\gessture
    .venv\\Scripts\\python.exe tests/test_stt_latency_benchmark.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs during benchmark

# ── Test phrases ──────────────────────────────────────────────────────────────
TEST_PHRASES = [
    "Hello how are you",
    "I need help",
    "Thank you very much",
    "Please wait a moment",
    "Where is the bathroom",
    "Can you repeat that",
    "I am feeling sick",
    "What time is it",
    "Good morning",
    "See you later",
]


def synthesize_audio(text: str, output_path: str):
    """Synthesize speech using Edge-TTS and save to file."""
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts not installed")

    async def _save():
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        await communicate.save(output_path)

    asyncio.run(_save())


def load_audio_for_whisper(audio_path: str) -> np.ndarray:
    """Load audio file and convert to numpy array for Whisper (16kHz mono float32)."""
    try:
        import wave
    except ImportError:
        raise RuntimeError("wave module not available")

    # Use ffmpeg to convert to 16kHz mono PCM
    tmp_wav = audio_path.replace(".mp3", "_16k.wav")
    os.system(f'ffmpeg -i "{audio_path}" -ar 16000 -ac 1 -y "{tmp_wav}" > nul 2>&1')

    with wave.open(tmp_wav, "rb") as wf:
        audio_bytes = wf.readframes(wf.getnframes())
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

    os.remove(tmp_wav)
    return audio_float


def transcribe_google_api(audio_path: str) -> tuple[str, float]:
    """Transcribe using Google Web Speech API (via speech_recognition)."""
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    t0 = time.time()
    try:
        text = recognizer.recognize_google(audio)
        elapsed = time.time() - t0
        return text.strip(), elapsed
    except sr.UnknownValueError:
        return "", time.time() - t0
    except sr.RequestError as e:
        print(f"Google API error: {e}")
        return "", time.time() - t0


def transcribe_whisper(audio_float: np.ndarray, model) -> tuple[str, float]:
    """Transcribe using Faster-Whisper."""
    t0 = time.time()
    segments, info = model.transcribe(
        audio_float,
        language="en",
        beam_size=1,
        vad_filter=False,
        without_timestamps=True,
    )
    text_parts = [seg.text.strip() for seg in segments]
    text = " ".join(text_parts).strip()
    elapsed = time.time() - t0
    return text, elapsed


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (Levenshtein distance on words)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Levenshtein distance on word level
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words) if ref_words else 0.0


def run_benchmark():
    """Run full STT benchmark comparing Google API vs Faster-Whisper."""
    print("\n" + "=" * 70)
    print("  STT LATENCY BENCHMARK: Google Web Speech API vs Faster-Whisper")
    print("=" * 70)

    # ── Load Whisper model ────────────────────────────────────────────────────
    try:
        from faster_whisper import WhisperModel
        print("\nLoading Faster-Whisper 'base' model...")
        t0 = time.time()
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print(f"  Loaded in {time.time() - t0:.2f}s\n")
    except Exception as e:
        print(f"Failed to load Faster-Whisper: {e}")
        return

    # ── Synthesize test audio ─────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="stt_bench_")
    print(f"Synthesizing {len(TEST_PHRASES)} test phrases...")
    audio_files = []
    for i, phrase in enumerate(TEST_PHRASES):
        audio_path = os.path.join(tmp_dir, f"phrase_{i:02d}.mp3")
        synthesize_audio(phrase, audio_path)
        audio_files.append(audio_path)
    print(f"  Audio files saved to {tmp_dir}\n")

    # ── Convert audio to WAV for speech_recognition ───────────────────────────
    wav_files = []
    for mp3_path in audio_files:
        wav_path = mp3_path.replace(".mp3", ".wav")
        os.system(f'ffmpeg -i "{mp3_path}" -ar 16000 -ac 1 -y "{wav_path}" > nul 2>&1')
        wav_files.append(wav_path)

    # ── Benchmark Google API ──────────────────────────────────────────────────
    print("Running Google Web Speech API benchmark...")
    google_results = []
    for i, (phrase, wav_path) in enumerate(zip(TEST_PHRASES, wav_files)):
        print(f"  [{i+1}/{len(TEST_PHRASES)}] '{phrase}'", end=" ")
        try:
            transcript, elapsed = transcribe_google_api(wav_path)
            wer = compute_wer(phrase, transcript)
            google_results.append({
                "phrase": phrase,
                "transcript": transcript,
                "latency": elapsed,
                "wer": wer,
            })
            print(f"→ '{transcript}' ({elapsed:.3f}s, WER={wer:.2%})")
        except Exception as e:
            print(f"ERROR: {e}")
            google_results.append({
                "phrase": phrase,
                "transcript": "",
                "latency": float("nan"),
                "wer": 1.0,
            })
    print()

    # ── Benchmark Faster-Whisper ──────────────────────────────────────────────
    print("Running Faster-Whisper benchmark...")
    whisper_results = []
    for i, (phrase, mp3_path) in enumerate(zip(TEST_PHRASES, audio_files)):
        print(f"  [{i+1}/{len(TEST_PHRASES)}] '{phrase}'", end=" ")
        try:
            audio_float = load_audio_for_whisper(mp3_path)
            transcript, elapsed = transcribe_whisper(audio_float, whisper_model)
            wer = compute_wer(phrase, transcript)
            whisper_results.append({
                "phrase": phrase,
                "transcript": transcript,
                "latency": elapsed,
                "wer": wer,
            })
            print(f"→ '{transcript}' ({elapsed:.3f}s, WER={wer:.2%})")
        except Exception as e:
            print(f"ERROR: {e}")
            whisper_results.append({
                "phrase": phrase,
                "transcript": "",
                "latency": float("nan"),
                "wer": 1.0,
            })
    print()

    # ── Aggregate results ─────────────────────────────────────────────────────
    google_latencies = [r["latency"] for r in google_results if not np.isnan(r["latency"])]
    whisper_latencies = [r["latency"] for r in whisper_results if not np.isnan(r["latency"])]

    google_wers = [r["wer"] for r in google_results if not np.isnan(r["latency"])]
    whisper_wers = [r["wer"] for r in whisper_results if not np.isnan(r["latency"])]

    google_mean_latency = np.mean(google_latencies) if google_latencies else float("nan")
    whisper_mean_latency = np.mean(whisper_latencies) if whisper_latencies else float("nan")

    google_mean_wer = np.mean(google_wers) if google_wers else float("nan")
    whisper_mean_wer = np.mean(whisper_wers) if whisper_wers else float("nan")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Google Web Speech API:")
    print(f"    Mean latency:  {google_mean_latency:.3f}s")
    print(f"    Mean WER:      {google_mean_wer:.2%}")
    print(f"    Min latency:   {min(google_latencies):.3f}s")
    print(f"    Max latency:   {max(google_latencies):.3f}s")

    print(f"\n  Faster-Whisper (base, int8):")
    print(f"    Mean latency:  {whisper_mean_latency:.3f}s")
    print(f"    Mean WER:      {whisper_mean_wer:.2%}")
    print(f"    Min latency:   {min(whisper_latencies):.3f}s")
    print(f"    Max latency:   {max(whisper_latencies):.3f}s")

    delta_latency = whisper_mean_latency - google_mean_latency
    delta_wer = whisper_mean_wer - google_mean_wer

    print(f"\n  DELTA (Whisper - Google):")
    print(f"    Latency:       {delta_latency:+.3f}s  "
          f"({'faster' if delta_latency < 0 else 'slower'})")
    print(f"    WER:           {delta_wer:+.2%}  "
          f"({'better' if delta_wer < 0 else 'worse'})")
    print("=" * 70)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    import shutil
    shutil.rmtree(tmp_dir)
    print(f"\nCleaned up {tmp_dir}")


if __name__ == "__main__":
    run_benchmark()
