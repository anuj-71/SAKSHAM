# Speech-to-Text Migration: Google Web Speech API → Faster-Whisper

## Summary

SAKSHAM's STT pipeline has been migrated from Google Web Speech API (network-based) to **Faster-Whisper** (local inference) with **Silero VAD** for voice activity detection.

---

## What Changed

### Before (Google Web Speech API)
- **Engine**: `src/speech/speech_engine.py` (`SpeechEngine`)
- **Backend**: Google Web Speech API (network round-trip per phrase)
- **VAD**: webrtcvad (energy-threshold-based)
- **Latency**: 500–1500ms network round-trip + queueing
- **Pause threshold**: 0.9s (conservative to avoid false phrase-ends)
- **Dependencies**: `SpeechRecognition`, `webrtcvad-wheels`

### After (Faster-Whisper)
- **Engine**: `src/speech/speech_engine_whisper.py` (`WhisperSpeechEngine`)
- **Backend**: Faster-Whisper (local ONNX inference, base model ~70M params)
- **VAD**: Silero VAD (neural VAD, more robust than energy thresholds)
- **Latency**: 100–200ms CPU inference (no network dependency)
- **Pause threshold**: 0.6s (reduced from 0.9s — network latency eliminated)
- **Non-speaking duration**: 0.3s (reduced from 0.5s — tighter phrase boundaries)
- **Dependencies**: `faster-whisper>=1.0.0`, `silero-vad>=5.0.0`, `pyaudio>=0.2.13`

---

## Expected Performance

| Metric | Google API | Faster-Whisper (base) | Δ |
|---|---|---|---|
| **Median latency** | ~800ms | ~150ms | **−650ms** |
| **P95 latency** | ~1500ms | ~250ms | **−1250ms** |
| **Word Error Rate** | ~5% (LibriSpeech) | ~6% (LibriSpeech) | +1% |
| **Network dependency** | Required | None | ✅ Offline-capable |
| **Pause threshold** | 0.9s | 0.6s | **−0.3s** (snappier turn-taking) |

### Model Size Tradeoffs

| Model | Params | CPU Latency | WER (LibriSpeech) | Recommendation |
|---|---|---|---|---|
| `tiny` | 39M | ~50ms | ~10% | Too inaccurate for ISL + English |
| **`base`** | **72M** | **~150ms** | **~6%** | **✅ Recommended (default)** |
| `small` | 244M | ~300ms | ~5% | Slower, minimal accuracy gain |

---

## Configuration

Edit `src/config/settings.py`:

```python
# Speech Recognition Settings
STT_BACKEND = "whisper"       # "whisper" (local) or "google" (network API)
WHISPER_MODEL_SIZE = "base"   # "tiny", "base", "small"
WHISPER_DEVICE = "cpu"        # "cpu" or "cuda" (GPU)
WHISPER_COMPUTE_TYPE = "int8" # "int8" (quantized) or "float32"
```

### Switching Back to Google API (if needed)

```python
STT_BACKEND = "google"  # Reverts to src/speech/speech_engine.py
```

---

## Architecture

Both engines use the same **producer/consumer** pipeline:

```
Listener Thread  →  audio_queue  →  Recognition Worker Thread
 (captures mic)                       (runs Whisper/Google API)
                                              │
                                      on_text_callback(text)
```

### Key Differences

| Component | Google API | Faster-Whisper |
|---|---|---|
| **Mic capture** | `speech_recognition.Microphone` | `pyaudio.PyAudio` |
| **VAD** | webrtcvad (energy-based) | Silero VAD (neural) |
| **Transcription** | `recognizer.recognize_google(audio)` | `model.transcribe(audio_float)` |
| **Audio format** | WAV (via speech_recognition) | int16 PCM → float32 numpy array |
| **Phrase detection** | Energy threshold + silence | VAD probability + silence |

---

## Testing

### Smoke Test (validates initialization)

```bash
cd e:\gessture
.venv\Scripts\python.exe tests/test_whisper_engine_smoke.py
```

**Expected output:**
```
✓ Engine initialized in ~30s (first run downloads base model ~140MB)
✓ Whisper model loaded: True
✓ Silero VAD loaded: True
✓ Microphone available: True
```

### Backend Switching Test

```bash
.venv\Scripts\python.exe tests/test_stt_backend_switch.py
```

Validates that `settings.STT_BACKEND` correctly toggles between engines.

### Latency Benchmark (requires network + mic)

```bash
.venv\Scripts\python.exe tests/test_stt_latency_benchmark.py
```

Synthesizes 10 test phrases, transcribes with both engines, measures latency and WER.

---

## Migration Checklist

- [x] Implement `WhisperSpeechEngine` in `src/speech/speech_engine_whisper.py`
- [x] Integrate Silero VAD for voice activity detection
- [x] Add configuration switch in `src/config/settings.py`
- [x] Update `src/main.py` to select engine based on `settings.STT_BACKEND`
- [x] Reduce `pause_threshold` (0.9s → 0.6s) and `non_speaking_duration` (0.5s → 0.3s)
- [x] Update `requirements.txt` with new dependencies
- [x] Create smoke test (`test_whisper_engine_smoke.py`)
- [x] Create backend switching test (`test_stt_backend_switch.py`)
- [x] Create latency benchmark harness (`test_stt_latency_benchmark.py`)
- [ ] Run live user acceptance test (manual validation with mic)
- [ ] Remove deprecated dependencies (`SpeechRecognition`, `webrtcvad-wheels`) after validation

---

## Troubleshooting

### Model download hangs on first run
**Symptom:** Engine initialization takes >2 minutes or hangs.  
**Cause:** Hugging Face model download is slow or network is unavailable.  
**Fix:** 
```bash
# Pre-download model manually:
.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu')"
```

### "Microphone unavailable" on headless/CI systems
**Expected behavior.** PyAudio requires a physical mic. For CI, mock the mic or skip STT tests.

### High CPU usage
**Cause:** Whisper inference is CPU-intensive.  
**Fix:** Use GPU if available (`WHISPER_DEVICE = "cuda"`), or switch to `tiny` model for lower CPU usage (with accuracy tradeoff).

### False phrase-ends (speech gets cut off mid-sentence)
**Cause:** `pause_threshold` is too aggressive.  
**Fix:** Increase `self.pause_threshold` in `speech_engine_whisper.py` (e.g., 0.6s → 0.8s).

---

## Why This Migration?

1. **Latency**: Eliminates 500–1500ms network round-trip → 100–200ms local inference.
2. **Reliability**: No dependency on Google API availability or rate limits.
3. **Privacy**: Audio never leaves the device.
4. **Offline capability**: Works without internet connection.
5. **Snappier UX**: Reduced pause thresholds enable faster turn-taking in conversations.

---

## References

- [Faster-Whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
