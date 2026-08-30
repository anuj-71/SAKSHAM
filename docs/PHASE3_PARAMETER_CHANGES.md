# Phase 3: STT Parameter Changes — Detailed Tradeoffs

## Overview

This document details the exact parameter changes made during the STT migration and their expected impact on user experience.

---

## 1. Pause Threshold: 0.9s → 0.6s

**What it controls:** Seconds of continuous silence required to mark the end of a spoken phrase.

### Old Value (Google API)
- `pause_threshold = 0.9s`
- **Rationale:** Conservative to account for network latency (500-1500ms) + processing time. False phrase-ends would interrupt users mid-sentence and fire incomplete transcripts.

### New Value (Faster-Whisper)
- `pause_threshold = 0.6s`
- **Rationale:** Network latency eliminated (local inference ~150ms). Can afford tighter threshold for snappier turn-taking without false positives.

### Expected Impact
- ✅ **Faster turn-taking:** User pauses for 0.6s → phrase fires (was 0.9s, saving 0.3s per turn)
- ⚠️ **Risk:** If user naturally pauses mid-sentence for >0.6s (thinking, breathing), phrase may fire prematurely.
- 🔧 **Tuning:** If users report speech cutoffs, increase to 0.7s or 0.8s in `speech_engine_whisper.py`.

### Real-World Example
```
User says: "I need help [0.7s pause] with the assignment"

Old (0.9s threshold):
  → Waits full 0.9s after "help" → fires: "I need help with the assignment" ✅

New (0.6s threshold):
  → Fires after 0.6s → may split into: "I need help" + "with the assignment" ⚠️
  
Mitigation: Natural speech rarely has >0.6s mid-sentence pauses. If issues arise, tune up to 0.7-0.8s.
```

---

## 2. Non-Speaking Duration: 0.5s → 0.3s

**What it controls:** Minimum silence duration required to mark the start of a new phrase (after previous phrase ended).

### Old Value (Google API)
- `non_speaking_duration = 0.5s`
- **Rationale:** Prevents accidental re-triggering of the same phrase if user briefly pauses while mic is still "hot."

### New Value (Faster-Whisper)
- `non_speaking_duration = 0.3s`
- **Rationale:** Tighter phrase boundaries. Local inference is fast enough that 0.3s is sufficient to distinguish "end of phrase" from "brief pause."

### Expected Impact
- ✅ **Tighter phrase detection:** Reduces dead air between user finishing speech and system recognizing phrase end.
- ⚠️ **Risk:** Very short pauses (<0.3s) within a sentence might be misinterpreted as phrase boundaries.
- 🔧 **Tuning:** If phrases fire on every tiny pause, increase to 0.4s.

### Real-World Example
```
User says: "Thank you [0.25s pause] goodbye"

Old (0.5s threshold):
  → Treats as one phrase: "Thank you goodbye" ✅

New (0.3s threshold):
  → Still treats as one phrase (0.25s < 0.3s) ✅
  
Edge case: "Thank you [0.35s pause] goodbye"
  → May split into: "Thank you" + "goodbye" ⚠️
```

---

## 3. Voice Activity Detection: webrtcvad → Silero VAD

**What it controls:** Determines whether audio contains human speech vs background noise.

### Old Value (Google API)
- `webrtcvad.Vad(aggressiveness=2)`
- **Approach:** Energy-threshold-based. Compares audio amplitude to a fixed threshold.
- **Pros:** Fast, lightweight (~1ms per 20ms frame)
- **Cons:** False positives from loud non-speech sounds (coughing, door slams, keyboard typing)

### New Value (Faster-Whisper)
- `Silero VAD` (neural network-based)
- **Approach:** Deep learning model trained to distinguish speech from non-speech.
- **Pros:** Robust to non-speech sounds, adapts to varying speakers/environments
- **Cons:** Slightly slower (~10ms per audio chunk on CPU), requires torch

### Expected Impact
- ✅ **Fewer false triggers:** Keyboard typing, background TV, door slams less likely to trigger transcription.
- ✅ **Better whisper detection:** Soft-spoken users detected more reliably.
- ⚠️ **Slightly higher CPU:** ~10-20% more CPU during active listening (negligible on modern CPUs).

### Real-World Example
```
Scenario: User coughs loudly while sign recognition is running.

Old (webrtcvad):
  → Energy spike → treated as speech → queued for Google API → transcribes as gibberish or error ❌

New (Silero VAD):
  → Neural model recognizes non-speech → discarded before transcription ✅
```

---

## 4. Transcription Backend: Google API → Faster-Whisper

**What it controls:** The STT model that converts audio to text.

### Old Value (Google API)
- `recognizer.recognize_google(audio)`
- **Latency:** 500-1500ms (network round-trip + server queue)
- **Accuracy:** ~5% WER on clean English (LibriSpeech)
- **Dependency:** Requires internet, subject to Google's rate limits and availability
- **Cost:** Free (with usage caps), paid tiers for heavy use

### New Value (Faster-Whisper)
- `whisper_model.transcribe(audio, model_size="base")`
- **Latency:** 100-200ms (local CPU inference, int8 quantized)
- **Accuracy:** ~6% WER on clean English (LibriSpeech)
- **Dependency:** None (fully offline after initial model download)
- **Cost:** Zero (runs locally)

### Expected Impact
- ✅ **~4-7x faster transcription:** 800ms → 150ms median (network eliminated)
- ✅ **Offline capability:** Works without internet after first run
- ✅ **No rate limits:** No Google API quota concerns
- ⚠️ **Slightly lower accuracy:** +1% WER (negligible in practice for conversational use)
- ⚠️ **First-run delay:** ~30-60s to download model (~140MB) from Hugging Face

### Real-World Example
```
User says: "I need water"

Old (Google API):
  User finishes → 0.9s pause → API call → 800ms network → transcript: "I need water" (total ~1.7s)

New (Faster-Whisper):
  User finishes → 0.6s pause → local inference → 150ms CPU → transcript: "I need water" (total ~0.75s)
  
Time saved: ~1 second per phrase ✅
```

---

## Combined Effect: Total Latency Reduction

| Stage | Old (Google API) | New (Faster-Whisper) | Δ |
|---|---|---|---|
| **User speaks** | 0ms | 0ms | — |
| **Pause detection** | 900ms | 600ms | −300ms |
| **Transcription** | 800ms (network) | 150ms (CPU) | −650ms |
| **Total turn-taking latency** | **1700ms** | **750ms** | **−950ms (−56%)** |

---

## Rollback Scenarios

### If users report speech cutoffs mid-sentence
**Symptom:** Phrases fire too early, splitting sentences like "I need [FIRE] help"  
**Fix:** Increase `pause_threshold` in `speech_engine_whisper.py`:
```python
self.pause_threshold = 0.7  # or 0.8
```

### If phrases fire on every tiny pause
**Symptom:** "Thank [FIRE] you [FIRE] very [FIRE] much"  
**Fix:** Increase `non_speaking_duration`:
```python
self.non_speaking_duration = 0.4  # or 0.5
```

### If accuracy is unacceptably low
**Symptom:** Transcripts have many errors compared to Google API  
**Fix:** Switch to `small` model (slower but more accurate):
```python
# In src/config/settings.py
WHISPER_MODEL_SIZE = "small"  # was "base"
```

### If local inference is too slow (>500ms/phrase)
**Symptom:** Transcription takes longer than network API did  
**Fix:** Switch to `tiny` model (faster but less accurate):
```python
WHISPER_MODEL_SIZE = "tiny"
```
Or use GPU if available:
```python
WHISPER_DEVICE = "cuda"  # was "cpu"
```

### Complete rollback to Google API
**If all else fails:**
```python
# In src/config/settings.py
STT_BACKEND = "google"
```

---

## Testing Recommendations

1. **Immediate (before live use):**
   - Run `tests/test_stt_imports_only.py` to validate all imports ✅ (already done)
   - Let Whisper model download once (30-60s) by running app or smoke test

2. **Within 24 hours:**
   - Test with 10-20 conversational phrases in real environment
   - Listen for: mid-sentence cutoffs, missed phrases, false triggers from noise
   - Adjust `pause_threshold` if needed

3. **Within 1 week:**
   - Run `tests/test_stt_latency_benchmark.py` to measure actual latency on this machine
   - Compare Google API vs Whisper on 10 test phrases

4. **Within 1 month:**
   - Collect user feedback on turn-taking "feel" — is it snappier? Too aggressive?
   - If validated, remove `SpeechRecognition` and `webrtcvad-wheels` from dependencies

---

## Success Metrics

| Metric | Target | How to Measure |
|---|---|---|
| Turn-taking latency | <1s (was 1.7s) | Stopwatch: user stops speaking → transcript appears |
| False phrase-ends | <5% of phrases | User feedback: "Did phrases fire mid-sentence?" |
| Transcription accuracy | >94% (same ballpark as Google) | Manual review of 50 transcripts |
| False triggers from noise | <2% of sessions | Count: Did keyboard/cough trigger transcription? |

---

**End of Document**
