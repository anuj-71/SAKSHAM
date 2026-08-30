# Code Audit Findings — Phase 1-3 Review

**Date:** 2026-07-19  
**Scope:** Full src/ audit for bugs, dead code, error-handling gaps

---

## Critical Issues

### 1. **Hardcoded 15-class references in main.py (breaks with 13-class model)**
**Location:** `src/main.py` lines 132-133  
**Issue:** SIGN_TO_PHRASE dict still has "mom" and "dad" entries, which no longer exist in the 13-class model.  
**Impact:** If sign recognition fires "mom" or "dad" (impossible now), the dict lookup would succeed but TTS would speak removed phrases.  
**Severity:** Low (signs don't exist in model anymore, but dict is stale)  
**Fix:** Remove "mom" and "dad" from SIGN_TO_PHRASE dict.

```python
# BEFORE:
SIGN_TO_PHRASE = {
    ...
    "mom": "I need my mom.",
    "dad": "I need my dad.",
    ...
}

# AFTER:
SIGN_TO_PHRASE = {
    "hello":    "Hello!",
    "bye":      "Goodbye!",
    "yes":      "Yes.",
    "no":       "No.",
    "please":   "Please.",
    "thankyou": "Thank you!",
    "water":    "I need water.",
    "food":     "I need food.",
    "drink":    "I need a drink.",
    "hungry":   "I am hungry.",
    "home":     "I want to go home.",
    "sick":     "I am feeling sick.",
    "where":    "Where are you?",
}
```

---

### 2. **Hardcoded 15-class default in training modules (dead code)**
**Location:**  
- `src/training/models.py` line 55, 144, 184, 219
- `src/training/trainer.py` line 296

**Issue:** Training modules have `num_classes=15` as default parameter. These modules are NOT used by the live app (tools/train.py is the live pipeline), but stale defaults could confuse future retraining.  
**Impact:** Low (dead code, not used in live pipeline)  
**Severity:** Low  
**Fix:** Update default to 13 or remove defaults entirely (require explicit num_classes).

```python
# src/training/models.py
def __init__(self, ..., num_classes: int = 13, ...):  # was 15

# src/training/trainer.py (test code at bottom)
num_classes = 13  # was 15
```

---

### 3. **Unicode encoding error in test_stability_regression.py**
**Location:** `tests/test_stability_regression.py` line 269  
**Issue:** Uses Unicode ≈ character that fails on Windows cmd/PowerShell with cp1252 encoding.  
**Impact:** Test fails to run on Windows.  
**Severity:** High (blocks test execution)  
**Fix:** ✅ FIXED — Replace "≈" with "~" (ASCII-safe approximation symbol)

---

### 4. **No error handling for Whisper model download failure**
**Location:** `src/speech/speech_engine_whisper.py` lines 60-73  
**Issue:** If Faster-Whisper model download fails (no internet, HF hub down, disk full), exception is raised but not gracefully handled. App would crash on startup.  
**Impact:** App crashes if model download fails on first run.  
**Severity:** High  
**Fix:** Catch download errors, log warning, fall back to Google API or disable STT.

```python
# CURRENT (raises exception):
try:
    from faster_whisper import WhisperModel
    self.whisper_model = WhisperModel(model_size, device=device, ...)
except Exception as e:
    logger.exception(f"Failed to load Faster-Whisper: {e}")
    raise  # ❌ Crashes app

# PROPOSED (graceful degradation):
try:
    from faster_whisper import WhisperModel
    logger.info(f"Loading Faster-Whisper model '{model_size}'...")
    self.whisper_model = WhisperModel(model_size, device=device, ...)
except Exception as e:
    logger.error(f"Failed to load Faster-Whisper: {e}")
    logger.error("Whisper STT unavailable. Please check internet connection or switch to STT_BACKEND='google'")
    self.mic_available = False
    self.last_error = f"Whisper model load failed: {e}"
    raise RuntimeError(f"Whisper initialization failed: {e}") from e
```

**Better approach:** Catch in main.py and fall back to Google API:
```python
# src/main.py
if settings.STT_BACKEND == "whisper":
    try:
        from src.speech.speech_engine_whisper import WhisperSpeechEngine as SpeechEngine
    except Exception as e:
        logger.error(f"Whisper engine init failed: {e}")
        logger.warning("Falling back to Google Web Speech API")
        from src.speech.speech_engine import SpeechEngine
else:
    from src.speech.speech_engine import SpeechEngine
```

---

### 5. **No validation that STT_BACKEND="google" still works after Whisper migration**
**Location:** `src/main.py` lines 17-28  
**Issue:** Conditional import logic exists, but we haven't verified that switching back to `STT_BACKEND="google"` actually works. speech_recognition import is still present in speech_engine.py, but could break if dependency is removed.  
**Impact:** Users can't roll back to Google API if Whisper fails.  
**Severity:** Medium  
**Fix:** Keep speech_recognition in requirements.txt with deprecation comment (already done). Add integration test.

---

### 6. **"unknown" class filtering in engine.py is fragile**
**Location:** `src/sign_language/engine.py` line 94  
**Issue:** Hardcoded check for `new_pred == "unknown"` to filter non-sign gestures. If label_map.json doesn't have "unknown" class, this breaks silently.  
**Impact:** Low (current 13-class model doesn't have "unknown", so this is dead code)  
**Severity:** Low  
**Fix:** Remove dead code or make it robust.

```python
# CURRENT:
if new_pred == "unknown":
    new_pred = None

# PROPOSED (check if class exists):
if new_pred == "unknown" or new_pred not in SIGN_TO_PHRASE:
    new_pred = None
```

---

### 7. **Legacy heuristic recognition still active as fallback**
**Location:** `src/sign_language/engine.py` lines 104-197  
**Issue:** `_attempt_recognition()` method implements hand-coded heuristics for detecting signs. This is dead code if ONNX model loads (which it always does), but could fire if model fails to load.  
**Impact:** Low (only runs if ONNX fails to load, which is rare)  
**Severity:** Low  
**Fix:** Document that this is a fallback, or remove it entirely if ONNX is mandatory.

---

## Minor Issues

### 8. **Duplicate state update calls in speech engines**
**Location:** `src/speech/speech_engine_whisper.py` lines 342, 373  
**Issue:** Multiple `self.update_state("Listening")` calls after errors. Harmless but redundant.  
**Severity:** Trivial  
**Fix:** No fix needed (defensive programming is fine).

---

### 9. **No timeout on torch.hub.load for Silero VAD**
**Location:** `src/speech/speech_engine_whisper.py` line 79  
**Issue:** `torch.hub.load()` can hang indefinitely if network is slow. No timeout configured.  
**Impact:** App startup hangs on slow network.  
**Severity:** Medium  
**Fix:** Add timeout or pre-download model.

```python
# PROPOSED:
import socket
socket.setdefaulttimeout(30)  # 30s timeout for network operations
try:
    self.vad_model, self.vad_utils = torch.hub.load(...)
except Exception as e:
    logger.error(f"Silero VAD load failed (timeout or network issue): {e}")
    raise
```

---

### 10. **Test imports block on model download**
**Location:** `tests/test_whisper_engine_smoke.py`  
**Issue:** First-time model download can take 1-5 minutes, causing test timeouts.  
**Severity:** Low (expected on first run)  
**Fix:** Document in test docstring. Already handled by test_stt_imports_only.py.

---

## Non-Issues (False Positives)

### ✅ speech_recognition import in speech_engine.py
**Status:** OK — still used when `STT_BACKEND="google"`. Kept for backward compatibility.

### ✅ Hardcoded num_classes in models.py/trainer.py  
**Status:** Low priority — these are dead training modules not used by live pipeline (tools/train.py is canonical).

---

## Recommendations

1. **Immediate fixes:**
   - ✅ Fix Unicode encoding in test_stability_regression.py
   - Remove "mom"/"dad" from SIGN_TO_PHRASE in main.py
   - Add graceful fallback for Whisper model download failure

2. **Before next release:**
   - Add integration test for `STT_BACKEND="google"` toggle
   - Document that training/* modules are legacy (tools/train.py is live)
   - Add timeout to torch.hub.load for Silero VAD

3. **Future cleanup:**
   - Remove legacy heuristic recognition in engine.py if ONNX is mandatory
   - Update training module defaults from 15 to 13 classes

---

## Summary

| Category | Count |
|---|---|
| Critical bugs | 1 (Whisper download error handling) |
| High-priority issues | 2 (Unicode encoding, STT backend validation) |
| Medium-priority issues | 3 (Silero VAD timeout, dead code, stale defaults) |
| Low-priority issues | 4 (dead code, redundant calls, minor inconsistencies) |
| **Total issues** | **10** |

All issues have been documented with fixes. Proceeding to implementation...
