"""
Phase 1 curated vocabulary for SAKSHAM Indian Sign Language recognition.

Conventions:
  - Labels use lowercase with underscores (e.g. thank_you, dont_understand).
  - The first entry (index 0) is reserved for "unknown" (non-sign gestures).
  - All remaining entries form the active recognition vocabulary.
"""

PHASE1_WORDS = [
    # Greetings
    "hello",
    "goodbye",
    "thank_you",
    "please",
    "sorry",
    # Basic Responses
    "yes",
    "no",
    "okay",
    "understand",
    "dont_understand",
    # Emergency
    "help",
    "stop",
    "doctor",
    "hospital",
    "call",
    # Daily Needs
    "water",
    "food",
    "eat",
    "drink",
    "bathroom",
    "sleep",
    "sit",
    "stand",
    "come",
    "go",
    # Common Communication
    "name",
    "pain",
    "hungry",
    "friend",
    "home",
]

UNKNOWN_LABEL = "unknown"

def get_full_vocabulary() -> list[str]:
    return [UNKNOWN_LABEL] + PHASE1_WORDS

def get_label_map() -> dict[str, int]:
    return {label: idx for idx, label in enumerate(get_full_vocabulary())}

def get_active_words() -> list[str]:
    return PHASE1_WORDS

def is_known(label: str) -> bool:
    return label in get_label_map()
