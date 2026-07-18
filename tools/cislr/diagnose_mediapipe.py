"""
Diagnostics for MediaPipe + HandTracker compatibility.

Prints mediapipe metadata, which solutions namespace was detected,
and attempts to initialize HandTracker and MediaPipe Hands.

Exit status:
  0 -> PASS (HandTracker initialized)
  1 -> FAIL (prints exception)

Do not start extraction in this script.
"""
import traceback
import sys
import importlib
import os
import pkgutil

# Ensure repository root is on sys.path so `src` imports work when run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mediapipe as mp

def locate_solutions():
    if hasattr(mp, "solutions"):
        return "mp.solutions", mp.solutions
    try:
        mp_python = importlib.import_module("mediapipe.python")
        if hasattr(mp_python, "solutions"):
            return "mediapipe.python.solutions", mp_python.solutions
    except Exception:
        pass
    try:
        sol = importlib.import_module("mediapipe.python.solutions")
        return "mediapipe.python.solutions", sol
    except Exception:
        pass
    return None, None


def main():
    print("mediapipe.__version__:", getattr(mp, "__version__", None))
    print("mediapipe.__file__:", getattr(mp, "__file__", None))
    print("hasattr(mp, 'solutions'):", hasattr(mp, "solutions"))

    which, sols = locate_solutions()
    print("detected_solutions:", which)

    # List mediapipe package modules for inspection
    try:
        names = [m.name for m in pkgutil.iter_modules(mp.__path__)]
    except Exception as e:
        names = [f"<error: {e}>"]
    print('mediapipe.__path__ modules:', names)

    # Try initializing HandTracker from project
    try:
        from src.sign_language.tracker import HandTracker
    except Exception as e:
        print("HandTracker import FAILED:")
        traceback.print_exc()
        sys.exit(1)

    try:
        ht = HandTracker()
        print("HandTracker() initialization: OK")
    except Exception as e:
        print("HandTracker() initialization FAILED:")
        traceback.print_exc()
        sys.exit(1)

    # Try to instantiate MediaPipe Hands via located solutions
    if sols is not None:
        try:
            hands = sols.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
            hands.close()
            print("mp_hands.Hands() instantiation: OK")
        except Exception:
            print("mp_hands.Hands() instantiation FAILED:")
            traceback.print_exc()
            # still treat as failure
            sys.exit(1)
    else:
        print("No mediapipe.solutions namespace available to instantiate Hands()")
        sys.exit(1)

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
