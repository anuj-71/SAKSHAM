import os
import cv2
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_SIGNS = [
    "Hello",
    "Goodbye",
    "Please",
    "Thank_You",
    "Yes",
    "No",
    "Help",
]
# Number of repetitions per sign (adjustable by user)
DEFAULT_REPS = 25
# Recording settings
FRAME_WIDTH = 1280  # 720p width
FRAME_HEIGHT = 720   # 720p height
FPS = 30
# Duration of each clip in seconds (you can also stop early by pressing 'q')
CLIP_DURATION = 2.5
# Output base directory – stored under the project 'dataset' folder
OUTPUT_BASE = Path(__file__).resolve().parents[2] / "dataset"

def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)

def countdown(seconds: int = 3) -> None:
    """Simple visual countdown printed to console."""
    for i in range(seconds, 0, -1):
        print(f"Recording starts in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("" * 30, end="\r")  # clear line

def record_clip(sign_name: str, rep_idx: int) -> None:
    """Record a single clip for a given sign.

    The clip is saved as `<OUTPUT_BASE>/<sign_name>/rep_{rep_idx}.mp4`.
    """
    sign_dir = OUTPUT_BASE / sign_name
    ensure_dir(sign_dir)
    out_path = sign_dir / f"rep_{rep_idx:03d}.mp4"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Ensure it is connected and not used by another program.")

    # Set resolution and FPS – may fail on some cameras, but we attempt.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break
        # Show the frame with a simple overlay indicating sign & rep
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"Sign: {sign_name}  Rep: {rep_idx}/{DEFAULT_REPS}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        # Blend overlay for readability
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.imshow("SAKSHAM Recording (press 'q' to abort)", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording aborted by user.")
            break
        if time.time() - start_time >= CLIP_DURATION:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out_path}")

def main() -> None:
    """Interactive script to record all target signs.

    The user is prompted sign‑by‑sign. For each sign a countdown is shown
    before the 2.5‑second clip is captured. The default number of repetitions
    can be overridden by passing an integer argument when running the script:

        python -m tools.record_signs.py [reps]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Record webcam clips for ISL signs.")
    parser.add_argument(
        "reps",
        type=int,
        nargs="?",
        default=DEFAULT_REPS,
        help=f"Number of repetitions per sign (default: {DEFAULT_REPS})",
    )
    args = parser.parse_args()
    reps = args.reps

    print("--- SAKSHAM Sign Recording Utility ---")
    print(f"Will record {reps} repetitions for each of {len(TARGET_SIGNS)} signs.")
    print("Press ENTER to start each sign, and 'q' during a clip to abort.")
    print("Make sure you are in a well‑lit area with a plain background for best MediaPipe results.\n")

    for sign in TARGET_SIGNS:
        input(f"Ready to record sign '{sign}'. Press ENTER when you are ready... ")
        for i in range(1, reps + 1):
            countdown(3)
            print(f"Recording {sign} – repetition {i}/{reps}")
            record_clip(sign, i)
            # Small pause between repetitions for the user to reset posture
            time.sleep(1)
        print(f"Finished recordings for '{sign}'.\n")

    print("All recordings completed! Check the 'dataset' folder for the saved videos.")

if __name__ == "__main__":
    main()
