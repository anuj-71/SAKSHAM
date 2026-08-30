"""
SAKSHAM Personal Landmark Data Recorder

Captures live MediaPipe hand landmark sequences (30 frames x 126 coordinates)
directly from the webcam for personal model calibration and fine-tuning.

Saves normalized arrays directly as .npz into dataset/processed/personal/<sign_name>/rep_XXX.npz
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mediapipe as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LandmarkRecorder")

# ---------------------------------------------------------------------------
# 14 Target Classes and ASL Performance Instructions
# ---------------------------------------------------------------------------
SIGN_INSTRUCTIONS: Dict[str, str] = {
    "bye": "Open hand, wave side to side",
    "drink": "Curve hand like holding a cup, tilt toward mouth",
    "food": "Bring fingertips to lips, tap twice",
    "hello": "Flat hand salute from forehead, move outward",
    "home": "Fingertips to chin, then to cheek",
    "hungry": "Curved hand on chest, move downward in C-shape",
    "no": "Index and middle finger tap thumb twice",
    "please": "Flat hand on chest, circular motion",
    "sick": "Middle finger to forehead, other to stomach",
    "sleep": "Open hand near face, close fingers downward",
    "thankyou": "Flat hand from chin, move forward",
    "water": "W handshape (3 fingers up), tap chin twice",
    "where": "Index finger wagging side to side",
    "yes": "Fist, nod up and down (wrist motion)",
}

SEQUENCE_LENGTH = 30
INPUT_FEATURES = 126  # 21 Left Hand (x,y,z) + 21 Right Hand (x,y,z)


def get_mediapipe_solutions():
    """Robustly retrieve mp.solutions or fallback import."""
    if hasattr(mp, "solutions"):
        return mp.solutions
    try:
        import importlib
        return importlib.import_module("mediapipe.python.solutions")
    except Exception:
        return None


def extract_frame_landmarks(results, image_width: int, image_height: int) -> Tuple[np.ndarray, bool]:
    """
    Extracts 126 normalized coordinates (21 left + 21 right hand landmarks x 3 coords).
    Returns (features_126_array, hands_present_bool).
    """
    features = np.zeros(INPUT_FEATURES, dtype=np.float32)
    hands_present = False

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return features, False

    hands_present = True
    for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label  # "Left" or "Right"
        
        # Google MediaPipe mirrors handedness in selfie mode unless flipped
        # Left Hand slots: 0..62 (indices 0..20 * 3)
        # Right Hand slots: 63..125 (indices 0..20 * 3 + 63)
        offset = 0 if label == "Left" else 63
        
        for idx, lm in enumerate(landmarks.landmark[:21]):
            slot = offset + (idx * 3)
            features[slot]     = lm.x
            features[slot + 1] = lm.y
            features[slot + 2] = lm.z

    return features, hands_present


def draw_ui_overlay(
    frame: np.ndarray,
    sign_name: str,
    instruction: str,
    rep_current: int,
    rep_total: int,
    state: str,
    countdown_val: int = 0,
    frames_recorded: int = 0,
    hands_present: bool = False,
) -> np.ndarray:
    """Renders informative instruction banner and recording status overlay."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top instruction panel
    panel_height = 110
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Sign name & Rep counter
    title_text = f"SIGN: {sign_name.upper()}"
    rep_text = f"Rep {rep_current} / {rep_total}"
    cv2.putText(frame, title_text, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, rep_text, (w - 200, 38), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    # ASL Instruction description
    cv2.putText(frame, f"How to sign: {instruction}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Hand detection status badge
    badge_color = (0, 220, 0) if hands_present else (0, 140, 255)
    badge_text = "Hands Detected" if hands_present else "No Hands Detected"
    cv2.circle(frame, (25, 96), 6, badge_color, -1)
    cv2.putText(frame, badge_text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, badge_color, 1, cv2.LINE_AA)

    # 2. Bottom status bar
    bot_height = 40
    cv2.rectangle(frame, (0, h - bot_height), (w, h), (15, 15, 15), -1)

    if state == "READY":
        hint = "Press [SPACE] to start countdown  |  [Q] Quit  |  [R] Redo previous"
        cv2.putText(frame, hint, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

    elif state == "COUNTDOWN":
        hint = "GET READY... Countdown in progress"
        cv2.putText(frame, hint, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        
        # Center countdown display
        count_str = str(countdown_val)
        (tw, th), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_DUPLEX, 3.5, 5)
        cx, cy = (w - tw) // 2, (h + th) // 2
        cv2.circle(frame, (w // 2, h // 2), 90, (0, 0, 0), -1)
        cv2.circle(frame, (w // 2, h // 2), 90, (0, 220, 255), 4)
        cv2.putText(frame, count_str, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 220, 255), 5, cv2.LINE_AA)

    elif state == "RECORDING":
        hint = f"PERFORM SIGN NOW -- Capturing {frames_recorded}/{SEQUENCE_LENGTH} frames"
        cv2.putText(frame, hint, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Top-right red recording indicator
        cv2.circle(frame, (w - 30, 96), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Progress bar across bottom of top panel
        progress_w = int((frames_recorded / SEQUENCE_LENGTH) * w)
        cv2.rectangle(frame, (0, panel_height - 4), (progress_w, panel_height), (0, 0, 255), -1)

    return frame


def record_sign_session(
    sign_name: str,
    instruction: str,
    num_reps: int,
    cap: cv2.VideoCapture,
    hands_detector,
    mp_draw,
    mp_hands,
    dest_dir: Path,
) -> bool:
    """
    Runs an interactive recording session for a single sign class.
    Returns True if completed, False if user requested quit.
    """
    sign_out_dir = dest_dir / sign_name
    sign_out_dir.mkdir(parents=True, exist_ok=True)

    # Determine starting rep index from existing files
    existing_reps = list(sign_out_dir.glob("rep_*.npz"))
    rep_idx = len(existing_reps) + 1
    total_reps_target = (rep_idx - 1) + num_reps

    log.info("Starting session for '%s' (%d repetitions target)", sign_name, num_reps)
    log.info("Instruction: %s", instruction)

    rep_counter = 1
    while rep_counter <= num_reps:
        current_rep_number = len(list(sign_out_dir.glob("rep_*.npz"))) + 1
        
        # -------------------------------------------------------------------
        # Phase 1: Ready / Preparation Screen
        # -------------------------------------------------------------------
        start_countdown = False
        while True:
            ret, frame = cap.read()
            if not ret:
                log.error("Failed to read from camera.")
                return False

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)

            # Draw MediaPipe hands overlay
            hands_present = bool(results.multi_hand_landmarks)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(255, 200, 0), thickness=2),
                    )

            draw_ui_overlay(
                frame,
                sign_name=sign_name,
                instruction=instruction,
                rep_current=current_rep_number,
                rep_total=total_reps_target,
                state="READY",
                hands_present=hands_present,
            )

            cv2.imshow("SAKSHAM Landmark Recorder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                start_countdown = True
                break
            elif key == ord("q"):
                log.info("User requested exit.")
                return False
            elif key == ord("r"):
                # Delete last rep if requested
                prev_reps = sorted(sign_out_dir.glob("rep_*.npz"))
                if prev_reps:
                    last_file = prev_reps[-1]
                    last_file.unlink(missing_ok=True)
                    log.info("Removed previous file: %s", last_file.name)
                    if rep_counter > 1:
                        rep_counter -= 1
                    current_rep_number = len(list(sign_out_dir.glob("rep_*.npz"))) + 1

        if not start_countdown:
            break

        # -------------------------------------------------------------------
        # Phase 2: 3-Second Visual Countdown
        # -------------------------------------------------------------------
        for count in range(3, 0, -1):
            t_end = time.time() + 1.0
            while time.time() < t_end:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(rgb_frame)

                hands_present = bool(results.multi_hand_landmarks)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
                            mp_draw.DrawingSpec(color=(255, 200, 0), thickness=2),
                        )

                draw_ui_overlay(
                    frame,
                    sign_name=sign_name,
                    instruction=instruction,
                    rep_current=current_rep_number,
                    rep_total=total_reps_target,
                    state="COUNTDOWN",
                    countdown_val=count,
                    hands_present=hands_present,
                )

                cv2.imshow("SAKSHAM Landmark Recorder", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    log.info("User aborted during countdown.")
                    return False

        # -------------------------------------------------------------------
        # Phase 3: Record 30 Frames of Landmarks
        # -------------------------------------------------------------------
        recorded_frames: List[np.ndarray] = []
        log.info("Capturing rep %d (%d frames)...", current_rep_number, SEQUENCE_LENGTH)

        while len(recorded_frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)

            features, hands_present = extract_frame_landmarks(results, w, h)
            recorded_frames.append(features)

            # Draw skeletal overlay
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 50, 255), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2),
                    )

            draw_ui_overlay(
                frame,
                sign_name=sign_name,
                instruction=instruction,
                rep_current=current_rep_number,
                rep_total=total_reps_target,
                state="RECORDING",
                frames_recorded=len(recorded_frames),
                hands_present=hands_present,
            )

            cv2.imshow("SAKSHAM Landmark Recorder", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                log.info("User aborted recording.")
                return False

        # Format into (30, 126) float32 numpy array
        features_array = np.array(recorded_frames, dtype=np.float32)  # Shape (30, 126)

        out_path = sign_out_dir / f"rep_{current_rep_number:03d}.npz"
        np.savez_compressed(
            out_path,
            features=features_array,
            isometric_landmarks=features_array,
        )
        log.info("Saved: %s  (shape: %s)", out_path.name, features_array.shape)

        # Brief confirmation flash
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                f"SAVED REP {current_rep_number}!",
                (w // 2 - 160, h // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("SAKSHAM Landmark Recorder", frame)
            cv2.waitKey(350)

        rep_counter += 1

    log.info("Successfully recorded all %d reps for '%s'!", num_reps, sign_name)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="SAKSHAM Landmark Data Recorder: Capture personal MediaPipe 30x126 landmark sequences"
    )
    parser.add_argument(
        "--sign_name",
        type=str,
        default=None,
        help="Specific sign to record (e.g. food, hello, water).",
    )
    parser.add_argument(
        "--num_reps",
        type=int,
        default=15,
        help="Number of repetitions to record per sign (default: 15).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sequentially record all 14 target signs.",
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="dataset/processed/personal",
        help="Output destination directory (default: dataset/processed/personal).",
    )
    args = parser.parse_args()

    # Determine signs to record
    if args.all:
        target_signs = sorted(list(SIGN_INSTRUCTIONS.keys()))
    elif args.sign_name:
        sign = args.sign_name.strip().lower()
        if sign not in SIGN_INSTRUCTIONS:
            log.error("Unknown sign '%s'. Available signs: %s", sign, list(SIGN_INSTRUCTIONS.keys()))
            sys.exit(1)
        target_signs = [sign]
    else:
        log.info("No sign specified. Use --sign_name <name> or --all.")
        print("\nAvailable 14 Signs:")
        for k, v in SIGN_INSTRUCTIONS.items():
            print(f"  - {k:<10}: {v}")
        print("\nExample: .venv\\Scripts\\python.exe tools/record_landmarks.py --sign_name food --num_reps 15\n")
        sys.exit(0)

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MediaPipe Hands
    mp_sol = get_mediapipe_solutions()
    if mp_sol is None or not hasattr(mp_sol, "hands"):
        log.error("MediaPipe solutions module could not be loaded.")
        sys.exit(1)

    mp_hands = mp_sol.hands
    mp_draw = mp_sol.drawing_utils

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        log.error("Could not open camera device %d.", args.camera_id)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    log.info("=" * 65)
    log.info("SAKSHAM LANDMARK RECORDER INITIALIZED")
    log.info("Target signs: %s", target_signs)
    log.info("Reps per sign: %d", args.num_reps)
    log.info("Output directory: %s", dest_dir.resolve())
    log.info("=" * 65)

    try:
        for idx, sign in enumerate(target_signs):
            instruction = SIGN_INSTRUCTIONS[sign]
            log.info("\n[%d/%d] Preparing to record sign: %s", idx + 1, len(target_signs), sign.upper())
            success = record_sign_session(
                sign_name=sign,
                instruction=instruction,
                num_reps=args.num_reps,
                cap=cap,
                hands_detector=hands_detector,
                mp_draw=mp_draw,
                mp_hands=mp_hands,
                dest_dir=dest_dir,
            )
            if not success:
                log.info("Session stopped by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_detector.close()
        log.info("Camera released. Exiting cleanly.")


if __name__ == "__main__":
    main()
