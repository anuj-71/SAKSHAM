"""Smoke test: create a UIRenderer, render a frame with some messages, save to disk."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.ui.renderer import UIRenderer
from src.session_manager import ConversationSession


def test_placeholder():
    """Renderer smoke test; see __main__ block below."""
    pass


def run_renderer_smoke_test():
    ui = UIRenderer(width=1280, height=720)
    session = ConversationSession()

    # Add some test messages
    session.add_message("Hearing Person", "Hello, how are you?")
    session.add_message("Hearing Person", "I wanted to ask about the project timeline and when we can expect the first deliverable to be ready for review.")
    session.add_message("Hearing Person", "That sounds great, thank you!")

    # Simulate a listening state
    session.mic_state = "Listening"
    session.set_draft("I am doing well thank you for...")

    # Fake camera frame
    fake_cam = np.full((720, 1280, 3), (80, 60, 40), dtype=np.uint8)
    cv2.putText(fake_cam, "CAMERA", (540, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)

    result = ui.render(camera_frame=fake_cam, session=session, fps=29.5, scroll_offset=0)

    cv2.imwrite("tests/ui_test_output.png", result)
    print(f"Rendered frame saved: tests/ui_test_output.png  shape={result.shape}")


if __name__ == "__main__":
    run_renderer_smoke_test()
