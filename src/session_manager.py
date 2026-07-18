import time
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import logging

DEFAULT_DATASET_LABELS = [
    "hello",
    "goodbye",
    "thank_you",
    "please",
    "sorry",
    "yes",
    "no",
    "okay",
    "understand",
    "dont_understand",
    "help",
    "stop",
    "doctor",
    "hospital",
    "call",
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
    "name",
    "pain",
    "hungry",
    "friend",
    "home",
]

DEFAULT_SIGNER_IDS = [f"SIGNER_{idx:02d}" for idx in range(1, 11)]

class ConversationSession:
    """
    Manages the state and history of a SAKSHAM communication session.
    """
    def __init__(self):
        self.messages: List[Dict] = []
        self.start_time = time.time()
        self.draft_message = ""
        self.mic_enabled = False
        self.mic_state = "Mic Off"
        self.toast_message: Optional[str] = None
        self.toast_expiry: float = 0.0
        self.last_export_path = ""
        self.last_export_status = ""  # "", "Success", "Failed"
        self.last_dataset_export_path = ""
        self.last_dataset_export_status = ""  # "", "Success", "Failed"
        self.typing_buffer = ""
        self.is_typing_focused = False
        self.request_mic_toggle = False
        self.last_error = ""

        # Dataset collection state
        self.dataset_mode = False
        self.dataset_labels = list(DEFAULT_DATASET_LABELS)
        self.dataset_label_index = 0
        self.dataset_signer_ids = list(DEFAULT_SIGNER_IDS)
        self.dataset_signer_index = 0
        self.dataset_status = "Idle"
        self.dataset_review_summary: Optional[Dict] = None
        self.dataset_clip_counts: Dict[str, int] = {}
        self.dataset_insights: Dict = {}
        self.dataset_live_feedback: Optional[Dict] = None
        self.live_hand_visibility = {
            "left_visible": False,
            "right_visible": False,
            "hands_present": 0,
        }

        # Live sign prediction state (from ONNX inference)
        self.live_prediction: Optional[str] = None
        self.live_confidence: float = 0.0
        self.live_top3: list[tuple[str, float]] = []
        self.buffer_progress: int = 0
        self.buffer_capacity: int = 64
        
        # Ensure export directory exists
        os.makedirs("exports", exist_ok=True)

    def add_message(self, sender: str, text: str, source: str = "System"):
        if not text.strip():
            return
            
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        self.messages.append({
            "sender": sender,
            "text": text,
            "time": time_str,
            "source": source
        })
        logging.info(f"[{time_str}] {sender} ({source}): {text}")
        
        # Clear draft when a final message is committed
        self.draft_message = ""

    def clear(self):
        """Clears the current conversation but keeps the session start time."""
        self.messages.clear()
        self.draft_message = ""
        self.set_toast("Conversation Cleared")
        logging.info("Conversation cleared.")

    def new_session(self):
        """Resets the entire session."""
        self.messages.clear()
        self.draft_message = ""
        self.start_time = time.time()
        self.set_toast("New Session Started")
        logging.info("New session started.")

    def get_duration_minutes(self) -> int:
        return int((time.time() - self.start_time) / 60)
        
    def get_message_count(self) -> int:
        return len(self.messages)

    def set_draft(self, text: str):
        self.draft_message = text

    def set_toast(self, message: str, duration: float = 3.0):
        self.toast_message = message
        self.toast_expiry = time.time() + duration

    @property
    def current_dataset_label(self) -> str:
        return self.dataset_labels[self.dataset_label_index]

    @property
    def current_signer_id(self) -> str:
        return self.dataset_signer_ids[self.dataset_signer_index]

    def cycle_dataset_label(self, delta: int):
        if not self.dataset_labels:
            return
        self.dataset_label_index = (self.dataset_label_index + delta) % len(self.dataset_labels)
        self.set_toast(f"Dataset Label: {self.current_dataset_label}")

    def cycle_dataset_signer(self, delta: int):
        if not self.dataset_signer_ids:
            return
        self.dataset_signer_index = (self.dataset_signer_index + delta) % len(self.dataset_signer_ids)
        self.set_toast(f"Signer: {self.current_signer_id}")

    def set_dataset_mode(self, enabled: bool):
        self.dataset_mode = enabled
        self.dataset_status = "Idle"
        self.dataset_review_summary = None
        self.dataset_live_feedback = None
        mode = "ON" if enabled else "OFF"
        self.set_toast(f"Dataset Mode: {mode}")

    def set_dataset_status(self, status: str):
        self.dataset_status = status

    def set_dataset_review_summary(self, summary: Optional[Dict]):
        self.dataset_review_summary = summary

    def set_dataset_insights(self, insights: Dict):
        self.dataset_insights = insights or {}
        statistics = self.dataset_insights.get("statistics", {})
        self.dataset_clip_counts = dict(statistics.get("clips_per_label", {}))

    def set_dataset_live_feedback(self, feedback: Optional[Dict]):
        self.dataset_live_feedback = feedback

    def set_live_hand_visibility(self, hand_data: Optional[Dict]):
        if not hand_data:
            self.live_hand_visibility = {
                "left_visible": False,
                "right_visible": False,
                "hands_present": 0,
            }
            return

        left_visible = bool(hand_data.get("left_hand", {}).get("present"))
        right_visible = bool(hand_data.get("right_hand", {}).get("present"))
        self.live_hand_visibility = {
            "left_visible": left_visible,
            "right_visible": right_visible,
            "hands_present": int(left_visible) + int(right_visible),
        }

    def mark_dataset_clip_saved(self, label: str):
        self.dataset_clip_counts[label] = self.dataset_clip_counts.get(label, 0) + 1

    def get_dataset_clip_count(self, label: str) -> int:
        statistics = self.dataset_insights.get("statistics", {})
        return statistics.get("clips_per_label", {}).get(label, self.dataset_clip_counts.get(label, 0))

    def set_dataset_export_result(self, export_base: str, success: bool):
        self.last_dataset_export_path = export_base
        self.last_dataset_export_status = "Success" if success else "Failed"

    def export(self) -> bool:
        """Exports the conversation to TXT and JSON in the exports/ directory."""
        if not self.messages:
            self.last_export_status = "Failed"
            self.set_toast("\u2717 Export Failed: Empty Chat")
            logging.warning("Export requested but conversation is empty.")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = f"exports/chat_{timestamp}.txt"
        json_path = f"exports/chat_{timestamp}.json"
        
        try:
            # TXT Export
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"SAKSHAM V1 Conversation Export\n\n")
                for msg in self.messages:
                    f.write(f"[{msg['time']}] {msg['sender']}\n{msg['text']}\n\n")

            # JSON Export
            export_data = {
                "session_start": datetime.fromtimestamp(self.start_time).isoformat(),
                "session_duration_minutes": self.get_duration_minutes(),
                "message_count": self.get_message_count(),
                "messages": self.messages
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4)
                
            # Verify Export Success
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 and \
               os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                self.last_export_path = f"chat_{timestamp}"
                self.last_export_status = "Success"
                self.set_toast(f"\u2713 Exported: chat_{timestamp} ({self.get_message_count()} msgs)", duration=4.0)
                logging.info(f"Session exported successfully to {txt_path} and {json_path}")
                return True
            else:
                self.last_export_status = "Failed"
                self.set_toast("\u2717 Export Failed: Verification Error")
                return False
                
        except Exception as e:
            self.last_export_status = "Failed"
            self.set_toast("\u2717 Export Failed: System Error")
            logging.error(f"Export failed with exception: {e}")
            return False
