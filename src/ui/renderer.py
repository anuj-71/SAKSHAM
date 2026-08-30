import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import src.config.settings as config
from src.session_manager import ConversationSession
from PIL import Image, ImageDraw, ImageFont
import os
import textwrap

class UIRenderer:
    """
    SAKSHAM V1.5 User Interface.
    Utilizes Pillow (PIL) for high-quality text rendering and chat bubbles.

    Layout (top to bottom):
        Header          (60px)
        Conversation    (fills remaining - bottom_section)
        ────────────────────────────────────────────
        Bottom section  (200px):
            Left:  Live Transcript panel  (65% width)
            Right: Camera preview (small) + Session Info
        ────────────────────────────────────────────
        Status bar      (36px)
    """
    def __init__(self, width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT):
        self.width = width
        self.height = height

        # ── Layout constants ──────────────────────────────────────────
        self.header_h = 56
        self.bottom_bar_h = 80  # Space reserved for bottom floating pill
        self.bottom_h = 240

        self.conv_y = self.header_h
        self.bottom_y = self.height - self.bottom_bar_h - self.bottom_h
        self.conv_h = self.bottom_y - self.conv_y

        # Bottom section split: 60% conversation/empty, 40% camera + stats
        self.live_panel_w = int(self.width * 0.60)
        self.right_panel_x = self.live_panel_w

        # Camera preview – fills right side of the bottom panel
        self.cam_h = 200
        self.cam_w = int(self.cam_h * 16 / 9)  # ~355px

        # Chat bubble max width: 80% of conversation area width
        self.max_bubble_w_ratio = 0.80

        # ── Fonts (Windows Defaults) ──────────────────────────────────
        font_path = "C:/Windows/Fonts/segoeui.ttf"
        font_bold_path = "C:/Windows/Fonts/segoeuib.ttf"

        if not os.path.exists(font_path):
            font_path = "arial.ttf"
            font_bold_path = "arialbd.ttf"

        try:
            self.font_header = ImageFont.truetype(font_bold_path, 22)
            self.font_subheader = ImageFont.truetype(font_path, 13)
            self.font_body = ImageFont.truetype(font_path, 17)
            self.font_body_bold = ImageFont.truetype(font_bold_path, 17)
            self.font_small = ImageFont.truetype(font_path, 13)
            self.font_small_bold = ImageFont.truetype(font_bold_path, 13)
            self.font_live = ImageFont.truetype(font_path, 18)
            self.font_live_state = ImageFont.truetype(font_bold_path, 15)
        except IOError:
            fallback = ImageFont.load_default()
            self.font_header = fallback
            self.font_subheader = fallback
            self.font_body = fallback
            self.font_body_bold = fallback
            self.font_small = fallback
            self.font_small_bold = fallback
            self.font_live = fallback
            self.font_live_state = fallback

    # ──────────────────────────────────────────────────────────────────
    #  Main render
    # ──────────────────────────────────────────────────────────────────
    def render(self,
               camera_frame: Optional[cv2.Mat],
               session: ConversationSession,
               fps: float,
               scroll_offset: int = 0,
               hand_data: Optional[Dict] = None) -> cv2.Mat:
        """Renders the SAKSHAM UI base canvas and returns a BGR numpy frame."""

        # 1. Base canvas
        ui_frame = np.full((self.height, self.width, 3), config.COLOR_BG, dtype=np.uint8)

        # Header background
        cv2.rectangle(ui_frame, (0, 0), (self.width, self.header_h), (20, 20, 20), -1)
        cv2.line(ui_frame, (0, self.header_h), (self.width, self.header_h), config.COLOR_DIVIDER, 1)

        # Bottom section divider (above camera & stats)
        cv2.line(ui_frame, (0, self.bottom_y), (self.width, self.bottom_y), config.COLOR_DIVIDER, 1)

        # Camera preview (right panel)
        cam_x = self.right_panel_x + 20
        cam_y = self.bottom_y + 12
        if camera_frame is not None:
            # 1. Draw MediaPipe hand skeleton directly on camera_frame (full size)
            if hand_data:
                left_hand = hand_data.get("left_hand", {})
                right_hand = hand_data.get("right_hand", {})
                if left_hand.get("present") and left_hand.get("pixel_landmarks"):
                    self._draw_hand_skeleton_on_feed(camera_frame, left_hand["pixel_landmarks"], line_color=(255, 0, 255), point_color=(0, 255, 255))
                if right_hand.get("present") and right_hand.get("pixel_landmarks"):
                    self._draw_hand_skeleton_on_feed(camera_frame, right_hand["pixel_landmarks"], line_color=(0, 255, 0), point_color=(0, 255, 255))

            # 2. Resize and composite camera frame into UI
            cam_resized = cv2.resize(camera_frame, (self.cam_w, self.cam_h))
            ui_frame[cam_y:cam_y + self.cam_h, cam_x:cam_x + self.cam_w] = cam_resized
            cv2.rectangle(ui_frame, (cam_x, cam_y),
                          (cam_x + self.cam_w, cam_y + self.cam_h), config.COLOR_DIVIDER, 1)

            # 3. Draw live debug HUD overlay on top of camera preview box
            raw_top3 = getattr(session, "raw_top3", [])
            if raw_top3 and len(raw_top3) > 0:
                top_label, top_score = raw_top3[0]
                cv2.rectangle(ui_frame, (cam_x + 5, cam_y + 5), (cam_x + self.cam_w - 5, cam_y + 30), (0, 0, 0), -1)
                cv2.putText(ui_frame, f"RAW: {top_label} ({top_score:.0%})", (cam_x + 10, cam_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. PIL overlay for all text rendering
        self._last_session_dataset_mode = session.dataset_mode
        self._last_session_dataset_label = session.current_dataset_label
        img_rgb = cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        self._draw_header(draw, fps, session.mic_state, camera_frame is not None)
        self._draw_conversation(draw, session, scroll_offset)
        self._draw_session_info(draw, session, cam_x, cam_y)
        self._draw_dataset_panel(draw, session, cam_x, cam_y + self.cam_h + 10)
        self._draw_toast(draw, session)

        # Convert back to BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ──────────────────────────────────────────────────────────────────
    #  Header (with clean status chips and shortcuts)
    # ──────────────────────────────────────────────────────────────────
    def _draw_header(self, draw: ImageDraw.ImageDraw, fps: float, mic_state: str, cam_ok: bool):
        draw.text((24, 12), "SAKSHAM V1",
                  font=self.font_header, fill=self._bgr_to_rgb(config.COLOR_ACCENT))
        draw.text((24, 38), "AI Communication Assistant",
                  font=self.font_subheader, fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))

        # Status indicator chips in header
        status_x = 240
        cam_col = config.COLOR_SUCCESS if cam_ok else config.COLOR_WARNING
        draw.text((status_x, 22), f"Camera: {'OK' if cam_ok else 'OFF'}",
                  font=self.font_small, fill=self._bgr_to_rgb(cam_col))

        ready_text = f"Dataset: {self._last_session_dataset_label}" if getattr(self, "_last_session_dataset_mode", False) else "System Ready"
        draw.text((status_x + 110, 22), ready_text,
                  font=self.font_small, fill=self._bgr_to_rgb(config.COLOR_TEXT_PRIMARY))

        if config.DEV_MODE:
            draw.text((status_x + 230, 22), f"FPS: {fps:.1f}",
                      font=self.font_small, fill=self._bgr_to_rgb(config.COLOR_WARNING))

        hint = "[M] Mic  [F] Fullscreen  [K] Dataset  [J/L] Label  [U/I] Signer  [R] Rec  [A/X] Review  [E] Export"
        if not getattr(self, "_last_session_dataset_mode", False):
            hint = "[M] Mic  [F] Fullscreen  [C] Clear  [E] Export  [O] Exports  [D] Dev Mode"
        _, _, w, _ = draw.textbbox((0, 0), hint, font=self.font_small)
        draw.text((self.width - w - 24, 22), hint,
                  font=self.font_small, fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))

    # ──────────────────────────────────────────────────────────────────
    #  Conversation panel  (chat bubbles, scrollable)
    # ──────────────────────────────────────────────────────────────────
    def _draw_conversation(self, draw: ImageDraw.ImageDraw,
                           session: ConversationSession, scroll_offset: int):
        padding_x = 24
        base_y = self.bottom_y - 12          # bottom edge with margin
        avail_w = self.width - (padding_x * 2)
        max_bubble_w = int(avail_w * self.max_bubble_w_ratio)
        # How many chars fit – rough estimate for textwrap
        chars_per_line = max(40, int(max_bubble_w / 10))

        bubble_pad = 10
        bubble_gap = 4

        # Build layout bottom-up (newest at bottom)
        rendered: List[dict] = []
        for msg in reversed(session.messages):
            source_tag = f" ({msg.get('source', '')})" if config.DEV_MODE and msg.get('source') else ""
            header = f"[{msg['time']}]  {msg['sender']}{source_tag}"
            wrapped = textwrap.fill(msg['text'], width=chars_per_line)
            _, _, hw, hh = draw.textbbox((0, 0), header, font=self.font_small)
            _, _, tw, th = draw.textbbox((0, 0), wrapped, font=self.font_body)
            bw = min(max(hw, tw) + bubble_pad * 2, max_bubble_w)
            bh = hh + th + bubble_pad * 2 + 4
            rendered.append({
                "header": header, "wrapped": wrapped, "sender": msg["sender"],
                "w": bw, "h": bh
            })

        # Apply scroll (skip N newest messages)
        visible = rendered[scroll_offset:] if scroll_offset < len(rendered) else []

        cur_y = base_y
        for b in visible:
            cur_y -= b["h"]
            if cur_y < self.conv_y + 4:
                break

            # Alignment and styling based on sender
            if b["sender"] == "Deaf User":
                bx = self.width - padding_x - b["w"]
                bg_color = (40, 50, 65, 255)       # slight bluish tint
                border_color = (60, 80, 100, 255)
            else:
                bx = padding_x
                bg_color = (42, 42, 48, 255)       # default gray
                border_color = (70, 70, 75, 255)

            # Bubble background
            draw.rounded_rectangle(
                [bx, cur_y, bx + b["w"], cur_y + b["h"]],
                radius=8,
                fill=bg_color,
                outline=border_color
            )
            # Header (timestamp + sender)
            draw.text((bx + bubble_pad, cur_y + bubble_pad),
                      b["header"], font=self.font_small,
                      fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))
            _, _, _, hh = draw.textbbox((0, 0), b["header"], font=self.font_small)
            # Body
            draw.text((bx + bubble_pad, cur_y + bubble_pad + hh + 4),
                      b["wrapped"], font=self.font_body,
                      fill=self._bgr_to_rgb(config.COLOR_TEXT_PRIMARY))

            cur_y -= bubble_gap

        # Empty-state placeholder
        if not session.messages:
            hint = "No messages yet – start speaking to begin a conversation."
            _, _, tw, _ = draw.textbbox((0, 0), hint, font=self.font_body)
            cx = (self.width - tw) // 2
            cy = self.conv_y + self.conv_h // 2
            draw.text((cx, cy), hint, font=self.font_body,
                      fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))

    # ──────────────────────────────────────────────────────────────────
    #  Live Transcript panel  (bottom-left, dedicated area)
    # ──────────────────────────────────────────────────────────────────
    # (Removed) _draw_live_transcript — replaced by floating LiveTranscriptWidget

    # ──────────────────────────────────────────────────────────────────
    #  Session Info panel  (bottom-right, below camera)
    # ──────────────────────────────────────────────────────────────────
    def _draw_session_info(self, draw: ImageDraw.ImageDraw,
                           session: ConversationSession,
                           cam_x: int, cam_y: int):
        if not config.DEV_MODE:
            return
            
        sx = cam_x + self.cam_w + 24
        sy = self.bottom_y + 15

        draw.text((sx, sy), "Current Session",
                  font=self.font_small_bold,
                  fill=self._bgr_to_rgb(config.COLOR_TEXT_PRIMARY))
        sy += 24

        info_lines = [
            f"Messages:  {session.get_message_count()}",
            f"Duration:  {session.get_duration_minutes()} min",
        ]

        # Export status
        if session.last_export_status == "Success" and session.last_export_path:
            info_lines.append(f"Last Export:  \u2713 {session.last_export_path}")
        elif session.last_export_status == "Failed":
            info_lines.append("Last Export:  \u2717 Failed")
        else:
            info_lines.append("Last Export:  \u2014")

        if session.last_dataset_export_status == "Success" and session.last_dataset_export_path:
            info_lines.append(f"Dataset Export:  \u2713 {os.path.basename(session.last_dataset_export_path)}")
        elif session.last_dataset_export_status == "Failed":
            info_lines.append("Dataset Export:  \u2717 Failed")
        else:
            info_lines.append("Dataset Export:  \u2014")

        for line in info_lines:
            draw.text((sx, sy), line, font=self.font_small,
                      fill=self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY))
            sy += 22

    def _draw_dataset_panel(self, draw: ImageDraw.ImageDraw,
                            session: ConversationSession,
                            cam_x: int, panel_y: int):
        if not session.dataset_mode and not session.dataset_review_summary:
            return

        panel_x = cam_x
        draw.text((panel_x, panel_y), "Dataset Collection",
                  font=self.font_small_bold,
                  fill=self._bgr_to_rgb(config.COLOR_TEXT_PRIMARY))
        panel_y += 18

        insights = getattr(session, "dataset_insights", {})
        statistics = insights.get("statistics", {})
        audit = insights.get("audit", {})
        two_hand = insights.get("two_hand_verification", {})
        visibility_metrics = statistics.get("hand_visibility_metrics", {})
        class_balance = statistics.get("class_balance", {})
        targets = statistics.get("targets", {})
        progress = statistics.get("progress", {})

        lines = [
            f"Label: {session.current_dataset_label}  ({session.get_dataset_clip_count(session.current_dataset_label)} saved)",
            f"Signer: {session.current_signer_id}",
            f"State: {session.dataset_status}",
        ]
        if session.dataset_review_summary:
            review = session.dataset_review_summary
            lines.append(
                f"Frames: {review['frame_count']} | L {review['left_presence_ratio']:.0%} | R {review['right_presence_ratio']:.0%}"
            )
            lines.append(
                f"Active: {review['active_hand_ratio']:.0%} | Both: {review['both_hands_ratio']:.0%}"
            )
            if review["passes_quality_checks"]:
                lines.append("Quality: PASS")
            else:
                lines.append("Quality: BLOCKED")
            if review["quality_blockers"]:
                lines.append(f"Blocker: {review['quality_blockers'][0]}")
            elif review["quality_warnings"]:
                lines.append(f"Warning: {review['quality_warnings'][0]}")
        else:
            lines.append("Review keys: [A] Accept  [X] Reject  [R] Re-record  [E] Export")

        if statistics:
            lines.extend([
                (
                    f"Dataset Stats: {statistics.get('total_clips', 0)}/{targets.get('target_dataset_size', 0)} clips | "
                    f"{statistics.get('unique_signer_count', 0)}/{targets.get('minimum_signers', 0)} signers"
                ),
                (
                    f"Frames Avg: {statistics.get('average_frame_count', 0.0):.1f} | "
                    f"Active Avg: {visibility_metrics.get('active_hand_avg', 0.0):.0%} | "
                    f"Both Avg: {visibility_metrics.get('both_hands_avg', 0.0):.0%}"
                ),
                (
                    f"Class Balance: min {class_balance.get('min_label') or '-'}={class_balance.get('min_count', 0)} | "
                    f"max {class_balance.get('max_label') or '-'}={class_balance.get('max_count', 0)} | "
                    f"ratio {class_balance.get('imbalance_ratio', 0.0):.2f}x"
                ),
                (
                    "Per Label: "
                    + ", ".join(
                        f"{label}:{count}"
                        for label, count in statistics.get("clips_per_label", {}).items()
                    )
                ),
                (
                    "Per Signer: "
                    + ", ".join(
                        f"{signer}:{count}"
                        for signer, count in statistics.get("clips_per_signer", {}).items()
                    ) if statistics.get("clips_per_signer") else "Per Signer: none yet"
                ),
                (
                    f"Quality Audit: {audit.get('verified_samples', 0)}/{audit.get('total_samples', 0)} verified | "
                    f"Warnings {audit.get('samples_with_warnings', 0)} | "
                    f"Issues {len(audit.get('issues', []))}"
                ),
                (
                    f"Two-Hand: {'VERIFIED' if two_hand.get('two_hand_capture_verified') else 'PENDING'} | "
                    f"Both-hand clips {two_hand.get('samples_with_both_hands', 0)}"
                ),
                (
                    f"Progress Remaining: {progress.get('clips_remaining', 0)} clips | "
                    f"Missing labels {len(class_balance.get('missing_labels', []))}"
                ),
            ])

        if session.last_dataset_export_status == "Success" and session.last_dataset_export_path:
            lines.append(f"Exported: {os.path.basename(session.last_dataset_export_path)}")

        panel_width = self.width - panel_x - 24
        for line in lines:
            panel_y = self._draw_wrapped_line(
                draw,
                line,
                panel_x,
                panel_y,
                panel_width,
                self.font_small,
                self._bgr_to_rgb(config.COLOR_TEXT_SECONDARY),
            )

    # ──────────────────────────────────────────────────────────────────
    #  Toast notification
    # ──────────────────────────────────────────────────────────────────
    def _draw_toast(self, draw: ImageDraw.ImageDraw, session: ConversationSession):
        if session.toast_message and time.time() < session.toast_expiry:
            msg = session.toast_message
            _, _, tw, th = draw.textbbox((0, 0), msg, font=self.font_body_bold)
            tx = (self.width - tw) // 2
            ty = self.header_h + 16
            draw.rounded_rectangle(
                [tx - 14, ty - 8, tx + tw + 14, ty + th + 8],
                radius=8, fill=(40, 160, 40, 220)
            )
            draw.text((tx, ty), msg, font=self.font_body_bold, fill=(255, 255, 255, 255))

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index
        (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]

    # ──────────────────────────────────────────────────────────────────
    #  Hand landmarks & skeleton overlay (OpenCV)
    # ──────────────────────────────────────────────────────────────────
    def _draw_hand_skeleton_on_feed(self, frame: cv2.Mat,
                                   pixel_landmarks: List[Tuple[int, int]],
                                   line_color: Tuple[int, int, int],
                                   point_color: Tuple[int, int, int]):
        if not pixel_landmarks or len(pixel_landmarks) < 21:
            return
        # Draw skeleton connections
        for p1_idx, p2_idx in self.HAND_CONNECTIONS:
            pt1 = pixel_landmarks[p1_idx]
            pt2 = pixel_landmarks[p2_idx]
            cv2.line(frame, pt1, pt2, line_color, 2, cv2.LINE_AA)
        # Draw landmark points
        for idx, pt in enumerate(pixel_landmarks):
            radius = 5 if idx in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, pt, radius, point_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_dev_landmarks(self, frame: cv2.Mat,
                            pixel_landmarks: List[Tuple[int, int]],
                            cx: int, cy: int, color: Tuple[int, int, int]):
        scale_x = self.cam_w / config.FRAME_WIDTH
        scale_y = self.cam_h / config.FRAME_HEIGHT
        for pt in pixel_landmarks:
            px = cx + int(pt[0] * scale_x)
            py = cy + int(pt[1] * scale_y)
            cv2.circle(frame, (px, py), 2, color, -1)

    # ──────────────────────────────────────────────────────────────────
    #  Utility
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _bgr_to_rgb(bgr_tuple: tuple) -> tuple:
        """Convert a BGR colour (used in config) to RGB for PIL."""
        return bgr_tuple[::-1]

    def _draw_wrapped_line(self,
                           draw: ImageDraw.ImageDraw,
                           text: str,
                           x: int,
                           y: int,
                           max_width: int,
                           font: ImageFont.ImageFont,
                           fill: tuple,
                           line_gap: int = 2) -> int:
        chars_per_line = max(18, int(max_width / 7))
        wrapped_lines = textwrap.wrap(text, width=chars_per_line) or [text]
        for line in wrapped_lines:
            draw.text((x, y), line, font=font, fill=fill)
            _, _, _, height = draw.textbbox((0, 0), line, font=font)
            y += height + line_gap
        return y
