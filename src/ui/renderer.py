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
    SAKSHAM V1 UI Renderer.
    High-contrast accessible interface for deaf/mute and hearing communication.
    Supports native High-DPI and Full-HD resolution scaling with razor-sharp rendering.
    """

    def __init__(self, width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT):
        self.width = width
        self.height = height

        # High-DPI Resolution Scaling Factor (1.0 at 720p, 1.5 at 1080p, 2.0 at 1440p)
        self.scale = max(1.0, min(self.width / 1280.0, self.height / 720.0))

        # Layout Geometry
        self.top_bar_h = int(52 * self.scale)
        self.bottom_bar_h = int(64 * self.scale)
        self.main_y = self.top_bar_h
        self.main_h = self.height - self.top_bar_h - self.bottom_bar_h

        # Left Panel: 45% of total width
        self.cam_w = int(self.width * 0.45)
        self.cam_h = self.main_h
        self.cam_x = 0
        self.cam_y = self.main_y

        # Camera scaling properties for aspect-ratio preservation
        self._cam_scale = 1.0
        self._cam_pad_x = 0
        self._cam_pad_y = 0
        self._cam_new_w = self.cam_w
        self._cam_new_h = self.cam_h

        # Right Panel: 55% of total width
        self.chat_x = self.cam_w
        self.chat_y = self.main_y
        self.chat_w = self.width - self.cam_w
        self.chat_h = self.main_h
        self.chat_header_h = int(48 * self.scale)
        self.col_w = self.chat_w // 2

        # Hit-test rects for top bar interactive controls
        self.shortcut_rects: Dict[str, Tuple[int, int, int, int]] = {}

        # Fonts (Windows Segoe UI with resolution-aware point sizes)
        font_path = "C:/Windows/Fonts/segoeui.ttf"
        font_bold_path = "C:/Windows/Fonts/segoeuib.ttf"
        font_semi_path = "C:/Windows/Fonts/segoeuisl.ttf"

        if not os.path.exists(font_path):
            font_path = "arial.ttf"
            font_bold_path = "arialbd.ttf"
            font_semi_path = "arial.ttf"

        try:
            self.font_brand_title = ImageFont.truetype(font_bold_path, max(12, int(20 * self.scale)))
            self.font_brand_sub = ImageFont.truetype(font_path, max(9, int(11 * self.scale)))
            self.font_status = ImageFont.truetype(font_path, max(10, int(13 * self.scale)))
            self.font_shortcut = ImageFont.truetype(font_bold_path, max(10, int(12 * self.scale)))
            self.font_col_header_left = ImageFont.truetype(font_path, max(11, int(15 * self.scale)))
            self.font_col_header_right = ImageFont.truetype(font_bold_path, max(11, int(15 * self.scale)))
            self.font_badge_live = ImageFont.truetype(font_bold_path, max(10, int(12 * self.scale)))
            self.font_badge_sign = ImageFont.truetype(font_bold_path, max(11, int(14 * self.scale)))
            self.font_badge_conf = ImageFont.truetype(font_bold_path, max(11, int(14 * self.scale)))
            self.font_chat_body = ImageFont.truetype(font_path, max(12, int(15 * self.scale)))
            self.font_placeholder = ImageFont.truetype(font_path, max(11, int(14 * self.scale)))
            self.font_timestamp = ImageFont.truetype(font_path, max(8, int(10 * self.scale)))
            self.font_toast = ImageFont.truetype(font_bold_path, max(11, int(14 * self.scale)))
        except IOError:
            fallback = ImageFont.load_default()
            self.font_brand_title = fallback
            self.font_brand_sub = fallback
            self.font_status = fallback
            self.font_shortcut = fallback
            self.font_col_header_left = fallback
            self.font_col_header_right = fallback
            self.font_badge_live = fallback
            self.font_badge_sign = fallback
            self.font_badge_conf = fallback
            self.font_chat_body = fallback
            self.font_placeholder = fallback
            self.font_timestamp = fallback
            self.font_toast = fallback

    def render(self,
               camera_frame: Optional[cv2.Mat],
               session: ConversationSession,
               fps: float,
               scroll_offset: int = 0,
               hand_data: Optional[Dict] = None) -> cv2.Mat:
        """Renders the SAKSHAM base interface canvas and returns a BGR image."""

        # 1. Base Dark Canvas (#080914 -> BGR: 20, 9, 8)
        ui_frame = np.full((self.height, self.width, 3), (20, 9, 8), dtype=np.uint8)

        # 2. Right Panel Background (#0A0C15 -> BGR: 21, 12, 10)
        cv2.rectangle(
            ui_frame,
            (self.chat_x, self.main_y),
            (self.width, self.main_y + self.main_h),
            (21, 12, 10),
            -1
        )

        # Right Panel Header Row Background (#080914 -> BGR: 20, 9, 8)
        cv2.rectangle(
            ui_frame,
            (self.chat_x, self.main_y),
            (self.width, self.main_y + self.chat_header_h),
            (20, 9, 8),
            -1
        )

        # Right Panel Header Bottom Border (#1a1a3a -> BGR: 58, 26, 26)
        cv2.line(
            ui_frame,
            (self.chat_x, self.main_y + self.chat_header_h),
            (self.width, self.main_y + self.chat_header_h),
            (58, 26, 26),
            max(1, int(1 * self.scale))
        )

        # Vertical Column Divider Line (#1a1a3a -> BGR: 58, 26, 26) between chat columns
        cv2.line(
            ui_frame,
            (self.chat_x + self.col_w, self.main_y),
            (self.chat_x + self.col_w, self.main_y + self.main_h),
            (58, 26, 26),
            max(1, int(1 * self.scale))
        )

        # Subtle divider between left camera panel and right chat panel (#1a1a3a)
        cv2.line(
            ui_frame,
            (self.cam_w, self.main_y),
            (self.cam_w, self.main_y + self.main_h),
            (58, 26, 26),
            max(1, int(1 * self.scale))
        )

        # 3. Left Panel: Aspect-Ratio-Preserved Camera Feed with Scaled Landmarks
        if camera_frame is not None:
            src_h, src_w = camera_frame.shape[:2]
            scale = min(self.cam_w / src_w, self.cam_h / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)

            cam_resized = cv2.resize(camera_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_x = (self.cam_w - new_w) // 2
            pad_y = (self.cam_h - new_h) // 2

            self._cam_scale = scale
            self._cam_pad_x = pad_x
            self._cam_pad_y = pad_y
            self._cam_new_w = new_w
            self._cam_new_h = new_h

            # Camera panel background (#0B0D16 -> BGR: 22, 13, 11)
            cam_panel = np.full((self.cam_h, self.cam_w, 3), (22, 13, 11), dtype=np.uint8)
            cam_panel[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cam_resized

            # Draw MediaPipe skeleton AFTER resize using clamped coordinates in cyan #00E5FF
            if hand_data:
                left_hand = hand_data.get("left_hand", {})
                right_hand = hand_data.get("right_hand", {})
                pose = hand_data.get("pose", {})

                # Pose / Body skeleton (cyan #00E5FF -> BGR: 255, 229, 0)
                if pose.get("present") and pose.get("pixel_landmarks"):
                    self._draw_pose_skeleton_on_feed(
                        cam_panel,
                        pose["pixel_landmarks"],
                        line_color=(255, 229, 0),
                        point_color=(255, 229, 0)
                    )

                # Hand skeletons (cyan #00E5FF -> BGR: 255, 229, 0)
                if left_hand.get("present") and left_hand.get("pixel_landmarks"):
                    self._draw_hand_skeleton_on_feed(
                        cam_panel,
                        left_hand["pixel_landmarks"],
                        line_color=(255, 229, 0),
                        point_color=(255, 229, 0)
                    )
                if right_hand.get("present") and right_hand.get("pixel_landmarks"):
                    self._draw_hand_skeleton_on_feed(
                        cam_panel,
                        right_hand["pixel_landmarks"],
                        line_color=(255, 229, 0),
                        point_color=(255, 229, 0)
                    )

            # Dark gradient overlay inside active video bounds
            grad_h = min(int(80 * self.scale), new_h)
            vid_bottom = pad_y + new_h
            if grad_h > 0:
                overlay_slice = cam_panel[vid_bottom - grad_h:vid_bottom, pad_x:pad_x + new_w].astype(np.float32)
                for gy in range(grad_h):
                    alpha = (gy / float(grad_h)) * 0.40
                    overlay_slice[gy, :] = overlay_slice[gy, :] * (1.0 - alpha)
                cam_panel[vid_bottom - grad_h:vid_bottom, pad_x:pad_x + new_w] = np.clip(overlay_slice, 0, 255).astype(np.uint8)

            ui_frame[self.main_y:self.main_y + self.cam_h, 0:self.cam_w] = cam_panel
        else:
            self._cam_scale = 1.0
            self._cam_pad_x = 0
            self._cam_pad_y = 0
            self._cam_new_w = self.cam_w
            self._cam_new_h = self.cam_h
            cam_panel = np.full((self.cam_h, self.cam_w, 3), (22, 13, 11), dtype=np.uint8)
            ui_frame[self.main_y:self.main_y + self.cam_h, 0:self.cam_w] = cam_panel

        # 4. Top Bar Background (#060711 -> BGR: 17, 7, 6)
        cv2.rectangle(ui_frame, (0, 0), (self.width, self.top_bar_h), (17, 7, 6), -1)
        cv2.line(ui_frame, (0, self.top_bar_h), (self.width, self.top_bar_h), (58, 26, 26), max(1, int(1 * self.scale)))

        # 5. Bottom Bar Top Border (#1a1a3a -> BGR: 58, 26, 26)
        cv2.line(
            ui_frame,
            (0, self.height - self.bottom_bar_h),
            (self.width, self.height - self.bottom_bar_h),
            (58, 26, 26),
            max(1, int(1 * self.scale))
        )
        cv2.rectangle(
            ui_frame,
            (0, self.height - self.bottom_bar_h),
            (self.width, self.height),
            (17, 7, 6),
            -1
        )

        # 6. High-Quality PIL Overlay for Typography, Badges, and Messages
        img_rgb = cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        # Render messages first, then headers and top bar on top for crisp edge clipping
        self._draw_chat_messages(draw, session, scroll_offset)
        self._draw_chat_headers(draw)
        self._draw_top_bar(draw, fps, session.mic_state, camera_frame is not None, session)
        self._draw_camera_overlays(draw, session)
        self._draw_toast(draw, session)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_top_bar(self, draw: ImageDraw.ImageDraw, fps: float, mic_state: str, cam_ok: bool, session: ConversationSession):
        """Renders the clean top bar with resolution-aware typography and crisp icons."""
        # Solid background overlay for top bar (#060711)
        draw.rectangle([0, 0, self.width, self.top_bar_h], fill=(6, 7, 17, 255), outline=(26, 26, 58, 255), width=max(1, int(1 * self.scale)))

        # 1. Left: Brand Logo & Title
        logo_x = int(16 * self.scale)
        logo_y = (self.top_bar_h - int(24 * self.scale)) // 2
        self._draw_asl_logo_icon(draw, logo_x, logo_y, color=(0, 229, 255, 255))

        title_x = logo_x + int(32 * self.scale)
        draw.text((title_x, int(8 * self.scale)), "SAKSHAM V1", font=self.font_brand_title, fill=(0, 229, 255, 255))
        draw.text((title_x, int(31 * self.scale)), "AI Communication Assistant", font=self.font_brand_sub, fill=(133, 137, 149, 255))

        # 2. Center: Status Indicators
        dot_r = max(3, int(4 * self.scale))
        center_x = self.width // 2 - int(150 * self.scale)
        dot_cy = self.top_bar_h // 2

        # Camera status dot
        cam_dot_color = (0, 255, 136, 255) if cam_ok else (255, 145, 0, 255)
        draw.ellipse([center_x, dot_cy - dot_r, center_x + dot_r * 2, dot_cy + dot_r], fill=cam_dot_color)
        cam_text = "Camera: OK" if cam_ok else "Camera: OFF"
        _, _, _, cth = draw.textbbox((0, 0), cam_text, font=self.font_status)
        draw.text((center_x + dot_r * 2 + int(6 * self.scale), (self.top_bar_h - cth) // 2), cam_text, font=self.font_status, fill=(255, 255, 255, 255))

        # System Ready status dot
        sys_x = center_x + int(120 * self.scale) + int(24 * self.scale)
        sys_dot_color = (0, 255, 136, 255)
        draw.ellipse([sys_x, dot_cy - dot_r, sys_x + dot_r * 2, dot_cy + dot_r], fill=sys_dot_color)
        ready_label = f"Dataset: {session.current_dataset_label}" if getattr(session, "dataset_mode", False) else "System Ready"
        _, _, _, rth = draw.textbbox((0, 0), ready_label, font=self.font_status)
        draw.text((sys_x + dot_r * 2 + int(6 * self.scale), (self.top_bar_h - rth) // 2), ready_label, font=self.font_status, fill=(255, 255, 255, 255))

        # 3. Right: Shortcut Pill Buttons (3 buttons only: [M] Mic, [C] Clear, [F] Fullscreen)
        shortcuts = [
            ("[M]  Mic", "mic"),
            ("[C]  Clear", "clear"),
            ("[F]  Fullscreen", "fullscreen")
        ]

        btn_h = int(28 * self.scale)
        btn_y = (self.top_bar_h - btn_h) // 2
        btn_pad_x = int(14 * self.scale)
        ctrl_w = int(96 * self.scale)

        curr_x = self.width - int(16 * self.scale) - ctrl_w
        self.shortcut_rects = {}
        for label, action in reversed(shortcuts):
            _, _, tw, th = draw.textbbox((0, 0), label, font=self.font_shortcut)
            bw = tw + btn_pad_x * 2
            bx = curr_x - bw - int(8 * self.scale)
            draw.rounded_rectangle(
                [bx, btn_y, bx + bw, btn_y + btn_h],
                radius=max(4, int(6 * self.scale)),
                fill=(15, 16, 32, 255),
                outline=(30, 32, 64, 255),
                width=max(1, int(1 * self.scale))
            )
            draw.text((bx + btn_pad_x, btn_y + (btn_h - th) // 2), label, font=self.font_shortcut, fill=(255, 255, 255, 255))
            self.shortcut_rects[action] = (bx, btn_y, bw, btn_h)
            curr_x = bx

        # 4. Far Right: Standard Window Control Icons
        wx = self.width - ctrl_w
        lw = max(1, int(1.5 * self.scale))
        cy = self.top_bar_h // 2

        # Minimize (_)
        draw.line([(wx + int(8 * self.scale), cy + int(4 * self.scale)), (wx + int(20 * self.scale), cy + int(4 * self.scale))], fill=(200, 200, 200, 255), width=lw)
        # Maximize (square)
        sq_s = int(12 * self.scale)
        draw.rectangle([wx + int(34 * self.scale), cy - sq_s // 2, wx + int(34 * self.scale) + sq_s, cy + sq_s // 2], outline=(200, 200, 200, 255), width=lw)
        # Close (X)
        cx1, cy1 = wx + int(64 * self.scale), cy - sq_s // 2
        cx2, cy2 = cx1 + sq_s, cy + sq_s // 2
        draw.line([(cx1, cy1), (cx2, cy2)], fill=(200, 200, 200, 255), width=lw)
        draw.line([(cx2, cy1), (cx1, cy2)], fill=(200, 200, 200, 255), width=lw)

        self.shortcut_rects["close"] = (wx + int(56 * self.scale), btn_y, int(28 * self.scale), btn_h)
        self.shortcut_rects["maximize"] = (wx + int(28 * self.scale), btn_y, int(28 * self.scale), btn_h)
        self.shortcut_rects["minimize"] = (wx, btn_y, int(28 * self.scale), btn_h)

    def _draw_camera_overlays(self, draw: ImageDraw.ImageDraw, session: ConversationSession):
        """Draws top-left LIVE pill and bottom-left detected sign badge inside actual video bounds."""
        # 1. Top-Left: LIVE Badge (inside video bounds)
        live_x = self._cam_pad_x + int(10 * self.scale)
        live_y = self.main_y + self._cam_pad_y + int(10 * self.scale)
        live_pill_w = int(64 * self.scale)
        live_pill_h = int(24 * self.scale)

        draw.rounded_rectangle(
            [live_x, live_y, live_x + live_pill_w, live_y + live_pill_h],
            radius=int(12 * self.scale),
            fill=(0, 0, 0, 136)
        )
        # Red filled circle
        dot_r = max(3, int(3.5 * self.scale))
        dot_cy = live_y + live_pill_h // 2
        draw.ellipse([live_x + int(8 * self.scale), dot_cy - dot_r, live_x + int(8 * self.scale) + dot_r * 2, dot_cy + dot_r], fill=(255, 35, 60, 255))
        _, _, _, lth = draw.textbbox((0, 0), "LIVE", font=self.font_badge_live)
        draw.text((live_x + int(21 * self.scale), live_y + (live_pill_h - lth) // 2), "LIVE", font=self.font_badge_live, fill=(255, 255, 255, 255))

        # 2. Bottom-Left: Detected Sign Recognition Badge (inside actual video bounds)
        pred_label = session.live_prediction
        pred_conf = session.live_confidence

        if not pred_label or pred_label.lower() == "null" or not pred_label.strip():
            raw_top3 = getattr(session, "raw_top3", [])
            if raw_top3 and len(raw_top3) > 0:
                top_label, top_score = raw_top3[0]
                if top_label and top_label.lower() != "null" and top_label.strip():
                    pred_label = top_label
                    pred_conf = top_score

        if pred_label and pred_label.lower() != "null" and pred_label.strip():
            clean_label = pred_label.replace("_", " ")
            conf_str = f"{int(pred_conf * 100)}%" if pred_conf > 0 else ""

            _, _, lw, lh = draw.textbbox((0, 0), clean_label, font=self.font_badge_sign)
            _, _, cw, _ = draw.textbbox((0, 0), conf_str, font=self.font_badge_conf) if conf_str else (0, 0, 0, 0)

            badge_pad_x = int(14 * self.scale)
            badge_w = lw + (int(10 * self.scale) + cw if conf_str else 0) + badge_pad_x * 2
            badge_h = int(28 * self.scale)
            badge_x = self._cam_pad_x + int(12 * self.scale)
            badge_y = self.main_y + self._cam_pad_y + self._cam_new_h - int(14 * self.scale) - badge_h

            draw.rounded_rectangle(
                [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                radius=int(8 * self.scale),
                fill=(0, 0, 0, 153)
            )

            # Label in white bold
            draw.text((badge_x + badge_pad_x, badge_y + (badge_h - lh) // 2), clean_label, font=self.font_badge_sign, fill=(255, 255, 255, 255))
            # Confidence in cyan #00E5FF bold
            if conf_str:
                draw.text((badge_x + badge_pad_x + lw + int(10 * self.scale), badge_y + (badge_h - lh) // 2), conf_str, font=self.font_badge_conf, fill=(0, 229, 255, 255))

    def _draw_chat_headers(self, draw: ImageDraw.ImageDraw):
        """Draws column headers for Hearing Person and Deaf User."""
        # Solid header row background (#080914)
        draw.rectangle(
            [self.chat_x, self.main_y, self.width, self.main_y + self.chat_header_h],
            fill=(8, 9, 20, 255)
        )
        # Bottom border line (#1a1a3a)
        draw.line(
            [(self.chat_x, self.main_y + self.chat_header_h), (self.width, self.main_y + self.chat_header_h)],
            fill=(26, 26, 58, 255),
            width=max(1, int(1 * self.scale))
        )
        # Column divider vertical line in header
        draw.line(
            [(self.chat_x + self.col_w, self.main_y), (self.chat_x + self.col_w, self.main_y + self.chat_header_h)],
            fill=(26, 26, 58, 255),
            width=max(1, int(1 * self.scale))
        )

        # Left column: Ear icon + Hearing Person (white 15px, left padding 20px)
        left_h_x = self.chat_x + int(20 * self.scale)
        _, _, _, hth = draw.textbbox((0, 0), "Hearing Person", font=self.font_col_header_left)
        h_y = self.main_y + (self.chat_header_h - hth) // 2
        self._draw_ear_icon(draw, left_h_x, h_y + 1, color=(255, 255, 255, 255))
        draw.text((left_h_x + int(24 * self.scale), h_y), "Hearing Person", font=self.font_col_header_left, fill=(255, 255, 255, 255))

        # Right column: Hand icon + Deaf User (cyan #00E5FF 15px bold, left padding 20px)
        right_h_x = self.chat_x + self.col_w + int(20 * self.scale)
        self._draw_hand_icon(draw, right_h_x, h_y + 1, color=(0, 229, 255, 255))
        draw.text((right_h_x + int(24 * self.scale), h_y), "Deaf User", font=self.font_col_header_right, fill=(0, 229, 255, 255))

    def _draw_chat_messages(self, draw: ImageDraw.ImageDraw, session: ConversationSession, scroll_offset: int):
        """Renders messages in a synchronized dual-column layout with column placeholders and crisp bubbles."""
        chat_top = self.main_y + self.chat_header_h + int(10 * self.scale)
        chat_bottom = self.main_y + self.chat_h - int(10 * self.scale)
        avail_h = chat_bottom - chat_top
        max_bubble_w = int(self.col_w * 0.80)
        chars_per_line = max(16, int(max_bubble_w / (9.2 * self.scale)))

        hearing_msgs = [m for m in session.messages if m.get("sender") != "Deaf User"]
        deaf_msgs = [m for m in session.messages if m.get("sender") == "Deaf User"]

        # If a column has no messages, draw centered placeholder
        if not hearing_msgs:
            hint_left = "Waiting for speech..."
            _, _, tw, th = draw.textbbox((0, 0), hint_left, font=self.font_placeholder)
            cx = self.chat_x + (self.col_w - tw) // 2
            cy = self.main_y + self.chat_header_h + (self.chat_h - self.chat_header_h - th) // 2
            draw.text((cx, cy), hint_left, font=self.font_placeholder, fill=(85, 85, 119, 255))

        if not deaf_msgs:
            hint_right = "Sign to communicate..."
            _, _, tw, th = draw.textbbox((0, 0), hint_right, font=self.font_placeholder)
            cx = self.chat_x + self.col_w + (self.col_w - tw) // 2
            cy = self.main_y + self.chat_header_h + (self.chat_h - self.chat_header_h - th) // 2
            draw.text((cx, cy), hint_right, font=self.font_placeholder, fill=(85, 85, 119, 255))

        if not session.messages:
            return

        # Pre-measure all message bubbles
        pad_x = int(16 * self.scale)
        pad_y = int(9 * self.scale)
        b_radius = max(8, int(14 * self.scale))

        rendered_items: List[dict] = []
        for msg in session.messages:
            sender = msg.get("sender", "Hearing Person")
            raw_text = msg.get("text", "")
            time_str = msg.get("time", "")

            wrapped = textwrap.fill(raw_text, width=chars_per_line)
            _, _, tw, th = draw.textbbox((0, 0), wrapped, font=self.font_chat_body)
            _, _, tmw, tmh = draw.textbbox((0, 0), time_str, font=self.font_timestamp)

            bw = max(tw + pad_x * 2, int(60 * self.scale))
            bw = min(bw, max_bubble_w)
            bh = th + pad_y * 2

            rendered_items.append({
                "sender": sender,
                "text": wrapped,
                "time": time_str,
                "bw": bw,
                "bh": bh,
                "th": tmh,
                "tw": tmw,
                "total_h": bh + tmh + int(4 * self.scale) + int(12 * self.scale)
            })

        total_content_h = sum(item["total_h"] for item in rendered_items)

        # Determine starting Y position based on total height and scroll offset
        if total_content_h <= avail_h:
            start_y = chat_top + int(4 * self.scale) + (scroll_offset * int(36 * self.scale))
        else:
            start_y = chat_bottom - total_content_h + (scroll_offset * int(36 * self.scale))

        cur_y = start_y
        for item in rendered_items:
            item_h = item["total_h"]
            bubble_y = cur_y
            time_y = bubble_y + item["bh"] + int(4 * self.scale)

            if chat_top - int(20 * self.scale) <= bubble_y <= chat_bottom:
                if item["sender"] == "Deaf User":
                    # Right aligned within right column with 16px right margin
                    bx = self.chat_x + self.chat_w - int(16 * self.scale) - item["bw"]
                    # Bubble background: #142B45 (RGB: 20, 43, 69)
                    draw.rounded_rectangle(
                        [bx, bubble_y, bx + item["bw"], bubble_y + item["bh"]],
                        radius=b_radius,
                        fill=(20, 43, 69, 255),
                        outline=(30, 58, 92, 255),
                        width=max(1, int(1 * self.scale))
                    )
                    # Text: white #ffffff 15px
                    draw.text((bx + pad_x, bubble_y + pad_y), item["text"], font=self.font_chat_body, fill=(255, 255, 255, 255))
                    # Timestamp below, right aligned, grey #858995 10px
                    draw.text((bx + item["bw"] - item["tw"], time_y), item["time"], font=self.font_timestamp, fill=(133, 137, 149, 255))
                else:
                    # Left aligned within left column with 16px left margin
                    bx = self.chat_x + int(16 * self.scale)
                    # Bubble background: #202127 (RGB: 32, 33, 39)
                    draw.rounded_rectangle(
                        [bx, bubble_y, bx + item["bw"], bubble_y + item["bh"]],
                        radius=b_radius,
                        fill=(32, 33, 39, 255),
                        outline=(46, 48, 56, 255),
                        width=max(1, int(1 * self.scale))
                    )
                    # Text: white #ffffff 15px
                    draw.text((bx + pad_x, bubble_y + pad_y), item["text"], font=self.font_chat_body, fill=(255, 255, 255, 255))
                    # Timestamp below, grey #858995 10px
                    draw.text((bx + int(2 * self.scale), time_y), item["time"], font=self.font_timestamp, fill=(133, 137, 149, 255))

            cur_y += item_h

    def _draw_toast(self, draw: ImageDraw.ImageDraw, session: ConversationSession):
        """Renders transient toast notifications centered near the top."""
        if session.toast_message and time.time() < session.toast_expiry:
            msg = session.toast_message
            _, _, tw, th = draw.textbbox((0, 0), msg, font=self.font_toast)
            tx = (self.width - tw) // 2
            ty = self.top_bar_h + int(16 * self.scale)
            draw.rounded_rectangle(
                [tx - int(18 * self.scale), ty - int(8 * self.scale), tx + tw + int(18 * self.scale), ty + th + int(8 * self.scale)],
                radius=int(8 * self.scale),
                fill=(16, 80, 50, 230),
                outline=(0, 200, 100, 255),
                width=max(1, int(1 * self.scale))
            )
            draw.text((tx, ty), msg, font=self.font_toast, fill=(255, 255, 255, 255))

    def _draw_asl_logo_icon(self, draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple):
        """Draws the SAKSHAM ASL vector brand icon with high-DPI scaling."""
        lw = max(2, int(2 * self.scale))
        s = self.scale
        draw.line([(x + int(6 * s), y + int(22 * s)), (x + int(6 * s), y + int(10 * s))], fill=color, width=lw)
        draw.line([(x + int(6 * s), y + int(10 * s)), (x + int(9 * s), y + int(5 * s))], fill=color, width=lw)
        draw.line([(x + int(9 * s), y + int(5 * s)), (x + int(12 * s), y + int(10 * s))], fill=color, width=lw)
        draw.line([(x + int(12 * s), y + int(10 * s)), (x + int(12 * s), y + int(18 * s))], fill=color, width=lw)
        draw.line([(x + int(16 * s), y + int(18 * s)), (x + int(16 * s), y + int(10 * s))], fill=color, width=lw)
        draw.line([(x + int(16 * s), y + int(10 * s)), (x + int(19 * s), y + int(5 * s))], fill=color, width=lw)
        draw.line([(x + int(19 * s), y + int(5 * s)), (x + int(22 * s), y + int(10 * s))], fill=color, width=lw)
        draw.line([(x + int(22 * s), y + int(10 * s)), (x + int(22 * s), y + int(22 * s))], fill=color, width=lw)
        draw.line([(x + int(6 * s), y + int(22 * s)), (x + int(22 * s), y + int(22 * s))], fill=color, width=lw)

    def _draw_ear_icon(self, draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple):
        """Draws ear audio symbol with high-DPI scaling."""
        lw = max(2, int(2 * self.scale))
        s = self.scale
        draw.arc([x, y, x + int(14 * s), y + int(16 * s)], start=135, end=360, fill=color, width=lw)
        draw.arc([x + int(3 * s), y + int(4 * s), x + int(11 * s), y + int(12 * s)], start=150, end=350, fill=color, width=max(1, int(1 * s)))
        draw.line([(x + int(2 * s), y + int(13 * s)), (x + int(6 * s), y + int(16 * s))], fill=color, width=lw)

    def _draw_hand_icon(self, draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple):
        """Draws ASL hand symbol with high-DPI scaling."""
        lw = max(2, int(2 * self.scale))
        s = self.scale
        draw.line([(x + int(2 * s), y + int(14 * s)), (x + int(2 * s), y + int(6 * s))], fill=color, width=lw)
        draw.line([(x + int(6 * s), y + int(14 * s)), (x + int(6 * s), y + int(3 * s))], fill=color, width=lw)
        draw.line([(x + int(10 * s), y + int(14 * s)), (x + int(10 * s), y + int(2 * s))], fill=color, width=lw)
        draw.line([(x + int(14 * s), y + int(14 * s)), (x + int(14 * s), y + int(5 * s))], fill=color, width=lw)
        draw.line([(x + int(2 * s), y + int(14 * s)), (x + int(16 * s), y + int(14 * s))], fill=color, width=lw)
        draw.line([(x + int(2 * s), y + int(14 * s)), (x + int(8 * s), y + int(18 * s))], fill=color, width=lw)
        draw.line([(x + int(8 * s), y + int(18 * s)), (x + int(16 * s), y + int(14 * s))], fill=color, width=lw)

    # ──────────────────────────────────────────────────────────────────
    #  Landmarks & skeleton overlay (OpenCV) - Scaled and Clamped
    # ──────────────────────────────────────────────────────────────────
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index
        (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]

    POSE_CONNECTIONS = [
        (11, 12),           # Shoulders
        (11, 13), (13, 15), # Left Arm
        (12, 14), (14, 16), # Right Arm
        (11, 23), (12, 24), (23, 24), # Torso
    ]

    def _clamp_point(self, raw_x: int, raw_y: int) -> Tuple[int, int, bool]:
        """Clamps scaled coordinates to the active video bounding box."""
        scaled_x = int(raw_x * self._cam_scale) + self._cam_pad_x
        scaled_y = int(raw_y * self._cam_scale) + self._cam_pad_y
        min_x = self._cam_pad_x
        max_x = self._cam_pad_x + self._cam_new_w - 1
        min_y = self._cam_pad_y
        max_y = self._cam_pad_y + self._cam_new_h - 1
        is_inside = (min_x <= scaled_x <= max_x and min_y <= scaled_y <= max_y)
        cx = max(min_x, min(max_x, scaled_x))
        cy = max(min_y, min(max_y, scaled_y))
        return cx, cy, is_inside

    def _draw_pose_skeleton_on_feed(self, frame: cv2.Mat,
                                    pixel_landmarks: List[Tuple[int, int]],
                                    line_color: Tuple[int, int, int],
                                    point_color: Tuple[int, int, int]):
        if not pixel_landmarks or len(pixel_landmarks) < 25:
            return

        scaled_points: Dict[int, Tuple[int, int]] = {}
        inside_map: Dict[int, bool] = {}
        for idx in range(11, min(25, len(pixel_landmarks))):
            ox, oy = pixel_landmarks[idx]
            if ox <= 0 and oy <= 0:
                continue
            cx, cy, inside = self._clamp_point(ox, oy)
            scaled_points[idx] = (cx, cy)
            inside_map[idx] = inside

        lw = max(1, int(1.5 * self.scale))
        dot_r = max(3, int(3.5 * self.scale))

        # Connection lines (skip if BOTH endpoints fall outside video bounds)
        for p1_idx, p2_idx in self.POSE_CONNECTIONS:
            if p1_idx in scaled_points and p2_idx in scaled_points:
                if not (inside_map[p1_idx] or inside_map[p2_idx]):
                    continue
                pt1 = scaled_points[p1_idx]
                pt2 = scaled_points[p2_idx]
                cv2.line(frame, pt1, pt2, line_color, lw, cv2.LINE_AA)

        # Joint dots (only draw if inside video bounds)
        for idx, pt in scaled_points.items():
            if inside_map[idx]:
                cv2.circle(frame, pt, dot_r, point_color, -1, cv2.LINE_AA)

    def _draw_hand_skeleton_on_feed(self, frame: cv2.Mat,
                                    pixel_landmarks: List[Tuple[int, int]],
                                    line_color: Tuple[int, int, int],
                                    point_color: Tuple[int, int, int]):
        if not pixel_landmarks or len(pixel_landmarks) < 21:
            return

        scaled_points: List[Tuple[int, int]] = []
        inside_list: List[bool] = []
        for ox, oy in pixel_landmarks:
            if ox <= 0 and oy <= 0:
                scaled_points.append((0, 0))
                inside_list.append(False)
                continue
            cx, cy, inside = self._clamp_point(ox, oy)
            scaled_points.append((cx, cy))
            inside_list.append(inside)

        lw = max(1, int(1.5 * self.scale))
        dot_r = max(3, int(3.5 * self.scale))

        # Connection lines (skip if BOTH endpoints fall outside video bounds)
        for p1_idx, p2_idx in self.HAND_CONNECTIONS:
            if p1_idx < len(scaled_points) and p2_idx < len(scaled_points):
                if not (inside_list[p1_idx] or inside_list[p2_idx]):
                    continue
                pt1 = scaled_points[p1_idx]
                pt2 = scaled_points[p2_idx]
                cv2.line(frame, pt1, pt2, line_color, lw, cv2.LINE_AA)

        # Landmark dots (only draw if inside video bounds)
        for idx, pt in enumerate(scaled_points):
            if inside_list[idx]:
                cv2.circle(frame, pt, dot_r, point_color, -1, cv2.LINE_AA)

    def handle_click(self, x: int, y: int) -> Optional[str]:
        """Hit-testing for top bar buttons."""
        for action, (bx, by, bw, bh) in self.shortcut_rects.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return action
        return None


