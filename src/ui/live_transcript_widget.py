"""
SAKSHAM Live Transcript Widget — Ultra-Crisp Modern Chat Input Bar
Renders a modern, razor-sharp chat-input pill bar at the bottom of the interface.
Features TrueType font antialiasing (PIL), crisp geometry, left '+' action button,
and right microphone with dropdown chevron matching the sleek design.
"""

import cv2
import numpy as np
import threading
import math
import time
import os
from typing import Optional
from PIL import Image, ImageDraw, ImageFont


class LiveTranscriptWidget:
    """
    Renders a modern, floating chat-input pill bar at the bottom of the SAKSHAM interface.
    Integrates live speech recognition preview and direct text typing seamlessly.
    """

    def __init__(self, bar_height: int = 56):
        self.bar_height = bar_height

        self._lock = threading.Lock()
        self._preview_text = ""
        self._last_final_text = ""
        self._mic_active = False
        self._pulse_phase = 0.0
        self._frame_count = 0

        # Click hit-box zones
        self.bar_rect = (0, 0, 0, 0)
        self.plus_rect = (0, 0, 0, 0)
        self.text_rect = (0, 0, 0, 0)
        self.mic_rect = (0, 0, 0, 0)

        # Colors (BGR for OpenCV, RGB for PIL)
        self.COLOR_BAR_BG       = (30, 30, 34)       # Charcoal pill background
        self.COLOR_BAR_BORDER   = (65, 65, 72)       # Subtle border outline
        self.COLOR_PLACEHOLDER  = (140, 140, 148)    # Crisp placeholder
        self.COLOR_TYPED_TEXT   = (245, 245, 250)    # Crisp white typed text
        self.COLOR_LIVE_TEXT    = (100, 230, 150)    # Active speech recognition green text
        self.COLOR_ICON_IDLE    = (190, 190, 195)    # Icon normal color
        self.COLOR_ICON_ACTIVE  = (70, 225, 120)     # Mic active green
        self.COLOR_CHEVRON      = (140, 140, 148)    # Small chevron arrow
        self.COLOR_CURSOR       = (100, 210, 255)    # Blinking text cursor

        # Load crisp TrueType fonts
        font_path = "C:/Windows/Fonts/segoeui.ttf"
        if not os.path.exists(font_path):
            font_path = "arial.ttf"

        try:
            self.font_text = ImageFont.truetype(font_path, 17)
        except IOError:
            self.font_text = ImageFont.load_default()

    def set_preview(self, text: str):
        """Called by STT thread with live transcription state or interim words."""
        with self._lock:
            self._preview_text = text
            self._mic_active = bool(text)

    def set_final(self, text: str):
        """Called when sentence transcription completes."""
        with self._lock:
            self._last_final_text = text
            self._preview_text = ""
            self._mic_active = False

    def clear(self):
        with self._lock:
            self._preview_text = ""
            self._last_final_text = ""
            self._mic_active = False

    def draw(self, frame: np.ndarray,
             session=None,
             mic_enabled: bool = False) -> np.ndarray:
        """
        Draws the modern chat input bar over the frame with crisp antialiased rendering.
        """
        h, w = frame.shape[:2]
        self._frame_count += 1

        # === Layout Geometry ===
        pad_x  = 28
        pad_y  = 16
        bar_h  = self.bar_height
        bar_w  = w - (pad_x * 2)
        bar_x  = pad_x
        bar_y  = h - bar_h - pad_y
        radius = bar_h // 2

        self.bar_rect = (bar_x, bar_y, bar_w, bar_h)

        # === 1. Draw Rounded Pill Container ===
        # Center filled rect
        cv2.rectangle(
            frame,
            (bar_x + radius, bar_y),
            (bar_x + bar_w - radius, bar_y + bar_h),
            self.COLOR_BAR_BG, -1
        )
        # Left circular cap
        cv2.circle(frame, (bar_x + radius, bar_y + radius), radius, self.COLOR_BAR_BG, -1, cv2.LINE_AA)
        # Right circular cap
        cv2.circle(frame, (bar_x + bar_w - radius, bar_y + radius), radius, self.COLOR_BAR_BG, -1, cv2.LINE_AA)

        # Container border
        cv2.ellipse(frame, (bar_x + radius, bar_y + radius), (radius, radius), 180, -90, 90, self.COLOR_BAR_BORDER, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (bar_x + bar_w - radius, bar_y + radius), (radius, radius), 0, -90, 90, self.COLOR_BAR_BORDER, 1, cv2.LINE_AA)
        cv2.line(frame, (bar_x + radius, bar_y), (bar_x + bar_w - radius, bar_y), self.COLOR_BAR_BORDER, 1, cv2.LINE_AA)
        cv2.line(frame, (bar_x + radius, bar_y + bar_h), (bar_x + bar_w - radius, bar_y + bar_h), self.COLOR_BAR_BORDER, 1, cv2.LINE_AA)

        # === 2. Left '+' Action Icon ===
        plus_cx = bar_x + radius + 4
        plus_cy = bar_y + radius
        plus_arm = 7
        self.plus_rect = (plus_cx - 16, bar_y, 32, bar_h)

        cv2.line(frame, (plus_cx - plus_arm, plus_cy), (plus_cx + plus_arm, plus_cy), self.COLOR_ICON_IDLE, 2, cv2.LINE_AA)
        cv2.line(frame, (plus_cx, plus_cy - plus_arm), (plus_cx, plus_cy + plus_arm), self.COLOR_ICON_IDLE, 2, cv2.LINE_AA)

        # === 3. Right Microphone & Dropdown Chevron ===
        mic_area_w = 64
        mic_btn_cx = bar_x + bar_w - radius - 18
        mic_btn_cy = bar_y + radius

        self.mic_rect = (bar_x + bar_w - radius - 36, bar_y, mic_area_w, bar_h)

        with self._lock:
            preview = self._preview_text
            internal_active = self._mic_active

        is_recording = mic_enabled or internal_active

        # Pulsing ring when mic is active
        self._pulse_phase = (self._pulse_phase + 0.16) % (2 * math.pi)
        if is_recording:
            pulse_radius = int(radius - 5 + 3 * abs(math.sin(self._pulse_phase)))
            cv2.circle(frame, (mic_btn_cx, mic_btn_cy), pulse_radius, (35, 120, 60), 1, cv2.LINE_AA)

        mic_col = self.COLOR_ICON_ACTIVE if is_recording else self.COLOR_ICON_IDLE

        # Draw crisp vector microphone
        mx, my = mic_btn_cx, mic_btn_cy
        # Capsule body
        cv2.rectangle(frame, (mx - 4, my - 8), (mx + 4, my + 2), mic_col, -1)
        cv2.circle(frame, (mx, my - 8), 4, mic_col, -1, cv2.LINE_AA)
        cv2.circle(frame, (mx, my + 2), 4, mic_col, -1, cv2.LINE_AA)
        # Arc stand
        cv2.ellipse(frame, (mx, my + 1), (7, 6), 0, 10, 170, mic_col, 2, cv2.LINE_AA)
        # Stem & base
        cv2.line(frame, (mx, my + 7), (mx, my + 11), mic_col, 2, cv2.LINE_AA)
        cv2.line(frame, (mx - 4, my + 11), (mx + 4, my + 11), mic_col, 2, cv2.LINE_AA)

        # Dropdown chevron 'v'
        chev_x = mic_btn_cx + 15
        chev_y = mic_btn_cy - 1
        cv2.line(frame, (chev_x - 3, chev_y - 2), (chev_x, chev_y + 2), self.COLOR_CHEVRON, 1, cv2.LINE_AA)
        cv2.line(frame, (chev_x, chev_y + 2), (chev_x + 3, chev_y - 2), self.COLOR_CHEVRON, 1, cv2.LINE_AA)

        # === 4. Middle Text Input / Live Transcript Zone ===
        text_x = plus_cx + 22
        text_y_offset = bar_y + (bar_h - 22) // 2
        text_max_w = bar_w - (radius * 2) - mic_area_w - 20
        self.text_rect = (text_x - 6, bar_y, text_max_w + 12, bar_h)

        is_focused = getattr(session, 'is_typing_focused', False) if session else False
        typed_buffer = getattr(session, 'typing_buffer', '') if session else ''

        if preview:
            display_text = preview
            text_color_rgb = (self.COLOR_LIVE_TEXT[2], self.COLOR_LIVE_TEXT[1], self.COLOR_LIVE_TEXT[0])
            show_cursor = True
        elif is_focused or typed_buffer:
            display_text = typed_buffer
            text_color_rgb = (self.COLOR_TYPED_TEXT[2], self.COLOR_TYPED_TEXT[1], self.COLOR_TYPED_TEXT[0])
            show_cursor = is_focused
        else:
            display_text = "Write a message..."
            text_color_rgb = (self.COLOR_PLACEHOLDER[2], self.COLOR_PLACEHOLDER[1], self.COLOR_PLACEHOLDER[0])
            show_cursor = False

        # Render text with PIL for crystal clear typography
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Truncate text from left if exceeding bar width
        display_str = display_text
        while display_str:
            _, _, tw, _ = draw.textbbox((0, 0), display_str, font=self.font_text)
            if tw <= text_max_w:
                break
            display_str = display_str[1:]

        draw.text((text_x, text_y_offset), display_str, font=self.font_text, fill=text_color_rgb)

        # Blinking cursor when typing / streaming
        if show_cursor and (self._frame_count // 16) % 2 == 0:
            _, _, tw, _ = draw.textbbox((0, 0), display_str, font=self.font_text)
            cursor_x = text_x + (tw + 2 if display_str else 0)
            draw.line(
                [(cursor_x, bar_y + 14), (cursor_x, bar_y + bar_h - 14)],
                fill=(self.COLOR_CURSOR[2], self.COLOR_CURSOR[1], self.COLOR_CURSOR[0]),
                width=2
            )

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def handle_click(self, x: int, y: int) -> Optional[str]:
        """
        Determines which sub-element was clicked in the pill bar.
        Returns: 'plus', 'mic', 'text', or None.
        """
        bx, by, bw, bh = self.bar_rect
        if not (bx <= x <= bx + bw and by <= y <= by + bh):
            return None

        # Check Plus button
        px, py, pw, ph = self.plus_rect
        if px <= x <= px + pw and py <= y <= py + ph:
            return 'plus'

        # Check Mic button
        mx, my, mw, mh = self.mic_rect
        if mx <= x <= mx + mw and my <= y <= my + mh:
            return 'mic'

        # Text area
        return 'text'
