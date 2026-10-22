"""
SAKSHAM Live Transcript Widget.
Renders the integrated 64px Bottom Bar with:
- Rounded input field (#0f1020, border #1e2040, radius 24px)
- Inside input right: Microphone icon (#555577 idle, #00FF88 active), 1px divider, send arrow icon
- Standalone Export button (#0f1020, border #1e2040, radius 8px, 110px width)
"""

import cv2
import numpy as np
import threading
import time
import os
from typing import Optional
from PIL import Image, ImageDraw, ImageFont


class LiveTranscriptWidget:
    """
    Renders the SAKSHAM integrated chat input and action bar.
    """

    def __init__(self, bar_height: int = 64):
        self.bar_height = bar_height
        self.scale = max(1.0, bar_height / 64.0)

        self._lock = threading.Lock()
        self._preview_text = ""
        self._last_final_text = ""
        self._mic_active = False
        self._frame_count = 0

        # Click hit-box zones
        self.mic_rect = (0, 0, 0, 0)
        self.input_rect = (0, 0, 0, 0)
        self.send_rect = (0, 0, 0, 0)
        self.export_rect = (0, 0, 0, 0)

        # Fonts with resolution scaling
        font_path = "C:/Windows/Fonts/segoeui.ttf"
        font_bold_path = "C:/Windows/Fonts/segoeuib.ttf"

        if not os.path.exists(font_path):
            font_path = "arial.ttf"
            font_bold_path = "arialbd.ttf"

        try:
            self.font_input = ImageFont.truetype(font_path, max(12, int(14 * self.scale)))
            self.font_btn = ImageFont.truetype(font_bold_path, max(11, int(13 * self.scale)))
        except IOError:
            fallback = ImageFont.load_default()
            self.font_input = fallback
            self.font_btn = fallback

    def set_preview(self, text: str):
        with self._lock:
            self._preview_text = text
            self._mic_active = bool(text)

    def set_final(self, text: str):
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
        Renders the integrated bottom bar with high-DPI vector precision and crisp typography.
        """
        h, w = frame.shape[:2]
        self._frame_count += 1

        bar_h = self.bar_height
        bar_y = h - bar_h
        s = self.scale

        with self._lock:
            preview = self._preview_text
            internal_active = self._mic_active

        is_recording = mic_enabled or internal_active

        # ── 1. Export Button Dimensions ──
        export_w = int(110 * s)
        export_h = int(44 * s)
        export_x = w - int(16 * s) - export_w
        export_y = bar_y + (bar_h - export_h) // 2
        self.export_rect = (export_x, export_y, export_w, export_h)

        # ── 2. Main Input Field Dimensions ──
        input_x = int(16 * s)
        input_w = (export_x - int(12 * s)) - input_x
        input_h = int(44 * s)
        input_y = bar_y + (bar_h - input_h) // 2
        self.input_rect = (input_x, input_y, input_w - int(76 * s), input_h)

        # Inside Input Right Coordinates
        mic_cx = input_x + input_w - int(60 * s)
        mic_cy = input_y + input_h // 2
        self.mic_rect = (mic_cx - int(16 * s), input_y, int(32 * s), input_h)

        send_btn_x = input_x + input_w - int(24 * s)
        send_btn_y = input_y + input_h // 2
        self.send_rect = (send_btn_x - int(14 * s), input_y, int(38 * s), input_h)

        # ── 3. High-DPI Vector Rendering via Pillow (RGBA) ──
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert('RGBA')
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        # Bottom Bar Background (#060711) and Top Border (#1a1a3a)
        draw.rectangle([0, bar_y, w, h], fill=(6, 7, 17, 255), outline=(26, 26, 58, 255), width=max(1, int(1 * s)))

        # Export Button Container (#0f1020, outline #1e2040, radius 8px)
        draw.rounded_rectangle(
            [export_x, export_y, export_x + export_w, export_y + export_h],
            radius=max(6, int(8 * s)),
            fill=(15, 16, 32, 255),
            outline=(30, 32, 64, 255),
            width=max(1, int(1.5 * s))
        )

        # Export Download Tray Vector Icon (White)
        ex_ic_x = export_x + int(16 * s)
        ex_ic_y = export_y + (export_h - int(16 * s)) // 2
        lw = max(2, int(2 * s))

        # Arrow down stem and head
        draw.line([(ex_ic_x + int(5 * s), ex_ic_y), (ex_ic_x + int(5 * s), ex_ic_y + int(8 * s))], fill=(255, 255, 255, 255), width=lw)
        draw.line([(ex_ic_x + int(2 * s), ex_ic_y + int(5 * s)), (ex_ic_x + int(5 * s), ex_ic_y + int(8 * s))], fill=(255, 255, 255, 255), width=lw)
        draw.line([(ex_ic_x + int(8 * s), ex_ic_y + int(5 * s)), (ex_ic_x + int(5 * s), ex_ic_y + int(8 * s))], fill=(255, 255, 255, 255), width=lw)
        # Tray base
        draw.line([(ex_ic_x, ex_ic_y + int(10 * s)), (ex_ic_x, ex_ic_y + int(13 * s))], fill=(255, 255, 255, 255), width=lw)
        draw.line([(ex_ic_x, ex_ic_y + int(13 * s)), (ex_ic_x + int(10 * s), ex_ic_y + int(13 * s))], fill=(255, 255, 255, 255), width=lw)
        draw.line([(ex_ic_x + int(10 * s), ex_ic_y + int(13 * s)), (ex_ic_x + int(10 * s), ex_ic_y + int(10 * s))], fill=(255, 255, 255, 255), width=lw)

        # Export Button Text (White bold)
        _, _, _, eth = draw.textbbox((0, 0), "Export", font=self.font_btn)
        draw.text((export_x + int(36 * s), export_y + (export_h - eth) // 2), "Export", font=self.font_btn, fill=(255, 255, 255, 255))

        # Main Input Field Container (#0f1020, outline #1e2040, radius 22px)
        draw.rounded_rectangle(
            [input_x, input_y, input_x + input_w, input_y + input_h],
            radius=int(22 * s),
            fill=(15, 16, 32, 255),
            outline=(30, 32, 64, 255),
            width=max(1, int(1.5 * s))
        )

        # Microphone Icon and Soft Glow
        if is_recording:
            mic_col = (0, 255, 136, 255)  # Active #00FF88
            glow_r = int(18 * s)
            draw.ellipse([mic_cx - glow_r, mic_cy - glow_r, mic_cx + glow_r, mic_cy + glow_r], fill=(0, 255, 136, 40))
        else:
            mic_col = (133, 137, 149, 255)  # Idle #858995

        # Crisp vector microphone
        cw = int(3 * s)
        ch = int(6 * s)
        draw.rounded_rectangle([mic_cx - cw, mic_cy - ch, mic_cx + cw, mic_cy + int(2 * s)], radius=int(3 * s), fill=mic_col)
        draw.arc([mic_cx - int(6 * s), mic_cy - int(3 * s), mic_cx + int(6 * s), mic_cy + int(5 * s)], start=0, end=180, fill=mic_col, width=max(1, int(1.5 * s)))
        draw.line([(mic_cx, mic_cy + int(5 * s)), (mic_cx, mic_cy + int(9 * s))], fill=mic_col, width=max(1, int(1.5 * s)))
        draw.line([(mic_cx - int(3 * s), mic_cy + int(9 * s)), (mic_cx + int(3 * s), mic_cy + int(9 * s))], fill=mic_col, width=max(1, int(1.5 * s)))

        # Thin Vertical Divider Line 1px #1e2040
        div_x = input_x + input_w - int(40 * s)
        draw.line([(div_x, input_y + int(12 * s)), (div_x, input_y + input_h - int(12 * s))], fill=(30, 32, 64, 255), width=max(1, int(1 * s)))

        # Send Arrow Icon (➤)
        is_focused = getattr(session, 'is_typing_focused', False) if session else False
        typed_buffer = getattr(session, 'typing_buffer', '') if session else ''
        has_text = bool(typed_buffer or preview)
        send_col = (0, 229, 255, 255) if has_text else (133, 137, 149, 255)

        sx, sy = send_btn_x - int(4 * s), send_btn_y - int(7 * s)
        draw.polygon([
            (sx, sy),
            (sx + int(14 * s), sy + int(7 * s)),
            (sx, sy + int(14 * s)),
            (sx + int(3 * s), sy + int(7 * s))
        ], fill=send_col)

        # Input Text / Placeholder / Live Speech Preview
        text_render_x = input_x + int(22 * s)
        text_max_w = input_w - int(95 * s)

        if preview:
            display_str = preview
            text_color = (0, 255, 136, 255)  # #00FF88
            show_cursor = True
        elif is_focused or typed_buffer:
            display_str = typed_buffer
            text_color = (255, 255, 255, 255)
            show_cursor = is_focused
        else:
            display_str = "Write a message..."
            text_color = (133, 137, 149, 255)  # Crisp secondary #858995
            show_cursor = False

        _, _, _, ith = draw.textbbox((0, 0), "Ag", font=self.font_input)
        text_render_y = input_y + (input_h - ith) // 2

        # Truncate if overflowing width
        truncated = display_str
        while truncated:
            _, _, tw, _ = draw.textbbox((0, 0), truncated, font=self.font_input)
            if tw <= text_max_w:
                break
            truncated = truncated[1:]

        draw.text((text_render_x, text_render_y), truncated, font=self.font_input, fill=text_color)

        # Blinking Cursor
        if show_cursor and (self._frame_count // 16) % 2 == 0:
            _, _, tw, _ = draw.textbbox((0, 0), truncated, font=self.font_input)
            cursor_x = text_render_x + (tw + int(2 * s) if truncated else 0)
            draw.line(
                [(cursor_x, input_y + int(12 * s)), (cursor_x, input_y + input_h - int(12 * s))],
                fill=(0, 229, 255, 255),
                width=max(1, int(2 * s))
            )

        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

    def handle_click(self, x: int, y: int) -> Optional[str]:
        """
        Determines which interactive bottom bar element was clicked.
        Returns: 'mic', 'send', 'export', 'text', or None.
        """
        # Mic icon inside input
        mx, my, mw, mh = self.mic_rect
        if mx <= x <= mx + mw and my <= y <= my + mh:
            return 'mic'

        # Send arrow icon inside input
        sx, sy, sw, sh = self.send_rect
        if sx <= x <= sx + sw and sy <= y <= sy + sh:
            return 'send'

        # Input field text area
        ix, iy, iw, ih = self.input_rect
        if ix <= x <= ix + iw and iy <= y <= iy + ih:
            return 'text'

        # Export button
        ex, ey, ew, eh = self.export_rect
        if ex <= x <= ex + ew and ey <= y <= ey + eh:
            return 'export'

        return None
