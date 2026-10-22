import cv2

# ==========================================
# SAKSHAM V1 CONFIGURATION SETTINGS
# ==========================================

# Camera & UI Settings
CAMERA_INDEX = 0          # Default system camera
FRAME_WIDTH = 1920        # Full HD width for crisp UI rendering
FRAME_HEIGHT = 1080       # Full HD height for crisp UI rendering
TARGET_FPS = 30           # Limit frame processing rate to save CPU

# Performance & Tracking Optimizations
TRACKING_FRAME_WIDTH = 480   # Resized width for MediaPipe processing
TRACKING_FRAME_HEIGHT = 270  # Resized height for MediaPipe processing

# Hand Tracking Settings (MediaPipe - for Phase 2 readiness)
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
MAX_NUM_HANDS = 2         # Sign language often uses two hands

# Developer Mode Settings
DEV_MODE = False             # Press 'd' to toggle Developer Mode
SHOW_FPS = False             # Controlled by Dev Mode
SHOW_TRACKING_LANDMARKS = False # Controlled by Dev Mode

import os
from dotenv import load_dotenv
load_dotenv()

# Speech Recognition Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
STT_BACKEND = os.getenv("STT_BACKEND", "groq")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Styling & Theme (Accessibility First - High Contrast Dark Mode BGR Format)
COLOR_BG = (20, 9, 8)                # Overall canvas background #080914
COLOR_TOPBAR = (17, 7, 6)            # Top bar background #060711
COLOR_BOTTOMBAR = (17, 7, 6)         # Bottom bar background #060711
COLOR_CAM_BG = (22, 13, 11)          # Camera panel background #0B0D16
COLOR_CHAT_BG = (21, 12, 10)         # Chat panel background #0A0C15
COLOR_TEXT_PRIMARY = (255, 255, 255) # Primary text #FFFFFF
COLOR_TEXT_SECONDARY = (149, 137, 133) # Secondary/timestamp text #858995 (BGR)
COLOR_ACCENT = (255, 229, 0)         # Primary cyan accent #00E5FF (BGR)
COLOR_SUCCESS = (136, 255, 0)        # Active/mic green #00FF88 (BGR)
COLOR_DIVIDER = (58, 26, 26)         # Borders and dividers #1a1a3a (BGR)

# Font settings
FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_HEADER = 0.7
FONT_SCALE_BODY = 0.6
FONT_SCALE_SMALL = 0.4
FONT_THICKNESS = 1
FONT_THICKNESS_BOLD = 2
