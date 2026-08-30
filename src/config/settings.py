import cv2

# ==========================================
# SAKSHAM V1 CONFIGURATION SETTINGS
# ==========================================

# Camera Settings
CAMERA_INDEX = 0          # Default system camera
FRAME_WIDTH = 1280        # Camera capture width
FRAME_HEIGHT = 720        # Camera capture height
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
COLOR_BG = (30, 30, 30)            # Dark grey background
COLOR_TEXT_PRIMARY = (240, 240, 240)  # Off-white text
COLOR_TEXT_SECONDARY = (150, 150, 150) # Medium grey for timestamps
COLOR_ACCENT = (255, 120, 50)       # Accent blue (BGR)
COLOR_SUCCESS = (50, 200, 50)      # Green for ready states
COLOR_WARNING = (0, 140, 255)      # Orange for warnings/processing
COLOR_DIVIDER = (60, 60, 60)    # Dark grey borders

# Font settings
FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_HEADER = 0.7
FONT_SCALE_BODY = 0.6
FONT_SCALE_SMALL = 0.4
FONT_THICKNESS = 1
FONT_THICKNESS_BOLD = 2
