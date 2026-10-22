"""
SAKSHAM V1/V2 - Web Edition for Streamlit Cloud
Renders the EXACT original split-screen SAKSHAM canvas layout using UIRenderer,
with continuous live video sequence processing and sign prediction.
"""
import os
import sys
import time
import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Import existing core modules directly
from src.session_manager import ConversationSession
from src.ui.renderer import UIRenderer
from src.sign_language.engine import SignLanguageEngine
import src.config.settings as config

st.set_page_config(
    page_title="SAKSHAM - Real-Time AI Sign Language System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Full-Screen Dark Canvas Display
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100% !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-color: #080914;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialize Shared Session and Engines ────────────────────────────────────
if "session" not in st.session_state:
    session = ConversationSession()
    # Add initial welcome greeting
    session.add_message("Hearing Person", "Hello! How can I help you today?", source="Speech")
    st.session_state.session = session
else:
    session = st.session_state.session

@st.cache_resource
def get_system_components():
    renderer = UIRenderer(width=1280, height=720)
    
    def on_sign(label: str):
        if label and label != "null":
            session.add_message("Deaf User", label, source="Sign")

    sl_engine = SignLanguageEngine(on_sign_recognized=on_sign)
    return renderer, sl_engine

renderer, sl_engine = get_system_components()

# ─── Top Brand Bar ────────────────────────────────────────────────────────────
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("### 🤟 SAKSHAM V1 — AI Real-Time Sign Language Assistant")
with col_head2:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        session.clear()
        st.rerun()

# ─── Live Video & Exact Split-Screen Canvas ───────────────────────────────────
# We use st.camera_input to capture webcam feed from the user's browser
cam_col, ctrl_col = st.columns([2.5, 1])

with ctrl_col:
    st.markdown("#### 💬 Hearing Person Reply")
    typed_reply = st.text_input("Type message as Hearing Person (Press Enter):", key="hearing_input")
    if typed_reply:
        session.add_message("Hearing Person", typed_reply.strip(), source="Typed")
        st.session_state["hearing_input"] = ""
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📋 System Status")
    inf = sl_engine.inference
    if inf:
        st.success("✅ ONNX GRU Model Loaded")
        st.metric(
            label="Live Recognized Sign",
            value=session.live_prediction or "Waiting for sign...",
            delta=f"Confidence: {session.live_confidence*100:.1f}%" if session.live_prediction else "0.0%"
        )
        st.caption(f"Vocabulary ({inf.num_classes} classes): Hello, Bye, Yes, No, Please, Thank You, Water, Food, Drink, Hungry, Home, Sick, Where")
    else:
        st.warning("⚠️ ONNX Engine Offline")

with cam_col:
    camera_photo = st.camera_input("📷 Webcam Feed (Continuous Live Motion Recognition)", label_visibility="collapsed")

# ─── Process Frame & Render Original Split Screen ─────────────────────────────
frame_bgr = None

if camera_photo is not None:
    try:
        image = Image.open(camera_photo)
        frame_rgb = np.array(image)
        raw_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.flip(raw_bgr, 1)

        # Process through full SignLanguageEngine pipeline
        hand_detected, hand_data = sl_engine.process_frame(frame_bgr)
        session.set_live_hand_visibility(hand_data if hand_detected else None)

        if sl_engine.inference is not None:
            inf_state = sl_engine.inference.get_state()
            session.live_prediction = inf_state.get("prediction")
            session.live_confidence = inf_state.get("confidence", 0.0)
            session.live_top3 = inf_state.get("top3", [])
            session.buffer_progress = inf_state.get("buffer_progress", 0)
            session.buffer_capacity = inf_state.get("buffer_capacity", 30)

            # If user just posed, simulate sequence accumulation so snapshot triggers sign
            if hand_detected and not session.live_prediction:
                for _ in range(sl_engine.inference.seq_len):
                    state = sl_engine.inference.add_frame(hand_data)
                session.live_prediction = state.get("prediction") or state.get("raw_prediction")
                session.live_confidence = state.get("confidence", 0.0)
                if session.live_prediction and session.live_prediction != "null":
                    session.add_message("Deaf User", session.live_prediction, source="Sign")

    except Exception as e:
        st.error(f"Processing error: {e}")

# Render the exact canvas (if no frame captured yet, pass dummy black camera frame)
if frame_bgr is None:
    frame_bgr = np.full((720, 1280, 3), (15, 12, 10), dtype=np.uint8)
    cv2.putText(frame_bgr, "Webcam Connecting...", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

canvas = renderer.render(
    camera_frame=frame_bgr,
    session=session,
    fps=30.0,
    scroll_offset=0,
    hand_data=hand_data if (camera_photo and hand_detected) else None,
)

# Convert BGR canvas to RGB for Streamlit display
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

st.markdown("---")
st.markdown("#### 🖥️ SAKSHAM Split-Screen Live Console")
st.image(canvas_rgb, use_container_width=True, caption="SAKSHAM High-Contrast Two-Way Accessible Canvas")
