"""
SAKSHAM - Real-Time Sign Language AI Web Application (Streamlit Cloud Edition)
"""
import os
import sys
import time
import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="SAKSHAM - AI Sign Language Communicator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark accessibility interface
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .detected-sign {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10b981;
    }
    .chat-bubble-deaf {
        background-color: #1e3a8a;
        color: #f8fafc;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
    }
    .chat-bubble-hearing {
        background-color: #334155;
        color: #f8fafc;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model & MediaPipe once (Cached) ────────────────────────────────────
@st.cache_resource
def load_sign_components():
    from src.sign_language.inference import ONNXInferenceEngine
    from src.sign_language.tracker import HandTracker

    model_path = os.path.join("models", "model.onnx")
    label_map_path = os.path.join("models", "label_map.json")

    engine = ONNXInferenceEngine(
        model_path=model_path,
        label_map_path=label_map_path,
        min_confidence=0.40,
        stability_frames=2,
    )
    tracker = HandTracker()
    return engine, tracker

# ─── Session State Initialization ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"sender": "System", "text": "SAKSHAM AI ready. Use webcam snapshot or video to translate signs.", "time": time.strftime("%H:%M:%S")},
    ]

if "last_recognized" not in st.session_state:
    st.session_state.last_recognized = "None"

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

# ─── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hand.png", width=64)
    st.title("🤟 SAKSHAM")
    st.markdown("**Real-Time Sign Language Assistant**")
    st.markdown("---")
    
    st.subheader("Model Vocabulary (13 Signs)")
    vocab = [
        "Hello", "Bye", "Yes", "No", "Please", "Thank You", 
        "Water", "Food", "Drink", "Hungry", "Home", "Sick", "Where"
    ]
    st.write(", ".join(vocab))
    st.markdown("---")

    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### About")
    st.caption("Bridging the communication barrier between deaf and hearing communities through computer vision & deep learning.")

# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🤟 SAKSHAM Communication Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sign Language Recognition & Two-Way Accessibility Interface</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📷 Camera Input")
    st.caption("Capture frames to detect sign language gestures live.")

    # Streamlit camera widget
    camera_photo = st.camera_input("Pose your sign to the camera:")

    if camera_photo is not None:
        try:
            engine, tracker = load_sign_components()
            image = Image.open(camera_photo)
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Process frame through MediaPipe
            detected, hand_data = tracker.process_frame(frame_bgr)

            if detected:
                # Add frame repeatedly to simulate sequence buffer for snapshot recognition
                state = {}
                for _ in range(engine.seq_len):
                    state = engine.add_frame(hand_data)

                pred = state.get("confirmed_prediction") or state.get("raw_prediction")
                conf = state.get("confidence", 0.0)

                if pred:
                    st.session_state.last_recognized = pred
                    st.session_state.last_confidence = conf

                    # Check if last message was already this
                    if not st.session_state.messages or st.session_state.messages[-1]["text"] != pred:
                        st.session_state.messages.append({
                            "sender": "Deaf User (Sign)",
                            "text": pred,
                            "time": time.strftime("%H:%M:%S")
                        })
                else:
                    st.info("✋ Hand/Body detected! Hold sign clearly for recognition.")
            else:
                st.warning("⚠️ No hands or body pose detected. Please step in front of the camera.")
        except Exception as e:
            st.error(f"Inference error: {e}")

with col2:
    st.subheader("📊 Live Recognition")
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #94a3b8; font-size: 0.9rem;">RECOGNIZED SIGN</div>
        <div class="detected-sign">{st.session_state.last_recognized}</div>
        <div style="color: #38bdf8; font-size: 1.1rem; margin-top: 0.5rem;">
            Confidence: {st.session_state.last_confidence * 100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("💬 Live Conversation")
    
    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.messages:
            sender = msg["sender"]
            text = msg["text"]
            t = msg.get("time", "")
            if "Deaf" in sender:
                st.markdown(f'<div class="chat-bubble-deaf"><b>🤟 {sender}</b> [{t}]<br>{text}</div>', unsafe_allow_html=True)
            elif "Hearing" in sender:
                st.markdown(f'<div class="chat-bubble-hearing"><b>🗣️ {sender}</b> [{t}]<br>{text}</div>', unsafe_allow_html=True)
            else:
                st.info(f"{text}")

    # Hearing person reply input
    hearing_reply = st.chat_input("Hearing person reply (type here):")
    if hearing_reply:
        st.session_state.messages.append({
            "sender": "Hearing Person (Text)",
            "text": hearing_reply,
            "time": time.strftime("%H:%M:%S")
        })
        st.rerun()
