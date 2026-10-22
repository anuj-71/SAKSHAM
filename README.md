# 🤟 SAKSHAM

### AI-Powered Real-Time Sign Language Communication System

> Bridging the communication gap between deaf and hearing individuals through real-time Sign Language Recognition, Speech Recognition, and Text-to-Speech.

---

## 📖 Overview

SAKSHAM is an AI-powered accessibility application designed to enable seamless communication between people who use sign language and those who communicate through speech.

The system combines Computer Vision, Deep Learning, Speech Recognition, and Text-to-Speech into a unified real-time application capable of translating sign language into spoken words while simultaneously converting speech into readable text.

The project is built with a modular architecture, allowing each subsystem—vision, speech, UI, and machine learning—to operate independently while communicating through a central application controller.

---

## ✨ Features

* 🤟 Real-time Sign Language Recognition
* 🎤 Live Speech-to-Text transcription
* 🔊 Instant Text-to-Speech synthesis
* 💬 Two-way communication interface
* 📷 Live webcam processing
* 🧠 MediaPipe landmark extraction
* ⚡ ONNX Runtime optimized inference
* 📁 Dataset collection mode
* 📤 Export conversations as JSON and TXT
* 🎨 Clean split-screen accessibility-focused interface

---

# System Architecture

```text
                    Webcam
                       │
                       ▼
              Camera Manager
                       │
                       ▼
          MediaPipe Hand Tracking
                       │
                       ▼
         Landmark Preprocessing
                       │
                       ▼
           ONNX GRU Classifier
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 Sign Prediction              Confidence Score
        │
        ▼
 Text Generation
        │
        ▼
 Edge Text-to-Speech
        │
        ▼
 Audio Output


                Microphone
                     │
                     ▼
          Speech Recognition
                     │
                     ▼
             Live Transcript
                     │
                     ▼
              Conversation UI
```

---

# Tech Stack

| Category           | Technologies           |
| ------------------ | ---------------------- |
| Language           | Python 3.12+           |
| Deep Learning      | PyTorch                |
| Inference          | ONNX Runtime           |
| Computer Vision    | OpenCV, MediaPipe      |
| Speech Recognition | SpeechRecognition      |
| Text-to-Speech     | Edge-TTS               |
| Audio              | pygame                 |
| Data               | Pandas, NumPy, PyArrow |
| Logging            | Rich                   |

---

# Project Structure

```text
SAKSHAM/
│
├── configs/                # Hyperparameter & application configurations
├── dataset/                # Active dataset partitions & processed samples
├── models/                 # Model checkpoints (.pt) & exported ONNX models
├── exports/                # Exported session transcripts & logs
├── outputs/                # Evaluation & inference outputs
├── reports/                # Benchmark and analysis reports
│
├── src/                    # Core application source code
│   ├── camera/             # Video capture & thread management
│   ├── sign_language/      # Landmark tracking, normalization & inference
│   ├── speech/             # Speech-to-Text & Text-to-Speech engines
│   ├── ui/                 # Split-screen renderer & live transcript widget
│   └── main.py             # Main application orchestrator
│
├── tools/                  # Data recording, extraction & training scripts
├── tests/                  # Unit and integration test suites
│
├── requirements.txt        # Python dependency manifest
├── INSTALLATION.md         # Environment setup instructions
└── README.md               # System documentation & quickstart
```

---

# AI Pipeline

### 1. Hand Detection

MediaPipe detects hand landmarks from webcam frames.

↓

### 2. Landmark Processing

* Normalize coordinates
* Wrist-relative positioning
* Sequence padding/truncation
* Hand presence encoding

↓

### 3. GRU Classifier

A lightweight two-layer GRU predicts the performed sign.

↓

### 4. ONNX Runtime

The trained model is exported to ONNX for fast CPU inference.

↓

### 5. Speech Generation

Predicted text is converted into natural speech using Microsoft Edge TTS.

---

# Sign Language Model

| Property        | Value        |
| --------------- | ------------ |
| Model           | 2-Layer GRU  |
| Input Features  | 128          |
| Sequence Length | 64 Frames    |
| Output Classes  | 13           |
| Runtime         | ONNX Runtime |

### Current Vocabulary

* Hello
* Bye
* Yes
* No
* Please
* Thank You
* Water
* Food
* Drink
* Hungry
* Home
* Sick
* Where

---

# Speech Pipeline

### Speech-to-Text

* Google Web Speech API
* Background listener threads
* Queue-based processing
* Ambient noise adaptation

### Text-to-Speech

* Microsoft Edge-TTS
* Cached audio generation
* Low-latency playback
* pygame audio engine

---

# Installation

## Clone the repository

```bash
git clone https://github.com/<your-username>/SAKSHAM.git

cd SAKSHAM
```

## Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

# Running the Application

Start the application:

```bash
python -m src.main
```

---

# Training the Model

Train the GRU classifier:

```bash
python -m tools.train \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3
```

After training, the following files are generated:

```text
dataset/models/

model.onnx
gru_best.pt
label_map.json
evaluation_results.json
confusion_matrix.png
training_history.json
```

---

# Dataset

The project currently uses:

* 5,771 training samples
* MediaPipe landmark sequences
* 13 Indian Sign Language classes
* Participant-aware training split

---

# Configuration

Important configuration files:

```text
src/config/settings.py
```

Model labels:

```text
SAKSHAM_DATASET/sign_map.json
```

---

# Current Status

| Module             | Status         |
| ------------------ | -------------- |
| Dataset            | ✅ Complete     |
| MediaPipe Pipeline | ✅ Complete     |
| Preprocessing      | ✅ Complete     |
| ONNX Inference     | ✅ Complete     |
| Speech Recognition | ✅ Complete     |
| Text-to-Speech     | ✅ Complete     |
| Real-Time UI       | ✅ Complete     |
| Model Training     | ✅ Complete     |
| Evaluation         | 🚧 In Progress |

---

# Roadmap

* Expand vocabulary beyond 13 signs
* Improve recognition accuracy
* Transformer-based sign recognition
* ONNX quantization
* Cross-platform packaging
* Multi-language support
* Larger ISL dataset
* Performance optimization

---

# Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# License

This project is intended for educational and research purposes.

---

## Author

**Anuj**

B.Tech Computer Science (AI & ML)

Vellore Institute of Technology, Chennai

Passionate about AI, Accessibility, Computer Vision, and Full-Stack Development.
