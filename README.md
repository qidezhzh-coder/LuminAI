# -LuminAI-Assistive-neck-wearable-device
LuminAI is a neck-worn assistive wearable designed for visually impaired users, integrating OCR, gesture recognition, and real-time obstacle detection into a single portable system. Powered by a Raspberry Pi 5 and Hailo AI acceleration, the device performs all processing locally on-device, enabling low-latency, privacy-focused, and cost-effective assistance for reading, navigation, and daily interaction.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [AI Modules](#-ai-modules)
5. [Hardware](#-hardware)
6. [Software Stack](#-software-stack)
7. [Installation & Setup](#-installation--setup)
8. [Usage](#-usage)
9. [Performance & Metrics](#-performance--metrics)
10. [Mechanical Design](#-mechanical-design)
11. [Companion App](#-companion-app)
12. [Team](#-team)

---

## Project Overview

**LuminAI** is a next-generation wearable assistive device designed for visually impaired individuals — particularly those with primary open-angle glaucoma (POAG), the leading cause of peripheral vision loss worldwide, affecting over 80 million people.

Current assistive solutions are fragmented: OCR devices only handle text reading, navigation aids detect obstacles but nothing else, and AI smart glasses are expensive and require manual activation. LuminAI addresses this gap by integrating three AI-powered modules in a single, affordable, ergonomic neck-wearable:

- **Optical Character Recognition (OCR)** — reads text aloud in real time
- **Gesture Recognition (GR)** — interprets non-verbal social cues (waving, pointing, handshakes)
- **Obstacle Detection (OD)** — alerts the user about objects in their path

The prototype has been validated at **TRL 5** through functional testing, with all AI inference running **fully on-device** for maximum privacy and low latency (~240ms end-to-end).

---

## ✅ Features

- **Multi-modal AI**: OCR + Gesture Recognition + Obstacle Detection in one device
- **On-device inference**: No cloud, no data transmission — full privacy by design
- **Hailo NPU acceleration**: Reduces CPU load to <26% during active inference
- **All-day battery**: 6.5h continuous operation (10Ah Li-ion, tested in OCR mode)
- **Ergonomic neck-wearable**: TPU 95A flexible casing, 3D-printed prototype
- **Companion Flutter app**: Wireless remote control over HTTP
- **Modular software**: Flask microservices — add new AI modules without touching core logic
- **Audio-first interface**: espeak TTS output, no screen required
- **6 gesture classes**: point_up, point_down, point_left, point_right, waving, handshake
- **Multilingual OCR**: PaddleOCR pipeline with SymSpell spell correction

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (Orchestrator)                  │
│  GPIO pin 17 → Mode selection (Reading / Pointing)             │
├───────────────┬─────────────────┬───────────────────────────────┤
│  Camera       │   Flask Server  │   Companion App (Flutter)     │
│  (V4L2 /      │   (On-device    │   HTTP endpoints:             │
│  libcamera)   │   REST API)     │   /status /language /volume   │
└───────┬───────┴────────┬────────┴───────────────────────────────┘
        │                │
        ▼                ▼
┌───────────────┐   ┌────────────────────────────────────┐
│ CameraManager │   │         Hailo AI HAT (NPU)          │
│  (exclusive   │   ├──────────────┬─────────────────────┤
│   access)     │   │  PaddleOCR   │  YOLO (Obstacle)    │
└───────────────┘   │  endpoint    │  endpoint           │
                    ├──────────────┴─────────────────────┤
                    │       Gesture Recognition           │
                    │  (MediaPipe + MLP, CPU/.venv)       │
                    └────────────────────────────────────┘
                                    │
                                    ▼
                           espeak (Audio output)
                           MAX98357A → Speaker
```

**Key design decisions:**
- The Hailo runtime is isolated behind a Flask REST API due to library incompatibilities with the main Python venv
- Each module (`gesture`, `ocr`, `obstacle`) runs in its own isolated virtual environment to avoid dependency conflicts
- `CameraManager` enforces exclusive access to prevent concurrent process conflicts on the Raspberry Pi

---

## 🤖 AI Modules

### 1️⃣ Gesture Recognition

Runs entirely on CPU within `.venv`.

- **MediaPipe** extracts 21 3D hand landmarks per frame
- Additional geometric features computed: inter-landmark distance vectors + angles relative to camera normal
- Custom **MLP classifier** trained on 187-dimensional feature vectors:

```
Input (187 features)
  → Dense(256) + ReLU + BatchNorm + Dropout(0.3)
  → Dense(128) + ReLU + BatchNorm + Dropout(0.3)
  → Dense(64)  + ReLU + BatchNorm + Dropout(0.2)
  → Dense(32)  + ReLU + BatchNorm
  → Dense(6)   + Softmax
Output (6 gesture classes)
```

| Feature Group | Count | Description |
|---|---|---|
| Normalized landmarks | 63 | 21 points × 3 coords (x, y, z), normalized by bounding box |
| Connection vectors | 84 | 21 connections × 4 values (dx, dy, dz, magnitude) |
| Angles | 40 | 5 fingers × 4 segments × 2 angles (horizontal/vertical) |

- **Model size:** ~500KB (`.tflite`)
- **Inference latency:** ~20ms on Raspberry Pi 5
- **Accuracy:** 86.5% (600-gesture test set)
- Privacy-by-design: inference on geometric vectors, no images stored or transmitted

---

### 2️⃣ OCR (PaddleOCR)

Runs in isolated `.venv-ocr` environment.

- Client issues HTTP POST to Flask OCR endpoint
- Server pre-processes frame to RGB tensor → PaddleOCR detection + recognition on Hailo NPU
- **SymSpell** spell correction applied before returning bounding boxes and confidence scores as JSON
- Forward compatibility ensured with parallel pre/post-processing functions for alternative Hailo firmware versions
- **Accuracy:** 76% under standard framing conditions (drops with poor framing — see Troubleshooting)

---

### 3️⃣ Obstacle Detection (YOLO)

Uses a **YOLOv8** `.hef` model deployed on the Hailo NPU.

- Minimal post-processing overhead due to YOLO's structured bounding box outputs
- **Temporal buffer** suppresses redundant consecutive alerts
- **Adaptive suppression**: mutes repeated detections of the same object class after 5 occurrences within 1 minute, resuming when the system infers the user has left the area
- **Accuracy:** 86% (500-object test set in clear conditions)

---

## 🔧 Hardware

### Bill of Materials (PoC — ~S$565.95)

| Component | Role | Power (W) |
|---|---|---|
| Raspberry Pi 5 (8GB) | Main processor | 3.39 |
| Hailo AI HAT | Neural Processing Unit (NPU) | 2.50 |
| Camera Module 3 | Image capture (CSI interface) | 0.66–0.83 |
| Active Cooler | Thermal management | 0.75 |
| MAX98357A | I2S Class-D audio amplifier | ~0 |
| 8Ω 3W oval speaker | Audio output | ~0 |
| 2× 5000mAh Li-ion (3.7V, parallel) | Battery (10Ah total) | — |
| IP5356 SoC | Power management (5V stable output) | — |
| 2× push buttons | On/Off + AI mode selection (GPIO) | — |

**Total system power:** 7.30–7.47W  
**Continuous runtime (OCR mode):** 6.5h (empirically validated)  
**Standby power (EEPROM POWER_OFF_ON_HALT):** 0.01W

### Electrical Safety

The IP5356 SoC integrates:
- Over-voltage protection (OVP) and under-voltage lockout (UVLO)
- Over-current protection (OCP) — short-circuit detection in 150–200µs
- Over-temperature protection (OTP) and ESD protection (4kV)
- Battery overcharge / overdischarge protection + NTC thermistor

The Raspberry Pi 5 manages voltage/current/temperature via DA9091 PMIC (BCM2712 chip), with soft thermal throttling at 80°C and hard throttling at 85°C. AWG16 wiring is used for power transmission under peak AI load.

---

## 💻 Software Stack

### Libraries & Frameworks

| Layer | Library / Tool | Purpose |
|---|---|---|
| AI / Vision | **MediaPipe** | Hand landmark extraction |
| AI / Vision | **PaddleOCR** | Text detection and recognition |
| AI / Vision | **YOLOv8 (Ultralytics)** | Obstacle detection |
| ML | **TensorFlow Lite** | On-device gesture inference |
| NLP | **SymSpell** | OCR spell correction |
| Backend | **Flask** | Internal REST API microservices |
| Mobile | **Flutter / Dart** | Companion app (HTTP control) |
| Camera | **libcamera / V4L2** | Camera capture stack |
| Audio | **espeak** | Text-to-speech output |
| Audio | **amixer** | Volume control |

### Python Virtual Environments

| venv | Modules |
|---|---|
| `.venv` | Main pipeline, gesture recognition, MediaPipe, Flask |
| `.venv-ocr` | PaddleOCR, OCR client wrappers |
| Hailo runtime | Isolated inside Flask server (`.hef` format incompatibility) |

---

## 🚀 Installation & Setup

### Prerequisites

```bash
# Raspberry Pi 5 — install system dependencies
sudo apt update
sudo apt install -y libcamera-dev espeak python3-venv python3-pip cmake
```

### Clone Repository

```bash
git clone https://github.com/your-org/luminai.git
cd luminai
```

### Create Virtual Environments

```bash
# Main environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate

# OCR environment
python3 -m venv .venv-ocr
source .venv-ocr/bin/activate
pip install -r requirements-ocr.txt
deactivate
```

### Hailo HAT Setup

```bash
# Install Hailo runtime (follow official Hailo documentation)
# Place .hef model files in models/
# ├── models/ocr_model.hef
# ├── models/yolo_obstacle.hef
```

### Gesture Model Training (Optional — pre-trained model included)

```bash
source .venv/bin/activate

# Collect training data
python3 src/data_builder.py

# Train model
python3 src/train_gesture_model.py \
  --data data/training_data.json \
  --model model/gesture_model.h5 \
  --tflite model/gesture_model.tflite \
  --epochs 100
```

---

## ▶️ Usage

### Start the Device

```bash
# Launch Flask server (Hailo context)
source .venv/bin/activate
python3 flask_server.py &

# Launch main pipeline
python3 main.py
```

### Mode Selection

- **GPIO pin 17 LOW** → Pointing mode (Gesture Recognition + Obstacle Detection)
- **GPIO pin 17 HIGH** → Reading mode (OCR pipeline)

Physical buttons:
- **Button 1** → On/Off
- **Button 2** → AI mode toggle

### Companion App (Flutter)

Connect to the same Wi-Fi network and open the app. Available controls:

| Endpoint | Action |
|---|---|
| `/status` | View system status (language, volume, camera state) |
| `/language` | Change TTS language |
| `/volume` | Adjust audio volume |
| `/startcam` | Start camera and launch `main.py` |
| `/stopcam` | Stop camera and terminate pipeline |

---

## 📊 Performance & Metrics

| Module | Accuracy | Test Set | Notes |
|---|---|---|---|
| Gesture Recognition | 86.5% | 600 gestures | MLP + MediaPipe |
| Obstacle Detection (YOLO) | 86.0% | 500 objects | Clear conditions |
| OCR (PaddleOCR) | 76.0% | Standard framing | Drops with poor framing |

**End-to-end latency:** ~240ms  
**CPU utilisation during inference (Hailo offload):** <26%  
**CPU utilisation without Hailo (CPU-only):** ~100% (all 4 cores saturated, ~10s latency — infeasible)

---

## 🔩 Mechanical Design

The enclosure is designed around two principles: **ergonomics** and **heat management**.

- **Form factor:** Over-the-shoulder neck-wearable
- **Material:** TPU 95A HF (Young's modulus 7.4–9.8 MPa, melting point 183°C, odourless, skin-compatible)
- **Prototype method:** FDM 3D printing (Ultimaker / Bambu Lab)
- **Thermal features:** Ventilation slits on computational and battery enclosures for passive heat dissipation
- **Active cooling:** Active cooler reduces operating temperature by up to 20°C, preventing thermal throttling under heavy AI load

Component distribution:
- **Cervical area:** Raspberry Pi 5 + Hailo AI HAT (heavier compute units)
- **Lateral sides:** Symmetric battery placement for balanced weight distribution
- **Front-facing:** Camera Module 3

Production roadmap transitions to **injection-molded TPU** with a **custom PCB** (ARM processor + dedicated NPU) replacing the Raspberry Pi 5 development board.

---

## 📱 Companion App

The Flutter companion app provides wireless remote control over HTTP.

**Features:**
- Real-time system status monitoring
- Language selection for TTS output
- Volume adjustment
- Remote camera start/stop
- Network resilience with timeout handling

**Requirements:**
- Flutter SDK 3.x+
- Device on same local network as Raspberry Pi

```bash
# Run Flutter app (development)
cd companion_app/
flutter run
```

---

## 🐛 Troubleshooting

**Camera not found**
```bash
# Check available devices
ls /dev/video*
# Install libcamera if missing
sudo apt install -y libcamera-dev
```

**Hailo model not loading**
```bash
# Ensure .hef files are in models/
# Check Hailo runtime version matches firmware
hailortcli fw-control identify
```

**Low OCR accuracy**
- Ensure text is well-framed and well-lit
- Hold camera steady (~30cm from text)
- OCR accuracy drops significantly with poor framing — this is a known limitation

**Low gesture accuracy**
```bash
# Collect more training data and retrain
python3 src/data_builder.py
python3 src/train_gesture_model.py --epochs 200
```

**Audio not working**
```bash
# Check espeak
espeak "test"
# Check amixer volume
amixer set Master 80%
```

---

## 👥 Team

| Name | Role |
|---|---|
| Yuhang Weng | Electronic engineer, CAD designer |
| Qide Zhengzhao | Electronic engineer, Software engineer |
| Alejandro López Lamata | AI engineer, Render design |
| Carmen Rodríguez García | CAD design, 3D printing, Materials engineer |
| Nathaniel Teophilus | Team coordination, Budget control |
| Lai Shi Hao (Mark) | CAD design, 3D printing, Materials engineer |
| Lim Suyi (Celine) | IP regulation, Business plan |
| Sharleen Goh | IP regulation, Business plan |

**Supervisors:** Ong Chi Wei, Song Juha, Park Seung Min  
**Course:** BG4102 Medical Device Design — NTU School of Chemistry, Chemical Engineering and Biotechnology (2026)

---

*LuminAI — Empowering independence for the visually impaired. 🌟*
