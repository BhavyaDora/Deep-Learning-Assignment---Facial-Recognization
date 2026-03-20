# 🔍 Facial Recognition Application

A complete end-to-end **Deep Learning** facial recognition system built with **FaceNet512** (via DeepFace), **OpenCV**, and **Streamlit**. Suitable for a university AI/ML submission.

---

## 📐 Project Overview

This application can:
- 🧠 **Detect** faces in webcam feeds or uploaded images
- 📐 **Generate** 512-dimensional face embeddings using FaceNet512
- 🗃️ **Store** embeddings with associated names in a local database
- 🔍 **Recognise** stored persons in real time using cosine similarity
- 🖥️ **Display** results through a Streamlit dashboard

---

## 🧠 Deep Learning Model Used

| Component | Technology |
|---|---|
| Face Detection | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Face Embedding | **FaceNet512** via [DeepFace](https://github.com/serengil/deepface) |
| Similarity Metric | Cosine Similarity (threshold ≥ 0.70) |
| Framework | TensorFlow / Keras (via DeepFace) |

### Why FaceNet512?
FaceNet maps a face image to a 512-dimensional vector (embedding) such that embeddings of the **same person** cluster close together, while embeddings of **different people** are far apart. This property is achieved through **triplet loss training** on millions of face pairs.

---

## 🗂️ Project Structure

```
facial-recognition-app/
│
├── dataset/              ← Stored face image samples per person
├── embeddings/           ← embeddings.pkl (name → vector database)
├── models/               ← (reserved for custom model weights)
│
├── src/
│   ├── __init__.py
│   ├── face_detection.py   ← Haar Cascade face detection + bounding boxes
│   ├── face_embedding.py   ← FaceNet512 embedding generation + similarity
│   ├── register_face.py    ← Webcam/image capture, embedding storage
│   └── recognize_face.py   ← Real-time recognition + cosine matching
│
├── app.py                ← Streamlit multi-page application
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python **3.9 – 3.11** (TensorFlow compatible range)
- A working **webcam**
- At least 4 GB RAM (FaceNet512 downloads ~88 MB weights on first run)

### Steps

```bash
# 1. Clone / navigate to the project folder
cd "c:\D drive\DL project\facial-recognition-app"

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** On first run, DeepFace will automatically download the FaceNet512 model weights (~88 MB) from the internet.

---

## 🚀 How to Run

```bash
# From the project root
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Non-UI Real-Time Recognition (terminal only)

```bash
python -m src.recognize_face
```
Press **`q`** to quit.

---

## 📱 Application Pages

| Page | Description |
|---|---|
| 📋 Register New Face | Capture webcam frames **or** upload images to enrol a person |
| 🎥 Real-Time Recognition | Live webcam stream with bounding box + name overlay |
| 👥 Registered Users | View, inspect, and delete enrolled persons |

---

## 🔬 Deep Learning Workflow

```
Webcam Frame
     │
     ▼
Face Detection (Haar Cascade CNN)
     │  → bounding box (x, y, w, h)
     ▼
Face Crop + Alignment
     │
     ▼
FaceNet512 CNN  ← pretrained on ~100M face pairs
     │  → 512-D embedding vector
     ▼
Cosine Similarity vs Database
     │  sim ≥ 0.70 → MATCH
     │  sim < 0.70 → Unknown
     ▼
Annotated Frame displayed in Streamlit
```

---

## 🛠️ Configuration

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `MODEL_NAME` | `face_embedding.py` | `"Facenet512"` | Change embedding model |
| `SIMILARITY_THRESHOLD` | `recognize_face.py` | `0.70` | Recognition strictness |
| `CAPTURE_FRAMES` | `register_face.py` | `10` | Frames captured per person |

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `deepface` | 0.0.93+ | FaceNet512 / ArcFace inference |
| `tensorflow` | 2.12+ | DL backend |
| `opencv-python` | 4.8+ | Webcam, image I/O, Haar Cascade |
| `streamlit` | 1.32+ | Web UI |
| `numpy` | 1.24+ | Numerical operations |
| `scikit-learn` | 1.3+ | Utility functions |

---

## 📝 Notes

- The **first recognition** may be slow (15-30 s) while FaceNet512 loads.
- For best accuracy, register **10+ varied images** per person.
- To switch from FaceNet512 to ArcFace, change `MODEL_NAME = "ArcFace"` in `face_embedding.py`.
- All embeddings are stored locally in `embeddings/embeddings.pkl` — no cloud required.

---

## 👨‍💻 Author

University AI/ML Deep Learning Project — 2026
Jothik:     23STUCHH011556
Bhavya:     23STUCHH010495
