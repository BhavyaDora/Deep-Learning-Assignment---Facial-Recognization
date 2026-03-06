"""
face_embedding.py
-----------------
Face Embedding Generation Module using DeepFace (wraps FaceNet / ArcFace).

Deep Learning Workflow:
  - A pre-trained convolutional neural network (FaceNet or ArcFace) processes
    a normalized face image and maps it to a high-dimensional embedding vector
    (e.g., 128-D for FaceNet, 512-D for ArcFace).
  - Similar faces produce embeddings that are close in Euclidean / cosine space.
  - We compare embeddings rather than raw pixels, making recognition robust to
    lighting, pose, and expression variations.

Primary library: deepface  (pip install deepface)
Fallback        : raw OpenCV + a lightweight face crop + direct model inference
"""

import cv2
import numpy as np
from pathlib import Path

# ── DeepFace wrapper ──────────────────────────────────────────────────────────
try:
    from deepface import DeepFace as _DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARNING] deepface not installed. Run:  pip install deepface")

# Model used for face representation
# Choices: "Facenet", "Facenet512", "ArcFace", "VGG-Face", "OpenFace", "DeepFace"
MODEL_NAME = "Facenet512"   # 512-D embeddings — good balance of speed & accuracy

# Detector backend used inside deepface (only for single-image API calls)
DETECTOR_BACKEND = "opencv"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_from_face(face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Generate a face embedding vector from a cropped face image.

    The function:
      1. Converts BGR → RGB (models trained on RGB).
      2. Feeds the face through the pre-trained FaceNet512 CNN.
      3. Returns the L2-normalised 512-D embedding vector.

    Args:
        face_bgr: Cropped face region as a BGR numpy array (H x W x 3).

    Returns:
        1-D numpy array (embedding) or None if extraction fails.
    """
    if not DEEPFACE_AVAILABLE:
        raise RuntimeError("deepface is not installed. Run: pip install deepface")

    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        result = _DeepFace.represent(
            img_path=face_rgb,          # accepts numpy array directly
            model_name=MODEL_NAME,
            enforce_detection=False,    # already cropped → skip internal detection
            detector_backend=DETECTOR_BACKEND,
            align=True,                 # align face landmarks before embedding
        )
        embedding = np.array(result[0]["embedding"], dtype=np.float32)
        # L2-normalise so cosine similarity == dot product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    except Exception as exc:
        print(f"[ERROR] Embedding extraction failed: {exc}")
        return None


def get_embedding_from_path(image_path: str | Path) -> np.ndarray | None:
    """
    Generate an embedding from an image file on disk.

    Args:
        image_path: Path to a JPEG/PNG image containing one face.

    Returns:
        1-D numpy array (embedding) or None on failure.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None
    return get_embedding_from_face(img)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Both vectors should already be L2-normalised (as returned by this module),
    in which case cosine_similarity == dot product.

    Returns:
        Float in [-1, 1]; higher means more similar.
    """
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two embedding vectors.

    Returns:
        Float; lower means more similar.
    """
    return float(np.linalg.norm(a - b))
