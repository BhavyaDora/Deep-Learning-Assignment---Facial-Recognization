"""
register_face.py
----------------
Face Registration Module

Workflow:
  1. Capture N frames from webcam (or load an existing image).
  2. Detect the face in each frame.
  3. Generate a FaceNet512 embedding for each detected face.
  4. Average the embeddings for robustness.
  5. Persist:
       dataset/<name>/  → raw face images
       embeddings/embeddings.pkl → {name: embedding_vector}
"""

import cv2
import os
import pickle
import time
from pathlib import Path

import numpy as np

from src.face_detection import detect_faces, crop_face
from src.face_embedding import get_embedding_from_face

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATASET_DIR    = BASE_DIR / "dataset"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.pkl"

# ── Constants ──────────────────────────────────────────────────────────────
CAPTURE_FRAMES  = 10   # number of frames to capture per person during webcam reg
CAPTURE_DELAY   = 0.3  # seconds between frame captures


# ─────────────────────────────────────────────────────────────────────────────
# Embedding storage helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings() -> dict[str, np.ndarray]:
    """
    Load the persisted embedding database from disk.

    Returns:
        Dict mapping person_name → embedding vector (np.ndarray).
        Returns empty dict if the file doesn't exist yet.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not EMBEDDINGS_FILE.exists():
        return {}
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def save_embeddings(db: dict[str, np.ndarray]) -> None:
    """Persist the embedding database to disk."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(db, f)
    print(f"[INFO] Embeddings saved -> {EMBEDDINGS_FILE}")


def list_registered_users() -> list[str]:
    """Return a list of all registered person names."""
    db = load_embeddings()
    return sorted(db.keys())


def delete_user(name: str) -> bool:
    """
    Remove a person from the embedding database.

    Args:
        name: Person's name as stored in the database.

    Returns:
        True if deleted, False if name was not found.
    """
    db = load_embeddings()
    if name not in db:
        return False
    del db[name]
    save_embeddings(db)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Core registration logic
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mean_embedding(face_images: list[np.ndarray]) -> np.ndarray | None:
    """
    Compute the average embedding across multiple face images.

    Averaging multiple samples of the same person stabilises the embedding
    and reduces the effect of noise from pose / lighting variations.

    Args:
        face_images: List of BGR face crop numpy arrays.

    Returns:
        Mean L2-normalised embedding vector or None on failure.
    """
    embeddings = []
    for img in face_images:
        emb = get_embedding_from_face(img)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return None

    mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb


def register_face_from_images(name: str,
                               image_paths: list[str | Path]) -> bool:
    """
    Register a person from a list of image file paths.

    Args:
        name:        Person's name (used as the database key).
        image_paths: List of paths to face images.

    Returns:
        True on success, False on failure.
    """
    name = name.strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        return False

    face_crops = []
    person_dir = DATASET_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARNING] Cannot read image: {path}")
            continue
        faces = detect_faces(img)
        if not faces:
            print(f"[WARNING] No face detected in: {path}")
            continue
        # Use the largest detected face
        largest = max(faces, key=lambda b: b[2] * b[3])
        crop = crop_face(img, largest)
        face_crops.append(crop)
        # Save a copy to dataset folder
        out_path = person_dir / f"img_{idx:03d}.jpg"
        cv2.imwrite(str(out_path), crop)

    if not face_crops:
        print("[ERROR] No usable face images found.")
        return False

    embedding = _compute_mean_embedding(face_crops)
    if embedding is None:
        print("[ERROR] Embedding generation failed.")
        return False

    db = load_embeddings()
    db[name] = embedding
    save_embeddings(db)
    print(f"[INFO] Registered '{name}' with {len(face_crops)} face sample(s).")
    return True


def register_face_from_webcam(name: str,
                               num_frames: int = CAPTURE_FRAMES,
                               progress_callback=None) -> tuple[bool, str]:
    """
    Register a person by capturing frames from the default webcam.

    Args:
        name:              Person's name.
        num_frames:        Number of frames to capture (default 10).
        progress_callback: Optional callable(frame_index, total, frame_bgr)
                           for UI updates during capture.

    Returns:
        (success: bool, message: str)
    """
    name = name.strip()
    if not name:
        return False, "Name cannot be empty."

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Could not open webcam."

    person_dir = DATASET_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    face_crops = []
    captured   = 0
    attempt    = 0
    max_attempts = num_frames * 5  # allow 5× attempts per desired frame

    print(f"[INFO] Capturing {num_frames} frames for '{name}'…")

    while captured < num_frames and attempt < max_attempts:
        ret, frame = cap.read()
        attempt += 1
        if not ret:
            continue

        faces = detect_faces(frame)
        if not faces:
            time.sleep(0.1)
            continue

        # Use the largest face
        largest = max(faces, key=lambda b: b[2] * b[3])
        crop = crop_face(frame, largest)
        face_crops.append(crop)

        out_path = person_dir / f"webcam_{captured:03d}.jpg"
        cv2.imwrite(str(out_path), crop)

        if progress_callback:
            progress_callback(captured, num_frames, frame)

        captured += 1
        time.sleep(CAPTURE_DELAY)

    cap.release()

    if not face_crops:
        return False, "No faces detected during webcam capture."

    embedding = _compute_mean_embedding(face_crops)
    if embedding is None:
        return False, "Embedding generation failed — ensure deepface is installed."

    db = load_embeddings()
    db[name] = embedding
    save_embeddings(db)
    return True, f"Successfully registered '{name}' with {len(face_crops)} sample(s)."
