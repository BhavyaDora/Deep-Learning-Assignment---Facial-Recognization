"""
recognize_face.py
-----------------
Face Recognition Module

Deep Learning Workflow:
  1. Detect faces in the current frame (OpenCV Haar/DNN).
  2. For each face, generate a FaceNet512 embedding.
  3. Compare the embedding with all stored embeddings using COSINE SIMILARITY.
  4. If the best match similarity exceeds the threshold → recognised person.
  5. Otherwise label the face "Unknown".

Cosine Similarity:
  sim(A, B) = (A · B) / (||A|| × ||B||)
  Both A,B are L2-normalised so sim(A,B) = A · B  ∈ [-1, 1].
  Threshold ≥ 0.70 is a reasonable default for FaceNet512.
"""

import cv2
import numpy as np
from pathlib import Path

from src.face_detection  import detect_faces, draw_face_boxes, crop_face
from src.face_embedding  import get_embedding_from_face, cosine_similarity
from src.register_face   import load_embeddings

# ── Tuning ─────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.70   # raise to be stricter, lower to be more lenient


# ─────────────────────────────────────────────────────────────────────────────
# Core recognition logic
# ─────────────────────────────────────────────────────────────────────────────

def identify_face(embedding: np.ndarray,
                  db: dict[str, np.ndarray],
                  threshold: float = SIMILARITY_THRESHOLD
                  ) -> tuple[str, float]:
    """
    Compare a query embedding against the stored database and return
    the best matching identity.

    Args:
        embedding: Query face embedding (1-D, L2-normalised).
        db:        {name: embedding} dictionary loaded from disk.
        threshold: Minimum cosine similarity to accept a match.

    Returns:
        (name, similarity) — name is "Unknown" when no match exceeds threshold.
    """
    best_name  = "Unknown"
    best_score = -1.0

    for name, stored_emb in db.items():
        score = cosine_similarity(embedding, stored_emb)
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score < threshold:
        return "Unknown", best_score

    return best_name, best_score


def recognize_frame(frame: np.ndarray,
                    db: dict[str, np.ndarray],
                    threshold: float = SIMILARITY_THRESHOLD
                    ) -> tuple[np.ndarray, list[dict]]:
    """
    Detect and recognise all faces in a single image frame.

    Args:
        frame:     BGR image as numpy array.
        db:        Embedding database {name: embedding}.
        threshold: Cosine similarity threshold.

    Returns:
        (annotated_frame, results)
        results is a list of dicts:
          {
            "box":        (x, y, w, h),
            "name":       str,
            "similarity": float,
          }
    """
    results = []
    faces   = detect_faces(frame)

    if not faces or not db:
        return frame, results

    labels = []
    for box in faces:
        crop    = crop_face(frame, box)
        emb     = get_embedding_from_face(crop)

        if emb is None:
            name, score = "Unknown", 0.0
        else:
            name, score = identify_face(emb, db, threshold)

        label = f"{name} ({score:.2f})"
        labels.append(label)

        results.append({
            "box":        box,
            "name":       name,
            "similarity": score,
        })

    color_map = {
        True:  (0, 220, 0),   # green  — recognised
        False: (0, 0, 220),   # red    — unknown
    }
    colors = [color_map[r["name"] != "Unknown"] for r in results]

    # Draw all boxes with a single colour toggle is not available in helper,
    # so we annotate manually:
    annotated = frame.copy()
    for (x, y, w, h), label, color in zip(faces, labels, colors):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x, y - th - 14), (x + tw + 6, y), color, -1)
        cv2.putText(annotated, label, (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated, results


# ─────────────────────────────────────────────────────────────────────────────
# Real-time webcam recognition (standalone, non-Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def run_realtime_recognition(threshold: float = SIMILARITY_THRESHOLD,
                              camera_index: int = 0) -> None:
    """
    Open the webcam and run continuous face recognition.
    Press 'q' to quit.

    Args:
        threshold:    Cosine similarity threshold.
        camera_index: OpenCV camera index (default 0).
    """
    db = load_embeddings()
    if not db:
        print("[WARNING] No embeddings found. Register faces first.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, results = recognize_frame(frame, db, threshold)

        # Overlay FPS / info text
        info = f"Registered: {len(db)}  |  Faces: {len(results)}  |  [q] Quit"
        cv2.putText(annotated, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Facial Recognition — Press Q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_recognition()
