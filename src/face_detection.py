"""
face_detection.py
-----------------
Face Detection Module using OpenCV's Haar Cascade or DNN-based detector.

Deep Learning Workflow:
  - A pre-trained deep neural network (or Haar Cascade) scans the image
    at multiple scales and locations to locate faces.
  - Returns bounding boxes (x, y, w, h) for each detected face.
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────
# Option A: Haar Cascade (fast, CPU-friendly)
# ─────────────────────────────────────────────
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_haar_detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


def detect_faces_haar(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect faces using Haar Cascade classifier.

    Args:
        frame: BGR image (H x W x 3) as numpy array.

    Returns:
        List of tuples (x, y, w, h) for each detected face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _haar_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,   # image is scaled down by 10% each step
        minNeighbors=5,    # minimum rectangles to consider a face
        minSize=(60, 60),  # smallest detectable face size
    )
    if len(faces) == 0:
        return []
    return [(x, y, w, h) for (x, y, w, h) in faces]


# ─────────────────────────────────────────────
# Main detect function (used throughout app)
# ─────────────────────────────────────────────
def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Primary face detection entry point.

    Args:
        frame: BGR image as numpy array.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    return detect_faces_haar(frame)


def draw_face_boxes(frame: np.ndarray,
                    faces: list[tuple[int, int, int, int]],
                    labels: list[str] | None = None,
                    color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw bounding boxes and optional labels on an image frame.

    Args:
        frame:  BGR image.
        faces:  List of (x, y, w, h) bounding boxes.
        labels: Optional list of label strings (one per face).
        color:  Box color in BGR, default green.

    Returns:
        Annotated image frame.
    """
    output = frame.copy()
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        if labels and idx < len(labels):
            label = labels[idx]
            # Background rectangle for text readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(output, (x, y - th - 12), (x + tw + 4, y), color, -1)
            cv2.putText(
                output, label,
                (x + 2, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 0), 2, cv2.LINE_AA
            )
    return output


def crop_face(frame: np.ndarray,
              box: tuple[int, int, int, int],
              margin: int = 20) -> np.ndarray:
    """
    Crop a face region from the image with optional margin.

    Args:
        frame:  BGR image.
        box:    (x, y, w, h) bounding box.
        margin: Pixels of padding around the face. Default 20.

    Returns:
        Cropped face image (BGR).
    """
    x, y, w, h = box
    H, W = frame.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(W, x + w + margin)
    y2 = min(H, y + h + margin)
    return frame[y1:y2, x1:x2]
