"""
app.py
------
Streamlit-based UI for the Facial Recognition Application.

Pages:
  📋 Page 1 — Register New Face  : Capture from webcam or upload images.
  🎥 Page 2 — Real-Time Recognition: Run recognition on live webcam feed.
  👥 Page 3 — View Registered Users: Manage the embedding database.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Ensure src/ is importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_detection  import detect_faces, draw_face_boxes, crop_face
from src.face_embedding  import get_embedding_from_face, cosine_similarity
from src.register_face   import (
    register_face_from_images,
    register_face_from_webcam,
    load_embeddings,
    save_embeddings,
    list_registered_users,
    delete_user,
    DATASET_DIR,
)
from src.recognize_face  import recognize_frame, SIMILARITY_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Facial Recognition App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark-mode look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: #0f111a;
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
  }
  [data-testid="stSidebar"] {
    background: #161b2e;
    border-right: 1px solid #2d3a5e;
  }
  /* ── Headings ── */
  h1, h2, h3 { color: #7dd3fc; }
  /* ── Buttons ── */
  div.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #818cf8);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    transition: opacity .2s;
  }
  div.stButton > button:hover { opacity: 0.85; }
  /* ── Cards ── */
  .card {
    background: #1e2640;
    border: 1px solid #2d3a5e;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
  }
  /* ── Badge ── */
  .badge-green {
    background:#16a34a; color:#fff; padding:2px 10px;
    border-radius:99px; font-size:0.75rem;
  }
  .badge-red {
    background:#dc2626; color:#fff; padding:2px 10px;
    border-radius:99px; font-size:0.75rem;
  }
  /* ── Progress bar ── */
  div[data-testid="stProgress"] > div { background: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 FaceID App")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📋 Register New Face", "🎥 Real-Time Recognition", "👥 Registered Users"],
)
st.sidebar.markdown("---")
db = load_embeddings()
st.sidebar.metric("Registered Persons", len(db))
st.sidebar.caption("Built with DeepFace · FaceNet512 · OpenCV · Streamlit")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Register New Face
# ═════════════════════════════════════════════════════════════════════════════
if page == "📋 Register New Face":
    st.title("📋 Register New Face")
    st.markdown("Capture webcam frames **or** upload images to register a person.")

    name = st.text_input("👤 Person Name", placeholder="e.g. Alice Johnson")

    tab_webcam, tab_upload = st.tabs(["📷 Webcam Capture", "🖼️ Upload Images"])

    # ── Tab 1: Webcam ──────────────────────────────────────────────────────
    with tab_webcam:
        num_frames = st.slider("Frames to capture", 5, 30, 10, 1)
        st.info("💡 Face the camera directly and ensure good lighting.")

        if st.button("▶ Start Capture & Register", key="btn_webcam"):
            if not name.strip():
                st.error("Please enter a name before capturing.")
            else:
                progress_bar = st.progress(0)
                status_text  = st.empty()
                frame_holder = st.empty()

                def progress_cb(idx, total, frame_bgr):
                    progress_bar.progress((idx + 1) / total)
                    status_text.write(f"Captured frame {idx + 1}/{total}")
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_holder.image(frame_rgb, channels="RGB", width=400)

                with st.spinner("Registering — please wait…"):
                    ok, msg = register_face_from_webcam(
                        name, num_frames=num_frames, progress_callback=progress_cb
                    )

                progress_bar.empty()
                status_text.empty()
                frame_holder.empty()

                if ok:
                    st.success(f"✅ {msg}")
                    st.balloons()
                else:
                    st.error(f"❌ {msg}")

    # ── Tab 2: Upload ──────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload face images (JPEG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploaded:
            cols = st.columns(min(len(uploaded), 5))
            for col, f in zip(cols, uploaded):
                col.image(f, use_container_width=True, caption=f.name)

        if st.button("✔ Register from Uploaded Images", key="btn_upload"):
            if not name.strip():
                st.error("Please enter a name.")
            elif not uploaded:
                st.error("Please upload at least one image.")
            else:
                # Save uploads to a temp location then pass paths
                tmp_dir = DATASET_DIR / "_tmp_upload"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                saved_paths = []
                for f in uploaded:
                    p = tmp_dir / f.name
                    p.write_bytes(f.read())
                    saved_paths.append(p)

                with st.spinner("Generating embeddings…"):
                    ok = register_face_from_images(name, saved_paths)

                if ok:
                    st.success(f"✅ '{name}' registered successfully from {len(saved_paths)} image(s)!")
                    st.balloons()
                else:
                    st.error("❌ Registration failed. Make sure face(s) are clearly visible.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Real-Time Recognition
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎥 Real-Time Recognition":
    st.title("🎥 Real-Time Face Recognition")

    db = load_embeddings()
    if not db:
        st.warning("⚠️ No registered faces found. Go to **Register New Face** first.")
        st.stop()

    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.50, max_value=0.99,
        value=SIMILARITY_THRESHOLD, step=0.01,
        help="Higher = stricter matching. Recommended: 0.65-0.75"
    )

    col_start, col_stop = st.columns([1, 1])
    start = col_start.button("▶ Start Recognition", key="btn_start")
    stop  = col_stop.button("⏹ Stop", key="btn_stop")

    frame_placeholder = st.empty()
    info_placeholder  = st.empty()

    # ── Session state initialisation ──────────────────────────────────────
    if "rec_running" not in st.session_state:
        st.session_state.rec_running = False
    if "rec_cap" not in st.session_state:
        st.session_state.rec_cap = None

    # ── Button handlers ───────────────────────────────────────────────────
    if start and not st.session_state.rec_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Make sure no other app is using it.")
        else:
            st.session_state.rec_cap     = cap
            st.session_state.rec_running = True

    if stop and st.session_state.rec_running:
        if st.session_state.rec_cap is not None:
            st.session_state.rec_cap.release()
            st.session_state.rec_cap = None
        st.session_state.rec_running = False
        info_placeholder.info("Recognition stopped.")

    # ── Running: grab one frame, display ───────────────────────────────────
    if st.session_state.rec_running:
        cap = st.session_state.rec_cap
        st.info("🟢 Recognition running — click **⏹ Stop** to end.")

        if cap is None or not cap.isOpened():
            st.error("❌ Webcam connection lost.")
            st.session_state.rec_running = False
        else:
            while st.session_state.rec_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Could not read frame from webcam.")
                    cap.release()
                    st.session_state.rec_cap     = None
                    st.session_state.rec_running = False
                    break
                
                annotated, results = recognize_frame(frame, db, threshold)
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb, channels="RGB", use_container_width=True
                )

                if results:
                    names  = [r["name"] for r in results]
                    scores = [f"{r['similarity']:.2f}" for r in results]
                    info_placeholder.markdown(
                        "**Detected:** "
                        + " | ".join(f"`{n}` ({s})" for n, s in zip(names, scores))
                    )
                else:
                    info_placeholder.markdown("_No faces detected in frame._")

                # Limit UI update rate to ~30 FPS
                time.sleep(0.03)
    else:
        if not stop:
            st.markdown(
                '<div class="card">Click <strong>▶ Start Recognition</strong> to begin live webcam recognition.</div>',
                unsafe_allow_html=True,
            )

    # ── Single-image test ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🖼️ Test on a Single Image")
    test_img = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"], key="test_img")
    if test_img and st.button("🔍 Recognise", key="btn_recognise"):
        img_bytes  = np.frombuffer(test_img.read(), np.uint8)
        img_bgr    = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        annotated, results = recognize_frame(img_bgr, db, threshold)
        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, channels="RGB", caption="Recognition Result", use_container_width=True)
        for r in results:
            badge = "badge-green" if r["name"] != "Unknown" else "badge-red"
            st.markdown(
                f'<span class="{badge}">{r["name"]}</span> &nbsp; similarity: **{r["similarity"]:.4f}**',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — View Registered Users
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Registered Users":
    st.title("👥 Registered Users")

    db = load_embeddings()
    if not db:
        st.info("No registered users yet.")
        st.stop()

    st.markdown(f"**Total registered:** {len(db)} person(s)")

    for person_name in sorted(db.keys()):
        with st.expander(f"👤 {person_name}"):
            emb = db[person_name]
            st.markdown(f"- Embedding dimensions: **{emb.shape[0]}**")
            st.markdown(f"- Embedding norm: **{np.linalg.norm(emb):.4f}** *(should be ≈ 1.0)*")

            # Show stored face images if present
            person_dir = DATASET_DIR / person_name
            face_images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            if face_images:
                st.markdown(f"- Stored samples: **{len(face_images)}**")
                cols = st.columns(min(len(face_images), 6))
                for col, img_path in zip(cols, face_images[:6]):
                    col.image(str(img_path), use_container_width=True)

            if st.button(f"🗑️ Delete {person_name}", key=f"del_{person_name}"):
                ok = delete_user(person_name)
                if ok:
                    st.success(f"'{person_name}' removed.")
                    st.rerun()
                else:
                    st.error("Could not delete user.")

    st.divider()
    if st.button("♻️ Reload Database", key="btn_reload"):
        st.rerun()
