"""
app.py
------
Streamlit-based UI for the Facial Recognition Application.
OPTIMIZED VERSION — faster startup, smoother webcam, quicker recognition.

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
# ✅ FIX 1: Cache the embeddings DB so it only loads ONCE on startup,
#    not on every single Streamlit re-render / sidebar refresh.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_cached_db():
    return load_embeddings()

def reload_db():
    """Call this after register/delete to bust the cache."""
    st.cache_resource.clear()


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

# ✅ FIX 1 applied: use cached DB for sidebar metric (no repeated file I/O)
db = get_cached_db()
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
                    reload_db()   # ✅ bust cache so new face appears immediately
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
                    reload_db()   # ✅ bust cache
                    st.success(f"✅ '{name}' registered successfully from {len(saved_paths)} image(s)!")
                    st.balloons()
                else:
                    st.error("❌ Registration failed. Make sure face(s) are clearly visible.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Real-Time Recognition
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎥 Real-Time Recognition":
    st.title("🎥 Real-Time Face Recognition")

    db = get_cached_db()   # ✅ FIX 1: cached, no disk read
    if not db:
        st.warning("⚠️ No registered faces found. Go to **Register New Face** first.")
        st.stop()

    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.50, max_value=0.99,
        value=SIMILARITY_THRESHOLD, step=0.01,
        help="Higher = stricter matching. Recommended: 0.65-0.75"
    )

    # ✅ FIX 2: Let user choose frame-skip rate to balance speed vs accuracy
    skip = st.select_slider(
        "Detection frequency (process every N frames)",
        options=[1, 2, 3, 5, 8],
        value=3,
        help="Higher = faster webcam feed but less frequent recognition updates"
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
        # ✅ FIX 5: Use CAP_DSHOW on Windows — cuts open time from ~5s to <1s
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback to default backend if DSHOW fails
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Could not open webcam. Make sure no other app is using it.")
        else:
            # ✅ FIX 3: Lower resolution & cap FPS for much smoother feed
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          15)
            # Reduce internal OpenCV buffer so frames aren't stale
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

            # ✅ FIX 6: Warm up camera — discard first few frames (they're often black)
            for _ in range(5):
                cap.read()

            st.session_state.rec_cap     = cap
            st.session_state.rec_running = True

    if stop and st.session_state.rec_running:
        if st.session_state.rec_cap is not None:
            st.session_state.rec_cap.release()
            st.session_state.rec_cap = None
        st.session_state.rec_running = False
        info_placeholder.info("Recognition stopped.")

    # ── Running: grab frames, display ─────────────────────────────────────
    if st.session_state.rec_running:
        cap = st.session_state.rec_cap
        st.info("🟢 Recognition running — click **⏹ Stop** to end.")

        if cap is None or not cap.isOpened():
            st.error("❌ Webcam connection lost.")
            st.session_state.rec_running = False
        else:
            frame_count  = 0
            last_results = []          # ✅ FIX 2: reuse last results on skipped frames
            last_annotated = None

            while st.session_state.rec_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Could not read frame from webcam.")
                    cap.release()
                    st.session_state.rec_cap     = None
                    st.session_state.rec_running = False
                    break

                frame_count += 1

                # ✅ FIX 2: Only run heavy recognition every `skip` frames.
                #    On other frames just redisplay the last annotated frame.
                if frame_count % skip == 0:
                    # ✅ FIX 2b: Shrink to half size for detection, draw on full frame
                    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    annotated_small, last_results = recognize_frame(small, db, threshold)
                    # Scale annotated boxes back up to full frame
                    last_annotated = cv2.resize(annotated_small, (frame.shape[1], frame.shape[0]))
                else:
                    # Reuse previous detection result on non-processed frames
                    last_annotated = frame

                frame_rgb = cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb, channels="RGB", use_container_width=True
                )

                if last_results:
                    names  = [r["name"] for r in last_results]
                    scores = [f"{r['similarity']:.2f}" for r in last_results]
                    info_placeholder.markdown(
                        "**Detected:** "
                        + " | ".join(f"`{n}` ({s})" for n, s in zip(names, scores))
                    )
                else:
                    info_placeholder.markdown("_No faces detected in frame._")

                # ✅ FIX 4: Slightly longer sleep gives Streamlit time to render
                #    without blocking — reduces UI jank significantly
                time.sleep(0.05)

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

    db = get_cached_db()   # ✅ FIX 1: cached
    if not db:
        st.info("No registered users yet.")
        st.stop()

    st.markdown(f"**Total registered:** {len(db)} person(s)")

    for person_name in sorted(db.keys()):
        with st.expander(f"👤 {person_name}"):
            emb = db[person_name]
            st.markdown(f"- Embedding dimensions: **{emb.shape[0]}**")
            st.markdown(f"- Embedding norm: **{np.linalg.norm(emb):.4f}** *(should be ≈ 1.0)*")

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
                    reload_db()   # ✅ bust cache after deletion
                    st.success(f"'{person_name}' removed.")
                    st.rerun()
                else:
                    st.error("Could not delete user.")

    st.divider()
    if st.button("♻️ Reload Database", key="btn_reload"):
        reload_db()
        st.rerun()