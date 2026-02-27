"""
Smart Detection App
- YOLOv8 object detection (water bottle, person, watch, mobile phone, laptop, plant, book)
- Face recognition from a folder of known person images
- Works on PC and Mobile browsers (accesses device camera via WebRTC)
- Streamlit hosted
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import logging
import sys
from pathlib import Path
import tempfile
import pickle

# ─── Logger Setup ─────────────────────────────────────────────────────────────
def setup_logger():
    """Configure and return the application logger."""
    logger = logging.getLogger("SmartVisionAI")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on Streamlit reruns
    if not logger.handlers:
        # Console handler (visible in terminal / Streamlit Cloud logs)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)

        # File handler — writes to smart_vision.log next to app.py
        log_file = Path(__file__).parent / "smart_vision.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s › %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


logger = setup_logger()
logger.info("=" * 60)
logger.info("Smart Vision AI starting up")
logger.info("=" * 60)


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Vision AI",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

* { font-family: 'Syne', sans-serif; }
code, pre { font-family: 'Space Mono', monospace; }

.stApp {
    background: #f5f6fa;
    color: #1a1a2e;
}
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e0e0f0;
}
h1, h2, h3 { color: #2c2c6c; letter-spacing: -0.02em; }

.detection-box {
    background: #ffffff;
    border: 1px solid #d0d0e8;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.person-badge {
    background: linear-gradient(90deg, #7b2ff7, #f707a8);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-block;
    margin: 3px;
    animation: pulse 2s infinite;
}
.object-badge {
    background: linear-gradient(90deg, #0066ff, #00aaff);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    display: inline-block;
    margin: 3px;
}
@keyframes pulse {
    0%,100%{ box-shadow: 0 0 6px #f707a8; }
    50%{ box-shadow: 0 0 18px #f707a8; }
}
.metric-card {
    background: #ffffff;
    border: 1px solid #d0d0e8;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.metric-num { font-size: 2rem; font-weight: 800; color: #2c2c6c; }
.metric-label { font-size: 0.78rem; color: #8888aa; text-transform: uppercase; letter-spacing: 0.1em; }
.stButton button {
    background: linear-gradient(90deg, #7b2ff7, #0066ff) !important;
    color: black !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: all 0.3s;
}
.stButton button:hover { opacity: 0.85; transform: translateY(-1px); }
.stFileUploader {
    background: #ffffff !important;
    border: 1px dashed #b0b0d0 !important;
    border-radius: 12px !important;
}

/* Fix Streamlit default dark overrides */
.stTextInput input, .stSlider, .stSelectbox {
    background: #ffffff !important;
    color: #1a1a2e !important;
}
[data-testid="stMarkdownContainer"] p { color: #1a1a2e; }
.stTabs [data-baseweb="tab"] { color: #2c2c6c; }
.stTabs [aria-selected="true"] { 
    border-bottom: 2px solid #7b2ff7 !important; 
    color: #7b2ff7 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load YOLO and face recognition models (cached)."""
    logger.info("Loading AI models …")

    yolo = None
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8x.pt")
        logger.info("YOLOv8 model loaded successfully")
        # st.success("✅ YOLOv8 model loaded")
    except ImportError:
        logger.error("ultralytics not installed — YOLOv8 unavailable")
        st.error("Install: pip install ultralytics")
    except Exception as e:
        logger.exception(f"Unexpected error loading YOLO: {e}")

    fr = None
    try:
        import face_recognition as _fr
        fr = _fr
        logger.info("face_recognition module loaded successfully")
        # st.success("✅ Face recognition ready")
    except ImportError:
        logger.warning("face_recognition not installed")
        st.warning("face_recognition not installed. Run: pip install face-recognition")
    except Exception as e:
        logger.exception(f"Unexpected error loading face_recognition: {e}")

    return yolo, fr


# ─── COCO class IDs for target objects ────────────────────────────────────────
TARGET_CLASSES = {
    39: "bottle",
    0:  "person",
    67: "cell phone",
    63: "laptop",
    58: "potted plant",
    # 73: "book",
    # 76: "watch",
}

CLASS_COLORS = {
    "bottle":       (0, 200, 255),
    "person":       (255, 50, 200),
    "cell phone":   (50, 255, 150),
    "laptop":       (255, 200, 0),
    "potted plant": (0, 255, 80),
    # "book":         (200, 100, 255),
    # "watch":        (255, 130, 30),
}


# ─── Face encoding helpers ────────────────────────────────────────────────────
def load_single_face_encoding(img_path: str, fr_module):
    """
    Load one image file and return its face encoding (first face found).
    Returns None if no face is detected or loading fails.
    """
    logger.debug(f"Loading face image: {img_path}")
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"cv2.imread returned None for: {img_path} — file may be corrupt or wrong path")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Failed to read image {img_path}: {e}")
        return None

    try:
        # dlib requires a C-contiguous uint8 array
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

        # upsample=1 helps detect smaller / lower-res faces
        locations = fr_module.face_locations(img_rgb, number_of_times_to_upsample=1, model="hog")
        logger.debug(f"  face_locations found {len(locations)} face(s) in {Path(img_path).name}")

        if not locations:
            logger.warning(f"  No face detected in {Path(img_path).name} — "
                           "ensure the image clearly shows the person's face and is well-lit")
            return None

        # Use known_face_locations keyword to avoid dlib landmark type mismatch
        encodings = fr_module.face_encodings(img_rgb,
                                             known_face_locations=locations,
                                             num_jitters=1)
        if not encodings:
            logger.warning(f"  face_encodings returned empty list for {Path(img_path).name}")
            return None

        logger.info(f"  ✅ Encoding built for {Path(img_path).name}")
        return encodings[0]

    except Exception as e:
        logger.exception(f"  Error encoding face in {Path(img_path).name}: {e}")
        return None


def build_face_database(folder_path: str, fr_module) -> dict:
    """
    Read all images from folder_path, build {display_name: encoding} dict.
    File name (without extension) becomes the display name.
    e.g. 'jaydeep.jpg'  →  'Jaydeep'
    """
    db = {}
    folder = Path(folder_path)
    logger.info(f"Building face database from folder: {folder.resolve()}")

    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [p for p in folder.glob("*") if p.suffix.lower() in supported]
    logger.info(f"Found {len(image_files)} image file(s): {[p.name for p in image_files]}")

    for img_path in image_files:
        name = img_path.stem.replace("_", " ").replace("-", " ").title()
        logger.info(f"Processing '{img_path.name}' → display name: '{name}'")
        enc = load_single_face_encoding(str(img_path), fr_module)
        if enc is not None:
            db[name] = enc
            logger.info(f"  Added '{name}' to face database")
        else:
            logger.warning(f"  Skipped '{name}' — no usable face encoding")

    logger.info(f"Face database ready: {list(db.keys())}")
    return db


# ─── Detection + Recognition ──────────────────────────────────────────────────
def process_frame(frame_bgr, yolo_model, face_db, fr_module, conf_threshold=0.4):
    """
    Run YOLO detection on frame_bgr.
    For 'person' detections, also run face recognition.
    Returns annotated frame + list of detection labels.
    """
    labels_found = []

    if yolo_model is None:
        logger.debug("process_frame: yolo_model is None, skipping")
        return frame_bgr, labels_found

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        results = yolo_model(frame_bgr, conf=conf_threshold, verbose=False)[0]
    except Exception as e:
        logger.error(f"YOLO inference failed: {e}")
        return frame_bgr, labels_found

    detected_classes = [TARGET_CLASSES.get(int(b.cls[0]), "?") for b in results.boxes
                        if int(b.cls[0]) in TARGET_CLASSES]
    if detected_classes:
        logger.debug(f"YOLO detections this frame: {detected_classes}")

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in TARGET_CLASSES:
            continue

        conf = float(box.conf[0])
        label = TARGET_CLASSES[cls_id]
        color = CLASS_COLORS.get(label, (200, 200, 200))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        display_name = label

        # ── Face recognition for person boxes ────────────────────────────────
        if label == "person" and fr_module is not None and face_db:
            logger.debug(f"Running face recognition on person box ({x1},{y1})-({x2},{y2})")
            try:
                # Clamp ROI to frame bounds
                h_frame, w_frame = frame_bgr.shape[:2]
                rx1 = max(0, x1)
                ry1 = max(0, y1)
                rx2 = min(w_frame, x2)
                ry2 = min(h_frame, y2)
                face_roi = frame_rgb[ry1:ry2, rx1:rx2]

                if face_roi.size == 0:
                    logger.warning("Face ROI is empty — bounding box may be outside frame")
                else:
                    roi_h, roi_w = face_roi.shape[:2]
                    logger.debug(f"Face ROI size: {roi_w}x{roi_h}")

                    # Ensure C-contiguous uint8 array — dlib requires this
                    face_roi = np.ascontiguousarray(face_roi, dtype=np.uint8)

                    # upsample=1 helps with smaller face regions
                    locs = fr_module.face_locations(face_roi,
                                                    number_of_times_to_upsample=1,
                                                    model="hog")
                    logger.debug(f"face_locations in ROI: {len(locs)} found")

                    if not locs:
                        logger.debug("No face locations found in ROI, skipping encoding")
                        continue

                    # Pass known_face_locations explicitly so dlib uses the
                    # pre-computed HOG locations (avoids the incompatible
                    # function argument TypeError with raw landmark objects)
                    encs = fr_module.face_encodings(face_roi,
                                                    known_face_locations=locs,
                                                    num_jitters=1)
                    logger.debug(f"face_encodings count: {len(encs)}")

                    for enc in encs:
                        known_encs  = list(face_db.values())
                        known_names = list(face_db.keys())

                        dists   = fr_module.face_distance(known_encs, enc)
                        matches = fr_module.compare_faces(known_encs, enc, tolerance=0.55)

                        logger.debug(f"Face distances: { {n: round(float(d),3) for n,d in zip(known_names, dists)} }")
                        logger.debug(f"Face matches:   { {n: bool(m) for n,m in zip(known_names, matches)} }")

                        if len(dists) > 0:
                            best_idx = int(np.argmin(dists))
                            best_dist = float(dists[best_idx])
                            if matches[best_idx]:
                                display_name = known_names[best_idx]
                                logger.info(f"✅ Recognised: '{display_name}' (distance={best_dist:.3f})")
                                color = CLASS_COLORS["person"]   # keep consistent color
                            else:
                                logger.debug(f"Best match '{known_names[best_idx]}' dist={best_dist:.3f} — "
                                             "below tolerance, labelled as 'person'")

            except Exception as e:
                logger.exception(f"Face recognition error: {e}")

        # ── Draw label tag ────────────────────────────────────────────────────
        tag = f"{display_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_bgr, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)

        labels_found.append(display_name)

    return frame_bgr, labels_found


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    logger.info("main() called — rendering UI")

    st.markdown("""
    <h1 style='text-align:center; font-size:2.5rem; margin-bottom:0;'>
        Smart Vision AI
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # with st.spinner("Loading AI models…"):
    yolo_model, fr_module = load_models()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### <span style='color:#000000;'>⚙️ Settings</span>", unsafe_allow_html=True)
        conf_threshold = 0.40

        # st.markdown("### 👤 Known Persons Folder")
        # st.caption("Put face images in the folder named below. "
        #            "File name = person's name  (e.g. `jaydeep.jpg`).")

        known_folder = "known_persons"

        uploaded_faces = st.file_uploader(
            "Or upload face images here",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="face_uploader",
        )

        face_db: dict = {}

        if fr_module:
            # ── Load from folder ──────────────────────────────────────────────
            if os.path.isdir(known_folder):
                logger.info(f"Loading known persons from folder: {known_folder}")
                face_db = build_face_database(known_folder, fr_module)
                if face_db:
                    st.success(f"✅ {len(face_db)} person(s) loaded from folder")
                    for name in face_db:
                        st.markdown(f"<span class='person-badge'>{name}</span>",
                                    unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Folder found but no faces could be encoded. "
                               "Check the log (smart_vision.log) for details.")
                    logger.warning(f"No faces encoded from folder: {known_folder}")
            else:
                logger.warning(f"Known persons folder not found: '{known_folder}'")
                st.warning(f"Folder '{known_folder}' not found.")

            # ── Load from direct uploads ──────────────────────────────────────
            if uploaded_faces:
                upload_key = tuple(sorted(f.name for f in uploaded_faces))

                if st.session_state.get("face_db_upload_key") != upload_key:
                    logger.info(f"Processing {len(uploaded_faces)} uploaded face image(s)")
                    
                    # Clear previous temp dir if exists
                    old_tmp = st.session_state.get("face_db_tmp_dir")
                    if old_tmp and os.path.isdir(old_tmp):
                        import shutil
                        shutil.rmtree(old_tmp, ignore_errors=True)

                    tmp_dir = tempfile.mkdtemp()
                    st.session_state["face_db_tmp_dir"] = tmp_dir  # persist path

                    for uf in uploaded_faces:
                        uf.seek(0)  # ← KEY FIX: reset buffer before reading
                        fp = os.path.join(tmp_dir, uf.name)
                        with open(fp, "wb") as f:
                            f.write(uf.read())
                        logger.debug(f"Saved uploaded file to: {fp}")

                    extra = build_face_database(tmp_dir, fr_module)
                    st.session_state["face_db_upload_key"] = upload_key
                    st.session_state["face_db_extra"] = extra
                    logger.info(f"Built {len(extra)} encoding(s) from uploads")

                extra = st.session_state.get("face_db_extra", {})
                face_db.update(extra)
                if extra:
                    st.success(f"✅ Added {len(extra)} person(s) from uploads")
                    for name in extra:
                        st.markdown(f"<span class='person-badge'>{name}</span>",
                                    unsafe_allow_html=True)
                elif st.session_state.get("face_db_upload_key") == upload_key:
                    st.warning("⚠️ Images uploaded but no faces detected. "
                            "Ensure images are clear, well-lit, and show the full face.")
        else:
            st.error("face_recognition module not available.")

        # ── Debug panel ───────────────────────────────────────────────────────
        # with st.expander("🔍 Debug — Face DB Status"):
        #     if face_db:
        #         st.write(f"**{len(face_db)} person(s) in database:**")
        #         for name in face_db:
        #             enc = face_db[name]
        #             st.write(f"- `{name}` — encoding shape: `{enc.shape}`")
        #     else:
        #         st.write("No faces in database.")
        #     log_file = Path(__file__).parent / "smart_vision.log"
        #     if log_file.exists():
        #         with open(log_file, "r") as lf:
        #             lines = lf.readlines()
        #         recent = "".join(lines[-60:])   # last 60 lines
        #         st.text_area("Recent log output", recent, height=250)

        st.markdown("---")
        st.markdown("### <span style='color:#000000;'>📦 Target Objects</span>", unsafe_allow_html=True)
        for cls_name in TARGET_CLASSES.values():
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            hex_c = "#{:02x}{:02x}{:02x}".format(*color)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
                f"<div style='width:12px;height:12px;border-radius:50%;background:{hex_c};'></div>"
                f"<span style='font-size:0.85rem;color:#000000;'>{cls_name.title()}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Main Tabs ─────────────────────────────────────────────────────────────
    tab1,  = st.tabs(["📹 Live Camera"])

    # ── TAB 1: Live Camera ────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 📷 Live Camera Detection")
        # st.info("ℹ️ On **mobile**: tap 'Start' and allow camera access.\n\n"
        #         "On **desktop**: it uses your webcam.")

        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av

            RTC_CONFIG = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            class VideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.conf   = conf_threshold
                    self.db     = face_db
                    self.labels = []
                    logger.info("VideoProcessor initialised")

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    annotated, lbls = process_frame(
                        img, yolo_model, self.db, fr_module, self.conf
                    )
                    self.labels = lbls
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            ctx = webrtc_streamer(
                key="smart-detection",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=False,
            )

            if ctx.video_processor:
                labels = ctx.video_processor.labels
                if labels:
                    st.markdown("**Detected:**")
                    for l in set(labels):
                        is_known = l not in TARGET_CLASSES.values()
                        badge_cls = "person-badge" if (l == "person" or is_known) else "object-badge"
                        st.markdown(f"<span class='{badge_cls}'>{l}</span>",
                                    unsafe_allow_html=True)

        except ImportError:
            logger.warning("streamlit-webrtc not installed — falling back to OpenCV")
            st.warning("⚠️ `streamlit-webrtc` not installed. Live camera unavailable.")
            st.code("pip install streamlit-webrtc", language="bash")

            st.markdown("**Fallback: OpenCV local camera (local run only)**")
            if st.button("▶ Start Webcam (local only)"):
                logger.info("Starting local OpenCV webcam")
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                stop_btn = st.button("⏹ Stop")
                while cap.isOpened() and not stop_btn:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Webcam frame read failed")
                        break
                    ann, _ = process_frame(frame, yolo_model, face_db, fr_module, conf_threshold)
                    stframe.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)
                cap.release()
                logger.info("Webcam stopped")

    # ── TAB 2: Upload Image ───────────────────────────────────────────────────
    # with tab2:
        # st.markdown("### 🖼️ Image Detection")
        # uploaded_img = st.file_uploader(
        #     "Upload an image", type=["jpg", "jpeg", "png", "webp"], key="img_upload"
        # )

        # if uploaded_img:
        #     logger.info(f"Image uploaded: {uploaded_img.name} ({uploaded_img.size} bytes)")
        #     img_pil = Image.open(uploaded_img).convert("RGB")
        #     img_np  = np.array(img_pil)
        #     img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        #     with st.spinner("Running detection…"):
        #         annotated, labels = process_frame(
        #             img_bgr, yolo_model, face_db, fr_module, conf_threshold
        #         )
        #     logger.info(f"Image detection results: {labels}")

        #     col1, col2 = st.columns(2)
        #     with col1:
        #         st.image(img_pil, caption="Original", use_container_width=True)
        #     with col2:
        #         result_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        #         st.image(result_rgb, caption="Detected", use_container_width=True)

        #     if labels:
        #         st.markdown("**Detections:**")
        #         for l in set(labels):
        #             is_known = l not in TARGET_CLASSES.values()
        #             badge_cls = "person-badge" if (l == "person" or is_known) else "object-badge"
        #             st.markdown(f"<span class='{badge_cls}'>{l}</span>",
        #                         unsafe_allow_html=True)
            # else:
            #     st.info("No target objects detected.")

    # ── TAB 3: Upload Video ───────────────────────────────────────────────────
    # with tab3:
        # st.markdown("### 🎬 Video Detection")
        # uploaded_vid = st.file_uploader(
        #     "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
        # )

        # if uploaded_vid:
        #     logger.info(f"Video uploaded: {uploaded_vid.name} ({uploaded_vid.size} bytes)")
        #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        #         tmp.write(uploaded_vid.read())
        #         tmp_path = tmp.name

        #     if st.button("🚀 Process Video"):
        #         logger.info(f"Starting video processing: {tmp_path}")
        #         cap         = cv2.VideoCapture(tmp_path)
        #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #         fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        #         w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #         h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #         logger.info(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

        #         out_path = tmp_path.replace(".mp4", "_out.mp4")
        #         fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        #         out      = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        #         progress   = st.progress(0, "Processing frames…")
        #         stframe    = st.empty()
        #         all_labels = set()
        #         frame_idx  = 0

        #         while True:
        #             ret, frame = cap.read()
        #             if not ret:
        #                 break
        #             ann, lbls = process_frame(
        #                 frame, yolo_model, face_db, fr_module, conf_threshold
        #             )
        #             all_labels.update(lbls)
        #             out.write(ann)

        #             if frame_idx % 5 == 0:
        #                 preview = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        #                 stframe.image(preview, channels="RGB", use_container_width=True)

        #             frame_idx += 1
        #             pct = min(frame_idx / max(total_frames, 1), 1.0)
        #             progress.progress(pct, f"Frame {frame_idx}/{total_frames}")

        #         cap.release()
        #         out.release()
        #         stframe.empty()
        #         progress.empty()

        #         logger.info(f"Video processing done. {frame_idx} frames. "
        #                     f"Unique detections: {all_labels}")
        #         st.success(f"✅ Done! Processed {frame_idx} frames.")

        #         if all_labels:
        #             st.markdown("**All unique detections in video:**")
        #             for l in sorted(all_labels):
        #                 is_known = l not in TARGET_CLASSES.values()
        #                 badge_cls = "person-badge" if (l == "person" or is_known) else "object-badge"
        #                 st.markdown(f"<span class='{badge_cls}'>{l}</span>",
        #                             unsafe_allow_html=True)

        #         with open(out_path, "rb") as f:
        #             st.download_button(
        #                 "⬇️ Download Annotated Video", f,
        #                 "detected_video.mp4", "video/mp4"
        #             )

        #         os.unlink(tmp_path)
        #         try:
        #             os.unlink(out_path)
        #         except Exception:
        #             pass


if __name__ == "__main__":
    main()