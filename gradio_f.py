"""
Smart Detection App — Gradio Version
- YOLOv8 object detection (bottle, person, cell phone, laptop, potted plant)
- Face recognition from uploaded images or a known_persons/ folder
- Works on PC and Mobile browsers via webcam
- Gradio hosted
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import logging
import sys
import tempfile
import shutil
from pathlib import Path

# ─── Logger Setup ─────────────────────────────────────────────────────────────
def setup_logger():
    logger = logging.getLogger("SmartVisionAI")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
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
logger.info("Smart Vision AI (Gradio) starting up")
logger.info("=" * 60)


# ─── COCO class IDs for target objects ────────────────────────────────────────
TARGET_CLASSES = {
    39: "bottle",
    0:  "person",
    67: "cell phone",
    63: "laptop",
    58: "potted plant",
}

CLASS_COLORS = {
    "bottle":       (0, 200, 255),
    "person":       (255, 50, 200),
    "cell phone":   (50, 255, 150),
    "laptop":       (255, 200, 0),
    "potted plant": (0, 255, 80),
}

# ─── Global model state ───────────────────────────────────────────────────────
_yolo_model = None
_fr_module = None
_face_db: dict = {}
_tmp_dir = None


def load_models():
    global _yolo_model, _fr_module
    if _yolo_model is not None:
        return _yolo_model, _fr_module

    logger.info("Loading AI models…")
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8x.pt")
        logger.info("YOLOv8 loaded successfully")
    except ImportError:
        logger.error("ultralytics not installed")
    except Exception as e:
        logger.exception(f"Error loading YOLO: {e}")

    try:
        import face_recognition as _fr
        _fr_module = _fr
        logger.info("face_recognition loaded successfully")
    except ImportError:
        logger.warning("face_recognition not installed")
    except Exception as e:
        logger.exception(f"Error loading face_recognition: {e}")

    return _yolo_model, _fr_module


# ─── Face encoding helpers ────────────────────────────────────────────────────
def load_single_face_encoding(img_path: str, fr_module):
    logger.debug(f"Loading face image: {img_path}")
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"cv2.imread returned None for: {img_path}")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Failed to read image {img_path}: {e}")
        return None

    try:
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        locations = fr_module.face_locations(img_rgb, number_of_times_to_upsample=1, model="hog")
        logger.debug(f"  face_locations found {len(locations)} face(s) in {Path(img_path).name}")
        if not locations:
            logger.warning(f"  No face detected in {Path(img_path).name}")
            return None
        encodings = fr_module.face_encodings(img_rgb, known_face_locations=locations, num_jitters=1)
        if not encodings:
            return None
        logger.info(f"  ✅ Encoding built for {Path(img_path).name}")
        return encodings[0]
    except Exception as e:
        logger.exception(f"  Error encoding face in {Path(img_path).name}: {e}")
        return None


def build_face_database(folder_path: str, fr_module) -> dict:
    db = {}
    folder = Path(folder_path)
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [p for p in folder.glob("*") if p.suffix.lower() in supported]
    logger.info(f"Found {len(image_files)} image(s) in {folder.resolve()}")
    for img_path in image_files:
        name = img_path.stem.replace("_", " ").replace("-", " ").title()
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
    labels_found = []
    if yolo_model is None:
        return frame_bgr, labels_found

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        results = yolo_model(frame_bgr, conf=conf_threshold, verbose=False)[0]
    except Exception as e:
        logger.error(f"YOLO inference failed: {e}")
        return frame_bgr, labels_found

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

        if label == "person" and fr_module is not None and face_db:
            try:
                h_frame, w_frame = frame_bgr.shape[:2]
                rx1, ry1 = max(0, x1), max(0, y1)
                rx2, ry2 = min(w_frame, x2), min(h_frame, y2)
                face_roi = frame_rgb[ry1:ry2, rx1:rx2]

                if face_roi.size > 0:
                    face_roi = np.ascontiguousarray(face_roi, dtype=np.uint8)
                    locs = fr_module.face_locations(face_roi, number_of_times_to_upsample=1, model="hog")
                    if locs:
                        encs = fr_module.face_encodings(face_roi, known_face_locations=locs, num_jitters=1)
                        for enc in encs:
                            known_encs  = list(face_db.values())
                            known_names = list(face_db.keys())
                            dists   = fr_module.face_distance(known_encs, enc)
                            matches = fr_module.compare_faces(known_encs, enc, tolerance=0.55)
                            if len(dists) > 0:
                                best_idx = int(np.argmin(dists))
                                if matches[best_idx]:
                                    display_name = known_names[best_idx]
                                    logger.info(f"✅ Recognised: '{display_name}' (dist={dists[best_idx]:.3f})")
            except Exception as e:
                logger.exception(f"Face recognition error: {e}")

        tag = f"{display_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_bgr, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
        labels_found.append(display_name)

    return frame_bgr, labels_found


# ─── Gradio callback functions ────────────────────────────────────────────────

def update_face_db(uploaded_files):
    """Called when user uploads face images. Returns status message."""
    global _face_db, _tmp_dir, _fr_module
    yolo_model, fr_module = load_models()

    if fr_module is None:
        return "❌ face_recognition module not available.", ""

    # Start fresh each upload
    if _tmp_dir and os.path.isdir(_tmp_dir):
        shutil.rmtree(_tmp_dir, ignore_errors=True)

    extra = {}

    # Load from known_persons folder first
    if os.path.isdir("known_persons"):
        extra.update(build_face_database("known_persons", fr_module))

    # Then from uploaded files
    if uploaded_files:
        _tmp_dir = tempfile.mkdtemp()
        for file_path in uploaded_files:
            dst = os.path.join(_tmp_dir, Path(file_path).name)
            shutil.copy(file_path, dst)
        extra.update(build_face_database(_tmp_dir, fr_module))

    _face_db = extra

    if not _face_db:
        status = "⚠️ No faces detected. Ensure images are clear, well-lit, and show the full face."
        names_html = ""
    else:
        status = f"✅ {len(_face_db)} person(s) loaded into face database."
        names_html = " ".join(
            f'<span style="background:linear-gradient(90deg,#7b2ff7,#f707a8);color:white;'
            f'padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.85rem;'
            f'display:inline-block;margin:3px;">{name}</span>'
            for name in _face_db
        )

    return status, names_html


def webcam_detection(frame):
    """Called on every webcam frame. Returns annotated frame + label HTML."""
    global _face_db
    yolo_model, fr_module = load_models()

    if frame is None:
        return None, ""

    # Gradio passes frames as RGB numpy arrays
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr, labels = process_frame(frame_bgr, yolo_model, _face_db, fr_module, conf_threshold=0.4)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    label_html = ""
    if labels:
        for l in sorted(set(labels)):
            is_known_person = l not in TARGET_CLASSES.values()
            if l == "person" or is_known_person:
                badge_style = ("background:linear-gradient(90deg,#7b2ff7,#f707a8);"
                               "animation:pulse 2s infinite;")
            else:
                badge_style = "background:linear-gradient(90deg,#0066ff,#00aaff);"
            label_html += (
                f'<span style="{badge_style}color:white;padding:4px 14px;border-radius:20px;'
                f'font-weight:700;font-size:0.9rem;display:inline-block;margin:4px;">{l}</span>'
            )

    return annotated_rgb, label_html


def image_detection(uploaded_image):
    """Run detection on a single uploaded image."""
    global _face_db
    yolo_model, fr_module = load_models()

    if uploaded_image is None:
        return None, ""

    img_rgb = np.array(uploaded_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr, labels = process_frame(img_bgr, yolo_model, _face_db, fr_module, conf_threshold=0.4)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    label_html = ""
    if labels:
        for l in sorted(set(labels)):
            is_known_person = l not in TARGET_CLASSES.values()
            badge_style = (
                "background:linear-gradient(90deg,#7b2ff7,#f707a8);"
                if (l == "person" or is_known_person)
                else "background:linear-gradient(90deg,#0066ff,#00aaff);"
            )
            label_html += (
                f'<span style="{badge_style}color:white;padding:4px 14px;border-radius:20px;'
                f'font-weight:700;font-size:0.9rem;display:inline-block;margin:4px;">{l}</span>'
            )
    else:
        label_html = "<span style='color:#888;'>No target objects detected.</span>"

    return annotated_rgb, label_html


def video_detection(video_path):
    """Run detection on an uploaded video and return an annotated output."""
    global _face_db
    yolo_model, fr_module = load_models()

    if video_path is None:
        return None, ""

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = video_path.replace(".mp4", "_out.mp4").replace(".avi", "_out.avi").replace(".mov", "_out.mov")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    all_labels = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, lbls = process_frame(frame, yolo_model, _face_db, fr_module, conf_threshold=0.4)
        all_labels.update(lbls)
        out.write(annotated)

    cap.release()
    out.release()

    label_html = ""
    for l in sorted(all_labels):
        is_known_person = l not in TARGET_CLASSES.values()
        badge_style = (
            "background:linear-gradient(90deg,#7b2ff7,#f707a8);"
            if (l == "person" or is_known_person)
            else "background:linear-gradient(90deg,#0066ff,#00aaff);"
        )
        label_html += (
            f'<span style="{badge_style}color:white;padding:4px 14px;border-radius:20px;'
            f'font-weight:700;font-size:0.9rem;display:inline-block;margin:4px;">{l}</span>'
        )

    if not label_html:
        label_html = "<span style='color:#888;'>No target objects detected.</span>"

    return out_path, label_html


# ─── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

* { font-family: 'Syne', sans-serif !important; }
code, pre { font-family: 'Space Mono', monospace !important; }

body, .gradio-container { background: #f5f6fa !important; color: #1a1a2e !important; }

h1, h2, h3 { color: #2c2c6c !important; letter-spacing: -0.02em; }

.tab-nav button {
    font-weight: 700 !important;
    color: #2c2c6c !important;
}
.tab-nav button.selected {
    border-bottom: 2px solid #7b2ff7 !important;
    color: #7b2ff7 !important;
}

button.primary {
    background: linear-gradient(90deg, #7b2ff7, #0066ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
button.primary:hover { opacity: 0.85; transform: translateY(-1px); }

.panel, .block { border-radius: 12px !important; }

@keyframes pulse {
    0%,100%{ box-shadow: 0 0 6px #f707a8; }
    50%{ box-shadow: 0 0 18px #f707a8; }
}
"""

TARGET_INFO_HTML = "".join(
    f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
    f'<div style="width:12px;height:12px;border-radius:50%;'
    f'background:#{CLASS_COLORS[c][0]:02x}{CLASS_COLORS[c][1]:02x}{CLASS_COLORS[c][2]:02x};"></div>'
    f'<span style="font-size:0.9rem;color:#1a1a2e;">{c.title()}</span></div>'
    for c in CLASS_COLORS
)

# ─── Build Gradio UI ──────────────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="Smart Vision AI") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px;">
        <h1 style="font-size:2.5rem;margin-bottom:0;color:#2c2c6c;">🔍 Smart Vision AI</h1>
        <p style="color:#8888aa;margin-top:6px;">YOLOv8 Object Detection + Face Recognition</p>
    </div>
    <hr style="border-color:#e0e0f0;">
    """)

    with gr.Row():
        # ── Left sidebar ───────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.HTML("<h3 style='color:#2c2c6c;'>⚙️ Settings</h3>")

            face_upload = gr.File(
                label="Upload Face Images (jpg/png/webp)",
                file_types=[".jpg", ".jpeg", ".png", ".webp"],
                file_count="multiple",
            )
            load_btn = gr.Button("Load Faces", variant="primary")
            face_status = gr.Textbox(label="Status", interactive=False, lines=2)
            loaded_names = gr.HTML(label="Loaded Persons")

            gr.HTML("<hr style='border-color:#e0e0f0;margin:16px 0;'>")
            gr.HTML(f"<h3 style='color:#2c2c6c;'>📦 Target Objects</h3>{TARGET_INFO_HTML}")

        # ── Main content ───────────────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():

                # ── Tab 1: Live Camera ─────────────────────────────────────────
                with gr.Tab("📹 Live Camera"):
                    gr.HTML("<h3 style='color:#2c2c6c;'>📷 Live Camera Detection</h3>")
                    gr.HTML("<p style='color:#666;font-size:0.9rem;'>Allow camera access when prompted. "
                            "Detection runs on each captured frame.</p>")
                    with gr.Row():
                        webcam_in = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="Camera Feed",
                            # mirror_webcam=False,
                        )
                        webcam_out = gr.Image(label="Detected Output", streaming=True)
                    webcam_labels = gr.HTML(label="Detections")

                    webcam_in.stream(
                        fn=webcam_detection,
                        inputs=[webcam_in],
                        outputs=[webcam_out, webcam_labels],
                        time_limit=60,
                        stream_every=0.1,
                    )

                # ── Tab 2: Image Upload ────────────────────────────────────────
                with gr.Tab("🖼️ Image Detection"):
                    gr.HTML("<h3 style='color:#2c2c6c;'>Upload an Image</h3>")
                    with gr.Row():
                        img_in = gr.Image(label="Upload Image", type="pil")
                        img_out = gr.Image(label="Detected Output")
                    img_labels = gr.HTML(label="Detections")
                    detect_btn = gr.Button("🚀 Run Detection", variant="primary")

                    detect_btn.click(
                        fn=image_detection,
                        inputs=[img_in],
                        outputs=[img_out, img_labels],
                    )

                # ── Tab 3: Video Upload ────────────────────────────────────────
                with gr.Tab("🎬 Video Detection"):
                    gr.HTML("<h3 style='color:#2c2c6c;'>Upload a Video</h3>")
                    gr.HTML("<p style='color:#888;font-size:0.85rem;'>Processing may take a moment "
                            "depending on video length.</p>")
                    vid_in = gr.Video(label="Upload Video")
                    vid_detect_btn = gr.Button("🚀 Process Video", variant="primary")
                    vid_out = gr.Video(label="Annotated Output")
                    vid_labels = gr.HTML(label="All Detected Objects")

                    vid_detect_btn.click(
                        fn=video_detection,
                        inputs=[vid_in],
                        outputs=[vid_out, vid_labels],
                    )

    # Wire up face loading
    load_btn.click(
        fn=update_face_db,
        inputs=[face_upload],
        outputs=[face_status, loaded_names],
    )

    gr.HTML("""
    <div style="text-align:center;padding:20px;color:#aaaacc;font-size:0.8rem;margin-top:16px;">
        Smart Vision AI &nbsp;•&nbsp; YOLOv8 + face_recognition &nbsp;•&nbsp; Gradio
    </div>
    """)


if __name__ == "__main__":
    load_models()  # pre-load models on startup
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
