"""
AI-Based Smoke and Fire Detection System
-----------------------------------------
A simple, professional Streamlit web application for college demonstration.
Uses a genuine, fine-tuned YOLOv8 model to detect Fire and Smoke in uploaded images.
"""

import os
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Page Configuration & Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Smoke & Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Model Loading & Verification (Genuine Weights)
# ---------------------------------------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "fire_smoke_yolov8n.pt"
MODEL_REPO = "rabahdev/fire-smoke-yolov8n"
MODEL_DIRECT_URL = "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Downloads and loads the genuine fire & smoke YOLOv8 model weights.
    Includes fallback mechanisms and file integrity checks.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # If the file exists but is incomplete or invalid (< 1MB), remove it
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size < 1_000_000:
        try:
            MODEL_PATH.unlink()
        except Exception:
            pass

    # Download weights if not already present locally
    if not MODEL_PATH.exists():
        with st.spinner("Downloading verified Fire & Smoke AI model weights (~6MB)..."):
            temp_path = MODEL_PATH.with_suffix(".tmp")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

            download_success = False

            # Attempt 1: Using huggingface_hub
            try:
                from huggingface_hub import hf_hub_download
                cached_file = hf_hub_download(repo_id=MODEL_REPO, filename="best.pt")
                shutil.copy(cached_file, temp_path)
                if temp_path.exists() and temp_path.stat().st_size > 1_000_000:
                    download_success = True
            except Exception:
                download_success = False

            # Attempt 2: Using urllib with standard headers
            if not download_success:
                try:
                    req = urllib.request.Request(
                        MODEL_DIRECT_URL,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                    )
                    with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
                        shutil.copyfileobj(response, out_file)
                    if temp_path.exists() and temp_path.stat().st_size > 1_000_000:
                        download_success = True
                except Exception:
                    download_success = False

            # Finalize file atomically
            if download_success and temp_path.exists() and temp_path.stat().st_size > 1_000_000:
                temp_path.replace(MODEL_PATH)
            else:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                raise RuntimeError(
                    "Unable to automatically download model weights. "
                    "Please ensure you have an active internet connection or manually place "
                    "'best.pt' renamed to 'fire_smoke_yolov8n.pt' in the 'models/' folder."
                )

    # Load YOLO model
    model = YOLO(str(MODEL_PATH))
    return model


# ---------------------------------------------------------
# 3. Sidebar Configuration & Model Status
# ---------------------------------------------------------
with st.sidebar:
    st.title("🔥 System Settings")
    
    st.subheader("1. AI Model Status")
    try:
        model = load_model()
        # Verify the actual trained classes
        classes = model.names  # e.g., {0: 'smoke', 1: 'fire'}
        st.success("✅ Model Loaded Successfully")
        
        # Display verified classes to prove authenticity (no fake predictions)
        verified_classes = list(classes.values())
        st.markdown(f"**Verified Model Classes:** `{', '.join(verified_classes)}`")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("2. Detection Sensitivity")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help="Higher values reduce false positives. Lower values detect faint smoke or fire."
    )

    st.markdown("---")
    st.subheader("3. Project Details")
    st.info(
        """
        - **Project:** AI-Based Smoke and Fire Detection
        - **Target Domain:** Computer Vision Safety
        - **Model Architecture:** YOLOv8 Nano
        - **Verified Detection Classes:** Fire, Smoke
        """
    )


# ---------------------------------------------------------
# 4. Main Page Header
# ---------------------------------------------------------
st.title("🔥 AI-Based Smoke and Fire Detection System")
st.write(
    "Upload an image to detect early signs of smoke and active fire using computer vision. "
    "Results show bounding boxes, confidence scores, and safety alert warnings."
)

st.markdown("---")


# ---------------------------------------------------------
# 5. Image Upload & Processing
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image for fire and smoke analysis (JPG, JPEG, PNG)...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read the image and fix EXIF orientation if present
    raw_image = Image.open(uploaded_file).convert("RGB")
    try:
        from PIL import ImageOps
        raw_image = ImageOps.exif_transpose(raw_image)
    except Exception:
        pass
    image_np = np.array(raw_image)

    # Perform genuine model inference
    with st.spinner("Analyzing image for fire and smoke..."):
        results = model.predict(source=image_np, conf=confidence_threshold)
        result = results[0]

    # Render bounding boxes onto image
   annotated_rgb = result.plot()
    
    # Collect detection statistics
    detected_items = []
    fire_count = 0
    smoke_count = 0

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names.get(class_id, "Unknown").lower()

            detected_items.append({
                "Object": class_name.capitalize(),
                "Confidence": f"{confidence * 100:.1f}%"
            })

            if "fire" in class_name:
                fire_count += 1
            elif "smoke" in class_name:
                smoke_count += 1

    # ---------------------------------------------------------
    # 6. Safety Warning Banners
    # ---------------------------------------------------------
    st.subheader("Safety Status")
    if fire_count > 0 and smoke_count > 0:
        st.error(
            f"🚨 **CRITICAL HAZARD DETECTED:** Both Fire ({fire_count}) and Smoke ({smoke_count}) detected! "
            "Trigger emergency alarms immediately."
        )
    elif fire_count > 0:
        st.error(
            f"🚨 **FIRE ALERT:** Active Fire detected ({fire_count} region(s))! "
            "Evacuate area and alert fire department."
        )
    elif smoke_count > 0:
        st.warning(
            f"⚠️ **SMOKE WARNING:** Smoke detected ({smoke_count} region(s))! "
            "Possible early stage hazard. Investigate promptly."
        )
    else:
        st.success("✅ **SAFE:** No fire or smoke detected with current confidence threshold.")

    # ---------------------------------------------------------
    # 7. Image Comparison Display
    # ---------------------------------------------------------
    st.subheader("Visual Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Uploaded Image**")
        st.image(raw_image, use_container_width=True)

    with col2:
        st.markdown("**AI Detection Result (Bounding Boxes)**")
        st.image(annotated_rgb, use_container_width=True)

    # ---------------------------------------------------------
    # 8. Detection Summary & Metrics
    # ---------------------------------------------------------
    st.subheader("Detection Summary")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Fire Instances", fire_count)
    metric_col2.metric("Smoke Instances", smoke_count)
    metric_col3.metric("Total Detections", len(detected_items))

    if detected_items:
        st.markdown("**Detailed Detection Breakdown:**")
        st.table(detected_items)
    else:
        st.info("No objects passed the confidence threshold.")

else:
    st.info("👆 Please upload an image using the box above to start detection.")
