import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
warnings.filterwarnings('ignore')

# ---------------------- STREAMLIT SETUP ----------------------
st.set_page_config(page_title="DV Photo Auto-Fix", layout="wide")
st.title("🧠 DV Photo Measurement Auto Adjust")

# Initialize session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_bg_removed" not in st.session_state:
    st.session_state.current_bg_removed = None
if "current_head_info" not in st.session_state:
    st.session_state.current_head_info = {}
if "fix_count" not in st.session_state:
    st.session_state.fix_count = 0
if "show_fixed_preview" not in st.session_state:
    st.session_state.show_fixed_preview = False

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ---------------------- UTILS ----------------------
def resize_for_display(image, max_width=600):
    w, h = image.size
    ratio = max_width / float(w)
    new_h = int(h * ratio)
    return image.resize((max_width, new_h))

def detect_face(image_pil):
    np_image = np.array(image_pil)
    image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

    if not results.detections:
        return None

    h, w, _ = image_rgb.shape
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    xmin, ymin, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

    keypoints = detection.location_data.relative_keypoints
    eye_left = (int(keypoints[0].x * w), int(keypoints[0].y * h))
    eye_right = (int(keypoints[1].x * w), int(keypoints[1].y * h))
    nose_tip = (int(keypoints[2].x * w), int(keypoints[2].y * h))

    head_top = ymin
    chin = ymin + height

    return {
        "bbox": (xmin, ymin, width, height),
        "head_top": head_top,
        "chin": chin,
        "eye_line": int((eye_left[1] + eye_right[1]) / 2),
        "image_height": h
    }

def remove_bg(image_pil):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    img_no_bg = remove(buffered.getvalue())
    return Image.open(io.BytesIO(img_no_bg)).convert("RGBA")

def draw_guidelines(image_pil, head_info):
    draw = ImageDraw.Draw(image_pil)

    if not head_info:
        return image_pil, 0, 0

    h = head_info["image_height"]
    head_top = head_info["head_top"]
    chin = head_info["chin"]
    eye_line = head_info["eye_line"]

    # Ratios (percentages)
    head_height_ratio = (chin - head_top) / h * 100
    eye_level_ratio = eye_line / h * 100

    # Draw lines
    draw.line([(0, head_top), (image_pil.width, head_top)], fill=(0, 255, 0), width=3)
    draw.line([(0, chin), (image_pil.width, chin)], fill=(255, 0, 0), width=3)
    draw.line([(0, eye_line), (image_pil.width, eye_line)], fill=(0, 0, 255), width=2)

    # Labels
    draw.text((10, head_top + 5), f"Head Top", fill=(0, 255, 0))
    draw.text((10, chin - 25), f"Chin", fill=(255, 0, 0))
    draw.text((10, eye_line - 20), f"Eyes", fill=(0, 0, 255))

    return image_pil, round(head_height_ratio, 2), round(eye_level_ratio, 2)

def auto_adjust_dv_photo(image_pil, force_correction=False):
    image_pil = image_pil.convert("RGB")

    head_info = detect_face(image_pil)
    if not head_info:
        raise ValueError("No face detected.")

    head_top = head_info["head_top"]
    chin = head_info["chin"]
    h = head_info["image_height"]

    current_head_ratio = (chin - head_top) / h * 100
    desired_head_ratio = 50  # DV spec target (example)
    scale_factor = desired_head_ratio / current_head_ratio if force_correction else 1

    new_h = int(h * scale_factor)
    new_img = image_pil.resize((image_pil.width, new_h))
    return new_img, detect_face(new_img)

# ---------------------- UI LAYOUT ----------------------
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📸 Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image

        with st.spinner("Removing background..."):
            bg_removed = remove_bg(image)
            st.session_state.current_bg_removed = bg_removed

        with st.spinner("Detecting and adjusting..."):
            processed, head_info = auto_adjust_dv_photo(bg_removed)
            st.session_state.current_image = processed
            st.session_state.current_head_info = head_info

# ---------------------- DISPLAY RESULTS ----------------------
if st.session_state.current_image:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Original (Background Removed)")
        st.image(resize_for_display(st.session_state.current_bg_removed), use_container_width=True)

    with col2:
        st.subheader("✅ Processed with Guidelines")
        final_preview, actual_head_ratio, actual_eye_ratio = draw_guidelines(
            st.session_state.current_image.copy(),
            st.session_state.current_head_info
        )
        st.session_state.current_head_info['actual_head_ratio'] = actual_head_ratio
        st.session_state.current_head_info['actual_eye_ratio'] = actual_eye_ratio

        st.image(resize_for_display(final_preview), use_container_width=True)
        st.markdown(f"""
        **Measurements**
        - Head Height: `{actual_head_ratio}%`
        - Eye Level: `{actual_eye_ratio}%`
        """)

    st.divider()

    # ---------------------- FIX BUTTON ----------------------
    if st.button("🛠️ Fix Photo Measurements", type="primary", use_container_width=True):
        st.session_state.fix_count += 1
        st.session_state.show_fixed_preview = True
        st.experimental_rerun()

    # ---------------------- APPLY FIX ----------------------
    if st.session_state.show_fixed_preview:
        with st.spinner("Auto-adjusting to DV specifications..."):
            force_correction = st.session_state.fix_count > 0
            processed, head_info = auto_adjust_dv_photo(st.session_state.current_bg_removed, force_correction)
            st.session_state.current_image = processed
            st.session_state.current_head_info = head_info
            st.session_state.show_fixed_preview = False

        st.experimental_rerun()

    # ---------------------- DOWNLOAD ----------------------
    output_buffer = io.BytesIO()
    st.session_state.current_image.save(output_buffer, format="JPEG")
    st.download_button(
        "💾 Download Adjusted Image",
        data=output_buffer.getvalue(),
        file_name="dv_photo_fixed.jpg",
        mime="image/jpeg",
        use_container_width=True
    )
