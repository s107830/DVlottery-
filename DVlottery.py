import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
warnings.filterwarnings('ignore')

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Auto Correction")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69

mp_face_mesh = mp.solutions.face_mesh

# ---------------------- HELPERS ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=True, min_detection_confidence=0.5) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face landmarks found")
        return results.multi_face_landmarks[0]

def get_head_eye_positions(landmarks, img_h, img_w):
    # Get head top (forehead) and chin positions
    top_y = int(landmarks.landmark[10].y * img_h)  # Forehead
    chin_y = int(landmarks.landmark[152].y * img_h)  # Chin
    
    # Get eye positions
    left_eye_y = int(landmarks.landmark[33].y * img_h)
    right_eye_y = int(landmarks.landmark[263].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2
    
    # Add buffer for hair/head top
    hair_buffer = int((chin_y - top_y) * 0.25)
    top_y = max(0, top_y - hair_buffer)
    
    return top_y, chin_y, eye_y

def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img_pil

# ---------------------- CORE PROCESSING ----------------------
def process_dv_photo(img_pil):
    cv_img = np.array(img_pil)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

    h, w = cv_img.shape[:2]
    landmarks = get_face_landmarks(cv_img)
    top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
    head_height = chin_y - top_y

    # Scale based on DV required head ratio
    scale_factor = (MIN_SIZE * 0.6) / head_height
    scale_factor = np.clip(scale_factor, 0.5, 2.0)
    resized = cv2.resize(cv_img, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_LANCZOS4)

    new_h, new_w = resized.shape[:2]
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)

    # Recalculate positions after scaling
    landmarks_resized = get_face_landmarks(resized)
    top_y, chin_y, eye_y = get_head_eye_positions(landmarks_resized, new_h, new_w)
    head_height = chin_y - top_y

    # Calculate placement to meet eye position requirements
    target_eye_y = MIN_SIZE - int(MIN_SIZE * ((EYE_MIN_RATIO + EYE_MAX_RATIO) / 2))
    y_offset = target_eye_y - eye_y
    x_offset = (MIN_SIZE - new_w) // 2

    # Place the image on canvas
    y_start = max(0, y_offset)
    y_end = min(MIN_SIZE, y_offset + new_h)
    x_start = max(0, x_offset)
    x_end = min(MIN_SIZE, x_offset + new_w)
    y_src_start = max(0, -y_offset)
    y_src_end = min(new_h, MIN_SIZE - y_offset)
    x_src_start = max(0, -x_offset)
    x_src_end = min(new_w, MIN_SIZE - x_offset)

    canvas[y_start:y_end, x_start:x_end] = resized[y_src_start:y_src_end, x_src_start:x_src_end]
    
    # Get final positions on the canvas
    final_top_y = top_y + y_offset
    final_chin_y = chin_y + y_offset
    final_eye_y = eye_y + y_offset
    
    result = Image.fromarray(canvas)
    result = ImageEnhance.Sharpness(result).enhance(1.1)

    head_info = {
        "top_y": final_top_y,
        "chin_y": final_chin_y,
        "eye_y": final_eye_y,
        "head_height": head_height,
        "canvas_size": MIN_SIZE
    }
    return result, head_info

# ---------------------- DRAW LINES ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = w // 2
    top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
    head_height, canvas_size = head_info["head_height"], head_info["canvas_size"]

    head_ratio = head_height / canvas_size
    eye_ratio = (canvas_size - eye_y) / canvas_size

    # Draw head vertical line (centered, full height)
    head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
    draw.line([(cx, top_y), (cx, chin_y)], fill=head_color, width=4)
    
    # Head ratio text
    head_text_y = (top_y + chin_y) // 2
    draw.text((cx + 10, head_text_y - 20), f"Head {int(head_ratio*100)}%", fill=head_color)
    draw.text((cx + 10, head_text_y), "Req 50-69%", fill="blue")

    # Draw eye position guidelines
    eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"
    eye_min_y = h - int(h * EYE_MAX_RATIO)  # 56% from top = 44% from bottom
    eye_max_y = h - int(h * EYE_MIN_RATIO)  # 69% from top = 31% from bottom
    
    # Eye range guidelines (green horizontal lines)
    draw.line([(cx - 200, eye_min_y), (cx + 200, eye_min_y)], fill="green", width=2)
    draw.line([(cx - 200, eye_max_y), (cx + 200, eye_max_y)], fill="green", width=2)
    
    # Actual eye position line (red/green horizontal line)
    draw.line([(0, eye_y), (w, eye_y)], fill=eye_color, width=3)
    
    # Eye ratio text
    draw.text((cx - 250, eye_y - 20), f"Eyes {int(eye_ratio*100)}%", fill=eye_color)
    draw.text((cx - 250, eye_y), "Req 56-69%", fill="blue")

    return img, head_ratio, eye_ratio

# ---------------------- STREAMLIT LOGIC ----------------------
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Use session state to store processing results
    if 'processed_data' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
        st.session_state.last_upload = uploaded_file.name
        orig = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("Processing photo..."):
            bg_removed = remove_background(orig)
            processed, head_info = process_dv_photo(bg_removed)
            processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
            
            # Store in session state
            st.session_state.processed_data = {
                'orig': orig,
                'processed': processed,
                'processed_with_lines': processed_with_lines,
                'head_info': head_info,
                'head_ratio': head_ratio,
                'eye_ratio': eye_ratio
            }
    
    # Get data from session state
    data = st.session_state.processed_data
    orig = data['orig']
    processed = data['processed']
    processed_with_lines = data['processed_with_lines']
    head_info = data['head_info']
    head_ratio = data['head_ratio']
    eye_ratio = data['eye_ratio']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original")
        st.image(orig)

    with col2:
        st.subheader("✅ Processed Photo")
        st.image(processed_with_lines, caption="Processed Preview")

        # Show metrics
        c1, c2 = st.columns(2)
        c1.metric("Head Height", f"{int(head_ratio*100)}%", "✅" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "❌")
        c2.metric("Eye Position", f"{int(eye_ratio*100)}%", "✅" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "❌")

        needs_fix = (head_ratio < HEAD_MIN_RATIO or head_ratio > HEAD_MAX_RATIO or
                     eye_ratio < EYE_MIN_RATIO or eye_ratio > EYE_MAX_RATIO)

        if needs_fix:
            st.warning("⚠️ Some measurements are out of range.")
            if st.button("🛠️ Fix Photo Measurements", use_container_width=True, type="primary"):
                # Force complete reprocessing
                del st.session_state.processed_data
                st.rerun()
        else:
            st.success("✅ All measurements within range!")

        # Save download - use the processed image WITHOUT guidelines for download
        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ Download Corrected DV Photo",
                           buf.getvalue(),
                           "dv_lottery_photo.jpg",
                           "image/jpeg",
                           use_container_width=True)
        
        # Also provide option to download with guidelines
        buf_with_guides = io.BytesIO()
        processed_with_lines.save(buf_with_guides, format="JPEG", quality=95)
        st.download_button("⬇️ Download with Guidelines",
                           buf_with_guides.getvalue(),
                           "dv_lottery_photo_with_guides.jpg",
                           "image/jpeg",
                           use_container_width=True)
else:
    st.info("👆 Upload a photo to start.")
    # Clear session state when no file is uploaded
    if 'processed_data' in st.session_state:
        del st.session_state.processed_data
    if 'last_upload' in st.session_state:
        del st.session_state.last_upload
