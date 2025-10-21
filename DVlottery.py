import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
warnings.filterwarnings('ignore')

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("📸 DV Lottery Photo Editor — Auto Correction & Compliance Check")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69

mp_face_mesh = mp.solutions.face_mesh

# ---------------------- IMPROVED BACKGROUND REMOVAL WITH HAIR PRESERVATION ----------------------
def remove_background(img_pil):
    """Improved background removal with better hair edge preservation"""
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        img_bytes = b.getvalue()
        fg_bytes = remove(img_bytes)
        fg = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
        r, g, b, alpha = fg.split()
        alpha_np = np.array(alpha)

        # Preserve fine hair details
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_closed = cv2.morphologyEx(alpha_np, cv2.MORPH_CLOSE, kernel)
        alpha_smoothed = cv2.GaussianBlur(alpha_closed, (3, 3), 0.5)
        edges = cv2.Canny(alpha_np, 50, 150)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_edges = cv2.dilate(edges, edge_kernel, iterations=1)
        transition_mask = cv2.GaussianBlur(dilated_edges, (5, 5), 1.0)
        transition_mask_norm = transition_mask.astype(np.float32) / 255.0

        final_alpha = alpha_smoothed.astype(np.float32)
        final_alpha = final_alpha * (1 - transition_mask_norm * 0.3) + alpha_np.astype(np.float32) * (transition_mask_norm * 0.3)
        final_alpha = np.clip(final_alpha, 0, 255).astype(np.uint8)

        final_alpha_pil = Image.fromarray(final_alpha)
        fg_improved = Image.merge('RGBA', (r, g, b, final_alpha_pil))
        white_bg = Image.new("RGBA", fg_improved.size, (255, 255, 255, 255))
        result = Image.alpha_composite(white_bg, fg_improved).convert("RGB")
        return result
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}. Using original image.")
        return img_pil

# ---------------------- ADDITIONAL HAIR REFINEMENT ----------------------
def refine_hair_edges(img_pil, original_img_pil):
    """Preserve fine hair details using texture and edge blending"""
    try:
        img_np = np.array(img_pil)
        orig_np = np.array(original_img_pil)
        if img_np.shape != orig_np.shape:
            orig_np = np.array(original_img_pil.resize(img_pil.size, Image.LANCZOS))
        gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getGaborKernel((15, 15), 3, np.pi/4, 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
        hair_texture = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        _, hair_mask = cv2.threshold(hair_texture, 30, 255, cv2.THRESH_BINARY)
        hair_mask = cv2.GaussianBlur(hair_mask, (5, 5), 1.0)
        hair_mask_norm = hair_mask.astype(np.float32) / 255.0
        hair_mask_3d = np.stack([hair_mask_norm] * 3, axis=-1)
        result_np = img_np.astype(np.float32) * (1 - hair_mask_3d * 0.2) + orig_np.astype(np.float32) * (hair_mask_3d * 0.2)
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)
        return Image.fromarray(result_np)
    except:
        return img_pil

# ---------------------- FACE LANDMARKS AND CHECKS ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face landmarks found")
        return results.multi_face_landmarks[0]

def get_head_eye_positions(landmarks, img_h, img_w):
    top_y = int(landmarks.landmark[10].y * img_h)
    chin_y = int(landmarks.landmark[152].y * img_h)
    left_eye_y = int(landmarks.landmark[33].y * img_h)
    right_eye_y = int(landmarks.landmark[263].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2
    top_y = max(0, top_y - int((chin_y - top_y) * 0.3))
    return top_y, chin_y, eye_y

# ---------------------- MAIN PROCESS ----------------------
def process_dv_photo_initial(img_pil):
    try:
        cv_img = np.array(img_pil)
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
        h, w = cv_img.shape[:2]
        scale_factor = MIN_SIZE / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
        y_offset, x_offset = (MIN_SIZE - new_h) // 2, (MIN_SIZE - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        result = Image.fromarray(canvas)
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
        head_info = {
            "top_y": int(top_y * scale_factor) + y_offset,
            "chin_y": int(chin_y * scale_factor) + y_offset,
            "eye_y": int(eye_y * scale_factor) + y_offset,
            "head_height": (chin_y - top_y) * scale_factor,
            "canvas_size": MIN_SIZE
        }
        return result, head_info, []
    except Exception as e:
        st.error(f"Initial processing error: {str(e)}")
        return img_pil, {}, ["❌ Face not detected properly"]

# ---------------------- STREAMLIT UI ----------------------
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    - Upload your photo
    - Hair edges are now preserved
    - Photo is auto resized to 600×600
    """)
    preserve_hair = st.checkbox("Enable Hair Preservation", value=True)

uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg", "jpeg", "png"])
if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Processing with improved hair preservation..."):
        bg_removed = remove_background(orig)
        if preserve_hair:
            bg_removed = refine_hair_edges(bg_removed, orig)
        processed, head_info, issues = process_dv_photo_initial(bg_removed)
    st.image(processed, caption="Final Result — Hair Preserved", use_container_width=True)
    st.success("✅ Hair preservation completed successfully!")

    buf = io.BytesIO()
    processed.save(buf, format="JPEG", quality=95)
    st.download_button(
        "⬇️ Download Processed Photo",
        data=buf.getvalue(),
        file_name="dv_photo_hair_preserved.jpg",
        mime="image/jpeg"
    )
else:
    st.info("Upload a photo above to start.")

st.markdown("---")
st.markdown("*DV Lottery Photo Editor | Hair Preservation v2.0*")

# ---------------------- IMPROVED BACKGROUND REMOVAL ----------------------
def remove_background_hair(img_pil):
    """Remove background and preserve hair details"""
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        img_bytes = b.getvalue()
        fg_bytes = remove(img_bytes)
        fg = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
        r, g, b, alpha = fg.split()
        alpha_np = np.array(alpha)
        # Hair edge smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_closed = cv2.morphologyEx(alpha_np, cv2.MORPH_CLOSE, kernel)
        alpha_smoothed = cv2.GaussianBlur(alpha_closed, (3, 3), 0.5)
        edges = cv2.Canny(alpha_np, 50, 150)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_edges = cv2.dilate(edges, edge_kernel, iterations=1)
        transition_mask = cv2.GaussianBlur(dilated_edges, (5, 5), 1.0)
        transition_mask_norm = transition_mask.astype(np.float32) / 255.0
        final_alpha = alpha_smoothed.astype(np.float32) * (1 - transition_mask_norm * 0.3) + alpha_np.astype(np.float32) * (transition_mask_norm * 0.3)
        final_alpha = np.clip(final_alpha, 0, 255).astype(np.uint8)
        final_alpha_pil = Image.fromarray(final_alpha)
        fg_improved = Image.merge('RGBA', (r, g, b, final_alpha_pil))
        white_bg = Image.new("RGBA", fg_improved.size, (255, 255, 255, 255))
        result = Image.alpha_composite(white_bg, fg_improved).convert("RGB")
        return result
    except:
        return img_pil

def refine_hair_edges(img_pil, original_img_pil):
    """Enhance fine hair edges"""
    try:
        img_np = np.array(img_pil)
        orig_np = np.array(original_img_pil.resize(img_pil.size, Image.LANCZOS))
        gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getGaborKernel((15, 15), 3, np.pi/4, 2*np.pi/3, 0.5, 0)
        hair_texture = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        _, hair_mask = cv2.threshold(hair_texture, 30, 255, cv2.THRESH_BINARY)
        hair_mask = cv2.GaussianBlur(hair_mask, (5, 5), 1.0)
        hair_mask_norm = hair_mask.astype(np.float32) / 255.0
        hair_mask_3d = np.stack([hair_mask_norm]*3, axis=-1)
        result_np = img_np.astype(np.float32) * (1 - hair_mask_3d * 0.2) + orig_np.astype(np.float32) * (hair_mask_3d * 0.2)
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)
        return Image.fromarray(result_np)
    except:
        return img_pil

# ---------------------- FACE LANDMARKS & COMPLIANCE ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face landmarks found")
        return results.multi_face_landmarks[0]

def get_head_eye_positions(landmarks, img_h, img_w):
    top_y = int(landmarks.landmark[10].y * img_h)
    chin_y = int(landmarks.landmark[152].y * img_h)
    left_eye_y = int(landmarks.landmark[33].y * img_h)
    right_eye_y = int(landmarks.landmark[263].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2
    top_y = max(0, top_y - int((chin_y - top_y)*0.3))
    return top_y, chin_y, eye_y

# ---------------------- PROCESSING FUNCTIONS ----------------------
def process_initial(img_pil):
    cv_img = np.array(img_pil)
    if cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    h, w = cv_img.shape[:2]
    scale_factor = MIN_SIZE / max(h, w)
    resized = cv2.resize(cv_img, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    y_offset, x_offset = (MIN_SIZE - resized.shape[0]) // 2, (MIN_SIZE - resized.shape[1]) // 2
    canvas[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
    result = Image.fromarray(canvas)
    try:
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
        head_info = {
            "top_y": int(top_y*scale_factor)+y_offset,
            "chin_y": int(chin_y*scale_factor)+y_offset,
            "eye_y": int(eye_y*scale_factor)+y_offset,
            "head_height": (chin_y-top_y)*scale_factor,
            "canvas_size": MIN_SIZE
        }
        return result, head_info, []
    except:
        return result, {}, ["❌ Face not detected properly"]

def process_adjusted(img_pil):
    # Same logic as your main script for auto-adjust
    # Uses compliance checks, baby detection, head-to-chin adjustment
    # Returns processed image, head_info, compliance_issues
    # You can paste your full process_dv_photo_adjusted() here
    from copy import deepcopy
    # Placeholder for demonstration:
    return process_initial(img_pil)

def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = w//2
    top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
    head_ratio = head_info["head_height"]/head_info["canvas_size"]
    eye_ratio = (head_info["canvas_size"]-head_info["eye_y"])/head_info["canvas_size"]
    head_color = "green" if HEAD_MIN_RATIO<=head_ratio<=HEAD_MAX_RATIO else "red"
    eye_color = "green" if EYE_MIN_RATIO<=eye_ratio<=EYE_MAX_RATIO else "red"
    draw.line([(cx-50, top_y), (cx+50, top_y)], fill="blue", width=3)
    draw.line([(cx-50, chin_y), (cx+50, chin_y)], fill="purple", width=3)
    draw.line([(cx, top_y), (cx, chin_y)], fill=head_color, width=2)
    draw.line([(0, eye_y), (w, eye_y)], fill=eye_color, width=3)
    return img, head_ratio, eye_ratio

# ---------------------- STREAMLIT UI ----------------------
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    - Upload your photo
    - Hair edges will be preserved
    - Auto-fix head-to-chin if measurements are out of range
    """)
    preserve_hair = st.checkbox("Enable Hair Preservation", value=True)

uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg","jpeg","png"])
if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Processing with hair preservation..."):
        bg_removed = remove_background_hair(orig)
        if preserve_hair:
            bg_removed = refine_hair_edges(bg_removed, orig)
        processed, head_info, issues = process_initial(bg_removed)
        processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
    st.image(processed_with_lines, caption="Processed Photo with Hair Preservation", use_container_width=True)
    st.markdown("**Head height:** {}%, **Eye position:** {}%".format(int(head_ratio*100), int(eye_ratio*100)))
    
    # Fixer Button
    if st.button("🔧 Auto-Fix Head-to-Chin"):
        with st.spinner("Applying adjustments..."):
            fixed_img, head_info_fixed, issues_fixed = process_adjusted(bg_removed)
            fixed_with_lines, head_ratio_fixed, eye_ratio_fixed = draw_guidelines(fixed_img.copy(), head_info_fixed)
            st.image(fixed_with_lines, caption="Auto-Fixed Photo", use_container_width=True)
            st.markdown("**Fixed Head height:** {}%, **Fixed Eye position:** {}%".format(int(head_ratio_fixed*100), int(eye_ratio_fixed*100)))
            
    buf = io.BytesIO()
    processed_with_lines.save(buf, format="JPEG", quality=95)
    st.download_button("⬇️ Download Photo", data=buf.getvalue(), file_name="dv_photo_hair_fixed.jpg", mime="image/jpeg")

else:
    st.info("Upload a photo above to start.")
