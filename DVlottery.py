import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
warnings.filterwarnings('ignore')

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("DV Lottery Photo Editor — Baby Aware + Auto Fit + Official Guidelines")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69
DPI = 300  # 2x2 inch photo at 300 DPI
mp_face_mesh = mp.solutions.face_mesh

# ---------------------- FACE UTILITIES ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.3, min_tracking_confidence=0.3
    ) as fm:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face landmarks found")
        return results.multi_face_landmarks[0]

def get_head_shoulder_positions(landmarks, img_h, img_w):
    # Head top (forehead)
    top_y = int(landmarks.landmark[10].y * img_h)
    
    # Chin
    chin_y = int(landmarks.landmark[152].y * img_h)
    
    # Eyes
    left_eye_y = int(landmarks.landmark[33].y * img_h)
    right_eye_y = int(landmarks.landmark[263].y * img_h)
    eye_y = (left_eye_y + right_eye_y) // 2
    
    # Shoulders - using neck points and estimating shoulder position
    neck_y = int(landmarks.landmark[10].y * img_h)  # Use forehead as reference
    shoulder_estimate = chin_y + (chin_y - top_y) * 0.8  # Shoulder is ~80% of head height below chin
    
    hair_buffer = int((chin_y - top_y) * 0.25)
    top_y = max(0, top_y - hair_buffer)
    
    return top_y, chin_y, eye_y, int(shoulder_estimate)

# ---------------------- BACKGROUND REMOVAL ----------------------
def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img_pil

# ---------------------- BABY DETECTION ----------------------
def is_baby_photo(landmarks, img_h, img_w):
    """Detects baby facial proportions based on eye and chin placement."""
    try:
        top_y = landmarks.landmark[10].y * img_h
        chin_y = landmarks.landmark[152].y * img_h
        left_eye_y = landmarks.landmark[33].y * img_h
        right_eye_y = landmarks.landmark[263].y * img_h
        eye_y = (left_eye_y + right_eye_y) / 2
        nose_y = landmarks.landmark[1].y * img_h

        face_height = chin_y - top_y
        eye_to_top = eye_y - top_y
        ratio = eye_to_top / face_height
        jaw_ratio = (chin_y - nose_y) / face_height

        # Babies have higher eyes and smaller chin area
        return ratio > 0.42 and jaw_ratio < 0.33
    except:
        return False

# ---------------------- AUTO CROP ----------------------
def auto_crop_dv(img_pil):
    """Auto-crops photo to DV specs with baby detection - shoulder to head only."""
    cv_img = np.array(img_pil)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

    h, w = cv_img.shape[:2]
    
    try:
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y, shoulder_y = get_head_shoulder_positions(landmarks, h, w)
        head_h = chin_y - top_y
        baby_mode = is_baby_photo(landmarks, h, w)
    except Exception as e:
        st.warning(f"Face detection limited: {str(e)}. Using safe crop.")
        # Center crop focusing on upper body
        scale = MIN_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
        # Position to show head and shoulders
        y_offset = max(0, (MIN_SIZE - new_h) // 3)  # Higher position to focus on head
        x_offset = (MIN_SIZE - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        head_info = {"is_baby": False, "top_y": y_offset+50, "chin_y": y_offset+150, 
                    "eye_y": y_offset+100, "head_height": 100, "shoulder_y": y_offset+200}
        return Image.fromarray(canvas), head_info

    # Calculate the area we want to capture (head to shoulders)
    if baby_mode:
        # For babies: head to shoulders only (not full body)
        body_height = shoulder_y - top_y
        target_body_height = MIN_SIZE * 0.75  # Head + shoulders take 75% of frame
    else:
        # For adults: standard head to shoulders
        body_height = shoulder_y - top_y
        target_body_height = MIN_SIZE * 0.70

    scale = target_body_height / body_height

    # Limit scale to prevent over-zooming
    max_scale = MIN_SIZE / min(h, w) * 0.9
    scale = min(scale, max_scale)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    
    # Target eye position - adjusted for shoulder-to-head composition
    if baby_mode:
        target_eye_min = MIN_SIZE - int(0.60 * MIN_SIZE)  # Higher position for babies
        target_eye_max = MIN_SIZE - int(0.50 * MIN_SIZE)
    else:
        target_eye_min = MIN_SIZE - int(EYE_MAX_RATIO * MIN_SIZE)
        target_eye_max = MIN_SIZE - int(EYE_MIN_RATIO * MIN_SIZE)
        
    target_eye = (target_eye_min + target_eye_max) // 2

    # Get positions in resized image
    landmarks_resized = get_face_landmarks(resized)
    top_y, chin_y, eye_y, shoulder_y = get_head_shoulder_positions(landmarks_resized, new_h, new_w)
    
    # Calculate offset to position eyes at target and ensure shoulders are visible but not too much body
    y_offset = target_eye - eye_y
    
    # Ensure we don't crop the head top and show shoulders appropriately
    max_attempts = 3
    for attempt in range(max_attempts):
        head_top_in_frame = top_y + y_offset >= 10  # Small margin from top
        shoulders_in_frame = shoulder_y + y_offset <= MIN_SIZE - 10  # Small margin from bottom
        
        if head_top_in_frame and shoulders_in_frame:
            break
            
        # Adjust positioning
        if not head_top_in_frame:
            y_offset = 10 - top_y  # Move down to show head top
        if not shoulders_in_frame:
            y_offset = MIN_SIZE - 10 - shoulder_y  # Move up to show shoulders
            
        # If still not fitting, zoom out
        if attempt == 1 and (not head_top_in_frame or not shoulders_in_frame):
            scale *= 0.85
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            landmarks_resized = get_face_landmarks(resized)
            top_y, chin_y, eye_y, shoulder_y = get_head_shoulder_positions(landmarks_resized, new_h, new_w)
            y_offset = target_eye - eye_y

    x_offset = (MIN_SIZE - new_w) // 2

    # Final safety check on positioning
    y_offset = max(10 - top_y, min(y_offset, MIN_SIZE - 10 - shoulder_y))
    x_offset = max(0, min(x_offset, MIN_SIZE - new_w))

    # Paste to canvas
    y_start_dst = max(0, y_offset)
    y_end_dst = min(MIN_SIZE, y_offset + new_h)
    x_start_dst = max(0, x_offset)
    x_end_dst = min(MIN_SIZE, x_offset + new_w)

    y_start_src = max(0, -y_offset)
    y_end_src = min(new_h, MIN_SIZE - y_offset)
    x_start_src = max(0, -x_offset)
    x_end_src = min(new_w, MIN_SIZE - x_offset)

    canvas[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
        resized[y_start_src:y_end_src, x_start_src:x_end_src]

    final_top_y = top_y + y_offset
    final_chin_y = chin_y + y_offset
    final_eye_y = eye_y + y_offset
    final_shoulder_y = shoulder_y + y_offset

    head_info = {
        "top_y": final_top_y,
        "chin_y": final_chin_y,
        "eye_y": final_eye_y,
        "shoulder_y": final_shoulder_y,
        "head_height": chin_y - top_y,
        "canvas_size": MIN_SIZE,
        "is_baby": baby_mode
    }
    return Image.fromarray(canvas), head_info

# ---------------------- DRAW DV GUIDELINES ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = w // 2
    top_y, chin_y, eye_y, shoulder_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"], head_info.get("shoulder_y", h-50)
    head_h = head_info["head_height"]
    head_ratio = head_h / h
    eye_ratio = (h - eye_y) / h

    head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
    eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"

    eye_min_px = int(1.125 * DPI)
    eye_max_px = int(1.375 * DPI)
    eye_band_top = h - eye_max_px
    eye_band_bottom = h - eye_min_px

    # Draw key facial lines
    draw.line([(0, top_y), (w, top_y)], fill="red", width=3)
    draw.line([(0, eye_y), (w, eye_y)], fill="red", width=3)
    draw.line([(0, chin_y), (w, chin_y)], fill="red", width=3)
    draw.line([(0, shoulder_y), (w, shoulder_y)], fill="blue", width=3)  # Shoulder line

    # Draw eye position guidelines
    for x in range(0, w, 20):
        draw.line([(x, eye_band_top), (x + 10, eye_band_top)], fill="green", width=2)
        draw.line([(x, eye_band_bottom), (x + 10, eye_band_bottom)], fill="green", width=2)

    # Labels
    draw.text((10, top_y - 25), "Top of Head", fill="red")
    draw.text((10, eye_y - 15), "Eye Line", fill="red")
    draw.text((10, chin_y - 20), "Chin", fill="red")
    draw.text((10, shoulder_y - 20), "Shoulders", fill="blue")
    draw.text((w - 240, eye_band_top - 20), "1 inch to 1-3/8 inch", fill="green")
    draw.text((w - 300, eye_band_bottom + 5), "1-1/8 inch to 1-3/8 inch from bottom", fill="green")

    # Frame and center line
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline="black", width=3)
    draw.line([(cx, 0), (cx, h)], fill="gray", width=1)

    # Inch rulers
    inch_px = DPI
    for i in range(3):
        y = i * inch_px
        draw.line([(0, y), (20, y)], fill="black", width=2)
        draw.text((25, y - 10), f"{i} in", fill="black")
        draw.line([(w - 20, y), (w, y)], fill="black", width=2)
        draw.text((w - 55, y - 10), f"{i} in", fill="black")

    # PASS/FAIL logic
    head_in_frame = (head_info["top_y"] > 5) and (head_info["chin_y"] < h - 5)
    shoulders_in_frame = head_info.get("shoulder_y", h) < h - 5
    
    passed = (
        HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
        and EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
        and head_in_frame
        and shoulders_in_frame
    )

    badge_color = "green" if passed else "red"
    status_text = "PASS" if passed else "FAIL"
    draw.rectangle([(10, 10), (200, 90)], fill="white", outline=badge_color, width=3)
    draw.text((20, 15), status_text, fill=badge_color)
    draw.text((20, 35), f"H:{int(head_ratio*100)}%  E:{int(eye_ratio*100)}%", fill="black")

    if not head_in_frame:
        draw.text((20, 55), "Head cropped - FAIL", fill="red")
    if not shoulders_in_frame:
        draw.text((20, 70), "Shoulders cropped - FAIL", fill="red")

    if head_info.get("is_baby", False):
        draw.text((20, 85), "Baby Mode: Shoulders to Head", fill="orange")

    return img, head_ratio, eye_ratio

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a front-facing photo.
2. Background auto-removed & cropped to 2x2 inch.
3. Detects baby faces & shows shoulders-to-head only.
4. Follows DV lottery requirements.

**DV Requirements:**
- Head height: 50–69% of image
- Eyes: 1-1/8–1-3/8 inch from bottom
- Shoulders to head composition
- White background, neutral expression

**Baby Photos:** Now shows shoulders to head only (no full body).
""")

uploaded = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded:
    orig = Image.open(uploaded).convert("RGB")
    with st.spinner("Processing photo... Shoulders to head composition for DV lottery."):
        bg_removed = remove_background(orig)
        processed, head_info = auto_crop_dv(bg_removed)
        overlay, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(orig, use_column_width=True)
    with col2:
        st.subheader("Processed (600x600 - Shoulders to Head)")
        st.image(overlay, use_column_width=True)

        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="Download Final 600x600 Photo",
            data=buf.getvalue(),
            file_name="dv_photo_final.jpg",
            mime="image/jpeg"
        )
        
    if head_info.get("is_baby", False):
        st.success("👶 Baby face detected - showing shoulders to head only (DV requirement)")
else:
    st.markdown("""
    ## Welcome to the DV Lottery Photo Editor  
    Upload your photo to generate a perfect 600x600 DV-compliant image.  
    
    **Features:**
    - Baby faces auto-detected
    - Shoulders to head composition only
    - No full body photos for babies
    - DV lottery requirement compliant
    
    **Output:** Professional shoulder-to-head portrait suitable for DV lottery application.
    """)

st.markdown("---")
st.caption("DV Lottery Photo Editor | Shoulders to Head | Baby Detection | Official Guidelines")
