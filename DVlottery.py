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
st.title("DV Lottery Photo Editor — Shoulders to Head Only")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69
DPI = 300
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
    
    # Shoulders - more accurate estimation
    # Use face height to estimate shoulder position
    face_height = chin_y - top_y
    shoulder_y = chin_y + int(face_height * 0.8)  # Shoulders are about 80% of face height below chin
    
    hair_buffer = int(face_height * 0.15)
    top_y = max(0, top_y - hair_buffer)
    
    return top_y, chin_y, eye_y, shoulder_y

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
    """Detects baby facial proportions."""
    try:
        top_y = landmarks.landmark[10].y * img_h
        chin_y = landmarks.landmark[152].y * img_h
        left_eye_y = landmarks.landmark[33].y * img_h
        right_eye_y = landmarks.landmark[263].y * img_h
        eye_y = (left_eye_y + right_eye_y) / 2

        face_height = chin_y - top_y
        eye_to_top = eye_y - top_y
        ratio = eye_to_top / face_height

        return ratio > 0.40  # Babies have higher eye placement
    except:
        return False

# ---------------------- AUTO CROP - SHOULDERS TO HEAD ONLY ----------------------
def auto_crop_dv(img_pil):
    """Auto-crops photo to show shoulders to head only."""
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
        # Fallback: crop center with focus on upper body
        scale = MIN_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
        # Position to show upper body only
        y_offset = max(0, MIN_SIZE - new_h) // 4  # Higher position for head focus
        x_offset = (MIN_SIZE - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        head_info = {"is_baby": False, "top_y": y_offset+50, "chin_y": y_offset+150, 
                    "eye_y": y_offset+100, "head_height": 100, "shoulder_y": y_offset+250}
        return Image.fromarray(canvas), head_info

    # Calculate the shoulder-to-head area we want to capture
    shoulder_to_head_height = shoulder_y - top_y
    
    # Calculate scale to fit shoulder-to-head area in the frame
    # We want shoulders to head to take about 80-90% of the frame height
    target_height = MIN_SIZE * 0.85
    scale = target_height / shoulder_to_head_height
    
    # Apply scaling
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Get new positions after scaling
    landmarks_resized = get_face_landmarks(resized)
    top_y, chin_y, eye_y, shoulder_y = get_head_shoulder_positions(landmarks_resized, new_h, new_w)
    
    # Create white canvas
    canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
    
    # Calculate positioning to center the shoulder-to-head area
    # We want the shoulder-to-head area centered vertically
    shoulder_to_head_center = (top_y + shoulder_y) // 2
    target_center = MIN_SIZE // 2
    y_offset = target_center - shoulder_to_head_center
    
    # Horizontal centering
    x_offset = (MIN_SIZE - new_w) // 2
    
    # Ensure we don't crop important parts
    if top_y + y_offset < 0:
        y_offset = -top_y + 10  # Add small margin from top
    if shoulder_y + y_offset > MIN_SIZE:
        y_offset = MIN_SIZE - shoulder_y - 10  # Add small margin from bottom
    
    # Final positioning limits
    y_offset = max(0, min(y_offset, MIN_SIZE - new_h))
    x_offset = max(0, min(x_offset, MIN_SIZE - new_w))
    
    # Paste the resized image onto canvas
    y_start_dst = max(0, y_offset)
    y_end_dst = min(MIN_SIZE, y_offset + new_h)
    x_start_dst = max(0, x_offset)
    x_end_dst = min(MIN_SIZE, x_offset + new_w)

    y_start_src = max(0, -y_offset)
    y_end_src = min(new_h, MIN_SIZE - y_offset)
    x_start_src = max(0, -x_offset)
    x_end_src = min(new_w, MIN_SIZE - x_offset)

    if y_end_src > y_start_src and x_end_src > x_start_src:
        canvas[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
            resized[y_start_src:y_end_src, x_start_src:x_end_src]

    # Calculate final positions for guidelines
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

# ---------------------- DRAW GUIDELINES ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    top_y = head_info["top_y"]
    chin_y = head_info["chin_y"] 
    eye_y = head_info["eye_y"]
    shoulder_y = head_info["shoulder_y"]
    head_h = head_info["head_height"]
    
    head_ratio = head_h / h
    eye_ratio = (h - eye_y) / h

    # Draw the key composition lines
    draw.line([(0, top_y), (w, top_y)], fill="red", width=2)
    draw.line([(0, eye_y), (w, eye_y)], fill="orange", width=2)
    draw.line([(0, chin_y), (w, chin_y)], fill="red", width=2)
    draw.line([(0, shoulder_y), (w, shoulder_y)], fill="blue", width=3)
    
    # Draw composition area
    draw.rectangle([(10, top_y), (w-10, shoulder_y)], outline="green", width=2)
    
    # Labels
    draw.text((15, top_y - 25), "HEAD TOP", fill="red")
    draw.text((15, eye_y - 15), "EYES", fill="orange") 
    draw.text((15, chin_y - 20), "CHIN", fill="red")
    draw.text((15, shoulder_y - 20), "SHOULDERS", fill="blue")
    draw.text((w//2 - 100, shoulder_y + 10), "SHOULDERS TO HEAD AREA", fill="green")
    
    # PASS/FAIL status
    head_in_frame = top_y > 10 and chin_y < h - 10
    shoulders_in_frame = shoulder_y < h - 10
    head_size_ok = HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
    eye_position_ok = EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
    
    passed = head_in_frame and shoulders_in_frame and head_size_ok and eye_position_ok
    
    # Status box
    status_color = "green" if passed else "red"
    draw.rectangle([(10, 10), (250, 100)], fill="white", outline=status_color, width=3)
    draw.text((20, 15), "PASS" if passed else "FAIL", fill=status_color)
    draw.text((20, 35), f"Head: {int(head_ratio*100)}%", fill="black")
    draw.text((20, 50), f"Eyes: {int(eye_ratio*100)}%", fill="black")
    draw.text((20, 65), f"Shoulders: {'OK' if shoulders_in_frame else 'CROPPED'}", fill="black")
    
    if head_info.get("is_baby", False):
        draw.text((20, 80), "BABY MODE", fill="orange")

    return img, head_ratio, eye_ratio

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.header("DV Lottery Photo Requirements")
st.sidebar.markdown("""
**Composition:**
- Shoulders to head only
- Head height: 50-69% of image
- Eyes: 56-69% from bottom
- Centered, front-facing
- White background

**For Baby Photos:**
- Same shoulder-to-head composition
- No full body photos
- Neutral expression
- Eyes open and visible
""")

uploaded = st.file_uploader("Upload Photo (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    original_img = Image.open(uploaded).convert("RGB")
    
    with st.spinner("🔄 Processing... Cropping shoulders to head only"):
        # Remove background
        no_bg_img = remove_background(original_img)
        
        # Auto crop to shoulders-to-head
        final_img, head_data = auto_crop_dv(no_bg_img)
        
        # Add guidelines
        guided_img, head_ratio, eye_ratio = draw_guidelines(final_img.copy(), head_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Photo")
        st.image(original_img, use_column_width=True)
        
    with col2:
        st.subheader("✅ Processed (600×600px)")
        st.image(guided_img, use_column_width=True)
        
        # Download button
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="📥 Download DV Photo",
            data=buf.getvalue(),
            file_name="dv_lottery_photo.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    
    # Show status information
    st.info(f"""
    **Photo Analysis:**
    - Head size: {int(head_ratio*100)}% of image {'✅' if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else '❌'}
    - Eye position: {int(eye_ratio*100)}% from bottom {'✅' if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else '❌'}
    - Composition: {'Shoulders to head ✅' if head_data.get('shoulder_y', 0) < MIN_SIZE - 10 else 'Body too long ❌'}
    - Baby detected: {'Yes 👶' if head_data.get('is_baby', False) else 'No'}
    """)

else:
    st.markdown("""
    ## 📷 DV Lottery Photo Editor
    
    **Get perfect shoulders-to-head photos for DV Lottery applications**
    
    Simply upload a photo and we'll automatically:
    - Remove background
    - Crop to show shoulders to head only  
    - Resize to 600×600 pixels (2×2 inches)
    - Position eyes correctly
    - Ensure DV lottery compliance
    
    *No full body photos - shoulders to head composition only*
    """)

st.markdown("---")
st.caption("DV Lottery Photo Editor v2.0 | Shoulders to Head Composition | Official Guidelines")
