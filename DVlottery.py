import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import mediapipe as mp
import io
import warnings
import sys
import cv2

# Check for cv2 import
try:
    import cv2
except ImportError as e:
    st.error(f"Failed to import OpenCV: {str(e)}. Please ensure opencv-python-headless is installed.")
    sys.exit(1)

# Check for rembg import
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError as e:
    REMBG_AVAILABLE = False
    st.error(f"Failed to import rembg: {str(e)}. Please ensure rembg is installed.")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("DV Lottery Photo Editor — Auto Correction & Official DV Guidelines")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69
DPI = 300  # 2x2 inch photo at 300 DPI
mp_face_mesh = mp.solutions.face_mesh

# ---------------------- FACE UTILITIES ----------------------
def get_face_landmarks(cv_img):
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.4, min_tracking_confidence=0.4
        ) as fm:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = fm.process(img_rgb)
            if not results.multi_face_landmarks:
                raise Exception("No face landmarks found")
            return results.multi_face_landmarks[0]
    except Exception as e:
        st.error(f"Face detection failed: {str(e)}")
        raise

def get_head_eye_positions(landmarks, img_h, img_w):
    try:
        top_y = int(landmarks.landmark[10].y * img_h)
        chin_y = int(landmarks.landmark[152].y * img_h)
        left_eye_y = int(landmarks.landmark[33].y * img_h)
        right_eye_y = int(landmarks.landmark[263].y * img_h)
        eye_y = (left_eye_y + right_eye_y) // 2
        hair_buffer = int((chin_y - top_y) * 0.25)
        top_y = max(0, top_y - hair_buffer)
        return top_y, chin_y, eye_y
    except Exception as e:
        st.error(f"Error calculating head/eye positions: {str(e)}")
        raise

def remove_background(img_pil, brightness_factor=1.0):
    try:
        if REMBG_AVAILABLE:
            # Convert to numpy array for processing
            cv_img = np.array(img_pil)
            h, w = cv_img.shape[:2]

            # Get face landmarks to determine hairline region
            landmarks = get_face_landmarks(cv_img)
            top_y, _, _ = get_head_eye_positions(landmarks, h, w)
            hair_region_height = int(h * 0.3)  # Adjust top 30% of image for hair brightness

            # Apply localized brightness boost to hair region
            if brightness_factor != 1.0:
                cv_img_top = cv_img[:hair_region_height, :, :].astype(float)
                cv_img_top = cv_img_top * brightness_factor
                cv_img_top = np.clip(cv_img_top, 0, 255).astype(np.uint8)
                cv_img[:hair_region_height, :, :] = cv_img_top

            # Convert back to PIL for rembg
            img_pil = Image.fromarray(cv_img)

            # Remove background with rembg
            b = io.BytesIO()
            img_pil.save(b, format="PNG")
            fg = Image.open(io.BytesIO(rembg_remove(b.getvalue()))).convert("RGBA")

            # Preserve hair edges with minimal processing
            fg_np = np.array(fg)
            alpha = fg_np[:, :, 3] / 255.0
            alpha = cv2.GaussianBlur(alpha, (3, 3), sigmaX=0.5)
            alpha = np.stack((alpha, alpha, alpha), axis=2)
            white_bg = np.full(fg_np.shape[:2] + (3,), 255, dtype=np.uint8)
            result_np = (fg_np[:, :, :3].astype(float) * alpha + white_bg.astype(float) * (1 - alpha)).astype(np.uint8)
            result = Image.fromarray(result_np)
            return result
        else:
            st.warning("No background removal available. Using original image.")
            return img_pil.convert("RGB")
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}. Using original image.")
        return img_pil.convert("RGB")

# ---------------------- AUTO CROP ----------------------
def auto_crop_dv(img_pil):
    try:
        cv_img = np.array(img_pil)
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

        h, w = cv_img.shape[:2]
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
        head_h = chin_y - top_y

        # Target head height ~63% of 600px = 378px
        target_head = MIN_SIZE * 0.63
        scale = target_head / head_h
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
        target_eye_min = MIN_SIZE - int(EYE_MAX_RATIO * MIN_SIZE)
        target_eye_max = MIN_SIZE - int(EYE_MIN_RATIO * MIN_SIZE)
        target_eye = (target_eye_min + target_eye_max) // 2

        landmarks_resized = get_face_landmarks(resized)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks_resized, new_h, new_w)
        y_offset = target_eye - eye_y
        x_offset = (MIN_SIZE - new_w) // 2

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

        head_info = {
            "top_y": final_top_y,
            "chin_y": final_chin_y,
            "eye_y": final_eye_y,
            "head_height": chin_y - top_y,
            "canvas_size": MIN_SIZE
        }
        return Image.fromarray(canvas), head_info
    except Exception as e:
        st.error(f"Auto-cropping failed: {str(e)}")
        raise

# ---------------------- DRAW DV GUIDELINES ----------------------
def draw_guidelines(img, head_info):
    try:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cx = w // 2
        top_y, chin_y, ear_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
        head_h = head_info["head_height"]
        head_ratio = head_h / h
        eye_ratio = (h - ear_y) / h

        head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
        eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"

        eye_min_px = int(1.125 * DPI)
        eye_max_px = int(1.375 * DPI)
        eye_band_top = h - eye_max_px
        eye_band_bottom = h - eye_min_px

        # Red guideline lines
        draw.line([(0, top_y), (w, top_y)], fill="red", width=3)
        draw.line([(0, ear_y), (w, ear_y)], fill="red", width=3)
        draw.line([(0, chin_y), (w, chin_y)], fill="red", width=3)

        # Green dashed eye band
        for x in range(0, w, 20):
            draw.line([(x, eye_band_top), (x + 10, eye_band_top)], fill="green", width=2)
            draw.line([(x, eye_band_bottom), (x + 10, eye_band_bottom)], fill="green", width=2)

        # Labels (ASCII-safe)
        draw.text((10, top_y - 25), "Top of Head", fill="red")
        draw.text((10, ear_y - 15), "Eye Line", fill="red")
        draw.text((10, chin_y - 20), "Chin", fill="red")
        draw.text((w - 240, eye_band_top - 20), "1 inch to 1-3/8 inch", fill="green")
        draw.text((w - 300, eye_band_bottom + 5), "1-1/8 inch to 1-3/8 inch from bottom", fill="green")

        # 2x2 box outline & center line
        draw.rectangle([(0, 0), (w - 1, h - 1)], outline="black", width=3)
        draw.line([(cx, 0), (cx, h)], fill="gray", width=1)

        # Vertical inch rulers (left & right)
        inch_px = DPI
        for i in range(3):
            y = i * inch_px
            draw.line([(0, y), (20, y)], fill="black", width=2)
            draw.text((25, y - 10), f"{i} in", fill="black")
            draw.line([(w - 20, y), (w, y)], fill="black", width=2)
            draw.text((w - 55, y - 10), f"{i} in", fill="black")

        # PASS/FAIL box (plain ASCII)
        passed = (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO) and (EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO)
        badge_color = "green" if passed else "red"
        status_text = "PASS" if passed else "FAIL"
        draw.rectangle([(10, 10), (170, 60)], fill="white", outline=badge_color, width=3)
        draw.text((25, 20), status_text, fill=badge_color)
        draw.text((25, 40), f"H:{int(head_ratio*100)}%  E:{int(eye_ratio*100)}%", fill="black")

        return img, head_ratio, eye_ratio
    except Exception as e:
        st.error(f"Error drawing guidelines: {str(e)}")
        raise

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a clear front-facing photo.
2. The tool removes the background & centers your face.
3. Crops & scales to official 2x2 inch (600x600 px) size.
4. Draws DV guidelines and compliance ruler.

**DV Requirements:**
- Head height: 50–69% of image  
- Eyes: 1-1/8–1-3/8 inch from bottom  
- Plain white background  
- Neutral expression, both eyes open  
- No glasses, hats, or shadows
""")

# Add brightness adjustment slider
st.sidebar.header("Adjustments")
brightness_factor = st.sidebar.slider("Brightness Adjustment", 0.8, 1.5, 1.2, 0.05)  # Increased default to 1.2 for hair

uploaded = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        orig = Image.open(uploaded).convert("RGB")
        if orig.size[0] < MIN_SIZE or orig.size[1] < MIN_SIZE:
            st.warning("Image is too small. Please upload a photo at least 600x600 pixels.")
        else:
            with st.spinner("Processing photo..."):
                bg_removed = remove_background(orig, brightness_factor=brightness_factor)
                processed, head_info = auto_crop_dv(bg_removed)
                overlay, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(orig, use_column_width=True)
            with col2:
                st.subheader("Processed (600x600)")
                st.image(overlay, use_column_width=True)

                # Save with DPI metadata for DV compliance
                buf = io.BytesIO()
                processed.save(buf, format="JPEG", quality=95, dpi=(DPI, DPI))
                st.download_button(
                    label="Download Final 600x600 Photo",
                    data=buf.getvalue(),
                    file_name="dv_photo_final.jpg",
                    mime="image/jpeg"
                )
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.markdown("""
    ## Welcome to the DV Lottery Photo Editor  
    Upload your photo above to generate a perfect 600x600 DV-compliant image  
    with official guideline lines, inch rulers, and pass/fail verification.
    """)

st.markdown("---")
st.caption("DV Lottery Photo Editor | Official 2x2 inch Compliance Visualizer (ASCII-safe)")
