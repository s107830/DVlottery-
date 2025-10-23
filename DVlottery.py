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
            min_detection_confidence=0.3, min_tracking_confidence=0.3  # Lowered confidence for better detection
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
        # Use more reliable landmarks for head top and chin
        top_y = int(landmarks.landmark[10].y * img_h)  # Forehead
        chin_y = int(landmarks.landmark[152].y * img_h)  # Chin
        
        # Use multiple eye landmarks for better accuracy
        left_eye_top = int(landmarks.landmark[159].y * img_h)
        right_eye_top = int(landmarks.landmark[386].y * img_h)
        eye_y = (left_eye_top + right_eye_top) // 2
        
        # More conservative hair buffer
        hair_buffer = int((chin_y - top_y) * 0.15)
        top_y = max(0, top_y - hair_buffer)
        
        return top_y, chin_y, eye_y
    except Exception as e:
        st.error(f"Error calculating head/eye positions: {str(e)}")
        raise

def get_ear_mask(landmarks, img_h, img_w):
    """Create a mask to protect ear regions based on MediaPipe landmarks."""
    try:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # Ear landmarks (more comprehensive ear coverage)
        left_ear_points = [
            (int(landmarks.landmark[234].x * img_w), int(landmarks.landmark[234].y * img_h)),
            (int(landmarks.landmark[132].x * img_w), int(landmarks.landmark[132].y * img_h)),
        ]
        right_ear_points = [
            (int(landmarks.landmark[454].x * img_w), int(landmarks.landmark[454].y * img_h)),
            (int(landmarks.landmark[361].x * img_w), int(landmarks.landmark[361].y * img_h)),
        ]
        
        # Create elliptical regions around ears
        for points in [left_ear_points, right_ear_points]:
            if len(points) >= 2:
                x_center = sum(p[0] for p in points) // len(points)
                y_center = sum(p[1] for p in points) // len(points)
                cv2.ellipse(mask, (x_center, y_center), (45, 70), 0, 0, 360, 255, -1)
        
        # Soften mask edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        
        return mask
    except Exception as e:
        st.warning(f"Error creating ear mask: {str(e)}")
        return np.zeros((img_h, img_w), dtype=np.uint8)

def remove_background(img_pil, brightness_factor=1.0):
    try:
        if not REMBG_AVAILABLE:
            st.warning("No background removal available. Using original image.")
            return img_pil.convert("RGB")

        # Convert to numpy array for processing
        cv_img = np.array(img_pil)
        original_h, original_w = cv_img.shape[:2]
        
        # Brightness adjustment
        if brightness_factor != 1.0:
            cv_img = cv2.convertScaleAbs(cv_img, alpha=brightness_factor, beta=0)
        
        # Gentle preprocessing - reduced to avoid artifacts
        cv_img = cv2.GaussianBlur(cv_img, (1, 1), 0)  # Very light blur
        
        # Get ear mask before background removal
        try:
            landmarks = get_face_landmarks(cv_img)
            ear_mask = get_ear_mask(landmarks, original_h, original_w)
        except Exception as e:
            st.warning(f"Could not create ear mask: {str(e)}")
            ear_mask = np.zeros((original_h, original_w), dtype=np.uint8)

        # Convert back to PIL for rembg
        img_for_rembg = Image.fromarray(cv_img)
        
        # Remove background with rembg
        b = io.BytesIO()
        img_for_rembg.save(b, format="PNG")
        fg = Image.open(io.BytesIO(rembg_remove(b.getvalue()))).convert("RGBA")

        # Post-process the alpha mask
        fg_np = np.array(fg)
        alpha = fg_np[:, :, 3]
        
        # Gentle alpha processing
        _, alpha = cv2.threshold(alpha, 150, 255, cv2.THRESH_BINARY)  # Lower threshold
        
        # Soften edges
        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
        
        # Protect ear regions
        if ear_mask.shape == alpha.shape:
            alpha = cv2.bitwise_or(alpha, ear_mask)
        
        # Fill small holes
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        fg_np[:, :, 3] = alpha
        fg = Image.fromarray(fg_np)

        # Composite onto white background
        white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        result = Image.alpha_composite(white_bg, fg).convert("RGB")
        
        return result
        
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}. Using original image.")
        return img_pil.convert("RGB")

# ---------------------- AUTO CROP ----------------------
def auto_crop_dv(img_pil):
    try:
        cv_img = np.array(img_pil)
        
        # Ensure RGB format
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

        h, w = cv_img.shape[:2]
        
        # Get face landmarks
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
        head_h = chin_y - top_y

        # Calculate scale to make head height ~63% of 600px
        target_head_height = MIN_SIZE * 0.63
        scale = target_head_height / head_h
        
        # Limit scaling to reasonable bounds
        scale = max(0.5, min(2.0, scale))
        
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Create white canvas
        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
        
        # Calculate eye position targets
        target_eye_min = MIN_SIZE - int(EYE_MAX_RATIO * MIN_SIZE)
        target_eye_max = MIN_SIZE - int(EYE_MIN_RATIO * MIN_SIZE)
        target_eye = (target_eye_min + target_eye_max) // 2

        # Get positions on resized image
        landmarks_resized = get_face_landmarks(resized)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks_resized, new_h, new_w)
        
        # Calculate offsets for centering
        y_offset = target_eye - eye_y
        x_offset = (MIN_SIZE - new_w) // 2

        # Calculate source and destination regions
        y_start_dst = max(0, y_offset)
        y_end_dst = min(MIN_SIZE, y_offset + new_h)
        x_start_dst = max(0, x_offset)
        x_end_dst = min(MIN_SIZE, x_offset + new_w)

        y_start_src = max(0, -y_offset)
        y_end_src = min(new_h, MIN_SIZE - y_offset)
        x_start_src = max(0, -x_offset)
        x_end_src = min(new_w, MIN_SIZE - x_offset)

        # Copy the image to canvas
        if (y_end_src > y_start_src and x_end_src > x_start_src and 
            y_end_dst > y_start_dst and x_end_dst > x_start_dst):
            canvas[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
                resized[y_start_src:y_end_src, x_start_src:x_end_src]

        # Calculate final positions
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
        # Fallback: simple center crop
        st.warning("Using fallback cropping method")
        img_pil.thumbnail((MIN_SIZE, MIN_SIZE), Image.LANCZOS)
        canvas = Image.new("RGB", (MIN_SIZE, MIN_SIZE), (255, 255, 255))
        canvas.paste(img_pil, ((MIN_SIZE - img_pil.width) // 2, (MIN_SIZE - img_pil.height) // 2))
        
        # Estimate head info for fallback
        head_info = {
            "top_y": MIN_SIZE // 4,
            "chin_y": 3 * MIN_SIZE // 4,
            "eye_y": MIN_SIZE // 2,
            "head_height": MIN_SIZE // 2,
            "canvas_size": MIN_SIZE
        }
        
        return canvas, head_info

# ---------------------- DRAW DV GUIDELINES ----------------------
def draw_guidelines(img, head_info):
    try:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cx = w // 2
        
        if not head_info:
            # Draw basic guidelines if no head info
            draw.rectangle([(0, 0), (w - 1, h - 1)], outline="black", width=3)
            draw.text((10, 10), "No face detection - manual check required", fill="red")
            return img, 0, 0
        
        top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
        head_h = head_info["head_height"]
        head_ratio = head_h / h if h > 0 else 0
        eye_ratio = (h - eye_y) / h if h > 0 else 0

        head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
        eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"

        # Eye position band (1-1/8 to 1-3/8 inch from bottom)
        eye_min_px = int(1.125 * DPI / 2)  # Convert to 600px canvas
        eye_max_px = int(1.375 * DPI / 2)
        eye_band_top = h - eye_max_px
        eye_band_bottom = h - eye_min_px

        # Head guideline lines
        draw.line([(0, top_y), (w, top_y)], fill="red", width=2)
        draw.line([(0, eye_y), (w, eye_y)], fill="red", width=2)
        draw.line([(0, chin_y), (w, chin_y)], fill="red", width=2)

        # Green dashed eye band
        for x in range(0, w, 20):
            if x + 10 <= w:
                draw.line([(x, eye_band_top), (x + 10, eye_band_top)], fill="green", width=2)
                draw.line([(x, eye_band_bottom), (x + 10, eye_band_bottom)], fill="green", width=2)

        # Labels
        draw.text((10, max(10, top_y - 25)), "Top of Head", fill="red")
        draw.text((10, max(10, eye_y - 15)), "Eye Line", fill="red")
        draw.text((10, min(h - 25, chin_y - 20)), "Chin", fill="red")
        
        # Eye band labels
        draw.text((w - 200, max(10, eye_band_top - 20)), "Eye Position Band", fill="green")
        draw.text((w - 280, min(h - 10, eye_band_bottom + 5)), "1-1/8\" to 1-3/8\" from bottom", fill="green")

        # 2x2 box outline & center line
        draw.rectangle([(0, 0), (w - 1, h - 1)], outline="black", width=3)
        draw.line([(cx, 0), (cx, h)], fill="gray", width=1)

        # Vertical inch rulers
        inch_px = DPI / 2  # For 600px 2x2 image
        for i in range(3):
            y = i * inch_px
            if y < h:
                # Left ruler
                draw.line([(0, y), (15, y)], fill="black", width=2)
                draw.text((18, max(10, y - 10)), f"{i}\"", fill="black")
                # Right ruler
                draw.line([(w - 15, y), (w, y)], fill="black", width=2)
                draw.text((w - 35, max(10, y - 10)), f"{i}\"", fill="black")

        # PASS/FAIL indicator
        passed = (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO) and (EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO)
        badge_color = "green" if passed else "red"
        status_text = "PASS" if passed else "FAIL"
        
        draw.rectangle([(10, 10), (160, 70)], fill="white", outline=badge_color, width=3)
        draw.text((20, 15), status_text, fill=badge_color, stroke_width=1)
        draw.text((20, 35), f"Head: {head_ratio:.1%}", fill="black")
        draw.text((20, 50), f"Eyes: {eye_ratio:.1%}", fill="black")

        return img, head_ratio, eye_ratio
        
    except Exception as e:
        st.error(f"Error drawing guidelines: {str(e)}")
        # Return original image if guidelines fail
        return img, 0, 0

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
brightness_factor = st.sidebar.slider("Brightness Adjustment", 0.8, 1.2, 1.0, 0.05)

uploaded = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        orig = Image.open(uploaded).convert("RGB")
        
        if orig.size[0] < MIN_SIZE or orig.size[1] < MIN_SIZE:
            st.warning("Image is too small. Please upload a photo at least 600x600 pixels.")
        else:
            with st.spinner("Processing photo..."):
                # Process image
                bg_removed = remove_background(orig, brightness_factor=brightness_factor)
                processed, head_info = auto_crop_dv(bg_removed)
                overlay, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
            
            # Display results
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
                
            # Show compliance status
            st.subheader("Compliance Check")
            head_ok = HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
            eye_ok = EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
            
            if head_ok and eye_ok:
                st.success("✓ Photo meets DV Lottery requirements!")
            else:
                st.error("✗ Photo does not meet all requirements:")
                if not head_ok:
                    st.error(f"- Head height should be {HEAD_MIN_RATIO:.0%}-{HEAD_MAX_RATIO:.0%} of image (currently {head_ratio:.1%})")
                if not eye_ok:
                    st.error(f"- Eyes should be {EYE_MIN_RATIO:.0%}-{EYE_MAX_RATIO:.0%} from bottom (currently {eye_ratio:.1%})")
                    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Try uploading a different photo with clear front-facing face detection")
else:
    st.markdown("""
    ## Welcome to the DV Lottery Photo Editor  
    Upload your photo above to generate a perfect 600x600 DV-compliant image  
    with official guideline lines, inch rulers, and pass/fail verification.
    """)

st.markdown("---")
st.caption("DV Lottery Photo Editor | Official 2x2 inch Compliance Visualizer")
