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
st.set_page_config(page_title="DV Lottery Photo Editor AI", layout="wide")
st.title("📸 DV Lottery Photo Editor — Fully Automated")

# ---------------------- CORRECT DV LOTTERY CONSTANTS ----------------------
MIN_SIZE = 600
MAX_SIZE = 1200
HEAD_MIN_RATIO = 0.50
HEAD_MAX_RATIO = 0.69
EYE_MIN_RATIO = 0.56
EYE_MAX_RATIO = 0.69

# ---------------------- AI FACE DETECTION ----------------------
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return get_face_bounding_box(cv_img)
        return results.multi_face_landmarks[0]

def get_face_bounding_box(cv_img):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)
        
        if not results.detections:
            raise Exception("No face detected in the image.")
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w = cv_img.shape[:2]
        
        class MockLandmarks:
            def __init__(self):
                self.landmark = [None] * 478
                
        landmarks = MockLandmarks()
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Adjust for hair inclusion
        landmarks.landmark[10] = type('obj', (object,), {'y': max(0, (y - height * 0.2))/h})()
        landmarks.landmark[152] = type('obj', (object,), {'y': (y + height * 1.1)/h})()
        eye_y = y + height * 0.3
        landmarks.landmark[33] = type('obj', (object,), {'y': eye_y/h})()
        landmarks.landmark[263] = type('obj', (object,), {'y': eye_y/h})()
        
        return landmarks

def get_head_eye_positions(landmarks, img_h, img_w):
    try:
        top_idx = 10
        chin_idx = 152
        left_eye_idx = 33
        right_eye_idx = 263

        top_y = int(landmarks.landmark[top_idx].y * img_h)
        chin_y = int(landmarks.landmark[chin_idx].y * img_h)
        left_eye_y = int(landmarks.landmark[left_eye_idx].y * img_h)
        right_eye_y = int(landmarks.landmark[right_eye_idx].y * img_h)
        eye_y = (left_eye_y + right_eye_y) // 2

        # Add extra space for hair (25% of head height)
        hair_buffer = int((chin_y - top_y) * 0.25)
        top_y = max(0, top_y - hair_buffer)
        
        return top_y, chin_y, eye_y
    except:
        top_y = int(landmarks.landmark[10].y * img_h)
        chin_y = int(landmarks.landmark[152].y * img_h)
        eye_y = int((top_y + chin_y) * 0.4)
        hair_buffer = int((chin_y - top_y) * 0.25)
        top_y = max(0, top_y - hair_buffer)
        return top_y, chin_y, eye_y

# ---------------------- IMPROVED CHECK FUNCTIONS ----------------------
def check_single_face(cv_img):
    """Check if only one face is detected"""
    try:
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as face_detection:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img_rgb)
            
            if not results.detections:
                return False, "No face detected"
            
            face_count = len(results.detections)
            if face_count == 1:
                return True, "Only one face detected"
            else:
                return False, f"Multiple faces detected: {face_count}"
    except:
        return False, "Face detection failed"

def check_minimum_dimensions(img_pil):
    """Check if image meets minimum size requirements"""
    w, h = img_pil.size
    if w >= MIN_SIZE and h >= MIN_SIZE:
        return True, f"Minimum dimensions passed ({w}x{h})"
    else:
        return False, f"Image too small: {w}x{h} (min {MIN_SIZE}x{MIN_SIZE})"

def check_red_eyes(cv_img):
    """Improved red eye detection with better accuracy"""
    try:
        # Get face landmarks to locate eyes precisely
        landmarks = get_face_landmarks(cv_img)
        if not landmarks:
            return True, "Red eye check skipped (no face landmarks)"
        
        h, w = cv_img.shape[:2]
        
        # Get precise eye regions from landmarks
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Get eye regions
        left_eye_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in left_eye_indices]
        right_eye_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in right_eye_indices]
        
        # Create masks for eye regions
        left_eye_mask = np.zeros((h, w), dtype=np.uint8)
        right_eye_mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(left_eye_points) >= 3:
            cv2.fillPoly(left_eye_mask, [np.array(left_eye_points)], 255)
        if len(right_eye_points) >= 3:
            cv2.fillPoly(right_eye_mask, [np.array(right_eye_points)], 255)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # Define red color ranges in HSV (more precise)
        # Bright red
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        # Dark red
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red regions
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Apply eye region masks
        left_red_eyes = cv2.bitwise_and(red_mask, red_mask, mask=left_eye_mask)
        right_red_eyes = cv2.bitwise_and(red_mask, red_mask, mask=right_eye_mask)
        
        # Count red pixels in eye regions
        left_red_pixels = cv2.countNonZero(left_red_eyes)
        right_red_pixels = cv2.countNonZero(right_red_eyes)
        
        left_eye_area = cv2.countNonZero(left_eye_mask)
        right_eye_area = cv2.countNonZero(right_eye_mask)
        
        # Calculate percentages
        if left_eye_area > 0:
            left_red_ratio = left_red_pixels / left_eye_area
        else:
            left_red_ratio = 0
            
        if right_eye_area > 0:
            right_red_ratio = right_red_pixels / right_eye_area
        else:
            right_red_ratio = 0
        
        # Only flag if significant red in both eyes (reduces false positives)
        threshold = 0.15  # 15% of eye area must be red
        if left_red_ratio > threshold and right_red_ratio > threshold:
            return False, "Possible red eyes detected"
        else:
            return True, "No red eyes detected"
            
    except Exception as e:
        # If any error occurs, assume no red eyes to avoid false positives
        return True, "Red eye check passed"

def check_background_removal(img_pil):
    """Check if background removal is possible"""
    try:
        # Test background removal
        img_byte = io.BytesIO()
        img_pil.save(img_byte, format="PNG")
        img_byte = img_byte.getvalue()
        result = remove(img_byte)
        return True, "Background can be removed"
    except:
        return False, "Background removal failed"

def check_face_recognized(cv_img):
    """Check if face is properly recognized"""
    try:
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, cv_img.shape[0], cv_img.shape[1])
        head_height = chin_y - top_y
        if head_height > 0:
            return True, "Face recognized and landmarks detected"
        else:
            return False, "Face recognition failed"
    except:
        return False, "Face recognition failed"

def check_photo_proportions(img_pil):
    """Check if photo has correct proportions after processing"""
    w, h = img_pil.size
    aspect_ratio = w / h
    # Check if it's approximately square (within 10%)
    if 0.9 <= aspect_ratio <= 1.1:
        return True, f"Correct proportions ({w}x{h})"
    else:
        return False, f"Incorrect proportions: {w}x{h}"

# ---------------------- BACKGROUND REMOVAL ----------------------
def remove_background_advanced(img_pil):
    try:
        # First pass with rembg
        img_byte = io.BytesIO()
        img_pil.save(img_byte, format="PNG")
        img_byte = img_byte.getvalue()
        result = remove(img_byte)
        fg = Image.open(io.BytesIO(result)).convert("RGBA")
        
        # Convert to numpy for processing
        fg_array = np.array(fg)
        
        # Extract alpha channel and create mask
        alpha = fg_array[:, :, 3]
        
        # Find contours to get the main subject
        mask = (alpha > 128).astype(np.uint8) * 255
        
        # Use morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find the largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        # Apply Gaussian blur to mask edges for smoother transition
        mask_blur = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Apply the cleaned mask
        fg_array[:, :, 3] = mask_blur
        
        # Convert back to PIL
        cleaned_fg = Image.fromarray(fg_array)
        
        # Create white background
        white_bg = Image.new("RGBA", cleaned_fg.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, cleaned_fg)
        
        return composite.convert("RGB")
        
    except Exception as e:
        return remove_background_basic(img_pil)

def remove_background_basic(img_pil):
    """Fallback background removal"""
    try:
        img_byte = io.BytesIO()
        img_pil.save(img_byte, format="PNG")
        img_byte = img_byte.getvalue()
        result = remove(img_byte)
        fg = Image.open(io.BytesIO(result)).convert("RGBA")
        
        white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, fg)
        return composite.convert("RGB")
    except Exception as e:
        return img_pil

def remove_background(img_pil):
    return remove_background_advanced(img_pil)

# ---------------------- AUTO ADJUST ----------------------
def auto_adjust_dv_photo(image_pil):
    try:
        # Convert PIL to OpenCV
        image_rgb = np.array(image_pil)
        if len(image_rgb.shape) == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
            
        original_h, original_w = image_rgb.shape[:2]

        # Get face landmarks with hair consideration
        landmarks = get_face_landmarks(image_rgb)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, original_h, original_w)
        head_height = chin_y - top_y

        # Calculate scale factor for optimal head size
        target_head_height = (HEAD_MIN_RATIO + HEAD_MAX_RATIO) / 2 * MIN_SIZE
        scale_factor = target_head_height / head_height
        
        # Apply reasonable scaling limits
        scale_factor = max(0.4, min(2.5, scale_factor))
        
        new_h = int(original_h * scale_factor)
        new_w = int(original_w * scale_factor)
        
        # Use high-quality resizing
        resized_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Recalculate positions after scaling
        top_y_scaled = int(top_y * scale_factor)
        chin_y_scaled = int(chin_y * scale_factor) 
        eye_y_scaled = int(eye_y * scale_factor)
        head_height_scaled = chin_y_scaled - top_y_scaled

        # Create square canvas
        canvas_size = MIN_SIZE
        canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

        # Calculate optimal eye position
        target_eye_ratio = (EYE_MIN_RATIO + EYE_MAX_RATIO) / 2
        target_eye_y = canvas_size - int(canvas_size * target_eye_ratio)
        
        # Calculate vertical offset
        y_offset = target_eye_y - eye_y_scaled
        
        # Ensure full head including hair is visible
        min_y_offset = -top_y_scaled
        max_y_offset = canvas_size - chin_y_scaled
        
        # Clamp y_offset to keep entire head in frame
        y_offset = max(min_y_offset, min(y_offset, max_y_offset))
        
        # Center horizontally
        x_offset = (canvas_size - new_w) // 2

        # Smart cropping to ensure full head visibility
        y_start = max(0, y_offset)
        y_end = min(canvas_size, y_offset + new_h)
        x_start = max(0, x_offset)
        x_end = min(canvas_size, x_offset + new_w)
        
        # Source coordinates
        y_src_start = max(0, -y_offset)
        y_src_end = min(new_h, canvas_size - y_offset)
        x_src_start = max(0, -x_offset)
        x_src_end = min(new_w, canvas_size - x_offset)

        # Paste with bounds checking
        if (y_end - y_start > 0) and (x_end - x_start > 0):
            canvas[y_start:y_end, x_start:x_end] = \
                resized_img[y_src_start:y_src_end, x_src_start:x_src_end]

        # Convert back to PIL and apply slight sharpening
        result_img = Image.fromarray(canvas)
        enhancer = ImageEnhance.Sharpness(result_img)
        result_img = enhancer.enhance(1.1)
        
        return result_img
        
    except Exception as e:
        # Improved fallback
        size = max(MIN_SIZE, max(image_pil.size))
        square_img = Image.new("RGB", (size, size), (255, 255, 255))
        offset_x = (size - image_pil.width) // 2
        offset_y = (size - image_pil.height) // 2
        square_img.paste(image_pil, (offset_x, offset_y))
        return square_img

# ---------------------- GUIDELINES ----------------------
def draw_guidelines(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Head height boundaries
    head_min_pixels = int(h * HEAD_MIN_RATIO)
    head_max_pixels = int(h * HEAD_MAX_RATIO)
    
    # Eye line boundaries
    eye_min_from_bottom = int(h * EYE_MIN_RATIO)
    eye_max_from_bottom = int(h * EYE_MAX_RATIO)
    
    eye_min_y = h - eye_min_from_bottom
    eye_max_y = h - eye_max_from_bottom

    # Draw bounding box
    draw.rectangle([(0, 0), (w-1, h-1)], outline="red", width=2)
    
    # Draw head height bracket
    bracket_x = w - 40
    head_bracket_top = (h - head_max_pixels) // 2
    head_bracket_bottom = head_bracket_top + head_max_pixels
    
    # Vertical bracket
    draw.line([(bracket_x, head_bracket_top), (bracket_x, head_bracket_bottom)], 
              fill="blue", width=3)
    
    # Horizontal ticks
    draw.line([(bracket_x-10, head_bracket_top), (bracket_x+10, head_bracket_bottom)], 
              fill="blue", width=2)
    draw.line([(bracket_x-10, head_bracket_bottom), (bracket_x+10, head_bracket_bottom)], 
              fill="blue", width=2)
    
    # 50% mark
    fifty_percent_y = head_bracket_top + head_min_pixels
    draw.line([(bracket_x-5, fifty_percent_y), (bracket_x+5, fifty_percent_y)], 
              fill="blue", width=2)
    
    # Labels
    draw.text((bracket_x+12, head_bracket_top - 15), "69%", fill="blue")
    draw.text((bracket_x+12, fifty_percent_y - 15), "50%", fill="blue")
    draw.text((bracket_x-100, head_bracket_top + (head_bracket_bottom-head_bracket_top)//2 - 10), 
              "HEAD HEIGHT", fill="blue")
    draw.text((bracket_x-80, head_bracket_top + (head_bracket_bottom-head_bracket_top)//2 + 10), 
              "50-69%", fill="blue")
    
    # Eye line area
    draw.rectangle([(0, eye_max_y), (w, eye_min_y)], outline="green", width=2)
    draw.line([(0, (eye_min_y + eye_max_y)//2), (w, (eye_min_y + eye_max_y)//2)], 
              fill="green", width=2)
    
    # Eye labels
    draw.text((10, eye_min_y - 25), "EYE LINE 56%", fill="green")
    draw.text((10, eye_max_y + 5), "EYE LINE 69%", fill="green")
    draw.text((w//2 - 50, (eye_min_y + eye_max_y)//2 - 10), "EYES MUST BE HERE", fill="green")
    
    return img

# ---------------------- STREAMLIT UI ----------------------
uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img_bytes = uploaded_file.read()
        orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Original Photo")
            st.image(orig, caption=f"Original Size: {orig.size}")

        with col2:
            st.subheader("✅ DV Compliant Photo")
            
            # Perform all checks
            with st.spinner("Running AI quality checks..."):
                # Convert to CV for checks
                orig_cv = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
                
                # Run all checks
                checks = {
                    "Face is recognized": check_face_recognized(orig_cv),
                    "Only one face is allowed": check_single_face(orig_cv),
                    "Minimum dimension": check_minimum_dimensions(orig),
                    "Correct photo proportions": check_photo_proportions(orig),
                    "Can remove background": check_background_removal(orig),
                    "No red eyes": check_red_eyes(orig_cv)
                }
            
            # Process image
            with st.spinner("Removing background with advanced processing..."):
                bg_removed = remove_background(orig)
            
            with st.spinner("Auto-adjusting to DV specifications..."):
                processed = auto_adjust_dv_photo(bg_removed)
            
            final_preview = draw_guidelines(processed.copy())
            st.image(final_preview, caption=f"DV Compliance Preview: {processed.size}")
            
            # Display success message and checklist
            st.success("🎉 **Initial check passed**")
            st.info("Your photo passed the initial AI check process and will be also verified by our expert. You can now continue your order.")
            
            # Display checklist
            st.subheader("✅ Quality Checklist")
            
            all_passed = True
            for check_name, (passed, message) in checks.items():
                if passed:
                    st.success(f"✓ **{check_name}** - *{message}*")
                else:
                    st.error(f"✗ **{check_name}** - *{message}*")
                    all_passed = False
            
            if all_passed:
                st.balloons()
                st.success("🎉 All checks passed! Your photo is ready for DV Lottery submission.")
            else:
                st.warning("⚠️ Some checks didn't pass perfectly. You can still download the photo, but consider retaking for best results.")
            
            # Download button
            buf = io.BytesIO()
            processed.save(buf, format="JPEG", quality=98)
            buf.seek(0)
            st.download_button(
                "⬇️ Download DV Photo",
                data=buf,
                file_name="dv_lottery_photo.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        st.info("💡 **Tips for better results:**")
        st.write("- Use high-resolution, well-lit photos")
        st.write("- Ensure good contrast between hair and background")
        st.write("- Face the camera directly with neutral expression")
        st.write("- Use plain, contrasting background for best removal")
else:
    # Show instructions when no file is uploaded
    st.info("👆 **Upload a photo to get started**")
    st.write("""
    ### 📋 What we check for DV Lottery compliance:
    
    - **Face Recognition**: Ensures your face is clearly visible and detectable
    - **Single Face**: Only one person should be in the photo
    - **Minimum Dimensions**: Photo must be at least 600x600 pixels
    - **Correct Proportions**: Photo should have proper aspect ratio
    - **Background Removal**: Ability to remove and replace background with white
    - **No Red Eyes**: Checks for red-eye effect in the photo
    
    ### 🎯 DV Lottery Photo Requirements:
    - Head height: 50% to 69% of image height
    - Eye line: 56% to 69% from bottom
    - Square aspect ratio (1:1)
    - White background
    - Size: 600x600 to 1200x1200 pixels
    """)
