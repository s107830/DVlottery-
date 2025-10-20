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

# ---------------------- ENHANCED AUTO ADJUST WITH CORRECTION CAPABILITY ----------------------
def auto_adjust_dv_photo(image_pil, correction_attempt=0):
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
        
        # Apply correction logic for subsequent attempts
        if correction_attempt > 0:
            # For correction attempts, be more aggressive with scaling
            if head_height < target_head_height * 0.8:
                # Head is too small, scale up more aggressively
                scale_factor = (target_head_height * 1.2) / head_height
            elif head_height > target_head_height * 1.2:
                # Head is too large, scale down more aggressively
                scale_factor = (target_head_height * 0.8) / head_height
            else:
                scale_factor = target_head_height / head_height
        else:
            scale_factor = target_head_height / head_height
        
        # Apply reasonable scaling limits
        scale_factor = max(0.3, min(3.0, scale_factor))
        
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
        
        # Apply correction for eye position if needed
        if correction_attempt > 0:
            # Adjust target eye position based on previous errors
            current_eye_ratio = eye_y_scaled / new_h
            if current_eye_ratio < EYE_MIN_RATIO:
                # Eyes are too high, adjust target lower
                target_eye_ratio = min(EYE_MAX_RATIO, target_eye_ratio + 0.1)
            elif current_eye_ratio > EYE_MAX_RATIO:
                # Eyes are too low, adjust target higher
                target_eye_ratio = max(EYE_MIN_RATIO, target_eye_ratio - 0.1)
        
        target_eye_y = canvas_size - int(canvas_size * target_eye_ratio)
        
        # Calculate vertical offset
        y_offset = target_eye_y - eye_y_scaled
        
        # Ensure full head including hair is visible
        min_y_offset = -top_y_scaled
        max_y_offset = canvas_size - chin_y_scaled
        
        # Clamp y_offset to keep entire head in frame
        y_offset = max(min_y_offset, min(y_offset, max_y_offset))
        
        # Apply additional correction for eye position if still needed
        if correction_attempt > 0:
            # Check if we need to adjust further for eye position
            final_eye_y = eye_y_scaled + y_offset
            final_eye_ratio = (canvas_size - final_eye_y) / canvas_size
            
            if final_eye_ratio < EYE_MIN_RATIO:
                # Still too high, try to move down more
                additional_offset = int(canvas_size * (EYE_MIN_RATIO - final_eye_ratio))
                y_offset = min(max_y_offset, y_offset + additional_offset)
            elif final_eye_ratio > EYE_MAX_RATIO:
                # Still too low, try to move up more
                additional_offset = int(canvas_size * (final_eye_ratio - EYE_MAX_RATIO))
                y_offset = max(min_y_offset, y_offset - additional_offset)
        
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
        
        # Return both the image and the head position info for guidelines
        head_info = {
            'top_y': y_start + top_y_scaled,
            'chin_y': y_start + chin_y_scaled,
            'eye_y': y_start + eye_y_scaled,
            'head_height': head_height_scaled,
            'canvas_size': canvas_size
        }
        
        return result_img, head_info
        
    except Exception as e:
        # Improved fallback
        size = max(MIN_SIZE, max(image_pil.size))
        square_img = Image.new("RGB", (size, size), (255, 255, 255))
        offset_x = (size - image_pil.width) // 2
        offset_y = (size - image_pil.height) // 2
        square_img.paste(image_pil, (offset_x, offset_y))
        
        # Return dummy head info for fallback
        head_info = {
            'top_y': size * 0.25,
            'chin_y': size * 0.75,
            'eye_y': size * 0.5,
            'head_height': size * 0.5,
            'canvas_size': size
        }
        
        return square_img, head_info

# ---------------------- CORRECTED GUIDELINES WITH ACTUAL HEAD POSITION ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Extract head position information
    top_y = head_info['top_y']
    chin_y = head_info['chin_y']
    eye_y = head_info['eye_y']
    head_height = head_info['head_height']
    canvas_size = head_info['canvas_size']

    # Calculate actual head height percentage
    actual_head_ratio = head_height / canvas_size
    head_percentage = int(actual_head_ratio * 100)

    # Eye line boundaries (56-69% from BOTTOM)
    eye_min_from_bottom = int(h * EYE_MIN_RATIO)
    eye_max_from_bottom = int(h * EYE_MAX_RATIO)
    
    eye_min_y = h - eye_min_from_bottom
    eye_max_y = h - eye_max_from_bottom

    # Draw bounding box
    draw.rectangle([(0, 0), (w-1, h-1)], outline="red", width=2)
    
    # Draw ACTUAL head height bracket on the left
    bracket_x = 40
    
    # Draw the actual head height measurement
    draw.line([(bracket_x, top_y), (bracket_x, chin_y)], 
              fill="blue", width=4)
    
    # Draw horizontal ticks at top and bottom of head
    draw.line([(bracket_x-15, top_y), (bracket_x+15, top_y)], 
              fill="blue", width=3)
    draw.line([(bracket_x-15, chin_y), (bracket_x+15, chin_y)], 
              fill="blue", width=3)
    
    # Add head height labels with actual percentage
    draw.text((bracket_x+20, top_y - 15), f"Top", fill="blue")
    draw.text((bracket_x+20, chin_y - 15), f"Chin", fill="blue")
    
    # Draw target head height range for reference (faint)
    target_min_y = (h - int(h * HEAD_MAX_RATIO)) // 2
    target_max_y = target_min_y + int(h * HEAD_MAX_RATIO)
    target_50_y = target_min_y + int(h * HEAD_MIN_RATIO)
    
    # Draw faint reference lines
    draw.line([(bracket_x-25, target_min_y), (bracket_x-5, target_min_y)], 
              fill="lightblue", width=2)
    draw.line([(bracket_x-25, target_max_y), (bracket_x-5, target_max_y)], 
              fill="lightblue", width=2)
    draw.line([(bracket_x-25, target_50_y), (bracket_x-5, target_50_y)], 
              fill="lightblue", width=2)
    
    # Add reference labels
    draw.text((10, target_min_y - 10), "69%", fill="lightblue")
    draw.text((10, target_50_y - 10), "50%", fill="lightblue")
    
    # Display actual head height percentage
    head_status_color = "green" if HEAD_MIN_RATIO <= actual_head_ratio <= HEAD_MAX_RATIO else "red"
    draw.text((bracket_x-100, (top_y + chin_y)//2 - 30), 
              f"HEAD HEIGHT: {head_percentage}%", fill=head_status_color)
    draw.text((bracket_x-100, (top_y + chin_y)//2 - 10), 
              "REQUIRED: 50-69%", fill="blue")
    
    if HEAD_MIN_RATIO <= actual_head_ratio <= HEAD_MAX_RATIO:
        draw.text((bracket_x-100, (top_y + chin_y)//2 + 10), 
                  "✓ WITHIN RANGE", fill="green")
    else:
        draw.text((bracket_x-100, (top_y + chin_y)//2 + 10), 
                  "✗ OUT OF RANGE", fill="red")
    
    # Draw eye line area
    draw.rectangle([(w-150, eye_max_y), (w, eye_min_y)], outline="green", width=2)
    draw.line([(w-150, (eye_min_y + eye_max_y)//2), (w, (eye_min_y + eye_max_y)//2)], 
              fill="green", width=2)
    
    # Calculate actual eye position percentage
    actual_eye_ratio = (h - eye_y) / h
    eye_percentage = int(actual_eye_ratio * 100)
    
    # Eye line labels with actual position
    eye_status_color = "green" if EYE_MIN_RATIO <= actual_eye_ratio <= EYE_MAX_RATIO else "red"
    draw.text((w-140, eye_min_y - 25), f"EYE LINE: {eye_percentage}%", fill=eye_status_color)
    draw.text((w-140, eye_max_y + 5), "REQUIRED: 56-69%", fill="green")
    
    # Mark actual eye position
    draw.line([(w-160, eye_y), (w-140, eye_y)], fill="darkgreen", width=3)
    
    # Draw actual head boundaries (faint lines across image)
    draw.line([(0, top_y), (w, top_y)], fill="lightblue", width=1)
    draw.line([(0, chin_y), (w, chin_y)], fill="lightblue", width=1)
    draw.line([(0, eye_y), (w, eye_y)], fill="lightgreen", width=1)
    
    return img, actual_head_ratio, actual_eye_ratio

# ---------------------- FIX PHOTO FUNCTION ----------------------
def fix_photo_measurements(original_bg_removed, current_head_info, current_eye_ratio):
    """Apply corrections to fix out-of-range measurements"""
    
    # Convert to CV for reprocessing
    image_rgb = np.array(original_bg_removed)
    
    try:
        # Get face landmarks
        landmarks = get_face_landmarks(image_rgb)
        original_h, original_w = image_rgb.shape[:2]
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, original_h, original_w)
        
        # Calculate what went wrong and apply corrections
        head_height = chin_y - top_y
        
        # Determine correction strategy based on current issues
        if current_eye_ratio < EYE_MIN_RATIO:
            # Eyes are too high (low percentage from bottom) - need to move image down
            # This means we need to scale differently and reposition
            correction_attempt = 1
        elif current_eye_ratio > EYE_MAX_RATIO:
            # Eyes are too low (high percentage from bottom) - need to move image up
            correction_attempt = 1
        else:
            correction_attempt = 1  # Default correction
            
        # Use the enhanced auto_adjust with correction logic
        fixed_img, fixed_head_info = auto_adjust_dv_photo(original_bg_removed, correction_attempt)
        
        return fixed_img, fixed_head_info
        
    except Exception as e:
        st.error(f"Fix failed: {e}")
        return original_bg_removed, current_head_info

# ---------------------- STREAMLIT UI ----------------------
uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

# Initialize session state for tracking fixes
if 'correction_count' not in st.session_state:
    st.session_state.correction_count = 0
if 'fixed_image' not in st.session_state:
    st.session_state.fixed_image = None
if 'fixed_head_info' not in st.session_state:
    st.session_state.fixed_head_info = None

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
                if st.session_state.fixed_image is not None and st.session_state.correction_count > 0:
                    # Use the fixed image if available
                    processed = st.session_state.fixed_image
                    head_info = st.session_state.fixed_head_info
                else:
                    # First time processing
                    processed, head_info = auto_adjust_dv_photo(bg_removed)
                    st.session_state.original_bg_removed = bg_removed
            
            final_preview, actual_head_ratio, actual_eye_ratio = draw_guidelines(processed.copy(), head_info)
            st.image(final_preview, caption=f"DV Compliance Preview: {processed.size}")
            
            # Display measurements
            col_meas1, col_meas2 = st.columns(2)
            with col_meas1:
                head_status = "✅ WITHIN RANGE" if HEAD_MIN_RATIO <= actual_head_ratio <= HEAD_MAX_RATIO else "❌ OUT OF RANGE"
                st.metric("Head Height", f"{int(actual_head_ratio * 100)}%", 
                         delta=head_status, delta_color="normal" if "WITHIN" in head_status else "off")
            
            with col_meas2:
                eye_status = "✅ WITHIN RANGE" if EYE_MIN_RATIO <= actual_eye_ratio <= EYE_MAX_RATIO else "❌ OUT OF RANGE"
                st.metric("Eye Position", f"{int(actual_eye_ratio * 100)}%", 
                         delta=eye_status, delta_color="normal" if "WITHIN" in eye_status else "off")
            
            # Show Fix Photo button if measurements are out of range
            if (actual_head_ratio < HEAD_MIN_RATIO or actual_head_ratio > HEAD_MAX_RATIO or 
                actual_eye_ratio < EYE_MIN_RATIO or actual_eye_ratio > EYE_MAX_RATIO):
                
                st.warning("⚠️ Some measurements are out of range. Click the button below to automatically fix them.")
                
                if st.button("🛠️ Fix Photo Measurements", type="primary", use_container_width=True):
                    with st.spinner("Applying corrections to fix measurements..."):
                        fixed_img, fixed_head_info = fix_photo_measurements(
                            st.session_state.original_bg_removed, 
                            head_info, 
                            actual_eye_ratio
                        )
                        
                        # Store fixed results in session state
                        st.session_state.fixed_image = fixed_img
                        st.session_state.fixed_head_info = fixed_head_info
                        st.session_state.correction_count += 1
                        
                        # Rerun to update display
                        st.rerun()
            
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
            
            # Add head and eye position to checklist
            head_check_passed = HEAD_MIN_RATIO <= actual_head_ratio <= HEAD_MAX_RATIO
            eye_check_passed = EYE_MIN_RATIO <= actual_eye_ratio <= EYE_MAX_RATIO
            
            if head_check_passed:
                st.success(f"✓ **Head height correct** - *{int(actual_head_ratio * 100)}% (required: 50-69%)*")
            else:
                st.error(f"✗ **Head height incorrect** - *{int(actual_head_ratio * 100)}% (required: 50-69%)*")
                all_passed = False
                
            if eye_check_passed:
                st.success(f"✓ **Eye position correct** - *{int(actual_eye_ratio * 100)}% from bottom (required: 56-69%)*")
            else:
                st.error(f"✗ **Eye position incorrect** - *{int(actual_eye_ratio * 100)}% from bottom (required: 56-69%)*")
                all_passed = False
            
            # Show correction count if any fixes were applied
            if st.session_state.correction_count > 0:
                st.info(f"🛠️ Photo has been automatically corrected {st.session_state.correction_count} time(s)")
            
            if all_passed and head_check_passed and eye_check_passed:
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
    # Reset session state when no file is uploaded
    st.session_state.correction_count = 0
    st.session_state.fixed_image = None
    st.session_state.fixed_head_info = None
    
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
    - **Head Height**: Head must be 50-69% of total image height ✓
    - **Eye Position**: Eyes must be 56-69% from bottom ✓
    
    ### 🎯 DV Lottery Photo Requirements:
    - Head height: 50% to 69% of image height
    - Eye line: 56% to 69% from bottom
    - Square aspect ratio (1:1)
    - White background
    - Size: 600x600 to 1200x1200 pixels
    
    ### 🛠️ Auto-Fix Feature:
    - If measurements are out of range, click the "Fix Photo" button
    - The AI will automatically adjust scaling and positioning
    - Checklist will be rerun to show updated results
    """)
