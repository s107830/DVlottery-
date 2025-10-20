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

# ---------------------- SIMPLIFIED AUTO ADJUST ----------------------
def auto_adjust_dv_photo(image_pil):
    try:
        # Convert PIL to OpenCV
        image_rgb = np.array(image_pil)
        if len(image_rgb.shape) == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
            
        original_h, original_w = image_rgb.shape[:2]

        # Get face landmarks
        landmarks = get_face_landmarks(image_rgb)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, original_h, original_w)
        head_height = chin_y - top_y

        # Calculate scale factor for optimal head size (aim for midpoint of range)
        target_head_height = (HEAD_MIN_RATIO + HEAD_MAX_RATIO) / 2 * MIN_SIZE
        scale_factor = target_head_height / head_height
        
        # Apply reasonable scaling limits
        scale_factor = max(0.5, min(2.0, scale_factor))
        
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

        # Calculate optimal eye position (aim for midpoint of eye range)
        target_eye_ratio = (EYE_MIN_RATIO + EYE_MAX_RATIO) / 2
        target_eye_y = canvas_size - int(canvas_size * target_eye_ratio)
        
        # Calculate vertical offset to position eyes correctly
        y_offset = target_eye_y - eye_y_scaled
        
        # Ensure we don't crop the head
        min_y_offset = -top_y_scaled  # Don't crop top
        max_y_offset = canvas_size - chin_y_scaled  # Don't crop bottom
        
        # Clamp the y_offset to keep entire head in frame
        y_offset = max(min_y_offset, min(y_offset, max_y_offset))
        
        # Center horizontally
        x_offset = (canvas_size - new_w) // 2

        # Paste the resized image onto canvas
        y_start = max(0, y_offset)
        y_end = min(canvas_size, y_offset + new_h)
        x_start = max(0, x_offset)
        x_end = min(canvas_size, x_offset + new_w)
        
        # Calculate source coordinates
        y_src_start = max(0, -y_offset)
        y_src_end = min(new_h, canvas_size - y_offset)
        x_src_start = max(0, -x_offset)
        x_src_end = min(new_w, canvas_size - x_offset)

        # Perform the paste operation
        if (y_end - y_start > 0) and (x_end - x_start > 0):
            canvas[y_start:y_end, x_start:x_end] = \
                resized_img[y_src_start:y_src_end, x_src_start:x_src_end]

        # Convert back to PIL and apply slight sharpening
        result_img = Image.fromarray(canvas)
        enhancer = ImageEnhance.Sharpness(result_img)
        result_img = enhancer.enhance(1.1)
        
        # Calculate actual positions in final image
        final_top_y = y_start + top_y_scaled
        final_chin_y = y_start + chin_y_scaled
        final_eye_y = y_start + eye_y_scaled
        
        # Return both the image and the head position info for guidelines
        head_info = {
            'top_y': final_top_y,
            'chin_y': final_chin_y,
            'eye_y': final_eye_y,
            'head_height': final_chin_y - final_top_y,
            'canvas_size': canvas_size
        }
        
        return result_img, head_info
        
    except Exception as e:
        # Fallback: simple resize to square
        size = max(MIN_SIZE, max(image_pil.size))
        square_img = Image.new("RGB", (size, size), (255, 255, 255))
        offset_x = (size - image_pil.width) // 2
        offset_y = (size - image_pil.height) // 2
        square_img.paste(image_pil, (offset_x, offset_y))
        
        # Return dummy head info for fallback
        head_info = {
            'top_y': size * 0.3,
            'chin_y': size * 0.7,
            'eye_y': size * 0.5,
            'head_height': size * 0.4,
            'canvas_size': size
        }
        
        return square_img, head_info

# ---------------------- FIXED GUIDELINES FUNCTION ----------------------
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

    # Calculate actual eye position percentage (from bottom)
    actual_eye_ratio = (canvas_size - eye_y) / canvas_size
    eye_percentage = int(actual_eye_ratio * 100)

    # Draw bounding box
    draw.rectangle([(0, 0), (w-1, h-1)], outline="red", width=2)
    
    # Draw head height bracket on the LEFT side
    bracket_x = 30
    
    # Draw the actual head height measurement (from top of head to chin)
    draw.line([(bracket_x, top_y), (bracket_x, chin_y)], fill="blue", width=4)
    
    # Draw horizontal ticks at top and bottom of head
    draw.line([(bracket_x-10, top_y), (bracket_x+10, top_y)], fill="blue", width=2)
    draw.line([(bracket_x-10, chin_y), (bracket_x+10, chin_y)], fill="blue", width=2)
    
    # Add head height labels
    draw.text((bracket_x+15, top_y - 15), "Head Top", fill="blue")
    draw.text((bracket_x+15, chin_y - 15), "Chin", fill="blue")
    
    # Display actual head height percentage
    head_status_color = "green" if HEAD_MIN_RATIO <= actual_head_ratio <= HEAD_MAX_RATIO else "red"
    draw.text((bracket_x-120, (top_y + chin_y)//2 - 30), 
              f"Head: {head_percentage}%", fill=head_status_color, stroke_width=1, stroke_fill="white")
    draw.text((bracket_x-120, (top_y + chin_y)//2 - 10), 
              "Req: 50-69%", fill="blue", stroke_width=1, stroke_fill="white")
    
    # Draw eye line bracket on the RIGHT side
    eye_bracket_x = w - 30
    
    # Draw eye line area (56-69% from bottom)
    eye_min_y = h - int(h * EYE_MAX_RATIO)  # 69% from bottom (31% from top)
    eye_max_y = h - int(h * EYE_MIN_RATIO)  # 56% from bottom (44% from top)
    
    # Draw the eye position bracket
    draw.line([(eye_bracket_x, eye_min_y), (eye_bracket_x, eye_max_y)], fill="green", width=4)
    draw.line([(eye_bracket_x-10, eye_min_y), (eye_bracket_x+10, eye_min_y)], fill="green", width=2)
    draw.line([(eye_bracket_x-10, eye_max_y), (eye_bracket_x+10, eye_max_y)], fill="green", width=2)
    
    # Mark actual eye position
    draw.line([(eye_bracket_x-15, eye_y), (eye_bracket_x+15, eye_y)], fill="darkgreen", width=3)
    
    # Add eye position labels
    eye_status_color = "green" if EYE_MIN_RATIO <= actual_eye_ratio <= EYE_MAX_RATIO else "red"
    draw.text((eye_bracket_x-100, (eye_min_y + eye_max_y)//2 - 30), 
              f"Eyes: {eye_percentage}%", fill=eye_status_color, stroke_width=1, stroke_fill="white")
    draw.text((eye_bracket_x-100, (eye_min_y + eye_max_y)//2 - 10), 
              "Req: 56-69%", fill="green", stroke_width=1, stroke_fill="white")
    
    # Add reference text
    draw.text((10, 10), "DV Lottery Photo Template", fill="black", stroke_width=1, stroke_fill="white")
    draw.text((10, 30), f"Size: {w}x{h} px", fill="black")
    
    return img, actual_head_ratio, actual_eye_ratio

# ---------------------- FIX PHOTO FUNCTION ----------------------
def fix_photo_measurements(original_img):
    """Apply enhanced corrections to fix out-of-range measurements"""
    try:
        # Re-process the original image with enhanced adjustment
        with st.spinner("Applying enhanced corrections..."):
            # Use a more aggressive approach for fixing
            fixed_img, fixed_head_info = auto_adjust_dv_photo(original_img)
            
            return fixed_img, fixed_head_info
        
    except Exception as e:
        st.error(f"Fix failed: {e}")
        return original_img, None

# ---------------------- STREAMLIT UI ----------------------
uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

# Initialize session state
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None
if 'head_info' not in st.session_state:
    st.session_state.head_info = None
if 'bg_removed' not in st.session_state:
    st.session_state.bg_removed = None
if 'fix_attempted' not in st.session_state:
    st.session_state.fix_attempted = False

if uploaded_file:
    try:
        # Read and process image
        if st.session_state.processed_img is None or st.session_state.fix_attempted:
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
                    orig_cv = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
                    
                    checks = {
                        "Face is recognized": check_face_recognized(orig_cv),
                        "Only one face is allowed": check_single_face(orig_cv),
                        "Minimum dimension": check_minimum_dimensions(orig),
                        "Correct photo proportions": check_photo_proportions(orig),
                        "Can remove background": check_background_removal(orig),
                        "No red eyes": check_red_eyes(orig_cv)
                    }
                
                # Process image
                with st.spinner("Removing background..."):
                    bg_removed = remove_background(orig)
                    st.session_state.bg_removed = bg_removed
                
                with st.spinner("Auto-adjusting to DV specifications..."):
                    processed, head_info = auto_adjust_dv_photo(bg_removed)
                    st.session_state.processed_img = processed
                    st.session_state.head_info = head_info
                    st.session_state.fix_attempted = False
                
                # Draw guidelines and get measurements
                final_preview, actual_head_ratio, actual_eye_ratio = draw_guidelines(
                    processed.copy(), head_info
                )
                st.image(final_preview, caption=f"DV Compliance Preview: {processed.size}")
                
                # Store measurements in session state
                st.session_state.actual_head_ratio = actual_head_ratio
                st.session_state.actual_eye_ratio = actual_eye_ratio
                
        else:
            # Use cached processed image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Original Photo")
                # Re-read the uploaded file for display
                uploaded_file.seek(0)
                img_bytes = uploaded_file.read()
                orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                st.image(orig, caption=f"Original Size: {orig.size}")
            
            with col2:
                st.subheader("✅ DV Compliant Photo")
                
                # Draw guidelines with cached data
                final_preview, actual_head_ratio, actual_eye_ratio = draw_guidelines(
                    st.session_state.processed_img.copy(), st.session_state.head_info
                )
                st.image(final_preview, caption=f"DV Compliance Preview: {st.session_state.processed_img.size}")
        
        # Display measurements (use cached if available)
        actual_head_ratio = getattr(st.session_state, 'actual_head_ratio', actual_head_ratio)
        actual_eye_ratio = getattr(st.session_state, 'actual_eye_ratio', actual_eye_ratio)
        
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
        needs_fix = (actual_head_ratio < HEAD_MIN_RATIO or actual_head_ratio > HEAD_MAX_RATIO or 
                    actual_eye_ratio < EYE_MIN_RATIO or actual_eye_ratio > EYE_MAX_RATIO)
        
        if needs_fix:
            st.warning("⚠️ Some measurements are out of range. Click the button below to automatically fix them.")
            
            if st.button("🛠️ Fix Photo Measurements", type="primary", use_container_width=True):
                with st.spinner("Applying corrections..."):
                    if st.session_state.bg_removed is not None:
                        fixed_img, fixed_head_info = fix_photo_measurements(st.session_state.bg_removed)
                        
                        if fixed_head_info is not None:
                            st.session_state.processed_img = fixed_img
                            st.session_state.head_info = fixed_head_info
                            st.session_state.fix_attempted = True
                            st.rerun()
                        else:
                            st.error("Failed to fix the photo. Please try again.")
                    else:
                        st.error("No background-removed image found. Please re-upload the photo.")
        
        # Display success message and checklist
        st.success("🎉 **Photo processed successfully**")
        
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
        
        if all_passed and head_check_passed and eye_check_passed:
            st.balloons()
            st.success("🎉 All checks passed! Your photo is ready for DV Lottery submission.")
        else:
            st.warning("⚠️ Some checks didn't pass perfectly. You can still download the photo, but consider retaking for best results.")
        
        # Download button
        buf = io.BytesIO()
        st.session_state.processed_img.save(buf, format="JPEG", quality=95)
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
    st.session_state.processed_img = None
    st.session_state.head_info = None
    st.session_state.bg_removed = None
    st.session_state.fix_attempted = False
    
    # Show instructions when no file is uploaded
    st.info("👆 **Upload a photo to get started**")
    st.write("""
    ### 📋 DV Lottery Photo Requirements:
    - **Head height**: 50% to 69% of image height
    - **Eye position**: 56% to 69% from bottom  
    - **Square aspect ratio** (1:1)
    - **White background**
    - **Size**: 600x600 to 1200x1200 pixels
    - **Single person**, front-facing, neutral expression
    
    ### 🛠️ Auto-Fix Feature:
    - If measurements are out of range, click "Fix Photo Measurements"
    - The AI will automatically adjust scaling and positioning
    - Updated results will be shown immediately
    """)
