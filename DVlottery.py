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
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.3,  # Lower confidence for babies
            min_tracking_confidence=0.3
        ) as fm:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = fm.process(img_rgb)
            if not results.multi_face_landmarks:
                raise Exception("No face landmarks found")
            return results.multi_face_landmarks[0]
    except Exception as e:
        st.error(f"Face detection error: {str(e)}")
        raise

def get_head_eye_positions(landmarks, img_h, img_w):
    try:
        # Get head top (forehead) and chin positions
        top_y = int(landmarks.landmark[10].y * img_h)  # Forehead
        chin_y = int(landmarks.landmark[152].y * img_h)  # Chin
        
        # Get eye positions
        left_eye_y = int(landmarks.landmark[33].y * img_h)
        right_eye_y = int(landmarks.landmark[263].y * img_h)
        eye_y = (left_eye_y + right_eye_y) // 2
        
        # Add buffer for hair/head top - larger buffer for babies
        hair_buffer = int((chin_y - top_y) * 0.5)  # Increased for babies
        top_y = max(0, top_y - hair_buffer)
        
        return top_y, chin_y, eye_y
    except Exception as e:
        st.error(f"Landmark processing error: {str(e)}")
        raise

def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}. Using original image.")
        return img_pil

def is_likely_baby_photo(cv_img, landmarks):
    """Detect if photo is likely a baby based on facial proportions"""
    try:
        h, w = cv_img.shape[:2]
        
        # Get facial features
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        nose_tip = landmarks.landmark[1]
        chin = landmarks.landmark[152]
        
        # Calculate proportions typical of babies
        eye_distance = abs(left_eye.x - right_eye.x) * w
        face_height = (chin.y - landmarks.landmark[10].y) * h
        
        # Babies typically have larger eyes relative to face
        eye_to_face_ratio = eye_distance / face_height
        
        # Babies have proportionally larger foreheads
        forehead_to_face_ratio = (landmarks.landmark[10].y - landmarks.landmark[151].y) / face_height
        
        return eye_to_face_ratio > 0.3 or forehead_to_face_ratio > 0.4
    except:
        return False

# ---------------------- CORE PROCESSING ----------------------
def process_dv_photo_initial(img_pil):
    """Initial processing without auto-adjustment - just resize to 600x600"""
    try:
        cv_img = np.array(img_pil)
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

        h, w = cv_img.shape[:2]
        
        # Simple resize to 600x600 without face adjustment
        scale_factor = MIN_SIZE / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas and center the image
        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
        y_offset = (MIN_SIZE - new_h) // 2
        x_offset = (MIN_SIZE - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        result = Image.fromarray(canvas)
        
        # Try to get face landmarks for display purposes only
        try:
            landmarks = get_face_landmarks(cv_img)
            top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
            head_height = chin_y - top_y
            
            # Scale positions for the resized image
            final_top_y = int(top_y * scale_factor) + y_offset
            final_chin_y = int(chin_y * scale_factor) + y_offset
            final_eye_y = int(eye_y * scale_factor) + y_offset
            
            head_info = {
                "top_y": final_top_y,
                "chin_y": final_chin_y,
                "eye_y": final_eye_y,
                "head_height": head_height * scale_factor,
                "canvas_size": MIN_SIZE,
                "is_baby": is_likely_baby_photo(cv_img, landmarks)
            }
        except:
            # If face detection fails, use default values
            head_info = {
                "top_y": MIN_SIZE // 4,
                "chin_y": MIN_SIZE * 3 // 4,
                "eye_y": MIN_SIZE // 2,
                "head_height": MIN_SIZE // 2,
                "canvas_size": MIN_SIZE,
                "is_baby": False
            }
        
        return result, head_info
    except Exception as e:
        st.error(f"Initial photo processing error: {str(e)}")
        return img_pil, {"top_y": 0, "chin_y": 0, "eye_y": 0, "head_height": 0, "canvas_size": MIN_SIZE, "is_baby": False}

def process_dv_photo_adjusted(img_pil):
    """Processing WITH auto-adjustment for head to chin ratio - BABY FRIENDLY"""
    try:
        cv_img = np.array(img_pil)
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

        h, w = cv_img.shape[:2]
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
        head_height = chin_y - top_y
        
        # Detect if it's a baby photo
        is_baby = is_likely_baby_photo(cv_img, landmarks)

        # Different scaling for babies vs adults
        if is_baby:
            # For babies, use more conservative scaling to avoid cutting head
            target_head_height = MIN_SIZE * 0.55  # Slightly smaller target
            scale_factor = target_head_height / head_height
            scale_factor = np.clip(scale_factor, 0.4, 2.5)  # Tighter bounds for babies
        else:
            # Normal scaling for adults
            target_head_height = MIN_SIZE * 0.6
            scale_factor = target_head_height / head_height
            scale_factor = np.clip(scale_factor, 0.3, 3.0)
        
        # Apply scaling
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Create white canvas
        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)

        # Recalculate positions after scaling
        landmarks_resized = get_face_landmarks(resized)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks_resized, new_h, new_w)
        head_height = chin_y - top_y

        # Calculate optimal eye position
        target_eye_min = MIN_SIZE - int(MIN_SIZE * EYE_MAX_RATIO)
        target_eye_max = MIN_SIZE - int(MIN_SIZE * EYE_MIN_RATIO)
        target_eye_y = (target_eye_min + target_eye_max) // 2

        # Calculate y_offset to position eyes at target
        y_offset = target_eye_y - eye_y
        
        # EXTRA PROTECTION FOR BABIES: Ensure the entire head is visible
        if is_baby:
            # For babies, be extra careful about head top
            head_top_margin = 20  # Larger margin for babies
            head_bottom_margin = 15
        else:
            head_top_margin = 10
            head_bottom_margin = 10
        
        # Check if top of head would be cut off
        if top_y + y_offset < head_top_margin:
            y_offset = -top_y + head_top_margin
        
        # Check if bottom would be cut off
        if chin_y + y_offset > MIN_SIZE - head_bottom_margin:
            y_offset = MIN_SIZE - chin_y - head_bottom_margin

        # Center horizontally
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

        # Place the image on canvas
        if (y_start_dst < y_end_dst and x_start_dst < x_end_dst and 
            y_start_src < y_end_src and x_start_src < x_end_src):
            
            canvas[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
                resized[y_start_src:y_end_src, x_start_src:x_end_src]
        else:
            # Fallback: center the image
            y_offset = max(0, (MIN_SIZE - new_h) // 2)
            x_offset = max(0, (MIN_SIZE - new_w) // 2)
            if y_offset + new_h <= MIN_SIZE and x_offset + new_w <= MIN_SIZE:
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

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
            "canvas_size": MIN_SIZE,
            "is_baby": is_baby
        }
        return result, head_info
    except Exception as e:
        st.error(f"Photo adjustment error: {str(e)}")
        return img_pil, {"top_y": 0, "chin_y": 0, "eye_y": 0, "head_height": 0, "canvas_size": MIN_SIZE, "is_baby": False}

# ---------------------- DRAW LINES ----------------------
def draw_guidelines(img, head_info):
    try:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cx = w // 2
        top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
        head_height, canvas_size = head_info["head_height"], head_info["canvas_size"]
        is_baby = head_info.get("is_baby", False)

        head_ratio = head_height / canvas_size
        eye_ratio = (canvas_size - eye_y) / canvas_size

        # Define colors based on compliance
        head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
        eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"

        # Draw head top line (short horizontal line)
        draw.line([(cx-50, top_y), (cx+50, top_y)], fill="blue", width=3)
        draw.text((cx+60, top_y-15), "Head Top", fill="blue")
        
        # Draw chin line (short horizontal line)  
        draw.line([(cx-50, chin_y), (cx+50, chin_y)], fill="purple", width=3)
        draw.text((cx+60, chin_y-15), "Chin", fill="purple")
        
        # Draw vertical head height line (from top to chin)
        draw.line([(cx, top_y), (cx, chin_y)], fill=head_color, width=2)
        
        # Head ratio text
        head_text_y = (top_y + chin_y) // 2
        draw.text((cx + 10, head_text_y - 20), f"Head: {int(head_ratio*100)}%", fill=head_color)
        draw.text((cx + 10, head_text_y), f"Req: {int(HEAD_MIN_RATIO*100)}-{int(HEAD_MAX_RATIO*100)}%", fill="blue")

        # Draw eye position guidelines
        eye_min_y = h - int(h * EYE_MAX_RATIO)
        eye_max_y = h - int(h * EYE_MIN_RATIO)
        
        # Eye range guidelines (dashed green lines)
        dash_length = 10
        for x in range(0, w, dash_length*2):
            if x + dash_length <= w:
                draw.line([(x, eye_min_y), (x+dash_length, eye_min_y)], fill="green", width=2)
        draw.text((10, eye_min_y-15), "56%", fill="green")
        
        for x in range(0, w, dash_length*2):
            if x + dash_length <= w:
                draw.line([(x, eye_max_y), (x+dash_length, eye_max_y)], fill="green", width=2)
        draw.text((10, eye_max_y-15), "69%", fill="green")
        
        # Actual eye position line
        draw.line([(0, eye_y), (w, eye_y)], fill=eye_color, width=3)
        
        # Eye ratio text
        draw.text((w-150, eye_y-15), f"Eyes: {int(eye_ratio*100)}%", fill=eye_color)

        # Show baby detection info
        if is_baby:
            draw.text((10, 10), "👶 Baby Photo Detected", fill="orange")

        return img, head_ratio, eye_ratio
    except Exception as e:
        st.error(f"Guideline drawing error: {str(e)}")
        return img, 0, 0

# ---------------------- STREAMLIT UI ----------------------

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** a clear front-facing photo
    2. **Check** the compliance results  
    3. **Fix** if measurements are out of range
    4. **Download** your corrected photo
    
    ### ✅ Requirements:
    - **Head Height**: 50% - 69% of photo
    - **Eye Position**: 56% - 69% from top
    - **Photo Size**: 600×600 pixels
    - **Background**: Plain white
    
    ### 👶 Baby Photos:
    - Works best with clear front-facing photos
    - Auto-detects baby facial features
    - Uses special adjustments for baby proportions
    """)
    
    st.header("⚙️ Settings")
    enhance_quality = st.checkbox("Enhance Image Quality", value=True)

# Main content
uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Initialize session state
    if 'processed_data' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
        st.session_state.last_upload = uploaded_file.name
        orig = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("🔄 Processing photo..."):
            try:
                bg_removed = remove_background(orig)
                processed, head_info = process_dv_photo_initial(bg_removed)
                processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
                
                head_compliant = HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
                eye_compliant = EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
                needs_fix = not (head_compliant and eye_compliant)
                
                st.session_state.processed_data = {
                    'orig': orig,
                    'processed': processed,
                    'processed_with_lines': processed_with_lines,
                    'head_info': head_info,
                    'head_ratio': head_ratio,
                    'eye_ratio': eye_ratio,
                    'needs_fix': needs_fix,
                    'head_compliant': head_compliant,
                    'eye_compliant': eye_compliant,
                    'bg_removed': bg_removed,
                    'is_adjusted': False
                }
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("💡 Tip: Try a different photo with clear facial features")
                st.stop()

    # Get data from session state
    data = st.session_state.processed_data
    
    # Show baby detection info
    if data['head_info'].get('is_baby', False):
        st.info("👶 **Baby photo detected** - Using special adjustments for infant facial proportions")
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Photo")
        st.image(data['orig'], use_column_width=True)
        st.info(f"**Original Size:** {data['orig'].size[0]}×{data['orig'].size[1]} pixels")

    with col2:
        status_text = "✅ Adjusted Photo" if data['is_adjusted'] else "📸 Initial Processed Photo"
        st.subheader(status_text)
        st.image(data['processed_with_lines'], use_column_width=True)
        st.info(f"**Final Size:** {MIN_SIZE}×{MIN_SIZE} pixels")
        if data['is_adjusted']:
            st.success("✅ Auto-adjustment applied")

    # Compliance Dashboard
    st.subheader("📊 Compliance Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        head_status = "✅ PASS" if data['head_compliant'] else "❌ FAIL"
        st.metric("Head Height", f"{int(data['head_ratio']*100)}%")
        st.write(head_status)
        st.progress(min(max(data['head_ratio'] / HEAD_MAX_RATIO, 0), 1.0))
        
    with col2:
        eye_status = "✅ PASS" if data['eye_compliant'] else "❌ FAIL"
        st.metric("Eye Position", f"{int(data['eye_ratio']*100)}%")
        st.write(eye_status)
        st.progress(min(max(data['eye_ratio'] / EYE_MAX_RATIO, 0), 1.0))
        
    with col3:
        overall_status = "✅ COMPLIANT" if not data['needs_fix'] else "❌ NEEDS FIXING"
        st.metric("Overall Status", overall_status)
        if not data['needs_fix']:
            st.success("🎉 Perfect! Your photo meets all requirements!")
        else:
            st.error("⚠️ Photo needs adjustment.")

    # Fix Section
    if data['needs_fix']:
        st.subheader("🛠️ Photo Correction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            issues = []
            if not data['head_compliant']:
                issues.append("Head height out of range")
            if not data['eye_compliant']:
                issues.append("Eye position out of range")
            
            st.warning(f"**Issues Detected:** - {' | '.join(issues)}")
            
        with col2:
            if st.button("🔧 Auto-Adjust Head to Chin", use_container_width=True, type="primary"):
                with st.spinner("🔄 Applying auto-adjustment..."):
                    try:
                        bg_removed = data['bg_removed']
                        processed, head_info = process_dv_photo_adjusted(bg_removed)
                        processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
                        
                        head_compliant = HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
                        eye_compliant = EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
                        needs_fix = not (head_compliant and eye_compliant)
                        
                        st.session_state.processed_data = {
                            'orig': data['orig'],
                            'processed': processed,
                            'processed_with_lines': processed_with_lines,
                            'head_info': head_info,
                            'head_ratio': head_ratio,
                            'eye_ratio': eye_ratio,
                            'needs_fix': needs_fix,
                            'head_compliant': head_compliant,
                            'eye_compliant': eye_compliant,
                            'bg_removed': data['bg_removed'],
                            'is_adjusted': True
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Adjustment failed: {str(e)}")
        
        if data['head_info'].get('is_baby', False):
            st.info("👶 **Baby photo tip:** Make sure the baby's face is clearly visible and looking directly at the camera")

    # Download Section
    st.subheader("📥 Download Corrected Photo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        buf = io.BytesIO()
        data['processed'].save(buf, format="JPEG", quality=95)
        st.download_button(
            label="⬇️ Download (No Guidelines)",
            data=buf.getvalue(),
            file_name="dv_lottery_photo.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    
    with col2:
        buf_with_guides = io.BytesIO()
        data['processed_with_lines'].save(buf_with_guides, format="JPEG", quality=95)
        st.download_button(
            label="⬇️ Download with Guidelines",
            data=b_with_guides.getvalue(),
            file_name="dv_lottery_photo_with_guides.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

else:
    # Welcome screen
    st.markdown("""
    ## 🎯 Welcome to DV Lottery Photo Editor
    
    This tool helps you create perfectly compliant photos for the Diversity Visa Lottery application.
    
    ### 🚀 How it works:
    1. **Upload** your photo
    2. **Automatic** background removal and resizing
    3. **Check** compliance results
    4. **Press Fix Button** for head-to-chin auto-adjustment
    5. **Download** your ready-to-use DV photo
    
    ### 👶 Baby Photos Supported!
    - Special detection for infant facial features
    - Adjusted proportions for baby photos
    - Extra head top protection
    
    **👆 Upload your photo above to get started!**
    """)
    
    # Clear session state
    if 'processed_data' in st.session_state:
        del st.session_state.processed_data
    if 'last_upload' in st.session_state:
        del st.session_state.last_upload

# Footer
st.markdown("---")
st.markdown("*DV Lottery Photo Editor | Now with better baby photo support*")
