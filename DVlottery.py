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
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                   refine_landmarks=True, min_detection_confidence=0.5) as fm:
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
        
        # Add buffer for hair/head top
        hair_buffer = int((chin_y - top_y) * 0.4)
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
                "canvas_size": MIN_SIZE
            }
        except:
            # If face detection fails, use default values
            head_info = {
                "top_y": MIN_SIZE // 4,
                "chin_y": MIN_SIZE * 3 // 4,
                "eye_y": MIN_SIZE // 2,
                "head_height": MIN_SIZE // 2,
                "canvas_size": MIN_SIZE
            }
        
        return result, head_info
    except Exception as e:
        st.error(f"Initial photo processing error: {str(e)}")
        return img_pil, {"top_y": 0, "chin_y": 0, "eye_y": 0, "head_height": 0, "canvas_size": MIN_SIZE}

def process_dv_photo_adjusted(img_pil):
    """Processing WITH auto-adjustment for head to chin ratio - FIXED VERSION"""
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

        # Calculate scale factor to make head height optimal (around 60% of canvas)
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

        # Calculate optimal eye position (middle of required range)
        target_eye_y = int(MIN_SIZE * ((1 - EYE_MIN_RATIO) + (1 - EYE_MAX_RATIO)) / 2)
        
        # Calculate y_offset to position eyes at target
        y_offset = target_eye_y - eye_y
        
        # CRITICAL FIX: Ensure the entire head is visible
        # Check if top of head would be cut off
        if top_y + y_offset < 0:
            # If head top would be cut off, adjust to show full head
            y_offset = -top_y + 5  # Small margin from top
        
        # Check if bottom would be cut off
        if chin_y + y_offset > MIN_SIZE - 5:
            # If chin would be cut off, adjust to show full chin
            y_offset = MIN_SIZE - chin_y - 5  # Small margin from bottom

        # Center horizontally
        x_offset = (MIN_SIZE - new_w) // 2

        # Calculate source and destination regions with bounds checking
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
            st.warning("Using fallback placement")
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
            "canvas_size": MIN_SIZE
        }
        return result, head_info
    except Exception as e:
        st.error(f"Photo adjustment error: {str(e)}")
        return img_pil, {"top_y": 0, "chin_y": 0, "eye_y": 0, "head_height": 0, "canvas_size": MIN_SIZE}

# ---------------------- DRAW LINES ----------------------
def draw_guidelines(img, head_info):
    try:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cx = w // 2
        top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
        head_height, canvas_size = head_info["head_height"], head_info["canvas_size"]

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
        eye_min_y = h - int(h * EYE_MAX_RATIO)  # 56% from top
        eye_max_y = h - int(h * EYE_MIN_RATIO)  # 69% from top
        
        # Eye range guidelines (dashed green lines)
        dash_length = 10
        # Top eye guideline (56%)
        for x in range(0, w, dash_length*2):
            if x + dash_length <= w:
                draw.line([(x, eye_min_y), (x+dash_length, eye_min_y)], fill="green", width=2)
        draw.text((10, eye_min_y-15), "56%", fill="green")
        
        # Bottom eye guideline (69%)
        for x in range(0, w, dash_length*2):
            if x + dash_length <= w:
                draw.line([(x, eye_max_y), (x+dash_length, eye_max_y)], fill="green", width=2)
        draw.text((10, eye_max_y-15), "69%", fill="green")
        
        # Actual eye position line (solid red/green line)
        draw.line([(0, eye_y), (w, eye_y)], fill=eye_color, width=3)
        
        # Eye ratio text
        draw.text((w-150, eye_y-15), f"Eyes: {int(eye_ratio*100)}%", fill=eye_color)

        return img, head_ratio, eye_ratio
    except Exception as e:
        st.error(f"Guideline drawing error: {str(e)}")
        return img, 0, 0

# ---------------------- STREAMLIT LOGIC ----------------------

# Sidebar for instructions
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
    - **Format**: JPEG recommended
    """)
    
    st.header("⚙️ Settings")
    enhance_quality = st.checkbox("Enhance Image Quality", value=True)

# Main content area
uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Use session state to store processing results
    if 'processed_data' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
        st.session_state.last_upload = uploaded_file.name
        orig = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("🔄 Processing photo... Removing background and resizing..."):
            try:
                bg_removed = remove_background(orig)
                # Use initial processing WITHOUT auto-adjustment
                processed, head_info = process_dv_photo_initial(bg_removed)
                processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
                
                # Calculate if fixes are needed
                head_compliant = HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
                eye_compliant = EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
                needs_fix = not (head_compliant and eye_compliant)
                
                # Store in session state
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
                    'bg_removed': bg_removed,  # Store for potential adjustment
                    'is_adjusted': False  # Track if adjustment has been applied
                }
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("Please try another photo with a clear front-facing face.")
                st.stop()
    
    # Get data from session state
    if 'processed_data' in st.session_state:
        data = st.session_state.processed_data
        orig = data['orig']
        processed = data['processed']
        processed_with_lines = data['processed_with_lines']
        head_info = data['head_info']
        head_ratio = data['head_ratio']
        eye_ratio = data['eye_ratio']
        needs_fix = data['needs_fix']
        head_compliant = data['head_compliant']
        eye_compliant = data['eye_compliant']
        is_adjusted = data['is_adjusted']
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Photo")
            st.image(orig, use_column_width=True)
            
            # Original photo info
            st.info(f"**Original Size:** {orig.size[0]}×{orig.size[1]} pixels")

        with col2:
            status_text = "✅ Adjusted Photo" if is_adjusted else "📸 Initial Processed Photo"
            st.subheader(status_text)
            st.image(processed_with_lines, use_column_width=True, caption="DV Lottery Photo with Guidelines")
            
            # Quick stats
            st.info(f"**Final Size:** {MIN_SIZE}×{MIN_SIZE} pixels | **Format:** JPEG")
            if is_adjusted:
                st.success("✅ Auto-adjustment applied")

        # Compliance Dashboard
        st.subheader("📊 Compliance Dashboard")
        
        # Create metrics columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            head_status = "✅ PASS" if head_compliant else "❌ FAIL"
            st.metric(
                label="Head Height", 
                value=f"{int(head_ratio*100)}%",
                delta=None
            )
            st.write(head_status)
            st.progress(min(max(head_ratio / HEAD_MAX_RATIO, 0), 1.0))
            
        with col2:
            eye_status = "✅ PASS" if eye_compliant else "❌ FAIL"
            st.metric(
                label="Eye Position", 
                value=f"{int(eye_ratio*100)}%",
                delta=None
            )
            st.write(eye_status)
            st.progress(min(max(eye_ratio / EYE_MAX_RATIO, 0), 1.0))
            
        with col3:
            overall_status = "✅ COMPLIANT" if not needs_fix else "❌ NEEDS FIXING"
            st.metric("Overall Status", overall_status)
            if not needs_fix:
                st.success("🎉 Perfect! Your photo meets all DV Lottery requirements!")
            else:
                st.error("⚠️ Photo needs adjustment to meet DV Lottery standards.")

        # Fix Section
        if needs_fix:
            st.subheader("🛠️ Photo Correction")
            
            fix_col1, fix_col2 = st.columns([2, 1])
            
            with fix_col1:
                issues = []
                if not head_compliant:
                    issues.append("Head height out of range")
                if not eye_compliant:
                    issues.append("Eye position out of range")
                
                st.warning(f"""
                **Issues Detected:**
                - {' | '.join(issues)}
                - Click the button below to apply automatic head-to-chin adjustment
                - Multiple attempts may be needed for optimal results
                """)
                
            with fix_col2:
                if st.button("🔧 Auto-Adjust Head to Chin", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Applying head-to-chin auto-adjustment..."):
                        try:
                            # Use the background-removed image and apply adjustment
                            bg_removed = data['bg_removed']
                            processed, head_info = process_dv_photo_adjusted(bg_removed)
                            processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
                            
                            # Recalculate compliance
                            head_compliant = HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
                            eye_compliant = EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
                            needs_fix = not (head_compliant and eye_compliant)
                            
                            # Update session state
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
                                'is_adjusted': True  # Mark as adjusted
                            }
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Adjustment failed: {str(e)}")
            
            st.info("💡 **Tip:** If auto-adjustment doesn't work after several attempts, try uploading a different photo with better lighting and clear facial features.")

        # Download Section
        st.subheader("📥 Download Corrected Photo")
        
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            # Save download - use the processed image WITHOUT guidelines for download
            buf = io.BytesIO()
            processed.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="⬇️ Download Corrected Photo (No Guidelines)",
                data=buf.getvalue(),
                file_name="dv_lottery_photo.jpg",
                mime="image/jpeg",
                use_container_width=True,
                help="Download the corrected photo without measurement guidelines"
            )
        
        with dl_col2:
            # Also provide option to download with guidelines
            buf_with_guides = io.BytesIO()
            processed_with_lines.save(buf_with_guides, format="JPEG", quality=95)
            st.download_button(
                label="⬇️ Download with Guidelines",
                data=buf_with_guides.getvalue(),  # FIXED: Changed from b_with_guides to buf_with_guides
                file_name="dv_lottery_photo_with_guides.jpg",
                mime="image/jpeg",
                use_container_width=True,
                help="Download the corrected photo with measurement guidelines for verification"
            )

        # Detailed Requirements Section
        with st.expander("📋 Detailed DV Lottery Photo Requirements"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📐 Technical Specifications:**
                - **Photo Size:** 600×600 pixels
                - **File Format:** JPEG recommended
                - **Color:** Color only (no black & white)
                - **Resolution:** High quality, no compression artifacts
                
                **👤 Pose & Composition:**
                - **Facing:** Directly facing camera
                - **Expression:** Neutral, both eyes open
                - **Background:** Plain white or off-white
                - **Lighting:** Even, no shadows
                - **Headwear:** None (religious exceptions)
                - **Glasses:** None if possible (no glare)
                """)
                
            with col2:
                st.markdown("""
                **📏 Measurement Requirements:**
                - **Head Height:** 50%
