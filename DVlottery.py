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
st.title("📸 DV Lottery Photo Editor — Auto Correction & Compliance Check")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# ---------------------- MOBILE CAMERA OVERLAY ----------------------
def add_dv_guidelines_to_camera():
    """Add DV guidelines overlay to camera input for mobile users"""
    st.markdown("""
    <style>
    .camera-guide {
        position: relative;
        border: 3px solid #00ff00;
        border-radius: 10px;
        padding: 10px;
        background: rgba(0, 255, 0, 0.1);
        margin-bottom: 20px;
    }
    .guide-text {
        color: #00ff00;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="camera-guide">', unsafe_allow_html=True)
    st.markdown('<div class="guide-text">🎯 DV Photo Guidelines - Position Face in Center</div>', unsafe_allow_html=True)
    
    # Camera input with mobile optimization
    camera_photo = st.camera_input(
        "Take DV Lottery Photo", 
        key="dv_camera",
        help="Position your face in the center. Keep neutral expression, eyes open, looking straight at camera."
    )
    
    st.markdown("""
    <div style="text-align: center; color: #00ff00; font-size: 14px; margin-top: 10px;">
    ✅ Green Area = Perfect Position<br>
    👁️  Eyes should be in middle section<br>
    📏 Head should fill 50-69% of height
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return camera_photo

def create_dv_overlay_image():
    """Create a visual guide image showing DV photo requirements"""
    # Create a template image with guidelines
    img = Image.new('RGB', (400, 500), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw head oval (green)
    head_center = (200, 250)
    head_width, head_height = 200, 280
    draw.ellipse([
        head_center[0] - head_width//2, 
        head_center[1] - head_height//2,
        head_center[0] + head_width//2, 
        head_center[1] + head_height//2
    ], outline='green', width=3)
    
    # Draw eye level band (semi-transparent green)
    eye_level = head_center[1] - 30
    band_height = 40
    eye_band = Image.new('RGBA', (400, 500), (0, 255, 0, 50))
    img = Image.alpha_composite(img.convert('RGBA'), eye_band).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw eye level line
    draw.line([(50, eye_level), (350, eye_level)], fill='green', width=2)
    
    # Add labels
    draw.text((160, head_center[1] - head_height//2 - 20), "Head Top", fill='blue')
    draw.text((170, head_center[1] + head_height//2 - 10), "Chin", fill='purple')
    draw.text((160, eye_level - 10), "Eye Level", fill='green')
    
    # Add instructions
    draw.text((50, 400), "✅ Center face in oval", fill='black')
    draw.text((50, 420), "✅ Align eyes with green band", fill='black')
    draw.text((50, 440), "✅ Neutral expression", fill='black')
    draw.text((50, 460), "✅ Look straight ahead", fill='black')
    
    return img

# ---------------------- COMPLIANCE CHECKERS ----------------------
def check_facing_direction(landmarks, img_w, img_h):
    """Check if face is directly facing camera"""
    try:
        nose_tip = landmarks.landmark[1]
        left_face = landmarks.landmark[234]
        right_face = landmarks.landmark[454]
        
        face_width = abs(right_face.x - left_face.x) * img_w
        nose_center_ratio = (nose_tip.x - left_face.x) / (right_face.x - left_face.x)
        
        return 0.4 <= nose_center_ratio <= 0.6
    except:
        return True

def check_eyes_open(landmarks, img_h, img_w):
    """Check if both eyes are open and visible"""
    try:
        left_eye_top = landmarks.landmark[159]
        left_eye_bottom = landmarks.landmark[145]
        right_eye_top = landmarks.landmark[386]
        right_eye_bottom = landmarks.landmark[374]
        
        left_eye_openness = abs(left_eye_top.y - left_eye_bottom.y) * img_h
        right_eye_openness = abs(right_eye_top.y - right_eye_bottom.y) * img_h
        
        min_eye_openness = 0.01 * img_h
        return left_eye_openness > min_eye_openness and right_eye_openness > min_eye_openness
    except:
        return True

def check_neutral_expression(landmarks):
    """Check for neutral facial expression"""
    try:
        mouth_top = landmarks.landmark[13]
        mouth_bottom = landmarks.landmark[14]
        mouth_openness = abs(mouth_top.y - mouth_bottom.y)
        return mouth_openness < 0.05
    except:
        return True

def check_image_quality(cv_img):
    """Check for blurriness, lighting issues"""
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        
        issues = []
        if blur_value < 100:
            issues.append("Image may be blurry")
        if brightness < 50 or brightness > 200:
            issues.append("Lighting issues detected")
            
        return issues
    except:
        return []

def comprehensive_compliance_check(cv_img, landmarks, head_info):
    """Run all compliance checks and return issues"""
    issues = []
    
    h, w = cv_img.shape[:2]
    
    # Face direction check
    if not check_facing_direction(landmarks, w, h):
        issues.append("❌ Face not directly facing camera - look straight ahead")
    
    # Eyes check
    if not check_eyes_open(landmarks, h, w):
        issues.append("❌ Eyes not fully open or clearly visible")
    
    # Expression check
    if not check_neutral_expression(landmarks):
        issues.append("❌ Non-neutral facial expression detected - maintain neutral expression")
    
    # Image quality check
    quality_issues = check_image_quality(cv_img)
    issues.extend([f"❌ {issue}" for issue in quality_issues])
    
    # Head and eye position compliance
    head_ratio = head_info["head_height"] / head_info["canvas_size"]
    eye_ratio = (head_info["canvas_size"] - head_info["eye_y"]) / head_info["canvas_size"]
    
    if not (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO):
        issues.append(f"❌ Head height {int(head_ratio*100)}% not in required range {int(HEAD_MIN_RATIO*100)}-{int(HEAD_MAX_RATIO*100)}%")
    
    if not (EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO):
        issues.append(f"❌ Eye position {int(eye_ratio*100)}% not in required range {int(EYE_MIN_RATIO*100)}-{int(EYE_MAX_RATIO*100)}%")
    
    return issues

# ---------------------- CORE FUNCTIONS ----------------------
def get_face_landmarks(cv_img):
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.3
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
        top_y = int(landmarks.landmark[10].y * img_h)
        chin_y = int(landmarks.landmark[152].y * img_h)
        
        left_eye_y = int(landmarks.landmark[33].y * img_h)
        right_eye_y = int(landmarks.landmark[263].y * img_h)
        eye_y = (left_eye_y + right_eye_y) // 2
        
        hair_buffer = int((chin_y - top_y) * 0.3)
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

def process_dv_photo_initial(img_pil):
    try:
        cv_img = np.array(img_pil)
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)

        h, w = cv_img.shape[:2]
        
        # Resize to 600x600
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
        
        # Try to get face landmarks
        try:
            landmarks = get_face_landmarks(cv_img)
            top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
            head_height = chin_y - top_y
            
            # Scale positions for resized image
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
            
            # Run compliance checks
            compliance_issues = comprehensive_compliance_check(cv_img, landmarks, head_info)
            
        except Exception as e:
            # If face detection fails
            head_info = {
                "top_y": MIN_SIZE // 4,
                "chin_y": MIN_SIZE * 3 // 4,
                "eye_y": MIN_SIZE // 2,
                "head_height": MIN_SIZE // 2,
                "canvas_size": MIN_SIZE
            }
            compliance_issues = ["❌ Cannot detect face properly - ensure clear front-facing photo"]
        
        return result, head_info, compliance_issues
    except Exception as e:
        st.error(f"Photo processing error: {str(e)}")
        return img_pil, {"top_y": 0, "chin_y": 0, "eye_y": 0, "head_height": 0, "canvas_size": MIN_SIZE}, ["❌ Processing error - try another photo"]

def draw_guidelines(img, head_info):
    try:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cx = w // 2
        top_y, chin_y, eye_y = head_info["top_y"], head_info["chin_y"], head_info["eye_y"]
        head_height, canvas_size = head_info["head_height"], head_info["canvas_size"]

        head_ratio = head_height / canvas_size
        eye_ratio = (canvas_size - eye_y) / canvas_size

        head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
        eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"

        # Draw head top and chin lines
        draw.line([(cx-50, top_y), (cx+50, top_y)], fill="blue", width=3)
        draw.line([(cx-50, chin_y), (cx+50, chin_y)], fill="purple", width=3)
        draw.line([(cx, top_y), (cx, chin_y)], fill=head_color, width=2)
        
        # Draw eye level lines
        eye_min_y = h - int(h * EYE_MAX_RATIO)
        eye_max_y = h - int(h * EYE_MIN_RATIO)
        
        dash_length = 10
        for x in range(0, w, dash_length*2):
            if x + dash_length <= w:
                draw.line([(x, eye_min_y), (x+dash_length, eye_min_y)], fill="green", width=2)
                draw.line([(x, eye_max_y), (x+dash_length, eye_max_y)], fill="green", width=2)
        
        draw.line([(0, eye_y), (w, eye_y)], fill=eye_color, width=3)
        
        # Add text labels
        draw.text((cx + 10, (top_y + chin_y) // 2 - 20), f"Head: {int(head_ratio*100)}%", fill=head_color)
        draw.text((w-150, eye_y-15), f"Eyes: {int(eye_ratio*100)}%", fill=eye_color)

        return img, head_ratio, eye_ratio
    except Exception as e:
        st.error(f"Guideline drawing error: {str(e)}")
        return img, 0, 0

# ---------------------- STREAMLIT UI ----------------------

# Sidebar
with st.sidebar:
    st.header("📋 DV Photo Requirements")
    st.markdown("""
    **📐 Measurements:**
    - Head Height: 50% - 69%
    - Eye Position: 56% - 69% from top
    
    **✅ Photo Quality:**
    - Plain white background
    - Face directly facing camera
    - Neutral expression, eyes open
    - No glasses/headwear
    - Good lighting
    - 600×600 pixels
    
    **📱 Mobile Tips:**
    - Hold phone at eye level
    - Center face in view
    - Use good natural light
    - Take multiple photos
    """)

# Main content
st.markdown("""
## 📸 DV Lottery Photo Assistant

**Two ways to get your perfect DV photo:**
1. **📱 Take New Photo** - Use camera with guidelines (Recommended for mobile)
2. **📤 Upload Existing** - Process photos you already have
""")

# Option 1: Camera with overlay
st.subheader("📱 Take Photo with DV Guidelines")
st.info("**Mobile-friendly**: Green guidelines help you position perfectly BEFORE capturing")

# Show visual guide
col1, col2 = st.columns([1, 2])
with col1:
    guide_img = create_dv_overlay_image()
    st.image(guide_img, caption="📐 Positioning Guide", use_column_width=True)

with col2:
    # Camera with styled overlay
    camera_photo = add_dv_guidelines_to_camera()

# Process camera photo if taken
if camera_photo:
    st.success("✅ Photo captured! Processing...")
    uploaded_file = Image.open(camera_photo).convert("RGB")
    st.session_state.camera_photo_taken = True

# Option 2: Upload existing photo
st.subheader("📤 Or Upload Existing Photo")
uploaded_file_manual = st.file_uploader("Choose photo file", type=["jpg", "jpeg", "png"], key="file_uploader")

# Use either camera photo or uploaded file
final_upload = uploaded_file if 'camera_photo_taken' in st.session_state else uploaded_file_manual

if final_upload:
    # Process the photo
    if isinstance(final_upload, Image.Image):
        orig = final_upload
    else:
        orig = Image.open(final_upload).convert("RGB")
    
    with st.spinner("🔄 Processing photo..."):
        try:
            # Remove background and process
            bg_removed = remove_background(orig)
            processed, head_info, compliance_issues = process_dv_photo_initial(bg_removed)
            processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
            
            head_compliant = HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
            eye_compliant = EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Your Photo")
                st.image(processed_with_lines, use_column_width=True)
                
                # Compliance status
                if compliance_issues:
                    st.error("❌ **Compliance Issues Found**")
                    for issue in compliance_issues:
                        st.write(f"- {issue}")
                else:
                    st.success("✅ **All requirements met!**")
            
            with col2:
                st.subheader("📊 Measurements")
                
                # Head height
                head_status = "✅ PASS" if head_compliant else "❌ FAIL"
                st.metric("Head Height", f"{int(head_ratio*100)}%", head_status)
                st.progress(min(max((head_ratio - HEAD_MIN_RATIO) / (HEAD_MAX_RATIO - HEAD_MIN_RATIO), 0), 1.0))
                
                # Eye position  
                eye_status = "✅ PASS" if eye_compliant else "❌ FAIL"
                st.metric("Eye Position", f"{int(eye_ratio*100)}%", eye_status)
                st.progress(min(max((eye_ratio - EYE_MIN_RATIO) / (EYE_MAX_RATIO - EYE_MIN_RATIO), 0), 1.0))
                
                # Overall status
                overall_ok = head_compliant and eye_compliant and not compliance_issues
                st.metric("Overall Status", "✅ READY" if overall_ok else "❌ NEEDS WORK")
                
                # Download buttons
                if overall_ok:
                    st.success("🎉 Your photo is DV Lottery ready!")
                    
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        buf = io.BytesIO()
                        processed.save(buf, format="JPEG", quality=95)
                        st.download_button(
                            "⬇️ Download Photo",
                            data=buf.getvalue(),
                            file_name="dv_lottery_photo.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                    
                    with col_d2:
                        buf_guide = io.BytesIO()
                        processed_with_lines.save(buf_guide, format="JPEG", quality=95)
                        st.download_button(
                            "⬇️ With Guidelines",
                            data=buf_guide.getvalue(),
                            file_name="dv_photo_with_guides.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Adjust photo and try again")
                    if st.button("🔄 Take New Photo", use_container_width=True):
                        if 'camera_photo_taken' in st.session_state:
                            del st.session_state.camera_photo_taken
                        st.rerun()
                        
        except Exception as e:
            st.error(f"❌ Error processing photo: {str(e)}")
            st.info("💡 Try a different photo with clear facial features")

# Footer
st.markdown("---")
st.markdown("*DV Lottery Photo Assistant - Mobile-friendly with live camera guidelines*")
