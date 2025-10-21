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
st.set_page_config(page_title="DV Lottery Photo Guide", layout="wide")
st.title("📸 DV Lottery Photo Positioning Guide")

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.50, 0.69
EYE_MIN_RATIO, EYE_MAX_RATIO = 0.56, 0.69

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# ---------------------- POSITIONING GUIDES ----------------------
def create_mobile_positioning_guide():
    """Create visual positioning guide for mobile camera"""
    st.markdown("""
    <style>
    .positioning-guide {
        border: 4px solid #00ff00;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        background: rgba(0, 255, 0, 0.05);
        text-align: center;
    }
    .guide-box {
        display: inline-block;
        width: 300px;
        height: 400px;
        border: 3px solid #00ff00;
        border-radius: 10px;
        position: relative;
        margin: 20px;
        background: white;
    }
    .head-outline {
        position: absolute;
        top: 50px;
        left: 50px;
        width: 200px;
        height: 250px;
        border: 2px solid #00ff00;
        border-radius: 45%;
    }
    .eye-level {
        position: absolute;
        top: 170px;
        left: 0;
        width: 300px;
        height: 40px;
        background: rgba(0, 255, 0, 0.2);
        border-top: 2px solid #00ff00;
        border-bottom: 2px solid #00ff00;
    }
    .guide-text {
        color: #00ff00;
        font-weight: bold;
        margin: 10px 0;
    }
    .mobile-steps {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
        text-align: center;
    }
    .step {
        flex: 1;
        padding: 10px;
        margin: 0 5px;
        background: rgba(0, 255, 0, 0.1);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_positioning_template():
    """Create a visual template showing perfect positioning"""
    # Create template image
    img = Image.new('RGB', (400, 500), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw camera frame
    draw.rectangle([10, 10, 390, 490], outline='black', width=3)
    
    # Draw head outline (green oval)
    center_x, center_y = 200, 250
    head_width, head_height = 240, 320
    draw.ellipse([
        center_x - head_width//2,
        center_y - head_height//2,
        center_x + head_width//2, 
        center_y + head_height//2
    ], outline='green', width=4)
    
    # Draw eye level band
    eye_y = center_y - 50
    band_height = 60
    draw.rectangle([
        20, eye_y - band_height//2,
        380, eye_y + band_height//2
    ], fill=(0, 255, 0, 50), outline='green', width=2)
    
    # Add labels
    draw.text((170, center_y - head_height//2 - 25), "HEAD TOP", fill='blue', stroke_width=1)
    draw.text((185, center_y + head_height//2 - 10), "CHIN", fill='purple', stroke_width=1)
    draw.text((165, eye_y - 10), "EYE LEVEL", fill='green', stroke_width=1)
    
    # Add measurement markers
    draw.text((10, eye_y - band_height//2 - 20), "56%", fill='green')
    draw.text((10, eye_y + band_height//2 - 10), "69%", fill='green')
    
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

def comprehensive_compliance_check(cv_img, landmarks, head_info):
    """Run all compliance checks and return issues"""
    issues = []
    
    h, w = cv_img.shape[:2]
    
    if not check_facing_direction(landmarks, w, h):
        issues.append("❌ Face not directly facing camera")
    
    if not check_eyes_open(landmarks, h, w):
        issues.append("❌ Eyes not fully open or clearly visible")
    
    head_ratio = head_info["head_height"] / head_info["canvas_size"]
    eye_ratio = (head_info["canvas_size"] - head_info["eye_y"]) / head_info["canvas_size"]
    
    if not (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO):
        issues.append(f"❌ Head height {int(head_ratio*100)}% not in range {int(HEAD_MIN_RATIO*100)}-{int(HEAD_MAX_RATIO*100)}%")
    
    if not (EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO):
        issues.append(f"❌ Eye position {int(eye_ratio*100)}% not in range {int(EYE_MIN_RATIO*100)}-{int(EYE_MAX_RATIO*100)}%")
    
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
        
        scale_factor = MIN_SIZE / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)
        y_offset = (MIN_SIZE - new_h) // 2
        x_offset = (MIN_SIZE - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        result = Image.fromarray(canvas)
        
        try:
            landmarks = get_face_landmarks(cv_img)
            top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
            head_height = chin_y - top_y
            
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
            
            compliance_issues = comprehensive_compliance_check(cv_img, landmarks, head_info)
            
        except Exception as e:
            head_info = {
                "top_y": MIN_SIZE // 4,
                "chin_y": MIN_SIZE * 3 // 4,
                "eye_y": MIN_SIZE // 2,
                "head_height": MIN_SIZE // 2,
                "canvas_size": MIN_SIZE
            }
            compliance_issues = ["❌ Cannot detect face properly"]
        
        return result, head_info, compliance_issues
    except Exception as e:
        st.error(f"Photo processing error: {str(e)}")
        return img_pil, {"top_y": 0, "chin_y": 0, "eye_y": 0, "head_height": 0, "canvas_size": MIN_SIZE}, ["❌ Processing error"]

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

        draw.line([(cx-50, top_y), (cx+50, top_y)], fill="blue", width=3)
        draw.line([(cx-50, chin_y), (cx+50, chin_y)], fill="purple", width=3)
        draw.line([(cx, top_y), (cx, chin_y)], fill=head_color, width=2)
        
        eye_min_y = h - int(h * EYE_MAX_RATIO)
        eye_max_y = h - int(h * EYE_MIN_RATIO)
        
        dash_length = 10
        for x in range(0, w, dash_length*2):
            if x + dash_length <= w:
                draw.line([(x, eye_min_y), (x+dash_length, eye_min_y)], fill="green", width=2)
                draw.line([(x, eye_max_y), (x+dash_length, eye_max_y)], fill="green", width=2)
        
        draw.line([(0, eye_y), (w, eye_y)], fill=eye_color, width=3)
        
        draw.text((cx + 10, (top_y + chin_y) // 2 - 20), f"Head: {int(head_ratio*100)}%", fill=head_color)
        draw.text((w-150, eye_y-15), f"Eyes: {int(eye_ratio*100)}%", fill=eye_color)

        return img, head_ratio, eye_ratio
    except Exception as e:
        return img, 0, 0

# ---------------------- STREAMLIT UI ----------------------

# Initialize positioning guide
create_mobile_positioning_guide()

# Sidebar
with st.sidebar:
    st.header("📋 DV Photo Requirements")
    st.markdown("""
    **📐 Perfect Positioning:**
    - Head fills 50-69% of frame
    - Eyes at 56-69% from top
    - Face centered, looking straight
    - Neutral expression
    - Plain white background
    
    **📱 Mobile Photo Tips:**
    - Hold phone at eye level
    - Use good natural light
    - Position using green guides
    - Take multiple shots
    """)
    
    # Show positioning template
    template_img = create_positioning_template()
    st.image(template_img, caption="🎯 Positioning Template", use_container_width=True)

# Main content - POSITIONING GUIDE
st.markdown("""
<div class="positioning-guide">
    <h2>📱 DV Lottery Photo Positioning Guide</h2>
    <p><strong>Use this visual guide to position yourself perfectly in your phone's camera</strong></p>
</div>
""", unsafe_allow_html=True)

# Visual positioning guide
st.markdown("""
<div style="text-align: center;">
    <h3>🎯 How to Position Your Face</h3>
</div>

<div class="mobile-steps">
    <div class="step">
        <h4>1️⃣</h4>
        <p>Open your phone's camera app</p>
    </div>
    <div class="step">
        <h4>2️⃣</h4>
        <p>Position face in green oval</p>
    </div>
    <div class="step">
        <h4>3️⃣</h4>
        <p>Align eyes with green band</p>
    </div>
    <div class="step">
        <h4>4️⃣</h4>
        <p>Take photo & upload below</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Interactive positioning demo
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="guide-box">
        <div class="head-outline"></div>
        <div class="eye-level"></div>
    </div>
    <div style="text-align: center;">
        <div class="guide-text">⬆️ POSITION YOUR FACE LIKE THIS ⬆️</div>
        <p>Head in oval • Eyes in green band • Neutral expression</p>
    </div>
    """, unsafe_allow_html=True)

# Photo upload for verification
st.markdown("---")
st.subheader("📤 Upload Your Photo for Verification")

uploaded_file = st.file_uploader(
    "Upload the photo you took using the positioning guide",
    type=["jpg", "jpeg", "png"],
    help="Take photo using your phone's camera app with the positioning guide, then upload here for verification"
)

if uploaded_file:
    st.success("✅ Photo received! Checking compliance...")
    
    with st.spinner("🔄 Analyzing your photo..."):
        try:
            orig = Image.open(uploaded_file).convert("RGB")
            
            # Remove background and process
            bg_removed = remove_background(orig)
            processed, head_info, compliance_issues = process_dv_photo_initial(bg_removed)
            processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(), head_info)
            
            head_compliant = HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO
            eye_compliant = EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Your Photo Analysis")
                st.image(processed_with_lines, use_container_width=True)
                
                if compliance_issues:
                    st.error("**Compliance Issues:**")
                    for issue in compliance_issues:
                        st.write(issue)
                else:
                    st.success("✅ **Perfect! All requirements met**")
            
            with col2:
                st.subheader("📐 Measurements")
                
                # Head height
                head_status = "✅ PASS" if head_compliant else "❌ FAIL"
                st.metric("Head Height", f"{int(head_ratio*100)}%", head_status)
                
                # Eye position  
                eye_status = "✅ PASS" if eye_compliant else "❌ FAIL"
                st.metric("Eye Position", f"{int(eye_ratio*100)}%", eye_status)
                
                # Overall status
                overall_ok = head_compliant and eye_compliant and not compliance_issues
                status_color = "🟢" if overall_ok else "🔴"
                st.metric("DV Lottery Ready", f"{status_color} {'YES' if overall_ok else 'NO'}")
                
                # Download section
                st.markdown("---")
                if overall_ok:
                    st.success("🎉 Your photo is perfect for DV Lottery!")
                    
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
                            "⬇️ With Guides",
                            data=buf_guide.getvalue(),
                            file_name="dv_photo_with_guides.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Some adjustments needed")
                    st.info("💡 Use the positioning guide above and take a new photo")
                    
        except Exception as e:
            st.error(f"❌ Error processing photo: {str(e)}")

else:
    # Instructions for using the guide
    st.markdown("""
    ## 📸 How to Use This Guide
    
    1. **Open your phone's camera app** (not this app's camera)
    2. **Position yourself** using the green guidelines above
    3. **Take the photo** with your phone's camera
    4. **Come back here** and upload the photo for verification
    5. **Get instant feedback** on DV Lottery compliance
    
    ### 💡 Pro Tips:
    - Use **natural lighting** - face a window
    - **Plain white background** behind you
    - **Neutral expression** - no smiling
    - **Both eyes open** and clearly visible
    - **No glasses** or headwear
    - **Hair** not covering face
    
    **Ready? Take your photo with your phone's camera, then upload it above!**
    """)

# Footer
st.markdown("---")
st.markdown("*DV Lottery Photo Positioning Guide - Use your phone's camera with our visual positioning aids*")
