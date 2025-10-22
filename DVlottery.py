import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, UnidentifiedImageError
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

# ---------------------- COMPLIANCE CHECK FUNCTIONS ----------------------
def check_facing_direction(landmarks, img_w, img_h):
    try:
        nose_tip = landmarks.landmark[1]
        left_face = landmarks.landmark[234]
        right_face = landmarks.landmark[454]
        nose_center_ratio = (nose_tip.x - left_face.x) / (right_face.x - left_face.x)
        return 0.4 <= nose_center_ratio <= 0.6
    except:
        return True

def check_eyes_open(landmarks, img_h, img_w):
    try:
        left_eye_top = landmarks.landmark[159]
        left_eye_bottom = landmarks.landmark[145]
        right_eye_top = landmarks.landmark[386]
        right_eye_bottom = landmarks.landmark[374]
        left_eye_openness = abs(left_eye_top.y - left_eye_bottom.y) * img_h
        right_eye_openness = abs(right_eye_top.y - right_eye_bottom.y) * img_h
        return left_eye_openness > 0.01*img_h and right_eye_openness > 0.01*img_h
    except:
        return True

def check_neutral_expression(landmarks):
    try:
        mouth_top = landmarks.landmark[13]
        mouth_bottom = landmarks.landmark[14]
        return abs(mouth_top.y - mouth_bottom.y) < 0.05
    except:
        return True

def check_hair_covering_eyes(landmarks, img_h, img_w):
    try:
        left_eye_inner = landmarks.landmark[133]
        left_eye_outer = landmarks.landmark[33]
        right_eye_inner = landmarks.landmark[362]
        right_eye_outer = landmarks.landmark[263]
        left_eye_width = abs(left_eye_outer.x - left_eye_inner.x) * img_w
        right_eye_width = abs(right_eye_outer.x - right_eye_inner.x) * img_w
        return left_eye_width >= 0.05*img_w and right_eye_width >= 0.05*img_w
    except:
        return True

def check_image_quality(cv_img):
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        issues = []
        if blur_value < 50:
            issues.append("Image may be blurry - use a clearer photo")
        if brightness < 40:
            issues.append("Photo may be too dark - improve lighting")
        elif brightness > 220:
            issues.append("Photo may be overexposed - reduce brightness")
        if contrast < 40:
            issues.append("Low contrast detected - ensure good lighting")
        return issues
    except:
        return []

def comprehensive_compliance_check(cv_img, landmarks, head_info):
    issues = []
    h, w = cv_img.shape[:2]
    if not check_facing_direction(landmarks, w, h):
        issues.append("❌ Face not directly facing camera")
    if not check_eyes_open(landmarks, h, w):
        issues.append("❌ Eyes not fully open or clearly visible")
    if not check_neutral_expression(landmarks):
        issues.append("❌ Non-neutral facial expression")
    if not check_hair_covering_eyes(landmarks, h, w):
        issues.append("❌ Hair may be covering eyes or face")
    issues.extend([f"❌ {i}" for i in check_image_quality(cv_img)])
    head_ratio = head_info["head_height"]/head_info["canvas_size"]
    eye_ratio = (head_info["canvas_size"]-head_info["eye_y"])/head_info["canvas_size"]
    if not (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO):
        issues.append(f"❌ Head height {int(head_ratio*100)}% not in required range {int(HEAD_MIN_RATIO*100)}-{int(HEAD_MAX_RATIO*100)}%")
    if not (EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO):
        issues.append(f"❌ Eye position {int(eye_ratio*100)}% not in required range {int(EYE_MIN_RATIO*100)}-{int(EYE_MAX_RATIO*100)}%")
    return issues

# ---------------------- HELPER FUNCTIONS ----------------------
def get_face_landmarks(cv_img):
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
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
        top_y = int(landmarks.landmark[10].y*img_h)
        chin_y = int(landmarks.landmark[152].y*img_h)
        left_eye_y = int(landmarks.landmark[33].y*img_h)
        right_eye_y = int(landmarks.landmark[263].y*img_h)
        eye_y = (left_eye_y+right_eye_y)//2
        hair_buffer = int((chin_y-top_y)*0.3)
        top_y = max(0, top_y-hair_buffer)
        return top_y, chin_y, eye_y
    except Exception as e:
        st.error(f"Landmark processing error: {str(e)}")
        raise

def remove_background(img_pil):
    try:
        b = io.BytesIO()
        img_pil.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255,255,255,255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except Exception as e:
        st.warning(f"Background removal failed: {str(e)}. Using original image.")
        return img_pil

def is_likely_baby_photo(cv_img, landmarks):
    try:
        h, w = cv_img.shape[:2]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        chin = landmarks.landmark[152]
        eye_distance = abs(left_eye.x - right_eye.x) * w
        face_height = (chin.y - landmarks.landmark[10].y)*h
        eye_to_face_ratio = eye_distance / face_height
        forehead_to_face_ratio = (landmarks.landmark[10].y - landmarks.landmark[151].y) / face_height
        return eye_to_face_ratio>0.35 and forehead_to_face_ratio>0.45
    except:
        return False

# ---------------------- PROCESSING FUNCTIONS ----------------------
def process_dv_photo_initial(img_pil):
    try:
        cv_img = np.array(img_pil)
        if len(cv_img.shape)==2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2]==4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
        h, w = cv_img.shape[:2]
        scale_factor = MIN_SIZE/max(h,w)
        new_w, new_h = int(w*scale_factor), int(h*scale_factor)
        resized = cv2.resize(cv_img,(new_w,new_h),interpolation=cv2.INTER_LANCZOS4)
        canvas = np.full((MIN_SIZE,MIN_SIZE,3),255,np.uint8)
        y_offset, x_offset = (MIN_SIZE-new_h)//2, (MIN_SIZE-new_w)//2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w]=resized
        result = Image.fromarray(canvas)
        try:
            landmarks = get_face_landmarks(cv_img)
            top_y, chin_y, eye_y = get_head_eye_positions(landmarks,h,w)
            head_height = chin_y - top_y
            final_top_y, final_chin_y, final_eye_y = int(top_y*scale_factor)+y_offset, int(chin_y*scale_factor)+y_offset, int(eye_y*scale_factor)+y_offset
            head_info = {"top_y": final_top_y,"chin_y": final_chin_y,"eye_y": final_eye_y,"head_height": head_height*scale_factor,"canvas_size": MIN_SIZE,"is_baby": is_likely_baby_photo(cv_img,landmarks)}
            compliance_issues = comprehensive_compliance_check(cv_img, landmarks, head_info)
        except:
            head_info = {"top_y": MIN_SIZE//4,"chin_y": MIN_SIZE*3//4,"eye_y": MIN_SIZE//2,"head_height": MIN_SIZE//2,"canvas_size": MIN_SIZE,"is_baby": False}
            compliance_issues = ["❌ Cannot detect face properly - ensure clear front-facing photo"]
        return result, head_info, compliance_issues
    except Exception as e:
        st.error(f"Initial photo processing error: {str(e)}")
        return img_pil, {"top_y":0,"chin_y":0,"eye_y":0,"head_height":0,"canvas_size":MIN_SIZE,"is_baby":False}, ["❌ Processing error - try another photo"]

def draw_guidelines(img, head_info):
    try:
        draw = ImageDraw.Draw(img)
        w,h = img.size
        cx = w//2
        top_y,chin_y,eye_y = head_info["top_y"],head_info["chin_y"],head_info["eye_y"]
        head_height,canvas_size = head_info["head_height"],head_info["canvas_size"]
        is_baby = head_info.get("is_baby",False)
        head_ratio = head_height/canvas_size
        eye_ratio = (canvas_size-eye_y)/canvas_size
        head_color = "green" if HEAD_MIN_RATIO<=head_ratio<=HEAD_MAX_RATIO else "red"
        eye_color = "green" if EYE_MIN_RATIO<=eye_ratio<=EYE_MAX_RATIO else "red"
        draw.line([(cx-50,top_y),(cx+50,top_y)],fill="blue",width=3)
        draw.text((cx+60,top_y-15),"Head Top",fill="blue")
        draw.line([(cx-50,chin_y),(cx+50,chin_y)],fill="purple",width=3)
        draw.text((cx+60,chin_y-15),"Chin",fill="purple")
        draw.line([(cx,top_y),(cx,chin_y)],fill=head_color,width=2)
        head_text_y=(top_y+chin_y)//2
        draw.text((cx+10,head_text_y-20),f"Head: {int(head_ratio*100)}%",fill=head_color)
        draw.text((cx+10,head_text_y),f"Req: {int(HEAD_MIN_RATIO*100)}-{int(HEAD_MAX_RATIO*100)}%",fill="blue")
        eye_min_y = h-int(h*EYE_MAX_RATIO)
        eye_max_y = h-int(h*EYE_MIN_RATIO)
        dash_length=10
        for x in range(0,w,dash_length*2):
            if x+dash_length<=w:
                draw.line([(x,eye_min_y),(x+dash_length,eye_min_y)],fill="green",width=2)
        draw.text((10,eye_min_y-15),"56%",fill="green")
        for x in range(0,w,dash_length*2):
            if x+dash_length<=w:
                draw.line([(x,eye_max_y),(x+dash_length,eye_max_y)],fill="green",width=2)
        draw.text((10,eye_max_y-15),"69%",fill="green")
        draw.line([(0,eye_y),(w,eye_y)],fill=eye_color,width=3)
        draw.text((w-150,eye_y-15),f"Eyes: {int(eye_ratio*100)}%",fill=eye_color)
        if is_baby:
            draw.text((10,10),"👶 Baby Photo Detected",fill="orange")
        return img, head_ratio, eye_ratio
    except Exception as e:
        st.error(f"Guideline drawing error: {str(e)}")
        return img, 0, 0

# ---------------------- STREAMLIT UI ----------------------
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** a clear front-facing photo
    2. **Check** the compliance results  
    3. **Fix** if measurements are out of range
    4. **Download** your corrected photo
    """)
    st.header("⚙️ Settings")
    enhance_quality = st.checkbox("Enhance Image Quality", value=True)

uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg","jpeg","png"])

def safe_show_image(img, caption=None):
    if isinstance(img, Image.Image):
        st.image(img, use_container_width=True, caption=caption)
    else:
        st.error("❌ Image could not be displayed (invalid format)")

if uploaded_file:
    if 'processed_data' not in st.session_state or st.session_state.get('last_upload')!=uploaded_file.name:
        st.session_state.last_upload = uploaded_file.name
        try:
            orig = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"❌ Failed to open image: {str(e)}")
            st.stop()
        with st.spinner("🔄 Processing photo and checking compliance..."):
            try:
                bg_removed = remove_background(orig)
                processed, head_info, compliance_issues = process_dv_photo_initial(bg_removed)
                processed_with_lines, head_ratio, eye_ratio = draw_guidelines(processed.copy(),head_info)
                head_compliant = HEAD_MIN_RATIO<=head_ratio<=HEAD_MAX_RATIO
                eye_compliant = EYE_MIN_RATIO<=eye_ratio<=EYE_MAX_RATIO
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
                    'is_adjusted': False,
                    'compliance_issues': compliance_issues
                }
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.stop()

    data = st.session_state.processed_data
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Photo")
        safe_show_image(data['orig'])
        if isinstance(data['orig'], Image.Image):
            st.info(f"**Original Size:** {data['orig'].size[0]}×{data['orig'].size[1]} pixels")
    with col2:
        status_text="✅ Adjusted Photo" if data['is_adjusted'] else "📸 Initial Processed Photo"
        st.subheader(status_text)
        safe_show_image(data['processed_with_lines'])
        st.info(f"**Final Size:** {MIN_SIZE}×{MIN_SIZE} pixels")
        if data['is_adjusted']:
            st.success("✅ Auto-adjustment applied")

    st.subheader("🔍 Compliance Check Results")
    if data
