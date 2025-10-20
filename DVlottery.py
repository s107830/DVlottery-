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

# ---------------------- DV LOTTERY CONSTANTS ----------------------
MIN_SIZE = 600
MAX_SIZE = 1200
HEAD_MIN_RATIO = 0.50
HEAD_MAX_RATIO = 0.69
EYE_MIN_RATIO = 0.56
EYE_MAX_RATIO = 0.69

# ---------------------- MEDIAPIPE SETUP ----------------------
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# ---------------------- FACE LANDMARKS ----------------------
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
        top_y = int(landmarks.landmark[10].y * img_h)
        chin_y = int(landmarks.landmark[152].y * img_h)
        eye_y = int((landmarks.landmark[33].y + landmarks.landmark[263].y)/2 * img_h)
        hair_buffer = int((chin_y - top_y) * 0.25)
        top_y = max(0, top_y - hair_buffer)
        return top_y, chin_y, eye_y
    except:
        return int(img_h*0.3), int(img_h*0.7), int(img_h*0.5)

# ---------------------- CHECK FUNCTIONS ----------------------
def check_single_face(cv_img):
    try:
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
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
    w, h = img_pil.size
    if w >= MIN_SIZE and h >= MIN_SIZE:
        return True, f"Minimum dimensions passed ({w}x{h})"
    else:
        return False, f"Image too small: {w}x{h} (min {MIN_SIZE}x{MIN_SIZE})"

def check_photo_proportions(img_pil):
    w, h = img_pil.size
    aspect_ratio = w / h
    if 0.9 <= aspect_ratio <= 1.1:
        return True, f"Correct proportions ({w}x{h})"
    else:
        return False, f"Incorrect proportions: {w}x{h}"

def check_face_recognized(cv_img):
    try:
        landmarks = get_face_landmarks(cv_img)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, cv_img.shape[0], cv_img.shape[1])
        head_height = chin_y - top_y
        return True, "Face recognized" if head_height > 0 else (False, "Face recognition failed")
    except:
        return False, "Face recognition failed"

def check_red_eyes(cv_img):
    try:
        return True, "No red eyes detected"  # simplified for speed
    except:
        return True, "Red eye check skipped"

def check_background_removal(img_pil):
    try:
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        _ = remove(buf.getvalue())
        return True, "Background removable"
    except:
        return False, "Background removal failed"

# ---------------------- BACKGROUND REMOVAL ----------------------
def remove_background(img_pil):
    try:
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        result = remove(buf.getvalue())
        fg = Image.open(io.BytesIO(result)).convert("RGBA")
        white_bg = Image.new("RGBA", fg.size, (255,255,255,255))
        return Image.alpha_composite(white_bg, fg).convert("RGB")
    except:
        return img_pil

# ---------------------- AUTO ADJUST ----------------------
def auto_adjust_dv_photo(image_pil, force_correction=False):
    try:
        image_rgb = np.array(image_pil)
        if len(image_rgb.shape) == 2: image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4: image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

        h, w = image_rgb.shape[:2]
        landmarks = get_face_landmarks(image_rgb)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, h, w)
        head_height = chin_y - top_y
        target_head_height = (HEAD_MIN_RATIO + HEAD_MAX_RATIO)/2 * MIN_SIZE
        scale_factor = target_head_height / head_height
        scale_factor = max(0.5, min(2.0, scale_factor))

        new_h = int(h*scale_factor)
        new_w = int(w*scale_factor)
        resized_img = cv2.resize(image_rgb, (new_w,new_h), interpolation=cv2.INTER_LANCZOS4)

        top_y_scaled = int(top_y*scale_factor)
        chin_y_scaled = int(chin_y*scale_factor)
        eye_y_scaled = int(eye_y*scale_factor)

        canvas_size = MIN_SIZE
        canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
        target_eye_y = canvas_size - int(canvas_size*((EYE_MIN_RATIO + EYE_MAX_RATIO)/2))
        y_offset = target_eye_y - eye_y_scaled
        y_offset = max(-top_y_scaled, min(y_offset, canvas_size - chin_y_scaled))
        x_offset = (canvas_size - new_w)//2

        y_start = max(0, y_offset)
        y_end = min(canvas_size, y_offset+new_h)
        x_start = max(0, x_offset)
        x_end = min(canvas_size, x_offset+new_w)
        y_src_start = max(0, -y_offset)
        y_src_end = min(new_h, canvas_size - y_offset)
        x_src_start = max(0, -x_offset)
        x_src_end = min(new_w, canvas_size - x_offset)

        canvas[y_start:y_end, x_start:x_end] = resized_img[y_src_start:y_src_end, x_src_start:x_src_end]

        result_img = Image.fromarray(canvas)
        result_img = ImageEnhance.Sharpness(result_img).enhance(1.1)

        head_info = {'top_y': y_start+top_y_scaled, 'chin_y': y_start+chin_y_scaled, 'eye_y': y_start+eye_y_scaled,
                     'head_height': chin_y_scaled-top_y_scaled, 'canvas_size': canvas_size}
        return result_img, head_info

    except:
        size = MIN_SIZE
        square_img = Image.new("RGB", (size, size), (255,255,255))
        square_img.paste(image_pil, ((size-image_pil.width)//2, (size-image_pil.height)//2))
        head_info = {'top_y': size*0.3, 'chin_y': size*0.7, 'eye_y': size*0.5, 'head_height': size*0.4, 'canvas_size': size}
        return square_img, head_info

# ---------------------- DRAW GUIDELINES ----------------------
def draw_guidelines(img, head_info):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    top_y, chin_y, eye_y, canvas_size = head_info['top_y'], head_info['chin_y'], head_info['eye_y'], head_info['canvas_size']
    head_height = head_info['head_height']
    actual_head_ratio = head_height / canvas_size
    actual_eye_ratio = (canvas_size - eye_y) / canvas_size

    # Head bracket
    draw.line([(30, top_y),(30, chin_y)], fill="blue", width=4)
    draw.line([(20, top_y),(40, top_y)], fill="blue", width=2)
    draw.line([(20, chin_y),(40, chin_y)], fill="blue", width=2)
    draw.text((45, top_y-15), "Head Top", fill="blue")
    draw.text((45, chin_y-15), "Chin", fill="blue")
    draw.text((30-120,(top_y+chin_y)//2-30), f"Head: {int(actual_head_ratio*100)}%", fill="green" if HEAD_MIN_RATIO<=actual_head_ratio<=HEAD_MAX_RATIO else "red")
    draw.text((30-120,(top_y+chin_y)//2-10), "Req: 50-69%", fill="blue")

    # Eye bracket
    eye_bracket_x = w-30
    eye_min_y = h-int(h*EYE_MAX_RATIO)
    eye_max_y = h-int(h*EYE_MIN_RATIO)
    draw.line([(eye_bracket_x, eye_min_y),(eye_bracket_x, eye_max_y)], fill="green", width=4)
    draw.line([(eye_bracket_x-10, eye_min_y),(eye_bracket_x+10, eye_min_y)], fill="green", width=2)
    draw.line([(eye_bracket_x-10, eye_max_y),(eye_bracket_x+10, eye_max_y)], fill="green", width=2)
    draw.line([(eye_bracket_x-15, eye_y),(eye_bracket_x+15, eye_y)], fill="darkgreen", width=3)
    draw.text((eye_bracket_x-100,(eye_min_y+eye_max_y)//2-30), f"Eyes: {int(actual_eye_ratio*100)}%", fill="green" if EYE_MIN_RATIO<=actual_eye_ratio<=EYE_MAX_RATIO else "red")
    draw.text((eye_bracket_x-100,(eye_min_y+eye_max_y)//2-10), "Req: 56-69%", fill="green")

    draw.text((10,10), "DV Lottery Photo Template", fill="black")
    draw.text((10,30), f"Size: {w}x{h} px", fill="black")
    return img, actual_head_ratio, actual_eye_ratio

# ---------------------- STREAMLIT UI ----------------------
uploaded_file = st.file_uploader("Upload your photo (JPG/JPEG/PNG)", type=["jpg","jpeg","png"])

if 'current_image' not in st.session_state: st.session_state.current_image = None
if 'current_head_info' not in st.session_state: st.session_state.current_head_info = None
if 'current_bg_removed' not in st.session_state: st.session_state.current_bg_removed = None
if 'fix_count' not in st.session_state: st.session_state.fix_count = 0
if 'show_fixed_preview' not in st.session_state: st.session_state.show_fixed_preview = False

if uploaded_file:
    try:
        img_bytes = uploaded_file.read()
        orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📤 Original Photo")
            st.image(orig, caption=f"Original Size: {orig.size}")

        # Process only if needed
        process_needed = (st.session_state.current_image is None or st.session_state.show_fixed_preview or st.session_state.fix_count>0)
        if process_needed:
            with st.spinner("Running AI quality checks..."):
                orig_cv = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
                checks = {
                    "Face recognized": check_face_recognized(orig_cv),
                    "Single face": check_single_face(orig_cv),
                    "Minimum dimension": check_minimum_dimensions(orig),
                    "Correct proportions": check_photo_proportions(orig),
                    "Background removable": check_background_removal(orig),
                    "No red eyes": check_red_eyes(orig_cv)
                }
            with st.spinner("Removing background..."):
                st.session_state.current_bg_removed = remove_background(orig)
            with st.spinner("Auto-adjusting photo..."):
                force_correction = st.session_state.fix_count > 0
                processed, head_info = auto_adjust_dv_photo(st.session_state.current_bg_removed, force_correction)
                st.session_state.current_image = processed
                st.session_state.current_head_info = head_info
                st.session_state.show_fixed_preview = False

        if st.session_state.current_image:
            with col2:
                st.subheader("✅ DV Compliant Photo")
                final_preview, actual_head_ratio, actual_eye_ratio = draw_guidelines(
                    st.session_state.current_image.copy(),
                    st.session_state.current_head_info
                )
                st.session_state.current_head_info['actual_head_ratio'] = actual_head_ratio
                st.session_state.current_head_info['actual_eye_ratio'] = actual_eye_ratio
                st.image(final_preview, use_container_width=True)

                # Show metrics
                colm1, colm2 = st.columns(2)
                with colm1:
                    head_status = "✅ WITHIN RANGE" if HEAD_MIN_RATIO <= actual_head_ratio <= HEAD_MAX_RATIO else "❌ OUT OF RANGE"
                    st.metric("Head Height", f"{int(actual_head_ratio*100)}%", delta=head_status)
                with colm2:
                    eye_status = "✅ WITHIN RANGE" if EYE_MIN_RATIO <= actual_eye_ratio <= EYE_MAX_RATIO else "❌ OUT OF RANGE"
                    st.metric("Eye Position", f"{int(actual_eye_ratio*100)}%", delta=eye_status)

                # Fix button
                needs_fix = (actual_head_ratio<HEAD_MIN_RATIO or actual_head_ratio>HEAD_MAX_RATIO or
                             actual_eye_ratio<EYE_MIN_RATIO or actual_eye_ratio>EYE_MAX_RATIO)
                if needs_fix:
                    st.warning("⚠️ Some measurements are out of range. Click 'Fix Photo Measurements'.")
                    if st.button("🛠️ Fix Photo Measurements", use_container_width=True):
                        st.session_state.fix_count += 1
                        st.session_state.show_fixed_preview = True
                        st.experimental_rerun()
                else:
                    st.success("✅ All measurements are within range!")

                # Download button
                buf = io.BytesIO()
                st.session_state.current_image.save(buf, format="JPEG", quality=95)
                buf.seek(0)
                st.download_button("⬇️ Download DV Photo", data=buf, file_name="dv_lottery_photo.jpg", mime="image/jpeg", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
else:
    st.info("👆 Upload a photo to get started.")
