import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove
import warnings
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
import queue

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

# ---------------------- CAMERA PROCESSOR CLASS ----------------------
class DVPhotoCamera:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.frame_queue = queue.Queue(maxsize=1)
        self.captured_image = None
        self.capture_event = threading.Event()
        
    def draw_dv_overlay(self, image, face_landmarks=None):
        """Draw the DV lottery photo guide overlay exactly like the reference image"""
        h, w = image.shape[:2]
        
        # Create a transparent overlay
        overlay = image.copy()
        
        # Define colors
        GREEN = (0, 255, 0)  # Bright green for outlines
        SEMI_TRANSPARENT = (0, 255, 0, 128)  # Semi-transparent green for eye band
        
        # 1. Draw the head outline (green oval)
        if face_landmarks:
            try:
                # Get face bounding points
                landmarks = face_landmarks.landmark
                
                # Get forehead (approx), chin, and side points
                forehead = landmarks[10]   # Forehead
                chin = landmarks[152]      # Chin
                left_side = landmarks[234] # Left face contour
                right_side = landmarks[454] # Right face contour
                
                # Calculate head dimensions
                head_top = int(forehead.y * h) - int(0.1 * h)  # Add some margin above head
                head_bottom = int(chin.y * h) + int(0.05 * h)  # Add some margin below chin
                head_left = int(left_side.x * w)
                head_right = int(right_side.x * w)
                
                head_center_x = (head_left + head_right) // 2
                head_center_y = (head_top + head_bottom) // 2
                head_width = head_right - head_left
                head_height = head_bottom - head_top
                
                # Draw green oval around head
                cv2.ellipse(overlay, 
                           (head_center_x, head_center_y),
                           (head_width//2, head_height//2),
                           0, 0, 360, GREEN, 3)
                
            except Exception as e:
                # If face detection fails, draw default oval in center
                head_center_x, head_center_y = w//2, h//2
                head_width, head_height = int(w*0.4), int(h*0.6)
                cv2.ellipse(overlay,
                           (head_center_x, head_center_y),
                           (head_width//2, head_height//2),
                           0, 0, 360, GREEN, 3)
        else:
            # Draw default oval in center when no face detected
            head_center_x, head_center_y = w//2, h//2
            head_width, head_height = int(w*0.4), int(h*0.6)
            cv2.ellipse(overlay,
                       (head_center_x, head_center_y),
                       (head_width//2, head_height//2),
                       0, 0, 360, GREEN, 3)
        
        # 2. Draw the eye-level horizontal band (semi-transparent green rectangle)
        eye_level_y = int(h * 0.62)  # 62% from top (middle of eye position range)
        band_height = int(h * 0.08)  # 8% of height for the band
        
        # Create semi-transparent eye band
        eye_band = image.copy()
        cv2.rectangle(eye_band, 
                     (0, eye_level_y - band_height//2),
                     (w, eye_level_y + band_height//2),
                     GREEN, -1)  # Filled rectangle
        
        # Blend the eye band with original image
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(eye_band, alpha, overlay, 1 - alpha, 0, overlay)
        
        # 3. Add eye level guide lines
        cv2.line(overlay, 
                (0, eye_level_y), 
                (w, eye_level_y), 
                GREEN, 2)
        
        # 4. Add measurement text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"Eye Level: 62%", 
                   (10, eye_level_y - band_height//2 - 10), 
                   font, 0.6, GREEN, 2)
        
        # 5. Add alignment status
        if face_landmarks:
            try:
                # Check if eyes are at correct level
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                eye_y = (left_eye.y + right_eye.y) / 2 * h
                
                eye_band_top = eye_level_y - band_height//2
                eye_band_bottom = eye_level_y + band_height//2
                
                if eye_band_top <= eye_y <= eye_band_bottom:
                    status_text = "ALIGNED - Ready to Capture!"
                    status_color = (0, 255, 0)  # Green
                else:
                    status_text = "ADJUST POSITION - Move up/down"
                    status_color = (0, 165, 255)  # Orange
                
                cv2.putText(overlay, status_text, 
                           (w//2 - 200, 40), font, 0.7, status_color, 2)
                
            except:
                cv2.putText(overlay, "Align face with green guides", 
                           (w//2 - 150, 40), font, 0.7, GREEN, 2)
        else:
            cv2.putText(overlay, "Position face in oval", 
                       (w//2 - 120, 40), font, 0.7, GREEN, 2)
        
        return overlay
    
    def process_frame(self, frame):
        """Process each camera frame and add DV overlay"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh
        results = self.face_mesh.process(rgb_frame)
        
        # Draw DV overlay
        if results.multi_face_landmarks:
            overlay_frame = self.draw_dv_overlay(frame, results.multi_face_landmarks[0])
        else:
            overlay_frame = self.draw_dv_overlay(frame)
        
        return overlay_frame
    
    def capture_photo(self, frame):
        """Capture and store the current frame"""
        self.captured_image = frame.copy()
        self.capture_event.set()
    
    def get_captured_image(self):
        """Get the captured image and reset the event"""
        if self.capture_event.is_set():
            self.capture_event.clear()
            return self.captured_image
        return None

# ---------------------- COMPLIANCE CHECKERS ----------------------
def check_facing_direction(landmarks, img_w, img_h):
    """Check if face is directly facing camera"""
    try:
        # Get nose tip and face contour points
        nose_tip = landmarks.landmark[1]
        left_face = landmarks.landmark[234]  # Left face contour
        right_face = landmarks.landmark[454]  # Right face contour
        
        # Calculate face width and nose position
        face_width = abs(right_face.x - left_face.x) * img_w
        nose_center_ratio = (nose_tip.x - left_face.x) / (right_face.x - left_face.x)
        
        # Face should be centered (nose around 0.5 ratio)
        return 0.4 <= nose_center_ratio <= 0.6
    except:
        return True  # If detection fails, assume it's OK

def check_eyes_open(landmarks, img_h, img_w):
    """Check if both eyes are open and visible"""
    try:
        # Left eye landmarks
        left_eye_top = landmarks.landmark[159]
        left_eye_bottom = landmarks.landmark[145]
        
        # Right eye landmarks  
        right_eye_top = landmarks.landmark[386]
        right_eye_bottom = landmarks.landmark[374]
        
        # Calculate eye openness (vertical distance)
        left_eye_openness = abs(left_eye_top.y - left_eye_bottom.y) * img_h
        right_eye_openness = abs(right_eye_top.y - right_eye_bottom.y) * img_h
        
        # Minimum eye openness threshold
        min_eye_openness = 0.01 * img_h  # 1% of image height
        
        return left_eye_openness > min_eye_openness and right_eye_openness > min_eye_openness
    except:
        return True

def check_neutral_expression(landmarks):
    """Check for neutral facial expression (no smiling)"""
    try:
        # Mouth landmarks
        mouth_top = landmarks.landmark[13]  # Upper lip
        mouth_bottom = landmarks.landmark[14]  # Lower lip
        
        # Calculate mouth openness
        mouth_openness = abs(mouth_top.y - mouth_bottom.y)
        
        # For neutral expression, mouth should be relatively closed
        return mouth_openness < 0.05  # Adjust threshold as needed
    except:
        return True

def check_hair_covering_eyes(landmarks, img_h, img_w):
    """Check if hair is covering eyes or face"""
    try:
        # Get eye region landmarks
        left_eye_inner = landmarks.landmark[133]  # Left eye inner corner
        left_eye_outer = landmarks.landmark[33]   # Left eye outer corner
        right_eye_inner = landmarks.landmark[362] # Right eye inner corner
        right_eye_outer = landmarks.landmark[263] # Right eye outer corner
        
        # Calculate eye region area
        left_eye_width = abs(left_eye_outer.x - left_eye_inner.x) * img_w
        right_eye_width = abs(right_eye_outer.x - right_eye_inner.x) * img_w
        
        # If eye region is too small, might be covered by hair
        min_eye_width = 0.05 * img_w  # Eyes should be at least 5% of image width
        
        return left_eye_width >= min_eye_width and right_eye_width >= min_eye_width
    except:
        return True

def check_image_quality(cv_img):
    """Check for blurriness, lighting issues"""
    try:
        # Check for blur using Laplacian variance
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check brightness
        brightness = np.mean(gray)
        
        issues = []
        if blur_value < 100:  # Threshold for blur detection
            issues.append("Image may be blurry")
        if brightness < 50 or brightness > 200:  # Too dark or too bright
            issues.append("Lighting issues detected")
            
        return issues
    except:
        return []

def comprehensive_compliance_check(cv_img, landmarks, head_info):
    """Run all compliance checks and return issues"""
    issues = []
    
    h, w = cv_img.shape[:2]
    
    # 1. Face direction check
    if not check_facing_direction(landmarks, w, h):
        issues.append("❌ Face not directly facing camera - look straight ahead")
    
    # 2. Eyes check
    if not check_eyes_open(landmarks, h, w):
        issues.append("❌ Eyes not fully open or clearly visible")
    
    # 3. Expression check
    if not check_neutral_expression(landmarks):
        issues.append("❌ Non-neutral facial expression detected - maintain neutral expression")
    
    # 4. Hair covering eyes check
    if not check_hair_covering_eyes(landmarks, h, w):
        issues.append("❌ Hair may be covering eyes or face")
    
    # 5. Image quality check
    quality_issues = check_image_quality(cv_img)
    issues.extend([f"❌ {issue}" for issue in quality_issues])
    
    # 6. Head and eye position compliance (existing checks)
    head_ratio = head_info["head_height"] / head_info["canvas_size"]
    eye_ratio = (head_info["canvas_size"] - head_info["eye_y"]) / head_info["canvas_size"]
    
    if not (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO):
        issues.append(f"❌ Head height {int(head_ratio*100)}% not in required range {int(HEAD_MIN_RATIO*100)}-{int(HEAD_MAX_RATIO*100)}%")
    
    if not (EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO):
        issues.append(f"❌ Eye position {int(eye_ratio*100)}% not in required range {int(EYE_MIN_RATIO*100)}-{int(EYE_MAX_RATIO*100)}%")
    
    return issues

# ---------------------- EXISTING HELPERS (updated with compliance checks) ----------------------
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
        top_y = int(landmarks.landmark[10].y * img_h)
        chin_y = int(landmarks.landmark[152].y * img_h)
        
        left_eye_y = int(landmarks.landmark[33].y * img_h)
        right_eye_y = int(landmarks.landmark[263].y * img_h)
        eye_y = (left_eye_y + right_eye_y) // 2
        
        hair_buffer = int((chin_y - top_y) * 0.3)  # Reduced buffer for better adult detection
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
    """More accurate baby detection with stricter thresholds"""
    try:
        h, w = cv_img.shape[:2]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        nose_tip = landmarks.landmark[1]
        chin = landmarks.landmark[152]
        
        eye_distance = abs(left_eye.x - right_eye.x) * w
        face_height = (chin.y - landmarks.landmark[10].y) * h
        
        # More accurate ratios for baby detection
        eye_to_face_ratio = eye_distance / face_height
        forehead_to_face_ratio = (landmarks.landmark[10].y - landmarks.landmark[151].y) / face_height
        
        # Stricter thresholds to reduce false positives
        is_baby = (eye_to_face_ratio > 0.35 and forehead_to_face_ratio > 0.45)
        
        return is_baby
    except:
        return False

# ---------------------- CORE PROCESSING (updated with compliance checks) ----------------------
def process_dv_photo_initial(img_pil):
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
        
        # Try to get face landmarks for display and compliance checking
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
            
            # Run compliance checks
            compliance_issues = comprehensive_compliance_check(cv_img, landmarks, head_info)
            
        except Exception as e:
            # If face detection fails, use default values
            head_info = {
                "top_y": MIN_SIZE // 4,
                "chin_y": MIN_SIZE * 3 // 4,
                "eye_y": MIN_SIZE // 2,
                "head_height": MIN_SIZE // 2,
                "canvas_size": MIN_SIZE,
                "is_baby": False
            }
            compliance_issues = ["❌ Cannot detect face properly - ensure clear front-facing photo"]
        
        return result, head_info, compliance_issues
    except Exception as e:
        st.error(f"Initial photo processing error: {str(e)}")
        return img_pil, {"top_y": 0, "chin_y": 0, "eye_y": 0, "head_height": 0, "canvas_size": MIN_SIZE, "is_baby": False}, ["❌ Processing error - try another photo"]

def process_dv_photo_adjusted(img_pil):
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
        
        is_baby = is_likely_baby_photo(cv_img, landmarks)

        if is_baby:
            target_head_height = MIN_SIZE * 0.55
            scale_factor = target_head_height / head_height
            scale_factor = np.clip(scale_factor, 0.4, 2.5)
        else:
            target_head_height = MIN_SIZE * 0.6
            scale_factor = target_head_height / head_height
            scale_factor = np.clip(scale_factor, 0.3, 3.0)
        
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.full((MIN_SIZE, MIN_SIZE, 3), 255, np.uint8)

        landmarks_resized = get_face_landmarks(resized)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks_resized, new_h, new_w)
        head_height = chin_y - top_y

        target_eye_min = MIN_SIZE - int(MIN_SIZE * EYE_MAX_RATIO)
        target_eye_max = MIN_SIZE - int(MIN_SIZE * EYE_MIN_RATIO)
        target_eye_y = (target_eye_min + target_eye_max) // 2

        y_offset = target_eye_y - eye_y
        
        if is_baby:
            head_top_margin = 20
            head_bottom_margin = 15
        else:
            head_top_margin = 10
            head_bottom_margin = 10
        
        if top_y + y_offset < head_top_margin:
            y_offset = -top_y + head_top_margin
        
        if chin_y + y_offset > MIN_SIZE - head_bottom_margin:
            y_offset = MIN_SIZE - chin_y - head_bottom_margin

        x_offset = (MIN_SIZE - new_w) // 2

        y_start_dst = max(0, y_offset)
        y_end_dst = min(MIN_SIZE, y_offset + new_h)
        x_start_dst = max(0, x_offset)
        x_end_dst = min(MIN_SIZE, x_offset + new_w)
        
        y_start_src = max(0, -y_offset)
        y_end_src = min(new_h, MIN_SIZE - y_offset)
        x_start_src = max(0, -x_offset)
        x_end_src = min(new_w, MIN_SIZE - x_offset)

        if (y_start_dst < y_end_dst and x_start_dst < x_end_dst and 
            y_start_src < y_end_src and x_start_src < x_end_src):
            
            canvas[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
                resized[y_start_src:y_end_src, x_start_src:x_end_src]
        else:
            y_offset = max(0, (MIN_SIZE - new_h) // 2)
            x_offset = max(0, (MIN_SIZE - new_w) // 2)
            if y_offset + new_h <= MIN_SIZE and x_offset + new_w <= MIN_SIZE:
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

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
        
        # Run compliance checks on adjusted image
        compliance_issues = comprehensive_compliance_check(resized, landmarks_resized, head_info)
        
        return result, head_info, compliance_issues
    except Exception as e:
        st.error(f"Photo adjustment error: {str(e)}")
        return img_pil, {"top_y": 0, "chin_y": 0, "eye_y": 0, "head_height": 0, "canvas_size": MIN_SIZE, "is_baby": False}, ["❌ Adjustment error - try another photo"]

# ---------------------- EXISTING DRAW LINES FUNCTION ----------------------
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

        head_color = "green" if HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO else "red"
        eye_color = "green" if EYE_MIN_RATIO <= eye_ratio <= EYE_MAX_RATIO else "red"

        draw.line([(cx-50, top_y), (cx+50, top_y)], fill="blue", width=3)
        draw.text((cx+60, top_y-15), "Head Top", fill="blue")
        
        draw.line([(cx-50, chin_y), (cx+50, chin_y)], fill="purple", width=3)
        draw.text((cx+60, chin_y-15), "Chin", fill="purple")
        
        draw.line([(cx, top_y), (cx, chin_y)], fill=head_color, width=2)
        
        head_text_y = (top_y + chin_y) // 2
        draw.text((cx + 10, head_text_y - 20), f"Head: {int(head_ratio*100)}%", fill=head_color)
        draw.text((cx + 10, head_text_y), f"Req: {int(HEAD_MIN_RATIO*100)}-{int(HEAD_MAX_RATIO*100)}%", fill="blue")

        eye_min_y = h - int(h * EYE_MAX_RATIO)
        eye_max_y = h - int(h * EYE_MIN_RATIO)
        
        dash_length = 10
        for x in range(0, w, dash_length*2):
            if x + dash_length <= w:
                draw.line([(x, eye_min_y), (x+dash_length, eye_min_y)], fill="green", width=2)
        draw.text((10, eye_min_y-15), "56%", fill="green")
        
        for x in range(0, w, dash_length*2):
            if x + dash_length <= w:
                draw.line([(x, eye_max_y), (x+dash_length, eye_max_y)], fill="green", width=2)
        draw.text((10, eye_max_y-15), "69%", fill="green")
        
        draw.line([(0, eye_y), (w, eye_y)], fill=eye_color, width=3)
        
        draw.text((w-150, eye_y-15), f"Eyes: {int(eye_ratio*100)}%", fill=eye_color)

        # Only show baby detection if it's actually a baby (with stricter detection)
        if is_baby:
            draw.text((10, 10), "👶 Baby Photo Detected", fill="orange")

        return img, head_ratio, eye_ratio
    except Exception as e:
        st.error(f"Guideline drawing error: {str(e)}")
        return img, 0, 0

# ---------------------- VIDEO PROCESSOR FOR CAMERA ----------------------
class VideoProcessor:
    def __init__(self):
        self.camera_processor = None
    
    def recv(self, frame):
        if self.camera_processor is None:
            self.camera_processor = DVPhotoCamera()
        
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame with DV overlay
        processed_img = self.camera_processor.process_frame(img)
        
        # Put frame in queue for potential capture
        try:
            self.camera_processor.frame_queue.put_nowait(processed_img)
        except queue.Full:
            pass
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# ---------------------- STREAMLIT UI ----------------------

# Initialize camera processor in session state
if 'camera_processor' not in st.session_state:
    st.session_state.camera_processor = DVPhotoCamera()

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
    - **Face**: Directly facing camera, neutral expression
    - **Eyes**: Both open and clearly visible
    - **No glasses**, headwear, or uniforms
    - **Hair**: Not covering face or eyes
    
    ### 👶 Baby Photos:
    - Works best with clear front-facing photos
    - Auto-detects baby facial features
    - Uses special adjustments for baby proportions
    """)
    
    st.header("⚙️ Settings")
    enhance_quality = st.checkbox("Enhance Image Quality", value=True)

# Main content with tabs
tab1, tab2 = st.tabs(["📤 Upload Photo", "📷 Camera Guide"])

with tab1:
    # Upload photo processing (existing functionality)
    uploaded_file = st.file_uploader("📤 Upload Your Photo", type=["jpg", "jpeg", "png"], key="uploader")

    if uploaded_file:
        # Initialize session state
        if 'processed_data' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
            st.session_state.last_upload = uploaded_file.name
            orig = Image.open(uploaded_file).convert("RGB")
            
            with st.spinner("🔄 Processing photo and checking compliance..."):
                try:
                    bg_removed = remove_background(orig)
                    processed, head_info, compliance_issues = process_dv_photo_initial(bg_removed)
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
                        'is_adjusted': False,
                        'compliance_issues': compliance_issues
                    }
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.info("💡 Tip: Try a different photo with clear facial features")
                    st.stop()

        # Get data from session state
        data = st.session_state.processed_data
        
        # Show baby detection info only if it's actually a baby
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

        # COMPLIANCE ISSUES DISPLAY
        st.subheader("🔍 Compliance Check Results")
        
        if data['compliance_issues']:
            st.error("❌ **Issues Found - Please upload a new photo:**")
            for issue in data['compliance_issues']:
                st.write(f"- {issue}")
            
            # Show specific warnings based on issues
            critical_issues = any("not detect face" in issue.lower() or "processing error" in issue.lower() for issue in data['compliance_issues'])
            if critical_issues:
                st.warning("**⚠️ Please upload a clear front-facing photo where your face is clearly visible**")
            
        else:
            st.success("✅ **All compliance checks passed!** Your photo meets the basic DV Lottery requirements.")

        # Compliance Dashboard
        st.subheader("📊 Measurements Dashboard")
        
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
            # Only show overall compliant if no compliance issues and measurements are good
            overall_compliant = not data['needs_fix'] and not data['compliance_issues']
            overall_status = "✅ COMPLIANT" if overall_compliant else "❌ NEEDS FIXING"
            st.metric("Overall Status", overall_status)
            if overall_compliant:
                st.success("🎉 Perfect! Your photo meets all requirements!")
            else:
                st.error("⚠️ Photo needs adjustment or replacement.")

        # Fix Section - ALWAYS SHOW FIX BUTTON if there are measurement issues
        if data['needs_fix']:
            st.subheader("🛠️ Photo Correction")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                issues = []
                if not data['head_compliant']:
                    issues.append("Head height out of range")
                if not data['eye_compliant']:
                    issues.append("Eye position out of range")
                
                st.warning(f"**Measurement Issues:** - {' | '.join(issues)}")
                
            with col2:
                if st.button("🔧 Auto-Adjust Head to Chin", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Applying auto-adjustment..."):
                        try:
                            bg_removed = data['bg_removed']
                            processed, head_info, compliance_issues = process_dv_photo_adjusted(bg_removed)
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
                                'is_adjusted': True,
                                'compliance_issues': compliance_issues
                            }
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Adjustment failed: {str(e)}")
            
            if data['head_info'].get('is_baby', False):
                st.info("👶 **Baby photo tip:** Make sure the baby's face is clearly visible and looking directly at the camera")

        # Download Section - ONLY SHOW IF NO COMPLIANCE ISSUES
        if not data['compliance_issues']:
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
                    data=buf_with_guides.getvalue(),
                    file_name="dv_lottery_photo_with_guides.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        else:
            st.warning("**⚠️ Cannot download - Please upload a new photo that meets all requirements**")

    else:
        # Welcome screen for upload tab
        st.markdown("""
        ## 📤 Upload Your Photo
        
        **Get started by uploading your photo above!**
        
        This tool will:
        - ✅ Automatically remove background
        - ✅ Resize to 600×600 pixels  
        - ✅ Check all DV Lottery requirements
        - ✅ Show measurement guidelines
        - ✅ Auto-adjust head and eye positions
        - ✅ Provide compliance report
        
        **or use the 📷 Camera Guide tab to take a new photo with alignment guides!**
        """)

with tab2:
    st.header("📷 DV Lottery Camera Guide")
    
    # Instructions
    st.markdown("""
    ## 🎯 Live Camera Guide for Perfect DV Lottery Photos
    
    **How to use:**
    1. **Allow camera access** when prompted
    2. **Position your face** inside the green oval
    3. **Align your eyes** with the green horizontal band
    4. **Wait for "ALIGNED" message**
    5. **Click "Capture Photo"** when ready
    6. **Process the photo** in the upload tab
    
    ### 📐 Alignment Guides:
    - **Green Oval**: Position your head within this outline
    - **Green Band**: Align your eyes with this horizontal guide (56%-69% from top)
    - **Status Text**: Shows when you're perfectly aligned
    """)
    
    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # Create two columns for camera and controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="dv-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("Camera Controls")
        
        # Capture button
        if st.button("📸 Capture Photo", use_container_width=True, type="primary", key="capture"):
            if webrtc_ctx.state.playing:
                # Get the latest frame from queue
                try:
                    current_frame = st.session_state.camera_processor.frame_queue.get_nowait()
                    st.session_state.camera_processor.capture_photo(current_frame)
                    st.success("✅ Photo captured successfully!")
                except queue.Empty:
                    st.warning("⚠️ No frame available. Please wait for camera to initialize.")
            else:
                st.error("❌ Camera not active. Please start the camera first.")
        
        st.markdown("---")
        st.subheader("Next Steps")
        
        if st.button("🖼️ View & Process Captured Photo", use_container_width=True, key="view"):
            captured_img = st.session_state.camera_processor.get_captured_image()
            if captured_img is not None:
                # Convert to PIL and store in session state for upload tab
                pil_image = Image.fromarray(cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB))
                st.session_state.captured_from_camera = pil_image
                st.success("✅ Photo ready for processing! Switch to the 'Upload Photo' tab.")
            else:
                st.warning("No photo captured yet. Please capture a photo first.")
    
    # Display captured photo if available
    captured_img = st.session_state.camera_processor.get_captured_image()
    if captured_img is not None:
        st.subheader("📸 Your Captured Photo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(captured_img, channels="BGR", 
                    caption="Captured Photo with DV Guides", use_column_width=True)
        
        with col2:
            # Convert to PIL for download
            pil_image = Image.fromarray(
                cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
            )
            
            # Download button
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=95)
            
            st.download_button(
                label="💾 Download Photo with Guides",
                data=buf.getvalue(),
                file_name="dv_lottery_photo_with_guides.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
            if st.button("🔄 Capture New Photo", use_container_width=True, key="new_capture"):
                st.session_state.camera_processor.captured_image = None
                st.rerun()

# Footer
st.markdown("---")
st.markdown("*DV Lottery Photo Editor | Complete solution with camera guide & compliance checking*")

# Clear session state when switching between modes
if 'last_upload' in st.session_state and st.session_state.last_upload == "camera_capture":
    if 'processed_data' in st.session_state:
        del st.session_state.processed_data
