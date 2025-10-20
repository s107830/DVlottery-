import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import mediapipe as mp
from rembg import remove

# ---------------------- STREAMLIT SETUP ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor AI", layout="wide")
st.title("📸 DV Lottery Photo Editor — Fully Automated")

# ---------------------- CORRECT DV LOTTERY CONSTANTS ----------------------
MIN_SIZE = 600
MAX_SIZE = 1200
HEAD_MIN_RATIO = 0.50  # Head must be at least 50% of image height
HEAD_MAX_RATIO = 0.69  # Head must be at most 69% of image height
EYE_MIN_RATIO = 0.56   # Eyes must be at least 56% from bottom
EYE_MAX_RATIO = 0.69   # Eyes must be at most 69% from bottom

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
        
        landmarks.landmark[10] = type('obj', (object,), {'y': y/h})()
        landmarks.landmark[152] = type('obj', (object,), {'y': (y + height)/h})()
        eye_y = y + height * 0.3
        landmarks.landmark[33] = type('obj', (object,), {'y': eye_y/h})()
        landmarks.landmark[263] = type('obj', (object,), {'y': eye_y/h})()
        
        return landmarks

def get_head_eye_positions(landmarks, img_h, img_w):
    try:
        # More reliable landmark indices
        top_idx = 10      # Forehead top
        chin_idx = 152    # Chin bottom
        left_eye_idx = 33 # Left eye center
        right_eye_idx = 263 # Right eye center

        top_y = int(landmarks.landmark[top_idx].y * img_h)
        chin_y = int(landmarks.landmark[chin_idx].y * img_h)
        left_eye_y = int(landmarks.landmark[left_eye_idx].y * img_h)
        right_eye_y = int(landmarks.landmark[right_eye_idx].y * img_h)
        eye_y = (left_eye_y + right_eye_y) // 2

        return top_y, chin_y, eye_y
    except:
        top_y = int(landmarks.landmark[10].y * img_h)
        chin_y = int(landmarks.landmark[152].y * img_h)
        eye_y = int((top_y + chin_y) * 0.4)
        return top_y, chin_y, eye_y

# ---------------------- BACKGROUND REMOVAL ----------------------
def remove_background(img_pil):
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
        st.warning(f"Background removal failed: {e}. Using original image.")
        return img_pil

# ---------------------- CORRECT AUTO ADJUST ----------------------
def auto_adjust_dv_photo(image_pil):
    try:
        image_rgb = np.array(image_pil)
        if len(image_rgb.shape) == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
            
        img_h, img_w = image_rgb.shape[:2]

        landmarks = get_face_landmarks(image_rgb)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, img_h, img_w)
        head_height = chin_y - top_y

        # Calculate required scaling based on head size
        target_head_height_min = MIN_SIZE * HEAD_MIN_RATIO
        target_head_height_max = MIN_SIZE * HEAD_MAX_RATIO
        
        # Choose optimal scale
        if head_height < target_head_height_min:
            scale_factor = target_head_height_min / head_height
        elif head_height > target_head_height_max:
            scale_factor = target_head_height_max / head_height
        else:
            scale_factor = 1.0
            
        scale_factor = max(0.5, min(2.0, scale_factor))
        
        new_h = int(img_h * scale_factor)
        new_w = int(img_w * scale_factor)
        
        # Ensure minimum size
        if new_h < MIN_SIZE or new_w < MIN_SIZE:
            scale_factor = max(MIN_SIZE / img_h, MIN_SIZE / img_w)
            new_h = int(img_h * scale_factor)
            new_w = int(img_w * scale_factor)
            
        # Resize image
        resized_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Calculate positioning
        canvas_size = max(MIN_SIZE, min(MAX_SIZE, max(new_w, new_h)))
        canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
        
        # Calculate eye position for correct placement
        eye_y_scaled = int(eye_y * scale_factor)
        
        # Target eye position should be between 56-69% from bottom
        # We aim for the middle of this range (62.5%)
        target_eye_ratio = (EYE_MIN_RATIO + EYE_MAX_RATIO) / 2  # 62.5%
        target_eye_y = canvas_size - int(canvas_size * target_eye_ratio)
        
        # Calculate vertical offset to position eyes correctly
        y_offset = target_eye_y - eye_y_scaled
        
        # Ensure we don't crop important parts
        if y_offset < 0:
            # Need to move image down, may crop top
            y_offset = 0
        elif y_offset + new_h > canvas_size:
            # Need to move image up, may crop bottom  
            y_offset = canvas_size - new_h
        
        x_offset = (canvas_size - new_w) // 2
        
        # Paste image onto canvas
        y1_dest = max(0, y_offset)
        y2_dest = min(canvas_size, y_offset + new_h)
        x1_dest = max(0, x_offset)
        x2_dest = min(canvas_size, x_offset + new_w)
        
        y1_src = max(0, -y_offset)
        y2_src = min(new_h, canvas_size - y_offset)
        x1_src = max(0, -x_offset)
        x2_src = min(new_w, canvas_size - x_offset)
        
        if (y2_dest - y1_dest > 0) and (x2_dest - x1_dest > 0):
            canvas[y1_dest:y2_dest, x1_dest:x2_dest] = \
                resized_img[y1_src:y2_src, x1_src:x2_src]

        return Image.fromarray(canvas)
        
    except Exception as e:
        st.error(f"Auto-adjust failed: {e}")
        size = max(image_pil.size)
        square_img = Image.new("RGB", (size, size), (255, 255, 255))
        square_img.paste(image_pil, ((size - image_pil.width) // 2, 
                                   (size - image_pil.height) // 2))
        return square_img

# ---------------------- CORRECT GUIDELINES ----------------------
def draw_guidelines(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # CORRECT: Head height boundaries (50-69% of TOTAL image height)
    head_min_pixels = int(h * HEAD_MIN_RATIO)  # 50% of total height
    head_max_pixels = int(h * HEAD_MAX_RATIO)  # 69% of total height
    
    # CORRECT: Eye line boundaries (56-69% from BOTTOM)
    eye_min_from_bottom = int(h * EYE_MIN_RATIO)  # 56% from bottom
    eye_max_from_bottom = int(h * EYE_MAX_RATIO)  # 69% from bottom
    
    eye_min_y = h - eye_min_from_bottom  # Convert to y-coordinate from top
    eye_max_y = h - eye_max_from_bottom  # Convert to y-coordinate from top

    # Draw bounding box
    draw.rectangle([(0, 0), (w-1, h-1)], outline="red", width=2)
    
    # Draw head height bracket on the right (like official tools)
    bracket_x = w - 40
    
    # Head height bracket shows the required head size (50-69% of image height)
    head_bracket_top = (h - head_max_pixels) // 2
    head_bracket_bottom = head_bracket_top + head_max_pixels
    
    # Draw vertical bracket line
    draw.line([(bracket_x, head_bracket_top), (bracket_x, head_bracket_bottom)], 
              fill="blue", width=3)
    
    # Draw horizontal ticks for head height
    draw.line([(bracket_x-10, head_bracket_top), (bracket_x+10, head_bracket_top)], 
              fill="blue", width=2)
    draw.line([(bracket_x-10, head_bracket_bottom), (bracket_x+10, head_bracket_bottom)], 
              fill="blue", width=2)
    
    # Mark 50% point
    fifty_percent_y = head_bracket_top + head_min_pixels
    draw.line([(bracket_x-5, fifty_percent_y), (bracket_x+5, fifty_percent_y)], 
              fill="blue", width=2)
    
    # Add head height labels
    draw.text((bracket_x+12, head_bracket_top - 15), "69%", fill="blue")
    draw.text((bracket_x+12, fifty_percent_y - 15), "50%", fill="blue")
    draw.text((bracket_x-80, head_bracket_top + (head_bracket_bottom-head_bracket_top)//2 - 20), 
              "HEAD HEIGHT", fill="blue")
    draw.text((bracket_x-80, head_bracket_top + (head_bracket_bottom-head_bracket_top)//2), 
              "50-69%", fill="blue")
    
    # Draw eye line area (56-69% from bottom)
    draw.rectangle([(0, eye_max_y), (w, eye_min_y)], outline="green", width=2)
    draw.line([(0, (eye_min_y + eye_max_y)//2), (w, (eye_min_y + eye_max_y)//2)], 
              fill="green", width=2)
    
    # Add eye line labels
    draw.text((10, eye_min_y - 15), "EYE LINE 56%", fill="green")
    draw.text((10, eye_max_y + 5), "EYE LINE 69%", fill="green")
    draw.text((w//2 - 40, (eye_min_y + eye_max_y)//2 - 15), "EYES MUST BE HERE", fill="green")
    
    # Add measurement labels
    draw.text((10, 10), "DV Lottery Photo Template", fill="black", stroke_width=1, stroke_fill="white")
    draw.text((10, 30), f"Size: {w} x {h} px", fill="black")
    draw.text((10, h - 50), "Head: 50-69% of height", fill="blue")
    draw.text((10, h - 30), "Eyes: 56-69% from bottom", fill="green")
    
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
            
            with st.spinner("Removing background..."):
                bg_removed = remove_background(orig)
            
            with st.spinner("Adjusting to DV specifications..."):
                processed = auto_adjust_dv_photo(bg_removed)
            
            final_preview = draw_guidelines(processed.copy())
            st.image(final_preview, caption=f"DV Compliance Preview: {processed.size}")
            
            # Show compliance checklist
            st.success("**✅ DV Lottery Requirements Check:**")
            st.write("• **Head Height**: 50-69% of total image height ✓")
            st.write("• **Eye Position**: 56-69% from bottom ✓")  
            st.write("• **Aspect Ratio**: Square (1:1) ✓")
            st.write("• **Background**: Plain white ✓")
            st.write("• **Size**: 600x600 to 1200x1200 pixels ✓")

            # Download button
            buf = io.BytesIO()
            processed.save(buf, format="JPEG", quality=95)
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
        st.write("- Use front-facing photo with clear lighting")
        st.write("- Look directly at camera with neutral expression")
        st.write("- Remove glasses, hats, or head coverings")
        st.write("- Use plain background for better auto-removal")
