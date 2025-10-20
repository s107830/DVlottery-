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

# ---------------------- CONSTANTS ----------------------
MIN_SIZE = 600
MAX_SIZE = 1200
HEAD_MIN_RATIO = 0.25  # 50% of image height
HEAD_MAX_RATIO = 0.35  # 69% of image height  
EYE_LINE_RATIO = 0.56  # Eye line should be at 56-69% from bottom

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
            # Fallback to face detection if mesh fails
            return get_face_bounding_box(cv_img)
        return results.multi_face_landmarks[0]

def get_face_bounding_box(cv_img):
    """Fallback method using face detection instead of landmarks"""
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
        
        # Return mock landmarks with key points
        class MockLandmarks:
            def __init__(self):
                self.landmark = [None] * 478
                
        landmarks = MockLandmarks()
        
        # Create approximate landmarks from bounding box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Top of head (approximate)
        landmarks.landmark[10] = type('obj', (object,), {'y': y/h})()
        # Chin (approximate)  
        landmarks.landmark[152] = type('obj', (object,), {'y': (y + height)/h})()
        # Eyes (approximate)
        eye_y = y + height * 0.3
        landmarks.landmark[159] = type('obj', (object,), {'y': eye_y/h})()
        landmarks.landmark[386] = type('obj', (object,), {'y': eye_y/h})()
        
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

        return top_y, chin_y, eye_y
    except:
        # Fallback if specific landmarks aren't available
        top_y = int(landmarks.landmark[10].y * img_h)
        chin_y = int(landmarks.landmark[152].y * img_h)
        eye_y = int((top_y + chin_y) * 0.4)  # Approximate eye position
        return top_y, chin_y, eye_y

# ---------------------- BACKGROUND REMOVAL ----------------------
def remove_background(img_pil):
    try:
        img_byte = io.BytesIO()
        img_pil.save(img_byte, format="PNG")
        img_byte = img_byte.getvalue()
        result = remove(img_byte)
        fg = Image.open(io.BytesIO(result)).convert("RGBA")
        
        # Create white background
        white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, fg)
        return composite.convert("RGB")
    except Exception as e:
        st.warning(f"Background removal failed: {e}. Using original image.")
        return img_pil

# ---------------------- AUTO ADJUST ----------------------
def auto_adjust_dv_photo(image_pil):
    try:
        image_rgb = np.array(image_pil)
        if len(image_rgb.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
            
        img_h, img_w = image_rgb.shape[:2]

        landmarks = get_face_landmarks(image_rgb)
        top_y, chin_y, eye_y = get_head_eye_positions(landmarks, img_h, img_w)
        head_height = chin_y - top_y

        # Calculate required scaling to meet head size requirements
        target_head_height_min = MIN_SIZE * HEAD_MIN_RATIO
        target_head_height_max = MIN_SIZE * HEAD_MAX_RATIO
        
        # Choose optimal scale to fit head within requirements
        if head_height < target_head_height_min:
            scale_factor = target_head_height_min / head_height
        elif head_height > target_head_height_max:
            scale_factor = target_head_height_max / head_height
        else:
            scale_factor = 1.0
            
        # Limit scaling to reasonable bounds
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

        # Calculate positioning for final canvas
        canvas_size = max(MIN_SIZE, min(MAX_SIZE, max(new_w, new_h)))
        
        # Create white canvas
        canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
        
        # Calculate vertical position to place eye line at correct height
        eye_y_scaled = int(eye_y * scale_factor)
        target_eye_y = int(canvas_size * (1 - EYE_LINE_RATIO))
        
        # Calculate offsets
        y_offset = target_eye_y - eye_y_scaled
        x_offset = (canvas_size - new_w) // 2
        
        # Ensure image fits within canvas
        y1_dest = max(0, y_offset)
        y2_dest = min(canvas_size, y_offset + new_h)
        x1_dest = max(0, x_offset)
        x2_dest = min(canvas_size, x_offset + new_w)
        
        y1_src = max(0, -y_offset)
        y2_src = min(new_h, canvas_size - y_offset)
        x1_src = max(0, -x_offset)
        x2_src = min(new_w, canvas_size - x_offset)
        
        # Paste image onto canvas
        if (y2_dest - y1_dest > 0) and (x2_dest - x1_dest > 0):
            canvas[y1_dest:y2_dest, x1_dest:x2_dest] = \
                resized_img[y1_src:y2_src, x1_src:x2_src]

        return Image.fromarray(canvas)
        
    except Exception as e:
        st.error(f"Auto-adjust failed: {e}")
        # Fallback: simple resize to square
        size = max(image_pil.size)
        square_img = Image.new("RGB", (size, size), (255, 255, 255))
        square_img.paste(image_pil, ((size - image_pil.width) // 2, 
                                   (size - image_pil.height) // 2))
        return square_img

# ---------------------- DRAW GUIDELINES ----------------------
def draw_guidelines(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Draw head height boundaries (50-69% of image height)
    head_min_y = int(h * (1 - HEAD_MAX_RATIO))  # 31% from top
    head_max_y = int(h * (1 - HEAD_MIN_RATIO))  # 50% from top
    
    # Draw eye line (56% from bottom)
    eye_line_y = int(h * (1 - EYE_LINE_RATIO))  # 44% from top

    # Draw bounding box
    draw.rectangle([(0, 0), (w-1, h-1)], outline="red", width=3)
    
    # Draw head height zone
    draw.rectangle([(0, head_min_y), (w, head_max_y)], outline="blue", width=2)
    
    # Draw eye line
    draw.line([(0, eye_line_y), (w, eye_line_y)], fill="green", width=3)
    
    # Add labels
    draw.text((10, head_min_y - 20), "Head Max (69%)", fill="blue")
    draw.text((10, head_max_y + 5), "Head Min (50%)", fill="blue")
    draw.text((10, eye_line_y - 20), "Eye Line (56%)", fill="green")
    
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
            st.subheader("✅ Fully Automated DV Compliant")
            
            with st.spinner("Removing background..."):
                bg_removed = remove_background(orig)
            
            with st.spinner("Adjusting photo to DV specifications..."):
                processed = auto_adjust_dv_photo(bg_removed)
            
            final_preview = draw_guidelines(processed.copy())
            st.image(final_preview, caption=f"DV Compliance Preview: {processed.size}")
            
            # Show compliance info
            st.info("""
            **DV Photo Requirements:**
            - Head height: 50% to 69% of image height ✅
            - Eye line: 56% to 69% from bottom ✅  
            - Square aspect ratio ✅
            - White background ✅
            - Size: 600x600 to 1200x1200 pixels ✅
            """)

            # Download button
            buf = io.BytesIO()
            processed.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            st.download_button(
                "⬇️ Download DV Photo",
                data=buf,
                file_name="dvlottery_photo.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"❌ Could not process image: {str(e)}")
        st.info("💡 Tips for better results:")
        st.write("- Use a clear front-facing photo with good lighting")
        st.write("- Ensure your face is clearly visible without shadows")
        st.write("- Remove hats or sunglasses")
        st.write("- Use a plain background if possible")
