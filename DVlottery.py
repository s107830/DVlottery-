import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io

# You may need: pip install face_recognition opencv-python-headless rembg
import face_recognition
from rembg import remove  # background removal library

def process_image(image: Image.Image) -> Image.Image:
    """Takes PIL image, replaces background white, crops/resizes for DV Lottery compliance."""
    # Convert to RGB
    image = image.convert("RGB")
    np_img = np.array(image)
    # 1. Background removal
    # Using rembg: returns RGBA (with alpha mask)
    result = remove(np_img)
    # Convert RGBA back to RGB with white background
    bgr = cv2.cvtColor(result[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = result[:, :, 3] / 255.0
    white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
    # Composite
    comp = (bgr * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
    rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
    # 2. Face detection & cropping
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        raise Exception("No face detected — please upload a clear frontal photo.")
    # We'll use the largest face
    top, right, bottom, left = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
    # Expand to include hair & shoulders
    height, width = rgb.shape[:2]
    # approximate expansions – you can tweak
    top_exp = max(int(top - 0.3*(bottom-top)), 0)
    bottom_exp = min(int(bottom + 0.2*(bottom-top)), height)
    left_exp = max(int(left - 0.2*(right-left)), 0)
    right_exp = min(int(right + 0.2*(right-left)), width)
    crop = rgb[top_exp:bottom_exp, left_exp:right_exp]
    # 3. Resize to square and ensure head height ratio
    crop_h, crop_w = crop.shape[:2]
    # Determine desired size
    final_size = 600
    # Resize preserving aspect ratio
    if crop_h > crop_w:
        new_w = int(crop_w * final_size / crop_h)
        resized = cv2.resize(crop, (new_w, final_size))
        pad = (final_size - new_w) // 2
        squared = cv2.copyMakeBorder(resized, 0, 0, pad, final_size-new_w-pad,
                                     cv2.BORDER_CONSTANT, value=[255,255,255])
    else:
        new_h = int(crop_h * final_size / crop_w)
        resized = cv2.resize(crop, (final_size, new_h))
        pad = (final_size - new_h) // 2
        squared = cv2.copyMakeBorder(resized, pad, final_size-new_h-pad, 0, 0,
                                     cv2.BORDER_CONSTANT, value=[255,255,255])
    # 4. Convert to PIL Image and return
    final_img = Image.fromarray(squared)
    return final_img

# Streamlit UI
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("DV Lottery Photo Editor — Auto-Crop & White Background")

uploaded_file = st.file_uploader("Upload your photo (jpg or jpeg)", type=["jpg","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    try:
        processed = process_image(image)
        st.image(processed, caption="Processed Image — ready for DV Lottery", use_column_width=True)
        # Provide download
        buf = io.BytesIO()
        processed.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        st.download_button(
            label="Download Ready Photo",
            data=buf,
            file_name="dvlottery_photo.jpg",
            mime="image/jpeg"
        )
    except Exception as e:
        st.error(f"Could not process image: {e}")
