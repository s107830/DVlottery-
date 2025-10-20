import streamlit as st
from PIL import Image, ImageDraw
import io
import numpy as np
import cv2
from rembg import remove

# -------------------------------
# 🎯 App Header
# -------------------------------
st.set_page_config(page_title="DV Lottery Photo Editor", page_icon="📸", layout="centered")
st.title("📸 DV Lottery Photo Editor — Auto Background, Crop & Guidelines")
st.write("Upload your photo (JPG/JPEG/PNG)")

# -------------------------------
# 📁 File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image file",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop file here. Limit 200MB per file • JPG, JPEG, PNG"
)

# -------------------------------
# ⚙️ Helper Functions
# -------------------------------
def remove_background_smooth(img: Image.Image) -> Image.Image:
    """Remove background and return a clean white background image."""
    img = img.convert("RGB")
    img_array = np.array(img)
    result = remove(img_array)
    no_bg = Image.open(io.BytesIO(result)).convert("RGBA")

    # Replace transparent areas with white
    white_bg = Image.new("RGBA", no_bg.size, (255, 255, 255, 255))
    white_bg.paste(no_bg, mask=no_bg.getchannel("A"))
    return white_bg.convert("RGB")

def crop_and_resize(img: Image.Image):
    """Crop around the face and resize to 600x600 for DV Lottery."""
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        raise Exception("No face detected! Please upload a clearer, front-facing photo.")

    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

    # Center crop box around the detected face
    cx, cy = x + w // 2, y + h // 2
    box_size = int(max(w, h) * 2.2)
    left = max(0, cx - box_size // 2)
    top = max(0, cy - box_size // 2)
    right = min(np_img.shape[1], cx + box_size // 2)
    bottom = min(np_img.shape[0], cy + box_size // 2)

    cropped = img.crop((left, top, right, bottom))
    cropped = cropped.resize((600, 600), Image.LANCZOS)
    return cropped, (x, y, w, h)

def draw_guidelines(img: Image.Image, face_box=None):
    """Draw head and eye-line guidelines on the photo."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Head height guidelines (300–414px)
    draw.line([(0, h - 414), (w, h - 414)], fill="red", width=2)
    draw.line([(0, h - 300), (w, h - 300)], fill="red", width=2)

    # Eye line guidelines (336–414px)
    draw.line([(0, h - 414), (w, h - 414)], fill="blue", width=1)
    draw.line([(0, h - 336), (w, h - 336)], fill="blue", width=1)

    # Face box (optional)
    if face_box:
        (x, y, w_box, h_box) = face_box
        draw.rectangle([x, y, x + w_box, y + h_box], outline="green", width=2)

    return img

# -------------------------------
# 🚀 Processing Section
# -------------------------------
if uploaded_file:
    try:
        img_bytes = uploaded_file.read()
        orig = Image.open(io.BytesIO(img_bytes))
        if orig.mode != "RGB":
            orig = orig.convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Original Photo")
            st.image(orig)

        with col2:
            st.subheader("✅ Final DV-Compliant Photo")

            # Process image
            orig_large = orig.resize((orig.width * 2, orig.height * 2), Image.LANCZOS)
            cleaned = remove_background_smooth(orig_large)
            cropped, face_box = crop_and_resize(cleaned)
            final_preview = draw_guidelines(cropped.copy(), face_box=face_box)

            # Display final output
            if isinstance(final_preview, Image.Image):
                st.image(final_preview, caption="DV 2x2 in format")
            else:
                st.warning("⚠️ Could not render final preview. Check face detection or image type.")

            # Download button
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            st.download_button(
                "⬇️ Download DV-Ready Photo (600x600)",
                data=buf,
                file_name="dvlottery_photo.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(f"❌ Could not process image: {e}")
else:
    st.info("👆 Upload your photo to start.")
