import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from rembg import remove
import io

st.set_page_config(page_title="📸 DV Lottery Photo Editor — Auto Background, Crop & Guidelines")

st.title("📸 DV Lottery Photo Editor — Auto Background, Crop & Guidelines")
st.write("Upload your photo (JPG/JPEG/PNG). The app will remove the background, crop to 2x2 in (600×600 px), and show DV guideline lines.")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

# ---- Utility: Draw guidelines ----
def draw_guidelines(image: Image.Image):
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # Define DV guideline zones
    head_top_y = int(h * 0.15)
    chin_y = int(h * 0.85)
    eye_y = int(h * 0.55)
    center_x = w // 2

    # Line style
    line_color = (0, 255, 0)
    text_color = (255, 0, 0)

    # Horizontal lines
    draw.line([(0, head_top_y), (w, head_top_y)], fill=line_color, width=2)
    draw.line([(0, eye_y), (w, eye_y)], fill=line_color, width=2)
    draw.line([(0, chin_y), (w, chin_y)], fill=line_color, width=2)
    draw.line([(center_x, 0), (center_x, h)], fill=line_color, width=2)

    # Label text
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Pillow 10.x uses textbbox instead of textsize
    def text_size(draw_obj, text, font):
        bbox = draw_obj.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    labels = [
        ("Head top", head_top_y),
        ("Eye line", eye_y),
        ("Chin", chin_y)
    ]

    for label, y in labels:
        tw, th = text_size(draw, label, font)
        draw.rectangle([(10, y - th - 4), (20 + tw, y)], fill=(255, 255, 255, 180))
        draw.text((15, y - th - 2), label, fill=text_color, font=font)

    return image

# ---- Process uploaded image ----
if uploaded_file is not None:
    try:
        # Read file bytes
        input_bytes = uploaded_file.read()
        input_image = Image.open(io.BytesIO(input_bytes)).convert("RGBA")

        # Remove background
        bg_removed = remove(input_image)

        # Resize to 600x600px (2x2 inch at 300dpi)
        final_img = bg_removed.resize((600, 600))

        # Clean transparent edges with white background
        white_bg = Image.new("RGBA", final_img.size, (255, 255, 255, 255))
        white_bg.paste(final_img, mask=final_img.getchannel("A"))
        cleaned = white_bg.convert("RGB")

        # Draw guidelines
        final_preview = draw_guidelines(cleaned.copy())

        # Display side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="📤 Uploaded Photo", use_container_width=True)
        with col2:
            st.image(final_preview, caption="✅ DV-Compliant Photo with Guidelines", use_container_width=True)

        # Download button
        buf = io.BytesIO()
        final_preview.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button("📥 Download Final Photo", data=byte_im, file_name="dv_lottery_photo.jpg", mime="image/jpeg")

    except Exception as e:
        st.error(f"❌ Could not process image: {e}")
else:
    st.info("Please upload a photo to begin.")
