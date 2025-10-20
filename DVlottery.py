import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from rembg import remove
import io
import math

# --------------------------
# Config
# --------------------------
FINAL_SIZE = 600
HEAD_MIN_RATIO = 0.50   # 50% of image height
HEAD_MAX_RATIO = 0.69   # 69% of image height
EYE_MIN_FROM_BOTTOM_RATIO = 0.56  # 56% from bottom
EYE_MAX_FROM_BOTTOM_RATIO = 0.69  # 69% from bottom
BG_COLOR = (255, 255, 255)

st.set_page_config(page_title="DV / Green Card Photo Checker", layout="centered", page_icon="📸")
st.title("📸 DV / Green Card Photo Checker — Official-style Guidelines")
st.write("Upload a frontal photo (JPG/JPEG/PNG). The tool will auto-remove background, center the face, and show official guidelines like GreenCardPhotoCheck.")

# --------------------------
# Helper: robust rembg -> cleaned image
# --------------------------
def remove_background_and_clean(img_pil: Image.Image) -> Image.Image:
    """
    Use rembg on PNG bytes, then clean alpha mask to remove faint halo while preserving hair.
    Returns RGB image composited over white background.
    """
    img_rgba = img_pil.convert("RGBA")
    b = io.BytesIO()
    img_rgba.save(b, format="PNG")
    input_bytes = b.getvalue()

    out_bytes = remove(input_bytes)
    fg = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    fg_np = np.array(fg)  # HxWx4

    # Extract raw alpha (0..1)
    alpha = fg_np[:, :, 3].astype(np.float32) / 255.0

    # Create binary-ish mask but keep hair: threshold then morphological close -> slight erosion -> blur
    thresh = 0.14  # tuned: lower keeps more hair, higher removes more fringe
    mask = (alpha > thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)

    # Rebuild RGB by blending with white background using cleaned mask
    fg_rgb = fg_np[:, :, :3].astype(np.float32)
    alpha_clean = (mask_blur.astype(np.float32) / 255.0)[..., None]
    white = np.ones_like(fg_rgb) * 255.0
    clean_rgb = (fg_rgb * alpha_clean) + (white * (1.0 - alpha_clean))
    clean_img = Image.fromarray(np.uint8(np.clip(clean_rgb, 0, 255)))
    return clean_img.convert("RGB")

# --------------------------
# Face detect & crop to square, return resized image and face_box in resized coords
# --------------------------
def detect_face_box(np_rgb):
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return (int(x), int(y), int(w), int(h))

def crop_center_face_and_resize(img_pil: Image.Image, desired_px=FINAL_SIZE):
    """
    Crop a square region around the main face and resize to desired_px.
    Returns (resized RGB PIL image, face_box_in_resized_coords, approx_alpha_mask_resized)
    """
    np_img = np.array(img_pil.convert("RGB"))
    h_full, w_full = np_img.shape[:2]
    face_box = detect_face_box(np_img)

    if face_box is None:
        # center crop fallback
        side = min(w_full, h_full)
        left = (w_full - side) // 2
        top = (h_full - side) // 2
        crop = img_pil.crop((left, top, left + side, top + side))
        resized = crop.resize((desired_px, desired_px), Image.LANCZOS)
        # alpha mask approximation: non-white pixels after compositing -> assume subject
        resized_np = np.array(resized.convert("RGB"))
        alpha_mask = (~np.all(resized_np >= 250, axis=2)).astype(np.uint8) * 255
        return resized.convert("RGB"), None, alpha_mask

    x, y, fw, fh = face_box
    cx = x + fw // 2
    cy = y + fh // 2

    # crop square region with multiplier tuned for shoulders & hair
    box_size = int(max(fw, fh) * 2.15)
    left = max(0, cx - box_size // 2)
    top = max(0, cy - box_size // 2)
    right = min(w_full, cx + box_size // 2)
    bottom = min(h_full, cy + box_size // 2)

    cropped = img_pil.crop((left, top, right, bottom))
    resized = cropped.resize((desired_px, desired_px), Image.LANCZOS)

    # approximate alpha mask by non-white detection (since we composited on white)
    resized_np = np.array(resized.convert("RGB"))
    alpha_mask = (~np.all(resized_np >= 250, axis=2)).astype(np.uint8) * 255

    # map face_box into resized coords
    scale_x = desired_px / (right - left)
    scale_y = desired_px / (bottom - top)
    fx = int((x - left) * scale_x)
    fy = int((y - top) * scale_y)
    fw_r = int(fw * scale_x)
    fh_r = int(fh * scale_y)
    face_box_resized = (fx, fy, fw_r, fh_r)

    return resized.convert("RGB"), face_box_resized, alpha_mask

# --------------------------
# Eye detection (within face box) for more accurate eye line
# --------------------------
def detect_eyes_y(np_rgb, face_box_resized):
    if face_box_resized is None:
        return None
    fx, fy, fw, fh = face_box_resized
    # crop face region from resized image
    h, w = np_rgb.shape[:2]
    x0 = max(0, fx); y0 = max(0, fy); x1 = min(w, fx + fw); y1 = min(h, fy + fh)
    face_roi = np_rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
    if len(eyes) == 0:
        return None
    # take the two highest-scoring (largest) eyes and compute their center y in resized coords
    eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    ys = []
    for ex, ey, ew, eh in eyes_sorted:
        ys.append(y0 + ey + eh // 2)
    if len(ys) == 0:
        return None
    return int(np.mean(ys))

# --------------------------
# Compute guideline positions from alpha mask and face box
# --------------------------
def compute_positions(resized_img: Image.Image, face_box, alpha_mask):
    w, h = resized_img.size

    head_top = None
    chin = None
    if alpha_mask is not None:
        rows_any = np.any(alpha_mask > 0, axis=1)
        if rows_any.any():
            ys = np.where(rows_any)[0]
            head_top = int(ys[0])
            chin = int(ys[-1])

    # fallback to face_box geometry
    if head_top is None or chin is None:
        if face_box is not None:
            fx, fy, fw, fh = face_box
            head_top = max(0, fy - int(0.25 * fh))
            chin = min(h - 1, fy + fh + int(0.06 * fh))
        else:
            head_top = int(h * 0.16)
            chin = int(h * 0.62)

    # eye line: try to detect eyes precisely
    np_rgb = np.array(resized_img)
    eye_y_detected = detect_eyes_y(np_rgb, face_box)
    if eye_y_detected is not None:
        eye_y = int(eye_y_detected)
    else:
        if face_box is not None:
            fx, fy, fw, fh = face_box
            eye_y = int(fy + 0.45 * fh)
        else:
            eye_y = int((head_top + chin) / 2)

    # clip
    head_top = int(np.clip(head_top, 0, h - 1))
    chin = int(np.clip(chin, 0, h - 1))
    eye_y = int(np.clip(eye_y, 0, h - 1))

    head_height_px = chin - head_top
    eye_from_bottom_px = h - eye_y

    return {
        "head_top": head_top,
        "chin": chin,
        "eye_y": eye_y,
        "head_height_px": head_height_px,
        "eye_from_bottom_px": eye_from_bottom_px,
        "img_w": w,
        "img_h": h,
    }

# --------------------------
# Draw exact GreenCardPhotoCheck-style guidelines & pass/fail badge
# --------------------------
def draw_guidelines_and_badge(img: Image.Image, pos: dict):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # dashed center vertical
    dash_h = 8
    gap = 6
    x_center = w // 2
    for y in range(0, h, dash_h + gap):
        draw.line([(x_center, y), (x_center, min(h, y + dash_h))], fill=(120, 120, 120), width=2)

    # Head top & chin lines (blue)
    draw.line([(0, pos["head_top"]), (w, pos["head_top"])], fill=(12, 102, 204, 255), width=3)
    draw.line([(0, pos["chin"]), (w, pos["chin"])], fill=(12, 102, 204, 255), width=3)

    # Eye line (green)
    draw.line([(0, pos["eye_y"]), (w, pos["eye_y"])], fill=(10, 150, 60, 255), width=3)

    # semi-transparent range highlight rectangles
    # head-height acceptable zone (centered vertically where head_top->chin would be acceptable)
    # Instead show text status instead of shading to keep clarity

    # labels: put px and inch equivalents
    # inches = px / 300 * 1 inch (assuming 300 DPI). But we just show px and percent
    hh = pos["head_height_px"]
    eyefb = pos["eye_from_bottom_px"]
    head_pct = round(hh / h * 100, 1)
    eyefb_pct = round(eyefb / h * 100, 1)

    # status text
    head_ok = (FINAL_SIZE * HEAD_MIN_RATIO) <= hh <= (FINAL_SIZE * HEAD_MAX_RATIO)
    eye_ok = (FINAL_SIZE * EYE_MIN_FROM_BOTTOM_RATIO) <= eyefb <= (FINAL_SIZE * EYE_MAX_FROM_BOTTOM_RATIO)
    overall_ok = head_ok and eye_ok

    # draw status banner top-left
    status_text = "✓ Compliant" if overall_ok else "✗ Not Compliant"
    status_color = (0, 160, 80) if overall_ok else (200, 30, 30)
    # small rectangle behind text
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        f = None
    tw, th = draw.textsize(status_text, font=f)
    draw.rectangle([(8, 8), (12 + tw, 12 + th)], fill=(255, 255, 255))
    draw.text((10, 10), status_text, fill=status_color, font=f)

    # detailed metrics (left side)
    metrics_y = 40
    draw.text((10, metrics_y), f"Head height: {hh}px ({head_pct}%)", fill=(12, 102, 204), font=f)
    draw.text((10, metrics_y + 18), f"Eye from bottom: {eyefb}px ({eyefb_pct}%)", fill=(10, 150, 60), font=f)
    draw.text((10, metrics_y + 36), f"Required head: {int(FINAL_SIZE*HEAD_MIN_RATIO)}–{int(FINAL_SIZE*HEAD_MAX_RATIO)} px", fill=(60,60,60), font=f)
    draw.text((10, metrics_y + 54), f"Required eye: {int(FINAL_SIZE*EYE_MIN_FROM_BOTTOM_RATIO)}–{int(FINAL_SIZE*EYE_MAX_FROM_BOTTOM_RATIO)} px from bottom", fill=(60,60,60), font=f)

    # small footer
    draw.text((w - 210, h - 20), f"2x2 inch ({w}x{h} px)", fill=(70, 70, 70), font=f)

    # border
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(90, 90, 90), width=2)
    return img

# --------------------------
# Streamlit flow
# --------------------------
uploaded = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        raw = uploaded.read()
        orig = Image.open(io.BytesIO(raw)).convert("RGB")
        st.subheader("Original")
        st.image(orig)

        # Step 1: remove background & clean halo
        with st.spinner("Removing background and cleaning hair..."):
            cleaned = remove_background_and_clean(orig)

        # Step 2: crop & resize to 600x600 and get alpha mask + face box
        resized, face_box_resized, alpha_mask = crop_center_face_and_resize(cleaned, desired_px=FINAL_SIZE)

        # Step 3: compute positions (head_top, chin, eye_y, head_height, eye_from_bottom)
        pos = compute_positions(resized, face_box_resized, alpha_mask)

        # Step 4: draw guidelines + badge
        guided = draw_guidelines_and_badge(resized.copy(), pos)

        st.subheader("Processed (600×600) with official-like guidelines")
        st.image(guided)

        # Download clean & guided
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ Download Clean 600×600", data=buf.getvalue(), file_name="dv_clean_600.jpg", mime="image/jpeg")

        buf2 = io.BytesIO()
        guided.save(buf2, format="JPEG", quality=95)
        st.download_button("⬇️ Download Guided 600×600", data=buf2.getvalue(), file_name="dv_guided_600.jpg", mime="image/jpeg")
    except Exception as e:
        st.error(f"❌ Could not process image: {e}")
else:
    st.info("Upload a clear, front-facing photo to run the DV check.")
