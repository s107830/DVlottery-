import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
from rembg import remove
import io

# --- Constants ---
FINAL_PX = 600
HEAD_MIN_PX = 300
HEAD_MAX_PX = 414
EYE_LINE_MIN_PX = 336
EYE_LINE_MAX_PX = 414
BG_COLOR = (255, 255, 255)

# --- Helper: Detect face using OpenCV Haar cascade ---
def detect_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(faces[0])  # (x, y, w, h)

# --- Background remover ---
def remove_background_smooth(img_pil):
    """Remove background and clean edge halo."""
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format="PNG")
    result = remove(img_byte.getvalue())

    fg = Image.open(io.BytesIO(result)).convert("RGBA")
    fg_np = np.array(fg)
    alpha = fg_np[:, :, 3].astype(np.float32) / 255.0

    # Edge cleanup
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    fg_rgb = fg_np[:, :, :3].astype(np.float32)
    bg_white = np.ones_like(fg_rgb) * 255.0
    clean_rgb = fg_rgb * alpha[..., None] + bg_white * (1 - alpha[..., None])

    return Image.fromarray(np.uint8(clean_rgb)).convert("RGB")

# --- Auto-center, scale & crop face ---
def auto_center_crop(image_pil, face_box):
    img_np = np.array(image_pil)
    h, w, _ = img_np.shape

    if not face_box:
        return image_pil.resize((FINAL_PX, FINAL_PX)), None

    x, y, fw, fh = face_box
    cx = x + fw // 2
    cy = y + fh // 2

    # Target head height ~360px → scale accordingly
    scale = 360 / fh
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cx = int(cx * scale)
    cy = int(cy * scale)

    # Crop around centered face
    crop_size = FINAL_PX
    left = max(0, cx - crop_size // 2)
    top = max(0, cy - crop_size // 2)
    right = left + crop_size
    bottom = top + crop_size

    # Pad if edges exceed
    pad_x = max(0, crop_size - (right - left))
    pad_y = max(0, crop_size - (bottom - top))

    cropped = np.full((crop_size, crop_size, 3), 255, dtype=np.uint8)
    crop_img = resized[top:bottom, left:right]

    y_off = (crop_size - crop_img.shape[0]) // 2
    x_off = (crop_size - crop_img.shape[1]) // 2
    cropped[y_off:y_off + crop_img.shape[0], x_off:x_off + crop_img.shape[1]] = crop_img

    return Image.fromarray(cropped), (x, y, fw, fh)

# --- Draw DV photo guidelines dynamically ---
def draw_guidelines(img, face_box=None):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
