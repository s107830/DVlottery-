import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2, io, warnings
import mediapipe as mp
from rembg import remove
warnings.filterwarnings("ignore")

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="DV Lottery Photo Editor", layout="wide")
st.title("DV Lottery Photo Editor — Accurate Head/Eye Calibration v3")

MIN_SIZE = 600
mp_face_mesh = mp.solutions.face_mesh
HEAD_RANGE = (0.50, 0.69)
EYE_RANGE  = (0.56, 0.69)

# ---------------------- LANDMARK HELPERS ----------------------
def get_face_landmarks(cv_img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=True, min_detection_confidence=0.4) as fm:
        res = fm.process(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            raise Exception("No face detected")
        return res.multi_face_landmarks[0]

def measure_positions(landmarks, h, w):
    top    = int(landmarks.landmark[10].y * h)
    chin   = int(landmarks.landmark[152].y * h)
    left_e = int(landmarks.landmark[33].y * h)
    right_e= int(landmarks.landmark[263].y * h)
    eye_y  = (left_e + right_e)//2
    face_h = chin - top
    # add 12 % buffer for hair
    top = max(0, top - int(face_h*0.12))
    return top, chin, eye_y

# ---------------------- BACKGROUND ----------------------
def remove_bg(img):
    try:
        b = io.BytesIO(); img.save(b, format="PNG")
        fg = Image.open(io.BytesIO(remove(b.getvalue()))).convert("RGBA")
        white = Image.new("RGBA", fg.size, (255,255,255,255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except:
        return img

# ---------------------- CORE AUTO-CROP ----------------------
def crop_dv(img_pil, auto=True):
    cv_img = np.array(img_pil.convert("RGB"))
    h, w = cv_img.shape[:2]
    try:
        lm = get_face_landmarks(cv_img)
        top, chin, eye_y = measure_positions(lm, h, w)
    except:
        resized = cv2.resize(cv_img,(MIN_SIZE,MIN_SIZE))
        return Image.fromarray(resized), dict(top_y=150,chin_y=400,eye_y=270,head_h=250)

    # target head = 60 % of frame, eyes ≈ 58 %
    head_h = chin - top
    target_head = 0.60 * MIN_SIZE
    scale = target_head / head_h
    new_w,new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(cv_img,(new_w,new_h))

    canvas = np.full((MIN_SIZE,MIN_SIZE,3),255,np.uint8)

    # vertical offset so that eyes land near 58 % of image height
    eye_scaled = eye_y * scale
    target_eye = MIN_SIZE * 0.58
    y_off = int(target_eye - eye_scaled)
    x_off = (MIN_SIZE - new_w)//2
    y_off = max(min(y_off,MIN_SIZE-new_h),0)
    canvas[y_off:y_off+new_h,x_off:x_off+new_w] = resized[:MIN_SIZE-y_off,:MIN_SIZE-x_off]

    info = dict(top_y=top*scale+y_off, chin_y=chin*scale+y_off,
                eye_y=eye_scaled+y_off, head_h=head_h*scale)
    return Image.fromarray(canvas), info

# ---------------------- DRAW + ANALYZE ----------------------
def annotate(img, info):
    draw = ImageDraw.Draw(img)
    w,h = img.size
    t,c,e = info["top_y"], info["chin_y"], info["eye_y"]
    head_r = (c - t)/h
    eye_r  = (h - e)/h
    ok = HEAD_RANGE[0]<=head_r<=HEAD_RANGE[1] and EYE_RANGE[0]<=eye_r<=EYE_RANGE[1]
    color = "green" if ok else "red"
    draw.line([(0,t),(w,t)],fill="green",width=2)
    draw.line([(0,e),(w,e)],fill="orange",width=2)
    draw.line([(0,c),(w,c)],fill="red",width=2)
    box=[(10,10),(220,85)]
    draw.rectangle(box,outline=color,fill="white",width=3)
    draw.text((20,15),"PASS" if ok else "FAIL",fill=color)
    draw.text((20,35),f"Head: {int(head_r*100)}%",fill="black")
    draw.text((20,50),f"Eyes: {int(eye_r*100)}%",fill="black")
    return img,head_r,eye_r,ok

# ---------------------- STREAMLIT UI ----------------------
st.sidebar.markdown("""
**DV Lottery Requirements**  
• Image 600×600 white background  
• Head 50–69 % of frame  
• Eyes 56–69 % from bottom  
• Centered, front-facing, neutral expression
""")

upl = st.file_uploader("Upload photo (JPG/PNG)",type=["jpg","jpeg","png"])
if upl:
    src = Image.open(upl).convert("RGB")
    with st.spinner("Processing photo …"):
        clean = remove_bg(src)
        cropped,info = crop_dv(clean)
        out,hr,er,ok = annotate(cropped.copy(),info)

    c1,c2=st.columns(2)
    c1.image(src,use_column_width=True,caption="Original")
    c2.image(out,use_column_width=True,caption="Processed 600×600")

    buf=io.BytesIO(); cropped.save(buf,format="JPEG",quality=95)
    st.download_button("📥 Download DV Photo",buf.getvalue(),"dv_photo.jpg","image/jpeg")

    st.info(f"**Analysis** Head {hr*100:.1f}% | Eyes {er*100:.1f}% → {'✅ Pass' if ok else '❌ Needs adjust'}")
else:
    st.write("📸 Upload a photo to auto-crop to DV Lottery standard.")
st.caption("DV Lottery Photo Editor v3 — Accurate Head/Eye Calibration")
