
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import pandas as pd
import os
import requests
from datetime import datetime
import time
from huggingface_hub import hf_hub_download

# ---------- Page config + background ----------
st.set_page_config(page_title="Fabric Defect Detection", page_icon="🧵", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #d7bde2 100%);
    color: white;
}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label {
    color: #ffffff !important;
}
.stButton>button {
    background-color: #ff7043;
    color: white;
    border-radius: 8px;
    border: none;
}
.stFileUploader, .stRadio, .stAlert {
    background-color: rgba(255,255,255,0.08);
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Model loader (Hugging Face option) ----------
@st.cache_resource
def load_model():
    st.write("Downloading model from Hugging Face... Please wait.")

    # Downloading your model from your HF repo
    model_path = hf_hub_download(
        repo_id="Umaar49/fabric-defect-yolov8",
        filename="best.pt",
        repo_type="model"
    )

    # Load the model into YOLO
    model = YOLO(model_path)
    st.write("Model loaded successfully!")
    return model


# Call the function
model = load_model()

# ---------- ESP8266 ----------
ESP8266_IP = "http://192.168.29.204"
last_signal_time = [0]  # mutable holder for throttle

def send_signal_to_esp8266(throttle_sec=3):
    now = time.time()
    if now - last_signal_time[0] < throttle_sec:
        return
    last_signal_time[0] = now
    try:
        r = requests.get(f"{ESP8266_IP}/defect_detected", timeout=2)
        st.toast(f"ESP8266: {r.text}")
    except Exception as e:
        st.warning(f"ESP8266 unreachable: {e}")

# ---------- CSV logging ----------
def log_defects(file_type, defect_counts):
    csv_file = "defect_logs.csv"
    entry = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "File Type": file_type,
             "Total Defects": sum(defect_counts.values())}
    entry.update(defect_counts)
    df = pd.DataFrame([entry])
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)
    return csv_file

defect_names = {0:"Hole", 1:"Stain", 2:"Broken Needle", 3:"Thread", 4:"Yarn Short"}

def resize_image(img, w, h):
    return cv2.resize(img, (w, h))

def count_defects(result):
    counts = {n:0 for n in defect_names.values()}
    for box in result[0].boxes:
        cls = int(box.cls[0])
        if cls in defect_names:
            counts[defect_names[cls]] += 1
    return counts

# ---------- UI ----------
st.title("🧵 Fabric Defect Detection Using Image Processing")
st.subheader("Detects: Hole • Stain • Broken Needle • Yarn Short • Thread")

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

file_type = st.radio("Select input type", ("Image", "Video", "Real-Time Image", "Real-Time Video"))

# ============ IMAGE UPLOAD ============
if file_type == "Image":
    f = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])
    if f:
        img = Image.open(f).convert("RGB")
        st.image(img, caption="Uploaded", use_column_width=True)
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_bgr = resize_image(img_bgr, 640, 640)
        result = model(img_bgr)
        ann = result[0].plot()
        ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        st.image(ann, caption="Detected", use_column_width=True)
        counts = count_defects(result)
        for k,v in counts.items():
            st.write(f"**{k}**: {v}")
        csv_path = log_defects("Image", counts)
        st.success(f"Logged → {csv_path}")
        st.dataframe(pd.read_csv(csv_path))
        if sum(counts.values()) > 0:
            send_signal_to_esp8266()

# ============ VIDEO UPLOAD ============
elif file_type == "Video":
    f = st.file_uploader("Upload a video", type=['mp4','avi','mov'])
    if f:
        with open("temp_video.mp4","wb") as tmp:
            tmp.write(f.read())
        cap = cv2.VideoCapture("temp_video.mp4")
        placeholder = st.empty()
        total = {n:0 for n in defect_names.values()}
        signal_sent = False
        progress = st.progress(0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = resize_image(frame, 640, 640)
            result = model(frame)
            ann = result[0].plot()
            ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            placeholder.image(ann, channels="RGB", use_column_width=True)
            counts = count_defects(result)
            for k,v in counts.items(): total[k] += v
            if not signal_sent and sum(counts.values())>0:
                send_signal_to_esp8266()
                signal_sent = True
            i += 1
            progress.progress(min(i/total_frames, 1.0))
        cap.release()
        st.success("✅ Video processed")
        st.write("### Total Defect Counts")
        for k,v in total.items(): st.write(f"**{k}**: {v}")
        csv_path = log_defects("Video", total)
        st.success(f"Saved → {csv_path}")
        st.dataframe(pd.read_csv(csv_path))

# ============ REAL-TIME IMAGE (camera_input) ============
elif file_type == "Real-Time Image":
    st.write("Capture an image using your webcam:")
    picture = st.camera_input("Take a picture")
    if picture:
        img = Image.open(picture).convert("RGB")
        st.image(img, caption="Captured", use_column_width=True)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_bgr = resize_image(img_bgr, 640, 640)
        result = model(img_bgr)
        ann = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
        st.image(ann, caption="Detected", use_column_width=True)
        counts = count_defects(result)
        for k,v in counts.items(): st.write(f"**{k}**: {v}")
        csv_path = log_defects("Real-Time Image", counts)
        st.success(f"Logged → {csv_path}")
        if sum(counts.values())>0:
            send_signal_to_esp8266()

# ============ REAL-TIME VIDEO ============
elif file_type == "Real-Time Video":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Webcam"):
            st.session_state.webcam_active = True
    with col2:
        if st.button("⏹ Stop Webcam"):
            st.session_state.webcam_active = False

    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        placeholder = st.empty()
        total = {n:0 for n in defect_names.values()}
        prev = {n:0 for n in defect_names.values()}
        signal_sent = False
        run_placeholder = st.empty()
        run_placeholder.write("🔴 Live webcam running — click Stop to end & save.")
        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not accessible")
                break
            frame = resize_image(frame, 640, 640)
            result = model(frame)
            ann = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
            placeholder.image(ann, channels="RGB", use_column_width=True)
            cur = count_defects(result)
            for k in total:
                if cur[k] > prev[k]:
                    total[k] += (cur[k]-prev[k])
            prev = cur
            if not signal_sent and sum(cur.values())>0:
                send_signal_to_esp8266()
                signal_sent = True
            time.sleep(0.05)
        cap.release()
        run_placeholder.empty()
        st.write("### Total Defect Counts")
        for k,v in total.items(): st.write(f"**{k}**: {v}")
        csv_path = log_defects("Real-Time Video", total)
        st.success(f"Saved → {csv_path}")
        st.dataframe(pd.read_csv(csv_path))
