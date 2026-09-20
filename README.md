# 🧵 Fabric Defect Detection System using YOLOv8
##📌 Overview
- This project is a real-time fabric defect detection system built using YOLOv8. The model was trained on self-collected and annotated raw fabric images gathered during an on-site industry visit. It achieves a 95% F1-score across both images and videos.

- The system is deployed via an interactive Streamlit interface that supports both uploaded files and live webcam inference. When a defect is detected, it triggers an ESP8266-based hardware alert via HTTP, activating live LED/alarm indicators. All detection results are automatically logged into a CSV file with timestamps, defect types, and frequencies.

---

## ✨ Key Features

- High Accuracy: Achieves a 95% F1-score in real-world industrial scenarios.
- Multi-Modal Inference: Supports Uploaded Images, Uploaded Videos, Real-Time Image Capture, and Real-Time Video Streaming.
- Hardware Integration (IoT): Sends HTTP requests to an ESP8266 to trigger physical alarms/LEDs when defects are found.
- Automated Logging: Generates a CSV report for every run, logging timestamps, total defects, and specific defect categories.
- Cloud-Hosted Model: Automatically fetches the model weights from Hugging Face Hub, making the app highly portable.

---

## 🛠️ Tech Stack

- Deep Learning: YOLOv8 (Ultralytics)
- Computer Vision: OpenCV
- Web Interface: Streamlit
- Hardware Communication: HTTP Requests (ESP8266/NodeMCU)
- Data Handling: Pandas, NumPy
- Model Hosting: Hugging Face Hub

---

## 📂 Defect Classes

The model is trained to detect 5 primary types of fabric defects:

- Hole
- Stain
- Broken Needle
- Thread
- Yarn Short

---

## 🚀 Installation & Setup

### Clone the Repository

- git clone https://github.com/your-username/your-repo-name.gitcd your-repo-name
- 
### Install Dependencies
  
- Create a virtual environment and install the required libraries (requirements.txt)

### streamlit run app.py

---

## ⚙️ Hardware Integration (ESP8266)
### The application communicates with an ESP8266 module over local Wi-Fi to trigger physical alerts.
- Default ESP8266 IP: http://192.168.29.204 (Change this in app.py to match your microcontroller's IP).
- Endpoint: /defect_detected
- Behavior: When a defect is detected in a frame, the app sends an HTTP GET request to the ESP8266, which triggers an LED or buzzer. The app throttles requests to prevent network flooding.

---

## 📊 CSV Logging

- Every time an image, video, or live stream is processed, the app appends a detailed log to defect_logs.csv.

### The CSV contains the following columns:

- Timestamp: Date and time of the inference.
- File Type: Image, Video, Real-Time Image, or Real-Time Video.
- Total Defects: Sum of all detected defects.
- Hole, Stain, Broken Needle, Thread, Yarn Short: Individual counts per defect type.

---

## 🤖 Model Access
- The pre-trained YOLOv8 weights (best.pt) are hosted securely on Hugging Face. The application automatically downloads and caches the model using the huggingface_hub library upon first run.

- Hugging Face Repo: Umaar49/fabric-defect-yolov8

---

## 🖥️ Interface Guide

- Select Input Type: Choose between Image, Video, Real-Time Image, or Real-Time Video using the radio buttons.
- Upload/Capture: Upload a file from your device or use your webcam to capture live media.
- View Results: The app displays the annotated media with bounding boxes in real-time.
- Check Logs: Scroll down to view the auto-generated pandas DataFrame showing your exact defect counts and timestamps.
