import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import ultralytics.nn.tasks

# Allow safe loading of custom model classes
torch.serialization.add_safe_globals({"DetectionModel": ultralytics.nn.tasks.DetectionModel})



model = YOLO("yolov8n.pt")  

def process_image(image):
    image = np.array(image)
    results = model(image)
    detected_objects = {}
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_objects[label] = detected_objects.get(label, 0) + 1
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, f"{label}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    
    return image, detected_objects

# Streamlit Page Configuration
st.set_page_config(page_title="VisionCount AI", page_icon="🔢", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    body { background-color: #F3F4F6; }
    .title { font-size: 36px; font-weight: bold; text-align: center; color: #1E88E5; margin-top: 20px; }
    .subtitle { font-size: 20px; text-align: center; color: #555; margin-bottom: 20px; }
    .upload-section { border: 2px dashed #1E88E5; padding: 20px; text-align: center; border-radius: 10px; background-color: #ffffff; }
    .object-results { background: #ffffff; padding: 15px; border-radius: 10px; margin-top: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); }
    .progress-bar { height: 10px; background-color: #E0E0E0; border-radius: 5px; margin-bottom: 10px; }
    .progress-value { height: 100%; background-color: #1E88E5; border-radius: 5px; transition: width 0.3s ease; }
    .footer { text-align: center; font-size: 14px; color: #888; margin-top: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
st.sidebar.title("VisionCount AI")
st.sidebar.subheader("Object Detection & Counting")
st.sidebar.write("Upload an image to see AI-powered object counting in action!")

# Main Layout
st.markdown("<div class='title'>VisionCount AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Object Detection & Counting System</div>", unsafe_allow_html=True)

# Upload Section
st.markdown("<div class='upload-section'>📤 Drag and drop your image Below</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Original Image")
        st.image(image, use_column_width=True)
    
    with st.spinner("Processing Image..."):
        progress_bar = st.empty()
        status_text = st.empty()
        for i in range(101):
            progress_bar.markdown(f"<div class='progress-bar'><div class='progress-value' style='width: {i}%'></div></div>", unsafe_allow_html=True)
            time.sleep(0.02)
        status_text.empty()
        processed_image, counts = process_image(image)
    
    with col2:
        st.markdown("### 🔍 Processed Image")
        st.image(processed_image, use_column_width=True, caption="Detected Objects")
    
    # Results Section
    st.markdown("## 📊 Detection Results")
    st.markdown("<div class='object-results'>", unsafe_allow_html=True)
    for obj, count in counts.items():
        st.markdown(f"**{obj}:** {count}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Download Processed Image
    processed_pil = Image.fromarray(processed_image)
    processed_pil.save("processed_output.png")
    with open("processed_output.png", "rb") as file:
        st.download_button(label="⬇️ Download Processed Image", data=file, file_name="visioncount_result.png", mime="image/png")

st.markdown("<div class='footer'>&copy; 2025 VisionCount AI</div>", unsafe_allow_html=True)
