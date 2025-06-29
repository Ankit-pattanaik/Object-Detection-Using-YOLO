import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import time
from collections import defaultdict
import plotly.express as px
from collections import Counter

# Streamlit page configuration
st.set_page_config(page_title="Object Detection Using YOLO", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme styling and larger video window
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 3px solid #ff4b4b;
    }
    .stApp {
        background: linear-gradient(to right, #2c2c2c, #1a1a1a);
    }
    .title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #ff4b4b, #4b79ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2em;
        color: #cccccc;
        text-align: center;
        margin-bottom: 20px;
    }
    .section {
        background-color: #2c2c2c;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4b79ff;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #e03a3a;
    }
    .footer {
        text-align: center;
        color: #999999;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 2px solid #4b79ff;
    }
    .stMarkdown, .stText, .stFileUploader, .stProgress, .stVideo {
        color: #ffffff;
    }
    .stSidebar .stMarkdown, .stSidebar .stText {
        color: #cccccc;
    }
    .status {
        color: #4b79ff;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .stImage>img {
        max-width: 1200px !important;
        width: 100%;
        height: auto;
        margin: auto;
    }
    .summary-box {
        background-color: #2c2c2c;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4b79ff;
        margin: 10px 0;
    }
    .summary-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
        color: #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title">Object Detection Using YOLO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a video to detect objects with our advanced YOLOv11 model</div>', unsafe_allow_html=True)

# Load YOLOv11 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Using official YOLOv8 nano model

model = load_model()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    frame_skip = st.slider("Frame Skip", 1, 10, 3, 1,
                          help="Process every nth frame for faster detection")
    st.divider()
    
    st.header("Instructions")
    st.markdown("""
    1. Upload a video file (MP4, AVI, or MOV)
    2. Adjust detection settings as needed
    3. Monitor real-time object detection
    4. View detailed detection summary
    """, unsafe_allow_html=True)

# Upload section
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"], key="video_uploader")
    
    # Show upload progress
    if uploaded_file is not None:
        st.markdown('<div class="status">Uploading video...</div>', unsafe_allow_html=True)
        upload_progress = st.progress(0)
        upload_percentage = st.empty()
        file_size = uploaded_file.size
        uploaded_size = 0
        chunk_size = file_size // 10 or 1
        while uploaded_size < file_size:
            uploaded_size = min(uploaded_size + chunk_size, file_size)
            progress = uploaded_size / file_size
            upload_progress.progress(min(progress, 1.0))
            upload_percentage.text(f"Upload Progress: {progress * 100:.1f}%")
            time.sleep(0.1)
        st.markdown('<div class="status">Upload complete!</div>', unsafe_allow_html=True)
        upload_percentage.text(f"Upload Progress: 100.0%")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Save video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    # Read video
    cap = cv2.VideoCapture(tfile.name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame display + progress
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Processing Video")
    st.markdown('<div class="status">Processing frames...</div>', unsafe_allow_html=True)
    stframe = st.empty()
    progress_bar = st.progress(0)
    progress_text = st.empty()
    processing_time = st.empty()

    # Detection collection with object tracking
    object_history = defaultdict(list)
    detected_objects_counter = Counter()
    frame_objects = defaultdict(set)
    last_id = 0
    max_distance = 50  # Max pixel distance to consider same object between frames

    start_time = time.time()
    frame_idx = 0
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue
            
        processed_frames += 1
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects with confidence threshold
        results = model(frame_rgb, conf=confidence_threshold)
        
        # Get current frame detections
        current_detections = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            current_detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'position': (x_center, y_center),
                'bbox': bbox
            })
        
        # Object tracking - match with previous detections
        used_ids = set()
        for detection in current_detections:
            min_distance = float('inf')
            matched_id = None
            pos = detection['position']
            
            # Find closest previous detection of same class
            for obj_id, history in object_history.items():
                if not history or history[-1]['class_name'] != detection['class_name']:
                    continue
                    
                last_pos = history[-1]['position']
                distance = ((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)**0.5
                
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    matched_id = obj_id
            
            # If match found, update history
            if matched_id is not None:
                detection['id'] = matched_id
                object_history[matched_id].append(detection)
                used_ids.add(matched_id)
            # Else create new object
            else:
                last_id += 1
                detection['id'] = last_id
                object_history[last_id] = [detection]
                used_ids.add(last_id)
        
        # Remove objects that disappeared
        disappeared_ids = set(object_history.keys()) - used_ids
        for obj_id in disappeared_ids:
            # Only count if object was tracked for at least 5 frames
            if len(object_history[obj_id]) >= 5:
                class_name = object_history[obj_id][0]['class_name']
                detected_objects_counter[class_name] += 1
            del object_history[obj_id]
        
        # Draw annotations
        annotated_frame = results[0].plot()
        
        # Add counters to frame
        counter_text = f"Objects: {sum(detected_objects_counter.values())} | Classes: {len(detected_objects_counter)}"
        cv2.putText(annotated_frame, counter_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show frame live
        stframe.image(annotated_frame, channels="RGB")

        # Progress
        progress = frame_idx / frame_count
        progress_bar.progress(min(progress, 1.0))
        elapsed = time.time() - start_time
        fps = processed_frames / elapsed if elapsed > 0 else 0
        progress_text.text(f"Processed: {frame_idx}/{frame_count} frames | FPS: {fps:.1f} | Objects: {sum(detected_objects_counter.values())}")

    # Release video capture
    cap.release()

    # Count remaining tracked objects
    for obj_id, history in object_history.items():
        if len(history) >= 5:  # Only count if tracked for sufficient frames
            class_name = history[0]['class_name']
            detected_objects_counter[class_name] += 1

    # Completion status
    total_time = time.time() - start_time
    st.markdown(f'<div class="status">Processing complete! Time: {total_time:.2f}s | FPS: {processed_frames/total_time:.1f}</div>', unsafe_allow_html=True)
    
    # Display detection summary
    st.subheader("Detection Summary")
    
    if detected_objects_counter:
        # Summary stats
        total_objects = sum(detected_objects_counter.values())
        unique_classes = len(detected_objects_counter)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Total Objects Detected:** {total_objects}")
        with col2:
            st.markdown(f"**Unique Classes:** {unique_classes}")
        
        # Sort by count descending
        sorted_counts = sorted(detected_objects_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Bar chart
        st.subheader("Object Distribution")
        classes, counts = zip(*sorted_counts)
        fig = px.bar(
            x=classes, 
            y=counts,
            labels={'x': 'Object Class', 'y': 'Count'},
            color=classes,
            text=counts
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed list
        st.subheader("Detailed Counts")
        for obj, count in sorted_counts:
            with st.container():
                st.markdown(f'<div class="summary-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-title">{obj}</div>', unsafe_allow_html=True)
                st.markdown(f"**Count:** {count}")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("No objects detected.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Clean temp file
    os.unlink(tfile.name)

# Footer
st.markdown('<div class="footer">Powered by YOLOv8 and Streamlit</div>', unsafe_allow_html=True)