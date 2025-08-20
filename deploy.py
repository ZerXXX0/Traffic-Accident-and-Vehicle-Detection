"""
Streamlit app — Multi-model YOLO + RCNN Video/Audio Inference
- upload a video
- sample frames at chosen FPS (timestamps recorded)
- extract short audio clips around each timestamp (keeps audio inside the video)
- run 3 YOLO models on frames and draw bounding boxes
- run 2 RCNN models on audio clips
- combine detections into a single timeline table and timeline chart
- preview frames with bounding boxes and play audio clips

Notes:
- Replace placeholder model-loading lines with your actual model paths/logic.
- This file is meant to be run with `streamlit run this_file.py` (or in Jupyter via streamlit magic if supported).
"""

import streamlit as st
import cv2
import tempfile
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from moviepy import VideoFileClip
from datetime import timedelta
from pathlib import Path
import base64
import io
import altair as alt

# -------------------------
# Helpers
# -------------------------

def format_ts(sec):
    return str(timedelta(seconds=round(sec, 2)))


def draw_boxes_on_frame(frame, detections, names=None):
    """detections: list of dicts {'xyxy': [x1,y1,x2,y2], 'conf': float, 'cls': int, 'label':str}
    Returns annotated BGR image with Apple-inspired styling.
    """
    img = frame.copy()
    
    # Apple-inspired color palette for different detection types
    color_map = {
        'ambulance': (255, 59, 48),      # Apple Red (RGB)
        'firetruck': (255, 149, 0),     # Apple Orange
        'smoke': (142, 142, 147),       # Apple Gray
        'fire': (255, 45, 85),          # Apple Pink
        'crash': (255, 204, 0),         # Apple Yellow
        'siren': (52, 199, 89),         # Apple Green
        'crash_sound': (0, 122, 255),   # Apple Blue
        'default': (88, 86, 214)        # Apple Purple
    }
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['xyxy'])
        conf = det.get('conf', 0)
        label = det.get('label', str(det.get('cls', '')))
        
        # Get color for this detection type (convert RGB to BGR for OpenCV)
        rgb_color = color_map.get(label.lower(), color_map['default'])
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        
        # Draw bounding box with thicker lines
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 3)
        
        # Create label background
        label_text = f"{label} {conf:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw filled rectangle for label background with some padding
        padding = 8
        label_bg_top_left = (x1, max(0, y1 - text_height - padding * 2))
        label_bg_bottom_right = (x1 + text_width + padding * 2, y1)
        
        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, label_bg_top_left, label_bg_bottom_right, bgr_color, -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Draw white text on colored background
        text_position = (x1 + padding, y1 - padding)
        cv2.putText(img, label_text, text_position, font, font_scale, (255, 255, 255), thickness)
        
        # Add corner indicators for better visibility
        corner_size = 20
        corner_thickness = 4
        
        # Top-left corner
        cv2.line(img, (x1, y1), (x1 + corner_size, y1), bgr_color, corner_thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_size), bgr_color, corner_thickness)
        
        # Top-right corner
        cv2.line(img, (x2, y1), (x2 - corner_size, y1), bgr_color, corner_thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_size), bgr_color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(img, (x1, y2), (x1 + corner_size, y2), bgr_color, corner_thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_size), bgr_color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2 - corner_size, y2), bgr_color, corner_thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_size), bgr_color, corner_thickness)
    
    return img


def array_to_audio_bytes(waveform, sr):
    # waveform: 1D numpy float32 in [-1,1]
    import soundfile as sf
    bio = io.BytesIO()
    sf.write(bio, waveform, sr, format='WAV')
    bio.seek(0)
    return bio.read()

# -------------------------
# Model loading (restored original functionality)
# -------------------------

@st.cache_resource
def load_yolo_models():
    """Load YOLOv12 models for vehicle and object detection"""
    models = []
    try:
        from ultralytics import YOLO
        models.append(YOLO('./model/ambulance_firetruck_best.pt'))
        models.append(YOLO('./model/smoke_and_fire_best.pt'))
        models.append(YOLO('./model/crash_best.pt'))
    except Exception:
        # fallback: dummy None list (user must replace)
        models = [None, None, None]
    return models

@st.cache_resource
def load_rcnn_models():
    """Load RCNN models for audio detection"""
    models = []
    try:
        m1 = torch.load('./model/crash_sound_best.pt', map_location='cpu')
        m2 = torch.load('./model/rcnn_siren_best.pt', map_location='cpu')
        m1.eval(); m2.eval()
        models = [m1, m2]
    except Exception:
        models = [None, None]
    return models

# -------------------------
# Extraction utilities
# -------------------------

def extract_frames_with_timestamps(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(orig_fps / target_fps)))
    frames = []
    timestamps = []
    idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts = idx / orig_fps
            frames.append(frame.copy())
            timestamps.append(ts)
            saved_idx += 1
        idx += 1
    cap.release()
    return frames, timestamps


def extract_audio_waveform_from_video(video_path, sr=16000):
    # Return mono waveform (numpy float32 in [-1,1]) and sample rate
    clip = VideoFileClip(video_path)
    audio = clip.audio
    arr = audio.to_soundarray(fps=sr)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr.astype('float32'), sr

# -------------------------
# Inference wrappers
# -------------------------

def run_yolo_on_frames(frames, models, conf_thres=0.25):
    all_results = [[ ] for _ in models]
    for i, model in enumerate(models):
        if model is None:
            continue
        try:
            # ultralytics YOLO supports passing numpy arrays one by one
            for frame in frames:
                res = model(frame)
                # parse res[0].boxes if ultralytics
                boxes = []
                try:
                    r0 = res[0]
                    if hasattr(r0, 'boxes'):
                        for b in r0.boxes.data.tolist():
                            x1,y1,x2,y2,score,cls = b
                            if score >= conf_thres:
                                boxes.append({'xyxy':[x1,y1,x2,y2],'conf':score,'cls':int(cls),'label':r0.names[int(cls)] if hasattr(r0,'names') else str(int(cls))})
                except Exception:
                    pass
                all_results[i].append(boxes)
        except Exception:
            # fallback for other model APIs (e.g., yolov5 hub models)
            try:
                for frame in frames:
                    results = model(frame)
                    dets = []
                    if hasattr(results, 'xyxy'):
                        for row in results.xyxy[0].cpu().numpy():
                            x1,y1,x2,y2,conf,cls = row
                            if conf >= conf_thres:
                                label = results.names[int(cls)] if hasattr(results, 'names') else str(int(cls))
                                dets.append({'xyxy':[x1,y1,x2,y2],'conf':float(conf),'cls':int(cls),'label':label})
                    all_results[i].append(dets)
            except Exception:
                # if model can't run, fill with empty lists
                all_results[i] = [[] for _ in frames]
    return all_results


def run_rcnn_on_audio_timestamps(waveform, sr, timestamps, clip_length_sec, models):
    results = [[ ] for _ in models]
    total_len = len(waveform)
    for idx, ts in enumerate(timestamps):
        start_sample = int(max(0, (ts - clip_length_sec/2) * sr))
        end_sample = int(min(total_len, start_sample + clip_length_sec * sr))
        segment = waveform[start_sample:end_sample]
        # normalize if needed
        if segment.size == 0:
            for j in range(len(models)):
                results[j].append(None)
            continue
        seg_t = torch.tensor(segment).float().unsqueeze(0).unsqueeze(0)  # shape [1,1,N]
        for j, m in enumerate(models):
            if m is None:
                results[j].append(None)
                continue
            with torch.no_grad():
                try:
                    out = m(seg_t)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                    results[j].append({'pred': pred, 'probs': probs.tolist()})
                except Exception:
                    # try other output shapes (binary)
                    try:
                        out = m(seg_t).cpu().numpy()
                        if out.size == 1:
                            prob = float(out)
                            pred = int(prob > 0.5)
                            results[j].append({'pred': pred, 'probs':[1-prob, prob]})
                        else:
                            pred = int(np.argmax(out))
                            results[j].append({'pred': pred, 'probs': out.tolist()})
                    except Exception:
                        results[j].append(None)
    return results

# -------------------------
# Streamlit UI with Apple Design System
# -------------------------

# Configure page with Apple-inspired styling
st.set_page_config(
    page_title="SOAR",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apple-inspired CSS styling with complete design system
st.markdown("""
<style>
    /* Import SF Pro Display font family */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Apple Design System Color Variables */
    :root {
        --apple-blue: #007AFF;
        --apple-green: #34C759;
        --apple-indigo: #5856D6;
        --apple-orange: #FF9500;
        --apple-pink: #FF2D92;
        --apple-purple: #AF52DE;
        --apple-red: #FF3B30;
        --apple-teal: #5AC8FA;
        --apple-yellow: #FFCC00;
        
        /* Grays */
        --apple-gray: #8E8E93;
        --apple-gray2: #AEAEB2;
        --apple-gray3: #C7C7CC;
        --apple-gray4: #D1D1D6;
        --apple-gray5: #E5E5EA;
        --apple-gray6: #F2F2F7;
        
        /* Background Colors */
        --apple-bg-primary: #FFFFFF;
        --apple-bg-secondary: #F2F2F7;
        --apple-bg-tertiary: #FFFFFF;
        
        /* Text Colors */
        --apple-text-primary: #000000;
        --apple-text-secondary: #3C3C43;
        --apple-text-tertiary: #3C3C4399;
    }
    
    /* Global styling with Apple background */
    .stApp {
        background: linear-gradient(135deg, var(--apple-bg-secondary) 0%, var(--apple-gray6) 50%, var(--apple-bg-secondary) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        color: var(--apple-text-primary);
    }
    
    /* Main title styling with Apple gradient */
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--apple-blue) 0%, var(--apple-purple) 50%, var(--apple-pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.375rem;
        color: var(--apple-text-secondary);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Card styling with Apple glassmorphism */
    .upload-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06), 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
    }
    
    /* Button styling with Apple colors */
    .stButton > button {
        background: linear-gradient(135deg, var(--apple-blue) 0%, var(--apple-indigo) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 20px rgba(0, 122, 255, 0.25);
        letter-spacing: -0.01em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 30px rgba(0, 122, 255, 0.35);
        background: linear-gradient(135deg, var(--apple-indigo) 0%, var(--apple-purple) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(1);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, rgba(0, 122, 255, 0.05) 0%, rgba(88, 86, 214, 0.05) 100%);
        border: 2px dashed var(--apple-blue);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        background: linear-gradient(135deg, rgba(0, 122, 255, 0.08) 0%, rgba(88, 86, 214, 0.08) 100%);
        border-color: var(--apple-indigo);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        background: var(--apple-bg-tertiary);
        border: 1px solid var(--apple-gray4);
        border-radius: 10px;
        padding: 0.875rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--apple-blue);
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--apple-text-primary);
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--apple-gray5);
        letter-spacing: -0.02em;
    }
    
    /* Timeline chart container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
        margin: 1.5rem 0;
        border: 1px solid var(--apple-gray5);
    }
    
    /* Data frame styling */
    .stDataFrame {
        background: var(--apple-bg-tertiary);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 16px rgba(0, 0, 0, 0.04);
        border: 1px solid var(--apple-gray5);
    }
    
    /* Success/info messages with Apple colors */
    .stSuccess {
        background: linear-gradient(135deg, rgba(52, 199, 89, 0.1) 0%, rgba(52, 199, 89, 0.05) 100%);
        border-radius: 12px;
        border-left: 4px solid var(--apple-green);
        color: var(--apple-text-primary);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 122, 255, 0.1) 0%, rgba(0, 122, 255, 0.05) 100%);
        border-radius: 12px;
        border-left: 4px solid var(--apple-blue);
        color: var(--apple-text-primary);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.1) 0%, rgba(255, 149, 0, 0.05) 100%);
        border-radius: 12px;
        border-left: 4px solid var(--apple-orange);
        color: var(--apple-text-primary);
    }
    
    /* Spinner with Apple colors */
    .stSpinner > div {
        border-color: var(--apple-blue) transparent var(--apple-blue) transparent;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 12px;
        background: var(--apple-bg-tertiary);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    /* Image styling with consistent sizing */
    .stImage > img {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        width: 100%;
        height: 280px;
        object-fit: cover;
        border: 1px solid var(--apple-gray5);
    }
    
    /* Pagination buttons */
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }
    
    /* Frame grid styling */
    .frame-container {
        background: var(--apple-bg-tertiary);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid var(--apple-gray5);
        transition: all 0.3s ease;
    }
    
    .frame-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    }
    
    /* Metrics styling */
    .stMetric {
        background: var(--apple-bg-tertiary);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        border: 1px solid var(--apple-gray5);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--apple-gray6);
        border-radius: 8px;
        font-weight: 600;
        color: var(--apple-text-primary);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: var(--apple-blue);
    }
</style>
""", unsafe_allow_html=True)

# Main title and subtitle
st.markdown('<h1 class="main-title">SOAR</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">System of a Road<br>Advanced AI-powered video and audio analysis platform</p>', unsafe_allow_html=True)

# Upload section with Apple-inspired card design
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded = st.file_uploader("📁 Upload your video file", type=['mp4','mov','avi','mkv'], help="Supported formats: MP4, MOV, AVI, MKV")

col1, col2 = st.columns(2)
with col1:
    target_fps = st.number_input("🎥 Frame sampling rate (FPS)", min_value=1, max_value=30, value=2, help="Higher values provide more temporal resolution")
with col2:
    audio_clip_len = st.number_input("🔊 Audio clip duration (seconds)", min_value=1, max_value=10, value=2, help="Length of audio segments for analysis")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is not None:
    tmp = Path(tempfile.gettempdir()) / uploaded.name
    with open(tmp, 'wb') as f:
        f.write(uploaded.read())

    # Video preview with styling
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📺 Video Preview</h3>', unsafe_allow_html=True)
    st.video(str(tmp))
    st.markdown('</div>', unsafe_allow_html=True)

    # Processing section
    with st.spinner('🔄 Extracting frames and audio...'):
        frames, frame_timestamps = extract_frames_with_timestamps(str(tmp), target_fps)
        waveform, sr = extract_audio_waveform_from_video(str(tmp), sr=16000)

    st.success(f"✅ Successfully extracted {len(frames)} frames and audio (sample rate: {sr} Hz)")

    # Load models
    yolo_models = load_yolo_models()
    rcnn_models = load_rcnn_models()

    # Run inference
    with st.spinner('🤖 Running AI models on video frames...'):
        yolo_all = run_yolo_on_frames(frames, yolo_models)

    with st.spinner('🎵 Analyzing audio segments...'):
        rcnn_all = run_rcnn_on_audio_timestamps(waveform, sr, frame_timestamps, audio_clip_len, rcnn_models)

    # Build a combined DataFrame timeline
    rows = []
    # Vision detections
    for mi, model_results in enumerate(yolo_all):
        for fi, dets in enumerate(model_results):
            ts = frame_timestamps[fi]
            if dets:
                for d in dets:
                    rows.append({
                        'timestamp': ts,
                        'time_str': format_ts(ts),
                        'source': f'YOLO_{mi+1}',
                        'type': 'vision',
                        'label': d.get('label', ''),
                        'confidence': d.get('conf', 0.0)
                    })
    # Audio detections
    for mi, model_results in enumerate(rcnn_all):
        for fi, res in enumerate(model_results):
            ts = frame_timestamps[fi]
            if res is not None:
                rows.append({
                    'timestamp': ts,
                    'time_str': format_ts(ts),
                    'source': f'RCNN_{mi+1}',
                    'type': 'audio',
                    'label': str(res.get('pred', '')),
                    'confidence': max(res.get('probs', [0])) if res.get('probs') is not None else 0.0
                })

    if not rows:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.info('🔍 No detections found from any model. Try adjusting the sampling parameters or check if models are loaded correctly.')
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <p style="color: #6b7280; font-size: 1.1rem;">SOAR analyzes your video for:</p>
            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">&#128657;</div>
                    <p style="color: #374151; font-weight: 500;">Emergency Vehicles</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">&#128293;</div>
                    <p style="color: #374151; font-weight: 500;">Fire & Smoke</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">&#128165;</div>
                    <p style="color: #374151; font-weight: 500;">Crash Detection</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">&#128680;</div>
                    <p style="color: #374151; font-weight: 500;">Siren Sounds</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values('timestamp')

        # Statistics section
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📈 Analysis Summary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_detections = len(df_sorted)
            st.metric("Total Detections", total_detections, delta=f"+{total_detections}")
        
        with col2:
            vision_detections = len(df_sorted[df_sorted['type'] == 'vision'])
            st.metric("Vision Events", vision_detections)
            
        with col3:
            audio_detections = len(df_sorted[df_sorted['type'] == 'audio'])
            st.metric("Audio Events", audio_detections)
            
        with col4:
            avg_confidence = df_sorted['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Timeline table with styling
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📊 Detection Timeline</h3>', unsafe_allow_html=True)
        st.dataframe(df_sorted.reset_index(drop=True), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Timeline chart with styling
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📈 Visual Timeline</h3>', unsafe_allow_html=True)
        base = alt.Chart(df_sorted).encode(x=alt.X('timestamp:Q', title='Time (seconds)'))
        points = base.mark_point(filled=True, size=120).encode(
            y=alt.Y('source:N', title='AI Model'),
            color=alt.Color('type:N', scale=alt.Scale(range=['#007AFF', '#FF3B30'])),  # Apple Blue and Red
            stroke=alt.value('white'),
            strokeWidth=alt.value(2),
            tooltip=['time_str', 'source', 'type', 'label', 'confidence']
        )
        st.altair_chart(points, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Preview frames section with styling and pagination
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🖼️ Annotated Frame Previews</h3>', unsafe_allow_html=True)
        
        total_frames = len(frame_timestamps)
        frames_per_page = 4
        total_pages = (total_frames + frames_per_page - 1) // frames_per_page  # Ceiling division
        
        # Initialize session state for pagination
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        
        # Page navigation controls
        if total_pages > 1:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️ First", disabled=(st.session_state.current_page == 0)):
                    st.session_state.current_page = 0
                    st.rerun()
            
            with col2:
                if st.button("◀️ Previous", disabled=(st.session_state.current_page == 0)):
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with col3:
                st.markdown(f"<div style='text-align: center; padding: 0.5rem; font-weight: 600;'>Page {st.session_state.current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            
            with col4:
                if st.button("Next ▶️", disabled=(st.session_state.current_page >= total_pages - 1)):
                    st.session_state.current_page += 1
                    st.rerun()
            
            with col5:
                if st.button("Last ⏭️", disabled=(st.session_state.current_page >= total_pages - 1)):
                    st.session_state.current_page = total_pages - 1
                    st.rerun()
        
        # Calculate frame indices for current page
        start_idx = st.session_state.current_page * frames_per_page
        end_idx = min(start_idx + frames_per_page, total_frames)
        
        # Display frames for current page
        for i in range(start_idx, end_idx, 2):  # Process 2 frames at a time for 2x2 grid
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                frame_idx = i + j
                if frame_idx < end_idx:
                    ts = frame_timestamps[frame_idx]
                    
                    # Collect all detections for this frame across models
                    dets_for_frame = []
                    for mi in range(len(yolo_all)):
                        if frame_idx < len(yolo_all[mi]):
                            dets_for_frame.extend(yolo_all[mi][frame_idx])
                    
                    # Always show frame, even without detections
                    if dets_for_frame:
                        ann = draw_boxes_on_frame(frames[frame_idx], dets_for_frame)
                        frame_to_show = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                        caption = f"🎯 Frame {frame_idx + 1} @ {format_ts(ts)} ({len(dets_for_frame)} detections)"
                    else:
                        frame_to_show = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
                        caption = f"� Frame {frame_idx + 1} @ {format_ts(ts)} (no detections)"
                    
                    with col:
                        st.markdown('<div class="frame-container">', unsafe_allow_html=True)
                        st.image(frame_to_show, caption=caption, use_column_width=True)
                        
                        # Audio section for each frame
                        with st.expander(f"🔊 Audio @ {format_ts(ts)}"):
                            start_s = max(0, ts - audio_clip_len/2)
                            end_s = min((len(waveform)/sr), start_s + audio_clip_len)
                            seg = waveform[int(start_s*sr):int(end_s*sr)]
                            
                            if len(seg) > 0:
                                audio_bytes = array_to_audio_bytes(seg, sr)
                                st.audio(audio_bytes)
                                
                                # Show audio detection results for this timestamp
                                audio_results = []
                                for mi, model_results in enumerate(rcnn_all):
                                    if frame_idx < len(model_results) and model_results[frame_idx] is not None:
                                        res = model_results[frame_idx]
                                        if res.get('pred', 0) == 1:
                                            model_name = 'Crash Sound' if mi == 0 else 'Siren Detection'
                                            confidence = max(res.get('probs', [0]))
                                            audio_results.append(f"✅ {model_name}: {confidence:.1%}")
                                
                                if audio_results:
                                    for result in audio_results:
                                        st.success(result)
                                else:
                                    st.info("🔇 No audio events detected")
                            else:
                                st.warning("⚠️ No audio data available")
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary at bottom with Apple styling
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0, 122, 255, 0.08) 0%, rgba(88, 86, 214, 0.08) 100%); border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; text-align: center; border: 1px solid var(--apple-gray4);">
            <p style="margin: 0; color: var(--apple-blue); font-weight: 700; font-size: 1.1rem;">&#128202; Frame Analysis Summary</p>
            <p style="margin: 1rem 0 0 0; color: var(--apple-text-secondary); font-size: 1rem;">
                Showing frames {start_idx + 1}-{end_idx} of {total_frames} total frames &#8226; 
                Page {st.session_state.current_page + 1} of {total_pages}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Download section with styling
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
        csv = df_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Timeline Data (CSV)",
            data=csv,
            file_name=f"soar_analysis_{uploaded.name}.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer with tips and information
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">💡 Tips & Information</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **&#127919; Optimization**
        - Higher frame sampling rates provide better temporal resolution
        - Increase audio clip length for longer events
        - Use GPU-enabled systems for faster processing
        """)
    
    with col2:
        st.markdown("""
        **&#128249; Video Tips**
        - Supported formats: MP4, MOV, AVI, MKV
        - Higher quality videos yield better results
        - Stable footage improves detection accuracy
        """)
    
    with col3:
        st.markdown("""
        **&#128266; Audio Analysis**
        - Clear audio improves siren detection
        - Background noise may affect results
        - Optimal clip length: 2-4 seconds
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with Apple styling
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding: 3rem; background: linear-gradient(135deg, rgba(242, 242, 247, 0.8) 0%, rgba(255, 255, 255, 0.9) 100%); border-radius: 20px; backdrop-filter: blur(20px);">
    <p style="margin: 0; font-size: 1.25rem; font-weight: 700; color: var(--apple-text-primary);">SOAR</p>
    <p style="margin: 0.5rem 0; font-size: 1rem; color: var(--apple-text-secondary);">System of a Road</p>
    <p style="margin: 1rem 0 0 0; font-size: 0.875rem; color: var(--apple-text-tertiary);">Powered by AI &#8226; Built with &#10084;&#65039; &#8226</p>
</div>
""", unsafe_allow_html=True)