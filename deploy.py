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
import streamlit.components.v1 as components

# -------------------------
# Helpers
# -------------------------

def format_ts(sec):
    return str(timedelta(seconds=round(sec, 2)))


def draw_boxes_on_frame(frame, detections, names=None):
    """detections: list of dicts {'xyxy': [x1,y1,x2,y2], 'conf': float, 'cls': int, 'label':str}
    Returns annotated BGR image.
    """
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['xyxy'])
        conf = det.get('conf', 0)
        label = det.get('label', str(det.get('cls', '')))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img


def array_to_audio_bytes(waveform, sr):
    # waveform: 1D numpy float32 in [-1,1]
    import soundfile as sf
    bio = io.BytesIO()
    sf.write(bio, waveform, sr, format='WAV')
    bio.seek(0)
    return bio.read()

# -------------------------
# Model loading (placeholders - replace with your code)
# -------------------------

@st.cache_resource
def load_yolo_models():
    """Load YOLO models with proper error handling and debugging"""
    models = []
    model_paths = [
        './model/ambulance_firetruck_best.pt',  # Fixed path
        './model/smoke_and_fire_best.pt',
        './model/crash_best.pt'
    ]
    
    for i, path in enumerate(model_paths):
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            models.append(model)
            st.write(f"✅ Model {i+1} loaded successfully: {path}")
        except FileNotFoundError:
            st.error(f"❌ Model file not found: {path}")
            models.append(None)
        except Exception as e:
            st.error(f"❌ Error loading model {i+1}: {str(e)}")
            models.append(None)
    
    return models

@st.cache_resource
def load_rcnn_models():
    """Load RCNN models with proper error handling and debugging"""
    models = []
    model_paths = [
        './model/crash_sound_best.pt',
        './model/rcnn_siren_best.pt'
    ]
    
    for i, path in enumerate(model_paths):
        try:
            model = torch.load(path, map_location='cpu')
            model.eval()
            models.append(model)
            st.write(f"✅ Audio Model {i+1} loaded successfully: {path}")
        except FileNotFoundError:
            st.error(f"❌ Audio model file not found: {path}")
            models.append(None)
        except Exception as e:
            st.error(f"❌ Error loading audio model {i+1}: {str(e)}")
            models.append(None)
    
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
    """Enhanced YOLO inference with debugging and lower confidence threshold"""
    all_results = [[] for _ in models]
    
    for i, model in enumerate(models):
        if model is None:
            st.warning(f"⚠️ Model {i+1} is None, skipping...")
            all_results[i] = [[] for _ in frames]
            continue
            
        try:
            st.write(f"🔍 Running Model {i+1} on {len(frames)} frames...")
            frame_count = 0
            
            for frame_idx, frame in enumerate(frames):
                # Ensure frame is in correct format (BGR for YOLO)
                if frame is None:
                    all_results[i].append([])
                    continue
                    
                # Run inference with lower confidence for debugging
                res = model(frame, conf=max(0.1, conf_thres-0.1), verbose=False)
                boxes = []
                
                try:
                    r0 = res[0]
                    if hasattr(r0, 'boxes') and r0.boxes is not None:
                        for b in r0.boxes.data.tolist():
                            if len(b) >= 6:  # Ensure we have all required values
                                x1, y1, x2, y2, score, cls = b[:6]
                                if score >= conf_thres:
                                    label = r0.names[int(cls)] if hasattr(r0, 'names') and int(cls) in r0.names else f"class_{int(cls)}"
                                    boxes.append({
                                        'xyxy': [x1, y1, x2, y2],
                                        'conf': score,
                                        'cls': int(cls),
                                        'label': label
                                    })
                                    frame_count += 1
                except Exception as e:
                    st.error(f"❌ Error parsing results for Model {i+1}, Frame {frame_idx}: {str(e)}")
                
                all_results[i].append(boxes)
            
            st.write(f"✅ Model {i+1}: Found {frame_count} detections across {len(frames)} frames")
            
        except Exception as e:
            st.error(f"❌ Critical error in Model {i+1}: {str(e)}")
            all_results[i] = [[] for _ in frames]
    
    return all_results


def run_rcnn_on_audio_timestamps(waveform, sr, timestamps, clip_length_sec, models):
    """Enhanced RCNN inference with debugging"""
    results = [[] for _ in models]
    total_len = len(waveform)
    
    st.write(f"🔊 Processing audio: {len(timestamps)} timestamps, {total_len} samples at {sr}Hz")
    
    for idx, ts in enumerate(timestamps):
        start_sample = int(max(0, (ts - clip_length_sec/2) * sr))
        end_sample = int(min(total_len, start_sample + clip_length_sec * sr))
        segment = waveform[start_sample:end_sample]
        
        # Check segment validity
        if segment.size == 0:
            st.warning(f"⚠️ Empty audio segment at timestamp {ts}")
            for j in range(len(models)):
                results[j].append(None)
            continue
            
        # Normalize audio segment
        if np.max(np.abs(segment)) > 0:
            segment = segment / np.max(np.abs(segment))
        
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
                    
                    # Handle different output formats
                    if hasattr(out, 'cpu'):
                        out = out.cpu()
                    
                    if out.dim() > 1 and out.shape[1] > 1:
                        # Multi-class classification
                        probs = torch.softmax(out, dim=1).numpy()[0]
                        pred = int(np.argmax(probs))
                        max_prob = float(np.max(probs))
                        results[j].append({'pred': pred, 'probs': probs.tolist(), 'confidence': max_prob})
                    else:
                        # Binary classification
                        prob = float(torch.sigmoid(out).numpy())
                        pred = int(prob > 0.5)
                        results[j].append({'pred': pred, 'probs': [1-prob, prob], 'confidence': prob})
                        
                except Exception as e:
                    st.error(f"❌ Audio Model {j+1} error at timestamp {ts}: {str(e)}")
                    results[j].append(None)
    
    # Summary
    for j, model_results in enumerate(results):
        valid_results = [r for r in model_results if r is not None]
        positive_detections = [r for r in valid_results if r.get('pred', 0) > 0]
        st.write(f"🎵 Audio Model {j+1}: {len(positive_detections)} positive detections out of {len(valid_results)} valid results")
    
    return results

# -------------------------
# Streamlit UI (Apple Reference Design)
# -------------------------

st.set_page_config(page_title="Video Analysis", layout="wide", initial_sidebar_state="auto")

# Apple Design System CSS
apple_css = """
<style>
    /* Apple Design Language */
    :root {
        --apple-bg: #ffffff;
        --apple-bg-secondary: #f5f5f7;
        --apple-bg-tertiary: #fbfbfd;
        --apple-text: #1d1d1f;
        --apple-text-secondary: #86868b;
        --apple-accent: #007aff;
        --apple-accent-hover: #0056cc;
        --apple-success: #30d158;
        --apple-warning: #ff9500;
        --apple-error: #ff3b30;
        --apple-border: #d2d2d7;
        --apple-separator: #f2f2f7;
        --apple-radius: 12px;
        --apple-radius-small: 8px;
        --apple-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        --apple-shadow-hover: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    
    /* SF Pro Display font stack */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
        background: var(--apple-bg-secondary);
        color: var(--apple-text);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .stApp > .main {
        background: var(--apple-bg-secondary);
        padding: 24px 32px;
    }
    
    /* Apple Card Component */
    .apple-card {
        background: var(--apple-bg);
        border-radius: var(--apple-radius);
        box-shadow: var(--apple-shadow);
        border: 1px solid var(--apple-separator);
        padding: 24px;
        margin-bottom: 20px;
        transition: box-shadow 0.3s ease;
    }
    
    .apple-card:hover {
        box-shadow: var(--apple-shadow-hover);
    }
    
    /* Apple Header */
    .apple-header {
        background: var(--apple-bg);
        border-radius: var(--apple-radius);
        box-shadow: var(--apple-shadow);
        border: 1px solid var(--apple-separator);
        padding: 32px;
        margin-bottom: 32px;
        text-align: center;
    }
    
    .apple-title {
        font-size: 48px;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--apple-text);
        margin: 0 0 8px 0;
        line-height: 1.1;
    }
    
    .apple-subtitle {
        font-size: 21px;
        font-weight: 400;
        color: var(--apple-text-secondary);
        margin: 0;
        line-height: 1.4;
    }
    
    /* Apple Metrics */
    .apple-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 24px 0;
    }
    
    .apple-metric {
        background: var(--apple-bg-tertiary);
        border-radius: var(--apple-radius-small);
        padding: 20px;
        text-align: center;
        border: 1px solid var(--apple-separator);
    }
    
    .apple-metric-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--apple-accent);
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
    }
    
    .apple-metric-label {
        font-size: 13px;
        font-weight: 500;
        color: var(--apple-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    /* Apple Section Headers */
    .apple-section-header {
        font-size: 28px;
        font-weight: 600;
        color: var(--apple-text);
        margin: 0 0 16px 0;
        letter-spacing: -0.02em;
    }
    
    .apple-section-subheader {
        font-size: 17px;
        font-weight: 400;
        color: var(--apple-text-secondary);
        margin: 0 0 24px 0;
        line-height: 1.5;
    }
    
    /* Apple Status */
    .apple-status {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
        margin: 4px 8px 4px 0;
    }
    
    .apple-status-success {
        background: rgba(48, 209, 88, 0.1);
        color: var(--apple-success);
    }
    
    .apple-status-warning {
        background: rgba(255, 149, 0, 0.1);
        color: var(--apple-warning);
    }
    
    .apple-status-error {
        background: rgba(255, 59, 48, 0.1);
        color: var(--apple-error);
    }
    
    /* Apple Navigation */
    .apple-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 24px 0;
    }
    
    .apple-nav-info {
        font-size: 15px;
        font-weight: 500;
        color: var(--apple-text-secondary);
    }
    
    /* Override Streamlit Styles */
    .stSidebar {
        background: var(--apple-bg);
        border-right: 1px solid var(--apple-separator);
    }
    
    .stSidebar .stMarkdown h1,
    .stSidebar .stMarkdown h2,
    .stSidebar .stMarkdown h3 {
        color: var(--apple-text);
        font-weight: 600;
    }
    
    .stButton > button {
        background: var(--apple-accent);
        color: white;
        border: none;
        border-radius: var(--apple-radius-small);
        font-weight: 500;
        font-size: 15px;
        padding: 12px 24px;
        transition: all 0.2s ease;
        box-shadow: none;
    }
    
    .stButton > button:hover {
        background: var(--apple-accent-hover);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }
    
    .stSelectbox label, 
    .stNumberInput label, 
    .stSlider label,
    .stFileUploader label {
        color: var(--apple-text) !important;
        font-weight: 500 !important;
        font-size: 15px !important;
    }
    
    .stDataFrame {
        border-radius: var(--apple-radius-small);
        overflow: hidden;
        box-shadow: var(--apple-shadow);
        border: 1px solid var(--apple-separator);
    }
    
    /* Apple Image Container */
    .apple-image-container {
        border-radius: var(--apple-radius-small);
        overflow: hidden;
        background: var(--apple-bg-tertiary);
        border: 1px solid var(--apple-separator);
        margin-bottom: 16px;
    }
    
    .apple-image-caption {
        padding: 12px 16px;
        background: var(--apple-bg-tertiary);
        border-top: 1px solid var(--apple-separator);
        font-size: 13px;
        color: var(--apple-text-secondary);
        text-align: center;
    }
</style>
"""
components.html(apple_css, height=0)

# Apple-style Header
components.html(
    """
    <div class='apple-header'>
        <h1 class='apple-title'>Video Analysis</h1>
        <p class='apple-subtitle'>Advanced multi-modal detection and analysis platform</p>
    </div>
    """,
    height=180,
)

# Apple-style Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    uploaded = st.file_uploader("Choose a video file", type=['mp4','mov','avi','mkv'], help="Upload your video for analysis")
    
    st.markdown("### Analysis Parameters")
    target_fps = st.number_input("Sample rate (FPS)", min_value=1, max_value=30, value=2, help="Frames per second to analyze")
    audio_clip_len = st.number_input("Audio window (seconds)", min_value=1, max_value=10, value=2, help="Audio clip duration around each frame")
    conf_thres = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01, help="Minimum confidence for detections")
    
    st.markdown("---")
    analyze_button = st.button("Start Analysis", type="primary")

# Initialize session state for navigation
if 'apple_page' not in st.session_state:
    st.session_state['apple_page'] = 0

# Main content area
main_content = st.container()
with main_content:
    if uploaded is None:
        st.markdown("""
        <div class='apple-card'>
            <h2 class='apple-section-header'>Welcome to Video Analysis</h2>
            <p class='apple-section-subheader'>
                Upload a video file to begin analysis. Our system will detect objects, analyze audio patterns, 
                and provide comprehensive insights with frame-by-frame annotations.
            </p>
            <div class='apple-metrics'>
                <div class='apple-metric'>
                    <div class='apple-metric-value'>3</div>
                    <div class='apple-metric-label'>Vision Models</div>
                </div>
                <div class='apple-metric'>
                    <div class='apple-metric-value'>2</div>
                    <div class='apple-metric-label'>Audio Models</div>
                </div>
                <div class='apple-metric'>
                    <div class='apple-metric-value'>5</div>
                    <div class='apple-metric-label'>Total Algorithms</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Process uploaded file
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        with open(tmp, 'wb') as f:
            f.write(uploaded.read())

        # Video preview
        st.markdown("""
        <div class='apple-card'>
            <h2 class='apple-section-header'>Video Preview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.video(str(tmp))

        if analyze_button:
            # Show model loading status first
            st.markdown("""
            <div class='apple-card'>
                <h2 class='apple-section-header'>Model Loading Status</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Load models with debug info
            yolo_models = load_yolo_models()
            rcnn_models = load_rcnn_models()
            
            # Check if any models loaded successfully
            valid_yolo_models = [m for m in yolo_models if m is not None]
            valid_rcnn_models = [m for m in rcnn_models if m is not None]
            
            if len(valid_yolo_models) == 0:
                st.error("🚨 No YOLO models loaded! Detection will fail.")
                st.stop()
            
            # Extract data
            with st.spinner('Extracting frames and audio...'):
                frames, frame_timestamps = extract_frames_with_timestamps(str(tmp), target_fps)
                waveform, sr = extract_audio_waveform_from_video(str(tmp), sr=16000)

            # Analysis results
            st.markdown(f"""
            <div class='apple-card'>
                <h2 class='apple-section-header'>Extraction Complete</h2>
                <div class='apple-metrics'>
                    <div class='apple-metric'>
                        <div class='apple-metric-value'>{len(frames)}</div>
                        <div class='apple-metric-label'>Frames</div>
                    </div>
                    <div class='apple-metric'>
                        <div class='apple-metric-value'>{sr:,}</div>
                        <div class='apple-metric-label'>Sample Rate</div>
                    </div>
                    <div class='apple-metric'>
                        <div class='apple-metric-value'>{len(waveform)/sr:.1f}s</div>
                        <div class='apple-metric-label'>Duration</div>
                    </div>
                    <div class='apple-metric'>
                        <div class='apple-metric-value'>{conf_thres}</div>
                        <div class='apple-metric-label'>Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Run analysis with debug info
            with st.spinner('Running vision analysis...'):
                yolo_all = run_yolo_on_frames(frames, yolo_models, conf_thres=conf_thres)

            with st.spinner('Running audio analysis...'):
                rcnn_all = run_rcnn_on_audio_timestamps(waveform, sr, frame_timestamps, audio_clip_len, rcnn_models)

            # Build results dataframe
            rows = []
            for mi, model_results in enumerate(yolo_all):
                for fi, dets in enumerate(model_results):
                    ts = frame_timestamps[fi]
                    if dets:
                        for d in dets:
                            rows.append({
                                'timestamp': ts,
                                'time_str': format_ts(ts),
                                'source': f'Vision Model {mi+1}',
                                'type': 'Visual',
                                'detection': d.get('label', 'Unknown'),
                                'confidence': d.get('conf', 0.0)
                            })
            
            for mi, model_results in enumerate(rcnn_all):
                for fi, res in enumerate(model_results):
                    ts = frame_timestamps[fi]
                    if res is not None:
                        conf = max(res.get('probs', [0])) if res.get('probs') is not None else 0.0
                        rows.append({
                            'timestamp': ts,
                            'time_str': format_ts(ts),
                            'source': f'Audio Model {mi+1}',
                            'type': 'Audio',
                            'detection': str(res.get('pred', 'Unknown')),
                            'confidence': conf
                        })

            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['timestamp','time_str','source','type','detection','confidence'])

            # Results summary
            st.markdown("""
            <div class='apple-card'>
                <h2 class='apple-section-header'>Analysis Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if df.empty:
                st.info('🎯 No detections found in this video')
            else:
                df_sorted = df.sort_values('timestamp').reset_index(drop=True)
                
                # Detection summary
                total_detections = len(df_sorted)
                high_conf = len(df_sorted[df_sorted['confidence'] > 0.7])
                visual_detections = len(df_sorted[df_sorted['type'] == 'Visual'])
                audio_detections = len(df_sorted[df_sorted['type'] == 'Audio'])
                
                st.markdown(f"""
                <div class='apple-metrics'>
                    <div class='apple-metric'>
                        <div class='apple-metric-value'>{total_detections}</div>
                        <div class='apple-metric-label'>Total Detections</div>
                    </div>
                    <div class='apple-metric'>
                        <div class='apple-metric-value'>{high_conf}</div>
                        <div class='apple-metric-label'>High Confidence</div>
                    </div>
                    <div class='apple-metric'>
                        <div class='apple-metric-value'>{visual_detections}</div>
                        <div class='apple-metric-label'>Visual</div>
                    </div>
                    <div class='apple-metric'>
                        <div class='apple-metric-value'>{audio_detections}</div>
                        <div class='apple-metric-label'>Audio</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(df_sorted, use_container_width=True)
                
                # Download option
                csv = df_sorted.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv, file_name="analysis_results.csv", mime="text/csv")

            # Frame viewer
            st.markdown("""
            <div class='apple-card'>
                <h2 class='apple-section-header'>Frame Analysis</h2>
                <p class='apple-section-subheader'>Review annotated frames with detected objects and play corresponding audio segments.</p>
            </div>
            """, unsafe_allow_html=True)
            
            frames_per_page = 4
            total_frames = len(frames)
            total_pages = max(1, (total_frames + frames_per_page - 1) // frames_per_page)

            # Navigation with proper key handling
            nav_cols = st.columns([1,3,1])
            
            # Previous button
            prev_clicked = nav_cols[0].button("Previous", key="prev_btn", disabled=(st.session_state['apple_page'] == 0))
            if prev_clicked and st.session_state['apple_page'] > 0:
                st.session_state['apple_page'] -= 1
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            
            # Page info
            nav_cols[1].markdown(f"""
            <div class='apple-nav-info' style='text-align: center; padding: 8px;'>
                Page {st.session_state['apple_page'] + 1} of {total_pages}
            </div>
            """, unsafe_allow_html=True)
            
            # Next button
            next_clicked = nav_cols[2].button("Next", key="next_btn", disabled=(st.session_state['apple_page'] >= total_pages - 1))
            if next_clicked and st.session_state['apple_page'] < total_pages - 1:
                st.session_state['apple_page'] += 1
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()

            # Display frames
            start_idx = st.session_state['apple_page'] * frames_per_page
            end_idx = min(total_frames, start_idx + frames_per_page)

            frame_cols = st.columns(2)
            for i in range(start_idx, end_idx):
                col = frame_cols[i % 2]
                
                # Aggregate detections for this frame
                frame_detections = []
                for mi in range(len(yolo_all)):
                    if i < len(yolo_all[mi]):
                        frame_detections += yolo_all[mi][i] or []
                
                # Draw annotations
                annotated_frame = draw_boxes_on_frame(frames[i], frame_detections)
                
                # Display with Apple-style container
                detection_count = len(frame_detections)
                confidence_avg = sum(d.get('conf', 0) for d in frame_detections) / max(1, detection_count)
                
                status_class = "success" if detection_count == 0 else "warning" if confidence_avg < 0.5 else "error"
                
                col.markdown(f"""
                <div class='apple-card'>
                    <div style='margin-bottom: 12px;'>
                        <span class='apple-status apple-status-{status_class}'>
                            {detection_count} detections
                        </span>
                        <span style='color: var(--apple-text-secondary); font-size: 13px;'>
                            {format_ts(frame_timestamps[i])}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Audio for this timestamp
                start_s = max(0, frame_timestamps[i] - audio_clip_len/2)
                end_s = min((len(waveform)/sr), start_s + audio_clip_len)
                audio_segment = waveform[int(start_s*sr):int(end_s*sr)]
                audio_bytes = array_to_audio_bytes(audio_segment, sr)
                col.audio(audio_bytes)

        st.markdown("""
        <div class='apple-card'>
            <h3 style='color: var(--apple-text-secondary); font-size: 17px; margin: 0 0 12px 0;'>Tips for Better Results</h3>
            <ul style='color: var(--apple-text-secondary); margin: 0; padding-left: 20px;'>
                <li>Use higher frame rates for more detailed analysis</li>
                <li>Ensure good video quality and lighting</li>
                <li>Audio analysis works best with clear sound</li>
                <li>Processing time depends on video length and frame rate</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
