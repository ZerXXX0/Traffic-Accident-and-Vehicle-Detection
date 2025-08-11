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
    """Replace with your YOLOv12 loading. Examples:
    - ultralytics: from ultralytics import YOLO; YOLO(path)
    - torch hub/yolov5: torch.hub.load(...)
    """
    models = []
    try:
        from ultralytics import YOLO
        models.append(YOLO('./model/ambulance_firetruck.pt'))
        models.append(YOLO('./model/smoke_and_fire_best.pt'))
        models.append(YOLO('./model/crash_best.pt'))
    except Exception:
        # fallback: dummy None list (user must replace)
        models = [None, None, None]
    return models

@st.cache_resource
def load_rcnn_models():
    # Replace with your RCNN model loading (torch.load or model definition + load_state_dict)
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
# Streamlit UI
# -------------------------

st.title("Multi-Model Video + Audio Inference (YOLO + RCNN)")
st.write("Upload a video, pick FPS and audio clip length. The app will run 3 YOLO models on frames and 2 RCNN models on audio segments and show a combined timeline.")

uploaded = st.file_uploader("Upload video (.mp4, .mov, .avi)", type=['mp4','mov','avi','mkv'])
col1, col2 = st.columns(2)
with col1:
    target_fps = st.number_input("Frame sampling FPS", min_value=1, max_value=30, value=2)
with col2:
    audio_clip_len = st.number_input("Audio clip length (sec)", min_value=1, max_value=10, value=2)

if uploaded is not None:
    tmp = Path(tempfile.gettempdir()) / uploaded.name
    with open(tmp, 'wb') as f:
        f.write(uploaded.read())

    st.video(str(tmp))

    with st.spinner('Extracting frames and audio...'):
        frames, frame_timestamps = extract_frames_with_timestamps(str(tmp), target_fps)
        waveform, sr = extract_audio_waveform_from_video(str(tmp), sr=16000)

    st.success(f"Extracted {len(frames)} frames and audio (sr={sr}).")

    # Load models
    yolo_models = load_yolo_models()
    rcnn_models = load_rcnn_models()

    # Run inference
    with st.spinner('Running YOLO models on frames...'):
        yolo_all = run_yolo_on_frames(frames, yolo_models)

    with st.spinner('Running RCNN models on audio around each timestamp...'):
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
        st.info('No detections from any model (or models not loaded).')
    else:
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values('timestamp')

        st.subheader('Combined timeline (table)')
        st.dataframe(df_sorted.reset_index(drop=True))

        st.subheader('Timeline chart')
        base = alt.Chart(df_sorted).encode(x=alt.X('timestamp:Q', title='Time (s)'))
        points = base.mark_point(filled=True, size=100).encode(
            y=alt.Y('source:N', title='Model'),
            color='type:N',
            tooltip=['time_str', 'source', 'type', 'label', 'confidence']
        )
        st.altair_chart(points, use_container_width=True)

        # show preview frames with bounding boxes for the first N detections
        st.subheader('Preview annotated frames')
        preview_n = st.slider('How many annotated frames to preview', min_value=1, max_value=min(10, len(frame_timestamps)), value=3)
        shown = 0
        for fi, ts in enumerate(frame_timestamps):
            # collect all detections for this frame across models
            dets_for_frame = []
            for mi in range(len(yolo_all)):
                dets_for_frame += yolo_all[mi][fi] if fi < len(yolo_all[mi]) else []
            if dets_for_frame:
                ann = draw_boxes_on_frame(frames[fi], dets_for_frame)
                st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"Annotated @ {format_ts(ts)}", use_column_width=True)
                # play audio clip for this timestamp
                start_s = max(0, ts - audio_clip_len/2)
                end_s = min((len(waveform)/sr), start_s + audio_clip_len)
                seg = waveform[int(start_s*sr):int(end_s*sr)]
                audio_bytes = array_to_audio_bytes(seg, sr)
                st.audio(audio_bytes)
                shown += 1
                if shown >= preview_n:
                    break

        # allow CSV download
        csv = df_sorted.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f"data:file/csv;base64,{b64}"
        st.markdown(f"[Download timeline CSV]({href})")

    st.markdown('----')
    st.write('Tips:')
    st.write('- If models are large, run the app on a machine with GPU and adjust model `.to(device)` lines.')
    st.write('- Increase `Frame sampling FPS` to get finer temporal resolution but it will increase compute.')
    st.write('- Increase `Audio clip length` if events are longer than the sampled instant.')
