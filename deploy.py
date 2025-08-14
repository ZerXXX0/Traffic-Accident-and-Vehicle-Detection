# deploy.py
import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import subprocess
import tempfile
import os
import io
import base64
import soundfile as sf
from pathlib import Path
from datetime import timedelta
import altair as alt
import torch.nn as nn
import torchaudio

# ----------------------------
# CONFIG (your exact paths)
# ----------------------------
YOLO_PATHS = [
    "./model/ambulance_firetruck_best.pt",
    "./model/smoke_and_fire_best.pt",
    "./model/crash_best.pt"
]
RCNN_PATHS = [
    "./model/crash_sound_best.pt",
    "./model/rcnn_siren_best.pt"
]
RCNN_SAMPLE_RATE = 16000   # same sample rate for both RCNN models (you said yes)
DEFAULT_FPS = 2
AUDIO_CLIP_SEC = 2
CONF_THRESH = 0.25

# ----------------------------
# Utilities
# ----------------------------
def format_ts(sec):
    return str(timedelta(seconds=round(sec, 2)))

def draw_boxes_on_frame(frame, detections):
    img = frame.copy()
    for d in detections:
        x1,y1,x2,y2 = map(int, d['xyxy'])
        label = d.get('label', '')
        conf = d.get('conf', 0.0)
        cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img

def array_to_wav_bytes(waveform, sr):
    bio = io.BytesIO()
    sf.write(bio, waveform, sr, format='WAV')
    bio.seek(0)
    return bio.read()

def extract_audio_with_ffmpeg(video_path, out_wav_path, sr=16000):
    """
    Extract audio from a video to a mono WAV file at the given sample rate.
    Will try system ffmpeg, otherwise uses imageio-ffmpeg's bundled binary.
    """
    import shutil
    
    # Try system ffmpeg first
    ffmpeg_path = shutil.which("ffmpeg")
    
    # If not found, fall back to imageio-ffmpeg
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg as ffmpeg
            ffmpeg_path = ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise RuntimeError(
                "ffmpeg not found on system and imageio-ffmpeg not installed.\n"
                "Install it via: pip install imageio-ffmpeg"
            )
    
    cmd = [
        ffmpeg_path,
        "-y",               # overwrite output if exists
        "-i", str(video_path),  # input file
        "-ac", "1",         # mono audio
        "-ar", str(sr),     # resample
        "-vn",              # no video
        str(out_wav_path)   # output file
    ]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def extract_frames_with_timestamps(video_path, target_fps):
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(orig_fps / target_fps)))
    frames = []
    timestamps = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts = idx / orig_fps
            frames.append(frame.copy())
            timestamps.append(ts)
        idx += 1
    cap.release()
    return frames, timestamps

# Model architecture
class RCNNClassifier(nn.Module):
    def __init__(self, n_mels=64, n_classes=3, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.lstm = nn.LSTM(input_size=32 * (n_mels // 4), hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):  # x: (B,1,n_mels,time)
        x = self.cnn(x)  # (B, C, M', T')
        B, C, M, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, M)
        x = x.view(B, T, C * M)  # (B, T, feat)
        x, _ = self.lstm(x)  # (B, T, hidden)
        x = x[:, -1, :]  # last timestep
        out = self.classifier(x)  # (B, n_classes)
        return out

# RCNN model
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x): return self.net(x)

class RCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.c1 = ConvBlock(1, 32)
        self.c2 = ConvBlock(32, 64)
        self.c3 = ConvBlock(64, 128)
        self.rnn_input = 128 * (64 // 8)
        self.rnn = nn.GRU(self.rnn_input, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C*M)
        x, _ = self.rnn(x)
        x = x.mean(1)
        return self.fc(x)

# ----------------------------
# Model load & check helpers
# ----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_and_load_models(yolo_paths, rcnn_paths, device):
    status = {"yolo": [], "rcnn": []}
    yolo_models = []
    rcnn_models = []

    # === YOLO Models ===
    for p in yolo_paths:
        info = {"path": p, "loaded": False, "error": None}
        try:
            from ultralytics import YOLO
            m = YOLO(p)
            yolo_models.append(m)
            info["loaded"] = True
        except Exception as e:
            info["error"] = repr(e)
            yolo_models.append(None)
        status["yolo"].append(info)

    # === RCNN Models ===
    # RCNN Model 1
    info1 = {"path": rcnn_paths[0], "loaded": False, "error": None}
    try:
        m1 = RCNNClassifier(n_mels=64, n_classes=3, hidden_size=128)
        m1.load_state_dict(torch.load(rcnn_paths[0], map_location=device, weights_only=False))
        m1.to(device)
        m1.eval()
        rcnn_models.append(m1)
        info1["loaded"] = True
    except Exception as e:
        info1["error"] = repr(e)
        rcnn_models.append(None)
    status["rcnn"].append(info1)

    # RCNN Model 2
    info2 = {"path": rcnn_paths[1], "loaded": False, "error": None}
    try:
        m2 = torch.load(rcnn_paths[1], map_location=device, weights_only=False)
        m2.to(device)
        m2.eval()
        rcnn_models.append(m2)
        info2["loaded"] = True
    except Exception as e:
        info2["error"] = repr(e)
        rcnn_models.append(None)
    status["rcnn"].append(info2)

    return yolo_models, rcnn_models, status

# ----------------------------
# Inference wrappers
# ----------------------------
def run_yolo_on_frames(frames, yolo_models, conf_thres=CONF_THRESH):
    all_results = []
    for model in yolo_models:
        model_results = []
        if model is None:
            model_results = [[] for _ in frames]
            all_results.append(model_results)
            continue

        for frame in frames:
            try:
                res = model.predict(frame, conf=conf_thres, verbose=False)
                boxes = []
                if len(res) > 0 and hasattr(res[0], 'boxes'):
                    names = res[0].names  # class names dict
                    for b in res[0].boxes.data.tolist():
                        x1, y1, x2, y2, score, cls = b
                        label = names.get(int(cls), str(int(cls)))
                        boxes.append({
                            'xyxy': [x1, y1, x2, y2],
                            'conf': float(score),
                            'cls': int(cls),
                            'label': label
                        })
                model_results.append(boxes)
            except Exception as e:
                print(f"YOLO inference error: {e}")
                model_results.append([])
        all_results.append(model_results)
    return all_results

def run_rcnn_on_audio_timestamps(waveform, sr, timestamps, clip_length_sec, rcnn_models, device):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=64
    ).to(device)
    
    results = []
    total_len = len(waveform)
    
    for m in rcnn_models:
        model_res = []
        for ts in timestamps:
            start = int(max(0, (ts - clip_length_sec / 2) * sr))
            end = int(min(total_len, start + clip_length_sec * sr))
            seg = waveform[start:end]

            # Ensure it's always a NumPy array (not a scalar)
            seg = np.array(seg, dtype=np.float32)

            # Skip empty segments or missing model
            if seg.size == 0 or m is None:
                model_res.append(None)
                continue

            # Convert to tensor safely
            seg_tensor = torch.as_tensor(seg, dtype=torch.float32, device=device)

            mel_spec = mel_transform(seg_tensor)  # (n_mels, time)
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # (B=1, C=1, n_mels, time)

            with torch.no_grad():
                out = m(mel_spec)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                model_res.append({'pred': pred, 'probs': probs.tolist()})

        results.append(model_res)
    
    return results


# ----------------------------
# Streamlit app UI
# ----------------------------
st.set_page_config(layout="wide", page_title="Multi-model YOLO + RCNN Inference")
st.title("Multi-model YOLO + RCNN Video/Audio Inference")

col_l, col_r = st.columns([1,2])
with col_l:
    st.header("Config")
    target_fps = st.number_input("Frame sampling FPS", min_value=1, max_value=30, value=DEFAULT_FPS)
    audio_clip_len = st.number_input("Audio clip length (s)", min_value=1, max_value=10, value=AUDIO_CLIP_SEC)
    conf_thresh = st.slider("YOLO confidence threshold", min_value=0.0, max_value=1.0, value=CONF_THRESH)
    device = get_device()
    st.write("Device:", device)

with col_r:
    st.header("Models")
    st.write("YOLO model paths (3):")
    for p in YOLO_PATHS: st.write("-", p)
    st.write("RCNN model paths (2):")
    for p in RCNN_PATHS: st.write("-", p)
    st.write(f"RCNN sample rate: {RCNN_SAMPLE_RATE} Hz (shared)")

st.markdown("---")
uploaded = st.file_uploader("Upload video (.mp4 .mov .avi .mkv)", type=['mp4','mov','avi','mkv'])

if uploaded is None:
    st.info("Upload a video file to run inference.")
    st.stop()

# save uploaded temp file
tmp_video = Path(tempfile.gettempdir()) / uploaded.name
with open(tmp_video, "wb") as f:
    f.write(uploaded.read())

st.video(str(tmp_video))

# load models and show load status
with st.spinner("Loading models / checking status..."):
    yolo_models, rcnn_models, load_status = check_and_load_models(YOLO_PATHS, RCNN_PATHS, device)
st.subheader("Model load status")
cols = st.columns(2)
with cols[0]:
    st.write("YOLO models")
    for s in load_status["yolo"]:
        if s["loaded"]:
            st.success(f"Loaded: {s['path']}")
        else:
            st.error(f"Failed: {s['path']} — {s['error']}")
with cols[1]:
    st.write("RCNN models")
    for s in load_status["rcnn"]:
        if s["loaded"]:
            st.success(f"Loaded: {s['path']}")
        else:
            st.error(f"Failed: {s['path']} — {s['error']}")

# extract frames
with st.spinner("Extracting frames..."):
    frames, frame_timestamps = extract_frames_with_timestamps(str(tmp_video), target_fps)
st.success(f"Extracted {len(frames)} frames")

# extract audio (resample once to RCNN_SAMPLE_RATE)
with st.spinner("Extracting and resampling audio with ffmpeg..."):
    tmp_wav = Path(tempfile.gettempdir()) / (tmp_video.stem + f"_sr{RCNN_SAMPLE_RATE}.wav")
    try:
        extract_audio_with_ffmpeg(tmp_video, tmp_wav, sr=RCNN_SAMPLE_RATE)
    except Exception as e:
        st.error(f"ffmpeg audio extraction failed: {e}")
        st.stop()
    sf_read = lambda p: sf.read(p)  # define shortcut function
    waveform, sr = sf_read(tmp_wav)
    # soundfile returns (data, sr), ensure mono float32
    if isinstance(waveform, tuple):
        arr, sr = waveform
    else:
        arr, sr = waveform, RCNN_SAMPLE_RATE
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    waveform = arr.astype('float32')

st.success(f"Audio ready (sr={sr}, len={len(waveform)} samples)")

# run YOLO
with st.spinner("Running YOLO models on frames..."):
    yolo_all = run_yolo_on_frames(frames, yolo_models, conf_thres=conf_thresh)
st.success("YOLO done")

# run RCNN
with st.spinner("Running RCNN models on audio around each frame timestamp..."):
    rcnn_all = run_rcnn_on_audio_timestamps(waveform, sr, frame_timestamps, audio_clip_len, rcnn_models, device)
st.success("RCNN done")

# Label mappings
YOLO_SOURCES = {
    0: "ambulance and fire truck detection vision",
    1: "smoke and fire detection",
    2: "crash detection vision"
}
YOLO2_LABELS = {0: "Smoke", 1: "Fire"}

RCNN_SOURCES = {
    0: "crash detection sound",
    1: "ambulance and fire truck detection sound"
}
RCNN1_LABELS = {0: "Skid", 1: "Crash", 2: "Background"}
RCNN2_LABELS = {0: "Ambulance", 1: "Fire Truck", 2: "Police"}

# build combined timeline
rows = []
jam = []
# vision
for mi, model_results in enumerate(yolo_all):
    for fi, dets in enumerate(model_results):
        ts = frame_timestamps[fi]
        if dets:
            for d in dets:
                label = d.get("label", "")
                # Apply YOLO_2 label remapping
                if mi == 1 and "cls" in d and d["cls"] in YOLO2_LABELS:
                    label = YOLO2_LABELS[d["cls"]]

                if label.lower() == "vehicle":
                    jam.append({
                        "timestamp": ts,
                        "time_str": format_ts(ts),
                        "source": YOLO_SOURCES[mi],
                        "type": "vision",
                        "label": label,
                        "confidence": d.get("conf", 0.0)
                    })
                    continue

                rows.append({
                    "timestamp": ts,
                    "time_str": format_ts(ts),
                    "source": YOLO_SOURCES[mi],
                    "type": "vision",
                    "label": label,
                    "confidence": d.get("conf", 0.0)
                })
# audio
for mi, model_results in enumerate(rcnn_all):
    for fi, res in enumerate(model_results):
        ts = frame_timestamps[fi]
        if res is not None:
            pred = res.get("pred", "")
            probs = res.get("probs", [0])
            label = str(pred)
            # Apply RCNN label remapping
            if mi == 0 and pred in RCNN1_LABELS:
                label = RCNN1_LABELS[pred]
            elif mi == 1 and pred in RCNN2_LABELS:
                label = RCNN2_LABELS[pred]
            
            # Skip background
            if label.lower() == "background":
                continue

            rows.append({
                "timestamp": ts,
                "time_str": format_ts(ts),
                "source": RCNN_SOURCES[mi],
                "type": "audio",
                "label": label,
                "confidence": max(probs) if probs else 0.0
            })


if not rows:
    st.info("No detections found (or models not loaded).")
else:
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    st.subheader("Combined timeline (table)")
    st.dataframe(df)

    st.subheader("Timeline chart")
    chart = alt.Chart(df).mark_point(filled=True, size=100).encode(
        x=alt.X("timestamp:Q", title="Time (s)"),
        y=alt.Y("source:N", title="Model"),
        color="type:N",
        tooltip=["time_str", "source", "type", "label", "confidence"]
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Annotated frame previews with audio")
    preview_n = st.slider("How many annotated frames to preview", 1, min(10, len(frame_timestamps)), 3)
    shown = 0
    for fi, ts in enumerate(frame_timestamps):
        # collect all detections across YOLO models for this frame
        dets_for_frame = []
        for mi in range(len(yolo_all)):
            if fi < len(yolo_all[mi]):
                dets_for_frame += yolo_all[mi][fi]
        if dets_for_frame:
            ann = draw_boxes_on_frame(frames[fi], dets_for_frame)
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"Annotated @ {format_ts(ts)}", use_column_width=True)
            # audio clip
            s = int(max(0, (ts - audio_clip_len/2) * sr))
            e = int(min(len(waveform), s + audio_clip_len * sr))
            seg = waveform[s:e]
            if seg.size > 0:
                st.audio(array_to_wav_bytes(seg, sr))
            shown += 1
            if shown >= preview_n:
                break

# =======================
# VEHICLE DETECTION TABLE
# =======================
if not jam:
    st.info("No vehicle found (or models not loaded).")
else:
    # Convert to DataFrame
    jam_df = pd.DataFrame(jam).sort_values("timestamp").reset_index(drop=True)

    # Count vehicles per timestamp
    vehicle_counts = jam_df.groupby("timestamp").size().reset_index(name="vehicle_count")

    # Classify traffic condition
    def classify_traffic(count):
        if 0 <= count <= 5:
            return "empty"
        elif 6 <= count <= 10:
            return "fluid"
        elif 11 <= count <= 15:
            return "moderate"
        elif 16 <= count <= 20:
            return "heavy"
        else:  # count >= 21
            return "jam"

    vehicle_counts["traffic_condition"] = vehicle_counts["vehicle_count"].apply(classify_traffic)

    # Merge classification back into jam_df
    jam_df = jam_df.merge(vehicle_counts, on="timestamp", how="left")

    st.subheader("Vehicle Detection Timeline")
    st.dataframe(jam_df)

    st.subheader("Vehicle Timeline Chart")
    jam_chart = alt.Chart(jam_df).mark_point(filled=True, size=100).encode(
        x=alt.X("timestamp:Q", title="Time (s)"),
        y=alt.Y("traffic_condition:N", title="Traffic Condition"),
        color=alt.Color("vehicle_count:Q", title="Vehicle Count", scale=alt.Scale(scheme="reds")),
        tooltip=["time_str", "vehicle_count", "traffic_condition", "confidence"]
    )
    st.altair_chart(jam_chart, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    st.markdown(f"[Download timeline CSV](data:file/csv;base64,{b64})")
