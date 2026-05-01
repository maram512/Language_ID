import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import joblib
import warnings
import tempfile
import os
import random

# Ignore warnings for a cleaner UI
warnings.filterwarnings("ignore")

# ==========================================
# 1. CNN Model Architecture
# Must match the exact architecture used during training
# ==========================================
class LanguageCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(LanguageCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), 
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), 
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))

# ==========================================
# 2. Model Loading Function
# Uses caching to load models only once (prevents server crashes)
# ==========================================
@st.cache_resource
def load_models():
    # Force PyTorch to be perfectly deterministic across different servers
    # This ensures math operations output the exact same numbers on Cloud vs PC
    torch.use_deterministic_algorithms(True)
    
    # --- Load CNN Model ---
    cnn_path = 'models/cnn_language_model4.pt'
    cnn_model = LanguageCNN(num_classes=3)
    checkpoint = torch.load(cnn_path, map_location='cpu', weights_only=False)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    cnn_model.eval()
    le_cnn = checkpoint["encoder"]
    
    # --- Load GMM Model ---
    gmm_path = 'models/gmm_language_model (3).pkl'
    gmm_data = joblib.load(gmm_path)
    gmm_models = gmm_data["models"]
    le_gmm = gmm_data["encoder"]
    
    return cnn_model, le_cnn, gmm_models, le_gmm

# ==========================================
# 3. CNN Feature Extraction Pipeline
# Bulletproof loading to match training environment exactly
# ==========================================
def extract_cnn_features(audio, sr):
    audio = np.array(audio, dtype=np.float32)
    
    # 1. Resample to 16kHz (Safety check, usually handled in librosa.load now)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
        
    # 2. Remove DC offset
    audio = audio - np.mean(audio)
    
    # 3. Pad/Trim to exactly 6 seconds
    target_len = sr * 6
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='reflect')
    else:
        audio = audio[:target_len]
        
    # 4. Normalize amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    
    # 5. Extract features
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=1.0, amin=1e-5, top_db=80)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=42)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    
    # 6. Normalize and stack to 128 rows
    def norm(f): return (f - np.mean(f)) / (np.std(f) + 1e-9)
    
    mel_db = norm(mel_db)
    mfcc_stack = np.vstack([norm(mfcc), norm(delta), norm(delta2), norm(centroid), norm(zcr)])
    
    # 7. Fix time frames to 300
    def fix_frames(x, target=300):
        if x.shape[1] < target:
            return np.pad(x, ((0, 0), (0, target - x.shape[1])), mode='reflect')
        return x[:, :target]
        
    # Final shape: (2, 128, 300)
    features = np.stack([fix_frames(mel_db), fix_frames(mfcc_stack)], axis=0)
    return features

# ==========================================
# 4. GMM Preprocessing & Feature Extraction
# (Strictly matches the training notebook)
# ==========================================
def preprocess_gmm_audio(audio, sr):
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = audio - np.mean(audio)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    intervals = librosa.effects.split(audio, top_db=25)
    if len(intervals) > 0:
        audio = np.concatenate([audio[start:end] for start, end in intervals])
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    target_len = sr * 6
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='reflect')
    else:
        audio = audio[:target_len]
    return audio, sr

def normalize_feature(feature):
    mean = np.mean(feature)
    std = np.std(feature)
    if std == 0: return feature - mean
    return (feature - mean) / std

def extract_gmm_features(audio, sr):
    audio, sr = preprocess_gmm_audio(audio, sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    mfcc = normalize_feature(mfcc)
    delta = normalize_feature(delta)
    delta2 = normalize_feature(delta2)
    zcr = normalize_feature(zcr)
    centroid = normalize_feature(centroid)
    features = np.vstack([mfcc, delta, delta2, zcr, centroid]).T
    return features

# ==========================================
# 5. Streamlit User Interface
# ==========================================
st.set_page_config(page_title="Language ID App", layout="wide")

st.title("🗣️ Spoken Language Identification System")
st.markdown("Compare **CNN** and **GMM** model predictions. Upload an audio file (WAV, MP3, M4A, etc.).")

# Load models once
cnn_model, le_cnn, gmm_models, le_gmm = load_models()

# File Upload Section
uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=["wav", "mp3", "m4a", "ogg", "flac"], 
    key="file_uploader"
)

audio = None
sr = None

if uploaded_file is not None:
    # Display the audio player
    st.audio(uploaded_file)
    
    try:
        # ROBUST FORMAT HANDLING:
        # Save to a temporary file first. This bypasses OS-level decoding 
        # errors and handles all formats safely.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            tmp.write(uploaded_file.read())
        
        # Force exact training format (Mono, 16kHz) regardless of file type
        # This is the key fix for PC vs Cloud discrepancies
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
        
        # Delete the temporary file to save server space
        os.remove(tmp_path)
        
    except Exception as e:
        # Catch any decoding errors and show a friendly message instead of crashing
        st.error(f"❌ Error processing audio file: {e}")
        st.info("Please try converting your file to a standard .wav format and upload again.")
        audio = None # Set to None so the prediction blocks don't run

# If audio is successfully loaded, run predictions
if audio is not None:
    
    col1, col2 = st.columns(2)
    
    # --- CNN Prediction ---
    with col1:
        st.subheader("🧠 CNN Model Prediction")
        with st.spinner('Analyzing with CNN...'):
            feats = extract_cnn_features(audio, sr)
            tensor = torch.tensor(feats[np.newaxis].astype(np.float32))
            
            with torch.no_grad():
                out = cnn_model(tensor)
                probs = F.softmax(out, dim=1).numpy()[0]
                pred_id = int(np.argmax(probs))
            
            lang = le_cnn.inverse_transform([pred_id])[0]
            conf = probs[pred_id]
        
        st.success(f"**Prediction: {lang}**")
        st.metric(label="Confidence", value=f"{conf*100:.2f}%")
        st.bar_chart({le_cnn.classes_[i]: probs[i] for i in range(len(probs))})
        
    # --- GMM Prediction ---
    with col2:
        st.subheader("📊 GMM Model Prediction")
        with st.spinner('Analyzing with GMM...'):
            feats_gmm = extract_gmm_features(audio, sr)
            scores = [
                gmm_models[i].score_samples(feats_gmm).mean() 
                for i in range(len(gmm_models))
            ]
            pred_id_gmm = int(np.argmax(scores))
            lang_gmm = le_gmm.inverse_transform([pred_id_gmm])[0]
            
            scores_arr = np.array(scores)
            exp_scores = np.exp(scores_arr - np.max(scores_arr))
            probs_gmm = exp_scores / exp_scores.sum()
            conf_gmm = probs_gmm[pred_id_gmm]
            
        st.success(f"**Prediction: {lang_gmm}**")
        st.metric(label="Confidence", value=f"{conf_gmm*100:.2f}%")
        st.bar_chart({le_gmm.classes_[i]: probs_gmm[i] for i in range(len(probs_gmm))})
