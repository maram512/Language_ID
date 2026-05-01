import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import joblib
import warnings

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
    # --- Load CNN Model ---
    # Exact path provided by user
    cnn_path = 'models/cnn_language_model4.pt'
    cnn_model = LanguageCNN(num_classes=3)
    checkpoint = torch.load(cnn_path, map_location='cpu', weights_only=False)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    cnn_model.eval()
    le_cnn = checkpoint["encoder"]
    
    # --- Load GMM Model ---
    # Exact path provided by user
    gmm_path = 'models/gmm_language_model (3).pkl'
    gmm_data = joblib.load(gmm_path)
    gmm_models = gmm_data["models"]
    le_gmm = gmm_data["encoder"]
    
    return cnn_model, le_cnn, gmm_models, le_gmm

# ==========================================
# 3. CNN Feature Extraction Pipeline
# Safe pipeline for external audio
# ==========================================
def extract_cnn_features(audio, sr):
    audio = np.array(audio, dtype=np.float32)
    
    # 1. Resample to 16kHz
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
# 4. GMM Preprocessing Pipeline
# Extracted EXACTLY from the provided GMM training script
# ==========================================
def preprocess_gmm_audio(audio, sr):
    # 1. Resample to target sampling rate
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # 2. Remove DC offset
    audio = audio - np.mean(audio)

    # 3. Apply pre-emphasis filter
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # 4. Remove silent parts of the signal
    intervals = librosa.effects.split(audio, top_db=25)
    if len(intervals) > 0:
        audio = np.concatenate([audio[start:end] for start, end in intervals])

    # 5. Normalize amplitude to range [-1, 1]
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # 6. Ensure fixed length (Pad at the END, like in training)
    target_len = sr * 6
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='reflect')
    else:
        audio = audio[:target_len]

    return audio, sr

# ==========================================
# 5. GMM Feature Extraction Pipeline
# Extracted EXACTLY from the provided GMM training script
# ==========================================
def normalize_feature(feature):
    mean = np.mean(feature)
    std = np.std(feature)
    if std == 0:
        return feature - mean
    return (feature - mean) / std

def extract_gmm_features(audio, sr):
    # Step 1: Preprocess (Exactly matching training order)
    audio, sr = preprocess_gmm_audio(audio, sr)

    # Step 2: Extract raw features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

    # Step 3: Normalize each feature separately
    mfcc = normalize_feature(mfcc)
    delta = normalize_feature(delta)
    delta2 = normalize_feature(delta2)
    zcr = normalize_feature(zcr)
    centroid = normalize_feature(centroid)

    # Step 4: Stack into shape (41, T) -> Transpose to (T, 41)
    features = np.vstack([mfcc, delta, delta2, zcr, centroid])
    
    return features.T


# ==========================================
# 6. Streamlit User Interface
# ==========================================
st.set_page_config(page_title="Language ID App", layout="wide")

st.title("🗣️ Spoken Language Identification System")
st.markdown("Compare **CNN** and **GMM** model predictions in real-time. Upload a file or use your microphone.")

# Load models once
cnn_model, le_cnn, gmm_models, le_gmm = load_models()

# Create Tabs for User Input
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone"])

audio = None
sr = None

# Tab 1: File Upload
with tab1:
    uploaded_file = st.file_uploader("Choose an audio file (WAV/MP3)", type=["wav", "mp3"], key="file_uploader")
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        audio, sr = librosa.load(uploaded_file, sr=None)

# Tab 2: Microphone Recording
with tab2:
    st.info("Click below to start recording. Speak clearly, then click stop.")
    mic_audio = st.audio_input("Record your voice", key="mic_recorder")
    if mic_audio is not None:
        st.success("Recording saved! Processing...")
        audio, sr = librosa.load(mic_audio, sr=None)

# If audio is provided, run predictions
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
            # Extract features (Returns shape T x 41)
            feats_gmm = extract_gmm_features(audio, sr)
            
            # Calculate mean Log-Likelihood for each of the 3 models
            scores = [
                gmm_models[i].score_samples(feats_gmm).mean() 
                for i in range(len(gmm_models))
            ]
            pred_id_gmm = int(np.argmax(scores))
            lang_gmm = le_gmm.inverse_transform([pred_id_gmm])[0]
            
            # Convert scores to readable percentages
            scores_arr = np.array(scores)
            exp_scores = np.exp(scores_arr - np.max(scores_arr))
            probs_gmm = exp_scores / exp_scores.sum()
            conf_gmm = probs_gmm[pred_id_gmm]
            
        st.success(f"**Prediction: {lang_gmm}**")
        st.metric(label="Confidence", value=f"{conf_gmm*100:.2f}%")
        st.bar_chart({le_gmm.classes_[i]: probs_gmm[i] for i in range(len(probs_gmm))})