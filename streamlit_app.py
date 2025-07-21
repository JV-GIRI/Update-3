import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.signal import butter, lfilter
from datetime import datetime
import json
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import tempfile

st.set_page_config(layout="wide")
st.title("🩺 PCG Analyzer with History + Patient Info")

# Directories
UPLOAD_FOLDER = "uploaded_audios"
PATIENT_DATA = "patient_data.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load/save patient data
def save_patient_data(data):
    if os.path.exists(PATIENT_DATA):
        with open(PATIENT_DATA, "r") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(data)
    with open(PATIENT_DATA, "w") as f:
        json.dump(existing, f)

def load_patient_data():
    if os.path.exists(PATIENT_DATA):
        with open(PATIENT_DATA, "r") as f:
            return json.load(f)
    return []

# Noise reduction
def reduce_noise(audio, sr):
    b, a = butter(6, 0.05)
    return lfilter(b, a, audio)

# Audio analysis
def analyze_audio(path, unique_id):
    sr, audio = wav.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    st.subheader("🔊 Audio Waveform")

    col1, col2 = st.columns(2)
    with col1:
        amplitude_factor = st.slider("Amplitude scaling", 0.1, 5.0, 1.0, key=f"amp_{unique_id}")
    with col2:
        duration_slider = st.slider("Adjust duration (seconds)", 1, int(len(audio) / sr), 5, key=f"dur_{unique_id}")

    adjusted_audio = audio[:duration_slider * sr] * amplitude_factor

    if st.button("🧹 Reduce Noise", key=f"noise_{unique_id}"):
        adjusted_audio = reduce_noise(adjusted_audio, sr)

    fig, ax = plt.subplots()
    times = np.linspace(0, duration_slider, len(adjusted_audio))
    ax.plot(times, adjusted_audio)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    st.audio(path, format="audio/wav")

# Upload sidebar
st.sidebar.header("📁 Upload or Record")

upload_file = st.sidebar.file_uploader("Upload WAV File", type=["wav"])

st.sidebar.markdown("""
📱 *You can also use the [**Song Meter Touch**](https://www.wildlifeacoustics.com/products/song-meter-touch) app (iOS/Android) to record PCG audio and upload the WAV here.*
""")

# Optional mic recording
st.sidebar.markdown("🎤 Or record directly using your microphone:")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.recorded_frames.append(audio)
        return frame

ctx = webrtc_streamer(
    key="pcg-recording",
    mode="sendonly",
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

recorded_audio_path = None
if ctx and ctx.state.playing and ctx.audio_processor:
    if st.sidebar.button("⏹ Stop and Save Recording"):
        frames = ctx.audio_processor.recorded_frames
        audio_np = np.concatenate(frames)
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=UPLOAD_FOLDER)
        sf.write(tmp_wav.name, audio_np, 48000)  # WebRTC uses 48kHz
        recorded_audio_path = tmp_wav.name
        st.success("Recording saved!")
        st.audio(tmp_wav.name, format="audio/wav")

if upload_file:
    path = os.path.join(UPLOAD_FOLDER, upload_file.name)
    with open(path, "wb") as f:
        f.write(upload_file.getbuffer())
    st.success("File uploaded successfully!")
    st.audio(path, format="audio/wav")

# Use uploaded or recorded
selected_audio_path = path if upload_file else recorded_audio_path

# Patient Info
if "patient_saved" not in st.session_state:
    st.session_state["patient_saved"] = False

with st.sidebar.expander("🧾 Add Patient Info"):
    name = st.text_input("Name")
    age = st.number_input("Age", 1, 120)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    notes = st.text_area("Clinical Notes")

    if st.button("💾 Save Patient Case"):
        if selected_audio_path:
            filename = os.path.basename(selected_audio_path)
            data = {
                "name": name,
                "age": age,
                "gender": gender,
                "notes": notes,
                "file": filename,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_patient_data(data)
            st.session_state["patient_saved"] = True
            st.success("Patient data saved. Now analyzing the audio...")
        else:
            st.warning("Please upload or record a PCG file before saving.")

# Case history jump button
if st.button("📚 View Case History"):
    st.session_state["show_history"] = True

# Perform analysis
if selected_audio_path and st.session_state["patient_saved"]:
    analyze_audio(selected_audio_path, unique_id=os.path.basename(selected_audio_path))

# History
if st.session_state.get("show_history", False):
    st.subheader("📚 Case History")

    patient_data = load_patient_data()
    if patient_data:
        for i, entry in enumerate(patient_data[::-1]):
            with st.expander(f"{entry['name']} ({entry['age']} y/o) - {entry['date']}"):
                st.write(f"**Gender:** {entry['gender']}")
                st.write(f"**Notes:** {entry['notes']}")
                file_path = os.path.join(UPLOAD_FOLDER, entry["file"])
                if os.path.exists(file_path):
                    st.audio(file_path, format="audio/wav")
                    analyze_audio(file_path, unique_id=f"history_{i}")
                else:
                    st.error("Audio file missing.")
    else:
        st.info("No history records found.")
