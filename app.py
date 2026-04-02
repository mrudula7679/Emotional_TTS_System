import streamlit as st
import subprocess
import os
import tempfile
import uuid
from PIL import Image
from emotion_detector import read_txt, read_docx, read_pdf, read_image, detect_emotion

st.title("Expressive Speech Synthesis (Multi-Input)")

# --- Input options ---
input_text = st.text_area("Enter text here:")
uploaded_file = st.file_uploader(
    "Or upload a file", type=["txt", "docx", "pdf", "jpg", "jpeg", "png"]
)

# --- Voice selection (removed child, added accents) ---
gender_options = ["female", "male"]
accent_options = ["american", "british", "australian", "indian", "neutral"]
voice_gender = st.selectbox("Choose gender:", gender_options)
voice_accent = st.selectbox("Choose accent:", accent_options)

def get_text_from_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    ext = os.path.splitext(temp_file.name)[1].lower()
    if ext == ".txt":
        text = read_txt(temp_path)
    elif ext == ".docx":
        text = read_docx(temp_path)
    elif ext == ".pdf":
        text = read_pdf(temp_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        text = read_image(temp_path)
    else:
        text = ""
    return text

if st.button("Generate Audio"):
    
    if uploaded_file is not None:
        text = get_text_from_file(uploaded_file)
    else:
        text = input_text

    if not text.strip():
        st.error("No input provided. Enter text or upload a file.")
        st.stop()

    
    emotion_label = detect_emotion(text)
    st.write(f"Detected Emotion: **{emotion_label}**")

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_in:
        temp_in.write(text.encode("utf-8"))
        temp_in_path = temp_in.name

    
    temp_out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")

    
    cmd = [
        "python",
        "-m", "src.main",
        "--input", temp_in_path,
        "--gender", voice_gender,
        "--accent", voice_accent,
        "--out", temp_out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    
    if result.returncode != 0:
        st.error("main.py failed")
        st.text(result.stderr)
    else:
        if os.path.exists(temp_out_path):
            st.audio(temp_out_path, format="audio/wav")
            st.success("Audio generated successfully!")
            with open(temp_out_path, "rb") as f:
                st.download_button(
                    label="Download Audio",
                    data=f,
                    file_name="speech.wav",
                    mime="audio/wav"
                )
        else:
            st.error("No audio file was created.")
            st.text(result.stdout)
