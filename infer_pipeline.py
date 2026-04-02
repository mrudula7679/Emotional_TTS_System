import os
import tempfile
import shutil
from TTS.api import TTS
from src.emotion_detector import detect_emotion
from src.style_mapper import StyleMapper
from src.voice_manager import get_voice_choice
from src.postprocess_audio import apply_prosody


def read_input(path_or_text: str):
    if os.path.exists(path_or_text):
        with open(path_or_text, 'r', encoding='utf-8') as f:
            return f.read()
    return path_or_text


def load_tts_model(model_name: str):
    """
    Safely load TTS model with fallback and cache handling
    """
    print(f" Loading TTS model: {model_name}")

    try:
        tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
        return tts
    except Exception as e:
        print(" Model loading failed:", e)

        # Clear corrupted cache
        cache_path = os.path.expanduser("~/.local/share/tts")
        if os.path.exists(cache_path):
            print(" Clearing corrupted TTS cache...")
            shutil.rmtree(cache_path)

        # Fallback model
        fallback_model = "tts_models/en/vctk/vits"
        print(f" Switching to fallback model: {fallback_model}")

        tts = TTS(model_name=fallback_model, progress_bar=True, gpu=False)
        return tts


def synthesize_text_to_emotional_speech(
    text_or_path: str,
    gender: str = 'female',
    accent: str = 'neutral',
    out_path: str = 'output/final_output.wav'
):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Read input
    text = read_input(text_or_path)

    # ---------------- EMOTION DETECTION ----------------
    print(" Detecting emotion...")
    emotion_label = detect_emotion(text)
    print("Detected emotion:", emotion_label)

    # ---------------- STYLE MAPPING ----------------
    mapper = StyleMapper()
    mapping = mapper.map(emotion_label, intensity=0.7)
    prosody = mapping['prosody']

    print(" Prosody settings:", prosody)

    # ---------------- VOICE SELECTION ----------------
    voice = get_voice_choice(gender, accent)
    print(" Voice config:", voice)

    model_name = voice.get('model', "tts_models/en/vctk/vits")
    speaker_id = voice.get('speaker')

    # ---------------- LOAD MODEL ----------------
    tts = load_tts_model(model_name)

    # ---------------- SYNTHESIS ----------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = os.path.join(tmp, "tts.wav")

        print(f"🗣️ Synthesizing speech ({gender}, {accent})...")

        try:
            # Use speaker only if supported
            if speaker_id is not None:
                tts.tts_to_file(
                    text=text,
                    speaker=speaker_id,
                    file_path=tmp_wav
                )
            else:
                tts.tts_to_file(
                    text=text,
                    file_path=tmp_wav
                )

        except Exception as e:
            print(" Error during synthesis:", e)
            print(" Retrying without speaker...")

            # Retry without speaker
            tts.tts_to_file(
                text=text,
                file_path=tmp_wav
            )

        # ---------------- POST-PROCESSING ----------------
        print("Applying emotion prosody...")
        apply_prosody(tmp_wav, out_path, prosody)

    print(f" Emotional speech saved at: {out_path}")
    return out_path