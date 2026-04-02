import torch
import numpy as np
import soundfile as sf
import librosa

ACOUSTIC_MAP = {
    "happy":   {"pitch": 1.2, "speed": 1.1, "energy": 1.2},
    "angry":   {"pitch": 1.1, "speed": 1.3, "energy": 1.5},
    "sad":     {"pitch": 0.9, "speed": 0.8, "energy": 0.7},
    "fear":    {"pitch": 1.3, "speed": 1.2, "energy": 0.9},
    "disgust": {"pitch": 0.8, "speed": 0.9, "energy": 0.8},
    "surprise":{"pitch": 1.4, "speed": 1.2, "energy": 1.3},
    "neutral": {"pitch": 1.0, "speed": 1.0, "energy": 1.0}
}

SPEAKER_MAP = {
    "male": 0,
    "female": 1,
    "child": 2
}

def adjust_acoustics(audio: np.ndarray, emotion: str, sr: int = 22050):
    params = ACOUSTIC_MAP.get(emotion, ACOUSTIC_MAP["neutral"])

    # Speed
    new_length = int(len(audio) / params["speed"])
    audio = np.interp(np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)), audio)

    # Pitch
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=(params["pitch"]-1)*2)

    # Energy
    audio = audio * params["energy"]
    audio = np.clip(audio, -1.0, 1.0)
    return audio

def synthesize(text, emotion="neutral", speaker="male", out_path="output.wav"):
    sr = 22050

    # Dummy waveform (replace with trained VITS output)
    duration = max(2, len(text) // 5)
    t = np.linspace(0, duration, int(sr*duration))
    audio = 0.2 * np.sin(2*np.pi*220*t)  # sine wave as placeholder

    # Apply acoustics
    audio = adjust_acoustics(audio, emotion, sr)

    # Save audio
    sf.write(out_path, audio, sr)
    print(f" Generated speech saved at {out_path}")
