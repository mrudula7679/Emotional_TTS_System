# src/postprocess_audio.py
import numpy as np
import soundfile as sf
import librosa

def apply_prosody(in_wav: str, out_wav: str, prosody: dict, sr: int = 22050):
    """
    prosody: {'f0_scale': float, 'rate_scale': float, 'energy_scale': float}
    """
    y, fs = librosa.load(in_wav, sr=sr, mono=True)
    # 1) time-stretch (rate)
    rate = prosody.get('rate_scale', 0.7)
    # librosa.effects.time_stretch speeds up if rate > 1.0
    if rate != 1.0:
        try:
            y = librosa.effects.time_stretch(y, rate)
        except Exception as e:
            print("time-stretch failed:", e)

    # 2) pitch shift (f0)
    f0_scale = prosody.get('f0_scale', 1.0)
    if f0_scale != 1.0:
        # convert scale -> semitone steps: n_steps = 12 * log2(f0_scale)
        n_steps = 12.0 * np.log2(float(f0_scale))
        try:
            y = librosa.effects.pitch_shift(y, sr, n_steps)
        except Exception as e:
            print("pitch-shift failed:", e)

    # 3) energy scaling
    energy = prosody.get('energy_scale', 0.6)
    if energy != 1.0:
        y = y * float(energy)

    # normalize to avoid clipping
    max_amp = np.max(np.abs(y)) + 1e-9
    if max_amp > 1.0:
        y = y / max_amp * 0.99

    sf.write(out_wav, y, sr)
    return out_wav
