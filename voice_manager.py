# High-precision VITS models with proper gender separation
VOICE_MODELS = {
    # Female voices - high quality VITS models
    "female": {
        "american": {"model": "tts_models/en/ljspeech/tacotron2-DDC", "speaker": None},
        "british": {"model": "tts_models/en/ek1/tacotron2", "speaker": None},
        "australian": {"model": "tts_models/en/ljspeech/vits", "speaker": None},
        "indian": {"model": "tts_models/en/vctk/vits", "speaker": "p236"},  # Female speaker
        "neutral": {"model": "tts_models/en/ljspeech/vits", "speaker": None}
    },
    # Male voices - distinct models/speakers
    "male": {
        "american": {"model": "tts_models/en/vctk/vits", "speaker": "p225"},  # Male speaker
        "british": {"model": "tts_models/en/vctk/vits", "speaker": "p251"},  # Male speaker
        "australian": {"model": "tts_models/en/vctk/vits", "speaker": "p339"},  # Male speaker
        "indian": {"model": "tts_models/en/vctk/vits", "speaker": "p295"},  # Male speaker
        "neutral": {"model": "tts_models/en/vctk/vits", "speaker": "p225"}  # Male default
    }
}

def get_voice_choice(gender: str, accent: str):
    """
    Returns TTS model config for specific gender + accent combination
    Uses VCTK dataset speakers for precise gender control
    """
    gender = gender.lower()
    accent = accent.lower()
    
    if gender in VOICE_MODELS and accent in VOICE_MODELS[gender]:
        return VOICE_MODELS[gender][accent]
    else:
        # Fallback to neutral models with proper gender
        return VOICE_MODELS["female"]["neutral"] if gender == "female" else VOICE_MODELS["male"]["neutral"]
