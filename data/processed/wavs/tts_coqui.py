# src/tts_coqui.py
import os
import tempfile
from TTS.api import TTS

# Map friendly voice choices to Coqui model names.
# You can change these to other available Coqui model ids.
VOICE_MODEL_MAP = {
    'female': "tts_models/en/ljspeech/tacotron2-DDC",   # single female voice
    'male':   "tts_models/en/vctk/vits",               # multi-speaker VCTK (select speaker optionally)
    'child':  "tts_models/en/ljspeech/tacotron2-DDC"   # we'll pitch-shift to simulate child
}

class CoquiSynth:
    def __init__(self, voice_choice: str = 'female'):
        self.voice_choice = voice_choice if voice_choice in VOICE_MODEL_MAP else 'female'
        self.model_name = VOICE_MODEL_MAP[self.voice_choice]
        # This will download model weights on first run and cache them locally.
        self.tts = TTS(self.model_name)

    def synthesize(self, text: str, out_path: str, speaker: str = None, style: str = None):
        """
        text: input text
        out_path: file path to save wav
        speaker: optional (for multi-speaker models you can pass speaker id or speaker_wav)
        style: optional string style token if model supports
        """
        kwargs = {}
        # For models supporting speakers, pass speaker if provided.
        if speaker is not None:
            # some Coqui models accept `speaker` as index or speaker_wav path
            kwargs['speaker'] = speaker
        if style is not None:
            kwargs['style'] = style

        # call tts_to_file (saves to out_path)
        self.tts.tts_to_file(text=text, file_path=out_path, **kwargs)
        return out_path
